# SPDX-License-Identifier: Apache-2.0
"""``BaseOp``: the engine behind every first-party op.

An op author implements a few hooks; the base class provides the contract verbs:

- ``plan``: calls :meth:`plan_samples` / :meth:`plan_sample` (metadata only) and
  :meth:`plan_manifest`, assembles the :class:`Plan`.
- ``preview``: materialises a few samples into a scratch directory (hardlinks, no copies),
  runs :meth:`apply_sample` there, diffs every changed array against the original, renders
  thumbnails, deletes the scratch.
- derived LOD levels follow level 0: when an op leaves ``SampleEditor.lods_stale`` set and the
  sample stores levels beyond 0, the engine rebuilds them from the new level 0
  (``frontier_prep.ops.mesh.rebuild_lods``) in the same worker, in ``run`` and in ``preview``.
  Level 0 keeps the op's determinism class; the rebuilt levels have the tolerance class of
  ``prep.mesh.decimate`` and the Result says so in ``tolerance`` (``lod_*`` keys).
- ``run``: makes sure the dataset is versioned and clean, applies :meth:`apply_sample` to
  every planned sample (worker processes, results merged in sorted sample order), then
  :meth:`apply_manifest`, then commits a version with the provenance op record, assertion
  records and the content hashes of what changed.
- ``explain``: a plain-language account built from the result; ops may override.

Hot paths (hashing, array diffs, per-sample transforms) are plain functions over numpy
arrays and paths so they can move to Rust without changing the contract.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from frontier_data.dataset import Dataset, Manifest
from frontier_data.versions import VersionStore, compare_arrays
from frontier_prep import kernels
from frontier_prep.io import SampleEditor, read_manifest, save_manifest, sync_manifest_fields
from frontier_prep.op import (
    Assertion,
    Change,
    Context,
    Cost,
    Determinism,
    Explanation,
    FieldDiff,
    Judgment,
    Machine,
    OpWarning,
    Params,
    Plan,
    Preview,
    Render,
    Result,
    Selection,
    Stopwatch,
    Timing,
    now_iso,
)

log = logging.getLogger(__name__)

MAX_RECORDED_CHANGES = 2000
#: a preview rebuilds stale LOD levels only below this level-0 face count (a 2M-face cascade
#: takes several seconds per sample); above it the preview says so in ``notes``
PREVIEW_LOD_MAX_FACES = 300_000


class SampleOutcome(BaseModel):
    """What :meth:`BaseOp.apply_sample` returns (picklable; crosses process boundaries)."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    changes: list[Change] = Field(default_factory=list)
    assertions: list[Assertion] = Field(default_factory=list)
    warnings: list[OpWarning] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    bytes_read: int = 0
    bytes_written: int = 0
    elapsed_s: float = 0.0
    skipped: bool = False
    #: LOD levels the engine rebuilt because the op changed level 0:
    #: ``{level: {n_faces, n_points, method, target, rel_area, seconds}}``
    lods_rebuilt: dict[int, dict[str, Any]] = Field(default_factory=dict)


class DatasetRef:
    """What ops receive: the dataset reader plus its version store and root path."""

    def __init__(self, dataset: Dataset | str | Path) -> None:
        self.ds = dataset if isinstance(dataset, Dataset) else Dataset(dataset)
        self.root = self.ds.root
        self._store: VersionStore | None = None

    @property
    def store(self) -> VersionStore:
        if self._store is None:
            self._store = VersionStore(self.root)
        return self._store

    @property
    def head(self) -> str | None:
        return self.store.head if self.store.initialized else None

    @property
    def manifest(self) -> Manifest:
        return self.ds.manifest

    @property
    def name(self) -> str:
        return self.ds.manifest.name

    def editor(self, sid: str) -> SampleEditor:
        return SampleEditor(self.ds.sample_dir(sid), sid)

    def sample_ids(self) -> list[str]:
        return list(self.ds.sample_ids)


def as_ref(dataset: Any) -> DatasetRef:
    return dataset if isinstance(dataset, DatasetRef) else DatasetRef(dataset)


class BaseOp:
    """Subclass, set the class attributes, implement the hooks you need."""

    id: ClassVar[str] = ""
    version: ClassVar[str] = "1.0.0"
    category: ClassVar[str] = "other"
    summary: ClassVar[str] = ""
    description: ClassVar[str] = ""
    when_to_use: ClassVar[str] = ""
    determinism: ClassVar[Determinism] = "bitwise"
    tolerance: ClassVar[dict[str, float] | None] = None
    Params: ClassVar[type[Params]] = Params
    changes_fields: ClassVar[bool] = False
    changes_geometry: ClassVar[bool] = False
    changes_manifest: ClassVar[bool] = False
    produces_assertions: ClassVar[bool] = False
    #: False for ops that only touch the manifest (split, select, ...)
    per_sample: ClassVar[bool] = True
    #: ops that only read (quality checks) never create a version unless they add assertions
    read_only: ClassVar[bool] = False
    source: str = "builtin"

    # -- hooks ---------------------------------------------------------------------------------

    def plan_samples(self, ref: DatasetRef, params: Any) -> list[str]:
        """Sample ids the op will touch (sorted, deterministic). Default: all."""
        return ref.sample_ids()

    def plan_sample(self, ed: SampleEditor, params: Any) -> list[Change]:
        """Intended changes of one sample from metadata (no array reads unless cheap)."""
        return []

    def plan_manifest(self, ref: DatasetRef, params: Any) -> list[Change]:
        return []

    def plan_extra(self, ref: DatasetRef, params: Any) -> tuple[list[OpWarning], list[Judgment]]:
        return [], []

    def apply_sample(self, ed: SampleEditor, params: Any, *, seed: int) -> SampleOutcome:
        return SampleOutcome(sample_id=ed.id, skipped=True)

    def apply_manifest(
        self, ref: DatasetRef, manifest: Manifest, params: Any, outcomes: list[SampleOutcome]
    ) -> tuple[list[Change], dict[str, Any]]:
        """Dataset-level step after every sample (main process). Mutate ``manifest`` and
        return (changes, stats); the engine saves it and syncs ``manifest.fields``."""
        return [], {}

    def assertions(
        self, ref: DatasetRef, params: Any, outcomes: list[SampleOutcome], stats: dict[str, Any]
    ) -> list[Assertion]:
        """Dataset-level assertions (``quality.*`` ops)."""
        return []

    def renders(
        self, before: SampleEditor, after: SampleEditor, params: Any, changes: list[Change]
    ) -> list[Render]:
        from frontier_prep.render import default_renders

        return default_renders(before, after, changes)

    def after_commit(self, ref: DatasetRef, params: Any, result: Result) -> None:
        """Hook after the version is committed (tags that must follow the output version)."""
        return None

    def summarize(self, pr: Plan | Result) -> str:
        n = len(pr.changes)
        return f"{self.id}: {n} change(s)"

    # -- verbs -----------------------------------------------------------------------------------

    def plan(self, dataset: Any, params: Any) -> Plan:
        ref = as_ref(dataset)
        p = self.coerce(params)
        t = Stopwatch()
        changes: list[Change] = []
        sids = self.plan_samples(ref, p) if self.per_sample else []
        for sid in sids:
            ed = ref.editor(sid)
            planned = self.plan_sample(ed, p)
            changes += planned
            changes += _planned_lod_rebuilds(ed, planned)
        changes += self.plan_manifest(ref, p)
        warnings, judgments = self.plan_extra(ref, p)
        plan = Plan(
            op=self.id,
            op_version=self.version,
            params=p.model_dump(mode="json"),
            dataset=ref.name,
            input_version=ref.head,
            changes=changes,
            cost=Cost(n_samples=len(sids), est_seconds=self.estimate_seconds(ref, sids, p)),
            warnings=warnings,
            judgments=judgments,
            determinism=self.determinism,
            tolerance=self.tolerance,
            noop=not changes and not self.produces_assertions,
        )
        plan.summary = self.summarize(plan)
        log.debug("plan %s in %.3fs", self.id, t())
        return plan

    def estimate_seconds(self, ref: DatasetRef, sids: list[str], params: Any) -> float | None:
        return None

    def preview(self, dataset: Any, params: Any, *, sample: Selection | None = None) -> Preview:
        ref = as_ref(dataset)
        p = self.coerce(params)
        sel = sample or Selection()
        t = Stopwatch()
        plan = self.plan(ref, p)
        planned = self.plan_samples(ref, p) if self.per_sample else []
        sids = list(sel.samples) if sel.samples else planned[: sel.max_samples]
        sids = [s for s in sids if s in planned][: sel.max_samples]
        diffs: list[FieldDiff] = []
        renders: list[Render] = []
        notes: list[str] = []
        stats: dict[str, Any] = {}
        truncated = False
        outcomes: list[SampleOutcome] = []
        with tempfile.TemporaryDirectory(prefix="fprep-preview-") as tmp:
            scratch = Path(tmp) / "ds"
            _scratch_dataset(ref, scratch, sids)
            for sid in sids:
                if t() > sel.time_budget_s:
                    truncated = True
                    notes.append(f"time budget {sel.time_budget_s}s hit before {sid}")
                    break
                before = ref.editor(sid)
                after = SampleEditor(scratch / "samples" / sid, sid)
                try:
                    out = self.apply_sample(after, p, seed=0)
                except Exception as exc:  # noqa: BLE001 - previews report, never raise
                    notes.append(f"{sid}: {type(exc).__name__}: {exc}")
                    continue
                try:
                    note = _follow_lods(after, out, max_faces=PREVIEW_LOD_MAX_FACES)
                except Exception as exc:  # noqa: BLE001
                    note = f"LOD rebuild failed: {type(exc).__name__}: {exc}"
                if note:
                    notes.append(f"{sid}: {note}")
                outcomes.append(out)
                diffs += _diff_changes(before, after, out.changes, sel)
                stats[sid] = out.stats
                if sel.renders:
                    try:
                        renders += self.renders(before, after, p, out.changes)
                    except Exception as exc:  # noqa: BLE001
                        notes.append(f"{sid}: render failed: {exc}")
            if not self.per_sample or outcomes:
                try:
                    m = read_manifest(scratch)
                    mchanges, mstats = self.apply_manifest(DatasetRef(scratch), m, p, outcomes)
                    if mstats:
                        stats["manifest"] = mstats
                    if mchanges and not plan.changes:
                        plan.changes = mchanges
                except Exception as exc:  # noqa: BLE001
                    notes.append(f"manifest step: {type(exc).__name__}: {exc}")
        return Preview(
            plan=plan,
            samples=sids,
            diffs=diffs,
            stats=stats,
            renders=renders,
            elapsed_s=t(),
            truncated=truncated,
            notes=notes,
        )

    def run(self, dataset: Any, params: Any, *, ctx: Context | None = None) -> Result:
        ref = as_ref(dataset)
        p = self.coerce(params)
        c = ctx or Context()
        started = now_iso()
        t = Stopwatch()
        store = ref.store
        if not store.initialized:
            store.init("baseline (auto, before first op)")
        if c.commit and not c.dry_run:
            st = store.status()
            if not st.clean:
                raise RuntimeError(
                    f"working tree of {ref.root} has uncommitted changes "
                    f"({len(st.added)} added, {len(st.removed)} removed, "
                    f"{len(st.modified)} modified); commit them with "
                    "`python -m frontier_data versions commit` or revert"
                )
        plan = self.plan(ref, p)
        plan_s = t()
        if plan.blocking:
            raise RuntimeError("plan is blocked: " + "; ".join(w.message for w in plan.blocking))
        input_version = ref.head
        if plan.noop:
            return Result(
                op=self.id,
                op_version=self.version,
                params=plan.params,
                dataset=ref.name,
                input_version=input_version,
                output_version=input_version,
                noop=True,
                summary=f"{self.id}: nothing to do",
                determinism=self.determinism,
                tolerance=self.tolerance,
                actor=c.actor,
                machine=Machine.local(),
                timing=Timing(started_at=started, elapsed_s=t(), plan_s=plan_s, workers=c.workers),
            )
        if c.dry_run:
            return Result(
                op=self.id,
                op_version=self.version,
                params=plan.params,
                dataset=ref.name,
                input_version=input_version,
                output_version=None,
                changes=plan.changes,
                summary="dry run: " + plan.summary,
                determinism=self.determinism,
                tolerance=self.tolerance,
                actor=c.actor,
            )
        sids = self.plan_samples(ref, p) if self.per_sample else []
        t_apply = Stopwatch()
        outcomes = self._apply_all(ref, p, sids, c)
        apply_s = t_apply()
        manifest = read_manifest(ref.root)
        mchanges, mstats = self.apply_manifest(ref, manifest, p, outcomes)
        manifest = sync_manifest_fields(ref.root, manifest)
        rebuilt = {o.sample_id: sorted(o.lods_rebuilt) for o in outcomes if o.lods_rebuilt}
        for e in manifest.samples:
            if e.id in rebuilt:
                e.lod_levels = sorted({0, *ref.editor(e.id).lod_levels_on_disk()})
        tolerance = self.tolerance
        if rebuilt:
            from frontier_prep.ops.mesh import LOD_TOLERANCE

            tolerance = {**(tolerance or {}), **{f"lod_{k}": v for k, v in LOD_TOLERANCE.items()}}
        if not self.read_only or mchanges or rebuilt:
            save_manifest(ref.root, manifest)
        changes = [ch for o in outcomes for ch in o.changes] + mchanges
        assertions = [a for o in outcomes for a in o.assertions]
        assertions += self.assertions(ref, p, outcomes, mstats)
        warnings = [w for o in outcomes for w in o.warnings] + plan.warnings
        stats: dict[str, Any] = {"samples": {o.sample_id: o.stats for o in outcomes if o.stats}}
        stats.update(mstats)
        if rebuilt:
            # timings stay out of the provenance record (they are not content)
            stats["lods_rebuilt"] = {
                "samples": len(rebuilt),
                "levels": sorted({lvl for v in rebuilt.values() for lvl in v}),
                "seconds": round(
                    sum(st["seconds"] for o in outcomes for st in o.lods_rebuilt.values()), 3
                ),
            }
        # commit
        t_commit = Stopwatch()
        output_version = input_version
        provenance_id = None
        content_hashes: dict[str, str] = {}
        if c.commit:
            rec = store.append_provenance(
                {
                    "kind": "op",
                    "op": self.id,
                    "op_version": self.version,
                    "params": plan.params,
                    "dataset": ref.name,
                    "input_version": input_version,
                    "determinism": self.determinism,
                    "tolerance": tolerance,
                    # which kernel implementation ran (native Rust or the numpy reference)
                    # and its version; the two are bitwise equal, this is for the audit trail
                    "kernels": kernels.describe(),
                    **({"lods_rebuilt": rebuilt} if rebuilt else {}),
                    "summary": plan.summary,
                    "actor": c.actor.model_dump(),
                    "message": c.message,
                    "n_changes": len(changes),
                    "changes": [ch.model_dump() for ch in changes[:MAX_RECORDED_CHANGES]],
                    "stats": _small(mstats),
                }
            )
            provenance_id = rec
            v = store.commit(
                c.message or plan.summary,
                op_record=rec,
                actor=c.actor.model_dump(),
                workers=max(2, c.workers),
            )
            if v is not None:
                output_version = v.id
                store.append_provenance(
                    {"kind": "note", "ref": rec, "output_version": v.id, "note": "output"}
                )
                content_hashes = _changed_hashes(store, input_version, v.id)
            for a in assertions:
                store.append_provenance(
                    {
                        "kind": "assertion",
                        "version": output_version,
                        "check": a.check,
                        "status": a.status,
                        "target": a.target,
                        "detail": _small(a.detail),
                        "op_record": rec,
                        "actor": c.actor.model_dump(),
                    }
                )
        commit_s = t_commit()
        result = Result(
            op=self.id,
            op_version=self.version,
            params=plan.params,
            dataset=ref.name,
            input_version=input_version,
            output_version=output_version,
            changes=changes,
            assertions=assertions,
            warnings=warnings,
            determinism=self.determinism,
            tolerance=tolerance,
            content_hashes=content_hashes,
            timing=Timing(
                started_at=started,
                elapsed_s=t(),
                plan_s=plan_s,
                apply_s=apply_s,
                commit_s=commit_s,
                workers=c.workers,
                bytes_read=sum(o.bytes_read for o in outcomes),
                bytes_written=sum(o.bytes_written for o in outcomes),
            ),
            machine=Machine.local(),
            actor=c.actor,
            provenance_id=provenance_id,
            noop=output_version == input_version and not assertions,
            stats=stats,
        )
        result.summary = self.summarize(result)
        if c.commit and output_version != input_version:
            self.after_commit(ref, p, result)
        return result

    def explain(self, result: Result) -> Explanation:
        steps = [f"Input version {result.input_version or '(unversioned)'}."]
        by_action: dict[str, int] = {}
        for ch in result.changes:
            by_action[ch.action] = by_action.get(ch.action, 0) + 1
        for action, n in sorted(by_action.items()):
            examples = [ch.target for ch in result.changes if ch.action == action][:3]
            steps.append(f"{n} {action} change(s), e.g. {', '.join(examples)}.")
        for a in result.assertions[:20]:
            steps.append(f"Assertion {a.check} on {a.target}: {a.status}.")
        if result.timing:
            steps.append(
                f"Ran in {result.timing.elapsed_s:.2f}s with {result.timing.workers} worker(s); "
                f"read {result.timing.bytes_read / 1e6:.1f} MB, wrote "
                f"{result.timing.bytes_written / 1e6:.1f} MB."
            )
        steps.append(
            f"Determinism: {result.determinism}"
            + (f" with tolerance {result.tolerance}" if result.tolerance else "")
            + f". Output version {result.output_version}."
        )
        return Explanation(
            op=result.op,
            summary=result.summary or f"{result.op} produced {result.output_version}",
            steps=steps,
            machine={
                "params": result.params,
                "n_changes": len(result.changes),
                "assertions": [a.model_dump() for a in result.assertions],
                "stats": _small(result.stats),
            },
        )

    # -- helpers ---------------------------------------------------------------------------------

    def coerce(self, params: Any) -> Any:
        if isinstance(params, self.Params):
            return params
        if params is None:
            return self.Params()
        if isinstance(params, Params):
            return self.Params.model_validate(params.model_dump())
        return self.Params.model_validate(params)

    def _apply_all(
        self, ref: DatasetRef, p: Any, sids: list[str], c: Context
    ) -> list[SampleOutcome]:
        if not sids:
            return []
        seeds = {sid: _sample_seed(c.seed, sid) for sid in sids}
        outcomes: dict[str, SampleOutcome] = {}
        done = 0
        if c.workers <= 1 or len(sids) == 1:
            for sid in sids:
                outcomes[sid] = self._apply_one(ref.root, sid, p, seeds[sid])
                done += 1
                _progress(c, done, len(sids), sid)
        else:
            params_json = p.model_dump(mode="json")
            n_workers = min(c.workers, len(sids))
            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=get_context("spawn"),
                initializer=_init_worker,
                initargs=(max(1, (os.cpu_count() or 1) // n_workers),),
            ) as ex:
                futs = {
                    ex.submit(_worker, self.id, params_json, str(ref.root), sid, seeds[sid]): sid
                    for sid in sids
                }
                for fut in futs:
                    sid = futs[fut]
                    outcomes[sid] = SampleOutcome.model_validate(fut.result())
                    done += 1
                    _progress(c, done, len(sids), sid)
        return [outcomes[sid] for sid in sids]  # planned (sorted) order, not completion order

    def _apply_one(self, root: Path, sid: str, p: Any, seed: int) -> SampleOutcome:
        ed = SampleEditor(Path(root) / "samples" / sid, sid)
        t = Stopwatch()
        out = self.apply_sample(ed, p, seed=seed)
        _follow_lods(ed, out)
        out.elapsed_s = t()
        out.bytes_read = out.bytes_read or ed.bytes_read
        out.bytes_written = out.bytes_written or ed.bytes_written
        return out


def _follow_lods(
    ed: SampleEditor, out: SampleOutcome, *, max_faces: int | None = None
) -> str | None:
    """Rebuild the stored LOD levels of a sample whose level 0 the op just changed, and add
    one ``modify`` change per level to the outcome. Surface levels are copies of geometry and
    fields, so any level-0 array change makes them stale; mesh levels are vertex maps, stale
    only when the geometry changed. Returns a note when the rebuild was skipped (previews of
    large meshes), else ``None``."""
    if not ed.lods_stale:
        return None
    try:
        kind = ed.kind
        levels = ed.lod_levels_on_disk()
    except (FileNotFoundError, KeyError, ValueError):
        return None
    if not levels or kind not in ("surface", "mesh") or (kind == "mesh" and not ed.geometry_stale):
        return None
    if max_faces is not None and int(ed.sample.cells.shape[0]) > max_faces:
        return (
            f"LOD levels {levels} not rebuilt in the preview (level 0 has more than "
            f"{max_faces} faces); `run` rebuilds them"
        )
    from frontier_prep.ops.mesh import rebuild_lods

    built = rebuild_lods(ed)
    out.lods_rebuilt = built
    out.changes += [
        Change(
            target=f"samples/{ed.id}/lod/{lvl}",
            action="modify",
            detail={"reason": "level 0 changed", "n_faces": st["n_faces"]},
        )
        for lvl, st in built.items()
    ]
    return None


def _planned_lod_rebuilds(ed: SampleEditor, planned: list[Change]) -> list[Change]:
    """What a plan says about the stored LOD levels of a sample: a change that rewrites,
    adds or removes level-0 data makes the engine rebuild every stored level from the new
    level 0 (a rename alone is applied to the levels as a directory rename and needs none).
    Listed so that a reviewer sees the full extent of a run before approving it."""
    if not any(ch.action in ("add", "modify", "remove") for ch in planned):
        return []
    try:
        levels = ed.lod_levels_on_disk()
    except OSError:
        return []
    return [
        Change(
            target=f"samples/{ed.id}/lod/{lvl}",
            action="modify",
            detail={"reason": "rebuilt from level 0 if the op changes level 0", "planned": True},
        )
        for lvl in levels
    ]


def _init_worker(kernel_threads: int) -> None:
    """Runs once in every worker process, before the first kernel call: the native kernels
    share the cores with the other workers instead of each starting one thread per core.
    An explicit ``RAYON_NUM_THREADS`` wins. The thread count never changes a result."""
    os.environ.setdefault("RAYON_NUM_THREADS", str(kernel_threads))


def _worker(op_id: str, params: dict[str, Any], root: str, sid: str, seed: int) -> dict[str, Any]:
    from frontier_prep.op import get_op

    op = get_op(op_id)
    return op._apply_one(Path(root), sid, op.coerce(params), seed).model_dump()  # noqa: SLF001


def _progress(c: Context, done: int, total: int, sid: str) -> None:
    if c.progress is not None:
        try:
            c.progress(done, total, sid)
        except Exception:  # noqa: BLE001
            pass


def _sample_seed(seed: int, sid: str) -> int:
    """Per-sample seed independent of worker assignment and sample order."""
    import hashlib

    h = hashlib.sha256(f"{seed}:{sid}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def _small(obj: Any, limit: int = 200_000) -> Any:
    """Keep provenance records bounded: drop giant nested blobs."""
    try:
        text = json.dumps(obj, default=str)
    except TypeError:
        return str(obj)[:limit]
    if len(text) <= limit:
        return json.loads(text)
    return {"truncated": True, "bytes": len(text)}


def _changed_hashes(store: VersionStore, a: str | None, b: str) -> dict[str, str]:
    vb = store.version(b)
    out: dict[str, str] = {}
    if a is None:
        return {"manifest.json": vb.files.get("manifest.json", ["", 0])[0]}
    va = store.version(a)
    for sid, th in sorted(vb.samples.items()):
        if va.samples.get(sid) != th:
            out[f"samples/{sid}"] = th
    for pth, e in sorted(vb.files.items()):
        if va.files.get(pth, [None])[0] != e[0]:
            out[pth] = e[0]
    return out


def _scratch_dataset(ref: DatasetRef, scratch: Path, sids: Iterable[str]) -> None:
    """A preview workspace: a manifest reduced to the selected samples plus those sample
    directories, hardlinked from the working tree (zarr chunks are never modified in place,
    so sharing inodes is safe)."""
    scratch.mkdir(parents=True)
    wanted = list(sids)
    m = read_manifest(ref.root)
    m.samples = [e for e in m.samples if e.id in set(wanted)]
    m.splits = {k: [i for i in v if i in set(wanted)] for k, v in m.splits.items()}
    save_manifest(scratch, m)
    for sid in wanted:
        _link_tree(ref.ds.sample_dir(sid), scratch / "samples" / sid)


def _link_tree(src: Path, dst: Path) -> None:
    import os

    for p in src.rglob("*"):
        rel = p.relative_to(src)
        if p.is_dir():
            (dst / rel).mkdir(parents=True, exist_ok=True)
        elif p.is_file():
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(p, dst / rel)
            except OSError:
                shutil.copyfile(p, dst / rel)


def _diff_changes(
    before: SampleEditor, after: SampleEditor, changes: list[Change], sel: Selection
) -> list[FieldDiff]:
    """Statistical diffs for every array a change names (``…/fields/<name>``, ``…/points``)."""
    out: list[FieldDiff] = []
    seen: set[str] = set()
    for ch in changes:
        target = ch.target
        rel = target.split(f"samples/{before.id}/", 1)[-1] if "samples/" in target else target
        if rel in seen or ch.action == "assert":
            continue
        seen.add(rel)
        a = _read_target(before, rel, sel) if ch.action != "add" else None
        b = _read_target(after, rel, sel) if ch.action != "remove" else None
        if a is None and b is None:
            continue
        st = compare_arrays(a, b, sel.max_points)
        out.append(
            FieldDiff(
                target=target,
                n_before=st.get("n_before"),
                n_after=st.get("n_after"),
                n_changed=st.get("n_changed"),
                frac_changed=st.get("frac_changed"),
                max_abs_diff=st.get("max_abs_diff"),
                rel_l2_diff=st.get("rel_l2_diff"),
                before=_floats(st.get("before")),
                after=_floats(st.get("after")),
                note=st.get("note"),
            )
        )
    return out


def _floats(d: Any) -> dict[str, float] | None:
    if not isinstance(d, dict):
        return None
    return {k: float(v) for k, v in d.items() if isinstance(v, int | float)}


def _read_target(ed: SampleEditor, rel: str, sel: Selection) -> np.ndarray | None:
    """Read ``fields/<name>``, ``cell_fields/<name>``, ``points``, ``times``, ... bounded by
    ``sel.max_steps`` time steps."""
    try:
        s = ed.reopen()
        if rel.startswith(("fields/", "cell_fields/", "static/")):
            group, name = rel.split("/", 1)
            if group == "static":
                return s.static(name)
            loc = "cell" if group == "cell_fields" else None
            if s.is_time_dependent(name, loc):
                return s.field(name, slice(0, sel.max_steps), location=loc)
            return s.field(name, location=loc)
        if rel in ("points", "faces", "cells", "times", "cell_area", "cell_types"):
            g: Any = s.group
            return np.asarray(g[rel][...]) if rel in g else None
        if rel.startswith("signals/"):
            return s.signal(rel.split("/", 1)[1])
    except (KeyError, FileNotFoundError):
        return None
    return None

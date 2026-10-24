# SPDX-License-Identifier: Apache-2.0
"""Quality ops: ``prep.quality.check`` wraps the frontier-data checks (closure checks first:
integrated forces vs reported, area vs STL, temporal continuity, frame hashes) as
assertions; ``prep.quality.gate`` decides from the recorded assertions whether a version may
be tagged ``train-ready``."""

from __future__ import annotations

import hashlib
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field

from arrayroute.ordered import update_digest
from frontier_data.dataset import Dataset, Manifest
from frontier_prep.base import BaseOp, DatasetRef, SampleOutcome
from frontier_prep.io import SampleEditor
from frontier_prep.op import Assertion, Change, Params, Result, register

#: closure checks: physics-level consistency, the detectors that found nearly every real defect
CLOSURE_CHECKS = (
    "force_consistency",
    "force_closure",
    "area_sum",
    "area_conservation",
    "mesh_cells",
    "freestream_consistency",
    "temporal_continuity",
    "initial_frame_duplicates",
    "duplicate_frames",
    "conservation",
    # a steady field inside its boundary values (maximum principle), declared per sample
    "field_bounds",
)
#: closure checks that describe what a transform did (recorded by the op that made the change);
#: they are not a property of the data that a later change can invalidate
TRANSFORM_CHECKS = frozenset({"area_conservation", "lod_area_conservation", "duplicate_frames"})
#: schema/sanity checks
BASIC_CHECKS = ("nan_inf", "nan_inf_per_step", "degenerate_faces", "watertight", "times_monotonic")
#: dataset-level checks a train-ready version must have
DATASET_CHECKS = ("split_leakage", "trajectory_split_leakage", "duplicate_geometry")

#: key of the pickled per-sample statistics inside ``SampleOutcome.stats`` (worker -> main)
_SAMPLE_STATS = "_quality_sample_stats"


@lru_cache(maxsize=4)
def _dataset(root: str, manifest_mtime_ns: int) -> Dataset:
    """The dataset a sample belongs to, opened once per worker process and manifest state
    (the manifest of a large dataset is not parsed again for every sample)."""
    return Dataset(root)


def _series_points(ds: Dataset) -> int:
    """``frontier_data.quality.series_points_for`` once per opened dataset."""
    cached = getattr(ds, "_prep_series_points", None)
    if cached is None:
        from frontier_data.quality import series_points_for

        cached = series_points_for(ds)
        ds._prep_series_points = cached  # type: ignore[attr-defined]  # noqa: SLF001
    return int(cached)


class CheckParams(Params):
    mean_outlier_action: Literal["fail", "warn"] = Field(
        default="fail",
        description=(
            "Severity of dataset-wide robust mean outliers. Use warn for measured sensor "
            "series with seasonal or operational shifts; finite-value, continuity and split "
            "checks remain blocking. The policy is recorded with each outlier assertion."
        ),
    )
    write_report: bool = Field(
        default=True, description="Write quality/report.json and report.html."
    )
    frame_hashes: bool = Field(
        default=True,
        description=(
            "Hash the first frame of transient samples to find duplicated initial conditions."
        ),
    )
    html: bool = True


@register
class QualityCheck(BaseOp):
    id = "prep.quality.check"
    version = "1.1.0"
    category = "quality"
    summary = (
        "Run every frontier-data quality check (closure checks first) and record each result as an "
        "assertion on the current version."
    )
    when_to_use = (
        "After ingest and after any op that touches arrays; the gate reads these assertions."
    )
    produces_assertions = True
    per_sample = True
    changes_manifest = True
    # the report carries timings and a timestamp: assertions are bitwise, the file is not
    determinism = "tolerance"
    tolerance: ClassVar[dict[str, float] | None] = {"report_timing_fields": 0.0}
    Params = CheckParams

    def plan_samples(self, ref: DatasetRef, params: CheckParams) -> list[str]:
        # every sample: the per-sample statistics of the quality report are computed in the
        # worker pool (``apply_sample``) and merged in sample order (``apply_manifest``)
        return ref.sample_ids()

    def plan_manifest(self, ref: DatasetRef, params: CheckParams) -> list[Change]:
        return [Change(target="quality/report.json", action="modify")]

    def apply_sample(self, ed: SampleEditor, params: CheckParams, *, seed: int) -> SampleOutcome:
        stats: dict[str, Any] = {}
        root = ed.dir.parent.parent
        transient = True  # without a manifest entry, hash as before
        if (root / "manifest.json").is_file():
            from frontier_data.quality import compute_sample_stats

            ds = _dataset(str(root), (root / "manifest.json").stat().st_mtime_ns)
            try:
                transient = ds.entry(ed.id).temporal == "transient"
            except KeyError:
                ds = None
            if ds is not None:
                # the per-sample pass of ``run_quality``, here so it runs in the worker
                # pool. A dataclass tree with arrays in it: it crosses the process boundary
                # pickled (as every worker result does) and ``apply_manifest`` takes it out
                # of the outcome again, so it never reaches a record.
                st = compute_sample_stats(ds, ed.id, series_points=_series_points(ds))
                stats[_SAMPLE_STATS] = pickle.dumps(st, protocol=pickle.HIGHEST_PROTOCOL)
        if params.frame_hashes and transient:
            s = ed.sample
            h = hashlib.sha256()
            for name in s.field_names():
                loc = "cell" if name in s.field_names("cell") else None
                if s.is_time_dependent(name, loc):
                    array = s.array(name, location=loc)
                    update_digest(
                        h,
                        s._reader(array),
                        (0, *(slice(None) for _ in array.shape[1:])),
                        stats=s.read_stats,
                    )
            stats["frame0"] = h.hexdigest()[:16]
        return SampleOutcome(sample_id=ed.id, stats=stats)

    def apply_manifest(
        self,
        ref: DatasetRef,
        manifest: Manifest,
        params: CheckParams,
        outcomes: list[SampleOutcome],
    ) -> tuple[list[Change], dict[str, Any]]:
        import tempfile

        from frontier_data.quality import run_quality

        ref.ds.manifest = manifest
        # per-sample statistics computed by the workers; a sample without one (a preview
        # covers a few samples only) is computed by ``run_quality`` itself
        precomputed = {
            o.sample_id: pickle.loads(o.stats.pop(_SAMPLE_STATS))  # noqa: S301 - own workers
            for o in outcomes
            if _SAMPLE_STATS in o.stats
        }
        if params.write_report:
            report = run_quality(
                ref.ds,
                out_dir=ref.root / "quality",
                write_html=params.html,
                precomputed=precomputed,
                mean_outlier_action=params.mean_outlier_action,
                dataset_location=".",
            )
            manifest.quality = "quality/report.json"
        else:
            with tempfile.TemporaryDirectory() as tmp:
                report = run_quality(
                    ref.ds,
                    out_dir=Path(tmp),
                    write_html=False,
                    precomputed=precomputed,
                    mean_outlier_action=params.mean_outlier_action,
                    dataset_location=".",
                )
        hashed = [o for o in outcomes if "frame0" in o.stats]
        groups: dict[str, list[str]] = {}
        for o in hashed:
            groups.setdefault(o.stats["frame0"], []).append(o.sample_id)
        dup = [v for v in groups.values() if len(v) > 1]
        assertions: list[dict[str, Any]] = []
        for row in report.get("samples", []):
            sid = row.get("id") or row.get("sample_id")
            for c in row.get("checks", []):
                assertions.append(
                    {
                        "check": str(c["check"]),
                        "status": _status(c["status"]),
                        "target": f"samples/{sid}",
                        "detail": _trim(c.get("detail", {})),
                    }
                )
        for c in report.get("dataset_checks", report.get("checks", [])):
            assertions.append(
                {
                    "check": str(c["check"]),
                    "status": _status(c["status"]),
                    "target": "dataset",
                    "detail": _trim(c.get("detail", {})),
                }
            )
        if hashed:
            assertions.append(
                {
                    "check": "initial_frame_duplicates",
                    "status": "warn" if dup else "pass",
                    "target": "dataset",
                    "detail": {"groups": dup[:20], "n_groups": len(dup)},
                }
            )
        stats = {
            "report": _report_summary(report),
            "initial_frame_duplicates": dup,
            "assertions": assertions,
        }
        changes = [Change(target="quality/report.json", action="modify", detail=stats["report"])]
        return changes if params.write_report else [], stats

    def assertions(
        self,
        ref: DatasetRef,
        params: CheckParams,
        outcomes: list[SampleOutcome],
        stats: dict[str, Any],
    ) -> list[Assertion]:
        return [Assertion.model_validate(a) for a in stats.get("assertions", [])]


def _status(s: Any) -> Literal["pass", "warn", "fail", "skip"]:
    return s if s in ("pass", "warn", "fail", "skip") else "skip"


def _trim(d: Any, limit: int = 4000) -> dict[str, Any]:
    import json

    if not isinstance(d, dict):
        return {"value": d}
    text = json.dumps(d, default=str)
    if len(text) <= limit:
        return d
    return {k: v for k, v in d.items() if not isinstance(v, list | dict)} | {"truncated": True}


def _report_summary(report: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in report.get("samples", []):
        for c in row.get("checks", []):
            counts[c["status"]] = counts.get(c["status"], 0) + 1
    return {"status_counts": counts, "n_samples": len(report.get("samples", []))}


# -- quality.gate ------------------------------------------------------------------------------


class GateParams(Params):
    required: list[str] | None = Field(
        default=None,
        description=(
            "Checks that must have been recorded (closure checks first). Default: the checks "
            "that apply to the dataset: area_sum for surfaces, force_consistency and "
            "force_closure when samples carry forces, temporal_continuity for transient data, "
            "mesh_cells for steady meshes, and predictable_fields, nan_inf and split_leakage "
            "for every dataset."
        ),
    )
    closure_required: bool = Field(
        default=True,
        description=(
            "At least one closure check (force/area/continuity/frame hash) must be present and not "
            "failing."
        ),
    )
    allow_warn: bool = Field(default=True, description="warn statuses do not block.")
    max_fail_frac: float = Field(
        default=0.0, description="Fraction of samples with failing checks tolerated (0 = none)."
    )
    tag: str = "train-ready"
    approver: str | None = Field(default=None, description="Record a human approval with the tag.")


@register
class QualityGate(BaseOp):
    id = "prep.quality.gate"
    version = "1.0.0"
    category = "quality"
    summary = (
        "Decide from recorded assertions whether this version is train-ready: closure checks "
        "present and passing, no failing samples beyond the tolerance; tags the version."
    )
    when_to_use = (
        "Last step of a cleaning session; the trainer and the hub refuse datasets without the tag. "
        "A failing gate names the assertions to fix."
    )
    produces_assertions = True
    per_sample = False
    read_only = True
    changes_manifest = True
    Params = GateParams

    def plan_manifest(self, ref: DatasetRef, params: GateParams) -> list[Change]:
        return [Change(target="manifest/stats/gate", action="modify", detail={"tag": params.tag})]

    @staticmethod
    def required_for(ref: DatasetRef) -> list[str]:
        """The checks that apply to this dataset's kind: a force closure cannot be asked of a
        dataset without forces, nor a time continuity of a steady one."""
        m = ref.manifest
        required: list[str] = []
        if m.sample_kind == "surface":
            if any(e.forces for e in m.samples):
                required += ["force_consistency", "force_closure"]
            required.append("area_sum")
        if m.temporal == "transient":
            required.append("temporal_continuity")
        elif m.sample_kind == "mesh":
            required.append("mesh_cells")
        return [*required, "predictable_fields", "nan_inf", "split_leakage"]

    def evaluate(self, ref: DatasetRef, params: GateParams) -> dict[str, Any]:
        required = params.required if params.required is not None else self.required_for(ref)
        recs = ref.store.effective_assertions("HEAD") if ref.store.initialized else []
        by_check: dict[str, list[dict[str, Any]]] = {}
        for r in recs:
            if r.get("check") == "train_ready":
                continue  # earlier gate verdicts are not evidence
            by_check.setdefault(str(r.get("check")), []).append(r)
        present = set(by_check)
        missing = [c for c in required if c not in present and not _alias_present(c, present)]
        failing_samples: set[str] = set()
        failing_checks: dict[str, int] = {}
        for check, rs in by_check.items():
            for r in rs:
                st = r.get("status")
                if st == "fail" or (st == "warn" and not params.allow_warn):
                    failing_checks[check] = failing_checks.get(check, 0) + 1
                    t = str(r.get("target", ""))
                    if t.startswith("samples/"):
                        failing_samples.add(t.split("/")[1])
        n = max(len(ref.manifest.samples), 1)
        closure_present = [c for c in CLOSURE_CHECKS if c in present]
        closure_failing = [c for c in closure_present if failing_checks.get(c)]
        not_evaluated = self.not_evaluated(ref, params, recs)
        ok = not missing and len(failing_samples) / n <= params.max_fail_frac
        if not_evaluated:
            ok = False  # fail closed: evidence that lapsed is not a pass
        if params.closure_required and (not closure_present or closure_failing):
            ok = False
        dataset_fail = [
            c
            for c in failing_checks
            if any(
                str(r.get("target")) == "dataset" for r in by_check[c] if r.get("status") == "fail"
            )
        ]
        if dataset_fail:
            ok = False
        return {
            "ok": ok,
            "required": required,
            "missing": missing,
            "failing_checks": failing_checks,
            "failing_samples": sorted(failing_samples)[:50],
            "n_failing_samples": len(failing_samples),
            "closure_present": closure_present,
            "closure_failing": closure_failing,
            "not_evaluated": not_evaluated,
            "dataset_failing": dataset_fail,
            "n_assertions": len(recs),
            "version": ref.head,
        }

    def summarize(self, pr: Any) -> str:
        gate = (getattr(pr, "stats", None) or {}).get("gate")
        if not gate:
            return f"{self.id}: evaluate the recorded assertions"
        if gate.get("ok"):
            return f"{self.id}: train-ready ({gate.get('n_assertions')} effective assertions)"
        why = []
        if gate.get("missing"):
            why.append("required checks never recorded: " + ", ".join(gate["missing"]))
        for check, targets in (gate.get("not_evaluated") or {}).items():
            why.append(
                f"{check} is not evaluated for {len(targets)} target(s) (e.g. {targets[0]}): "
                "the data changed after it was last checked; run the check again"
            )
        if gate.get("failing_checks"):
            why.append("failing: " + ", ".join(sorted(gate["failing_checks"])))
        if not gate.get("closure_present"):
            why.append("no closure check present")
        return f"{self.id}: not train-ready; " + "; ".join(why)

    def not_evaluated(
        self, ref: DatasetRef, params: GateParams, effective: list[dict[str, Any]]
    ) -> dict[str, list[str]]:
        """Evidence that existed in this version's history and no longer holds for its content:
        ``{check: [targets]}``. A later op that changes a sample (an LOD rebuild, a unit
        conversion) invalidates the assertions made on it. When the lapsed record is a closure
        check, or its last verdict was a failure, the gate treats the check as *not evaluated*
        and fails until it is run again: a failing force closure must never turn into a pass
        because the record that held it stopped applying."""
        if not ref.store.initialized:
            return {}
        have = {(str(r.get("check")), str(r.get("target", "dataset"))) for r in effective}
        order = {v.id: i for i, v in enumerate(ref.store.log())}  # 0 = HEAD, ancestors after
        latest: dict[tuple[str, str], dict[str, Any]] = {}
        for r in ref.store.provenance("assertion"):
            age = order.get(str(r.get("version")))
            check = str(r.get("check"))
            if age is None or check == "train_ready":
                continue
            key = (check, str(r.get("target", "dataset")))
            cur = latest.get(key)
            if cur is None or age <= order[str(cur.get("version"))]:
                latest[key] = r
        samples = {e.id for e in ref.manifest.samples}
        out: dict[str, list[str]] = {}
        for (check, target), r in sorted(latest.items()):
            if (check, target) in have or _alias_have(check, target, have):
                continue
            if target.startswith("samples/") and target.split("/")[1] not in samples:
                continue  # the sample is gone
            status = r.get("status")
            failed = status == "fail" or (status == "warn" and not params.allow_warn)
            verifies = check in CLOSURE_CHECKS and check not in TRANSFORM_CHECKS
            if failed or verifies:
                out.setdefault(check, []).append(target)
        return {k: v[:50] for k, v in out.items()}

    def apply_manifest(
        self, ref: DatasetRef, manifest: Manifest, params: GateParams, outcomes: list[SampleOutcome]
    ) -> tuple[list[Change], dict[str, Any]]:
        res = self.evaluate(ref, params)
        manifest.stats["gate"] = {**res, "tag": params.tag if res["ok"] else None}
        if res["ok"] and ref.store.initialized:
            ref.store.tag(params.tag, "HEAD")
            if params.approver:
                ref.store.approve("HEAD", params.approver, note=f"gate {params.tag}")
        elif (
            ref.store.initialized
            and params.tag in ref.store.tags()
            and ref.store.tags()[params.tag] == ref.head
        ):
            ref.store.untag(params.tag)
        return [
            Change(
                target="manifest/stats/gate",
                action="modify",
                detail={"ok": res["ok"], "tag": params.tag if res["ok"] else None},
            )
        ], {"gate": res}

    def after_commit(self, ref: DatasetRef, params: GateParams, result: Result) -> None:
        """The gate evaluated its input version and tagged it; its own commit (stats.gate in
        the manifest, the assertion) is the new head, which differs from the tagged content
        only by the gate record itself. Move the tag so HEAD carries it and a reader can ask
        `tags()[tag] == head`."""
        if (
            result.output_version
            and ref.store.initialized
            and ref.store.tags().get(params.tag) == result.input_version
        ):
            ref.store.tag(params.tag, result.output_version)

    def assertions(
        self,
        ref: DatasetRef,
        params: GateParams,
        outcomes: list[SampleOutcome],
        stats: dict[str, Any],
    ) -> list[Assertion]:
        res = stats.get("gate", {})
        return [
            Assertion(
                check="train_ready",
                status="pass" if res.get("ok") else "fail",
                target="dataset",
                detail=res,
            )
        ]


def _alias_have(check: str, target: str, have: set[tuple[str, str]]) -> bool:
    """`nan_inf` and `nan_inf_per_step`, `split_leakage` and `trajectory_split_leakage` are
    the same question asked of steady and transient data: either one counts as evaluated. A
    force closure is not replaced by the weaker force consistency."""
    same = {
        "nan_inf": "nan_inf_per_step",
        "nan_inf_per_step": "nan_inf",
        "split_leakage": "trajectory_split_leakage",
        "trajectory_split_leakage": "split_leakage",
    }
    other = same.get(check)
    return other is not None and (other, target) in have


def _alias_present(check: str, present: set[str]) -> bool:
    aliases = {
        "force_consistency": {"force_closure"},
        "force_closure": {"force_consistency"},
        "nan_inf": {"nan_inf_per_step"},
        "split_leakage": {"trajectory_split_leakage"},
        "area_sum": {"area_conservation"},
    }
    return bool(aliases.get(check, set()) & present)

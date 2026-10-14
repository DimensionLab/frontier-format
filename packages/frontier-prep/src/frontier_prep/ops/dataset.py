# SPDX-License-Identifier: Apache-2.0
"""Dataset-level ops: dedup, split (with leakage proof), coverage (with DoE suggestions),
balance, select, merge, summarize. Most work on the manifest; dedup fingerprints and
coverage read small per-sample arrays."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field

from frontier_data.dataset import Manifest, SampleEntry, write_json
from frontier_data.splits import find_leakage, make_split
from frontier_prep.base import BaseOp, DatasetRef, SampleOutcome
from frontier_prep.io import SampleEditor, read_manifest, write_text_atomic
from frontier_prep.op import Assertion, Change, OpWarning, Params, register


def _natural(s: str) -> tuple[Any, ...]:
    import re

    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s))


def _remove_samples(
    ref: DatasetRef, manifest: Manifest, ids: set[str], *, delete_dirs: bool
) -> None:
    manifest.samples = [e for e in manifest.samples if e.id not in ids]
    for k in list(manifest.splits):
        manifest.splits[k] = [i for i in manifest.splits[k] if i not in ids]
    for fam in manifest.split_families.values():
        if isinstance(fam, dict) and "splits" in fam:
            fam["splits"] = {k: [i for i in v if i not in ids] for k, v in fam["splits"].items()}
    if delete_dirs:
        for sid in ids:
            d = ref.root / "samples" / sid
            if d.exists():
                shutil.rmtree(d)
    manifest.stats["n_samples"] = len(manifest.samples)


# -- dataset.dedup -----------------------------------------------------------------------------


class DedupParams(Params):
    action: Literal["report", "drop"] = "report"
    exact: bool = True
    near: bool = Field(
        default=True, description="Near-duplicates by geometry invariants and point-cloud distance."
    )
    invariant_tol: float = Field(
        default=0.01, description="Relative tolerance on area/bbox/moments (1 % rule)."
    )
    chamfer_tol: float = Field(
        default=1e-3, description="Chamfer distance / bbox diagonal threshold."
    )
    n_probe: int = Field(default=2000, description="Seeded probe points for the distance test.")
    compare_parameters: bool = Field(
        default=True,
        description=(
            "Equal parameters make an exact duplicate; different parameters with the same geometry "
            "are a 'rerun' judgment."
        ),
    )
    delete_dirs: bool = True


@register
class Dedup(BaseOp):
    id = "prep.dataset.dedup"
    version = "1.0.0"
    category = "dataset"
    summary = (
        "Find exact (content hash) and near (geometry invariants + point-cloud distance) duplicate "
        "samples; report groups or drop later members."
    )
    when_to_use = (
        "Campaign archives with re-runs and repacks; before splitting (duplicates across splits "
        "are leakage). Dropping is a policy: report first, decide, then drop."
    )
    changes_manifest = True
    produces_assertions = True
    Params = DedupParams

    def apply_sample(self, ed: SampleEditor, params: DedupParams, *, seed: int) -> SampleOutcome:
        s = ed.sample
        pts = s.points().astype(np.float64)
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(s.points()).tobytes())
        if s.kind in ("surface", "mesh"):
            h.update(np.ascontiguousarray(s.cells).tobytes())
        exact = h.hexdigest()[:16]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        ext = hi - lo
        diag = float(np.linalg.norm(ext)) or 1.0
        c = pts.mean(axis=0)
        cov = np.cov((pts - c).T) if len(pts) > 1 else np.zeros((pts.shape[1],) * 2)
        eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
        inv = {
            "n_points": int(len(pts)),
            "n_cells": int(len(s.cells)) if s.kind != "point_cloud" else 0,
            "extent": (ext / diag).tolist(),
            "eig": (np.sqrt(np.maximum(eig, 0)) / diag).tolist(),
            "area": float(s.attrs.get("area", 0.0)) / (diag * diag),
        }
        rng = np.random.default_rng(
            0
        )  # fixed probe selection: a dataset-wide constant, not per-sample
        probe = rng.choice(len(pts), min(params.n_probe, len(pts)), replace=False)
        norm = ((pts[np.sort(probe)] - lo) / diag).astype(np.float32)
        hp = hashlib.sha256(json.dumps(ed.meta.parameters, sort_keys=True).encode()).hexdigest()[
            :16
        ]
        return SampleOutcome(
            sample_id=ed.id,
            stats={"exact": exact, "params": hp, "inv": inv, "probe": norm.tolist()},
        )

    def apply_manifest(
        self,
        ref: DatasetRef,
        manifest: Manifest,
        params: DedupParams,
        outcomes: list[SampleOutcome],
    ) -> tuple[list[Change], dict[str, Any]]:
        from scipy.spatial import KDTree

        groups: list[dict[str, Any]] = []
        assigned: dict[str, int] = {}
        ids = [o.sample_id for o in outcomes]
        by = {o.sample_id: o.stats for o in outcomes}
        if params.exact:
            seen: dict[str, str] = {}
            for sid in ids:
                key = by[sid]["exact"] + (by[sid]["params"] if params.compare_parameters else "")
                if key in seen:
                    g = assigned[seen[key]]
                    groups[g]["members"].append(sid)
                    assigned[sid] = g
                else:
                    seen[key] = sid
                    assigned[sid] = len(groups)
                    groups.append({"kind": "exact", "members": [sid]})
        else:
            for sid in ids:
                assigned[sid] = len(groups)
                groups.append({"kind": "exact", "members": [sid]})
        if params.near:
            reps = [g["members"][0] for g in groups]
            for i, a in enumerate(reps):
                for j in range(i + 1, len(reps)):
                    b = reps[j]
                    if groups[assigned[a]] is groups[assigned[b]]:
                        continue
                    ia, ib = by[a]["inv"], by[b]["inv"]
                    if not _invariants_close(ia, ib, params.invariant_tol):
                        continue
                    pa, pb = np.asarray(by[a]["probe"]), np.asarray(by[b]["probe"])
                    d1, _ = KDTree(pb).query(pa)
                    d2, _ = KDTree(pa).query(pb)
                    chamfer = float(0.5 * (d1.mean() + d2.mean()))
                    if chamfer <= params.chamfer_tol:
                        ga, gb = assigned[a], assigned[b]
                        same_params = by[a]["params"] == by[b]["params"]
                        groups[ga]["members"] += groups[gb]["members"]
                        groups[ga]["kind"] = "near" if same_params else "rerun?"
                        groups[ga]["chamfer"] = chamfer
                        for m in groups[gb]["members"]:
                            assigned[m] = ga
                        groups[gb] = {"kind": "merged", "members": []}
        dup_groups = [g for g in groups if len(g["members"]) > 1]
        for g in dup_groups:
            g["members"] = sorted(g["members"], key=_natural)
        dropped: list[str] = []
        if params.action == "drop":
            for g in dup_groups:
                if g["kind"] == "rerun?":
                    continue  # a judgment, never dropped automatically
                dropped += g["members"][1:]
            _remove_samples(ref, manifest, set(dropped), delete_dirs=params.delete_dirs)
        manifest.stats["duplicates"] = {
            "groups": dup_groups,
            "dropped": dropped,
            "params": params.model_dump(),
        }
        changes = [
            Change(
                target=f"samples/{sid}", action="remove", detail={"duplicate_of": g["members"][0]}
            )
            for g in dup_groups
            for sid in g["members"][1:]
            if sid in dropped
        ]
        if dup_groups:
            changes.append(
                Change(
                    target="manifest/stats/duplicates",
                    action="modify",
                    detail={"n_groups": len(dup_groups)},
                )
            )
        return changes, {"n_groups": len(dup_groups), "groups": dup_groups, "dropped": dropped}

    def assertions(
        self,
        ref: DatasetRef,
        params: DedupParams,
        outcomes: list[SampleOutcome],
        stats: dict[str, Any],
    ) -> list[Assertion]:
        n = stats.get("n_groups", 0)
        return [
            Assertion(
                check="duplicate_geometry",
                status="pass" if n == 0 else "warn",
                detail={
                    "n_groups": n,
                    "groups": stats.get("groups", [])[:20],
                    "dropped": stats.get("dropped", []),
                },
            )
        ]


def _invariants_close(a: dict[str, Any], b: dict[str, Any], tol: float) -> bool:
    for key in ("extent", "eig"):
        x, y = np.asarray(a[key]), np.asarray(b[key])
        if x.shape != y.shape or np.any(np.abs(x - y) > tol * np.maximum(np.abs(x), 1e-9) + 1e-6):
            return False
    if a["area"] and b["area"] and abs(a["area"] - b["area"]) > tol * max(a["area"], b["area"]):
        return False
    return abs(a["n_points"] - b["n_points"]) <= 0.1 * max(a["n_points"], b["n_points"])


# -- dataset.split -----------------------------------------------------------------------------


class SplitParams(Params):
    strategy: Literal["random", "parameter_holdout", "group"] = "random"
    fractions: list[float] = Field(default_factory=lambda: [0.8, 0.1, 0.1])
    seed: int = 0
    parameter: str | None = Field(default=None, description="parameter_holdout: which parameter.")
    holdout_fraction: float = 0.15
    side: Literal["top", "bottom"] = "top"
    group_key: Literal["group", "geometry_hash"] = Field(
        default="group",
        description=(
            "group: manifest `group` (trajectory id); geometry_hash: samples with identical "
            "geometry."
        ),
    )
    family: str | None = Field(
        default=None, description="Name under split_families (default: strategy)."
    )
    activate: bool = True


@register
class Split(BaseOp):
    id = "prep.dataset.split"
    version = "1.0.0"
    category = "dataset"
    summary = (
        "Assign train/val/test (random, parameter hold-out or group-aware) and prove no leakage: "
        "id overlap, group straddling, duplicate geometry, nearest-neighbour parameter distances."
    )
    when_to_use = (
        "After dedup, before normalisation. Use parameter_holdout to measure extrapolation; group "
        "for trajectories cut into windows or design families."
    )
    changes_manifest = True
    produces_assertions = True
    determinism = "seeded"
    per_sample = False
    Params = SplitParams

    def plan_manifest(self, ref: DatasetRef, params: SplitParams) -> list[Change]:
        return [
            Change(
                target="manifest/splits",
                action="modify",
                detail={"strategy": params.strategy, "seed": params.seed},
            )
        ]

    def plan_extra(self, ref: DatasetRef, params: SplitParams) -> tuple[list[OpWarning], list[Any]]:
        w = []
        if params.strategy == "parameter_holdout" and (
            params.parameter is None
            or params.parameter not in (ref.manifest.parameters + ref.manifest.geometry_params)
        ):
            w.append(
                OpWarning(
                    severity="fail",
                    code="parameter",
                    message=f"parameter_holdout needs a known parameter, got {params.parameter!r}",
                )
            )
        return w, []

    def apply_manifest(
        self,
        ref: DatasetRef,
        manifest: Manifest,
        params: SplitParams,
        outcomes: list[SampleOutcome],
    ) -> tuple[list[Change], dict[str, Any]]:
        ids = sorted((e.id for e in manifest.samples), key=_natural)
        pmap = {e.id: {**e.geometry_params, **e.parameters} for e in manifest.samples}
        fr = tuple(params.fractions)
        if params.strategy == "random":
            res = make_split("random", ids, fractions=fr, seed=params.seed)
        elif params.strategy == "parameter_holdout":
            assert params.parameter is not None
            res = make_split(
                "parameter_holdout",
                ids,
                pmap,
                parameter=params.parameter,
                holdout_fraction=params.holdout_fraction,
                side=params.side,
                seed=params.seed,
            )
        else:
            if params.group_key == "group":
                groups: dict[str, str | None] = {e.id: e.group for e in manifest.samples}
            else:
                groups = {e.id: _geometry_hash(ref, e.id) for e in manifest.samples}
            res = make_split("group", ids, groups=groups, fractions=fr, seed=params.seed)
        proof = leakage_proof(ref, manifest, res.splits, pmap)
        name = params.family or params.strategy
        manifest.split_families[name] = {
            "splits": res.splits,
            "strategy": res.strategy,
            "leakage": proof,
        }
        changes = []
        if params.activate:
            before = {k: len(v) for k, v in manifest.splits.items()}
            manifest.splits = res.splits
            manifest.split_strategy = res.strategy
            changes.append(
                Change(
                    target="manifest/splits",
                    action="modify",
                    before=before,
                    after={k: len(v) for k, v in res.splits.items()},
                )
            )
        changes.append(
            Change(target=f"manifest/split_families/{name}", action="add", detail=res.strategy)
        )
        return changes, {
            "splits": {k: len(v) for k, v in res.splits.items()},
            "leakage": proof,
            "strategy": res.strategy,
        }

    def assertions(
        self,
        ref: DatasetRef,
        params: SplitParams,
        outcomes: list[SampleOutcome],
        stats: dict[str, Any],
    ) -> list[Assertion]:
        proof = stats.get("leakage", {})
        return [Assertion(check="split_leakage", status=proof.get("status", "skip"), detail=proof)]


def _geometry_hash(ref: DatasetRef, sid: str) -> str:
    s = ref.ds.sample(sid)
    h = hashlib.sha256(np.ascontiguousarray(s.points()).tobytes())
    return h.hexdigest()[:16]


def leakage_proof(
    ref: DatasetRef,
    manifest: Manifest,
    splits: dict[str, list[str]],
    pmap: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Id overlap, group straddling, duplicate geometry across splits (from stats.duplicates
    when dedup ran), and nearest-neighbour distance test->train in normalised parameter space
    compared with train->train."""
    out: dict[str, Any] = {"id_overlap": find_leakage(splits)}
    status = "fail" if out["id_overlap"] else "pass"
    where = {sid: k for k, v in splits.items() for sid in v}
    groups: dict[str, set[str]] = {}
    for e in manifest.samples:
        if e.group and e.id in where:
            groups.setdefault(e.group, set()).add(where[e.id])
    straddle = sorted(g for g, ks in groups.items() if len(ks) > 1)
    out["group_straddling"] = straddle
    if straddle:
        status = "fail"
    dup = manifest.stats.get("duplicates", {}).get("groups", [])
    cross = [
        g["members"] for g in dup if len({where.get(m) for m in g["members"] if m in where}) > 1
    ]
    out["duplicates_across_splits"] = cross
    if cross:
        status = "fail"
    keys = sorted({k for p in pmap.values() for k in p})
    train = [s for s in splits.get("train", []) if s in pmap]
    if keys and len(train) >= 2:
        x = np.array(
            [[pmap[s].get(k, np.nan) for k in keys] for s in sorted(pmap)], dtype=np.float64
        )
        lo, hi = np.nanmin(x, axis=0), np.nanmax(x, axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        norm = {s: (np.array([pmap[s].get(k, np.nan) for k in keys]) - lo) / span for s in pmap}
        tr = np.array([norm[s] for s in train])
        tr = np.nan_to_num(tr, nan=0.0)
        from scipy.spatial import KDTree

        tree = KDTree(tr)
        d_tt, _ = tree.query(tr, k=2)
        nn: dict[str, Any] = {"train_to_train_median": float(np.median(d_tt[:, 1]))}
        for k in ("val", "test"):
            ids = [s for s in splits.get(k, []) if s in norm]
            if ids:
                d, _ = tree.query(np.nan_to_num(np.array([norm[s] for s in ids]), nan=0.0))
                nn[f"{k}_to_train_median"] = float(np.median(d))
                nn[f"{k}_to_train_min"] = float(np.min(d))
                nn[f"{k}_exact_matches"] = int(np.sum(d == 0))
                if np.sum(d == 0) and status == "pass":
                    status = "warn"
        out["parameter_nn"] = nn
    out["status"] = status
    return out


# -- dataset.coverage --------------------------------------------------------------------------


class CoverageParams(Params):
    parameters: list[str] | None = None
    bins: int = Field(default=10, ge=2)
    n_suggest: int = Field(
        default=0,
        ge=0,
        description=(
            "Propose this many new parameter points (maximin over a seeded LHS candidate set)."
        ),
    )
    n_candidates: int = 2000
    seed: int = 0
    split: str | None = Field(default=None, description="Restrict to a split (e.g. train).")


@register
class Coverage(BaseOp):
    id = "prep.dataset.coverage"
    version = "1.0.0"
    category = "dataset"
    summary = (
        "Parameter-space coverage: empty bins per parameter, nearest-neighbour gaps, 2-D "
        "projections, and maximin suggestions for the next cases (DoE)."
    )
    when_to_use = (
        "Planning the next simulations, or explaining why hold-out cases are out of distribution."
    )
    changes_manifest = True
    produces_assertions = True
    determinism = "seeded"
    per_sample = False
    read_only = True
    Params = CoverageParams

    def plan_manifest(self, ref: DatasetRef, params: CoverageParams) -> list[Change]:
        return [Change(target="manifest/stats/coverage", action="modify")]

    def apply_manifest(
        self,
        ref: DatasetRef,
        manifest: Manifest,
        params: CoverageParams,
        outcomes: list[SampleOutcome],
    ) -> tuple[list[Change], dict[str, Any]]:
        ids = (
            manifest.splits.get(params.split, [])
            if params.split
            else [e.id for e in manifest.samples]
        )
        pmap = {
            e.id: {**e.geometry_params, **e.parameters}
            for e in manifest.samples
            if e.id in set(ids)
        }
        keys = params.parameters or sorted({k for p in pmap.values() for k in p})
        keys = [k for k in keys if sum(k in p for p in pmap.values()) >= 2]
        if not keys:
            return [], {"coverage": {}}
        sids = sorted(pmap, key=_natural)
        x = np.array([[pmap[s].get(k, np.nan) for k in keys] for s in sids], dtype=np.float64)
        lo, hi = np.nanmin(x, axis=0), np.nanmax(x, axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        u = (x - lo) / span
        per: dict[str, Any] = {}
        for j, k in enumerate(keys):
            col = u[:, j]
            col = col[np.isfinite(col)]
            hist, _ = np.histogram(col, bins=params.bins, range=(0, 1))
            per[k] = {
                "min": float(lo[j]),
                "max": float(hi[j]),
                "empty_bins": int((hist == 0).sum()),
                "histogram": hist.tolist(),
                "std_normalized": float(col.std()) if len(col) else 0.0,
            }
        uf = np.nan_to_num(u, nan=0.0)
        from scipy.spatial import KDTree

        tree = KDTree(uf)
        d, _ = tree.query(uf, k=min(2, len(uf)))
        nn = d[:, 1] if d.ndim == 2 and d.shape[1] > 1 else np.zeros(len(uf))
        gaps = {
            "nn_median": float(np.median(nn)),
            "nn_max": float(nn.max()),
            "most_isolated": [sids[i] for i in np.argsort(-nn)[:5]],
        }
        suggestions = []
        if params.n_suggest:
            rng = np.random.default_rng(params.seed)
            cand = _lhs(params.n_candidates, len(keys), rng)
            pts = uf.copy()
            for _ in range(params.n_suggest):
                dc, _ = KDTree(pts).query(cand)
                j = int(np.argmax(dc))
                pts = np.vstack([pts, cand[j]])
                suggestions.append(
                    {k: float(lo[i] + cand[j, i] * span[i]) for i, k in enumerate(keys)}
                    | {"min_distance": float(dc[j])}
                )
        cov = {
            "parameters": per,
            "gaps": gaps,
            "suggestions": suggestions,
            "n": len(sids),
            "split": params.split,
        }
        manifest.stats["coverage"] = cov
        return [
            Change(target="manifest/stats/coverage", action="modify", detail={"parameters": keys})
        ], {"coverage": cov}

    def assertions(
        self,
        ref: DatasetRef,
        params: CoverageParams,
        outcomes: list[SampleOutcome],
        stats: dict[str, Any],
    ) -> list[Assertion]:
        cov = stats.get("coverage", {})
        empty = {
            k: v["empty_bins"] for k, v in cov.get("parameters", {}).items() if v["empty_bins"]
        }
        return [
            Assertion(
                check="parameter_coverage",
                status="warn" if empty else "pass",
                detail={"empty_bins": empty, "gaps": cov.get("gaps")},
            )
        ]


def _lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    out = np.empty((n, d))
    for j in range(d):
        out[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return out


# -- dataset.balance ---------------------------------------------------------------------------


class BalanceParams(Params):
    parameter: str
    bins: int = Field(default=5, ge=2)
    edges: list[float] | None = Field(
        default=None, description="Explicit bin edges instead of quantiles."
    )
    split: str = "train"
    method: Literal["inverse_frequency", "sqrt_inverse"] = "inverse_frequency"


@register
class Balance(BaseOp):
    id = "prep.dataset.balance"
    version = "1.0.0"
    category = "dataset"
    summary = (
        "Regime balancing: bin a parameter, count samples per regime and record per-sample weights "
        "(inverse frequency) for the trainer."
    )
    when_to_use = (
        "Campaigns dominated by one regime (e.g. low Reynolds); the weights are a hint the trainer "
        "may use, the data is untouched."
    )
    changes_manifest = True
    per_sample = False
    read_only = True
    Params = BalanceParams

    def plan_manifest(self, ref: DatasetRef, params: BalanceParams) -> list[Change]:
        return [
            Change(
                target="manifest/stats/weights",
                action="modify",
                detail={"parameter": params.parameter},
            )
        ]

    def apply_manifest(
        self,
        ref: DatasetRef,
        manifest: Manifest,
        params: BalanceParams,
        outcomes: list[SampleOutcome],
    ) -> tuple[list[Change], dict[str, Any]]:
        ids = manifest.splits.get(params.split) or [e.id for e in manifest.samples]
        val = {
            e.id: {**e.geometry_params, **e.parameters}.get(params.parameter)
            for e in manifest.samples
            if e.id in set(ids)
        }
        xs = np.array([v for v in val.values() if v is not None], dtype=np.float64)
        if xs.size == 0:
            return [], {}
        edges = (
            np.asarray(params.edges)
            if params.edges
            else np.quantile(xs, np.linspace(0, 1, params.bins + 1))
        )
        edges[-1] = np.nextafter(edges[-1], np.inf)
        regime = {
            sid: int(np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2))
            for sid, v in val.items()
            if v is not None
        }
        counts = np.bincount(list(regime.values()), minlength=len(edges) - 1)
        w = {sid: float(len(regime) / (len(counts) * counts[r])) for sid, r in regime.items()}
        if params.method == "sqrt_inverse":
            w = {k: float(np.sqrt(v)) for k, v in w.items()}
        manifest.stats["weights"] = {
            "parameter": params.parameter,
            "edges": edges.tolist(),
            "counts": counts.tolist(),
            "method": params.method,
            "split": params.split,
            "weights": w,
            "regime": regime,
        }
        return [
            Change(
                target="manifest/stats/weights", action="modify", detail={"counts": counts.tolist()}
            )
        ], {"counts": counts.tolist()}


# -- dataset.select ----------------------------------------------------------------------------


class Predicate(Params):
    key: str = Field(
        description=(
            "parameters.<name> | geometry_params.<name> | forces.<name> | n_points | n_steps | id"
        )
    )
    op: Literal["==", "!=", "<", "<=", ">", ">=", "in", "matches"]
    value: Any


class SelectParams(Params):
    ids: list[str] | None = None
    where: list[Predicate] = Field(
        default_factory=list, description="All predicates must hold (AND)."
    )
    quality: Literal["any", "no_fail", "pass_only"] = Field(
        default="any", description="Filter by recorded assertions on the current version."
    )
    mode: Literal["keep", "drop"] = "keep"
    delete_dirs: bool = True


@register
class Select(BaseOp):
    id = "prep.dataset.select"
    version = "1.0.0"
    category = "dataset"
    summary = (
        "Keep or drop samples by id, predicates on parameters/forces/sizes, or recorded quality "
        "assertions."
    )
    when_to_use = (
        "Remove failed cases after quality checks, carve a subset for an experiment, or build a "
        "regime-specific dataset."
    )
    changes_manifest = True
    per_sample = False
    Params = SelectParams

    def _match(
        self, ref: DatasetRef, e: SampleEntry, params: SelectParams, failing: dict[str, str]
    ) -> bool:
        if params.ids is not None and e.id not in params.ids:
            return False
        for p in params.where:
            v = _lookup(e, p.key)
            if v is None:
                return False
            if p.op == "==" and not v == p.value:
                return False
            if p.op == "!=" and not v != p.value:
                return False
            if p.op in ("<", "<=", ">", ">=") and not _cmp(v, p.op, p.value):
                return False
            if p.op == "in" and v not in p.value:
                return False
            if p.op == "matches":
                import re

                if not re.search(str(p.value), str(v)):
                    return False
        if params.quality == "no_fail" and failing.get(e.id) == "fail":
            return False
        if params.quality == "pass_only" and failing.get(e.id) in ("fail", "warn"):
            return False
        return True

    def plan_manifest(self, ref: DatasetRef, params: SelectParams) -> list[Change]:
        failing = _assertion_status(ref)
        keep = [e.id for e in ref.manifest.samples if self._match(ref, e, params, failing)]
        drop = (
            [e.id for e in ref.manifest.samples if e.id not in keep]
            if params.mode == "keep"
            else keep
        )
        return [Change(target=f"samples/{sid}", action="remove") for sid in drop]

    def apply_manifest(
        self,
        ref: DatasetRef,
        manifest: Manifest,
        params: SelectParams,
        outcomes: list[SampleOutcome],
    ) -> tuple[list[Change], dict[str, Any]]:
        failing = _assertion_status(ref)
        matched = {e.id for e in manifest.samples if self._match(ref, e, params, failing)}
        drop = {e.id for e in manifest.samples} - matched if params.mode == "keep" else matched
        _remove_samples(ref, manifest, drop, delete_dirs=params.delete_dirs)
        manifest.stats.setdefault("selection", []).append(
            {"dropped": sorted(drop, key=_natural), "params": params.model_dump()}
        )
        return [
            Change(target=f"samples/{sid}", action="remove") for sid in sorted(drop, key=_natural)
        ], {"dropped": sorted(drop, key=_natural), "kept": len(manifest.samples)}


def _lookup(e: SampleEntry, key: str) -> Any:
    if key == "id":
        return e.id
    if key in ("n_points", "n_steps", "n_faces", "n_cells"):
        return getattr(e, key)
    sec, _, name = key.partition(".")
    d = getattr(e, sec, None)
    return d.get(name) if isinstance(d, dict) else None


def _cmp(v: Any, op: str, w: Any) -> bool:
    try:
        return {"<": v < w, "<=": v <= w, ">": v > w, ">=": v >= w}[op]
    except TypeError:
        return False


def _assertion_status(ref: DatasetRef) -> dict[str, str]:
    """Worst recorded assertion status per sample on HEAD."""
    out: dict[str, str] = {}
    if not ref.store.initialized:
        return out
    rank = {"pass": 0, "skip": 0, "warn": 1, "fail": 2}
    for rec in ref.store.effective_assertions("HEAD"):
        t = str(rec.get("target", ""))
        if t.startswith("samples/"):
            sid = t.split("/")[1]
            st = str(rec.get("status", "pass"))
            if rank.get(st, 0) >= rank.get(out.get(sid, "pass"), 0):
                out[sid] = st
    return out


# -- dataset.merge -----------------------------------------------------------------------------


class MergeParams(Params):
    source: str = Field(description="Path of the dataset to merge in.")
    prefix: str | None = Field(
        default=None, description="Prefix for incoming sample ids (default: source name + '/')."
    )
    on_conflict: Literal["fail", "skip", "rename"] = "fail"
    require_same_units: bool = True
    merge_splits: bool = True


@register
class Merge(BaseOp):
    id = "prep.dataset.merge"
    version = "1.0.0"
    category = "dataset"
    summary = (
        "Merge another dataset into this one with schema reconciliation (same kind, field units "
        "must agree), prefixed ids and merged splits."
    )
    when_to_use = (
        "Combining campaigns or hub subsets after both were mapped to the canonical vocabulary and "
        "units."
    )
    changes_manifest = True
    per_sample = False
    Params = MergeParams

    def plan_extra(self, ref: DatasetRef, params: MergeParams) -> tuple[list[OpWarning], list[Any]]:
        w: list[OpWarning] = []
        try:
            src = read_manifest(Path(params.source))
        except FileNotFoundError as exc:
            return [OpWarning(severity="fail", code="source", message=str(exc))], []
        if src.sample_kind != ref.manifest.sample_kind:
            w.append(
                OpWarning(
                    severity="fail",
                    code="kind",
                    message=f"sample kinds differ: {ref.manifest.sample_kind} vs {src.sample_kind}",
                )
            )
        units = {f.name: f.units for f in ref.manifest.fields}
        for f in src.fields:
            if f.name in units and units[f.name] != f.units and params.require_same_units:
                w.append(
                    OpWarning(
                        severity="fail",
                        code="units",
                        message=(
                            f"field {f.name}: {units[f.name]} vs {f.units}; "
                            "run prep.units.convert first"
                        ),
                        target=f.name,
                    )
                )
            if f.name not in units:
                w.append(
                    OpWarning(
                        severity="info",
                        code="new-field",
                        message=f"field {f.name} only in source",
                        target=f.name,
                    )
                )
        return w, []

    def plan_manifest(self, ref: DatasetRef, params: MergeParams) -> list[Change]:
        try:
            src = read_manifest(Path(params.source))
        except FileNotFoundError:
            return []
        prefix = params.prefix if params.prefix is not None else f"{src.name}/"
        return [Change(target=f"samples/{prefix}{e.id}", action="add") for e in src.samples]

    def apply_manifest(
        self,
        ref: DatasetRef,
        manifest: Manifest,
        params: MergeParams,
        outcomes: list[SampleOutcome],
    ) -> tuple[list[Change], dict[str, Any]]:
        from frontier_prep.base import _link_tree

        src_root = Path(params.source)
        src = read_manifest(src_root)
        prefix = params.prefix if params.prefix is not None else f"{src.name}/"
        have = {e.id for e in manifest.samples}
        changes = []
        added = []
        for e in src.samples:
            sid = f"{prefix}{e.id}"
            if sid in have:
                if params.on_conflict == "fail":
                    raise RuntimeError(f"sample {sid} exists")
                if params.on_conflict == "skip":
                    continue
                sid = f"{sid}~{src.name}"
            dst = ref.root / "samples" / sid
            _link_tree(src_root / e.path, dst)
            meta = json.loads((dst / "meta.json").read_text())
            meta["id"] = sid
            meta.setdefault("extra", {})["merged_from"] = {
                "dataset": src.name,
                "id": e.id,
                "path": str(src_root),
            }
            write_json(dst / "meta.json", meta)
            ne = e.model_copy(update={"id": sid, "path": f"samples/{sid}"})
            manifest.samples.append(ne)
            added.append(sid)
            changes.append(Change(target=f"samples/{sid}", action="add"))
            if params.merge_splits:
                for k, v in src.splits.items():
                    if e.id in v:
                        manifest.splits.setdefault(k, []).append(sid)
        for key in ("parameters", "conditions", "geometry_params"):
            cur = getattr(manifest, key)
            for k in getattr(src, key):
                if k not in cur:
                    cur.append(k)
        manifest.samples.sort(key=lambda e: _natural(e.id))
        manifest.stats["n_samples"] = len(manifest.samples)
        manifest.stats.setdefault("merged", []).append(
            {"source": src.name, "n": len(added), "prefix": prefix}
        )
        return changes, {"added": len(added)}


# -- dataset.summarize -------------------------------------------------------------------------


class SummarizeParams(Params):
    write_card: bool = Field(default=True, description="Write CARD.md next to the manifest.")


@register
class Summarize(BaseOp):
    id = "prep.dataset.summarize"
    version = "1.0.0"
    category = "dataset"
    summary = (
        "Dataset card: counts, kinds, fields with units, parameter ranges, splits, quality and "
        "provenance summary (manifest.stats.summary + CARD.md)."
    )
    when_to_use = (
        "Before sharing or registering a version; after a cleaning session to document what the "
        "data now is."
    )
    changes_manifest = True
    per_sample = False
    Params = SummarizeParams

    def plan_manifest(self, ref: DatasetRef, params: SummarizeParams) -> list[Change]:
        return [Change(target="manifest/stats/summary", action="modify")] + (
            [Change(target="CARD.md", action="modify")] if params.write_card else []
        )

    def apply_manifest(
        self,
        ref: DatasetRef,
        manifest: Manifest,
        params: SummarizeParams,
        outcomes: list[SampleOutcome],
    ) -> tuple[list[Change], dict[str, Any]]:
        samples = manifest.samples
        pr: dict[str, Any] = {}
        for k in manifest.parameters + manifest.geometry_params:
            vals = [{**e.geometry_params, **e.parameters}.get(k) for e in samples]
            vals = [v for v in vals if v is not None]
            if vals:
                pr[k] = {"min": float(min(vals)), "max": float(max(vals)), "n": len(vals)}
        kinds: dict[str, int] = {}
        for e in samples:
            kinds[e.sample_kind] = kinds.get(e.sample_kind, 0) + 1
        assertions = ref.store.effective_assertions("HEAD") if ref.store.initialized else []
        qa: dict[str, dict[str, int]] = {}
        for a in assertions:
            c = qa.setdefault(str(a.get("check")), {})
            c[str(a.get("status"))] = c.get(str(a.get("status")), 0) + 1
        summary = {
            "name": manifest.name,
            "version": ref.head,
            "n_samples": len(samples),
            "kinds": kinds,
            "temporal": manifest.temporal,
            "fields": [
                {"name": f.name, "units": f.units, "location": f.location, "channels": f.channels}
                for f in manifest.fields
            ],
            "parameters": pr,
            "splits": {k: len(v) for k, v in manifest.splits.items()},
            "split_strategy": manifest.split_strategy,
            "n_points_total": int(sum(e.n_points for e in samples)),
            "n_steps_total": int(sum(e.n_steps for e in samples)),
            "quality": qa,
            "n_versions": len(ref.store.log()) if ref.store.initialized else 0,
            "license": manifest.license,
            "citation": manifest.citation,
        }
        manifest.stats["summary"] = summary
        changes = [Change(target="manifest/stats/summary", action="modify")]
        if params.write_card:
            write_text_atomic(ref.root / "CARD.md", _card_md(summary, ref))
            changes.append(Change(target="CARD.md", action="modify"))
        return changes, {"summary": summary}


def _card_md(s: dict[str, Any], ref: DatasetRef) -> str:
    lines = [
        f"# {s['name']}",
        "",
        f"Version `{s['version']}`; {s['n_samples']} samples "
        f"({', '.join(f'{v} {k}' for k, v in s['kinds'].items())}), {s['temporal']}.",
        "",
    ]
    if s["license"] or s["citation"]:
        lines += [f"License: {s['license']}. Citation: {s['citation']}", ""]
    lines += ["## Fields", "", "| name | units | location | channels |", "|---|---|---|---|"]
    lines += [
        f"| {f['name']} | {f['units']} | {f['location']} | {f['channels']} |" for f in s["fields"]
    ]
    if s["parameters"]:
        lines += ["", "## Parameters", "", "| name | min | max | n |", "|---|---|---|---|"]
        lines += [
            f"| {k} | {v['min']:.6g} | {v['max']:.6g} | {v['n']} |"
            for k, v in s["parameters"].items()
        ]
    lines += [
        "",
        "## Splits",
        "",
        ", ".join(f"{k}: {v}" for k, v in s["splits"].items()) or "none",
        f"Strategy: `{json.dumps(s['split_strategy'])}`",
    ]
    if s["quality"]:
        lines += [
            "",
            "## Quality (assertions on this version)",
            "",
            "| check | pass | warn | fail |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {k} | {v.get('pass', 0)} | {v.get('warn', 0)} | {v.get('fail', 0)} |"
            for k, v in sorted(s["quality"].items())
        ]
    if ref.store.initialized:
        lines += ["", "## Provenance", ""]
        for v in ref.store.log(10):
            rec = ref.store.record(v.op_record) if v.op_record else None
            lines.append(
                f"- `{v.id}` {v.created_at[:19]} {rec.get('op') if rec else ''}: {v.message}"
            )
    return "\n".join(lines) + "\n"

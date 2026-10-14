# SPDX-License-Identifier: Apache-2.0
"""Quality checks for transient / non-surface samples (``frontier_data.quality_transient``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from frontier_data.dataset import Dataset, DatasetWriter, MeshStats, Provenance, SampleMeta
from frontier_data.mesh import make_box_mesh
from frontier_data.quality import run_quality
from frontier_data.quality_report import render_html, svg_timeseries
from frontier_data.quality_transient import (
    CFL_TINY,
    CFL_WARN,
    HIGH_K_WARN,
    SERIES_MAX_POINTS,
    check_times,
    classify_stationarity,
    integrated_autocorr_time,
    isotropic_spectrum,
    mser_truncation,
    transient_sample_pass,
)
from frontier_data.splits import (
    TrajectoryFingerprint,
    find_trajectory_leakage,
    group_split,
    make_split,
)
from frontier_data.synthetic import spectral_vorticity_2d, write_synthetic_dataset

PERIODIC = {"x": "periodic", "y": "periodic"}
TRANSIENT_CHECKS = {
    "nan_inf_per_step",
    "temporal_continuity",
    "cfl",
    "stationarity",
    "spectra",
    "conservation",
    "temporal_subsampling",
}


@pytest.fixture(scope="module")
def base() -> dict[str, Any]:
    """One smooth 2D Navier-Stokes trajectory (32 steps on 32 x 32) to doctor."""
    return spectral_vorticity_2d(n=32, n_steps=32, dt=0.1, nu=2e-3, seed=3)


def _copy(d: dict[str, Any]) -> dict[str, Any]:
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def _cut(d: dict[str, Any], sl: slice) -> dict[str, Any]:
    return {**d, **{k: d[k][sl] for k in ("velocity", "vorticity", "times")}}


def _grid_dataset(
    root: Path,
    samples: dict[str, dict[str, Any]],
    splits: dict[str, list[str]] | None = None,
    **common: Any,
) -> Dataset:
    """Write solver-style dicts (``velocity``, ``vorticity``, ``times``, ``spacing`` plus
    optional writer keywords under ``kw``) as grid samples."""
    w = DatasetWriter(root, name=root.name)
    for sid, d in samples.items():
        kw = {"boundary_conditions": PERIODIC, **common, **d.get("kw", {})}
        w.write_grid_sample(
            sid,
            fields={"velocity": d["velocity"], "vorticity": d["vorticity"]},
            times=d["times"],
            spacing=d["spacing"],
            **kw,
        )
    w.finalize()
    ds = Dataset(root)
    if splits is not None:
        ds.manifest.splits = splits
        ds.save_manifest()
    return ds


def _scalar_dataset(root: Path, series: dict[str, np.ndarray], n: int = 12) -> Dataset:
    """Grid samples with one scalar field ``q`` = fixed pattern x amplitude(t) + weak noise."""
    rng = np.random.default_rng(0)
    x = 2 * np.pi * np.arange(n) / n
    pattern = 1.0 + 0.5 * np.sin(x)[:, None] * np.cos(x)[None, :]
    w = DatasetWriter(root, name=root.name)
    for sid, amp in series.items():
        q = pattern[None] * amp[:, None, None] + 1e-3 * rng.standard_normal((amp.size, n, n))
        w.write_grid_sample(
            sid,
            fields={"q": q},
            times=0.1 * np.arange(amp.size),
            spacing=[1.0, 1.0],
            boundary_conditions=PERIODIC,
        )
    w.finalize()
    return Dataset(root)


def _checks(report: dict[str, Any], sid: str) -> dict[str, dict[str, Any]]:
    row = next(r for r in report["samples"] if r["id"] == sid)
    return {c["check"]: c for c in row["checks"]}


def _dataset_checks(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["check"]: c for c in report["dataset_checks"]}


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(token: str) -> None:
        raise ValueError(f"non-standard JSON constant {token}")

    return json.loads(path.read_text(), parse_constant=reject)


# -- run_quality on every synthetic kind ----------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "size"), [("grid", 32), ("mesh", 200), ("surface", 4), ("point_cloud", 200)]
)
def test_run_quality_on_synthetic_kinds(tmp_path: Path, kind: str, size: int) -> None:
    ds = write_synthetic_dataset(tmp_path / kind, kind=kind, n_samples=3, n_steps=24, size=size)
    report = run_quality(ds)

    # the top-level shape is unchanged, plus the "transient" object
    for key in ("summary", "dataset_checks", "field_statistics", "recommended_normalization"):
        assert key in report
    assert report["summary"]["status"] in ("pass", "warn")
    assert report["summary"]["fail"] == 0
    dchecks = _dataset_checks(report)
    assert dchecks["trajectory_split_leakage"]["status"] == "pass"
    assert dchecks["temporal_consistency"]["status"] == "pass"
    assert dchecks["temporal_consistency"]["detail"]["dt"]

    tr = report["transient"]
    assert tr["n_trajectories"] == 3 and tr["steps"] == {
        "min": 24,
        "median": 24.0,
        "max": 24,
        "total": 72,
    }
    assert len(tr["dt"]["values"]) == 1 and tr["dt"]["uniform"]
    assert tr["recommended_stride"] >= 1 and tr["recommended_burn_in"] >= 0
    assert 0 < tr["training_pairs"] <= tr["training_pairs_stride_1"] == 3 * 23
    assert tr["recommendations"] and all(isinstance(r, str) for r in tr["recommendations"])

    fields = {"grid": "velocity", "mesh": "velocity", "point_cloud": "pressure", "surface": "p"}
    name = fields[kind]
    assert report["recommended_normalization"]["fields"][name]["method"] == "zscore"
    assert report["field_statistics"][name]["std"] > 0

    for row in report["samples"]:
        checks = {c["check"]: c for c in row["checks"]}
        assert TRANSIENT_CHECKS <= set(checks)
        assert checks["nan_inf_per_step"]["status"] == "pass"
        assert checks["temporal_continuity"]["status"] == "pass"
        assert checks["conservation"]["status"] == "pass"
        sub = checks["temporal_subsampling"]
        assert sub["status"] == "pass"
        assert {"recommended_stride", "decorrelation_steps", "lag1_autocorr", "rule"} <= set(
            sub["detail"]
        )
        t = row["transient"]
        assert t["n_steps"] == 24 and t["kind"] == kind
        for block in t["series"].values():
            for values in _series_lists(block):
                assert len(values) <= SERIES_MAX_POINTS
        assert {"step_change", "running_mean", "integrals"} <= set(t["series"])
        if kind == "grid":
            assert checks["spectra"]["status"] == "pass"
            assert checks["spectra"]["detail"]["window"] == "none (periodic)"
            assert checks["spectra"]["detail"]["enstrophy"]["source"] == "vorticity"
            assert "spectrum" in t["series"]
            assert checks["cfl"]["status"] == "pass" and checks["cfl"]["detail"]["regime"] == "ok"
        else:
            assert checks["spectra"]["status"] == "skip"
        if kind in ("mesh", "point_cloud"):
            assert checks["cfl"]["status"] == "pass"
            assert CFL_TINY < checks["cfl"]["detail"]["cfl_max"] < CFL_WARN
        if kind == "surface":
            assert checks["cfl"]["status"] == "skip"
            assert checks["degenerate_faces"]["status"] == "pass"
            assert checks["force_consistency"]["status"] == "skip"

    # report.json is strict JSON (no NaN / Infinity), the HTML has the Transient section
    on_disk = _strict_json(ds.root / "quality" / "report.json")
    assert on_disk["transient"]["n_trajectories"] == 3
    page = (ds.root / "quality" / "report.html").read_text()
    assert "<h2>Transient</h2>" in page and "<svg" in page and "<polyline" in page
    assert "<script" not in page and "http://www.w3.org/2000/svg" in page
    assert "Recommendations" in page and "traj_000" in page


def _series_lists(block: Any) -> list[list[Any]]:
    if isinstance(block, list):
        return [block]
    if isinstance(block, dict):
        return [v for b in block.values() for v in _series_lists(b)]
    return []


def test_steady_non_surface_sample(tmp_path: Path) -> None:
    w = DatasetWriter(tmp_path / "steady", name="steady")
    rng = np.random.default_rng(0)
    for i in range(3):
        w.write_points_sample(
            f"pc_{i}", points=rng.random((50, 3)), fields={"T": rng.random(50) + i}
        )
    bad = rng.random(50)
    bad[4] = np.inf
    w.write_points_sample("pc_bad", points=rng.random((50, 3)), fields={"T": bad})
    w.finalize()
    report = run_quality(Dataset(tmp_path / "steady"))
    assert "transient" not in report
    assert [c["check"] for c in report["dataset_checks"]] == ["parameter_coverage", "split_leakage"]
    assert _checks(report, "pc_0")["nan_inf_per_step"]["status"] == "pass"
    bad_check = _checks(report, "pc_bad")["nan_inf_per_step"]
    assert bad_check["status"] == "fail" and bad_check["detail"]["time_independent"] == {"T": 1}
    assert "T" in report["recommended_normalization"]["fields"]
    assert "<h2>Transient</h2>" not in (tmp_path / "steady/quality/report.html").read_text()


# -- injected defects -----------------------------------------------------------------------


def test_defects_nan_frozen_duplicate_jump_times(tmp_path: Path, base: dict[str, Any]) -> None:
    nan = _copy(base)
    nan["velocity"][7, 3, 4, 0] = np.nan
    frozen = _copy(base)
    for f in ("velocity", "vorticity"):
        frozen[f][20] = frozen[f][19]
    dup = _copy(base)
    dup["velocity"][28] = dup["velocity"][10]
    jump = _copy(base)
    jump["velocity"][25] *= 50.0
    times = _copy(base)
    times["times"][24:] += 0.05
    ds = _grid_dataset(
        tmp_path / "defects",
        {"clean": base, "nan": nan, "frozen": frozen, "dup": dup, "jump": jump, "times": times},
    )
    report = run_quality(ds)
    status = {r["id"]: r["status"] for r in report["samples"]}

    clean = _checks(report, "clean")
    assert status["clean"] in ("pass", "warn")
    for name in ("nan_inf_per_step", "temporal_continuity", "spectra", "conservation", "cfl"):
        assert clean[name]["status"] == "pass", name

    c = _checks(report, "nan")["nan_inf_per_step"]
    assert c["status"] == "fail" and status["nan"] == "fail"
    assert c["detail"]["fields"]["velocity"]["first_steps"] == [{"step": 7, "count": 1}]
    assert "vorticity" not in c["detail"]["fields"]
    # a single NaN cell is not a blow-up
    assert _checks(report, "nan")["conservation"]["status"] == "pass"

    c = _checks(report, "frozen")["temporal_continuity"]
    vel = c["detail"]["fields"]["velocity"]
    assert c["status"] == "fail" and vel["frozen_kind"] == "interior"
    assert vel["frozen_steps"] == [20] and vel["n_duplicated"] == 0

    c = _checks(report, "dup")["temporal_continuity"]
    vel = c["detail"]["fields"]["velocity"]
    assert c["status"] == "fail" and vel["n_frozen"] == 0
    assert vel["duplicated_steps"] == [{"step": 28, "same_as": 10, "constant_frame": False}]
    assert c["detail"]["fields"]["vorticity"]["n_duplicated"] == 0

    c = _checks(report, "jump")["temporal_continuity"]
    vel = c["detail"]["fields"]["velocity"]
    assert c["status"] == "fail" and 25 in [j["step"] for j in vel["jumps"]]
    assert vel["jumps"][0]["ratio_to_neighbours"] > 100
    assert c["detail"]["fields"]["vorticity"]["n_jumps"] == 0
    series = next(r for r in report["samples"] if r["id"] == "jump")["transient"]["series"]
    assert 25 in series["step_change"]["jumps"]["velocity"]

    c = _checks(report, "times")["temporal_continuity"]
    assert c["status"] == "warn" and status["times"] == "warn"
    assert c["detail"]["times"]["uniform"] is False
    assert c["detail"]["times"]["non_uniform_steps"] == [24]
    assert c["detail"]["times"]["strictly_increasing"] is True
    assert clean["temporal_continuity"]["detail"]["times"]["uniform"] is True

    cons = _dataset_checks(report)["temporal_consistency"]
    assert cons["status"] == "warn" and cons["detail"]["non_uniform_dt_samples"] == ["times"]
    recs = " ".join(report["transient"]["recommendations"])
    assert "NaN" in recs and "jumps, duplicated or frozen" in recs

    # the report draws the marked jump; samples are ordered worst first
    page = render_html(report)
    assert "<h2>Transient</h2>" in page and "stroke-dasharray" in page
    first_plotted = page.index("<h3>nan ")
    assert first_plotted < page.index("<h3>clean ")


def test_stationarity_classes(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 200
    t = np.arange(n)
    noise = np.zeros(n)
    for i in range(1, n):  # AR(1): a statistically stationary forced signal
        noise[i] = 0.5 * noise[i - 1] + rng.standard_normal()
    ds = _scalar_dataset(
        tmp_path / "stat",
        {
            "stationary": 1.0 + 0.05 * noise,
            "drifting": 1.0 + 2.0 * t / n + 0.05 * rng.standard_normal(n),
            "startup": 1.0 + 0.6 * np.exp(-t / 8.0) + 0.02 * rng.standard_normal(n),
            "short": 1.0 + 0.05 * rng.standard_normal(10),
        },
    )
    report = run_quality(ds)
    st = {sid: _checks(report, sid)["stationarity"] for sid in ds.sample_ids}

    assert st["stationary"]["detail"]["class"] == "stationary"
    assert st["stationary"]["status"] == "pass"
    assert st["stationary"]["detail"]["n_eff_min"] > 20
    mean = st["stationary"]["detail"]["fields"]["q"]["mean"]
    assert {"half_z", "drift_last_third", "tau_int", "n_eff"} <= set(mean)
    assert 0.5 <= mean["tau_int"] < 4.0  # AR(1) with phi = 0.5 has tau_int = 1.5

    assert st["drifting"]["detail"]["class"] == "drifting"
    assert st["drifting"]["status"] == "warn"  # never a failure
    assert st["drifting"]["detail"]["fields"]["q"]["mean"]["half_z"] > 3.5

    assert st["startup"]["detail"]["class"] == "transient-start"
    assert 5 <= st["startup"]["detail"]["burn_in_steps"] <= 60
    assert st["short"]["detail"]["class"] == "too-short" and st["short"]["status"] == "skip"

    tr = report["transient"]
    assert tr["stationarity"] == {
        "stationary": 1,
        "drifting": 1,
        "transient-start": 1,
        "too-short": 1,
    }
    assert tr["burn_in_max"] == st["startup"]["detail"]["burn_in_steps"]
    assert any("start-up transient" in r for r in tr["recommendations"])
    row = next(r for r in report["samples"] if r["id"] == "startup")
    assert row["transient"]["burn_in"] == tr["burn_in_max"]
    assert len(row["transient"]["series"]["running_mean"]["fields"]["q"]) <= SERIES_MAX_POINTS


def test_stationarity_helpers() -> None:
    rng = np.random.default_rng(0)
    white = rng.standard_normal(4000)
    assert integrated_autocorr_time(white) == pytest.approx(0.5, abs=0.15)
    ar = np.zeros(4000)
    for i in range(1, ar.size):
        ar[i] = 0.8 * ar[i - 1] + white[i]
    assert integrated_autocorr_time(ar) == pytest.approx(4.5, rel=0.35)  # (1 + phi) / 2(1 - phi)
    x = np.concatenate([np.linspace(5.0, 0.0, 30), np.zeros(170)]) + 0.1 * white[:200]
    assert 20 <= mser_truncation(x) <= 40
    assert mser_truncation(white[:200]) < 20
    assert classify_stationarity(np.full(50, 3.0))["class"] == "stationary"
    assert classify_stationarity(np.arange(8.0))["class"] == "too-short"
    assert classify_stationarity(np.linspace(0.0, 1.0, 100))["class"] == "drifting"


def test_energy_blow_up_and_spectral_pile_up(tmp_path: Path, base: dict[str, Any]) -> None:
    blow = _copy(base)
    blow["velocity"][-5:] *= np.array([3.0, 10.0, 30.0, 100.0, 300.0])[:, None, None, None]
    rng = np.random.default_rng(0)
    noise = _copy(base)
    noise["velocity"] = rng.standard_normal(base["velocity"].shape).astype(np.float32)
    open_bc = {**base, "kw": {"boundary_conditions": {"x": ["wall", "wall"], "y": "open"}}}
    ds = _grid_dataset(
        tmp_path / "energy", {"clean": base, "blow": blow, "noise": noise, "open": open_bc}
    )
    report = run_quality(ds)

    clean = _checks(report, "clean")
    ke = clean["conservation"]["detail"]["quantities"]["kinetic_energy"]
    assert clean["conservation"]["status"] == "pass"
    assert ke["monotonic"] == "decreasing" and -0.5 < ke["rel_drift"] < 0  # decaying turbulence
    assert "enstrophy" in clean["conservation"]["detail"]["quantities"]
    assert "mean:vorticity" in clean["conservation"]["detail"]["quantities"]

    c = _checks(report, "blow")["conservation"]
    ke = c["detail"]["quantities"]["kinetic_energy"]
    assert c["status"] == "fail" and ke["growth_vs_median"] > 100
    assert ke["exploded_at_step"] in (28, 29) and ke["monotonic"] == "no"
    assert c["detail"]["quantities"]["enstrophy"]["growth_vs_median"] < 2
    assert any("blow up" in r for r in report["transient"]["recommendations"])

    spec = clean["spectra"]
    assert spec["status"] == "pass" and spec["detail"]["high_k_fraction_max"] < 0.01
    assert spec["detail"]["slope_mid"] < -2.0
    first = spec["detail"]["steps"]["first"]
    # Parseval: the spectrum sums to the kinetic energy of the first frame
    assert first["total"] == pytest.approx(
        clean["conservation"]["detail"]["quantities"]["kinetic_energy"]["first"], rel=1e-3
    )
    assert spec["detail"]["energy_evolution"]["last_over_first"] < 1.0

    c = _checks(report, "noise")["spectra"]
    assert c["status"] == "warn" and c["detail"]["high_k_fraction_max"] > 10 * HIGH_K_WARN
    assert c["detail"]["pile_up"] is True
    assert _checks(report, "noise")["temporal_subsampling"]["detail"]["recommended_stride"] == 1
    assert any("grid cutoff" in r for r in report["transient"]["recommendations"])

    # non-periodic boundaries: Hann window, the smooth field still passes
    c = _checks(report, "open")["spectra"]
    assert c["detail"]["window"] == "hann" and c["status"] == "pass"

    row = next(r for r in report["samples"] if r["id"] == "clean")
    sp = row["transient"]["series"]["spectrum"]
    assert len(sp["k"]) == len(sp["first"]) == len(sp["last"]) == 16
    assert "wavenumber k" in render_html(report)


def test_isotropic_spectrum_identifies_a_single_mode() -> None:
    n = 32
    x = 2 * np.pi * np.arange(n) / n
    f = np.cos(5 * x)[:, None] * np.ones(n)[None, :]
    k, e = isotropic_spectrum(f[..., None])
    assert int(k[np.argmax(e)]) == 5
    assert e.sum() == pytest.approx(0.5 * np.mean(f**2))
    k3, e3 = isotropic_spectrum(np.broadcast_to(f[:16:2, :16, None, None], (8, 16, 16, 1)))
    # anisotropic box: shells are in units of the shortest side's fundamental
    assert k3[-1] == 4 and e3.sum() == pytest.approx(0.5 * np.mean(f[:16:2] ** 2))
    _, ew = isotropic_spectrum(f[..., None], window=True)
    assert int(np.argmax(ew)) == 5


def test_cfl_regimes(tmp_path: Path, base: dict[str, Any]) -> None:
    dx = base["spacing"][0]
    ds = _grid_dataset(
        tmp_path / "cfl",
        {
            "ok": base,
            "coarse": {**base, "spacing": [dx / 200, dx / 200]},
            "oversampled": {**base, "spacing": [dx * 500, dx * 500]},
        },
    )
    report = run_quality(ds)
    ok = _checks(report, "ok")["cfl"]
    assert ok["status"] == "pass" and ok["detail"]["regime"] == "ok"
    assert ok["detail"]["velocity_field"] == "velocity"
    assert ok["detail"]["cfl_p99"] <= ok["detail"]["cfl_max"]

    coarse = _checks(report, "coarse")["cfl"]
    assert coarse["status"] == "warn" and coarse["detail"]["regime"] == "coarse"
    assert coarse["detail"]["cfl_p99"] > CFL_WARN

    over = _checks(report, "oversampled")["cfl"]
    assert over["status"] == "pass" and over["detail"]["regime"] == "oversampled"
    assert over["detail"]["cfl_max"] < CFL_TINY and over["detail"]["recommended_stride"] > 10
    recs = " ".join(report["transient"]["recommendations"])
    assert "saved-step CFL above" in recs and "oversampled in time" in recs


def test_temporal_subsampling_recommends_a_stride(tmp_path: Path) -> None:
    n, n_steps, period = 16, 160, 80
    x = 2 * np.pi * np.arange(n) / n
    t = np.arange(n_steps)
    phase = 2 * np.pi * t / period
    wave = np.cos(x[None, :, None] - phase[:, None, None]) * np.ones((1, 1, n))
    w = DatasetWriter(tmp_path / "wave", name="wave")
    w.write_grid_sample("wave", fields={"q": wave}, times=0.01 * t, spacing=[1.0, 1.0])
    w.finalize()
    ds = Dataset(tmp_path / "wave")
    report = run_quality(ds)
    d = _checks(report, "wave")["temporal_subsampling"]["detail"]
    # rho(s) = cos(2 pi s / 80): >= 0.9 up to s = 5, below 1/e from s = 16
    assert d["recommended_stride"] == 5 and d["stride_cap"] == 10
    assert d["lag1_autocorr"] == pytest.approx(np.cos(2 * np.pi / period), abs=2e-3)
    assert d["decorrelation_steps"] == 16 and d["effective_steps"] == 32
    assert "0.9" in d["rule"]
    tr = report["transient"]
    assert tr["recommended_stride"] == 5 and tr["training_pairs"] == n_steps - 5
    assert tr["training_pairs_disjoint"] == (n_steps - 1) // 5
    assert any("stride of 5" in r for r in tr["recommendations"])
    assert _checks(report, "wave")["stationarity"]["detail"]["class"] == "stationary"

    # long trajectories are scanned with a stride that the details report
    res = transient_sample_pass(ds.sample("wave"), max_scan_steps=40)
    by_name = {c.check: c for c in res.checks}
    assert res.transient is not None and res.transient["scan_stride"] == 4
    assert by_name["nan_inf_per_step"].detail["stride"] == 4
    assert by_name["nan_inf_per_step"].detail["n_steps_checked"] == 41  # + the last step
    assert by_name["temporal_continuity"].detail["stride"] == 4
    assert by_name["temporal_subsampling"].detail["lag_unit_steps"] == 4
    assert by_name["temporal_subsampling"].detail["recommended_stride"] == 4  # rho(8) < 0.9


def test_check_times() -> None:
    assert check_times(None, {})["status"] == "skip"
    ok = check_times(0.5 * np.arange(10), {"u": 10})
    assert ok["status"] == "pass" and ok["uniform"] and ok["dt_median"] == 0.5
    dup = check_times(np.array([0.0, 1.0, 1.0, 2.0]), {"u": 4})
    assert dup["status"] == "fail" and dup["n_duplicates"] == 1 and dup["duplicate_steps"] == [2]
    back = check_times(np.array([0.0, 1.0, 0.5, 2.0]), {"u": 4})
    assert back["status"] == "fail" and back["strictly_increasing"] is False
    assert check_times(np.arange(5.0), {"u": 6})["length_mismatch"] == {"u": 6}
    assert check_times(np.array([0.0, 1.0, 3.0, 4.0]), {})["status"] == "warn"


# -- splits ---------------------------------------------------------------------------------


def test_trajectory_split_leakage(tmp_path: Path, base: dict[str, Any]) -> None:
    other = spectral_vorticity_2d(n=32, n_steps=16, dt=0.1, nu=2e-3, seed=11)

    def leak(report: dict[str, Any]) -> dict[str, Any]:
        return _dataset_checks(report)["trajectory_split_leakage"]

    # (a) group ids: one trajectory cut in two, the halves in different splits
    grouped = {
        "a0": {**_cut(base, slice(0, 12)), "kw": {"group": "traj_a"}},
        "a1": {**_cut(base, slice(20, 32)), "kw": {"group": "traj_a"}},
        "b0": {**other, "kw": {"group": "traj_b"}},
    }
    ds = _grid_dataset(
        tmp_path / "grouped", grouped, {"train": ["a0", "b0"], "val": [], "test": ["a1"]}
    )
    report = run_quality(ds)
    c = leak(report)
    assert c["status"] == "fail" and report["summary"]["status"] == "fail"
    assert c["detail"]["group_leaks"] == {"traj_a": {"train": ["a0"], "test": ["a1"]}}
    assert c["detail"]["content_leaks"] == []
    assert report["transient"]["n_trajectories"] == 2 and report["transient"]["n_samples"] == 3
    assert any("group_split" in r for r in report["transient"]["recommendations"])
    assert _dataset_checks(report)["split_leakage"]["status"] == "pass"  # ids do not overlap
    assert _dataset_checks(report)["temporal_consistency"]["detail"]["mixed_n_steps"] is True

    # keeping the trajectory together passes
    ds.manifest.splits = {"train": ["a0", "a1"], "val": [], "test": ["b0"]}
    ds.save_manifest()
    assert leak(run_quality(ds))["status"] == "pass"

    # (b) no group ids: the content gives the cut away
    ungrouped = {
        "p0": _cut(base, slice(0, 17)),  # ends on frame 16 ...
        "p1": _cut(base, slice(16, 32)),  # ... which is the first frame of p1
        "q0": _cut(base, slice(0, 8)),  # same initial condition as p0
        "r0": other,
    }
    ds = _grid_dataset(
        tmp_path / "ungrouped",
        ungrouped,
        {"train": ["p0", "r0"], "val": ["q0"], "test": ["p1"]},
    )
    c = leak(run_quality(ds))
    assert c["status"] == "fail"
    kinds = {(v["kind"], v["a"], v["b"]): v for v in c["detail"]["content_leaks"]}
    assert set(kinds) == {
        ("shared_frame", "p0", "p1"),
        ("identical_initial_condition", "p0", "q0"),
    }
    assert kinds[("shared_frame", "p0", "p1")]["severity"] == "fail"
    assert kinds[("identical_initial_condition", "p0", "q0")]["same_parameters"] is True

    # a cut without a shared frame is still a continuation of the same trajectory
    halves = {"h0": _cut(base, slice(0, 16)), "h1": _cut(base, slice(16, 32)), "r0": other}
    ds = _grid_dataset(
        tmp_path / "halves", halves, {"train": ["h0", "r0"], "val": [], "test": ["h1"]}
    )
    c = leak(run_quality(ds))
    assert c["status"] == "fail"
    assert [(v["kind"], v["a"], v["b"]) for v in c["detail"]["content_leaks"]] == [
        ("continuation", "h0", "h1")
    ]
    ds.manifest.splits = {"train": ["h0", "h1"], "val": [], "test": ["r0"]}
    ds.save_manifest()
    assert leak(run_quality(ds))["status"] == "pass"


def test_find_trajectory_leakage_parameters_and_trivial_frames() -> None:
    rng = np.random.default_rng(0)
    ic, end_a, end_b = rng.standard_normal((3, 64))

    def fp(first: np.ndarray, last: np.ndarray, tag: str, **kw: Any) -> TrajectoryFingerprint:
        return TrajectoryFingerprint(
            key="u:8x8x1",
            first_hash=f"h{hash(first.tobytes())}",
            last_hash=f"h{hash(last.tobytes())}-{tag}",
            first=first,
            last=last,
            step_change=0.05,
            **kw,
        )

    splits = {"train": ["a"], "test": ["b"]}
    fps = {"a": fp(ic, end_a, "a"), "b": fp(ic, end_b, "b")}
    # a parameter sweep from one initial state only warns ...
    out = find_trajectory_leakage(splits, None, fps, {"a": {"Re": 100.0}, "b": {"Re": 200.0}})
    assert out["status"] == "warn" and out["content_leaks"][0]["same_parameters"] is False
    # ... the same parameters, the same initial condition and the same end state is the same
    # trajectory stored twice
    same = {"a": {"Re": 100.0}, "b": {"Re": 100.0}}
    twice = {"a": fp(ic, end_a, "a"), "b": fp(ic, end_a * (1 + 1e-3), "b")}
    out = find_trajectory_leakage(splits, None, twice, same)
    assert out["status"] == "fail"
    assert out["content_leaks"][0]["kind"] == "identical_initial_condition"
    # ... but runs that share step 0 and then diverge are an ensemble from one state: warn
    out = find_trajectory_leakage(splits, None, fps, same)
    assert out["status"] == "warn" and out["content_leaks"][0]["diverged"] is True
    # fluid at rest is no evidence
    zero = np.zeros(64)
    rest = {
        "a": fp(zero, end_a, "a", trivial_first=True),
        "b": fp(zero, end_b, "b", trivial_first=True),
    }
    assert find_trajectory_leakage(splits, None, rest)["status"] == "pass"
    # groups alone, without any content
    out = find_trajectory_leakage({"train": ["a", "c"], "val": ["b"]}, {"a": "g", "b": "g"})
    assert out["status"] == "fail" and out["group_leaks"] == {"g": {"train": ["a"], "val": ["b"]}}
    assert (
        find_trajectory_leakage({"train": ["a", "b"], "val": ["c"]}, {"a": "g", "b": "g"})["status"]
        == "pass"
    )


def test_group_split_never_separates_a_group() -> None:
    ids = [f"s{i:03d}" for i in range(60)]
    groups: dict[str, str | None] = {i: f"traj_{k // 4}" for k, i in enumerate(ids)}
    for seed in range(5):
        res = group_split(ids, groups, (0.7, 0.15, 0.15), seed=seed)
        assert sorted(sum(res.splits.values(), [])) == ids
        owner: dict[str, str] = {}
        for name, members in res.splits.items():
            for i in members:
                assert owner.setdefault(str(groups[i]), name) == name
        assert all(len(v) % 4 == 0 and v for v in res.splits.values())
        assert find_trajectory_leakage(res.splits, groups)["status"] == "pass"
    res = group_split(ids, groups, (0.8, 0.1, 0.1), seed=1)
    assert res.strategy["name"] == "group" and res.strategy["n_groups"] == 15
    assert res.strategy["groups_per_split"] == {"train": 11, "val": 2, "test": 2}
    assert group_split(ids, groups, seed=1).splits == res.splits  # deterministic
    assert group_split(ids, groups, seed=2).splits != res.splits

    via = make_split("group", ids, groups=groups, fractions=(0.8, 0.1, 0.1), seed=1)
    assert via.splits == res.splits
    with pytest.raises(ValueError):
        make_split("group", ids)

    # samples without a group are their own trajectory
    mixed: dict[str, str | None] = {"a": "g", "b": "g", "c": None}
    res = group_split(["a", "b", "c", "d", "e"], mixed, seed=0)
    assert res.strategy["n_groups"] == 4
    assert len({name for name, v in res.splits.items() if {"a", "b"} & set(v)}) == 1


def test_temporal_consistency_mixed_dt(tmp_path: Path, base: dict[str, Any]) -> None:
    fast = {**base, "times": base["times"] * 0.5}
    fewer = _copy(base)
    del fewer["vorticity"]
    w = DatasetWriter(tmp_path / "mixed", name="mixed")
    for sid, d in {"a": base, "b": fast}.items():
        w.write_grid_sample(
            sid,
            fields={"velocity": d["velocity"], "vorticity": d["vorticity"]},
            times=d["times"],
            spacing=d["spacing"],
            boundary_conditions=PERIODIC,
        )
    w.write_grid_sample(
        "c",
        fields={"velocity": fewer["velocity"][:20]},
        times=fewer["times"][:20],
        spacing=fewer["spacing"],
        boundary_conditions=PERIODIC,
    )
    w.finalize()
    report = run_quality(Dataset(tmp_path / "mixed"))
    c = _dataset_checks(report)["temporal_consistency"]
    assert c["status"] == "warn"
    assert c["detail"]["mixed_dt"] and c["detail"]["dt"] == {"0.1": 2, "0.05": 1}
    assert c["detail"]["mixed_fields"] and c["detail"]["n_steps"] == {"32": 2, "20": 1}
    assert report["transient"]["dt"]["values"] == [0.05, 0.1]
    assert any("lead time" in r for r in report["transient"]["recommendations"])
    assert _dataset_checks(report)["trajectory_split_leakage"]["status"] == "skip"  # no splits


# -- report ---------------------------------------------------------------------------------


def test_svg_timeseries() -> None:
    svg = svg_timeseries(
        {"velocity": [1.0, 0.5, None, 0.0, 2e-3, float("nan"), 4.0], "p <raw>": [3.0] * 7},
        "step change & more",
        xlabel="step",
        ylabel="rel. change",
        log_y=True,
        x=[0, 1, 2, 3, 4, 5, 6],
        markers={"jump": [3]},
    )
    assert svg.startswith("<figure><svg") and svg.endswith("</figure>")
    # gaps break the first series into one line and two isolated points
    assert svg.count("<polyline") == 2 and svg.count("<circle") == 2
    assert "velocity" in svg and "p &lt;raw&gt;" in svg and "jump" in svg
    assert "step change &amp; more" in svg and "rel. change" in svg and "1e0" in svg
    assert "stroke-dasharray" in svg and "<script" not in svg and "nan" not in svg.lower()

    spec = svg_timeseries(
        {"first": [1.0, 0.1, 0.01, 1e-3]}, "spectrum", "k", "E", log_y=True, x=[1, 2, 4, 8],
        log_x=True, width=300, height=160,
    )  # fmt: skip
    assert 'width="300"' in spec and 'height="160"' in spec and "1e-3" in spec
    per_series = svg_timeseries({"a": [1, 2, 3], "b": [2, 1]}, "t", x={"a": [0, 1, 2], "b": [5, 6]})
    assert per_series.count("<polyline") == 2
    assert "no data" in svg_timeseries({"a": [None, float("inf")]}, "empty")
    assert "<svg" in svg_timeseries({"flat": [2.0, 2.0, 2.0]}, "flat")


def test_steady_surface_reports_are_unchanged(tmp_path: Path) -> None:
    writer = DatasetWriter(tmp_path / "steady", name="steady-surface")
    for i in range(4):
        mesh = make_box_mesh(subdivisions=6)
        mesh.cell_fields["p"] = mesh.cell_fields["p"] * (1 + 0.1 * i)
        meta = SampleMeta(
            id=f"box_{i}",
            dataset="steady-surface",
            mesh=MeshStats(
                n_points=mesh.n_points,
                n_faces=mesh.n_faces,
                bounds=list(mesh.bounds),
                area=mesh.area,
                stl_area=mesh.area,
            ),
            geometry_params={"slant": 10.0 + i},
            provenance=Provenance(source="test"),
        )
        writer.write_sample(f"box_{i}", {0: mesh}, meta, {})
    writer.finalize()
    ds = Dataset(tmp_path / "steady")
    ds.manifest.splits = {"train": ["box_0", "box_1"], "val": ["box_2"], "test": ["box_3"]}
    ds.save_manifest()
    report = run_quality(ds)
    assert list(report) == [
        "dataset",
        "root",
        "generated_at",
        "elapsed_s",
        "summary",
        "dataset_checks",
        "field_statistics",
        "recommended_normalization",
        "samples",
    ]
    assert [c["check"] for c in report["dataset_checks"]] == ["parameter_coverage", "split_leakage"]
    for row in report["samples"]:
        assert "transient" not in row and "kind" not in row
        assert [c["check"] for c in row["checks"]] == [
            "nan_inf",
            "degenerate_faces",
            "area_sum",
            "force_consistency",
            "predictable_fields",
            "field_outliers",
            "duplicate_geometry",
        ]
        assert row["status"] in ("pass", "warn")
    page = (tmp_path / "steady" / "quality" / "report.html").read_text()
    assert "Transient" not in page and "<svg" in page


def test_predictable_fields_and_steady_mesh_checks() -> None:
    """A trajectory whose fields never change, or a sample without a real-valued field, has
    nothing to learn from; a steady mesh is checked against its own points and reference speed."""
    from types import SimpleNamespace

    from frontier_data.quality import check_freestream, check_mesh_cells
    from frontier_data.quality_transient import check_predictable_fields

    def scan(name: str, dnorm: float) -> Any:
        return SimpleNamespace(name=name, norm=np.array([2.0, 2.0]), dnorm=np.array([dnorm]))

    assert check_predictable_fields([], None).status == "fail"
    assert check_predictable_fields(["p"], None).status == "pass"
    frozen = check_predictable_fields(["u", "p"], [scan("u", 0.0), scan("p", 0.0)])
    assert frozen.status == "fail" and frozen.detail["reason"] == "no field changes over time"
    part = check_predictable_fields(["u", "p"], [scan("u", 0.5), scan("p", 0.0)])
    assert part.status == "warn" and part.detail == {"fields": ["u"], "constant": ["p"]}
    assert check_predictable_fields(["u"], [scan("u", 0.5)]).status == "pass"
    # an integer label that is stored per step is not a field to predict
    assert check_predictable_fields([], [scan("node_type", 0.0)]).status == "fail"

    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1]], dtype=np.float32)
    quads = np.array([[0, 1, 2, 3], [1, 4, 5, 2]])
    assert check_mesh_cells(pts, quads).status == "pass"
    assert check_mesh_cells(pts, np.array([[0, 1, 2, 9]])).status == "fail"  # no such point
    flat = check_mesh_cells(pts, np.array([[0, 1, 1, 0]] * 3 + [[0, 1, 2, 3]]))
    assert flat.status == "fail" and flat.detail["degenerate"] == 3
    ring = np.stack([np.cos(np.linspace(0, 6.28, 400)), np.sin(np.linspace(0, 6.28, 400))], 1)
    cloud = np.concatenate([ring * 50, np.random.default_rng(0).normal(size=(4000, 2))])
    vel = np.tile([30.0, 0.0], (len(cloud), 1))
    assert check_freestream(cloud, vel, {"u_inf": 30.0}).status == "pass"
    assert check_freestream(cloud, vel, {"u_inf": 3.0}).status == "fail"  # a factor of ten
    assert check_freestream(cloud, None, {"u_inf": 30.0}).status == "skip"

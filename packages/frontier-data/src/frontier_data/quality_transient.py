# SPDX-License-Identifier: Apache-2.0
"""Quality checks for transient (time-dependent) and non-surface samples (format v1.1).

:func:`transient_sample_pass` is the per-sample pass that :func:`frontier_data.quality.run_quality`
uses for every sample that is not a steady surface. Field scans use bounded ArrayRoute
batches across steps; spectra select the spatial crop before reading. Other field summaries
and static non-finite checks stream bounded reads. Signal normalization uses bounded fitted
readers; time axes, retained reports and FFT workspaces remain separate. The pass yields:

* ``nan_inf_per_step``: non-finite values per step and field.
* ``temporal_continuity``: jumps, frozen frames, duplicated steps, and the ``times`` axis.
* ``cfl``: ``|u| dt / dx`` of the *saved* steps (how far the flow moves between two frames).
* ``stationarity``: is the window statistically stationary, and how many independent samples
  does it hold (``stationary`` / ``transient-start`` / ``drifting`` / ``too-short``).
* ``spectra``: shell-averaged energy / enstrophy spectra of regular grids (resolution check).
* ``conservation``: drift of mean, kinetic energy, enstrophy; fails only on a blow-up.
* ``temporal_subsampling``: a recommended training stride from the lag autocorrelation.

Dataset level: :func:`check_temporal_consistency` (same ``dt`` / steps / fields across samples)
and :func:`summarize_transient` (the ``report["transient"]`` object). The trajectory leakage
check lives in :mod:`frontier_data.splits`.

Cost model: the cheap statistics (non-finite count, norms, step change, hash, spatial mean and
RMS) are taken at every step up to :data:`MAX_SCAN_STEPS` steps (longer trajectories are scanned
with a stride, reported as ``stride`` in the check details). Autocorrelations use at most
:data:`EXPENSIVE_MAX_STEPS` time origins and spectra a handful of steps. Percentiles, the CFL
p99 for time-dependent velocity and the autocorrelation use fixed probe points. Static or
steady velocity CFL uses exact full-selection speed percentiles through a derived reader.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from arrayroute.backends import NumpyBackend
from arrayroute.derived import VectorMagnitudeBackend
from arrayroute.hashing import DigestPipeline
from arrayroute.ordered import OrderedSelection, read_flat
from arrayroute.quantiles import count_nonfinite, finite_statistics
from arrayroute.reader import ArrayReader, ReadStats
from arrayroute.rows import read_rows
from arrayroute.standardized import ALGORITHM as SIGNAL_NORMALIZATION_ALGORITHM
from arrayroute.standardized import StandardizedChannelsBackend
from arrayroute.statistics import CompensatedSum
from frontier_data.quality import CheckResult, Status, field_stats, streamed_field_stats
from frontier_data.splits import TrajectoryFingerprint

if TYPE_CHECKING:
    from frontier_data.arrays import Sample

FIELD_SCAN_ALGORITHM = "c-order-compensated-field-scan-v1"

# -- thresholds -----------------------------------------------------------------------------

#: cheap per-step statistics are taken at every step up to this many, then with a stride
MAX_SCAN_STEPS = 4096
#: time origins used for autocorrelations ("expensive" checks) on long trajectories
EXPENSIVE_MAX_STEPS = 256
#: every series stored in the report is downsampled to at most this many points
SERIES_MAX_POINTS = 200
#: probe points per field (fewer for long trajectories: at most PROBE_BUDGET values in memory)
PROBE_POINTS = 2048
PROBE_BUDGET = 4_000_000
#: probe values per frame in a trajectory fingerprint (leakage check)
LEAK_PROBE_VALUES = 512

#: a step whose relative change to the previous one is below this is a frozen (duplicated) frame
FROZEN_REL = 1e-7
#: jump detection: step change vs the median of the +-JUMP_WINDOW neighbouring step changes
JUMP_WINDOW = 5
JUMP_Z = 6.0  # robust z-score of the log10 ratio (median/MAD over the trajectory)
JUMP_MIN_RATIO = 4.0  # ... and at least this many times the neighbours -> warn
JUMP_FAIL_RATIO = 20.0  # ... this many times -> fail
JUMP_SIGMA_FLOOR = 0.05  # decades; keeps a perfectly regular series from flagging noise
TIME_UNIFORM_RTOL = 1e-4  # same tolerance as frontier_data.arrays.time_info

#: saved-step CFL above this: frames are far apart relative to advection -> warn
CFL_WARN = 20.0
#: saved-step CFL below this: oversampled in time -> pass with a recommended stride
CFL_TINY = 0.05
#: the stride recommended for oversampled data brings the saved-step CFL to about this
CFL_TARGET = 0.5

STATIONARITY_MIN_STEPS = 16  # shorter windows are "too-short" (check skipped)
HALF_Z_DRIFT = 3.5  # first-half vs second-half mean, in standard errors -> drifting
HALF_Z_SOFT = 2.0  # ... a weaker half difference counts together with DRIFT_WARN
DRIFT_WARN = 0.5  # drift of the running mean over the last third / fluctuation std
N_EFF_MIN = 10.0  # fewer independent samples than this -> too-short
CONSTANT_REL_TOL = 1e-6  # a series varying less than this (relative) is constant
TRANSIENT_START_SIGMA = 2.0  # head mean vs tail mean, in tail std, for a start-up transient
#: check status per stationarity class (never "fail": decaying flows are legitimate data)
STATIONARITY_STATUS: dict[str, Status] = {
    "stationary": "pass",
    "transient-start": "warn",
    "drifting": "warn",
    "too-short": "warn",
}

SPECTRA_MAX_SIDE = 256  # larger grids are cropped (and windowed) before the FFT
SPECTRA_AVG_STEPS = 8  # steps in the time-averaged spectrum
HIGH_K_WARN = 0.05  # fraction of energy in the top third of wavenumbers -> warn
PILEUP_MIN_FRACTION = 0.005  # a spectrum rising towards the cutoff with this much energy -> warn

EXPLODE_FACTOR = 100.0  # growth of a positive integral vs its median over time -> fail
INITIAL_SPIKE_FRACTION = 0.05  # a peak within this leading fraction of the steps: start-up
NON_FINITE_FRACTION_FAIL = 0.01  # a step with this fraction of non-finite values: blow-up

SUBSAMPLE_MIN_CORR = 0.9  # frames `stride` apart keep at least this autocorrelation
SUBSAMPLE_MIN_STEPS = 16  # ... and the subsampled trajectory keeps at least this many frames
DECORRELATION_LEVEL = math.exp(-1.0)

_STATIONARITY_ORDER = ("stationary", "too-short", "transient-start", "drifting")


def worst_status(statuses: Sequence[Status]) -> Status:
    order = {"pass": 0, "skip": 0, "warn": 1, "fail": 2}
    out: Status = "pass"
    for s in statuses:
        if order[s] > order[out]:
            out = s
    return out


# -- small numeric helpers ------------------------------------------------------------------


def _num(v: Any, digits: int = 6) -> float | None:
    """A JSON-safe float: non-finite -> ``None``, rounded to ``digits`` significant digits."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return float(f"{f:.{digits}g}")


def _clean(obj: Any) -> Any:
    """Make a detail structure strictly JSON-serialisable (no NaN/inf, no numpy types)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_ | bool):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


def downsample(values: Sequence[float] | np.ndarray, n: int = SERIES_MAX_POINTS) -> np.ndarray:
    """Indices of at most ``n`` evenly spaced entries of ``values`` (first and last kept)."""
    m = len(values)
    if m <= n:
        return np.arange(m)
    return np.unique(np.round(np.linspace(0, m - 1, n)).astype(np.int64))


def _series(values: np.ndarray, idx: np.ndarray) -> list[float | None]:
    return [_num(v, 5) for v in np.asarray(values, dtype=np.float64)[idx]]


def _bucket_max(values: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Downsample by taking the maximum of ``n`` buckets, so isolated spikes survive.
    Returns (index of the last entry of each bucket, bucket maximum)."""
    m = len(values)
    if m <= n:
        return np.arange(m), np.asarray(values, dtype=np.float64)
    edges = np.round(np.linspace(0, m, n + 1)).astype(np.int64)
    edges = np.unique(edges)
    v = np.where(np.isfinite(values), values, -np.inf)
    mx = np.maximum.reduceat(v, edges[:-1])
    return edges[1:] - 1, np.where(np.isfinite(mx), mx, np.nan)


def integrated_autocorr_time(x: np.ndarray, c: float = 5.0) -> float:
    """Integrated autocorrelation time ``tau = 1/2 + sum_k rho(k)`` with Sokal's automatic
    window (the sum stops at the first lag ``M >= c * tau(M)``). ``tau = 0.5`` for white
    noise; the number of independent samples in ``T`` steps is ``T / (2 tau)``."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 4:
        return 0.5
    x = x - x.mean()
    var = float(x @ x) / n
    if not math.isfinite(var) or var <= 0:
        return 0.5
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / (n * var)
    taus = 0.5 + np.cumsum(acf[1 : max(n // 2, 2)])
    lags = np.arange(1, taus.size + 1)
    hit = np.flatnonzero(lags >= c * taus)
    tau = float(taus[hit[0]]) if hit.size else float(taus[-1])
    return max(tau, 0.5)


def mser_truncation(x: np.ndarray) -> int:
    """MSER truncation point: the number of initial steps ``d`` (``0 <= d <= T/2``) whose
    removal minimises the squared standard error ``var(x[d:]) / (T - d)`` of the mean."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 4:
        return 0
    # suffix sums give var(x[d:]) for every d in O(T)
    s1 = np.cumsum(x[::-1])[::-1]
    s2 = np.cumsum((x * x)[::-1])[::-1]
    m = n - np.arange(n)
    var = np.maximum(s2 / m - (s1 / m) ** 2, 0.0)
    score = (var / m)[: n // 2 + 1]
    best = int(np.argmin(score))
    # ties and round-off: do not truncate unless it helps by a clear margin
    return best if score[best] < 0.999 * score[0] else 0


def _series_stats(x: np.ndarray) -> dict[str, float]:
    """Stationarity statistics of one time series (see :func:`classify_stationarity`)."""
    n = x.size
    t = np.arange(n, dtype=np.float64)
    std = float(x.std())
    resid = x - np.polyval(np.polyfit(t, x, 1), t)
    tau = integrated_autocorr_time(resid)
    sd_r = float(resid.std())
    h = n // 2
    diff = abs(float(x[:h].mean()) - float(x[h:].mean()))
    se = sd_r * math.sqrt(2.0 * tau * (1.0 / h + 1.0 / (n - h)))
    floor = 1e-12 * max(abs(float(x.mean())), std, 1e-300)
    z = diff / se if se > floor else (0.0 if diff <= floor else math.inf)
    running = np.cumsum(x) / (t + 1.0)
    drift = abs(float(running[-1] - running[(2 * n) // 3 - 1])) / std if std > 0 else 0.0
    return {
        "half_z": float(z),
        "drift_last_third": float(drift),
        "tau_int": float(tau),
        "n_eff": float(n / (2.0 * tau)),
        "std": std,
    }


def _is_drifting(st: dict[str, float]) -> bool:
    return st["half_z"] >= HALF_Z_DRIFT or (
        st["drift_last_third"] >= DRIFT_WARN and st["half_z"] >= HALF_Z_SOFT
    )


def classify_stationarity(x: np.ndarray, scale: float | None = None) -> dict[str, Any]:
    """Classify a time series as ``stationary`` / ``transient-start`` / ``drifting`` /
    ``too-short``.

    * ``half_z``: mean of the first half vs the second half in standard errors, the standard
      error being ``std * sqrt(2 tau_int / n)`` with ``tau_int`` and ``std`` taken from the
      linearly detrended series (a trend would otherwise inflate both and hide itself).
    * ``drift_last_third``: change of the running (cumulative) mean over the last third of the
      window, in units of the fluctuation std.
    * ``n_eff = T / (2 tau_int)``: independent samples in the window.
    * ``burn_in``: MSER truncation point, reported when dropping it leaves a stationary tail
      whose mean differs from the head by more than ``TRANSIENT_START_SIGMA`` tail stds.

    ``scale`` is the magnitude the variation is compared with to decide that a series is
    constant (defaults to the series' own RMS).
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    out: dict[str, Any] = {"n": n, "burn_in": 0}
    if n < STATIONARITY_MIN_STEPS:
        return {**out, "class": "too-short", "reason": f"fewer than {STATIONARITY_MIN_STEPS} steps"}
    ref = scale if scale is not None and scale > 0 else float(np.sqrt(np.mean(x * x)))
    if float(np.ptp(x)) <= CONSTANT_REL_TOL * max(ref, 1e-300):
        return {
            **out,
            "class": "stationary",
            "constant": True,
            "half_z": 0.0,
            "drift_last_third": 0.0,
            "tau_int": 0.5,
            "n_eff": float(n),
        }
    d = mser_truncation(x)
    if d >= max(3, math.ceil(0.02 * n)) and n - d >= STATIONARITY_MIN_STEPS:
        tail = x[d:]
        st = _series_stats(tail)
        gap = abs(float(x[:d].mean()) - float(tail.mean()))
        if gap >= TRANSIENT_START_SIGMA * st["std"] and not _is_drifting(st):
            return {**out, **st, "class": "transient-start", "burn_in": int(d)}
    st = _series_stats(x)
    if _is_drifting(st):
        cls = "drifting"
    elif st["n_eff"] < N_EFF_MIN:
        cls = "too-short"
    else:
        cls = "stationary"
    return {**out, **st, "class": cls}


# -- the per-step scan ----------------------------------------------------------------------


@dataclass
class FieldScan:
    """Per-step statistics of one time-dependent field (``S`` scanned steps)."""

    name: str
    n_components: int
    step_shape: tuple[int, ...]
    steps: np.ndarray  # (S,) step indices that were read
    non_finite: np.ndarray  # (S,) number of non-finite values
    norm: np.ndarray  # (S,) L2 norm of the finite values
    mean: np.ndarray  # (S,) spatial mean over all components
    rms: np.ndarray  # (S,) sqrt(mean(u^2)) over all components
    vmin: np.ndarray
    vmax: np.ndarray
    dnorm: np.ndarray  # (S-1,) ||u[s+1] - u[s]||
    hashes: list[str]
    probes: np.ndarray  # (S, P, C) float32 values at the probe points
    n_nan: int = 0
    n_inf: int = 0
    speed_max: np.ndarray | None = None  # (S,) max |u| for a velocity field
    leak_first: np.ndarray = field(default_factory=lambda: np.zeros(0))
    leak_last: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: measured signals (sensors, SCADA): a sudden change is physics (a gust, a start), so
    #: jumps and coincidentally equal frames only warn; a stalled frame still fails
    measured: bool = False
    normalization: dict[str, Any] | None = None

    @property
    def n_values(self) -> int:
        return int(np.prod(self.step_shape)) if self.step_shape else 1

    @property
    def rel_change(self) -> np.ndarray:
        """``||u[s+1] - u[s]|| / ||u[s]||`` (``/ ||u[s+1]||`` when ``u[s]`` is identically 0)."""
        a, b = self.norm[:-1], self.norm[1:]
        den = np.where(a > 0, a, b)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(den > 0, self.dnorm / np.where(den > 0, den, 1.0), 0.0)
        return r

    @property
    def trivial(self) -> np.ndarray:
        """Steps whose frame is a single constant value."""
        return self.vmin == self.vmax


@dataclass
class SampleInfo:
    """What the checks need to know about a sample's layout, looked up once (every
    ``sample.array`` / ``sample.dims`` call goes back to the store's metadata)."""

    sample: Sample
    is_transient: bool
    names: list[str]  # field_names(): point/grid fields, then cell fields
    arrays: dict[str, Any]  # name -> zarr array
    dims: dict[str, list[str]]
    static_names: list[str]
    velocity: tuple[str, str] | None = None  # see find_velocity_field

    def time_dependent(self, name: str) -> bool:
        return self.is_transient and self.dims[name][:1] == ["time"]


def inspect_sample(sample: Sample | SampleInfo) -> SampleInfo:
    """Build the :class:`SampleInfo` of a sample (a no-op when given one)."""
    if isinstance(sample, SampleInfo):
        return sample
    names = sample.field_names()
    arrays = {n: sample.array(n) for n in names}
    dims: dict[str, list[str]] = {}
    for n, arr in arrays.items():
        d = arr.attrs.get("dims")
        dims[n] = [str(x) for x in d] if isinstance(d, list) else sample.dims(n)
    info = SampleInfo(
        sample=sample,
        is_transient=sample.is_transient,
        names=names,
        arrays=arrays,
        dims=dims,
        static_names=sample.static_names,
    )
    info.velocity = find_velocity_field(info)
    return info


def is_signal_sample(sample: Sample) -> bool:
    """A one-cell transient grid whose state is its global time series ``signals/<name>``
    ``(T, C)`` (SCADA, test-bench and historian channels); its grid fields are bookkeeping
    (an observed-value mask), not state."""
    if not sample.is_transient or sample.kind != "grid" or int(np.prod(sample.shape)) != 1:
        return False
    names = sample.signal_names
    return bool(names) and any(sample.array(f"signals/{n}").ndim == 2 for n in names)


def signal_reader(
    sample: Sample, *, workspace_bytes: int = 32 << 20,
) -> tuple[ArrayReader, list[str], dict[str, Any]]:
    """Fit finite channel statistics, then read standardized signals lazily in source order."""
    sources, names, source_names = [], [], []
    for n in sample.signal_names:
        a = sample.array(f"signals/{n}")
        if a.ndim != 2:
            continue
        sources.append(sample._reader(a))
        source_names.append(n)
        names += [n] if a.shape[1] == 1 else [f"{n}_{i}" for i in range(a.shape[1])]
    backend = StandardizedChannelsBackend(
        sources, workspace_bytes=workspace_bytes, source_stats=sample.read_stats,
    )
    details = {
        "algorithm": SIGNAL_NORMALIZATION_ALGORITHM, "sources": source_names,
        "channels": names, "parameters": [asdict(p) for p in backend.parameters],
        "workspace_bytes": workspace_bytes, "finite_policy": "exclude_nan_and_infinity",
        "fingerprint_scope": "native_source_bytes_during_fitting",
    }
    return ArrayReader(backend, buffer_bytes=16 << 20), names, details


def scan_signals(
    info: SampleInfo, steps: np.ndarray, *, workspace_bytes: int = 32 << 20,
) -> FieldScan:
    """The scan of a signal sample's standardised channel vector, named ``signals``."""
    z, _, details = signal_reader(info.sample, workspace_bytes=workspace_bytes)
    info.arrays["signals"] = z
    info.dims["signals"] = ["time", "component"]
    try:
        scan = scan_field(info, "signals", steps, velocity=False)
    finally:
        del info.arrays["signals"], info.dims["signals"]
    scan.measured = True
    scan.normalization = details
    return scan


def scan_steps(n_steps: int, max_steps: int = MAX_SCAN_STEPS) -> tuple[np.ndarray, int]:
    """Step indices to scan and the stride between them (the last step is always included)."""
    stride = max(1, math.ceil(n_steps / max(max_steps, 1)))
    steps = np.arange(0, n_steps, stride, dtype=np.int64)
    if steps.size and steps[-1] != n_steps - 1:
        steps = np.append(steps, n_steps - 1)
    return steps, stride


def _probe_index(n_entities: int, n: int, salt: int = 0) -> np.ndarray:
    """A deterministic random subset of entity indices: samples with the same number of
    points get the same probe points, so their probe values are comparable."""
    n = min(n, n_entities)
    rng = np.random.default_rng(1_000_003 * salt + n_entities)
    return np.sort(rng.choice(n_entities, size=n, replace=False))


def scan_field(
    sample: Sample | SampleInfo, name: str, steps: np.ndarray, *, velocity: bool | None = None
) -> FieldScan:
    """Scan fixed C-order reduction blocks within bounded I/O batches across steps.

    Hashes/probes/counts are exact. Compensated block reductions have a versioned fixed
    partition independent of storage/read budgets; they can differ in last bits from a
    whole-frame NumPy reduction. Numeric scratch is bounded separately from reader
    allowances and retained per-step report/probe arrays. ``velocity`` records max |u|.
    """
    info = inspect_sample(sample)
    arr = info.arrays[name]
    reader = (
        arr if isinstance(arr, ArrayReader) else
        ArrayReader(NumpyBackend(arr), buffer_bytes=64 << 20)
        if isinstance(arr, np.ndarray) else info.sample._reader(arr)
    )
    if velocity is None:
        velocity = info.velocity == ("field", name)
    step_shape = reader.backend.spec.shape[1:]
    n_comp = step_shape[-1] if info.dims[name][-1:] == ["component"] and step_shape else 1
    if n_comp * 64 > 8 << 20:
        raise MemoryError("Field components exceed the quality-scan workspace allowance")
    read_stats = ReadStats() if isinstance(arr, ArrayReader) else info.sample.read_stats
    n_ent = max(int(np.prod(step_shape)) // max(n_comp, 1), 1)
    n_s = int(steps.size)
    n_probe = int(np.clip(PROBE_BUDGET // max(n_s * n_comp, 1), 64, PROBE_POINTS))
    pidx = _probe_index(n_ent, n_probe)
    lidx = _probe_index(n_ent, max(LEAK_PROBE_VALUES // n_comp, 1), salt=1)

    scan = FieldScan(
        name=name,
        n_components=n_comp,
        step_shape=step_shape,
        steps=steps,
        non_finite=np.zeros(n_s, dtype=np.int64),
        norm=np.zeros(n_s),
        mean=np.full(n_s, np.nan),
        rms=np.full(n_s, np.nan),
        vmin=np.full(n_s, np.nan),
        vmax=np.full(n_s, np.nan),
        dnorm=np.zeros(max(n_s - 1, 0)),
        hashes=[],
        probes=np.zeros((n_s, pidx.size, n_comp), dtype=np.float32),
        speed_max=np.zeros(n_s) if velocity else None,
    )
    sources = [
        OrderedSelection.create(reader, (int(i), *(slice(None) for _ in step_shape))) for i in steps
    ]
    block_rows = min(65536, max(1, (8 << 20) // (64 * n_comp)))
    sums = [CompensatedSum() for _ in steps]
    squares = [CompensatedSum() for _ in steps]
    deltas = [CompensatedSum() for _ in range(max(0, n_s - 1))]
    digests = [hashlib.blake2b(digest_size=8) for _ in steps]
    scan.leak_first = np.zeros((lidx.size, n_comp), dtype=np.float64).reshape(-1)
    scan.leak_last = scan.leak_first.copy()
    # I/O batches may vary with the reader allowance; reduction leaves never do.
    io_bytes = min(16 << 20, reader.buffer_bytes // 2)
    io_rows = max(
        block_rows, (io_bytes // (n_comp * max(8, reader.backend.dtype.itemsize)))
        // block_rows * block_rows
    )
    def frames(first_entity: int, last_entity: int):
        if n_ent == 1:
            row_bytes = n_comp * reader.backend.dtype.itemsize
            batch_rows = max(1, (reader.buffer_bytes // 2) // max(1, row_bytes))
            for first in range(0, n_s, batch_rows):
                batch = read_rows(
                    reader, steps[first:first + batch_rows].tolist(), stats=read_stats,
                )
                for offset in range(len(batch)):
                    yield first + offset, batch[offset].reshape(-1)
                del batch
        else:
            for index, source in enumerate(sources):
                yield index, read_flat(
                    source, first_entity * n_comp, last_entity * n_comp, read_stats, None,
                )

    with DigestPipeline(buffer_bytes=io_rows * n_comp * reader.backend.dtype.itemsize) as hashing:
        for outer in range(0, n_ent, io_rows):
            end = min(n_ent, outer + io_rows)
            prev: np.ndarray | None = None
            for j, raw in frames(outer, end):
                hashing.update(digests[j], raw)
                current = raw.astype(np.float64).reshape(end - outer, n_comp)
                del raw
                for start in range(outer, end, block_rows):
                    stop = min(end, start + block_rows)
                    p0, p1 = np.searchsorted(pidx, [start, stop])
                    l0, l1 = np.searchsorted(lidx, [start, stop])
                    a = current[start - outer : stop - outer]
                    finite = np.isfinite(a)
                    n_bad = int(a.size - np.count_nonzero(finite))
                    if n_bad:
                        scan.non_finite[j] += n_bad
                        n_nan = int(np.isnan(a).sum())
                        scan.n_nan += n_nan
                        scan.n_inf += n_bad - n_nan
                        good = a[finite]
                        a[~finite] = 0.0
                    else:
                        good = a
                    if good.size:
                        squares[j].add(float(np.einsum("ij,ij->", a, a)))
                        sums[j].add(float(a.sum()))
                        lo, hi = float(good.min()), float(good.max())
                        scan.vmin[j] = lo if np.isnan(scan.vmin[j]) else min(scan.vmin[j], lo)
                        scan.vmax[j] = hi if np.isnan(scan.vmax[j]) else max(scan.vmax[j], hi)
                    if scan.speed_max is not None:
                        scan.speed_max[j] = max(
                            scan.speed_max[j], float(np.sqrt(np.einsum("ij,ij->i", a, a).max()))
                        )
                    scan.probes[j, p0:p1] = a[pidx[p0:p1] - start]
                    if j == 0:
                        scan.leak_first.reshape(-1, n_comp)[l0:l1] = a[lidx[l0:l1] - start]
                    if j == n_s - 1:
                        scan.leak_last.reshape(-1, n_comp)[l0:l1] = a[lidx[l0:l1] - start]
                    if prev is not None:
                        d = a - prev[start - outer : stop - outer]
                        deltas[j - 1].add(float(np.einsum("ij,ij->", d, d)))
                        del d
                    del finite, good, a
                prev = current
                del current
            del prev
    scan.hashes = [digest.hexdigest() for digest in digests]
    for j in range(n_s):
        n_good = n_ent * n_comp - int(scan.non_finite[j])
        if n_good:
            scan.norm[j] = math.sqrt(squares[j].value)
            scan.mean[j] = sums[j].value / n_good
            scan.rms[j] = math.sqrt(squares[j].value / n_good)
        if j:
            scan.dnorm[j - 1] = math.sqrt(deltas[j - 1].value)
    return scan


def _scan_field_stats(scan: FieldScan) -> dict[str, float]:
    """Dataset-style field statistics from a scan: mean/std/min/max over every scanned step,
    percentiles from the probe points."""
    n_good = scan.n_values - scan.non_finite
    total = float(n_good.sum())
    ok = np.isfinite(scan.mean)
    if total <= 0 or not ok.any():
        return {
            **field_stats(np.array([np.nan])),
            "n_nan": float(scan.n_nan),
            "n_inf": float(scan.n_inf),
        }
    mean = float((scan.mean[ok] * n_good[ok]).sum() / total)
    msq = float((scan.rms[ok] ** 2 * n_good[ok]).sum() / total)
    p01, p99 = np.percentile(scan.probes, [1, 99])
    return {
        "mean": mean,
        "std": math.sqrt(max(msq - mean * mean, 0.0)),
        "min": float(np.nanmin(scan.vmin)),
        "max": float(np.nanmax(scan.vmax)),
        "p01": float(p01),
        "p99": float(p99),
        "n_nan": float(scan.n_nan),
        "n_inf": float(scan.n_inf),
    }


# -- field roles ----------------------------------------------------------------------------


def is_velocity_name(name: str) -> bool:
    leaf = name.rsplit("/", 1)[-1]
    return "velocity" in leaf.lower() or leaf in ("u", "U")


def space_dim(sample: Sample) -> int:
    if sample.kind == "grid":
        return len(sample.shape)
    if "points" in sample.group:
        pts: Any = sample.group["points"]
        return int(pts.shape[-1])
    return 3


def find_velocity_field(sample: Sample | SampleInfo) -> tuple[str, str] | None:
    """``(source, name)`` of the velocity-like field: a field whose name contains ``velocity``
    (or is ``u`` / ``U``) with as many components (>= 2) as space dimensions. ``source`` is
    ``"field"`` (time-dependent), ``"steady"`` (in ``fields/`` without a time axis) or
    ``"static"`` (under ``static/``)."""
    if isinstance(sample, SampleInfo):
        info, sample = sample, sample.sample
    else:
        return inspect_sample(sample).velocity
    dim = space_dim(sample)

    def fits(n_comp: int) -> bool:
        return n_comp >= 2 and n_comp == dim

    for name in info.names:
        if is_velocity_name(name) and fits(int(info.arrays[name].shape[-1])):
            return ("field" if info.time_dependent(name) else "steady", name)
    for name in info.static_names:
        if is_velocity_name(name):
            arr = sample.array(f"static/{name}")
            if arr.ndim >= 2 and fits(int(arr.shape[-1])):
                return ("static", name)
    return None


def median_edge_length(points: np.ndarray, cells: np.ndarray, max_cells: int = 200_000) -> float:
    """Median edge length of a mesh from its ``(K, P)`` connectivity padded with -1 (the edges
    between consecutive cell nodes, on a random subset of at most ``max_cells`` cells)."""
    cells = np.asarray(cells)
    if cells.ndim != 2 or cells.shape[0] == 0 or cells.shape[1] < 2:
        return math.nan
    if cells.shape[0] > max_cells:
        rng = np.random.default_rng(0)
        cells = cells[rng.choice(cells.shape[0], size=max_cells, replace=False)]
    pts = np.asarray(points, dtype=np.float64)
    lengths: list[np.ndarray] = []
    n_nodes = (cells >= 0).sum(axis=1)
    for j in range(cells.shape[1]):
        a = cells[:, j]
        nxt = cells[:, (j + 1) % cells.shape[1]]
        b = np.where((j + 1 < n_nodes) & (nxt >= 0), nxt, cells[:, 0])
        ok = (a >= 0) & (b >= 0) & (a != b) & (j < n_nodes) & (n_nodes > 2) | (
            (n_nodes == 2) & (j == 0)
        )
        if ok.any():
            lengths.append(np.linalg.norm(pts[a[ok]] - pts[b[ok]], axis=1))
    if not lengths:
        return math.nan
    ell = np.concatenate(lengths)
    ell = ell[np.isfinite(ell) & (ell > 0)]
    return float(np.median(ell)) if ell.size else math.nan


def characteristic_spacing(sample: Sample) -> tuple[float, str] | None:
    """``(dx, how)``: the smallest grid spacing, the median mesh edge length, or the median
    nearest-neighbour distance of a point cloud. ``None`` when it cannot be determined."""
    if sample.kind == "grid":
        sp = sample.spacing
        if sp is None:
            sp = [float(np.median(np.diff(c))) for c in sample.coords.values() if len(c) > 1]
        sp = [abs(float(s)) for s in sp if math.isfinite(float(s)) and float(s) != 0]
        return (min(sp), "min grid spacing") if sp else None
    if sample.kind in ("mesh", "surface"):
        dx = median_edge_length(sample.points(), sample.cells)
        return (dx, "median edge length") if math.isfinite(dx) and dx > 0 else None
    if sample.kind == "point_cloud":
        from scipy.spatial import KDTree

        pts = np.asarray(sample.points(), dtype=np.float64)
        if len(pts) < 2 or len(pts) > 2_000_000:
            return None
        query = pts[_probe_index(len(pts), 2000)]
        dist, _ = KDTree(pts).query(query, k=2)
        nn = np.asarray(dist)[:, 1]
        nn = nn[np.isfinite(nn) & (nn > 0)]
        return (float(np.median(nn)), "median nearest-neighbour distance") if nn.size else None
    return None


# -- per-sample checks ----------------------------------------------------------------------


#: a field counts as changing when one step differs from the previous by more than this
#: fraction of the field's norm
VARIATION_REL = 1e-9


def check_predictable_fields(float_fields: list[str], scans: list[FieldScan] | None) -> CheckResult:
    """There is something to learn from the sample: at least one real-valued field and, for a
    transient sample (``scans`` given), at least one of them changes over time. A trajectory
    whose fields are constant, or whose only stored field is a label, fails."""
    if not float_fields:
        return CheckResult(
            "predictable_fields", "fail", {"reason": "no real-valued field", "fields": []}
        )
    if scans is None:
        return CheckResult("predictable_fields", "pass", {"fields": float_fields})
    varying, constant = [], []
    for scan in scans:
        if scan.name not in float_fields:
            continue
        scale = float(np.nanmax(scan.norm)) if scan.norm.size else 0.0
        change = float(np.nanmax(scan.dnorm)) if scan.dnorm.size else 0.0
        (varying if change > VARIATION_REL * max(scale, 1e-300) else constant).append(scan.name)
    if not varying:
        return CheckResult(
            "predictable_fields",
            "fail",
            {"reason": "no field changes over time", "fields": [], "constant": constant},
        )
    return CheckResult(
        "predictable_fields",
        "warn" if constant else "pass",
        {"fields": sorted(varying), "constant": sorted(constant)},
    )


def check_nan_inf_per_step(
    scans: Sequence[FieldScan], other_bad: dict[str, int] | None = None, stride: int = 1
) -> CheckResult:
    """Non-finite values per step for every time-dependent field (``other_bad``: counts for
    time-independent arrays). Fails on any."""
    fields: dict[str, Any] = {}
    bad_any = False
    for s in scans:
        bad = np.flatnonzero(s.non_finite)
        if bad.size:
            bad_any = True
            fields[s.name] = {
                "n_bad_steps": int(bad.size),
                "total_non_finite": int(s.non_finite.sum()),
                "n_nan": s.n_nan,
                "n_inf": s.n_inf,
                "first_steps": [
                    {"step": int(s.steps[b]), "count": int(s.non_finite[b])} for b in bad[:10]
                ],
            }
    other = {k: int(v) for k, v in (other_bad or {}).items() if v}
    detail = {
        "fields": fields,
        "time_independent": other,
        "n_steps_checked": int(max((s.steps.size for s in scans), default=1)),
        "stride": stride,
    }
    return CheckResult("nan_inf_per_step", "fail" if bad_any or other else "pass", detail)


def check_times(times: np.ndarray | None, n_steps_arrays: dict[str, int]) -> dict[str, Any]:
    """The ``times`` axis: strictly increasing, duplicates, uniform, same length as arrays."""
    if times is None:
        return {"status": "skip", "reason": "no times array"}
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    out: dict[str, Any] = {"n": int(t.size), "status": "pass"}
    mismatch = {k: v for k, v in n_steps_arrays.items() if v != t.size}
    if mismatch:
        out["length_mismatch"] = mismatch
        out["status"] = "fail"
    if not np.isfinite(t).all():
        out["non_finite"] = int((~np.isfinite(t)).sum())
        out["status"] = "fail"
    if t.size < 2:
        return {**out, "uniform": True, "strictly_increasing": True, "n_duplicates": 0}
    d = np.diff(t)
    dt = float(np.median(d))
    n_dup = int((d == 0).sum())
    increasing = bool(np.all(d > 0))
    non_uniform = np.abs(d - dt) > TIME_UNIFORM_RTOL * max(abs(dt), 1e-300)
    out.update(
        {
            "strictly_increasing": increasing,
            "n_duplicates": n_dup,
            "duplicate_steps": [int(i) + 1 for i in np.flatnonzero(d == 0)[:10]],
            "n_decreasing": int((d < 0).sum()),
            "uniform": not bool(non_uniform.any()),
            "n_non_uniform": int(non_uniform.sum()),
            "non_uniform_steps": [int(i) + 1 for i in np.flatnonzero(non_uniform)[:10]],
            "dt_median": dt,
            "dt_min": float(d.min()),
            "dt_max": float(d.max()),
        }
    )
    if not increasing:
        out["status"] = "fail"
    elif non_uniform.any() and out["status"] == "pass":
        out["status"] = "warn"
    return out


def _find_jumps(r: np.ndarray, frozen: np.ndarray) -> list[dict[str, Any]]:
    """Indices into the step-change series ``r`` that stand out from their neighbours."""
    ok = np.isfinite(r) & (r > 0) & ~frozen
    if ok.sum() < 2 * JUMP_WINDOW:
        return []
    lg = np.where(ok, np.log10(np.where(ok, r, 1.0)), np.nan)
    w = JUMP_WINDOW
    padded = np.concatenate([np.full(w, np.nan), lg, np.full(w, np.nan)])
    win = np.lib.stride_tricks.sliding_window_view(padded, 2 * w + 1).copy()
    win[:, w] = np.nan  # leave the centre out
    with warnings.catch_warnings():  # all-NaN windows are expected next to frozen runs
        warnings.simplefilter("ignore", RuntimeWarning)
        local = np.nanmedian(win, axis=1)
    dev = lg - local
    good = np.isfinite(dev)
    if good.sum() < 3:
        return []
    med = float(np.median(dev[good]))
    mad = float(np.median(np.abs(dev[good] - med)))
    sigma = max(1.4826 * mad, JUMP_SIGMA_FLOOR)
    out = []
    for i in np.flatnonzero(good):
        z = (dev[i] - med) / sigma
        ratio = 10.0 ** dev[i]
        if z >= JUMP_Z and ratio >= JUMP_MIN_RATIO:
            out.append({"index": int(i), "ratio": float(ratio), "z": float(z)})
    return out


def _runs(idx: np.ndarray) -> list[tuple[int, int]]:
    """Consecutive runs ``(first, last)`` of a sorted integer index array."""
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [idx.size - 1]])
    return [(int(idx[a]), int(idx[b])) for a, b in zip(starts, ends, strict=True)]


def check_temporal_continuity(
    scans: Sequence[FieldScan], times: np.ndarray | None = None, stride: int = 1
) -> CheckResult:
    """Jumps, frozen frames and duplicated steps of every time-dependent field, plus the
    ``times`` axis.

    * jump (``warn``, ``fail`` above ``JUMP_FAIL_RATIO``): a step change that is an outlier
      against the median of its ``+-JUMP_WINDOW`` neighbours.
    * frozen frame: a step equal to the previous one (relative change ``< FROZEN_REL``).
      ``fail`` in the interior of a trajectory (a duplicated or stalled frame), ``warn`` for a
      run that touches the start or the end (initial hold, converged tail) or a field that
      never changes.
    * duplicated step (``fail``; ``warn`` for constant frames): a frame bit-identical to an
      earlier, non-adjacent one.
    """
    if not scans:
        return CheckResult("temporal_continuity", "skip", {"reason": "no time-dependent field"})
    statuses: list[Status] = []
    fields: dict[str, Any] = {}
    for s in scans:
        n = int(s.steps.size)
        d: dict[str, Any] = {"n_steps": n}
        if n < 2:
            fields[s.name] = {**d, "note": "single step"}
            continue
        r = s.rel_change
        frozen = (s.dnorm == 0) | (r < FROZEN_REL)
        fidx = np.flatnonzero(frozen)
        st: Status = "pass"
        d["median_rel_change"] = float(np.median(r))
        d["max_rel_change"] = float(np.max(r))
        d["n_frozen"] = int(fidx.size)
        if fidx.size == n - 1:
            d["frozen_kind"] = "constant"
            st = "warn"
        elif fidx.size:
            interior = [
                (a, b) for a, b in _runs(fidx) if a > 0 and b < n - 2
            ]  # diff index b is the pair (b, b+1); the last pair is n-2
            d["frozen_kind"] = "interior" if interior else "start-or-end"
            d["frozen_steps"] = [int(s.steps[i + 1]) for i in fidx[:20]]
            st = "fail" if interior else "warn"
        jumps = _find_jumps(r, frozen)
        d["n_jumps"] = len(jumps)
        if jumps:
            d["jumps"] = [
                {
                    "step": int(s.steps[j["index"] + 1]),
                    "rel_change": float(r[j["index"]]),
                    "ratio_to_neighbours": j["ratio"],
                    "z": j["z"],
                }
                for j in sorted(jumps, key=lambda j: -j["ratio"])[:20]
            ]
            top = max(j["ratio"] for j in jumps)
            hard = top >= JUMP_FAIL_RATIO and not s.measured
            st = worst_status([st, "fail" if hard else "warn"])
        # bit-identical frames that are not neighbours
        seen: dict[str, list[int]] = {}
        for j, h in enumerate(s.hashes):
            seen.setdefault(h, []).append(j)
        dups: list[dict[str, Any]] = []
        trivial = s.trivial
        for members in seen.values():
            runs = _runs(np.asarray(members))
            for a, _ in runs[1:]:
                dups.append(
                    {
                        "step": int(s.steps[a]),
                        "same_as": int(s.steps[runs[0][0]]),
                        "constant_frame": bool(trivial[a]),
                    }
                )
        d["n_duplicated"] = len(dups)
        if dups:
            dups.sort(key=lambda v: v["step"])
            d["duplicated_steps"] = dups[:20]
            real = any(not v["constant_frame"] for v in dups) and not s.measured
            st = worst_status([st, "fail" if real else "warn"])
        statuses.append(st)
        d["status"] = st
        fields[s.name] = d
    tinfo = check_times(times, {s.name: int(s.steps[-1]) + 1 for s in scans if s.steps.size})
    if tinfo["status"] != "skip":
        statuses.append(tinfo["status"])
    return CheckResult(
        "temporal_continuity",
        worst_status(statuses),
        {"fields": fields, "times": tinfo, "stride": stride},
    )


def _median_dt(times: np.ndarray | None) -> float | None:
    if times is None or len(times) < 2:
        return None
    d = np.diff(np.asarray(times, dtype=np.float64))
    dt = float(np.median(d))
    return dt if math.isfinite(dt) and dt > 0 else None


def check_cfl(
    sample: Sample | SampleInfo, scans: Sequence[FieldScan], times: np.ndarray | None = None,
    *, statistics_workspace_bytes: int = 32 << 20,
) -> CheckResult:
    """Saved-step CFL number ``|u| dt / dx``: ``dt`` is the time between two *stored* frames
    (not the solver step), ``dx`` the characteristic spacing. Large values mean the flow moves
    many cells between frames (``warn`` above ``CFL_WARN`` on the p99); values below
    ``CFL_TINY`` mean the data is oversampled in time (``pass`` with a recommended stride)."""
    name = "cfl"
    dt = _median_dt(times)
    if dt is None:
        return CheckResult(name, "skip", {"reason": "no time step (steady sample or one step)"})
    info = inspect_sample(sample)
    sample = info.sample
    vel = info.velocity
    if vel is None:
        return CheckResult(name, "skip", {"reason": "no velocity-like field"})
    spacing = characteristic_spacing(sample)
    if spacing is None:
        return CheckResult(name, "skip", {"reason": "no spacing / connectivity"})
    dx, how = spacing
    source, vname = vel
    statistics: dict[str, Any] = {}
    if source == "field":
        scan = next((s for s in scans if s.name == vname), None)
        if scan is None or scan.speed_max is None:
            return CheckResult(name, "skip", {"reason": f"field {vname} was not scanned"})
        u_max = float(np.nanmax(scan.speed_max))
        speed = np.sqrt((scan.probes.astype(np.float64) ** 2).sum(axis=-1))
        u_p99 = float(np.percentile(speed, 99))
    else:
        arr = sample.array(f"static/{vname}" if source == "static" else vname)
        magnitude = VectorMagnitudeBackend(sample._reader(arr), source_stats=sample.read_stats)
        summary = finite_statistics(
            ArrayReader(magnitude, buffer_bytes=8 << 20), quantiles=(0.99,),
            workspace_bytes=statistics_workspace_bytes,
        )
        if not summary.count:
            return CheckResult(name, "skip", {"reason": f"{vname} has no finite values"})
        u_max, u_p99 = summary.maximum, summary.quantiles[0]
        statistics = {
            "magnitude_algorithm": magnitude.algorithm,
            "magnitude_reduction": magnitude.reduction,
            "statistics_algorithm": summary.algorithm,
            "percentile_scope": "full_selection", "percentile_method": "linear",
            "finite_count": summary.count, "passes": summary.passes,
            "magnitude_bytes_sha256": summary.source_sha256,
            "statistics_workspace_bytes": statistics_workspace_bytes,
            "magnitude_workspace_bytes": magnitude.workspace_bytes,
        }
    cfl_max, cfl_p99 = u_max * dt / dx, u_p99 * dt / dx
    detail: dict[str, Any] = {
        "velocity_field": vname,
        "velocity_source": source,
        "dt": dt,
        "dx": dx,
        "dx_kind": how,
        "u_max": u_max,
        "u_p99": u_p99,
        "cfl_max": cfl_max,
        "cfl_p99": cfl_p99,
        "regime": "ok",
    }
    if statistics:
        detail["statistics"] = statistics
    status: Status = "pass"
    if not math.isfinite(cfl_p99):
        return CheckResult(name, "skip", {**detail, "reason": "non-finite velocity"})
    if cfl_p99 > CFL_WARN:
        status = "warn"
        detail["regime"] = "coarse"
        detail["note"] = (
            f"the flow crosses about {cfl_p99:.0f} cells between saved frames: temporal "
            "subsampling is aggressive for a next-step model"
        )
    elif 0 < cfl_max < CFL_TINY:
        stride = max(1, int(CFL_TARGET / cfl_max))
        detail["regime"] = "oversampled"
        detail["recommended_stride"] = stride
        detail["note"] = (
            f"frames are {cfl_max:.3g} cells apart: oversampled in time, a stride of about "
            f"{stride} keeps the saved-step CFL near {CFL_TARGET}"
        )
    return CheckResult(name, status, detail)


def check_stationarity(scans: Sequence[FieldScan]) -> CheckResult:
    """Is the window statistically stationary? Classifies the spatial-mean and spatial-RMS time
    series of every field (:func:`classify_stationarity`) and reports the worst class.
    ``drifting`` is a warning at most: decaying and developing flows are legitimate data."""
    if not scans:
        return CheckResult("stationarity", "skip", {"reason": "no time-dependent field"})
    fields: dict[str, Any] = {}
    classes: list[str] = []
    burn = 0
    n_eff: list[float] = []
    for s in scans:
        scale = float(np.nanmedian(s.rms)) if np.isfinite(s.rms).any() else None
        per: dict[str, Any] = {}
        for stat, x in (("mean", s.mean), ("rms", s.rms)):
            c = classify_stationarity(x, scale)
            per[stat] = c
            classes.append(c["class"])
            if c["class"] == "transient-start":
                burn = max(burn, int(c["burn_in"]) * _step_stride(s))
            if "n_eff" in c and not c.get("constant"):
                n_eff.append(float(c["n_eff"]))
        fields[s.name] = per
    cls = max(classes, key=_STATIONARITY_ORDER.index)
    n = int(max(s.steps.size for s in scans))
    detail = {
        "class": cls,
        "burn_in_steps": burn if cls == "transient-start" else 0,
        "n_eff_min": min(n_eff) if n_eff else None,
        "n_steps": n,
        "fields": fields,
        "stride": _step_stride(scans[0]),
    }
    if n < STATIONARITY_MIN_STEPS:
        return CheckResult("stationarity", "skip", detail)
    return CheckResult("stationarity", STATIONARITY_STATUS[cls], detail)


def _step_stride(scan: FieldScan) -> int:
    return int(scan.steps[1] - scan.steps[0]) if scan.steps.size > 1 else 1


# -- spectra --------------------------------------------------------------------------------


class SpectralGrid:
    """Shell bookkeeping for isotropic spectra on a regular grid of ``shape`` with physical
    side lengths ``lengths``. Shell ``k`` collects the modes with
    ``round(|kappa| L_min) == k`` (``kappa`` in cycles per unit length), up to the largest
    shell that fits inside the Nyquist box on every axis."""

    def __init__(self, shape: Sequence[int], lengths: Sequence[float], window: bool) -> None:
        self.shape = tuple(int(n) for n in shape)
        self.ndim = len(self.shape)
        self.window = window
        lmin = min(lengths)
        freqs = [np.fft.fftfreq(n, d=1.0 / n) for n in self.shape[:-1]]
        freqs.append(np.fft.rfftfreq(self.shape[-1], d=1.0 / self.shape[-1]))
        # wavenumber in units of the fundamental of the shortest side
        self.k_axes = [f * (lmin / ell) for f, ell in zip(freqs, lengths, strict=True)]
        mesh = np.meshgrid(*self.k_axes, indexing="ij")
        kk = np.sqrt(sum(m * m for m in mesh))
        self.k_max = int(
            min((n // 2) * (lmin / ell) for n, ell in zip(shape, lengths, strict=True))
        )
        self.shell = np.minimum(np.round(kk).astype(np.int64), self.k_max + 1)
        # rfftn keeps half of the last axis: count the conjugate modes
        w = np.full(self.shape[-1] // 2 + 1, 2.0)
        w[0] = 1.0
        if self.shape[-1] % 2 == 0:
            w[-1] = 1.0
        self.weight = np.broadcast_to(w, self.shell.shape)
        self.angular = [2.0 * np.pi * f / ell for f, ell in zip(freqs, lengths, strict=True)]
        self.win: np.ndarray | None = None
        self.win_norm = 1.0
        if window:
            win = np.ones(self.shape)
            for ax, n in enumerate(self.shape):
                shp = [1] * self.ndim
                shp[ax] = n
                win = win * np.hanning(n).reshape(shp)
            self.win = win
            self.win_norm = float(np.mean(win * win)) or 1.0

    def transform(self, comp: np.ndarray) -> np.ndarray:
        """FFT of one scalar component, normalised so that ``sum |f_hat|^2 = <f^2>``."""
        f = np.asarray(comp, dtype=np.float64)
        f = np.where(np.isfinite(f), f, 0.0)
        if self.win is not None:
            f = (f - f.mean()) * self.win
        fh = np.fft.rfftn(f) / f.size
        return fh / math.sqrt(self.win_norm) if self.win is not None else fh

    def power(self, values: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """``sum_c |f_hat_c|^2`` per mode of a ``(..., C)`` field, and the transforms."""
        hats = [self.transform(values[..., c]) for c in range(values.shape[-1])]
        total = np.zeros(self.shell.shape)
        for h in hats:
            total += np.abs(h) ** 2
        return total, hats

    def shells(self, power: np.ndarray) -> np.ndarray:
        """Sum ``power`` (``|f_hat|^2`` per mode) over shells ``0..k_max``."""
        out = np.bincount(
            self.shell.ravel(), weights=(power * self.weight).ravel(), minlength=self.k_max + 2
        )
        return out[: self.k_max + 1]


def isotropic_spectrum(
    values: np.ndarray, lengths: Sequence[float] | None = None, window: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Shell-summed spectrum of a grid field ``(X, Y[, Z], C)``: returns ``(k, E)`` with
    ``E[k] = 0.5 * sum_{modes in shell k} |f_hat|^2`` so that ``sum(E) = 0.5 <|f|^2>`` (for
    a velocity field the kinetic energy; exact without a window, up to the removed mean and
    the corner modes beyond the largest shell). ``window=True`` applies a Hann window (after
    removing the mean) for non-periodic data."""
    values = np.asarray(values)
    shape = values.shape[:-1]
    grid = SpectralGrid(
        shape, list(lengths) if lengths is not None else [float(n) for n in shape], window
    )
    power, _ = grid.power(values)
    return np.arange(grid.k_max + 1), 0.5 * grid.shells(power)


def _slope(k: np.ndarray, e: np.ndarray, lo: float, hi: float) -> float | None:
    sel = (k >= lo) & (k <= hi) & (e > 0) & np.isfinite(e)
    if sel.sum() < 3:
        return None
    return float(np.polyfit(np.log(k[sel]), np.log(e[sel]), 1)[0])


def spectrum_metrics(k: np.ndarray, e: np.ndarray) -> dict[str, Any]:
    """Mid-range slope, tail slope and the fraction of ``sum(E[1:])`` in the top third."""
    k_max = int(k[-1]) if k.size else 0
    total = float(e[1:].sum())
    top = k > (2.0 / 3.0) * k_max
    out: dict[str, Any] = {
        "total": total,
        "slope_mid": _slope(k, e, max(2.0, k_max / 8.0), k_max / 2.0),
        "slope_tail": _slope(k, e, (2.0 / 3.0) * k_max, float(k_max)),
        "high_k_fraction": float(e[top].sum() / total) if total > 0 else 0.0,
        "k_peak": int(k[1:][np.argmax(e[1:])]) if k.size > 1 and total > 0 else None,
    }
    return out


def _is_periodic(bc: dict[str, Any], axes: Sequence[str]) -> bool:
    def one(v: Any) -> bool:
        if isinstance(v, str):
            return v.lower() == "periodic"
        if isinstance(v, list | tuple) and v:
            return all(one(x) for x in v)
        return False

    return bool(axes) and all(one(bc.get(a)) for a in axes)


def check_spectra(
    sample: Sample | SampleInfo, scans: Sequence[FieldScan]
) -> tuple[CheckResult, dict[str, Any] | None]:
    """Isotropic energy spectrum of a grid sample at the first, middle and last step and
    averaged over ``SPECTRA_AVG_STEPS`` steps (velocity field, or the first field when there
    is none; a Hann window is applied unless every axis is periodic), plus the enstrophy
    spectrum from a ``vorticity`` field or derived from a 2D velocity.

    ``warn`` when the top third of the wavenumbers holds more than ``HIGH_K_WARN`` of the
    energy, or the spectrum rises towards the grid cutoff: under-resolved data or aliasing.
    Returns the check and the spectrum series for plotting."""
    name = "spectra"
    info = inspect_sample(sample)
    sample = info.sample
    if sample.kind != "grid":
        return CheckResult(name, "skip", {"reason": "spectra need a regular grid"}), None
    shape = sample.shape
    if len(shape) not in (2, 3) or min(shape) < 8:
        return CheckResult(name, "skip", {"reason": f"grid shape {list(shape)} not 2D/3D"}), None
    by_name = {s.name: s for s in scans}
    if not by_name:
        return CheckResult(name, "skip", {"reason": "no time-dependent field"}), None
    vel = info.velocity
    is_velocity = vel is not None and vel[0] == "field" and vel[1] in by_name
    target = vel[1] if vel is not None and is_velocity else next(iter(by_name))
    vort = next((n for n in by_name if "vorticity" in n.lower() and n != target), None)
    scan = by_name[target]
    if scan.step_shape[:-1] != tuple(shape):
        return CheckResult(name, "skip", {"reason": f"{target} is not on the grid"}), None

    crop = tuple(min(n, SPECTRA_MAX_SIDE) for n in shape)
    periodic = _is_periodic(sample.boundary_conditions, sample.axes) and crop == tuple(shape)
    sp = sample.spacing or [1.0] * len(shape)
    lengths = [abs(float(d)) * n or float(n) for d, n in zip(sp, crop, strict=True)]
    grid = SpectralGrid(crop, lengths, window=not periodic)
    sl = tuple(slice(0, n) for n in crop)
    k = np.arange(grid.k_max + 1)
    derive_enstrophy = vort is None and is_velocity and len(shape) == 2

    target_arr = info.arrays[target]
    vort_arr = info.arrays[vort] if vort is not None else None

    def spectra_at(step: int) -> tuple[np.ndarray, np.ndarray | None]:
        u = info.sample._reader(target_arr).read(
            (step, *sl, slice(None)), stats=info.sample.read_stats
        )
        power, hats = grid.power(u)
        energy = 0.5 * grid.shells(power)
        ens: np.ndarray | None = None
        if vort_arr is not None:
            v = info.sample._reader(vort_arr).read(
                (step, *sl, slice(None)), stats=info.sample.read_stats
            )
            ens = 0.5 * grid.shells(grid.power(v)[0])
        elif derive_enstrophy:
            kx = grid.angular[0][:, None]
            ky = grid.angular[1][None, :]
            wh = 1j * (kx * hats[1] - ky * hats[0])
            ens = 0.5 * grid.shells(np.abs(wh) ** 2)
        return energy, ens

    steps = scan.steps
    picks = {"first": int(steps[0]), "middle": int(steps[steps.size // 2]), "last": int(steps[-1])}
    avg_steps = sorted(
        {int(steps[i]) for i in downsample(steps, SPECTRA_AVG_STEPS)} | set(picks.values())
    )
    cache = {st: spectra_at(st) for st in avg_steps}
    mean_e = np.mean([cache[st][0] for st in avg_steps], axis=0)
    per_step: dict[str, Any] = {}
    for label, st in picks.items():
        e, z = cache[st]
        m = {"step": st, **spectrum_metrics(k, e)}
        if z is not None:
            zm = spectrum_metrics(k, z)
            m["enstrophy_total"] = zm["total"]
            m["enstrophy_high_k_fraction"] = zm["high_k_fraction"]
            m["enstrophy_slope_mid"] = zm["slope_mid"]
        per_step[label] = m
    mean_metrics = spectrum_metrics(k, mean_e)
    per_step["time_average"] = {"n_steps": len(avg_steps), **mean_metrics}

    worst_frac = max(per_step[lbl]["high_k_fraction"] for lbl in picks)
    pile_up = any(
        (per_step[lbl]["slope_tail"] or 0.0) > 0
        and per_step[lbl]["high_k_fraction"] >= PILEUP_MIN_FRACTION
        for lbl in picks
    )
    status: Status = "warn" if worst_frac >= HIGH_K_WARN or pile_up else "pass"
    e0, e1 = per_step["first"]["total"], per_step["last"]["total"]
    totals = [float(cache[st][0][1:].sum()) for st in avg_steps]
    detail: dict[str, Any] = {
        "field": target,
        "quantity": "kinetic energy" if is_velocity else f"0.5 <{target}^2>",
        "window": "none (periodic)" if periodic else "hann",
        "grid": list(crop),
        "cropped": crop != tuple(shape),
        "k_max": grid.k_max,
        "k_unit": 2.0 * math.pi / min(lengths),
        "steps": per_step,
        "high_k_fraction_max": worst_frac,
        "pile_up": pile_up,
        "slope_mid": mean_metrics["slope_mid"],
        "energy_evolution": {
            "steps": avg_steps,
            "total": totals,
            "last_over_first": e1 / e0 if e0 > 0 else None,
        },
        "enstrophy": (
            None
            if cache[avg_steps[0]][1] is None
            else {
                "source": vort if vort is not None else f"curl of {target}",
                "steps": avg_steps,
                "total": [float(np.asarray(cache[st][1])[1:].sum()) for st in avg_steps],
            }
        ),
    }
    if status == "warn":
        detail["note"] = (
            f"{worst_frac:.1%} of the energy sits in the top third of the resolved wavenumbers"
            + (" and the spectrum rises towards the grid cutoff" if pile_up else "")
            + ": under-resolved data or aliasing"
        )
    series: dict[str, Any] = {
        "k": [int(v) for v in k[1:]],
        "field": target,
        "quantity": detail["quantity"],
        "first": [_num(v, 5) for v in cache[picks["first"]][0][1:]],
        "last": [_num(v, 5) for v in cache[picks["last"]][0][1:]],
        "mean": [_num(v, 5) for v in mean_e[1:]],
    }
    z0, z1 = cache[picks["first"]][1], cache[picks["last"]][1]
    if z0 is not None and z1 is not None:
        series["enstrophy_first"] = [_num(v, 5) for v in z0[1:]]
        series["enstrophy_last"] = [_num(v, 5) for v in z1[1:]]
    if len(series["k"]) > SERIES_MAX_POINTS:
        idx = downsample(series["k"], SERIES_MAX_POINTS)
        for key, val in list(series.items()):
            if isinstance(val, list):
                series[key] = [val[i] for i in idx]
    return CheckResult(name, status, detail), series


# -- conservation ---------------------------------------------------------------------------


def integral_quantities(scans: Sequence[FieldScan]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Time series of the integral quantities, as ``{name: (steps, values)}``:
    ``kinetic_energy = 0.5 <|u|^2>`` for a velocity-like field, ``enstrophy = 0.5 <w^2>``
    for a field named ``*vorticity*``, ``mean_square:<f>`` for every other field and
    ``mean:<f>`` (the mass-like integral) for every single-component field."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for s in scans:
        msq = s.n_components * s.rms**2  # <|u|^2> per point
        leaf = s.name.rsplit("/", 1)[-1].lower()
        if s.speed_max is not None and "kinetic_energy" not in out:
            out["kinetic_energy"] = (s.steps, 0.5 * msq)
        elif "vorticity" in leaf and "enstrophy" not in out:
            out["enstrophy"] = (s.steps, 0.5 * msq)
        else:
            out[f"mean_square:{s.name}"] = (s.steps, msq)
        if s.n_components == 1:
            out[f"mean:{s.name}"] = (s.steps, s.mean)
    return out


def _monotonic(x: np.ndarray) -> str:
    d = np.diff(x)
    tol = 1e-12 * max(float(np.max(np.abs(x))), 1e-300)
    if np.all(np.abs(d) <= tol):
        return "constant"
    if np.all(d >= -tol):
        return "increasing"
    if np.all(d <= tol):
        return "decreasing"
    return "no"


def check_conservation(scans: Sequence[FieldScan]) -> CheckResult:
    """Drift of the integral quantities (:func:`integral_quantities`) over the trajectory.

    Informational (``pass`` with numbers: many systems are forced or dissipative) unless a
    positive quantity blows up: it exceeds ``EXPLODE_FACTOR`` times its median over time
    (``fail``; a flow that starts from rest and saturates does not trip this), a step has more
    than ``NON_FINITE_FRACTION_FAIL`` non-finite values (``fail``), or it grows by more than
    ``EXPLODE_FACTOR`` from a non-zero initial value without leaving the median behind
    (``warn``: a large but saturating growth)."""
    if not scans or max(s.steps.size for s in scans) < 2:
        return CheckResult("conservation", "skip", {"reason": "fewer than two steps"})
    rms0 = {s.name: float(s.rms[0]) if math.isfinite(float(s.rms[0])) else 0.0 for s in scans}
    statuses: list[Status] = []
    quantities: dict[str, Any] = {}
    for qname, (_, x) in integral_quantities(scans).items():
        ok = np.isfinite(x)
        if ok.sum() < 2:
            quantities[qname] = {"non_finite": True}
            continue
        v = x[ok]
        first, last = float(v[0]), float(v[-1])
        positive = not qname.startswith("mean:")
        scale = abs(first)
        if not positive:  # a mean near zero is compared with the field's own magnitude
            scale = max(scale, rms0.get(qname.split(":", 1)[1], 0.0))
        scale = scale if scale > 0 else max(float(np.max(np.abs(v))), 1e-300)
        q: dict[str, Any] = {
            "first": first,
            "last": last,
            "min": float(v.min()),
            "max": float(v.max()),
            "rel_drift": (last - first) / scale,
            "max_rel_deviation": float(np.max(np.abs(v - first))) / scale,
            "monotonic": _monotonic(v),
        }
        st: Status = "pass"
        if positive:
            pos = v[v > 0]  # a flow that starts from rest has leading zeros
            med = float(np.median(pos)) if pos.size else 0.0
            g_med = float(v.max()) / med if med > 0 else 1.0
            g_first = float(v.max()) / first if first > 0 else None
            q["growth_vs_median"] = g_med
            q["growth_vs_first"] = g_first
            if g_med > EXPLODE_FACTOR:
                st = "fail"
                q["exploded_at_step"] = None
                if int(np.argmax(v)) < max(2, int(INITIAL_SPIKE_FRACTION * v.size)):
                    # the peak is at the very start and the quantity then relaxes: a start-up
                    # artefact (e.g. the first pressure frames of an impulsively started
                    # solver), to be dropped as burn-in, not a blow-up
                    st = "warn"
                    q["initial_spike"] = True
            elif g_first is not None and g_first > EXPLODE_FACTOR:
                st = "warn"
        quantities[qname] = q
        statuses.append(st)
    blown: dict[str, Any] = {}
    for s in scans:
        frac = s.non_finite / max(s.n_values, 1)
        bad = np.flatnonzero(frac >= NON_FINITE_FRACTION_FAIL)
        if bad.size:
            blown[s.name] = {"first_step": int(s.steps[bad[0]]), "fraction": float(frac[bad[0]])}
            statuses.append("fail")
    for qname, (steps, x) in integral_quantities(scans).items():
        q = quantities.get(qname, {})
        if "exploded_at_step" in q:
            xs = np.nan_to_num(x, nan=0.0, posinf=0.0)
            med = float(np.median(xs[xs > 0])) if (xs > 0).any() else 0.0
            hit = np.flatnonzero(xs > EXPLODE_FACTOR * med)
            q["exploded_at_step"] = int(steps[hit[0]]) if hit.size else None
    detail = {
        "quantities": quantities,
        "non_finite_blow_up": blown,
        "stride": _step_stride(scans[0]),
    }
    return CheckResult("conservation", worst_status(statuses), detail)


# -- temporal subsampling -------------------------------------------------------------------


def lag_autocorrelation(probes: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    """Pooled space-time autocorrelation of the fluctuation at the given lags.

    ``probes`` is ``(S, ...)``: the field at fixed probe points for ``S`` steps. The time mean
    of every probe value is removed, then
    ``rho(s) = sum_t <u'_t, u'_{t+s}> / sqrt(sum_t |u'_t|^2 * sum_t |u'_{t+s}|^2)`` over at most
    ``EXPENSIVE_MAX_STEPS`` evenly spaced time origins ``t``. NaN when the field is constant."""
    p = np.asarray(probes, dtype=np.float64).reshape(len(probes), -1)
    p = p - p.mean(axis=0, keepdims=True)
    n = p.shape[0]
    sq = np.einsum("ij,ij->i", p, p)
    out = np.full(len(lags), np.nan)
    for j, lag in enumerate(lags):
        if lag >= n:
            continue
        origins = np.arange(0, n - lag)
        if origins.size > EXPENSIVE_MAX_STEPS:
            origins = origins[downsample(origins, EXPENSIVE_MAX_STEPS)]
        den = math.sqrt(float(sq[origins].sum()) * float(sq[origins + lag].sum()))
        if den > 0:
            out[j] = float(np.einsum("ij,ij->", p[origins], p[origins + lag])) / den
    return out


def _lags(n: int) -> list[int]:
    """Every lag up to 32, then geometrically spaced, up to half the window."""
    top = max(n // 2, 1)
    lags = set(range(1, min(32, top) + 1))
    x = 32.0
    while x < top:
        x *= 1.25
        lags.add(min(int(round(x)), top))
    return sorted(lags)


def check_temporal_subsampling(scans: Sequence[FieldScan], n_steps: int) -> CheckResult:
    """Recommend a training stride (never fails).

    Rule: the largest stride ``s`` for which the pooled space-time autocorrelation of every
    field's fluctuation (field minus its time mean, at random probe points) between frames
    ``s`` apart is still ``>= SUBSAMPLE_MIN_CORR`` (frames ``s`` apart then differ by about
    ``sqrt(2 (1 - 0.9)) = 45 %`` of the fluctuation amplitude: learnable, not redundant),
    capped so that at least ``SUBSAMPLE_MIN_STEPS`` frames remain. ``decorrelation_steps`` is
    the first lag where the autocorrelation drops below ``1/e``."""
    name = "temporal_subsampling"
    usable = [s for s in scans if s.steps.size >= 4]
    if not usable:
        return CheckResult(name, "skip", {"reason": "fewer than 4 steps"})
    rule = (
        f"largest stride with lag autocorrelation of the fluctuation >= {SUBSAMPLE_MIN_CORR} "
        f"for every field, keeping at least {SUBSAMPLE_MIN_STEPS} frames"
    )
    cap = max(1, n_steps // SUBSAMPLE_MIN_STEPS)
    per: dict[str, Any] = {}
    strides: list[int] = []
    decorr: list[float] = []
    lag1: list[float] = []
    for s in usable:
        stride0 = _step_stride(s)
        n = int(s.steps.size)
        if stride0 > 1 and s.steps[-1] - s.steps[-2] != stride0:
            n -= 1  # the appended last step is not on the scan stride
        lags = _lags(n)
        rho = lag_autocorrelation(s.probes[:n], lags)
        if not np.isfinite(rho).any():
            per[s.name] = {"constant": True}
            continue
        keep = [lg for lg, r in zip(lags, rho, strict=True) if math.isfinite(r)]
        # the autocorrelation must stay above the level for every lag up to the stride
        good = 0
        for lg, r in zip(lags, rho, strict=True):
            if not math.isfinite(r) or r < SUBSAMPLE_MIN_CORR:
                break
            good = lg
        below = [lg for lg, r in zip(lags, rho, strict=True) if r < DECORRELATION_LEVEL]
        dec = below[0] * stride0 if below else None
        # with a scan stride the lags are multiples of it; below one scan stride nothing is
        # known, so no subsampling is recommended
        stride = good * stride0 if good > 0 else 1
        per[s.name] = {
            "lag1_autocorr": float(rho[0]),
            "recommended_stride": stride,
            "decorrelation_steps": dec,
            "max_lag": keep[-1] * stride0 if keep else None,
        }
        strides.append(stride)
        lag1.append(float(rho[0]))
        if dec is not None:
            decorr.append(float(dec))
    if not strides:
        return CheckResult(
            name, "skip", {"reason": "every field is constant in time", "fields": per}
        )
    stride = int(max(1, min(min(strides), cap)))
    detail: dict[str, Any] = {
        "recommended_stride": stride,
        "decorrelation_steps": min(decorr) if decorr else None,
        "lag1_autocorr": min(lag1),
        "rule": rule,
        "effective_steps": int(math.ceil(n_steps / stride)),
        "n_steps": n_steps,
        "stride_cap": cap,
        "lag_unit_steps": _step_stride(usable[0]),
        "fields": per,
    }
    if detail["decorrelation_steps"] is None:
        detail["note"] = "the autocorrelation stays above 1/e over half the window"
    if min(lag1) < SUBSAMPLE_MIN_CORR:
        detail["note"] = (
            f"consecutive frames are already weakly correlated (lag-1 = {min(lag1):.2f}): "
            "the data is coarse in time for a next-step model, do not subsample"
        )
    return CheckResult(name, "pass", detail)


# -- the per-sample pass --------------------------------------------------------------------


@dataclass
class TransientSampleResult:
    """Everything :func:`frontier_data.quality.run_quality` needs from one sample."""

    sample_id: str
    kind: str
    is_transient: bool
    n_steps: int
    dt: float | None
    uniform_dt: bool
    fields: dict[str, dict[str, float]]
    checks: list[CheckResult]
    field_shapes: dict[str, list[int]] = field(default_factory=dict)
    transient: dict[str, Any] | None = None
    fingerprint: TrajectoryFingerprint | None = None
    group: str | None = None
    parameters: dict[str, float] = field(default_factory=dict)
    recommended_stride: int | None = None
    burn_in: int = 0
    stationarity: str | None = None


def _check(checks: Sequence[CheckResult], name: str) -> CheckResult | None:
    return next((c for c in checks if c.check == name), None)


def _plot_series(
    scans: Sequence[FieldScan], checks: Sequence[CheckResult], n_points: int
) -> dict[str, Any]:
    """The small time series a report needs to plot a sample (<= ``n_points`` points each)."""
    out: dict[str, Any] = {}
    shown = list(scans)[:4]
    if not shown or shown[0].steps.size < 2:
        return out
    ref = shown[0]
    cont = _check(checks, "temporal_continuity")
    cfields = cont.detail.get("fields", {}) if cont is not None else {}
    sc: dict[str, Any] = {"step": [], "fields": {}, "jumps": {}, "frozen": {}}
    for s in shown:
        if s.steps.size != ref.steps.size:
            continue
        idx, mx = _bucket_max(s.rel_change, n_points)
        sc["step"] = [int(v) for v in s.steps[1:][idx]]
        sc["fields"][s.name] = [_num(v, 5) for v in mx]
        d = cfields.get(s.name, {})
        if d.get("jumps"):
            sc["jumps"][s.name] = [int(j["step"]) for j in d["jumps"]]
        if d.get("frozen_steps"):
            sc["frozen"][s.name] = [int(v) for v in d["frozen_steps"]]
    out["step_change"] = sc
    idx = downsample(ref.steps, n_points)
    rm: dict[str, Any] = {"step": [int(v) for v in ref.steps[idx]], "fields": {}}
    for s in shown:
        if s.steps.size != ref.steps.size:
            continue
        x = np.where(np.isfinite(s.rms), s.rms, 0.0)
        running = np.cumsum(x) / np.arange(1, x.size + 1)
        final = running[-1]
        rm["fields"][s.name] = _series(running / final if final > 0 else running, idx)
    rm["quantity"] = "running mean of the spatial RMS / its final value"
    out["running_mean"] = rm
    quantities = {
        k: v
        for k, v in integral_quantities(scans).items()
        if k in ("kinetic_energy", "enstrophy") or k.startswith("mean_square:")
    }
    if quantities:
        steps0 = next(iter(quantities.values()))[0]
        idx = downsample(steps0, n_points)
        out["integrals"] = {
            "step": [int(v) for v in steps0[idx]],
            "quantities": {
                k: _series(v, idx) for k, (st, v) in quantities.items() if st.size == steps0.size
            },
        }
    return out


def transient_sample_pass(
    sample: Sample,
    *,
    group: str | None = None,
    parameters: dict[str, float] | None = None,
    series_points: int = SERIES_MAX_POINTS,
    max_scan_steps: int = MAX_SCAN_STEPS,
    statistics_workspace_bytes: int = 32 << 20,
) -> TransientSampleResult:
    """Run every per-sample check on a v1.1 sample of any kind (steady samples only get
    ``nan_inf_per_step`` and field statistics) and collect the series for the report."""
    info = inspect_sample(sample)
    transient = info.is_transient
    times = sample.times if transient else None
    fields: dict[str, dict[str, float]] = {}
    shapes: dict[str, list[int]] = {}
    other_bad: dict[str, int] = {}
    summaries: dict[str, dict[str, Any]] = {}
    scans: list[FieldScan] = []
    stride = 1
    signal = transient and is_signal_sample(sample)
    for fname in info.names:
        arr = info.arrays[fname]
        if info.time_dependent(fname) and arr.shape[0] > 0 and not signal:
            steps, st = scan_steps(int(arr.shape[0]), max_scan_steps)
            stride = max(stride, st)
            scan = scan_field(info, fname, steps)
            scans.append(scan)
            fields[fname] = _scan_field_stats(scan)
            shapes[fname] = [int(v) for v in arr.shape[1:]]
        else:
            fields[fname], summaries[fname] = streamed_field_stats(
                sample._reader(arr), stats=sample.read_stats,
                workspace_bytes=statistics_workspace_bytes,
            )
            shapes[fname] = [int(v) for v in arr.shape]
            other_bad[fname] = int(fields[fname]["n_nan"] + fields[fname]["n_inf"])
    for sname in info.static_names:
        arr = sample.array(f"static/{sname}")
        if arr.dtype.kind in "biuf":
            other_bad[f"static/{sname}"] = sum(
                count_nonfinite(sample._reader(arr), stats=sample.read_stats)
            )
    if "points" in sample.group and sample.kind != "grid":
        other_bad["points"] = sum(
            count_nonfinite(sample._reader(sample.group["points"]), stats=sample.read_stats)
        )

    float_fields = sorted(n for n in info.names if np.dtype(info.arrays[n].dtype).kind == "f")
    if signal:
        # the state of a signal sample is its channels; the grid fields only describe them
        steps, st = scan_steps(sample.n_steps, max_scan_steps)
        stride = max(stride, st)
        scans.insert(0, scan_signals(info, steps, workspace_bytes=statistics_workspace_bytes))
        for n in sample.signal_names:
            arr = sample.array(f"signals/{n}")
            key = f"signals/{n}"
            fields[key], summaries[key] = streamed_field_stats(
                sample._reader(arr), stats=sample.read_stats,
                workspace_bytes=statistics_workspace_bytes,
            )
            shapes[key] = [int(v) for v in arr.shape[1:]]
            other_bad[key] = int(fields[key]["n_nan"] + fields[key]["n_inf"])
        float_fields = ["signals"]
    checks: list[CheckResult] = [
        check_nan_inf_per_step(scans, other_bad, stride),
        check_predictable_fields(float_fields, scans if transient else None),
    ]
    if summaries:
        checks[0].detail["field_statistics"] = summaries
    if signal:
        checks[0].detail["signal_normalization"] = scans[0].normalization
    n_steps = int(len(times)) if times is not None else 1
    dt = _median_dt(times)
    result = TransientSampleResult(
        sample_id=sample.id,
        kind=sample.kind,
        is_transient=transient,
        n_steps=n_steps,
        dt=dt,
        uniform_dt=True,
        fields=fields,
        checks=checks,
        field_shapes=shapes,
        group=group,
        parameters=dict(parameters or {}),
    )
    if not transient:
        return result

    continuity = check_temporal_continuity(scans, times, stride)
    stationarity = check_stationarity(scans)
    spectra, spectrum_series = check_spectra(info, scans)
    subsampling = check_temporal_subsampling(scans, n_steps)
    cfl = check_cfl(info, scans, times, statistics_workspace_bytes=statistics_workspace_bytes)
    checks += [continuity, cfl, stationarity, spectra, check_conservation(scans), subsampling]
    for c in checks:
        c.detail = _clean(c.detail)

    result.uniform_dt = bool(continuity.detail.get("times", {}).get("uniform", True))
    result.recommended_stride = subsampling.detail.get("recommended_stride")
    result.stationarity = stationarity.detail.get("class")
    result.burn_in = int(stationarity.detail.get("burn_in_steps") or 0)
    # Measured signals get no content fingerprint: an operating state recurs (a turbine at
    # rated power on two different days has nearly the same channel vector), so a frame match
    # is no evidence of a shared trajectory. Their splits are kept apart in time by the writer.
    if scans and not signal:
        first = scans[0]
        r = first.rel_change
        moving = r[r >= FROZEN_REL]
        trivial = first.trivial
        result.fingerprint = TrajectoryFingerprint(
            key=f"{first.name}:{'x'.join(str(v) for v in first.step_shape)}",
            first_hash=first.hashes[0],
            last_hash=first.hashes[-1],
            first=first.leak_first,
            last=first.leak_last,
            step_change=float(np.median(moving)) / _step_stride(first) if moving.size else 0.0,
            trivial_first=bool(trivial[0]),
            trivial_last=bool(trivial[-1]),
        )
    series = _plot_series(scans, checks, series_points)
    if spectrum_series is not None:
        series["spectrum"] = spectrum_series
    result.transient = _clean(
        {
            "kind": sample.kind,
            "n_steps": n_steps,
            "dt": dt,
            "uniform_dt": result.uniform_dt,
            "t0": float(times[0]) if times is not None and len(times) else None,
            "group": group,
            "fields": [s.name for s in scans],
            "scan_stride": stride,
            "field_scan_algorithm": FIELD_SCAN_ALGORITHM,
            "stationarity": result.stationarity,
            "burn_in": result.burn_in,
            "n_eff": stationarity.detail.get("n_eff_min"),
            "recommended_stride": result.recommended_stride,
            "decorrelation_steps": subsampling.detail.get("decorrelation_steps"),
            "lag1_autocorr": subsampling.detail.get("lag1_autocorr"),
            "cfl_max": cfl.detail.get("cfl_max"),
            "cfl_p99": cfl.detail.get("cfl_p99"),
            "high_k_fraction": spectra.detail.get("high_k_fraction_max"),
            "spectral_slope": spectra.detail.get("slope_mid"),
            "series": series,
        }
    )
    return result


# -- dataset level --------------------------------------------------------------------------


def _round_sig(v: float, digits: int = 6) -> float:
    return float(f"{v:.{digits}g}")


def check_temporal_consistency(results: Sequence[TransientSampleResult]) -> CheckResult:
    """Do the transient samples share ``dt``, the number of steps, the fields and their
    shapes? ``warn`` on mixed ``dt`` (a model with a fixed lead time needs lead-time
    conditioning or resampling), non-uniform time axes or mixed field sets; a varying number
    of steps or varying shapes (different meshes) is only reported."""
    name = "temporal_consistency"
    tr = [r for r in results if r.is_transient]
    if not tr:
        return CheckResult(name, "skip", {"reason": "no transient samples"})
    dts = Counter("none" if r.dt is None else repr(_round_sig(r.dt, 4)) for r in tr)
    steps = Counter(str(r.n_steps) for r in tr)
    field_sets = Counter(",".join(sorted(r.field_shapes)) for r in tr)
    shapes = Counter(
        ";".join(f"{k}:{'x'.join(map(str, v))}" for k, v in sorted(r.field_shapes.items()))
        for r in tr
    )
    non_uniform = [r.sample_id for r in tr if not r.uniform_dt]

    def top(c: Counter[str], n: int = 12) -> dict[str, int]:
        return dict(c.most_common(n))

    detail: dict[str, Any] = {
        "n_transient_samples": len(tr),
        "dt": top(dts),
        "n_distinct_dt": len(dts),
        "n_steps": top(steps),
        "n_distinct_n_steps": len(steps),
        "field_sets": top(field_sets),
        "n_distinct_shapes": len(shapes),
        "non_uniform_dt_samples": non_uniform[:20],
        "n_non_uniform_dt": len(non_uniform),
        "mixed_dt": len(dts) > 1,
        "mixed_n_steps": len(steps) > 1,
        "mixed_fields": len(field_sets) > 1,
        "mixed_shapes": len(shapes) > 1,
    }
    notes: list[str] = []
    if len(dts) > 1:
        notes.append(
            "samples have different time steps: a fixed-lead-time model needs lead-time "
            "conditioning or resampling to a common dt"
        )
    if non_uniform:
        notes.append(f"{len(non_uniform)} sample(s) have a non-uniform time axis")
    if len(field_sets) > 1:
        notes.append("samples do not all store the same fields")
    detail["notes"] = notes
    return CheckResult(name, "warn" if notes else "pass", detail)


def summarize_transient(
    results: Sequence[TransientSampleResult],
    dataset_checks: Sequence[CheckResult] = (),
    time_units: str = "s",
) -> dict[str, Any] | None:
    """The ``report["transient"]`` object: trajectory counts, steps, ``dt``, the dataset-wide
    recommended stride (median of the per-sample strides) and burn-in (median of the per-sample
    MSER truncation, 0 for samples without a start-up transient), the number of training pairs
    at that stride and plain-language recommendations. ``None`` without transient samples."""
    tr = [r for r in results if r.is_transient]
    if not tr:
        return None
    groups = {r.group if r.group is not None else f"\0{r.sample_id}" for r in tr}
    n_steps = np.array([r.n_steps for r in tr])
    dts = sorted({_round_sig(r.dt, 4) for r in tr if r.dt is not None})
    strides = [r.recommended_stride for r in tr if r.recommended_stride]
    stride = int(max(1, round(float(np.median(strides))))) if strides else 1
    burns = np.array([r.burn_in for r in tr])
    burn = int(np.median(burns))
    usable = np.maximum(n_steps - burn, 0)
    pairs = int(np.maximum(usable - stride, 0).sum())
    pairs_disjoint = int((np.maximum(usable - 1, 0) // stride).sum())
    classes = Counter(r.stationarity or "unknown" for r in tr)

    def count(check: str, statuses: tuple[str, ...]) -> int:
        n = 0
        for r in tr:
            c = _check(r.checks, check)
            n += bool(c is not None and c.status in statuses)
        return n

    def regime(name: str) -> int:
        n = 0
        for r in tr:
            c = _check(r.checks, "cfl")
            n += bool(c is not None and c.detail.get("regime") == name)
        return n

    rec: list[str] = []
    n = len(tr)
    bad = count("nan_inf_per_step", ("fail",))
    if bad:
        rec.append(f"{bad} of {n} trajectories contain NaN/inf values: drop or repair them.")
    cont_fail, cont_warn = (
        count("temporal_continuity", ("fail",)),
        count("temporal_continuity", ("warn",)),
    )
    if cont_fail:
        rec.append(
            f"{cont_fail} of {n} trajectories have jumps, duplicated or frozen frames: inspect "
            "them before training (see the per-sample temporal_continuity detail)."
        )
    if cont_warn:
        rec.append(
            f"{cont_warn} of {n} trajectories have continuity warnings (non-uniform times, "
            "frozen start/end, mild jumps)."
        )
    leak = _check(dataset_checks, "trajectory_split_leakage")
    if leak is not None and leak.status == "fail":
        rec.append(
            "Trajectories straddle the train/val/test splits: re-split with "
            "frontier_data.splits.group_split so a trajectory stays in one split."
        )
    elif leak is not None and leak.status == "warn":
        rec.append(
            "Samples in different splits share an initial condition (with different "
            "parameters): fine for a parameter sweep, a leak otherwise."
        )
    cons = _check(dataset_checks, "temporal_consistency")
    if cons is not None and cons.detail.get("mixed_dt"):
        rec.append(
            f"Time steps differ across samples ({len(dts)} distinct dt): condition the model "
            "on the lead time or resample to a common dt."
        )
    if stride > 1:
        rec.append(
            f"Consecutive frames are highly correlated: train with a stride of {stride} "
            f"({pairs} input/target pairs remain)."
        )
    else:
        rec.append(f"Use every saved step (stride 1): {pairs} input/target pairs are available.")
    if burn > 0:
        rec.append(f"Drop the first {burn} steps of each trajectory (start-up transient).")
    elif classes.get("transient-start"):
        rec.append(
            f"{classes['transient-start']} of {n} trajectories have a start-up transient; see "
            "their per-sample burn_in."
        )
    if classes.get("drifting"):
        rec.append(
            f"{classes['drifting']} of {n} trajectories are not statistically stationary "
            "(decaying or developing): fine for forecasting, but time-averaged statistics "
            "are not converged."
        )
    if classes.get("too-short"):
        rec.append(
            f"{classes['too-short']} of {n} trajectories hold fewer than {N_EFF_MIN:.0f} "
            "independent samples: too short to judge stationarity or to average over."
        )
    if regime("coarse"):
        rec.append(
            f"{regime('coarse')} of {n} trajectories have a saved-step CFL above {CFL_WARN:.0f}: "
            "frames are far apart relative to advection, expect a hard next-step problem."
        )
    if regime("oversampled"):
        rec.append(
            f"{regime('oversampled')} of {n} trajectories have a saved-step CFL below "
            f"{CFL_TINY}: oversampled in time, a larger stride loses little."
        )
    spec = count("spectra", ("warn",))
    if spec:
        rec.append(
            f"{spec} of {n} trajectories pile energy up at the grid cutoff: under-resolved or "
            "aliased, expect the model to learn grid-scale noise."
        )
    blow = count("conservation", ("fail",))
    if blow:
        rec.append(f"{blow} of {n} trajectories blow up (energy grows by more than 100x).")

    cfl_max = [
        v for v in ((r.transient or {}).get("cfl_max") for r in tr) if isinstance(v, int | float)
    ]
    return _clean(
        {
            "n_trajectories": len(groups),
            "n_samples": n,
            "sample_kinds": dict(Counter(r.kind for r in tr)),
            "steps": {
                "min": int(n_steps.min()),
                "median": float(np.median(n_steps)),
                "max": int(n_steps.max()),
                "total": int(n_steps.sum()),
            },
            "dt": {
                "values": dts[:20],
                "n_distinct": len(dts),
                "uniform": all(r.uniform_dt for r in tr),
                "units": time_units,
            },
            "recommended_stride": stride,
            "recommended_burn_in": burn,
            "burn_in_max": int(burns.max()),
            "training_pairs": pairs,
            "training_pairs_disjoint": pairs_disjoint,
            "training_pairs_stride_1": int(np.maximum(n_steps - 1, 0).sum()),
            "stationarity": dict(classes),
            "cfl_max_median": float(np.median(cfl_max)) if cfl_max else None,
            "recommendations": rec,
        }
    )

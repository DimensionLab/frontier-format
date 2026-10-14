# SPDX-License-Identifier: Apache-2.0
"""Self-contained HTML rendering of a quality report (tables + inline SVG histograms, and
inline SVG time-series plots for transient datasets). No JavaScript, no external assets."""

from __future__ import annotations

import html
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:24px;
color:#1b1f23;background:#fff}
h1,h2,h3{font-weight:600} table{border-collapse:collapse;font-size:13px;margin:8px 0 20px}
th,td{border:1px solid #d0d7de;padding:4px 8px;text-align:left;vertical-align:top}
th{background:#f6f8fa} .pass{color:#1a7f37} .warn{color:#9a6700} .fail{color:#cf222e}
.skip{color:#57606a}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600}
.badge.pass{background:#dafbe1}.badge.warn{background:#fff8c5}.badge.fail{background:#ffebe9}.badge.skip{background:#eaeef2}
.grid{display:flex;flex-wrap:wrap;gap:16px} figure{margin:0}
figcaption{font-size:12px;color:#57606a}
code{font-size:12px;background:#f6f8fa;padding:1px 4px;border-radius:3px} details{margin:4px 0}
ul.rec{font-size:13px;margin:4px 0 16px;padding-left:20px} ul.rec li{margin:2px 0}
"""

#: line colours of :func:`svg_timeseries` (the first one is the histogram blue)
SERIES_COLORS = ("#0969da", "#bc4c00", "#1a7f37", "#8250df", "#bf3989", "#57606a")
SPECTRUM_PLOT_FLOOR = 1e-12  # spectra are plotted down to this fraction of their maximum
MAX_TRANSIENT_SAMPLES = 6  # samples plotted in the "Transient" section (worst status first)


def _badge(status: str) -> str:
    return f'<span class="badge {status}">{status}</span>'


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.4g}"
    return html.escape(str(v))


def svg_histogram(
    values: list[float], title: str, bins: int = 12, width: int = 260, height: int = 120
) -> str:
    vals = np.array([v for v in values if v is not None and math.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return f"<figure><figcaption>{html.escape(title)}: no data</figcaption></figure>"
    hist, edges = np.histogram(vals, bins=bins)
    top = max(int(hist.max()), 1)
    pad_l, pad_b = 6, 18
    bw = (width - 2 * pad_l) / bins
    bars = []
    for i, h in enumerate(hist):
        bh = (height - pad_b - 4) * h / top
        x = pad_l + i * bw
        y = height - pad_b - bh
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw - 1:.1f}" height="{bh:.1f}" fill="#0969da">'
            f"<title>[{edges[i]:.4g}, {edges[i + 1]:.4g}): {int(h)}</title></rect>"
        )
    axis = (
        f'<text x="{pad_l}" y="{height - 4}" font-size="10" fill="#57606a">{edges[0]:.3g}</text>'
        f'<text x="{width - pad_l}" y="{height - 4}" font-size="10" fill="#57606a" '
        f'text-anchor="end">{edges[-1]:.3g}</text>'
    )
    return (
        f'<figure><svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">{"".join(bars)}{axis}</svg>'
        f"<figcaption>{html.escape(title)} (n={vals.size})</figcaption></figure>"
    )


def _nice_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    """About ``n`` round tick values covering ``[lo, hi]``."""
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo]
    raw = (hi - lo) / max(n, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1.0, 2.0, 5.0, 10.0)), key=lambda c: abs(math.log(c / raw)))
    first = math.ceil(lo / step - 1e-9) * step
    ticks = []
    v = first
    while v <= hi + 1e-9 * step:
        ticks.append(0.0 if abs(v) < 1e-12 * step else v)
        v += step
    return ticks or [lo]


def _log_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Decade ticks covering ``[lo, hi]`` (both > 0), thinned to about ``n``."""
    a, b = math.ceil(math.log10(lo) - 1e-9), math.floor(math.log10(hi) + 1e-9)
    if b - a < 1:  # less than two decades in range: 1-2-5 ticks
        lo_e, hi_e = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
        ticks = [m * 10.0**e for e in range(lo_e, hi_e + 1) for m in (1.0, 2.0, 5.0)]
        ticks = [t for t in ticks if lo <= t <= hi]
        return ticks or [lo, hi]
    every = max(1, math.ceil((b - a + 1) / n))
    return [10.0**e for e in range(a, b + 1, every)]


def _tick_label(v: float, log: bool) -> str:
    if log:
        e = math.log10(v)
        return f"1e{round(e)}" if abs(e - round(e)) < 1e-6 else f"{v:.1g}"
    return f"{v:.3g}"


def svg_timeseries(
    series: Mapping[str, Sequence[float | None]],
    title: str,
    xlabel: str = "step",
    ylabel: str = "",
    log_y: bool = False,
    width: int = 360,
    height: int = 200,
    *,
    x: Sequence[float] | Mapping[str, Sequence[float]] | None = None,
    log_x: bool = False,
    markers: Mapping[str, Sequence[float]] | None = None,
) -> str:
    """Inline SVG line plot (no JavaScript) of one or more series, as a ``<figure>``.

    ``series`` maps a legend label to its y values; ``None`` / non-finite values (and values
    ``<= 0`` on a log axis) break the line. ``x`` gives the abscissa: one sequence shared by
    every series, a mapping per label, or nothing for the index ``0..n-1``. ``markers`` maps a
    legend label to x positions drawn as dashed vertical lines (e.g. detected jumps).
    Axes carry tick labels, the legend sits above the plot.
    """
    palette = SERIES_COLORS
    marker_colors = ("#cf222e", "#9a6700", "#8250df")

    def usable(v: Any, log: bool) -> bool:
        return isinstance(v, int | float) and math.isfinite(v) and (not log or v > 0)

    lines: list[tuple[str, list[tuple[float, float] | None]]] = []
    for label, ys in series.items():
        xs_any = x.get(label) if isinstance(x, Mapping) else x
        xs = list(xs_any) if xs_any is not None else list(range(len(ys)))
        pts: list[tuple[float, float] | None] = []
        for xv, yv in zip(xs, ys, strict=False):
            if usable(xv, log_x) and usable(yv, log_y):
                pts.append((float(xv), float(yv)))  # type: ignore[arg-type]
            else:
                pts.append(None)
        lines.append((str(label), pts))
    flat = [p for _, pts in lines for p in pts if p is not None]
    cap = f"<figcaption>{html.escape(title)}</figcaption>"
    if not flat:
        return f"<figure><figcaption>{html.escape(title)}: no data</figcaption></figure>"

    def fwd(v: float, log: bool) -> float:
        return math.log10(v) if log else v

    x_lo, x_hi = min(p[0] for p in flat), max(p[0] for p in flat)
    y_lo, y_hi = min(p[1] for p in flat), max(p[1] for p in flat)
    if x_hi <= x_lo:
        x_lo, x_hi = (x_lo / 2, x_hi * 2) if log_x else (x_lo - 0.5, x_hi + 0.5)
    if y_hi <= y_lo:
        pad = abs(y_lo) * 0.1 or 1.0
        y_lo, y_hi = (y_lo / 2, y_hi * 2) if log_y else (y_lo - pad, y_hi + pad)
    elif not log_y:
        pad = 0.05 * (y_hi - y_lo)
        y_lo, y_hi = y_lo - pad, y_hi + pad

    n_legend = len(lines) + len(markers or {})
    legend_rows = max(1, math.ceil(n_legend / 3))
    pad_l, pad_r, pad_t, pad_b = 46, 10, 6 + 13 * legend_rows, 30
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    fx_lo, fx_hi = fwd(x_lo, log_x), fwd(x_hi, log_x)
    fy_lo, fy_hi = fwd(y_lo, log_y), fwd(y_hi, log_y)

    def px(v: float) -> float:
        return pad_l + pw * (fwd(v, log_x) - fx_lo) / (fx_hi - fx_lo)

    def py(v: float) -> float:
        return pad_t + ph * (1.0 - (fwd(v, log_y) - fy_lo) / (fy_hi - fy_lo))

    grey = 'font-size="10" fill="#57606a"'
    out = [
        f'<rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" fill="none" stroke="#d0d7de"/>'
    ]
    for tv in _log_ticks(x_lo, x_hi) if log_x else _nice_ticks(x_lo, x_hi, 5):
        if not x_lo <= tv <= x_hi:
            continue
        out.append(
            f'<line x1="{px(tv):.1f}" y1="{pad_t + ph}" x2="{px(tv):.1f}" y2="{pad_t + ph + 3}" '
            f'stroke="#57606a"/><text x="{px(tv):.1f}" y="{pad_t + ph + 13}" {grey} '
            f'text-anchor="middle">{_tick_label(tv, log_x)}</text>'
        )
    for tv in _log_ticks(y_lo, y_hi) if log_y else _nice_ticks(y_lo, y_hi, 4):
        if not y_lo <= tv <= y_hi:
            continue
        out.append(
            f'<line x1="{pad_l}" y1="{py(tv):.1f}" x2="{pad_l + pw}" y2="{py(tv):.1f}" '
            f'stroke="#eaeef2"/><text x="{pad_l - 4}" y="{py(tv) + 3:.1f}" {grey} '
            f'text-anchor="end">{_tick_label(tv, log_y)}</text>'
        )
    out.append(
        f'<text x="{pad_l + pw / 2:.1f}" y="{height - 4}" {grey} text-anchor="middle">'
        f"{html.escape(xlabel)}</text>"
    )
    if ylabel:
        out.append(
            f'<text x="10" y="{pad_t + ph / 2:.1f}" {grey} text-anchor="middle" '
            f'transform="rotate(-90 10 {pad_t + ph / 2:.1f})">{html.escape(ylabel)}</text>'
        )
    legend: list[tuple[str, str, bool]] = []
    for k, (label, values) in enumerate((markers or {}).items()):
        color = marker_colors[k % len(marker_colors)]
        legend.append((label, color, True))
        for xv in values:
            if usable(xv, log_x) and x_lo <= xv <= x_hi:
                out.append(
                    f'<line x1="{px(xv):.1f}" y1="{pad_t}" x2="{px(xv):.1f}" y2="{pad_t + ph}" '
                    f'stroke="{color}" stroke-dasharray="3 2"><title>{html.escape(label)}: '
                    f"{xv:g}</title></line>"
                )
    for k, (label, pts) in enumerate(lines):
        color = palette[k % len(palette)]
        legend.insert(k, (label, color, False))
        segs: list[list[tuple[float, float]]] = [[]]
        for p in pts:
            if p is None:
                segs.append([])
            else:
                segs[-1].append(p)
        for seg in segs:
            if len(seg) == 1:
                out.append(
                    f'<circle cx="{px(seg[0][0]):.1f}" cy="{py(seg[0][1]):.1f}" r="1.5" '
                    f'fill="{color}"/>'
                )
            elif seg:
                path = " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in seg)
                out.append(
                    f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="1.3"/>'
                )
    col_w = pw / 3
    for k, (label, color, dashed) in enumerate(legend):
        lx = pad_l + (k % 3) * col_w
        ly = 10 + 13 * (k // 3)
        dash = ' stroke-dasharray="3 2"' if dashed else ""
        text = label if len(label) <= 18 else label[:17] + "…"
        out.append(
            f'<line x1="{lx:.1f}" y1="{ly - 3}" x2="{lx + 14:.1f}" y2="{ly - 3}" stroke="{color}" '
            f'stroke-width="1.5"{dash}/><text x="{lx + 18:.1f}" y="{ly}" font-size="10" '
            f'fill="#1b1f23">{html.escape(text)}</text>'
        )
    return (
        f'<figure><svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">{"".join(out)}</svg>{cap}</figure>'
    )


def _opt(v: Any) -> str:
    return "n/a" if v is None else _fmt(v)


def _normalised(values: Sequence[float | None]) -> list[float | None]:
    """Divide by the first non-zero value (by the maximum when there is none) for plotting."""
    finite = [v for v in values if v is not None and math.isfinite(v)]
    ref = next((v for v in finite if v != 0), None)
    if ref is None:
        return list(values)
    return [None if v is None else v / ref for v in values]


def _transient_figures(sample: dict[str, Any]) -> list[str]:
    """The time-series plots of one sample from its ``transient["series"]`` block."""
    series = sample["transient"].get("series", {})
    figs: list[str] = []
    sc = series.get("step_change")
    if sc and sc.get("fields"):
        marks: dict[str, list[float]] = {}
        jumps = sorted({v for vs in sc.get("jumps", {}).values() for v in vs})
        frozen = sorted({v for vs in sc.get("frozen", {}).values() for v in vs})
        if jumps:
            marks["jump"] = jumps
        if frozen:
            marks["frozen frame"] = frozen
        figs.append(
            svg_timeseries(
                sc["fields"],
                "relative step change |u(t+1) - u(t)| / |u(t)|",
                "step",
                "rel. change",
                log_y=True,
                x=sc["step"],
                markers=marks,
            )
        )
    rm = series.get("running_mean")
    if rm and rm.get("fields"):
        figs.append(
            svg_timeseries(
                rm["fields"],
                rm.get("quantity", "running mean"),
                "step",
                "running mean",
                x=rm["step"],
            )
        )
    integ = series.get("integrals")
    if integ and integ.get("quantities"):
        figs.append(
            svg_timeseries(
                {k: _normalised(v) for k, v in integ["quantities"].items()},
                "integral quantities / initial value",
                "step",
                "q / q0",
                x=integ["step"],
            )
        )
    spec = series.get("spectrum")
    if spec and spec.get("k"):
        curves = {"first step": spec["first"], "last step": spec["last"]}
        if "mean" in spec:
            curves["time average"] = spec["mean"]
        # round-off level shells (dealiased modes) would stretch the log axis
        top = max((v for c in curves.values() for v in c if v is not None), default=0.0)
        curves = {
            k: [v if v is not None and v > top * SPECTRUM_PLOT_FLOOR else None for v in c]
            for k, c in curves.items()
        }
        figs.append(
            svg_timeseries(
                curves,
                f"isotropic spectrum of {spec.get('field', '')} ({spec.get('quantity', '')})",
                "wavenumber k",
                "E(k)",
                log_y=True,
                x=spec["k"],
                log_x=True,
            )
        )
    return figs


def render_transient_section(report: dict[str, Any]) -> str:
    """The "Transient" section: dataset-level temporal summary, recommendations and time-series
    plots for up to ``MAX_TRANSIENT_SAMPLES`` samples (worst status first)."""
    tr = report.get("transient")
    if not tr:
        return ""
    steps, dt = tr.get("steps", {}), tr.get("dt", {})
    dt_txt = ", ".join(_opt(float(v)) for v in dt.get("values", [])) or "n/a"
    if dt.get("n_distinct", 0) > len(dt.get("values", [])):
        dt_txt += ", …"
    rows = [
        ["trajectories", f"{tr.get('n_trajectories')} ({tr.get('n_samples')} samples)"],
        [
            "steps per sample (min / median / max)",
            f"{steps.get('min')} / {_opt(steps.get('median'))} / {steps.get('max')}",
        ],
        [
            f"dt [{html.escape(str(dt.get('units', 's')))}]",
            dt_txt + ("" if dt.get("uniform", True) else " (non-uniform time axes present)"),
        ],
        ["recommended stride", str(tr.get("recommended_stride"))],
        ["recommended burn-in (steps to drop)", str(tr.get("recommended_burn_in"))],
        [
            "training pairs at that stride",
            f"{tr.get('training_pairs')} (stride 1: {tr.get('training_pairs_stride_1')})",
        ],
        [
            "stationarity",
            ", ".join(f"{html.escape(k)}: {v}" for k, v in tr.get("stationarity", {}).items()),
        ],
        ["median of max saved-step CFL", _opt(tr.get("cfl_max_median"))],
    ]
    parts = ["<h2>Transient</h2>", _table(["quantity", "value"], rows)]
    recs = tr.get("recommendations") or []
    if recs:
        parts.append(
            "<h3>Recommendations</h3><ul class='rec'>"
            + "".join(f"<li>{html.escape(r)}</li>" for r in recs)
            + "</ul>"
        )
    order = {"fail": 0, "warn": 1, "pass": 2}
    candidates = [r for r in report["samples"] if r.get("transient")]
    candidates.sort(key=lambda r: order.get(r["status"], 3))  # stable: keeps the dataset order
    for r in candidates[:MAX_TRANSIENT_SAMPLES]:
        t = r["transient"]
        flagged = ", ".join(
            f"<span class='{c['status']}'>{html.escape(c['check'])}</span>"
            for c in r["checks"]
            if c["status"] in ("warn", "fail")
        )
        facts = (
            f"{html.escape(str(t.get('kind')))} &middot; {t.get('n_steps')} steps &middot; "
            f"dt {_opt(t.get('dt'))} &middot; {html.escape(str(t.get('stationarity')))} &middot; "
            f"stride {t.get('recommended_stride')} &middot; burn-in {t.get('burn_in')}"
            f" &middot; CFL max {_opt(t.get('cfl_max'))}"
        )
        parts.append(
            f"<h3>{html.escape(r['id'])} {_badge(r['status'])}</h3><p>{facts}"
            + (f" &middot; flagged: {flagged}" if flagged else "")
            + "</p><div class='grid'>"
            + "".join(_transient_figures(r))
            + "</div>"
        )
    if len(candidates) > MAX_TRANSIENT_SAMPLES:
        parts.append(
            f"<p>{len(candidates) - MAX_TRANSIENT_SAMPLES} more transient samples are not "
            "plotted; their series are in report.json.</p>"
        )
    return "".join(parts)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_html(report: dict[str, Any]) -> str:
    s = report["summary"]
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Quality report: {html.escape(report['dataset'])}</title>"
        f"<style>{_CSS}</style></head><body>",
        f"<h1>Quality report: {html.escape(report['dataset'])} {_badge(s['status'])}</h1>",
        f"<p>{s['n_samples']} samples &middot; pass {s['pass']} &middot; warn {s['warn']}"
        f" &middot; fail {s['fail']} &middot; generated {html.escape(report['generated_at'])}"
        f" in {report['elapsed_s']:.1f}s</p>",
    ]
    # dataset checks
    rows = []
    for c in report["dataset_checks"]:
        d = dict(c["detail"])
        if c["check"] == "parameter_coverage" and "parameters" in d:
            d = {k: v for k, v in d.items() if k != "parameters"}
        rows.append(
            [
                html.escape(c["check"]),
                _badge(c["status"]),
                f"<code>{html.escape(_fmt_detail(d))}</code>",
            ]
        )
    parts.append("<h2>Dataset checks</h2>" + _table(["check", "status", "detail"], rows))
    parts.append(render_transient_section(report))

    # histograms
    samples = report["samples"]
    figs = [
        svg_histogram([r.get("cd_reference") for r in samples], "Cd (solver CSV)"),
        svg_histogram([r.get("cd_integrated") for r in samples], "Cd (integrated from surface)"),
        svg_histogram([r["n_faces"] for r in samples], "faces per sample"),
        svg_histogram([r["area"] for r in samples], "surface area"),
    ]
    for name, st in report["field_statistics"].items():
        figs.append(
            svg_histogram(
                [r["fields"].get(name, {}).get("mean") for r in samples], f"{name}: per-sample mean"
            )
        )
        _ = st
    cov = next((c for c in report["dataset_checks"] if c["check"] == "parameter_coverage"), None)
    if cov and "parameters" in cov["detail"]:
        for pname, pd in cov["detail"]["parameters"].items():
            hist = pd["histogram"]
            fake = []
            lo, hi = pd["min"], pd["max"]
            nb = len(hist)
            for i, h in enumerate(hist):
                fake.extend([lo + (hi - lo) * (i + 0.5) / nb] * int(h))
            figs.append(svg_histogram(fake, f"param {pname}", bins=nb))
    parts.append("<h2>Distributions</h2><div class='grid'>" + "".join(figs) + "</div>")

    # field stats
    rows = [
        [
            html.escape(n),
            _fmt(v["mean"]),
            _fmt(v["std"]),
            _fmt(v["min"]),
            _fmt(v["max"]),
            _fmt(v["p01_median"]),
            _fmt(v["p99_median"]),
        ]
        for n, v in report["field_statistics"].items()
    ]
    parts.append(
        "<h2>Field statistics</h2>"
        + _table(["field", "mean", "std", "min", "max", "p01", "p99"], rows)
    )
    norm = report["recommended_normalization"]
    parts.append(
        "<details><summary>Recommended normalization</summary><pre>"
        + html.escape(_fmt_detail(norm))
        + "</pre></details>"
    )

    # samples
    rows = []
    for r in samples:
        checks = " ".join(
            f"<span class='{c['status']}' title='{html.escape(_fmt_detail(c['detail']))}'>"
            f"{html.escape(c['check'])}:{c['status']}</span>"
            for c in r["checks"]
        )
        rows.append(
            [
                html.escape(r["id"]),
                _badge(r["status"]),
                str(r["n_points"]),
                str(r["n_faces"]),
                _fmt(r["area"]),
                _fmt(r.get("cd_reference")),
                _fmt(r.get("cd_integrated")),
                checks,
            ]
        )
    parts.append(
        "<h2>Samples</h2>"
        + _table(["id", "status", "points", "faces", "area", "Cd csv", "Cd integ.", "checks"], rows)
    )
    parts.append("</body></html>")
    return "".join(parts)


def _fmt_detail(d: Any, limit: int = 600) -> str:
    import json

    txt = json.dumps(d, default=str, sort_keys=True)
    if len(txt) > limit:
        txt = txt[:limit] + "…"
    return txt

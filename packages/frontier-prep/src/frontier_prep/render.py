# SPDX-License-Identifier: Apache-2.0
"""Small thumbnails for previews: SVG surface/mesh renders (painter's algorithm over a LOD),
PNG grid slices (zlib-only encoder, no imaging dependency) and SVG time series (reusing
``frontier_data.quality_report.svg_timeseries``). Bounded: at most ``MAX_TRIANGLES`` triangles,
``MAX_PIXELS`` pixels, output under ~400 kB."""

from __future__ import annotations

import base64
import struct
import zlib
from typing import Any

import numpy as np

from frontier_data.quality_report import svg_timeseries
from frontier_prep.io import SampleEditor
from frontier_prep.op import Change, Render

MAX_TRIANGLES = 6000
MAX_PIXELS = 256 * 256
W, H = 320, 240


def _colormap(v: np.ndarray) -> np.ndarray:
    """coolwarm-like map of values in [0, 1] -> (N, 3) uint8."""
    v = np.clip(np.nan_to_num(v, nan=0.5), 0, 1)
    r = np.interp(v, [0, 0.5, 1], [59, 221, 180])
    g = np.interp(v, [0, 0.5, 1], [76, 221, 4])
    b = np.interp(v, [0, 0.5, 1], [192, 221, 38])
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def _normalize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    x = values.astype(np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return np.full_like(x, 0.5), 0.0, 0.0
    lo, hi = np.percentile(x[finite], [2, 98])
    if hi <= lo:
        hi = lo + 1e-12
    return (x - lo) / (hi - lo), float(lo), float(hi)


def svg_surface(
    points: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray | None = None,
    *,
    caption: str = "",
    axis: int = 2,
    width: int = W,
    height: int = H,
) -> str:
    """Orthographic projection along ``axis`` (default: looking down z), depth-sorted."""
    pts = np.asarray(points, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(f) > MAX_TRIANGLES:
        idx = np.linspace(0, len(f) - 1, MAX_TRIANGLES).astype(np.int64)
        f = f[idx]
        if values is not None and len(values) == len(faces):
            values = np.asarray(values)[idx]
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts))])
    keep = [i for i in range(3) if i != axis]
    xy = pts[:, keep]
    depth = pts[:, axis]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    scale = min((width - 10) / span[0], (height - 10) / span[1])
    sx = 5 + (xy[:, 0] - lo[0]) * scale
    sy = height - 5 - (xy[:, 1] - lo[1]) * scale
    tri_depth = depth[f].mean(axis=1)
    order = np.argsort(tri_depth, kind="stable")
    if values is not None:
        vals = np.asarray(values, dtype=np.float64).reshape(len(values), -1)[:, 0]
        if len(vals) == len(pts):
            tri_vals = vals[f].mean(axis=1)
        else:
            tri_vals = vals
        norm, vmin, vmax = _normalize(tri_vals)
        colors = _colormap(norm)
        legend = f"{vmin:.3g} … {vmax:.3g}"
    else:
        # flat shading from the face normal
        e1 = pts[f[:, 1]] - pts[f[:, 0]]
        e2 = pts[f[:, 2]] - pts[f[:, 0]]
        n = np.cross(e1, e2)
        nn = np.linalg.norm(n, axis=1)
        nn[nn == 0] = 1
        shade = 0.35 + 0.65 * np.abs(n[:, axis] / nn)
        colors = np.stack([shade * 200] * 3, axis=1).astype(np.uint8)
        legend = ""
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" shape-rendering="crispEdges">',
        f'<rect width="{width}" height="{height}" fill="#101418"/>',
    ]
    for i in order:
        a, b, c = f[i]
        col = colors[i]
        parts.append(
            f'<polygon points="{sx[a]:.1f},{sy[a]:.1f} {sx[b]:.1f},{sy[b]:.1f} '
            f'{sx[c]:.1f},{sy[c]:.1f}" fill="rgb({col[0]},{col[1]},{col[2]})"/>'
        )
    if caption:
        parts.append(
            f'<text x="6" y="14" font-size="11" fill="#ddd" font-family="monospace">'
            f"{_esc(caption)}</text>"
        )
    if legend:
        parts.append(
            f'<text x="6" y="{height - 6}" font-size="10" fill="#bbb" font-family="monospace">'
            f"{_esc(legend)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an (H, W, 3) uint8 array as PNG."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def png_field(values: np.ndarray, *, max_pixels: int = MAX_PIXELS) -> tuple[str, int, int]:
    """A 2-D field slice -> base64 PNG (downsampled by striding to ``max_pixels``)."""
    a = np.asarray(values, dtype=np.float64)
    while a.ndim > 2:
        a = a[..., 0] if a.shape[-1] <= 4 else a[a.shape[0] // 2]
    if a.ndim == 1:
        a = a[None, :]
    stride = int(np.ceil(np.sqrt(a.size / max_pixels))) if a.size > max_pixels else 1
    a = a[::stride, ::stride]
    norm, _, _ = _normalize(a.reshape(-1))
    rgb = _colormap(norm).reshape(a.shape[0], a.shape[1], 3)
    rgb = np.ascontiguousarray(np.transpose(rgb, (1, 0, 2))[::-1])  # x right, y up
    return base64.b64encode(png_bytes(rgb)).decode(), rgb.shape[1], rgb.shape[0]


def render_sample(ed: SampleEditor, field: str | None = None, caption: str = "") -> Render | None:
    s = ed.reopen()
    if s.kind == "grid":
        name = field or (s.field_names()[0] if s.field_names() else None)
        if name is None:
            return None
        step = s.field(name, 0) if s.is_time_dependent(name) else s.field(name)
        data, w, h = png_field(step)
        return Render(
            kind="png", caption=caption or f"{ed.id} {name}", data=data, width=w, height=h
        )
    lod = 0
    levels: Any = s.attrs.get("lod_levels") or [0]
    if s.kind == "surface" and len(levels) > 1:
        lod = int(max(levels))
    pts = s.points(lod=lod)
    tri = _triangles(s, lod)
    vals = None
    if field and field in s.field_names(lod=lod):
        loc = "cell" if field in s.field_names("cell", lod=lod) else None
        v = (
            s.field(field, 0, location=loc, lod=lod)
            if s.is_time_dependent(field, loc)
            else s.field(field, location=loc, lod=lod)
        )
        vals = v
    return Render(
        kind="svg",
        caption=caption or f"{ed.id}" + (f" {field}" if field else ""),
        data=svg_surface(pts, tri, vals, caption=caption or f"{ed.id} {field or ''}"),
    )


def _triangles(s: Any, lod: int) -> np.ndarray:
    if lod == 0:
        return s.triangles()
    g: Any = s.group["lod"][str(lod)]
    return np.asarray(g["faces"][...])


def render_series(
    series: dict[str, list[float]], title: str, x: list[float] | None = None
) -> Render:
    svg = svg_timeseries({k: list(v) for k, v in series.items()}, title, x=x, width=W, height=200)
    return Render(kind="svg", caption=title, data=svg, width=W, height=200)


def default_renders(
    before: SampleEditor, after: SampleEditor, changes: list[Change]
) -> list[Render]:
    """Before/after thumbnails of the first changed field (or the geometry)."""
    field = None
    for ch in changes:
        if "/fields/" in ch.target or "/cell_fields/" in ch.target:
            field = ch.target.rsplit("/", 1)[-1]
            break
    out: list[Render] = []
    try:
        r = render_sample(
            before,
            field if field and field in before.reopen().field_names() else None,
            caption=f"{before.id} before",
        )
        if r:
            out.append(r)
        r = render_sample(
            after,
            field if field and field in after.reopen().field_names() else None,
            caption=f"{after.id} after",
        )
        if r:
            out.append(r)
    except (KeyError, FileNotFoundError, ValueError):
        pass
    return out

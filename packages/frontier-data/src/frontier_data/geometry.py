# SPDX-License-Identifier: Apache-2.0
"""Parametric geometry: a surface from a few numbers.

A **provider** turns a parameter vector into a triangulated surface. The Input node
(``frontier_data.inputs``) uses one when the user types parameters instead of pointing at a
file, and the Optimise step moves the parameters and asks for the surface again at every
evaluation. The first provider is the Ahmed body; a NURBS or CAD provider (control points
moved by the optimiser, the result going back to CAD) plugs in through the same interface:
register it under the entry-point group ``frontier_data.geometry`` or pass it to
:func:`register`.

A provider says which parameters it takes (name, unit, range, default), builds a surface
for a vector of them, and names the parameters it exposes to models as the sample's
``geometry_params``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from frontier_data.mesh import SurfaceMesh

ENTRY_POINT_GROUP = "frontier_data.geometry"


@dataclass(frozen=True)
class ParameterSpec:
    """One parameter a provider takes."""

    name: str
    default: float
    min: float
    max: float
    units: str = ""
    description: str = ""

    def clamp(self, value: float) -> float:
        return float(min(self.max, max(self.min, value)))

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "units": self.units,
            "description": self.description,
        }


@runtime_checkable
class GeometryProvider(Protocol):
    """The interface every provider implements."""

    id: str
    title: str
    #: what the surface's lengths are in
    length_units: str

    def parameters(self) -> list[ParameterSpec]: ...

    def surface(self, params: dict[str, float], *, resolution: int = 24) -> SurfaceMesh:
        """The surface for ``params`` (missing ones take their default; values outside the
        range are clamped). ``resolution`` is the number of segments along the longest
        edge; the provider decides what that means for its shape."""
        ...

    def conditions(self, params: dict[str, float]) -> dict[str, float]:
        """The parameters as models trained on public datasets of this shape name them,
        derived from ``params`` (the Ahmed body publishes AhmedML's eight, in millimetres),
        so a case made here conditions such a model without a mapping by hand."""
        ...


@dataclass
class AhmedBody:
    """The Ahmed reference body: a box with a rounded nose and a slanted rear, standing on
    stilts. Parameters: the slant angle of the rear upper surface, the taper of the rear
    sides, the body height, the length, the width and the ground clearance. Metres.

    The body is the standard shape of automotive aerodynamics (Ahmed, Ramm and Faltin,
    1984): 1.044 m long, 0.389 m wide, 0.288 m high, rear slant 0.222 m long. Its drag
    depends strongly on the slant angle, which is what makes it the example for shape
    optimisation."""

    id: str = "ahmed"
    title: str = "Ahmed body"
    length_units: str = "m"
    _specs: list[ParameterSpec] = field(
        default_factory=lambda: [
            ParameterSpec("slant_angle", 25.0, 0.0, 45.0, "deg", "Angle of the rear upper slant"),
            ParameterSpec(
                "rear_taper",
                0.0,
                0.0,
                0.3,
                "",
                "Narrowing of the rear sides, as a fraction of the width",
            ),
            ParameterSpec("height", 0.288, 0.2, 0.4, "m", "Body height"),
            ParameterSpec("length", 1.044, 0.8, 1.3, "m", "Body length"),
            ParameterSpec("width", 0.389, 0.3, 0.5, "m", "Body width"),
            ParameterSpec("ride_height", 0.05, 0.02, 0.1, "m", "Ground clearance"),
        ]
    )

    def parameters(self) -> list[ParameterSpec]:
        return list(self._specs)

    def conditions(self, params: dict[str, float]) -> dict[str, float]:
        """AhmedML's parameters (`body-length`, `body-height`, `body-width`,
        `front-arc-diameter`, `slant-angle-length`, `slant-angle-height`,
        `slant-surface-length`, `slant-angle-degrees`), in millimetres, from the body's
        own. The front arc follows the reference body's proportion to the height."""
        v = self.values(params)
        slant_len = v["length"] * (0.222 / 1.044)
        drop = min(slant_len * math.tan(math.radians(v["slant_angle"])), 0.9 * v["height"])
        return {
            "body-length": 1000.0 * v["length"],
            "body-height": 1000.0 * v["height"],
            "body-width": 1000.0 * v["width"],
            "front-arc-diameter": 1000.0 * v["height"] * (0.19 / 0.288),
            "slant-angle-length": 1000.0 * slant_len,
            "slant-angle-height": 1000.0 * drop,
            "slant-surface-length": 1000.0 * math.hypot(slant_len, drop),
            "slant-angle-degrees": v["slant_angle"],
        }

    def values(self, params: dict[str, float] | None) -> dict[str, float]:
        """Every parameter, with defaults filled in and values clamped to the range."""
        given = params or {}
        return {s.name: s.clamp(float(given.get(s.name, s.default))) for s in self._specs}

    def surface(self, params: dict[str, float], *, resolution: int = 24) -> SurfaceMesh:
        from frontier_data.mesh import SurfaceMesh

        v = self.values(params)
        points, faces = ahmed_surface(
            length=v["length"],
            width=v["width"],
            height=v["height"],
            slant_deg=v["slant_angle"],
            taper=v["rear_taper"],
            ride=v["ride_height"],
            resolution=max(4, int(resolution)),
        )
        return SurfaceMesh(points=points, faces=faces)


def _grid_patch(
    corner: np.ndarray, u: np.ndarray, v: np.ndarray, nu: int, nv: int, flip: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """A planar quadrilateral patch (corner + s*u + t*v) as a triangle grid."""
    s = np.linspace(0.0, 1.0, nu + 1)
    t = np.linspace(0.0, 1.0, nv + 1)
    ss, tt = np.meshgrid(s, t, indexing="ij")
    pts = corner[None, :] + ss.reshape(-1, 1) * u[None, :] + tt.reshape(-1, 1) * v[None, :]
    idx = np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    tri = np.concatenate([np.stack([a, b, c], -1), np.stack([a, c, d], -1)], axis=0).reshape(-1, 3)
    if flip:
        tri = tri[:, ::-1]
    return pts.astype(np.float32), tri.astype(np.int32)


def ahmed_surface(
    *,
    length: float,
    width: float,
    height: float,
    slant_deg: float,
    taper: float,
    ride: float,
    resolution: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """The Ahmed body as a closed triangulated surface: x along the body (nose at 0), y
    across (centred), z up (underside at ``ride``). The rear slant takes the last
    ``0.222 / 1.044`` of the length, as on the reference body; ``taper`` narrows the rear
    sides linearly over the slant's length. Outward-facing triangles."""
    slant_len = length * (0.222 / 1.044)
    x_s = length - slant_len
    drop = slant_len * float(np.tan(np.deg2rad(slant_deg)))
    drop = min(drop, 0.9 * height)
    z0, z1 = ride, ride + height
    half = width / 2.0
    half_rear = half * (1.0 - taper)
    n = resolution
    nx_front = max(2, int(n * x_s / length))
    nx_slant = max(2, n - nx_front)
    ny = max(2, int(n * width / length))
    nz = max(2, int(n * height / length))
    patches: list[tuple[np.ndarray, np.ndarray]] = []
    ex, ey, ez = np.eye(3)
    # nose, bottom, top (front part), two sides (front part)
    patches.append(_grid_patch(np.array([0, -half, z0]), half * 2 * ey, height * ez, ny, nz, True))
    patches.append(
        _grid_patch(np.array([0, -half, z0]), x_s * ex, half * 2 * ey, nx_front, ny, True)
    )
    patches.append(_grid_patch(np.array([0, -half, z1]), x_s * ex, half * 2 * ey, nx_front, ny))
    patches.append(_grid_patch(np.array([0, -half, z0]), x_s * ex, height * ez, nx_front, nz))
    patches.append(_grid_patch(np.array([0, half, z0]), x_s * ex, height * ez, nx_front, nz, True))

    # the rear: a slanted top (planar), two tapered sides and a tapered bottom (each a
    # ruled trapezoid: planar when taper is 0, two planar triangles per cell otherwise),
    # and the flat base. Every edge of the result is shared by exactly two triangles.
    def ruled(
        near: tuple[np.ndarray, np.ndarray],
        far: tuple[np.ndarray, np.ndarray],
        n_along: int,
        n_across: int,
        flip: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        s = np.linspace(0.0, 1.0, n_along + 1)[:, None, None]
        t = np.linspace(0.0, 1.0, n_across + 1)[None, :, None]
        a0, a1 = near
        b0, b1 = far
        pts = (1 - s) * ((1 - t) * a0 + t * a1) + s * ((1 - t) * b0 + t * b1)
        pts = pts.reshape(-1, 3)
        idx = np.arange((n_along + 1) * (n_across + 1)).reshape(n_along + 1, n_across + 1)
        a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
        tri = np.concatenate([np.stack([a, b, c], -1), np.stack([a, c, d], -1)], 0).reshape(-1, 3)
        if flip:
            tri = tri[:, ::-1]
        return pts.astype(np.float32), tri.astype(np.int32)

    z_base = z1 - drop
    nl, nr = np.array([x_s, -half, 0.0]), np.array([x_s, half, 0.0])
    fl, fr = np.array([length, -half_rear, 0.0]), np.array([length, half_rear, 0.0])
    up = lambda p, z: p + np.array([0, 0, z])  # noqa: E731
    # top: near edge at z1, far edge at z_base
    patches.append(
        ruled((up(nl, z1), up(nr, z1)), (up(fl, z_base), up(fr, z_base)), nx_slant, ny, False)
    )
    # bottom: near and far edges at z0
    patches.append(ruled((up(nl, z0), up(nr, z0)), (up(fl, z0), up(fr, z0)), nx_slant, ny, True))
    # sides: across = up the body; the left side's far top is lower (the slant)
    patches.append(
        ruled((up(nl, z0), up(nl, z1)), (up(fl, z0), up(fl, z_base)), nx_slant, nz, False)
    )
    patches.append(
        ruled((up(nr, z0), up(nr, z1)), (up(fr, z0), up(fr, z_base)), nx_slant, nz, True)
    )
    # base
    patches.append(ruled((up(fl, z0), up(fr, z0)), (up(fl, z_base), up(fr, z_base)), nz, ny, True))
    points = np.concatenate([p for p, _ in patches], axis=0)
    offsets = np.cumsum([0] + [len(p) for p, _ in patches[:-1]])
    faces = np.concatenate([f + o for (_, f), o in zip(patches, offsets, strict=True)], axis=0)
    # patches meet along their edges: merge the points they share, so the surface is closed
    key = np.round(points.astype(np.float64) / (1e-6 * length)).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    merged = points[np.sort(first)]
    renumber = np.empty(len(first), dtype=np.int64)
    renumber[np.argsort(first)] = np.arange(len(first))
    faces = renumber[inverse.reshape(-1)][faces]
    keep = (
        (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    )
    return merged.astype(np.float32), faces[keep].astype(np.int32)


_PROVIDERS: dict[str, GeometryProvider] = {}


def register(provider: GeometryProvider) -> GeometryProvider:
    _PROVIDERS[provider.id] = provider
    return provider


def providers() -> dict[str, GeometryProvider]:
    """Every provider: the built-in ones and those installed under the entry-point group."""
    if "ahmed" not in _PROVIDERS:
        register(AhmedBody())
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group=ENTRY_POINT_GROUP):
            if ep.name not in _PROVIDERS:
                loaded = ep.load()
                register(loaded() if isinstance(loaded, type) else loaded)
    except Exception:  # noqa: BLE001 - a broken plugin must not take the built-ins down
        pass
    return dict(_PROVIDERS)


def provider(id_: str) -> GeometryProvider:
    found = providers().get(id_)
    if found is None:
        known = ", ".join(sorted(providers())) or "(none)"
        raise KeyError(f"no geometry provider {id_!r}; known: {known}")
    return found


__all__ = [
    "ENTRY_POINT_GROUP",
    "AhmedBody",
    "GeometryProvider",
    "ParameterSpec",
    "ahmed_surface",
    "provider",
    "providers",
    "register",
]

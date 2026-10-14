# SPDX-License-Identifier: Apache-2.0
"""Small synthetic transient generators (tests, smoke configs, demos; no network).

* :func:`spectral_vorticity_2d`: 2D incompressible Navier-Stokes in vorticity form on a periodic
  box, pseudo-spectral with 2/3 dealiasing and a Crank-Nicolson / Heun step.
* :func:`advection_diffusion_2d`: exact spectral solution of a scalar advected by a uniform
  velocity with diffusion.
* :func:`vortex_street_mesh`: analytic von Karman street surrogate behind a cylinder on an
  unstructured triangle mesh (the layout of DeepMind's ``cylinder_flow``).
* :func:`transient_surface`: travelling pressure wave on a closed box surface.
* :func:`write_synthetic_dataset`: a whole ``frontier-dataset`` v1.1 directory from any of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from frontier_data.mesh import SurfaceMesh, make_box_mesh

#: node types of the MeshGraphNets datasets (``common.NodeType``)
NODE_NORMAL, NODE_OBSTACLE, NODE_AIRFOIL, NODE_HANDLE, NODE_INFLOW, NODE_OUTFLOW, NODE_WALL = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
)


def _wavenumbers(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return kx, ky, kx**2 + ky**2


def velocity_from_vorticity(omega: np.ndarray) -> np.ndarray:
    """``(…, X, Y)`` vorticity on a 2π-periodic box -> velocity ``(…, X, Y, 2)``."""
    n = omega.shape[-1]
    kx, ky, k2 = _wavenumbers(n)
    k2 = np.where(k2 == 0, 1.0, k2)
    psi_h = np.fft.fft2(omega) / k2
    u = np.real(np.fft.ifft2(1j * ky * psi_h))
    v = np.real(np.fft.ifft2(-1j * kx * psi_h))
    return np.stack([u, v], axis=-1)


def random_vorticity(n: int, seed: int = 0, k_peak: float = 4.0, amplitude: float = 1.0):
    """Band-limited random vorticity field with unit-order RMS."""
    rng = np.random.default_rng(seed)
    _, _, k2 = _wavenumbers(n)
    k = np.sqrt(k2)
    spec = k * np.exp(-((k / k_peak) ** 2))
    w = np.fft.ifft2(spec * np.fft.fft2(rng.standard_normal((n, n)))).real
    return amplitude * w / (w.std() + 1e-12)


def spectral_vorticity_2d(
    n: int = 64,
    n_steps: int = 40,
    dt: float = 0.05,
    nu: float = 1e-3,
    seed: int = 0,
    forcing: float = 0.0,
    forcing_k: int = 4,
    substeps: int | None = None,
    omega0: np.ndarray | None = None,
) -> dict[str, Any]:
    """Integrate ``dω/dt + u·∇ω = ν∇²ω + f`` on ``[0, 2π)²`` and return ``n_steps`` snapshots
    ``dt`` apart: ``{"vorticity" (T,n,n,1), "velocity" (T,n,n,2), "times", "spacing"}``.
    ``forcing`` adds a Kolmogorov forcing ``f = forcing * k cos(k y)``."""
    kx, ky, k2 = _wavenumbers(n)
    k2s = np.where(k2 == 0, 1.0, k2)
    kmax = n // 3
    dealias = (np.abs(kx) <= kmax) & (np.abs(ky) <= kmax)
    w = random_vorticity(n, seed) if omega0 is None else np.asarray(omega0, dtype=np.float64)
    wh = np.fft.fft2(w) * dealias
    x = 2 * np.pi * np.arange(n) / n
    fh = np.fft.fft2(np.broadcast_to(forcing * forcing_k * np.cos(forcing_k * x)[None, :], (n, n)))

    def nonlinear(wh_: np.ndarray) -> np.ndarray:
        psi = wh_ / k2s
        u = np.fft.ifft2(1j * ky * psi).real
        v = np.fft.ifft2(-1j * kx * psi).real
        wx = np.fft.ifft2(1j * kx * wh_).real
        wy = np.fft.ifft2(1j * ky * wh_).real
        return (-np.fft.fft2(u * wx + v * wy) + fh) * dealias

    if substeps is None:  # CFL ~ 0.3 for |u| ~ 1.5
        substeps = max(1, int(np.ceil(dt / (0.3 * (2 * np.pi / n) / 1.5))))
    h = dt / substeps
    a = 1.0 / (1.0 + 0.5 * h * nu * k2)
    b = 1.0 - 0.5 * h * nu * k2
    out = np.empty((n_steps, n, n), dtype=np.float32)
    for t in range(n_steps):
        out[t] = np.fft.ifft2(wh).real
        for _ in range(substeps):
            n1 = nonlinear(wh)
            pred = a * (b * wh + h * n1)
            wh = a * (b * wh + 0.5 * h * (n1 + nonlinear(pred)))
    return {
        "vorticity": out[..., None],
        "velocity": velocity_from_vorticity(out.astype(np.float64)).astype(np.float32),
        "times": dt * np.arange(n_steps, dtype=np.float64),
        "spacing": [2 * np.pi / n, 2 * np.pi / n],
        "parameters": {"nu": nu, "forcing": forcing},
    }


def advection_diffusion_2d(
    n: int = 48,
    n_steps: int = 32,
    dt: float = 0.05,
    velocity: tuple[float, float] = (1.0, 0.5),
    kappa: float = 5e-3,
    seed: int = 0,
) -> dict[str, Any]:
    """Exact solution of ``∂c/∂t + u·∇c = κ∇²c`` on ``[0, 2π)²`` (uniform ``u``)."""
    kx, ky, k2 = _wavenumbers(n)
    c0 = np.fft.fft2(random_vorticity(n, seed, k_peak=3.0))
    times = dt * np.arange(n_steps, dtype=np.float64)
    phase = -1j * (kx * velocity[0] + ky * velocity[1])
    out = np.stack(
        [np.fft.ifft2(c0 * np.exp((phase - kappa * k2) * t)).real for t in times]
    ).astype(np.float32)
    vel = np.broadcast_to(np.asarray(velocity, dtype=np.float32), (n, n, 2)).copy()
    return {
        "concentration": out[..., None],
        "velocity": vel,  # static
        "times": times,
        "spacing": [2 * np.pi / n, 2 * np.pi / n],
        "parameters": {"u": velocity[0], "v": velocity[1], "kappa": kappa},
    }


def cylinder_mesh(
    n_target: int = 900,
    seed: int = 0,
    length: float = 1.6,
    height: float = 0.41,
    centre: tuple[float, float] = (0.33, 0.2),
    radius: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangle mesh of a channel with a cylinder: ``points (N,2)``, ``cells (K,3)``,
    ``node_type (N,)``."""
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(seed)
    h = float(np.sqrt(length * height / max(n_target, 16)))
    nx, ny = max(int(length / h), 4), max(int(height / h), 3)
    gx, gy = np.meshgrid(np.linspace(0, length, nx), np.linspace(0, height, ny), indexing="ij")
    jit = rng.uniform(-0.25, 0.25, size=(nx, ny, 2)) * [length / nx, height / ny]
    jit[[0, -1], :, :] = 0.0
    jit[:, [0, -1], :] = 0.0
    pts = np.stack([gx, gy], axis=-1) + jit
    pts = pts.reshape(-1, 2)
    c = np.asarray(centre)
    pts = pts[np.linalg.norm(pts - c, axis=1) > radius * 1.25]
    n_ring = max(12, int(2 * np.pi * radius / (0.5 * h)))
    ang = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    ring = c + radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    pts = np.concatenate([pts, ring]).astype(np.float64)
    tri = Delaunay(pts).simplices.astype(np.int32)
    cen = pts[tri].mean(axis=1)
    tri = tri[np.linalg.norm(cen - c, axis=1) > radius * 0.999]
    node_type = np.full(len(pts), NODE_NORMAL, dtype=np.int32)
    eps = 1e-9
    node_type[(pts[:, 1] < eps) | (pts[:, 1] > height - eps)] = NODE_WALL
    node_type[pts[:, 0] > length - eps] = NODE_OUTFLOW
    node_type[pts[:, 0] < eps] = NODE_INFLOW
    node_type[len(pts) - n_ring :] = NODE_WALL
    return pts.astype(np.float32), tri, node_type


def vortex_street_mesh(
    n_target: int = 900,
    n_steps: int = 40,
    dt: float = 0.02,
    u_inf: float = 1.0,
    strouhal: float = 0.2,
    seed: int = 0,
    radius: float = 0.05,
    centre: tuple[float, float] = (0.33, 0.2),
) -> dict[str, Any]:
    """Analytic wake surrogate: parabolic channel flow deflected around the cylinder (potential
    doublet) plus alternating Lamb-Oseen vortices shed at ``St = f D / U`` and convected
    downstream. Not a solution of Navier-Stokes, but periodic, smooth and mesh-based."""
    pts, cells, node_type = cylinder_mesh(n_target, seed, centre=centre, radius=radius)
    height = float(pts[:, 1].max())
    x, y = pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)
    dx, dy = x - centre[0], y - centre[1]
    r2 = np.maximum(dx**2 + dy**2, radius**2)
    profile = 4.0 * u_inf * y * (height - y) / height**2
    base_u = profile * (1.0 - radius**2 * (dx**2 - dy**2) / r2**2)
    base_v = profile * (-2.0 * radius**2 * dx * dy / r2**2)
    freq = strouhal * u_inf / (2 * radius)
    period = 1.0 / freq
    uc = 0.8 * u_inf
    gamma = 1.6 * u_inf * radius * 2
    core = 0.6 * radius
    times = dt * np.arange(n_steps, dtype=np.float64)
    vel = np.empty((n_steps, len(pts), 2), dtype=np.float32)
    length = float(pts[:, 0].max())
    n_vort = int(np.ceil(length / (uc * period / 2))) + 2
    for i, t in enumerate(times):
        u, v = base_u.copy(), base_v.copy()
        for j in range(-1, n_vort):
            age = t - (np.floor(t / (period / 2)) - j) * (period / 2)  # time since shedding
            if age < 0:
                continue
            k = int(np.floor(t / (period / 2))) - j
            sign = 1.0 if k % 2 == 0 else -1.0
            vx = centre[0] + radius + uc * age
            if vx > length + 4 * core:
                continue
            vy = centre[1] + sign * 0.6 * radius
            ddx, ddy = x - vx, y - vy
            rr2 = ddx**2 + ddy**2
            grow = core * np.sqrt(1.0 + 4.0 * age / period)
            strength = sign * gamma * (1.0 - np.exp(-rr2 / grow**2)) / (2 * np.pi * (rr2 + 1e-12))
            ramp = min(1.0, age / (0.25 * period))
            u += -strength * ddy * ramp
            v += strength * ddx * ramp
        wall = node_type == NODE_WALL
        u[wall], v[wall] = 0.0, 0.0
        vel[i, :, 0], vel[i, :, 1] = u, v
    speed2 = (vel.astype(np.float64) ** 2).sum(axis=-1)
    pressure = (0.5 * (u_inf**2 - speed2)).astype(np.float32)[..., None]
    return {
        "points": pts,
        "cells": cells,
        "node_type": node_type,
        "velocity": vel,
        "pressure": pressure,
        "times": times,
        "parameters": {"u_inf": u_inf, "strouhal": strouhal, "radius": radius},
    }


def transient_surface(
    subdivisions: int = 6, n_steps: int = 24, dt: float = 0.05, seed: int = 0
) -> dict[str, Any]:
    """Closed box surface with a travelling pressure wave and a shear vector field."""
    rng = np.random.default_rng(seed)
    mesh: SurfaceMesh = make_box_mesh(subdivisions=subdivisions)
    kvec = rng.uniform(4.0, 9.0, size=3)
    omega = float(rng.uniform(4.0, 8.0))
    times = dt * np.arange(n_steps, dtype=np.float64)
    phase = mesh.points.astype(np.float64) @ kvec
    p = np.cos(phase[None, :] - omega * times[:, None])
    tau = np.stack(
        [np.sin(phase[None, :] - omega * times[:, None]) * a for a in (0.02, 0.01, 0.005)], axis=-1
    )
    return {
        "mesh": mesh,
        "p": p.astype(np.float32)[..., None],
        "wallShearStress": tau.astype(np.float32),
        "times": times,
        "parameters": {"omega": omega},
    }


def write_synthetic_dataset(
    root: str | Path,
    kind: str = "grid",
    n_samples: int = 6,
    n_steps: int = 24,
    seed: int = 0,
    size: int | None = None,
    name: str | None = None,
    split: bool = True,
) -> Any:
    """Write a transient synthetic dataset and return the opened
    :class:`~frontier_data.dataset.Dataset`.

    ``kind``: ``grid`` (2D Navier-Stokes vorticity, ``size`` = grid points per side),
    ``grid_advection`` (advection-diffusion), ``mesh`` (cylinder wake, ``size`` = target node
    count), ``surface`` (box, ``size`` = subdivisions) or ``point_cloud`` (the mesh nodes
    without connectivity).
    """
    from frontier_data.dataset import Dataset, DatasetWriter
    from frontier_data.splits import random_split

    rng = np.random.default_rng(seed)
    writer = DatasetWriter(
        root,
        name=name or f"synthetic-{kind}",
        source={"kind": "synthetic", "generator": kind, "seed": seed},
        domain="cfd",
        license="CC0-1.0",
        citation="frontier_data.synthetic",
    )
    for i in range(n_samples):
        sid = f"traj_{i:03d}"
        s = seed * 1000 + i
        if kind == "grid":
            nu = float(10 ** rng.uniform(-3.3, -2.3))
            d = spectral_vorticity_2d(n=size or 32, n_steps=n_steps, dt=0.1, nu=nu, seed=s)
            writer.write_grid_sample(
                sid,
                fields={"vorticity": d["vorticity"], "velocity": d["velocity"]},
                times=d["times"],
                spacing=d["spacing"],
                boundary_conditions={"x": "periodic", "y": "periodic"},
                parameters=d["parameters"],
                meta={"solver": "frontier-spectral", "physics": "incompressible-ns-2d"},
            )
        elif kind == "grid_advection":
            ang = rng.uniform(0, 2 * np.pi)
            d = advection_diffusion_2d(
                n=size or 32,
                n_steps=n_steps,
                velocity=(float(np.cos(ang)), float(np.sin(ang))),
                kappa=float(10 ** rng.uniform(-3, -2)),
                seed=s,
            )
            writer.write_grid_sample(
                sid,
                fields={"concentration": d["concentration"]},
                static={"velocity": d["velocity"]},
                times=d["times"],
                spacing=d["spacing"],
                boundary_conditions={"x": "periodic", "y": "periodic"},
                parameters=d["parameters"],
                meta={"solver": "analytic", "physics": "advection-diffusion"},
            )
        elif kind in ("mesh", "point_cloud"):
            d = vortex_street_mesh(
                n_target=size or 500,
                n_steps=n_steps,
                u_inf=float(rng.uniform(0.7, 1.4)),
                seed=s,
                radius=float(rng.uniform(0.04, 0.06)),
                centre=(0.33, float(rng.uniform(0.17, 0.24))),
            )
            common: dict[str, Any] = {
                "points": d["points"],
                "fields": {"velocity": d["velocity"], "pressure": d["pressure"]},
                "static": {"node_type": d["node_type"]},
                "times": d["times"],
                "parameters": d["parameters"],
                "meta": {"solver": "analytic", "physics": "vortex-street-surrogate"},
            }
            if kind == "mesh":
                writer.write_mesh_sample(sid, cells=d["cells"], **common)
            else:
                writer.write_points_sample(sid, **common)
        elif kind == "surface":
            d = transient_surface(subdivisions=size or 6, n_steps=n_steps, seed=s)
            writer.write_surface_sample(
                sid,
                d["mesh"],
                fields={"p": d["p"], "wallShearStress": d["wallShearStress"]},
                times=d["times"],
                parameters=d["parameters"],
                meta={"solver": "analytic", "physics": "travelling-wave"},
            )
        else:
            raise ValueError(f"unknown synthetic kind {kind!r}")
    manifest = writer.finalize()
    ds = Dataset(root)
    if split:
        res = random_split(manifest.sample_ids(), seed=seed)
        ds.manifest.splits = res.splits
        ds.manifest.split_strategy = res.strategy
        ds.manifest.split_families = {"random": res.as_family()}
        ds.save_manifest()
    return ds

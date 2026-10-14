# SPDX-License-Identifier: Apache-2.0
"""An input dataset: cases made from typed parameters and geometry, in the layout of any
other dataset, so Inference, Adapt, viewers and a served board treat them like data
(``docs/spec/model-package.md``, "Model contract").

:func:`write_input` takes

* ``parameters``: one case (``{name: value}``) or many (a list of them, or a CSV/JSON table
  with one row per case);
* ``geometry``: a parametric provider (``frontier_data.geometry``: the parameters that the
  provider takes shape the surface, the rest are the case's parameters), one or many
  surface files (STL, OBJ, VTP, … read through the connectors), or samples of a dataset;

and writes a ``frontier-dataset`` whose every sample is in the ``test`` split and whose
manifest says ``source.type: input``. When a contract is given, the sample's
``geometry_params`` carry exactly the parameters the model takes, and missing ones fail with
the name that is missing.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from frontier_data.dataset import DatasetWriter, MeshStats, Provenance, SampleMeta
from frontier_data.geometry import provider as geometry_provider
from frontier_data.lod import LOD_TARGET_FACES, build_lods
from frontier_data.mesh import SurfaceMesh

#: levels of detail an input sample gets, as a hub conversion gives them
LEVELS = (0, 1, 2, 3)


class InputError(ValueError):
    """What the person typed cannot become a case: one line of what, and what to do."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


def read_table(path: str | Path) -> list[dict[str, float]]:
    """Cases from a CSV (one row each, a header of names) or a JSON list of objects."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        rows = json.loads(text)
        if isinstance(rows, dict):
            rows = rows.get("cases") or rows.get("rows") or [rows]
        return [{str(k): float(v) for k, v in r.items() if _is_number(v)} for r in rows]
    out = []
    for r in csv.DictReader(text.splitlines()):
        out.append({str(k).strip(): float(v) for k, v in r.items() if _is_number(v)})
    return out


def _is_number(v: Any) -> bool:
    try:
        float(v)
    except (TypeError, ValueError):
        return False
    return not isinstance(v, bool)


def _cases(parameters: Any) -> list[dict[str, float]]:
    if parameters is None:
        return [{}]
    if isinstance(parameters, str | Path):
        return read_table(parameters) or [{}]
    if isinstance(parameters, dict):
        return [{str(k): float(v) for k, v in parameters.items() if _is_number(v)}]
    return [{str(k): float(v) for k, v in c.items() if _is_number(v)} for c in parameters] or [{}]


def read_surface_file(path: str | Path, connector: str | None = None) -> SurfaceMesh:
    """One surface file as a mesh, through the connector that recognises it (STL, OBJ,
    VTP, PLY, …), else meshio."""
    p = Path(path)
    try:
        from frontier_connectors.registry import resolve

        conn = resolve(p, connector)
        for s in conn.read(p):
            if getattr(s, "kind", "") == "surface" and s.points is not None and s.faces is not None:
                return SurfaceMesh(
                    points=np.asarray(s.points, np.float32), faces=np.asarray(s.faces, np.int32)
                )
    except Exception:  # noqa: BLE001 - the generic reader below says what is wrong
        pass
    try:
        import meshio

        m = meshio.read(str(p))
    except Exception as exc:  # noqa: BLE001
        raise InputError(
            f"{p.name} is not a surface file that can be read ({exc}).",
            hint="use STL, OBJ, PLY or VTP, or convert the case with a connector first",
        ) from None
    tris = [c.data for c in m.cells if c.type == "triangle"]
    if not tris:
        raise InputError(f"{p.name} has no triangles.", hint="export the surface as triangles")
    return SurfaceMesh(
        points=np.asarray(m.points, np.float32)[:, :3], faces=np.concatenate(tris).astype(np.int32)
    )


def _contract_params(contract: dict[str, Any] | None) -> list[str]:
    if not contract:
        return []
    return [
        str(p["name"])
        for p in (contract.get("inputs") or {}).get("parameters") or []
        if p.get("required", True)
    ]


def _lods(mesh: SurfaceMesh) -> dict[int, SurfaceMesh]:
    wanted = [lv for lv in LEVELS if lv == 0 or (LOD_TARGET_FACES.get(lv) or 0) < mesh.n_faces]
    lods, _ = build_lods(mesh, levels=wanted)
    return lods


def write_input(
    out_dir: str | Path,
    parameters: Any = None,
    *,
    geometry: str | None = None,
    geometry_files: list[str | Path] | None = None,
    geometry_params: dict[str, float] | None = None,
    dataset: str | Path | None = None,
    sample_ids: list[str] | None = None,
    contract: dict[str, Any] | None = None,
    name: str = "input",
    connector: str | None = None,
    resolution: int = 24,
) -> Path:
    """Write the input dataset to ``out_dir`` (replaced when it exists). Exactly one of
    ``geometry`` (a provider id), ``geometry_files`` or ``dataset`` gives the shapes:

    * a provider: one case per row of ``parameters``; the provider's own parameters come
      from the row, then ``geometry_params``, then the provider's defaults;
    * files: one case per file; with one row of ``parameters`` every file takes it, with
      as many rows as files each file takes its row;
    * a dataset: its samples (``sample_ids``, else all), each with its own parameters, the
      rows of ``parameters`` laid over them in order.

    Returns the dataset directory."""
    out = Path(out_dir)
    cases = _cases(parameters)
    needed = _contract_params(contract)
    sources = sum(x is not None for x in (geometry, geometry_files, dataset))
    if sources != 1:
        raise InputError(
            "Say where the geometry comes from: a parametric shape, surface files or a dataset.",
            hint="set one of geometry, geometry_files, dataset",
        )
    if out.exists():
        shutil.rmtree(out)
    writer = DatasetWriter(
        out,
        name,
        source={
            "type": "input",
            "geometry": geometry or ("files" if geometry_files else "dataset"),
        },
    )
    made: list[str] = []

    def add(sid: str, mesh: SurfaceMesh, params: dict[str, float], extra: dict[str, Any]) -> None:
        missing = [n for n in needed if n not in params]
        if missing:
            raise InputError(
                f"Case {sid} does not give the parameter {missing[0]} the model needs.",
                hint="add a value for it, or a column of that name to the table",
            )
        lods = _lods(mesh)
        lo, hi = mesh.points.min(axis=0), mesh.points.max(axis=0)
        meta = SampleMeta(
            id=sid,
            dataset=name,
            solver="input",
            fidelity="none",
            mesh=MeshStats(
                n_points=mesh.n_points,
                n_faces=mesh.n_faces,
                bounds=[float(v) for pair in zip(lo, hi, strict=True) for v in pair],
                area=float(mesh.areas().sum()),
            ),
            geometry_params=dict(params),
            provenance=Provenance(source="frontier_data.inputs", ingest_tool="frontier-data"),
            sample_kind="surface",
            temporal="steady",
            lod_levels=sorted(lods),
            extra={"input": extra},
        )
        writer.write_surface_sample(sid, lods, meta=meta, forces={}, parameters=dict(params))
        made.append(sid)

    if geometry is not None:
        prov = geometry_provider(geometry)
        own = {s.name for s in prov.parameters()}
        for i, row in enumerate(cases):
            shape_params = {k: v for k, v in {**(geometry_params or {}), **row}.items() if k in own}
            mesh = prov.surface(shape_params, resolution=resolution)
            # the case's parameters: the shape's, as the provider settled them, under the row
            values = cast(
                Callable[[dict[str, float]], Mapping[str, float]] | None,
                getattr(prov, "values", None),
            )
            settled: dict[str, float] = (
                {str(k): float(v) for k, v in values(shape_params).items()}
                if callable(values)
                else {s.name: shape_params.get(s.name, s.default) for s in prov.parameters()}
            )
            # the shape's own parameters, the names models trained on public datasets of
            # this shape use, and the row (which wins)
            published = cast(
                Callable[[dict[str, float]], Mapping[str, float]] | None,
                getattr(prov, "conditions", None),
            )
            named = published(settled) if callable(published) else {}
            full = {**settled, **{str(k): float(v) for k, v in named.items()}, **row}
            add(
                f"case_{i + 1:04d}",
                mesh,
                full,
                {"provider": prov.id, "length_units": prov.length_units},
            )
    elif geometry_files is not None:
        files = [Path(f) for f in geometry_files]
        if len(cases) not in (1, len(files)):
            raise InputError(
                f"{len(cases)} rows of parameters for {len(files)} geometries.",
                hint="give one row for all, or one row per geometry",
            )
        taken: set[str] = set()
        for i, f in enumerate(files):
            row = cases[0] if len(cases) == 1 else cases[i]
            sid = _sid(f.stem, i)
            if sid in taken:  # two files of the same name: the second is numbered
                sid = f"{sid}-{i + 1}"
            taken.add(sid)
            add(sid, read_surface_file(f, connector), row, {"file": str(f)})
    else:
        from frontier_data.dataset import read_surface

        root = Path(str(dataset))
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        entries = {s["id"]: s for s in manifest.get("samples") or []}
        ids = sample_ids or list(entries)
        if len(cases) not in (1, len(ids)) and parameters is not None:
            raise InputError(
                f"{len(cases)} rows of parameters for {len(ids)} samples.",
                hint="give one row per sample",
            )
        for i, sid in enumerate(ids):
            if sid not in entries:
                raise InputError(
                    f"The dataset has no sample {sid}.", hint="choose one of its samples"
                )
            mesh = read_surface(root / "samples" / sid / "surface.zarr", lod=0, fields=False)
            row = cases[0] if len(cases) == 1 else cases[i]
            params = {**(entries[sid].get("geometry_params") or {}), **row}
            add(sid, mesh, params, {"dataset": str(root), "sample": sid})
    writer.manifest.splits = {"train": [], "val": [], "test": list(made)}
    writer.finalize()  # writes manifest.json
    return out


def _sid(stem: str, i: int) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in stem).strip("-")
    return safe or f"case_{i + 1:04d}"


__all__ = ["InputError", "LEVELS", "read_surface_file", "read_table", "write_input"]

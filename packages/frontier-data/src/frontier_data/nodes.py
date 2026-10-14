# SPDX-License-Identifier: Apache-2.0
"""Built-in pipeline nodes: ``hf_source``, ``local_source``, ``synthetic_source``, ``vtk_reader``,
``lod``, ``dataset_sink``, ``split``, ``quality_checks``.

Items are plain dicts. The AhmedML-specific chain populates:
``id, run, files{role: path}, provenance[...]`` (source) -> ``mesh, stl_area, forces,
geometry_params, native_cell_area_sum`` (vtk_reader) -> ``lods, lod_timings`` (lod) ->
``entry`` (dataset_sink).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from frontier_data import ahmedml
from frontier_data.dataset import (
    Dataset,
    DatasetWriter,
    FieldInfo,
    MeshStats,
    Provenance,
    RawFileRecord,
    SampleEntry,
    SampleMeta,
    write_sample,
)
from frontier_data.lod import build_lods
from frontier_data.mesh import SurfaceMesh, make_box_mesh
from frontier_data.pipeline import NodeContext, register
from frontier_data.sources import huggingface as hf

log = logging.getLogger(__name__)


# -- sources ------------------------------------------------------------------------------


def _runs_param(ctx: NodeContext) -> list[int]:
    runs = ctx.param("runs", "1-8")
    if isinstance(runs, str):
        return hf.parse_runs(runs)
    return [int(r) for r in runs]


@register(
    "hf_source",
    "source",
    "sample",
    description="Download AhmedML runs from HuggingFace (resumable, sha256 provenance).",
    params={
        "runs": "1-8",
        "raw_dir": str(hf.DEFAULT_RAW_DIR),
        "revision": None,
        "download_workers": 4,
    },
)
def hf_source(ctx: NodeContext) -> list[dict[str, Any]]:
    runs = _runs_param(ctx)
    raw_dir = Path(ctx.param("raw_dir", hf.DEFAULT_RAW_DIR))
    revision = ctx.param("revision")
    workers = int(ctx.param("download_workers", 4))
    t0 = time.perf_counter()
    got = hf.download_runs(
        runs,
        raw_dir,
        workers=workers,
        revision=revision,
        progress=lambda msg, d, t: ctx.emit(msg, d, t),
    )
    log.info("downloaded %d runs in %.1fs", len(got), time.perf_counter() - t0)
    items = []
    for run, files in got.items():
        items.append(
            {
                "id": ahmedml.run_id(run),
                "run": run,
                "files": {role: f.local_path for role, f in files.items()},
                "provenance": [
                    RawFileRecord(
                        role=f.role,
                        repo_path=f.repo_path,
                        url=f.url,
                        sha256=f.sha256,
                        size=f.size,
                        revision=f.revision,
                    ).model_dump()
                    for f in files.values()
                ],
                "source": f"huggingface:{ahmedml.HF_REPO}",
            }
        )
    return items


@register(
    "local_source",
    "source",
    "sample",
    description="Use already-downloaded AhmedML runs under raw_dir without touching the network.",
    params={"runs": "1-8", "raw_dir": str(hf.DEFAULT_RAW_DIR)},
)
def local_source(ctx: NodeContext) -> list[dict[str, Any]]:
    runs = _runs_param(ctx)
    raw_dir = Path(ctx.param("raw_dir", hf.DEFAULT_RAW_DIR))
    store = hf.ProvenanceStore(raw_dir)
    items = []
    for run in runs:
        files: dict[str, str] = {}
        prov = []
        for role, rel in ahmedml.run_files(run).items():
            p = raw_dir / rel
            if not p.exists():
                if role in ahmedml.OPTIONAL_ROLES:
                    continue
                raise FileNotFoundError(f"{p} missing (run local download first)")
            files[role] = str(p)
            rec = store.record(
                rel, p, hf.resolve_url(ahmedml.HF_REPO, rel, None, ahmedml.HF_REPO_TYPE), None
            )
            prov.append(
                RawFileRecord(
                    role=role, repo_path=rel, url=rec.url, sha256=rec.sha256, size=rec.size
                ).model_dump()
            )
        items.append(
            {
                "id": ahmedml.run_id(run),
                "run": run,
                "files": files,
                "provenance": prov,
                "source": f"local:{raw_dir}",
            }
        )
    store.save()
    return items


@register(
    "synthetic_source",
    "source",
    "sample",
    description="Generate small synthetic box meshes (tests/demos, no I/O).",
    params={"n": 4, "subdivisions": 6, "seed": 0},
)
def synthetic_source(ctx: NodeContext) -> list[dict[str, Any]]:
    n = int(ctx.param("n", 4))
    return [
        {
            "id": f"box_{i}",
            "run": i,
            "files": {},
            "provenance": [],
            "source": "synthetic",
            "subdivisions": int(ctx.param("subdivisions", 6)),
            "seed": int(ctx.param("seed", 0)) + i,
        }
        for i in range(1, n + 1)
    ]


# -- transforms ---------------------------------------------------------------------------


def _ahmedml_meta(
    item: dict[str, Any],
    mesh: SurfaceMesh,
    stats: MeshStats,
    forces: dict[str, Any],
    geometry: dict[str, float],
) -> SampleMeta:
    fields = [
        FieldInfo(
            name=n,
            units=ahmedml.FIELDS.get(n, ("-", ""))[0],
            description=ahmedml.FIELDS.get(n, ("-", ""))[1],
            channels=int(arr.shape[1]),
            location="cell",
        )
        for n, arr in mesh.cell_fields.items()
    ]
    for n, arr in mesh.point_fields.items():
        if n not in mesh.cell_fields:
            fields.append(FieldInfo(name=n, channels=int(arr.shape[1]), location="point"))
    prov = Provenance(
        source=str(item.get("source", f"huggingface:{ahmedml.HF_REPO}")),
        license=ahmedml.LICENSE,
        citation=ahmedml.PAPER_URL,
        files=[RawFileRecord.model_validate(p) for p in item.get("provenance", [])],
    )
    return SampleMeta(
        id=item["id"],
        dataset="ahmedml",
        solver=str(ahmedml.SOLVER["solver"]),
        solver_version=str(ahmedml.SOLVER["solver_version"]),
        application=str(ahmedml.SOLVER["application"]),
        turbulence_model=str(ahmedml.SOLVER["turbulence_model"]),
        turbulence_family=str(ahmedml.SOLVER["turbulence_family"]),
        physics=str(ahmedml.SOLVER["physics"]),
        fidelity=str(ahmedml.SOLVER["fidelity"]),
        fidelity_tier=str(ahmedml.SOLVER["fidelity_tier"]),
        mesh=stats,
        geometry_params=geometry,
        geometry_param_units=ahmedml.GEOMETRY_PARAM_UNITS,
        conditions=dict(ahmedml.CONDITIONS),
        boundary_conditions={
            "inlet": "fixedValue U=(1 0 0)",
            "outlet": "zeroGradient / fixed p=0",
            "body": "no-slip wall (patch group ahmedGroup)",
            "ground": "stationary wall",
            "source": "openfoam-casesetup.tgz in the dataset repository",
        },
        forces=forces,
        fields=fields,
        field_roles=dict(ahmedml.FIELD_ROLES),
        provenance=prov,
        extra={
            "volume_mesh_cells_approx": ahmedml.SOLVER["mesh_cells_volume_approx"],
            "citation": ahmedml.PAPER_URL,
        },
    )


@register(
    "vtk_reader",
    "transform",
    "sample",
    description="Read boundary VTP + STL + CSVs (+ native cell areas) into a SurfaceMesh.",
    params={"dataset": "ahmedml", "keep_fields": None},
)
def vtk_reader(ctx: NodeContext, item: dict[str, Any]) -> dict[str, Any]:
    from frontier_data.readers import vtk as rd

    files: dict[str, str] = item.get("files", {})
    timings: dict[str, float] = {}
    if "surface" in files:
        t0 = time.perf_counter()
        mesh = rd.read_vtp(files["surface"])
        timings["read_vtp_s"] = time.perf_counter() - t0
    else:  # synthetic item
        mesh = make_box_mesh(subdivisions=int(item.get("subdivisions", 6)))
        rng = np.random.default_rng(int(item.get("seed", 0)))
        mesh.cell_fields["p"] = mesh.cell_fields["p"] * float(rng.uniform(0.8, 1.2))
        mesh.point_fields["p"] = mesh.cell_to_point(mesh.cell_fields["p"])
    keep = ctx.param("keep_fields")
    if keep:
        mesh.cell_fields = {k: v for k, v in mesh.cell_fields.items() if k in keep}
        mesh.point_fields = {k: v for k, v in mesh.point_fields.items() if k in keep}
    mesh.areas()

    stats = MeshStats(
        n_points=mesh.n_points, n_faces=mesh.n_faces, bounds=list(mesh.bounds), area=mesh.area
    )
    if "cell_area" in files:  # prefer the solver's own polygon areas, split over triangles
        native = rd.read_cell_area_npy(files["cell_area"])
        stats.native_n_cells = int(native.shape[0])
        stats.native_area_sum = float(native.sum(dtype=np.float64))
        if mesh.face_parent is not None and native.shape[0] == int(mesh.face_parent.max()) + 1:
            mesh.cell_area = rd.distribute_parent_areas(mesh, native)
            stats.area = mesh.area
    if "stl" in files:
        t0 = time.perf_counter()
        stl = rd.read_stl(files["stl"])
        stats.stl_n_faces = stl.n_faces
        stats.stl_area = stl.area
        timings["read_stl_s"] = time.perf_counter() - t0
    forces: dict[str, Any] = {}
    if "forces" in files:
        forces.update({k.lower(): v for k, v in rd.read_scalar_csv(files["forces"]).items()})
    if "forces_varref" in files:
        forces.update(
            {
                f"{k.lower()}_varref": v
                for k, v in rd.read_scalar_csv(files["forces_varref"]).items()
            }
        )
    if forces:
        forces["reference"] = {
            "area": ahmedml.CONDITIONS["reference_area"],
            "u_inf": ahmedml.CONDITIONS["u_inf"],
            "rho": ahmedml.CONDITIONS["rho"],
            "note": "cd/cl use the constant Aref; *_varref use the per-geometry frontal area",
        }
    geometry = rd.read_scalar_csv(files["geometry"]) if "geometry" in files else {}
    if not files:  # synthetic: fake geometry params + forces from the integrated pressure
        rng = np.random.default_rng(int(item.get("seed", 0)))
        geometry = {
            "slant-angle-degrees": float(rng.uniform(10, 40)),
            "body-length": float(rng.uniform(900, 1300)),
        }
        n = mesh.face_normals()
        a = mesh.areas().astype(np.float64)
        fx = float((mesh.cell_fields["p"][:, 0] * n[:, 0] * a).sum()) - float(
            (mesh.cell_fields["wallShearStress"][:, 0] * a).sum()
        )
        forces = {"cd": 2 * fx / 0.112032, "cl": 0.0}
        stats.stl_area = mesh.area

    meta = (
        _ahmedml_meta(item, mesh, stats, forces, geometry)
        if ctx.param("dataset", "ahmedml") == "ahmedml"
        else None
    )
    if meta is None:
        raise ValueError(f"unsupported dataset {ctx.param('dataset')!r}")
    if not files:
        meta.conditions = {**meta.conditions}
        meta.field_roles = {"pressure": "p", "wall_shear_stress": "wallShearStress"}
    meta.extra["timings"] = timings
    item.update(
        {
            "mesh": mesh,
            "meta": meta,
            "forces": forces,
            "geometry_params": geometry,
            "stl_area": stats.stl_area,
            "native_cell_area_sum": stats.native_area_sum,
            "mesh_stats": stats.model_dump(),
        }
    )
    return item


@register(
    "lod",
    "transform",
    "sample",
    description="Build LOD levels (0 full, 1 ~100k, 2 ~20k, 3 ~5k faces) carrying the fields.",
    params={"levels": [0, 2, 3], "method": "auto"},
)
def lod_node(ctx: NodeContext, item: dict[str, Any]) -> dict[str, Any]:
    levels = [int(x) for x in ctx.param("levels", [0, 2, 3])]
    if 0 not in levels:
        levels.append(0)
    lods, timings = build_lods(item["mesh"], levels, method=str(ctx.param("method", "auto")))
    item["lods"] = lods
    item["lod_timings"] = [t.__dict__ for t in timings]
    meta: SampleMeta = item["meta"]
    meta.lod_stats = {
        str(t.level): {
            "n_faces": t.n_faces,
            "n_points": t.n_points,
            "seconds": t.seconds,
            "method": t.method,
        }
        for t in timings
    }
    return item


# -- sinks --------------------------------------------------------------------------------


def _sink_finalize(ctx: NodeContext, results: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir = Path(ctx.param("out_dir", "datasets/out"))
    writer = DatasetWriter(
        out_dir,
        name=str(ctx.param("name", out_dir.name)),
        source=ctx.param(
            "source",
            {"repo": ahmedml.HF_REPO, "license": ahmedml.LICENSE, "citation": ahmedml.PAPER_URL},
        ),
    )
    for r in results:
        if "entry" not in r:
            continue
        entry = SampleEntry.model_validate(r["entry"])
        meta = SampleMeta.model_validate(r["meta"]) if isinstance(r.get("meta"), dict) else None
        writer.add_entry(entry, meta)
    writer.manifest.pipeline = "pipeline.json"
    ctx.pipeline.save(out_dir / "pipeline.json")
    manifest = writer.finalize()
    return {"n_samples": len(manifest.samples), "out_dir": str(out_dir)}


@register(
    "dataset_sink",
    "sink",
    "sample",
    description="Write samples/<id>/ (zarr v3 surface, meta.json, forces.json) and the manifest.",
    params={"out_dir": "datasets/out", "name": None, "copy_stl": False},
    finalize=_sink_finalize,
)
def dataset_sink(ctx: NodeContext, item: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(ctx.param("out_dir", "datasets/out"))
    lods: dict[int, SurfaceMesh] = item.get("lods") or {0: item["mesh"]}
    stl = (
        Path(item["files"]["stl"])
        if ctx.param("copy_stl", False) and "stl" in item.get("files", {})
        else None
    )
    entry = write_sample(out_dir, item["id"], lods, item["meta"], item["forces"], stl_path=stl)
    item["entry"] = entry.model_dump()
    item["meta"] = item["meta"].model_dump()
    item.pop("mesh", None)
    item.pop("lods", None)
    return item


# -- dataset-level ------------------------------------------------------------------------


@register(
    "split",
    "transform",
    "dataset",
    description="Assign train/val/test splits (random or parameter_holdout) into the manifest.",
    params={
        "out_dir": "datasets/out",
        "strategy": "random",
        "fractions": [0.8, 0.1, 0.1],
        "seed": 0,
        "parameter": ahmedml.GENERALIZATION_PARAMETER,
        "holdout_fraction": 0.15,
        "side": "top",
        "val_fraction": 0.1,
        "families": ["random", "parameter_holdout"],
    },
)
def split_node(ctx: NodeContext, results: list[dict[str, Any]]) -> dict[str, Any]:
    from frontier_data.splits import make_split

    ds = Dataset(Path(ctx.param("out_dir", "datasets/out")))
    ids = ds.sample_ids
    params = ds.geometry_params()
    seed = int(ctx.param("seed", 0))
    families: dict[str, Any] = {}
    for fam in ctx.param("families", ["random", "parameter_holdout"]):
        try:
            if fam == "random":
                res = make_split(
                    "random",
                    ids,
                    fractions=tuple(ctx.param("fractions", [0.8, 0.1, 0.1])),
                    seed=seed,
                )
            else:
                res = make_split(
                    "parameter_holdout",
                    ids,
                    params,
                    parameter=str(ctx.param("parameter") or ahmedml.GENERALIZATION_PARAMETER),
                    holdout_fraction=float(ctx.param("holdout_fraction", 0.15)),
                    side=str(ctx.param("side", "top")),
                    val_fraction=float(ctx.param("val_fraction", 0.1)),
                    seed=seed,
                )
        except ValueError as exc:
            log.warning("split family %s skipped: %s", fam, exc)
            continue
        families[fam] = res.as_family()
    active = str(ctx.param("strategy", "random"))
    if active not in families:
        raise ValueError(f"active split strategy {active!r} could not be built")
    ds.manifest.split_families = families
    ds.manifest.splits = families[active]["splits"]
    ds.manifest.split_strategy = families[active]["strategy"]
    ds.save_manifest()
    return {
        "active": active,
        "sizes": {k: len(v) for k, v in ds.manifest.splits.items()},
        "families": list(families),
    }


@register(
    "quality_checks",
    "transform",
    "dataset",
    description="Run the quality checks and write quality/report.{json,html}.",
    params={"out_dir": "datasets/out", "html": True},
)
def quality_node(ctx: NodeContext, results: list[dict[str, Any]]) -> dict[str, Any]:
    from frontier_data.quality import run_quality

    ds = Dataset(Path(ctx.param("out_dir", "datasets/out")))
    report = run_quality(
        ds, write_html=bool(ctx.param("html", True)), progress=lambda msg, d, t: ctx.emit(msg, d, t)
    )
    return report["summary"]


# -- convenience ----------------------------------------------------------------------------


def ahmedml_pipeline(
    out_dir: str | Path,
    runs: str = "1-8",
    *,
    raw_dir: str | Path = hf.DEFAULT_RAW_DIR,
    lod_levels: list[int] | None = None,
    split_strategy: str = "random",
    quality: bool = True,
    source_kind: str = "hf_source",
    name: str | None = None,
    revision: str | None = None,
    copy_stl: bool = False,
) -> Any:
    """Build the standard AhmedML ingest pipeline (returned, not run)."""
    from frontier_data.pipeline import Pipeline

    out_dir = Path(out_dir)
    p = Pipeline(
        name=name or out_dir.name, description=f"AhmedML runs {runs} -> frontier-dataset v1"
    )
    p.add("source", source_kind, runs=runs, raw_dir=str(raw_dir), revision=revision)
    p.add("read", "vtk_reader", ["source"], dataset="ahmedml")
    p.add("lod", "lod", ["read"], levels=lod_levels or [0, 2, 3])
    p.add(
        "sink",
        "dataset_sink",
        ["lod"],
        out_dir=str(out_dir),
        name=name or out_dir.name,
        copy_stl=copy_stl,
    )
    p.add("split", "split", ["sink"], out_dir=str(out_dir), strategy=split_strategy)
    if quality:
        p.add("quality", "quality_checks", ["split"], out_dir=str(out_dir))
    return p

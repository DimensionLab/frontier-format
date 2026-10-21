# SPDX-License-Identifier: Apache-2.0
"""``python -m frontier_data`` command line interface (typer + rich)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.table import Table

from frontier_data import joblog
from frontier_data.pipeline import PipelineEvent

app = typer.Typer(
    help="Ingest, inspect, validate and version Frontier datasets",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
ingest_app = typer.Typer(
    help="Download a public dataset and convert it to the Frontier dataset format",
    no_args_is_help=True,
)
app.add_typer(ingest_app, name="ingest")
from frontier_data.cli_versions import versions_app  # noqa: E402

app.add_typer(versions_app, name="versions")
console = joblog.console("frontier_data")


@app.command("query-field")
def query_field(
    dataset: Path,
    sample: str,
    field: str,
    select: Annotated[
        list[str] | None, typer.Option(help="Named index/slice, e.g. time=10 x=20:80")
    ] = None,
    engine: str = "zarr-python",
    budget_mib: Annotated[int, typer.Option(min=1)] = 128,
    max_output_mib: Annotated[
        int, typer.Option(min=1, help="Maximum exported array size; read buffers remain bounded")
    ] = 1024,
    out: Annotated[
        Path | None, typer.Option(help="Write .npy; omit to inspect read costs only")
    ] = None,
) -> None:
    """Inspect or read a field region without loading a whole simulation."""
    from frontier_data.dataset import Dataset
    from frontier_data.query import FieldReader, parse_selectors

    try:
        reader = FieldReader(
            Dataset(dataset).sample(sample), field, engine=engine, buffer_bytes=budget_mib << 20
        )
        _query_field_output(
            reader, parse_selectors(select or []), sample, field, engine, out, max_output_mib
        )
    except (ValueError, KeyError, OSError, ImportError, MemoryError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _query_field_output(reader, selectors, sample, field, engine, out, max_output_mib):
    from dataclasses import asdict

    from arrayroute import ReadStats
    from arrayroute.export import save_npy

    cost = reader.plan(selectors)
    result = {
        "sample": sample,
        "snapshot_id": reader.snapshot_id,
        "field": field,
        "dimensions": reader.dimensions,
        "selected_dimensions": [
            d
            for d, s in zip(reader.dimensions, cost.selection, strict=True)
            if isinstance(s, slice)
        ],
        "units": reader.units,
        "association": reader.association,
        "output_shape": cost.output_shape,
        "output_bytes": cost.output_bytes,
        "chunks_touched": cost.chunks_touched,
        "decoded_bytes_upper_bound": cost.decoded_bytes_upper_bound,
        "decode_amplification": cost.decode_amplification,
        "engine": engine,
    }
    if out is not None:
        stats = ReadStats()
        save_npy(
            reader.reader,
            reader.selection(selectors),
            out,
            max_output_bytes=max_output_mib << 20,
            stats=stats,
        )
        result["read"] = asdict(stats)
    typer.echo(json.dumps(result, indent=2))


@app.command("query-snapshot")
def query_snapshot(
    repository: Path,
    array_path: str,
    snapshot_id: str | None = None,
    branch: str | None = None,
    select: Annotated[list[str] | None, typer.Option(help="Named index/slice")] = None,
    budget_mib: Annotated[int, typer.Option(min=1)] = 128,
    max_output_mib: Annotated[int, typer.Option(min=1)] = 1024,
    out: Annotated[
        Path | None, typer.Option(help="Write .npy; omit for metadata and read costs")
    ] = None,
) -> None:
    """Pin an Icechunk field revision and inspect or export a scientific region."""
    if (snapshot_id is None) == (branch is None):
        raise typer.BadParameter("Choose exactly one of --snapshot-id or --branch")
    try:
        import icechunk as ic
    except ImportError as exc:
        raise typer.BadParameter("Use Python 3.12+ and install frontier-data[snapshots] to read revisions") from exc
    from arrayroute.snapshot import FilesystemRepository, Snapshot, SnapshotReadSession
    from frontier_data.query import SnapshotFieldReader, parse_selectors

    try:
        location = FilesystemRepository(str(repository))
        pinned = (
            Snapshot(location, snapshot_id)
            if snapshot_id is not None
            else Snapshot.resolve(location, branch=branch or "main")
        )
        reader = SnapshotFieldReader(
            SnapshotReadSession(pinned, buffer_bytes=budget_mib << 20), array_path
        )
        _query_field_output(
            reader, parse_selectors(select or []), None, array_path, "icechunk", out, max_output_mib
        )
    except (ValueError, KeyError, OSError, MemoryError, ic.IcechunkError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("compare-field")
def compare_field(
    reference: Path,
    candidate: Path,
    sample: str,
    field: str,
    tolerance: Annotated[float, typer.Option(help="Absolute tolerance in the stored field units")],
    select: Annotated[list[str] | None, typer.Option(help="Named index/slice")] = None,
    candidate_sample: str | None = None,
    engine: str = "zarr-python",
    budget_mib: Annotated[int, typer.Option(min=1)] = 128,
) -> None:
    """Measure error between explicitly paired fields with the same geometry/index space."""
    from dataclasses import asdict

    from frontier_data.dataset import Dataset
    from frontier_data.query import FieldReader, compare_fields, parse_selectors

    try:
        expected = FieldReader(
            Dataset(reference).sample(sample), field, engine=engine, buffer_bytes=budget_mib << 20
        )
        actual = FieldReader(
            Dataset(candidate).sample(candidate_sample or sample),
            field,
            engine=engine,
            buffer_bytes=budget_mib << 20,
        )
        result = compare_fields(
            expected, actual, parse_selectors(select or []), absolute_tolerance=tolerance
        )
        typer.echo(
            json.dumps(
                {
                    "field": field,
                    "units": expected.units,
                    "association": expected.association,
                    "assessment": {**asdict(result), "passed": result.passed},
                },
                indent=2,
            )
        )
        if not result.passed:
            raise typer.Exit(1)
    except (ValueError, KeyError, OSError, ImportError, MemoryError, ArithmeticError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _setup_logging(verbose: bool) -> None:
    joblog.setup(verbose)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def _parse_levels(text: str) -> list[int]:
    return sorted({int(x) for x in text.split(",") if x.strip()})


class _Progress:
    """Console progress printer for pipeline events."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self._last: dict[str, float] = {}

    def __call__(self, ev: PipelineEvent) -> None:
        t = time.perf_counter() - self.t0
        if ev.status == "start":
            console.print(f"[dim]{t:7.1f}s[/] [bold]{ev.node_id}[/] start")
        elif ev.status == "done":
            console.print(
                f"[dim]{t:7.1f}s[/] [bold]{ev.node_id}[/] done "
                f"({ev.done}/{ev.total}, {ev.elapsed_s:.1f}s in node)"
                if ev.total
                else f"[dim]{t:7.1f}s[/] [bold]{ev.node_id}[/] done ({ev.elapsed_s:.1f}s)"
            )
        elif ev.status == "error":
            console.print(
                f"[dim]{t:7.1f}s[/] [red]{ev.node_id}[/] {ev.item_id or ''} ERROR {ev.message}"
            )
        elif ev.status == "progress" and ev.total:
            key = f"{ev.node_id}:{ev.item_id}"
            if key in self._last:
                return
            self._last[key] = t
            if ev.item_id:
                console.print(
                    f"[dim]{t:7.1f}s[/] {ev.node_id} {ev.item_id} [{ev.done}/{ev.total}] "
                    f"{ev.elapsed_s:.1f}s"
                    if ev.elapsed_s
                    else f"[dim]{t:7.1f}s[/] {ev.node_id} {ev.item_id} [{ev.done}/{ev.total}]"
                )
            elif ev.message:
                console.print(f"[dim]{t:7.1f}s[/] {ev.node_id} {ev.message} [{ev.done}/{ev.total}]")


@ingest_app.command("ahmedml")
def ingest_ahmedml(
    runs: Annotated[str, typer.Option(help='Runs to ingest, e.g. "1-40" or "1,3,5-8"')] = "1-8",
    out: Annotated[Path, typer.Option(help="Write the dataset to this directory")] = Path(
        "datasets/ahmedml-8"
    ),
    raw_dir: Annotated[Path, typer.Option(help="Keep raw downloads in this directory")] = Path(
        "datasets/raw/ahmedml"
    ),
    lod: Annotated[str, typer.Option(help="Store these LOD levels, comma separated")] = "0,2,3",
    workers: Annotated[int, typer.Option(help="Set the number of worker processes")] = 4,
    split: Annotated[
        str, typer.Option(help="Set the active split: random | parameter_holdout")
    ] = "random",
    quality: Annotated[bool, typer.Option(help="Run quality checks after ingest")] = True,
    local: Annotated[
        bool, typer.Option(help="Use the files already in --raw-dir instead of downloading")
    ] = False,
    revision: Annotated[
        str | None, typer.Option(help="Pin the Hugging Face revision (commit or tag)")
    ] = None,
    copy_stl: Annotated[bool, typer.Option(help="Copy geometry.stl into each sample")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Log debug messages")] = False,
) -> None:
    """Download AhmedML runs and convert them to a surface dataset"""
    from frontier_data.nodes import ahmedml_pipeline
    from frontier_data.pipeline import PipelineRunner

    _setup_logging(verbose)
    pipeline = ahmedml_pipeline(
        out,
        runs,
        raw_dir=raw_dir,
        lod_levels=_parse_levels(lod),
        split_strategy=split,
        quality=quality,
        source_kind="local_source" if local else "hf_source",
        revision=revision,
        copy_stl=copy_stl,
    )
    console.print(
        f"[bold]ingest ahmedml[/] runs={runs} out={out} lod={_parse_levels(lod)} workers={workers}"
    )
    runner = PipelineRunner(workers=workers, on_event=_Progress())
    result = runner.run(pipeline)
    table = Table(title="node timings")
    table.add_column("node")
    table.add_column("seconds (sum over items)", justify="right")
    for nid, secs in result.timings.items():
        table.add_row(nid, f"{secs:.1f}")
    console.print(table)
    console.print(
        f"finished in {result.elapsed_s:.1f}s, {len(result.items)} samples, "
        f"{len(result.errors)} errors"
    )
    for err in result.errors:
        console.print(f"[red]{err.get('item_id') or err.get('node_id')}: {err['error']}[/]")
    for nid, out_ in result.outputs.items():
        if out_:
            console.print(f"  {nid}: {json.dumps(out_, default=str)[:300]}")
    if result.errors:
        raise typer.Exit(code=1)


def _ingest_summary(ds: object) -> None:
    from frontier_data.dataset import Dataset

    assert isinstance(ds, Dataset)
    m = ds.manifest
    console.print(
        f"[bold]{m.name}[/] v{m.version}.{m.minor} {m.sample_kind}/{m.temporal}: "
        f"{len(m.samples)} samples, "
        + ", ".join(f"{k}={len(v)}" for k, v in m.splits.items())
        + f" at {ds.root}"  # no size walk: stores have many chunk files (slow on network disks)
    )
    if m.time:
        console.print(f"time: {m.time.n_steps} steps, dt={m.time.dt}, uniform={m.time.uniform}")
    console.print(f"fields: {[(f.name, f.channels) for f in m.fields]}  params: {m.parameters}")


@ingest_app.command("the-well")
def ingest_the_well_cmd(
    name: Annotated[
        str, typer.Argument(help="Name the The Well dataset, e.g. turbulent_radiative_layer_2D")
    ],
    out: Annotated[Path | None, typer.Option(help="Output dataset directory")] = None,
    trajectories: Annotated[
        int, typer.Option(help="Limit the total number of trajectories (0 = all)")
    ] = 8,
    source_splits: Annotated[
        str, typer.Option(help="Read these source split folders, comma separated")
    ] = "train,valid,test",
    per_file: Annotated[
        int | None, typer.Option(help="Limit the trajectories taken from each HDF5 file")
    ] = None,
    resplit: Annotated[
        bool, typer.Option(help="Split by trajectory instead of keeping the source splits")
    ] = False,
    raw_dir: Annotated[Path | None, typer.Option(help="Raw download cache")] = None,
    time_stride: Annotated[int, typer.Option(help="Keep every n-th step")] = 1,
    max_steps: Annotated[int | None, typer.Option(help="Keep at most this many steps")] = None,
    max_gb: Annotated[
        float | None, typer.Option(help="Stop downloading after this many GB")
    ] = None,
    fields: Annotated[
        str | None, typer.Option(help="Keep only these fields, comma separated")
    ] = None,
    quality: Annotated[bool, typer.Option(help="Run quality checks after ingest")] = True,
    local: Annotated[
        bool, typer.Option(help="Use the files already in --raw-dir instead of downloading")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Log debug messages")] = False,
) -> None:
    """Download part of a The Well dataset and convert it to transient grid samples"""
    from frontier_data.ingest.the_well import ingest_the_well

    _setup_logging(verbose)
    ds = ingest_the_well(
        name,
        out or Path("datasets") / f"the-well-{name}",
        trajectories=trajectories or None,
        source_splits=[s.strip() for s in source_splits.split(",") if s.strip()],
        per_file=per_file,
        resplit=resplit,
        raw_dir=raw_dir,
        time_stride=time_stride,
        max_steps=max_steps,
        max_bytes=max_gb * 1e9 if max_gb else None,
        fields=[f.strip() for f in fields.split(",")] if fields else None,
        quality=quality,
        download=not local,
        progress=lambda msg, d, t: console.print(f"  [{d}/{t}] {msg}"),
    )
    _ingest_summary(ds)


@ingest_app.command("meshgraphnets")
def ingest_meshgraphnets_cmd(
    name: Annotated[
        str, typer.Argument(help="Name the dataset: cylinder_flow | airfoil | flag_simple | ...")
    ] = ("cylinder_flow"),
    out: Annotated[Path | None, typer.Option(help="Output dataset directory")] = None,
    trajectories: Annotated[
        int, typer.Option(help="Limit the total number of trajectories, split 80/10/10")
    ] = 12,
    train: Annotated[int | None, typer.Option(help="Set the number of train trajectories")] = None,
    val: Annotated[int | None, typer.Option(help="Set the number of val trajectories")] = None,
    test: Annotated[int | None, typer.Option(help="Set the number of test trajectories")] = None,
    raw_dir: Annotated[Path | None, typer.Option(help="Raw download cache")] = None,
    time_stride: Annotated[int, typer.Option(help="Keep every n-th step")] = 1,
    max_steps: Annotated[int | None, typer.Option(help="Keep at most this many steps")] = None,
    quality: Annotated[bool, typer.Option(help="Run quality checks after ingest")] = True,
    local: Annotated[
        bool, typer.Option(help="Use the files already in --raw-dir instead of downloading")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Log debug messages")] = False,
) -> None:
    """Download part of a MeshGraphNets dataset and convert it to transient mesh samples"""
    from frontier_data.ingest.meshgraphnets import ingest_meshgraphnets

    _setup_logging(verbose)
    counts: int | dict[str, int] = trajectories
    if train is not None or val is not None or test is not None:
        counts = {"train": train or 0, "val": val or 0, "test": test or 0}
    ds = ingest_meshgraphnets(
        name,
        out or Path("datasets") / f"meshgraphnets-{name}",
        trajectories=counts,
        raw_dir=raw_dir,
        time_stride=time_stride,
        max_steps=max_steps,
        quality=quality,
        download=not local,
        progress=lambda msg, d, t: console.print(f"  [{d}/{t}] {msg}"),
    )
    _ingest_summary(ds)


@app.command("inspect")
def inspect_cmd(
    dataset: Annotated[Path, typer.Argument(help="Dataset directory")],
    sample: Annotated[str | None, typer.Option(help="Print meta.json of this sample")] = None,
) -> None:
    """Print the manifest summary, splits and a table of samples"""
    from frontier_data.dataset import Dataset

    ds = Dataset(dataset)
    m = ds.manifest
    size = ds.size_bytes()
    console.print(f"[bold]{m.name}[/] ({m.format} v{m.version}) at {ds.root}")
    console.print(
        f"samples: {len(m.samples)}  size: {size / 1e6:.1f} MB  "
        f"fields: {[f.name for f in m.fields]}"
    )
    console.print(f"geometry params: {m.geometry_params}")
    if m.splits:
        console.print(
            f"splits ({(m.split_strategy or {}).get('name')}): "
            + ", ".join(f"{k}={len(v)}" for k, v in m.splits.items())
        )
        if m.split_families:
            console.print(f"split families: {list(m.split_families)}")
    if m.quality:
        rep = json.loads((ds.root / m.quality).read_text())
        console.print(f"quality: {rep['summary']}")
    table = Table(title="samples")
    if m.minor >= 1 and (m.temporal == "transient" or m.sample_kind != "surface"):
        console.print(
            f"kind: {m.sample_kind}  temporal: {m.temporal}  domain: {m.domain}  "
            f"license: {m.license}  hub_id: {m.hub_id}"
        )
        if m.time:
            console.print(
                f"time: up to {m.time.n_steps} steps, dt={m.time.dt} {m.time.units}, "
                f"uniform={m.time.uniform}"
            )
        for col in ("id", "kind", "steps", "points", "cells", "shape", "parameters"):
            table.add_column(col, justify="left" if col in ("id", "parameters") else "right")
        for s in m.samples:
            table.add_row(
                s.id,
                s.sample_kind,
                str(s.n_steps),
                str(s.n_points),
                str(s.n_cells),
                "x".join(map(str, s.shape)) if s.shape else "-",
                ", ".join(f"{k}={v:.4g}" for k, v in s.parameters.items()),
            )
    else:
        for col in ("id", "points", "faces", "lods", "cd", "cl", "slant"):
            table.add_column(col, justify="right" if col != "id" else "left")
        for s in m.samples:
            table.add_row(
                s.id,
                str(s.n_points),
                str(s.n_faces),
                ",".join(map(str, s.lod_levels)),
                f"{s.forces.get('cd', float('nan')):.4f}",
                f"{s.forces.get('cl', float('nan')):.4f}",
                f"{s.geometry_params.get('slant-angle-degrees', float('nan')):.1f}",
            )
    console.print(table)
    if sample:
        console.print_json(json.dumps(ds.meta(sample).model_dump(), default=str))


@app.command("validate")
def validate_cmd(
    dataset: Annotated[Path, typer.Argument(help="Dataset directory")],
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON")] = False,
    shallow: Annotated[bool, typer.Option(help="Skip reading connectivity arrays")] = False,
    strict: Annotated[bool, typer.Option(help="Exit non-zero on warnings too")] = False,
) -> None:
    """Check a dataset against the Frontier dataset format"""
    from frontier_data.validate import validate_dataset

    rep = validate_dataset(dataset, deep=not shallow)
    if as_json:
        print(json.dumps(rep.to_dict(), indent=2))
    else:
        for issue in rep.issues:
            colour = "red" if issue.level == "error" else "yellow"
            console.print(f"[{colour}]{issue.level:7s}[/] {issue.where}: {issue.message}")
        colour = "green" if rep.ok else "red"
        console.print(
            f"[{colour}]{'valid' if rep.ok else 'INVALID'}[/] frontier-dataset v{rep.version}: "
            f"{rep.n_samples} samples, {rep.n_arrays} field arrays, "
            f"{len(rep.errors)} errors, {len(rep.warnings)} warnings"
        )
    if not rep.ok or (strict and rep.warnings):
        raise typer.Exit(code=1)


@app.command("synth")
def synth_cmd(
    out: Annotated[Path, typer.Argument(help="Output dataset directory")],
    kind: Annotated[
        str,
        typer.Option(help="Choose the kind: grid | grid_advection | mesh | surface | point_cloud"),
    ] = "grid",
    samples: Annotated[int, typer.Option(help="Set the number of trajectories")] = 8,
    steps: Annotated[int, typer.Option(help="Set the steps per trajectory")] = 32,
    size: Annotated[int | None, typer.Option(help="Set the grid side or target node count")] = None,
    seed: Annotated[int, typer.Option(help="Set the random seed")] = 0,
) -> None:
    """Write a small synthetic transient dataset"""
    from frontier_data.synthetic import write_synthetic_dataset

    ds = write_synthetic_dataset(out, kind, n_samples=samples, n_steps=steps, seed=seed, size=size)
    console.print(f"wrote {len(ds)} {ds.manifest.sample_kind} trajectories to {ds.root}")


@app.command("quality")
def quality_cmd(
    dataset: Annotated[Path, typer.Argument(help="Dataset directory")],
    html: Annotated[bool, typer.Option(help="Write report.html too")] = True,
) -> None:
    """Run quality checks and write quality/report.json and report.html"""
    from frontier_data.dataset import Dataset
    from frontier_data.quality import run_quality

    _setup_logging(False)
    ds = Dataset(dataset)
    rep = run_quality(
        ds, write_html=html, progress=lambda msg, d, t: console.print(f"  [{d}/{t}] {msg}")
    )
    s = rep["summary"]
    colour = {"pass": "green", "warn": "yellow", "fail": "red"}[s["status"]]
    console.print(
        f"[{colour}]{s['status']}[/] pass={s['pass']} warn={s['warn']} fail={s['fail']} "
        f"dataset={s['dataset_checks']} -> {ds.root / 'quality'}"
    )
    table = Table(title="findings")
    table.add_column("sample")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    for row in rep["samples"]:
        for c in row["checks"]:
            if c["status"] in ("warn", "fail"):
                table.add_row(
                    row["id"], c["check"], c["status"], json.dumps(c["detail"], default=str)[:120]
                )
    for c in rep["dataset_checks"]:
        if c["status"] in ("warn", "fail"):
            d = {k: v for k, v in c["detail"].items() if k != "parameters"}
            table.add_row("<dataset>", c["check"], c["status"], json.dumps(d, default=str)[:120])
    if table.row_count:
        console.print(table)


@app.command("pipeline")
def pipeline_cmd(
    path: Annotated[
        Path | None, typer.Argument(help="Run this pipeline.json (omit with --json or --stdin)")
    ] = None,
    workers: Annotated[int, typer.Option(help="Set the number of worker processes")] = 4,
    json_text: Annotated[
        str | None, typer.Option("--json", help="Run this inline pipeline JSON")
    ] = None,
    stdin: Annotated[
        bool, typer.Option("--stdin", help="Read the pipeline JSON from stdin")
    ] = False,
) -> None:
    """Run a pipeline from a file, --json or --stdin"""
    import sys

    from frontier_data.pipeline import Pipeline, PipelineRunner

    _setup_logging(False)
    if json_text is not None:
        doc = json.loads(json_text)
    elif stdin:
        doc = json.load(sys.stdin)
    elif path is not None:
        doc = json.loads(path.read_text())
    else:
        raise typer.BadParameter("No pipeline given. Pass a pipeline.json path, --json or --stdin.")
    from frontier_data.board_run import is_board_document, run_board

    if is_board_document(doc):
        # a board's run: every step reports its state as events (and may train or evaluate)
        if not run_board(doc, workers=workers):
            raise typer.Exit(code=1)
        return
    pipeline = Pipeline.model_validate(doc)
    result = PipelineRunner(workers=workers, on_event=_Progress()).run(pipeline)
    console.print(f"finished in {result.elapsed_s:.1f}s with {len(result.errors)} errors")
    if result.errors:
        raise typer.Exit(code=1)


@app.command("mesh-lods")
def mesh_lods_cmd(
    dataset: Annotated[Path, typer.Argument(help="Dataset directory (with manifest.json)")],
    levels: Annotated[str, typer.Option(help="Add these LOD levels, comma separated")] = "1,2,3",
    samples: Annotated[
        str | None, typer.Option(help="Process only these sample ids, comma separated")
    ] = None,
    force: Annotated[bool, typer.Option(help="Rebuild levels that already exist")] = False,
) -> None:
    """Add coarser LOD levels to the mesh samples of a dataset"""
    from frontier_data.lod import add_mesh_lods

    _setup_logging(False)
    ids = [s.strip() for s in samples.split(",")] if samples else None
    done = add_mesh_lods(
        dataset, _parse_levels(levels), sample_ids=ids, force=force, log_fn=console.print
    )
    console.print(f"{len(done)} mesh samples have LOD levels")


@app.command("nodes")
def nodes_cmd() -> None:
    """List the built-in pipeline node kinds"""
    import frontier_data.nodes  # noqa: F401
    from frontier_data.pipeline import REGISTRY

    table = Table(title="node kinds")
    for col in ("kind", "role", "scope", "params", "description"):
        table.add_column(col)
    for spec in REGISTRY.values():
        table.add_row(
            spec.kind, spec.role, spec.scope, json.dumps(spec.params, default=str), spec.description
        )
    console.print(table)


@app.command("inputs")
def inputs_cmd(
    out: Annotated[Path, typer.Option("--out", help="Write the input dataset here")],
    parameters: Annotated[
        str | None,
        typer.Option(
            "--parameters", help="A JSON object (one case), a JSON list, or a CSV/JSON file"
        ),
    ] = None,
    geometry: Annotated[
        str | None, typer.Option("--geometry", help="A parametric shape (ahmed)")
    ] = None,
    files: Annotated[
        list[Path] | None, typer.Option("--file", help="A surface file; repeatable")
    ] = None,
    dataset: Annotated[
        Path | None, typer.Option("--dataset", help="Take the shapes of this dataset")
    ] = None,
    ids: Annotated[
        str | None, typer.Option("--ids", help="Samples of the dataset, comma separated")
    ] = None,
    contract: Annotated[
        Path | None,
        typer.Option("--contract", help="JSON file: the model contract the cases are for"),
    ] = None,
    name: Annotated[str, typer.Option("--name", help="Dataset name")] = "input",
    resolution: Annotated[
        int, typer.Option("--resolution", help="Segments along a parametric shape")
    ] = 24,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    """Write typed parameters and geometry as an input dataset for inference"""
    from frontier_data.inputs import write_input

    params: Any = None
    if parameters:
        text = parameters.strip()
        params = json.loads(text) if text.startswith(("{", "[")) else Path(text)
    doc = json.loads(contract.read_text(encoding="utf-8")) if contract else None
    made = write_input(
        out,
        params,
        geometry=geometry,
        geometry_files=list(files) if files else None,
        dataset=dataset,
        sample_ids=[i for i in (ids or "").split(",") if i.strip()] or None,
        contract=doc,
        name=name,
        resolution=resolution,
    )
    manifest = json.loads((made / "manifest.json").read_text(encoding="utf-8"))
    if as_json:
        print(json.dumps({"dataset": str(made), "samples": [s["id"] for s in manifest["samples"]]}))
    else:
        console.print(f"input dataset {made}: {len(manifest['samples'])} case(s)")


def main() -> None:
    """Console entry point: an expected failure ends in one line, not a traceback."""
    from frontier_data.cli_errors import run

    run(app)


if __name__ == "__main__":  # pragma: no cover
    main()

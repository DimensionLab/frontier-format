# SPDX-License-Identifier: Apache-2.0
"""``python -m frontier_data versions …``: init, status, log, diff, revert, checkout, lineage,
gc, tag, approve, export-prov (``docs/spec/data-prep.md`` sections 3 to 5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from frontier_data.versions import VersionStore, export_prov_json, migrate

versions_app = typer.Typer(
    help="List, compare, revert and approve dataset versions", no_args_is_help=True
)
console = Console()

DatasetArg = Annotated[Path, typer.Argument(help="Dataset directory")]
JsonOpt = Annotated[bool, typer.Option("--json", help="Print JSON")]
MessageOpt = typer.Option("-m", "--message", help="Describe the version")


def gate_hint(gate: dict[str, Any], dataset: Path) -> str:
    """What to tell the user when HEAD does not carry the train-ready tag."""
    where = f"the tag is on {gate['tagged']}" if gate["tagged"] else "no version carries the tag"
    return (
        f"HEAD {gate['head']} is not tagged train-ready ({where}); "
        f"run `python -m frontier_prep run prep.quality.gate -d {dataset}`"
    )


def _print(obj: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, indent=2, default=str))
    else:
        console.print_json(json.dumps(obj, default=str))


@versions_app.command("init")
def init_cmd(
    dataset: DatasetArg,
    message: Annotated[str, MessageOpt] = "baseline",
    as_json: JsonOpt = False,
) -> None:
    """Record the baseline version of an existing dataset"""
    v = VersionStore(dataset).init(message)
    _print(
        {"version": v.id, "message": v.message, "stats": v.stats, "tags": v.tags}, as_json
    ) if as_json else console.print(f"[bold]{v.id}[/] {v.message}  {v.stats}")


@versions_app.command("migrate")
def migrate_cmd(
    root: Annotated[Path, typer.Argument(help="Directory that holds datasets")],
    message: Annotated[str, MessageOpt] = "baseline (migration)",
    as_json: JsonOpt = False,
) -> None:
    """Record a baseline version for every dataset under ROOT that has none"""
    rows = migrate(root, message=message)
    if as_json:
        _print(rows, True)
        return
    for r in rows:
        extra = ""
        if r["action"] == "initialized":
            st = r["stats"]
            extra = f"  {st.get('n_files')} files, {st.get('n_bytes', 0) / 1e9:.2f} GB"
        console.print(f"{r['action']:12s} {r['version'] or '-':20s} {r['path']}{extra}")


@versions_app.command("status")
def status_cmd(dataset: DatasetArg, as_json: JsonOpt = False) -> None:
    """Show HEAD and the files changed since it"""
    st = VersionStore(dataset).status()
    if as_json:
        _print(st.to_dict(), True)
        return
    console.print(f"HEAD {st.head}  {'clean' if st.clean else 'dirty'}")
    for kind, paths in (("added", st.added), ("removed", st.removed), ("modified", st.modified)):
        for p in paths[:50]:
            console.print(f"  {kind:8s} {p}")
        if len(paths) > 50:
            console.print(f"  … {len(paths) - 50} more {kind}")


@versions_app.command("commit")
def commit_cmd(
    dataset: DatasetArg,
    message: Annotated[str, MessageOpt] = "",
    as_json: JsonOpt = False,
) -> None:
    """Record changes made outside of ops as a new version"""
    v = VersionStore(dataset).commit(message or "manual commit")
    if v is None:
        console.print("nothing to commit")
        raise typer.Exit(0)
    _print({"version": v.id, "stats": v.stats}, as_json) if as_json else console.print(
        f"[bold]{v.id}[/] {v.message}"
    )


@versions_app.command("log")
def log_cmd(
    dataset: DatasetArg,
    limit: Annotated[int | None, typer.Option("-n", help="Show at most this many")] = None,
    as_json: JsonOpt = False,
) -> None:
    """List versions"""
    vs = VersionStore(dataset)
    rows = []
    for v in vs.log(limit):
        rec = vs.record(v.op_record) if v.op_record else None
        rows.append(
            {
                "version": v.id,
                "parent": v.parent,
                "created_at": v.created_at,
                "message": v.message,
                "op": rec.get("op") if rec else None,
                "actor": v.actor,
                "tags": v.tags,
                "n_new_bytes": v.stats.get("n_new_bytes"),
                "n_files": v.stats.get("n_files"),
            }
        )
    if as_json:
        _print(rows, True)
        return
    table = Table("version", "created", "op", "actor", "tags", "new bytes", "message")
    for r in rows:
        table.add_row(
            r["version"],
            str(r["created_at"])[:19],
            str(r["op"] or ""),
            str(r["actor"].get("name", "")),
            ",".join(r["tags"]),
            str(r["n_new_bytes"]),
            r["message"],
        )
    console.print(table)


@versions_app.command("diff")
def diff_cmd(
    dataset: DatasetArg,
    a: Annotated[str, typer.Option(help="Compare from this version")] = "HEAD~1",
    b: Annotated[str, typer.Option(help="Compare to this version")] = "HEAD",
    arrays: Annotated[bool, typer.Option(help="Scan full arrays for exact change counts")] = True,
    percentile_samples: Annotated[int, typer.Option(
        min=1, max=4_000_000, help="Deterministic sample limit for percentiles only",
    )] = 65536,
    read_engine: Annotated[str, typer.Option(
        help="zarr-python, tensorstore or zarrista",
    )] = "zarr-python",
    read_buffer_mib: Annotated[int, typer.Option(min=1, help="Read-buffer allowance in MiB")] = 128,
    as_json: JsonOpt = False,
) -> None:
    """Show what changed between two versions"""
    d = VersionStore(dataset).diff(
        a, b, arrays=arrays, percentile_samples=percentile_samples,
        read_engine=read_engine, read_buffer_bytes=read_buffer_mib << 20,
    )
    if as_json:
        _print(d, True)
        return
    console.print(f"[bold]{d['a']} -> {d['b']}[/]  files {d['files']}")
    if d["manifest"]:
        console.print("manifest:", d["manifest"])
    s = d["samples"]
    if s["added"] or s["removed"]:
        console.print(f"samples added {s['added']} removed {s['removed']}")
    for sid, entry in s["modified"].items():
        console.print(f"[bold]{sid}[/] files {entry['files']}")
        for name, st in entry.get("arrays", {}).items():
            console.print(f"  {name}: {_short_stats(st)}")
        for key in ("meta", "forces"):
            if key in entry:
                console.print(f"  {key}: {entry[key]}")


def _short_stats(st: dict[str, Any]) -> str:
    if st.get("note"):
        return st["note"]
    parts = [f"changed {st.get('frac_changed', 0):.1%}"]
    if st.get("max_abs_diff") is not None:
        parts.append(f"max|d| {st['max_abs_diff']:.4g}")
    if st.get("rel_l2_diff") is not None:
        parts.append(f"relL2 {st['rel_l2_diff']:.3g}")
    b, a = st.get("before", {}), st.get("after", {})
    if b and a:
        parts.append(f"mean {b.get('mean', 0):.4g} -> {a.get('mean', 0):.4g}")
    return ", ".join(parts)


@versions_app.command("revert")
def revert_cmd(
    dataset: DatasetArg,
    version: Annotated[str, typer.Argument(help="Version id or tag")],
    message: Annotated[str | None, MessageOpt] = None,
    as_json: JsonOpt = False,
) -> None:
    """Create a new version with the content of VERSION, keeping history"""
    store = VersionStore(dataset)
    v = store.revert(version, message)
    gate = store.gate_status()
    hint = None if gate["train_ready"] else gate_hint(gate, dataset)
    if as_json:
        _print({"version": v.id, "message": v.message, "tags": v.tags, **gate, "hint": hint}, True)
        return
    console.print(f"[bold]{v.id}[/] {v.message}")
    if gate["train_ready"]:
        console.print(f"HEAD {v.id} is tagged train-ready")
    else:
        console.print(hint, markup=False, highlight=False, soft_wrap=True)


@versions_app.command("checkout")
def checkout_cmd(
    dataset: DatasetArg, version: Annotated[str, typer.Argument(help="Version id or tag")]
) -> None:
    """Set the working tree and HEAD to VERSION without recording a version"""
    v = VersionStore(dataset).checkout(version)
    console.print(f"HEAD -> {v.id} ({v.message})")


@versions_app.command("lineage")
def lineage_cmd(
    dataset: DatasetArg,
    target: Annotated[
        str, typer.Option(help="Path inside the dataset, e.g. samples/<id>")
    ] = "manifest.json",
    as_json: JsonOpt = False,
) -> None:
    """List the versions and ops that changed a file, sample or field"""
    rows = VersionStore(dataset).lineage(target)
    if as_json:
        _print(rows, True)
        return
    table = Table("version", "created", "op", "+", "-", "~", "message")
    for r in rows:
        table.add_row(
            r["version"],
            str(r["created_at"])[:19],
            str(r["op"] or ""),
            str(r["added"]),
            str(r["removed"]),
            str(r["modified"]),
            r["message"],
        )
    console.print(table)


@versions_app.command("gc")
def gc_cmd(
    dataset: DatasetArg,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print the plan and exit")] = False,
    as_json: JsonOpt = False,
) -> None:
    """Delete stored objects that no version or working file uses"""
    _print(VersionStore(dataset).gc(dry_run=dry_run), as_json or True)


@versions_app.command("tag")
def tag_cmd(
    dataset: DatasetArg,
    name: Annotated[str, typer.Argument(help="Tag name")],
    version: Annotated[str, typer.Option(help="Version id or tag")] = "HEAD",
) -> None:
    """Set a tag on a version"""
    vid = VersionStore(dataset).tag(name, version)
    console.print(f"{name} -> {vid}")


@versions_app.command("approve")
def approve_cmd(
    dataset: DatasetArg,
    approved_by: Annotated[str, typer.Option("--by", help="Name the approver")],
    version: Annotated[str, typer.Option(help="Version id or tag")] = "HEAD",
    decision: Annotated[str, typer.Option(help="Record this decision")] = "approve",
    note: Annotated[str, typer.Option(help="Add a note")] = "",
) -> None:
    """Record an approval or rejection of a version"""
    rid = VersionStore(dataset).approve(version, approved_by, decision, note)
    console.print(f"approval {rid} ({decision}) recorded for {version}")


@versions_app.command("provenance")
def provenance_cmd(
    dataset: DatasetArg,
    kind: Annotated[
        str | None, typer.Option(help="Print only: version | op | assertion | approval | note")
    ] = None,
) -> None:
    """Print the provenance log, one JSON record per line"""
    for rec in VersionStore(dataset).provenance(kind):
        print(json.dumps(rec, default=str))


@versions_app.command("export-prov")
def export_prov_cmd(
    dataset: DatasetArg,
    out: Annotated[Path | None, typer.Option("-o", "--out", help="Write to this file")] = None,
) -> None:
    """Export the provenance log as W3C PROV-JSON"""
    text = json.dumps(export_prov_json(VersionStore(dataset)), indent=2, default=str)
    if out is not None:
        out.write_text(text)
        console.print(f"wrote {out}")
    else:
        print(text)

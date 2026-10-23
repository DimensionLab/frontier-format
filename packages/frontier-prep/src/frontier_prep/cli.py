# SPDX-License-Identifier: Apache-2.0
"""``python -m frontier_prep``: ops, plan/preview/run, recipes, versions
(``docs/spec/data-prep.md`` section 7). Exit codes: 0 ok, 1 error, 2 blocked (a plan has
a ``fail`` warning, the gate fails, or a recipe step fails)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.table import Table

from frontier_data import joblog
from frontier_data.cli_versions import versions_app
from frontier_prep.op import REGISTRY, Actor, Context, Selection, export_tools, op_info

app = typer.Typer(
    help=(
        "Plan, preview and run data-preparation ops on Frontier datasets.\n\n"
        "Exit codes: 0 ok, 1 error, 2 blocked."
    ),
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
ops_app = typer.Typer(help="List and describe ops", no_args_is_help=True)
recipe_app = typer.Typer(help="Run, plan and show saved recipes", no_args_is_help=True)
app.add_typer(ops_app, name="ops")
app.add_typer(recipe_app, name="recipe")
app.add_typer(versions_app, name="versions")
console = joblog.console("frontier_prep")
err = joblog.console("frontier_prep", stderr=True)

OpArg = Annotated[str, typer.Argument(help="Op id, e.g. prep.units.convert")]
DatasetOpt = Annotated[Path, typer.Option("--dataset", "-d", help="Dataset directory")]
ParamsOpt = Annotated[
    str | None, typer.Option("--params", "-p", help="Pass op parameters as JSON or a .json file")
]
SetOpt = Annotated[
    list[str] | None, typer.Option("--set", help="Set one parameter: key=value (JSON value)")
]
JsonOpt = Annotated[bool, typer.Option("--json", help="Print JSON")]
RecipeArg = Annotated[Path, typer.Argument(help="Recipe JSON file")]


def _dump(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _parse_params(params: str | None, sets: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if params:
        p = Path(params)
        try:
            is_file = p.suffix == ".json" and p.exists()
        except OSError:  # inline JSON longer than a file name
            is_file = False
        out = json.loads(p.read_text()) if is_file else json.loads(params)
    for item in sets or []:
        if "=" not in item:
            raise typer.BadParameter(f"--set expects key=value, got {item!r}")
        key, _, raw = item.partition("=")
        try:
            value: Any = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        cur = out
        parts = key.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value
    return out


# -- ops -------------------------------------------------------------------------------------


@ops_app.command("list")
def ops_list(
    as_json: JsonOpt = False,
    category: Annotated[str | None, typer.Option(help="List only this category")] = None,
) -> None:
    """List ops"""
    infos = [i for i in REGISTRY.infos() if category is None or i.category == category]
    if as_json:
        _dump([i.model_dump() for i in infos])
        return
    table = Table("id", "version", "category", "determinism", "summary")
    for i in infos:
        table.add_row(i.id, i.version, i.category, i.determinism, i.summary)
    console.print(table)


@ops_app.command("show")
def ops_show(op_id: OpArg, as_json: JsonOpt = False) -> None:
    """Show the description and parameters of an op"""
    info = op_info(REGISTRY.get(op_id))
    if as_json:
        _dump(info.model_dump())
        return
    console.print(f"[bold]{info.id}[/] v{info.version} ({info.category}, {info.determinism})")
    console.print(info.summary)
    if info.description:
        console.print(info.description)
    if info.when_to_use:
        console.print(f"[dim]When to use:[/] {info.when_to_use}")
    console.print_json(json.dumps(info.params_schema))


@ops_app.command("export-tools")
def ops_export_tools(
    as_json: JsonOpt = True,
    per_mode: Annotated[
        bool, typer.Option("--per-mode", help="Export one tool per op and verb")
    ] = False,
    out: Annotated[Path | None, typer.Option("-o", "--out", help="Write to this file")] = None,
) -> None:
    """Export the ops as tool definitions for an agent"""
    text = json.dumps(export_tools(per_mode=per_mode), indent=2)
    if out is not None:
        out.write_text(text)
        console.print(f"wrote {out}")
    else:
        print(text)


@ops_app.command("vocabulary")
def ops_vocabulary() -> None:
    """Print the canonical field vocabulary as JSON"""
    from frontier_prep.vocabulary import vocabulary_json

    _dump(vocabulary_json())


def _op_and_params(op_id: str, params: str | None, sets: list[str] | None) -> tuple[Any, Any]:
    """The op and its validated params, or a message and exit code 1: an unknown op lists the
    close matches, a wrong parameter names the valid keys (no traceback)."""
    import difflib

    from pydantic import ValidationError

    try:
        op = REGISTRY.get(op_id)
    except KeyError:
        close = difflib.get_close_matches(op_id, REGISTRY.ids(), n=5, cutoff=0.4)
        hint = f"; did you mean: {', '.join(close)}" if close else ""
        err.print(
            f"[red]unknown op {op_id!r}[/red]{hint} (`ops list` shows all {len(REGISTRY.ids())})"
        )
        raise typer.Exit(1) from None
    try:
        raw = _parse_params(params, sets)
    except json.JSONDecodeError as e:
        err.print(f"[red]--params is not valid JSON[/red]: {e.msg} at character {e.pos}")
        raise typer.Exit(1) from None
    try:
        return op, op.Params.model_validate(raw)
    except ValidationError as e:
        fields = op.Params.model_fields
        err.print(f"[red]invalid parameters for {op.id}[/red]")
        for item in e.errors():
            loc = ".".join(str(x) for x in item["loc"]) or "(params)"
            msg = item["msg"]
            if item["type"] == "extra_forbidden":
                close = difflib.get_close_matches(str(item["loc"][-1]), list(fields), n=3)
                msg = "unknown parameter" + (f"; did you mean {', '.join(close)}?" if close else "")
            err.print(f"  {loc}: {msg}")
        err.print("valid parameters: " + ", ".join(sorted(fields)))
        err.print(f"see `python -m frontier_prep ops show {op.id}` for types and defaults")
        raise typer.Exit(1) from None


# -- plan / preview / run --------------------------------------------------------------------


@app.command("plan")
def plan_cmd(
    op_id: OpArg,
    dataset: DatasetOpt,
    params: ParamsOpt = None,
    sets: SetOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """List the changes, warnings and open questions of an op without running it"""
    op, p = _op_and_params(op_id, params, sets)
    plan = op.plan(dataset, p)
    if as_json:
        _dump(plan.model_dump())
    else:
        console.print(
            f"[bold]{plan.op}[/] v{plan.op_version} on {plan.dataset} @ {plan.input_version}"
        )
        console.print(plan.summary)
        for ch in plan.changes[:40]:
            console.print(
                f"  {ch.action:7s} {ch.target}  {ch.before or ''} -> {ch.after or ch.detail or ''}"
            )
        if len(plan.changes) > 40:
            console.print(f"  … {len(plan.changes) - 40} more")
        for w in plan.warnings:
            console.print(
                f"[{'red' if w.severity == 'fail' else 'yellow'}]{w.severity}[/] "
                f"{w.code}: {w.message}"
            )
        for j in plan.judgments:
            console.print(f"[cyan]judgment[/] {j.question} options={j.options} -> param {j.param}")
    if plan.blocking:
        raise typer.Exit(2)


@app.command("preview")
def preview_cmd(
    op_id: OpArg,
    dataset: DatasetOpt,
    params: ParamsOpt = None,
    sets: SetOpt = None,
    samples: Annotated[
        str | None, typer.Option(help="Preview these sample ids, comma separated")
    ] = None,
    max_samples: Annotated[int, typer.Option(help="Preview at most this many samples")] = 3,
    max_steps: Annotated[int, typer.Option(help="Compare at most this many time steps")] = 8,
    percentile_samples: Annotated[int, typer.Option(help="Samples for percentiles only")] = 65_536,
    read_engine: Annotated[
        str, typer.Option(help="zarr-python, tensorstore or zarrista")
    ] = "zarr-python",
    read_buffer_mib: Annotated[
        int, typer.Option(help="Comparison read allowance per array (MiB)")
    ] = 128,
    renders: Annotated[bool, typer.Option(help="Render before and after images")] = True,
    render_dir: Annotated[
        Path | None, typer.Option(help="Write the renders (svg, png) to this directory")
    ] = None,
    as_json: JsonOpt = False,
) -> None:
    """Apply an op to a few samples in a temporary copy and show the differences"""
    op, p = _op_and_params(op_id, params, sets)
    sel = Selection.model_validate({
        "samples": samples.split(",") if samples else None, "max_samples": max_samples,
        "max_steps": max_steps, "renders": renders, "percentile_samples": percentile_samples,
        "read_engine": read_engine, "read_buffer_bytes": read_buffer_mib << 20,
    })
    pv = op.preview(dataset, p, sample=sel)
    if render_dir is not None:
        render_dir.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(pv.renders):
            if r.kind == "svg":
                (render_dir / f"{i:02d}.svg").write_text(r.data)
            else:
                import base64

                (render_dir / f"{i:02d}.png").write_bytes(base64.b64decode(r.data))
    if as_json:
        _dump(pv.model_dump())
        return
    console.print(
        f"[bold]{pv.plan.op}[/] preview on {pv.samples} "
        f"({pv.elapsed_s:.1f}s{', truncated' if pv.truncated else ''})"
    )
    console.print(pv.plan.summary)
    for d in pv.diffs[:40]:
        line = f"  {d.target}: "
        if d.note:
            line += d.note
        else:
            line += f"changed {d.frac_changed or 0:.1%}"
            if d.max_abs_diff is not None:
                line += (f", max|d| {d.max_abs_diff:.4g} "
                         f"{d.difference_units or '(units unspecified)'}")
        if d.before and d.after and "mean" in d.before and "mean" in d.after:
            line += (f", mean {d.before['mean']:.4g} {d.units_before or '(units unspecified)'}"
                     f" -> {d.after['mean']:.4g} {d.units_after or '(units unspecified)'}")
        if d.comparison_scope == "full_selection":
            line += " (selected time window; all spatial values)"
        console.print(line)
    for n in pv.notes:
        console.print(f"[yellow]note[/] {n}")
    if pv.renders:
        console.print(
            f"{len(pv.renders)} render(s)"
            + (f" written to {render_dir}" if render_dir else " (use --render-dir)")
        )


@app.command("run")
def run_cmd(
    op_id: OpArg,
    dataset: DatasetOpt,
    params: ParamsOpt = None,
    sets: SetOpt = None,
    seed: Annotated[int, typer.Option(help="Set the random seed")] = 0,
    workers: Annotated[int, typer.Option(help="Set the number of worker processes")] = 1,
    message: Annotated[str, typer.Option("-m", "--message", help="Describe the version")] = "",
    actor: Annotated[str, typer.Option(help="Record this name as the actor")] = "",
    agent: Annotated[bool, typer.Option(help="Record the actor as an agent")] = False,
    approve: Annotated[str | None, typer.Option(help="Record this name as the approver")] = None,
    allow_dirty: Annotated[
        bool, typer.Option(help="Commit changes made outside of ops first")
    ] = False,
    dry_run: Annotated[bool, typer.Option(help="Print the plan and exit")] = False,
    record: Annotated[Path | None, typer.Option(help="Append this step to a recipe file")] = None,
    explain: Annotated[bool, typer.Option(help="Print what the op did, step by step")] = False,
    as_json: JsonOpt = False,
) -> None:
    """Run an op and record the result as a new dataset version"""
    import os

    op, validated = _op_and_params(op_id, params, sets)
    p = validated.model_dump(mode="json", exclude_unset=True)
    who = Actor(
        kind="agent" if agent else "human",
        name=actor or os.environ.get("USER", ""),
        approved_by=approve,
    )
    ctx = Context(
        seed=seed,
        workers=workers,
        actor=who,
        message=message,
        dry_run=dry_run,
        progress=_progress if not as_json else None,
    )
    if allow_dirty:
        from frontier_data.versions import VersionStore

        vs = VersionStore(dataset)
        if vs.initialized and not vs.status().clean:
            vs.commit("manual changes before " + op_id)
    res = op.run(dataset, p, ctx=ctx)
    if record is not None and not dry_run and not res.noop:
        from frontier_prep.recipes import record_step

        record_step(record, op_id, p, message=message, dataset=str(dataset), actor=who)
    if as_json:
        _dump(
            res.model_dump()
            if not explain
            else {"result": res.model_dump(), "explanation": op.explain(res).model_dump()}
        )
    else:
        console.print(
            f"[bold]{res.op}[/] {res.input_version} -> {res.output_version}  {res.summary}"
        )
        if res.timing:
            console.print(
                f"  {res.timing.elapsed_s:.2f}s (plan {res.timing.plan_s:.2f}, "
                f"apply {res.timing.apply_s:.2f}, commit {res.timing.commit_s:.2f}); "
                f"{res.timing.workers} worker(s); read {res.timing.bytes_read / 1e6:.1f} MB, "
                f"wrote {res.timing.bytes_written / 1e6:.1f} MB"
            )
        counts: dict[str, int] = {}
        for a in res.assertions:
            counts[a.status] = counts.get(a.status, 0) + 1
        if counts:
            console.print(f"  assertions: {counts}")
        for w in res.warnings[:10]:
            console.print(f"  [yellow]{w.severity}[/] {w.code}: {w.message}")
        if explain:
            for step in op.explain(res).steps:
                console.print(f"  - {step}")
    if not res.ok:
        raise typer.Exit(2)


def _progress(done: int, total: int, sid: str) -> None:
    err.print(f"  [{done}/{total}] {sid}", end="\r")


@app.command("explain")
def explain_cmd(
    dataset: DatasetOpt,
    record: Annotated[str, typer.Option(help="Version id or tag")] = "HEAD",
) -> None:
    """Show the op, parameters and assertions that produced a version"""
    from frontier_data.versions import VersionStore

    vs = VersionStore(dataset)
    v = vs.version(record)
    rec = vs.record(v.op_record) if v.op_record else None
    if rec is None:
        console.print(f"{v.id}: no op record ({v.message})")
        return
    console.print(f"[bold]{v.id}[/] {rec.get('op')} v{rec.get('op_version')} by {rec.get('actor')}")
    console.print(rec.get("summary", ""))
    console.print_json(json.dumps(rec.get("params", {}), default=str))
    console.print(f"{rec.get('n_changes', 0)} change(s); determinism {rec.get('determinism')}")
    for a in vs.assertions(v.id)[:30]:
        console.print(f"  assertion {a['check']} {a['status']} {a.get('target')}")


# -- recipes ---------------------------------------------------------------------------------


@recipe_app.command("run")
def recipe_run(
    recipe: RecipeArg,
    dataset: DatasetOpt,
    workers: Annotated[int, typer.Option(help="Set the number of worker processes")] = 1,
    seed: Annotated[int, typer.Option(help="Set the random seed")] = 0,
    as_json: JsonOpt = False,
    result_file: Annotated[
        Path | None,
        typer.Option(help="Save the complete recipe result as JSON, including failures"),
    ] = None,
) -> None:
    """Run every step of a recipe on a dataset"""
    from frontier_prep.recipes import Recipe, run_recipe

    r = Recipe.load(recipe)

    def on_step(sr: Any) -> None:
        if not as_json:
            v = sr.result.output_version if sr.result else "-"
            console.print(f"  [{sr.index}] {sr.op}: {sr.status} -> {v} {sr.error or ''}")

    out = run_recipe(r, dataset, ctx=Context(seed=seed, workers=workers), on_step=on_step)
    if result_file is not None:
        result_file.write_text(out.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if as_json:
        _dump(out.model_dump())
    else:
        console.print(
            f"recipe {out.recipe}: {out.input_version} -> {out.output_version} "
            f"({'ok' if out.ok else 'FAILED'})"
        )
    if not out.ok:
        raise typer.Exit(2)


@recipe_app.command("diff")
def recipe_diff(
    recipe: RecipeArg,
    dataset: DatasetOpt,
    as_json: JsonOpt = False,
) -> None:
    """Plan every step of a recipe on a dataset without running it"""
    from frontier_prep.recipes import Recipe, plan_recipe

    steps = plan_recipe(Recipe.load(recipe), dataset)
    if as_json:
        _dump([s.model_dump() for s in steps])
        return
    for s in steps:
        summary = s.plan.summary if s.plan else s.error
        console.print(f"  [{s.index}] {s.op}: {s.status}  {summary}")


@recipe_app.command("show")
def recipe_show(recipe: RecipeArg) -> None:
    """Print the steps of a recipe"""
    from frontier_prep.recipes import Recipe

    r = Recipe.load(recipe)
    console.print(f"[bold]{r.name}[/] ({len(r.steps)} steps) {r.description}")
    for i, s in enumerate(r.steps):
        console.print(f"  [{i}] {s.op} {json.dumps(s.params)}")


@app.command("adapt")
def adapt_cmd(
    dataset: DatasetOpt,
    out: Annotated[Path, typer.Option("--out", help="Write the adapted dataset here")],
    fixes: Annotated[
        Path, typer.Option("--fixes", help="JSON file: the problems of a compatibility verdict")
    ],
    contract: Annotated[
        Path | None, typer.Option("--contract", help="JSON file: the model contract adapted to")
    ] = None,
    sets: SetOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Adapt a dataset to a model: apply the fixes of a compatibility verdict as a recipe"""
    from frontier_prep.adapt import adapt

    doc = json.loads(fixes.read_text(encoding="utf-8"))
    problems = doc.get("problems", doc) if isinstance(doc, dict) else doc
    values = {}
    for item in sets or []:
        key, _, value = item.partition("=")
        values[key.strip()] = float(value)
    contract_doc = json.loads(contract.read_text(encoding="utf-8")) if contract else {}
    out_dir, run = adapt(dataset, contract_doc, out, fixes=list(problems), values=values or None)
    if as_json:
        _dump(
            {
                "dataset": str(out_dir),
                "ok": run.ok,
                "output_version": run.output_version,
                "steps": [s.model_dump(mode="json") for s in run.steps],
            }
        )
    else:
        for s in run.steps:
            console.print(f"{s.status:8} {s.op}" + (f"  {s.error}" if s.error else ""))
        console.print(f"adapted -> {out_dir} @ {run.output_version}")
    if not run.ok:
        raise typer.Exit(1)


def main() -> None:
    """Console entry point: an expected failure ends in one line, not a traceback."""
    from frontier_data.cli_errors import run

    run(app, code=1)


if __name__ == "__main__":  # pragma: no cover
    main()

# SPDX-License-Identifier: Apache-2.0
"""ArrayRoute comparisons for preparation previews, with physical metadata and time limits."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from fnmatch import fnmatchcase

from arrayroute.difference import compare_array_readers
from arrayroute.plan import Selection as ArraySelection
from arrayroute.reader import ArrayReader
from arrayroute.session import LocalReadSession
from frontier_data.arrays import array_names

from .io import SampleEditor
from .op import Change, FieldDiff, Selection


def _expand_changes(
    before: SampleEditor,
    after: SampleEditor,
    changes: list[Change],
) -> Iterator[Change]:
    for change in changes:
        if change.action == "assert":
            continue
        rel = change.target.removeprefix(f"samples/{before.id}/")
        prefix, sep, name = rel.partition(":")
        if sep and prefix in ("point", "cell", "grid"):
            rel = ("cell_fields/" if prefix == "cell" else "fields/") + name
        if any(c in rel for c in "*?["):
            group, _, pattern = rel.partition("/")
            ed = before if change.action == "remove" else after
            for name in array_names(ed.sample.group[group]):
                if fnmatchcase(name, pattern):
                    yield change.model_copy(update={"target": f"samples/{ed.id}/{group}/{name}"})
        else:
            yield change.model_copy(update={"target": f"samples/{before.id}/{rel}"})


def compare_changes(
    before: SampleEditor,
    after: SampleEditor,
    changes: list[Change],
    sel: Selection,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> list[FieldDiff]:
    """Statistical diffs for every array a change names (``…/fields/<name>``, ``…/points``)."""
    out: list[FieldDiff] = []
    seen: set[str] = set()
    session = LocalReadSession(sel.read_engine, buffer_bytes=sel.read_buffer_bytes)
    try:
        for ch in _expand_changes(before, after, changes):
            target = ch.target
            rel = target.split(f"samples/{before.id}/", 1)[-1] if "samples/" in target else target
            if rel in seen:
                continue
            seen.add(rel)
            old_rel = rel
            if ch.action == "rename" and ch.before and "name" in ch.before:
                old_rel = rel.split("/", 1)[0] + "/" + ch.before["name"]
            a, sa, ua = (
                _read_target(before, old_rel, sel, session)
                if ch.action != "add"
                else (None, None, None)
            )
            b, sb, ub = (
                _read_target(after, rel, sel, session)
                if ch.action != "remove"
                else (None, None, None)
            )
            if a is None and b is None:
                continue
            st = compare_array_readers(
                a,
                b,
                before_selection=sa,
                after_selection=sb,
                percentile_samples=sel.percentile_samples,
                cancelled=cancelled,
            )
            units_changed = a is not None and b is not None and ua != ub
            note = st.get("note")
            if units_changed:
                note = "; ".join(
                    filter(
                        None,
                        [
                            note,
                            "Units changed; physical differences not computed.",
                        ],
                    )
                )
            out.append(
                FieldDiff(
                    target=target,
                    n_before=st.get("n_before"),
                    n_after=st.get("n_after"),
                    n_changed=st.get("n_changed"),
                    n_bitwise_changed=st.get("n_bitwise_changed"),
                    frac_changed=st.get("frac_changed"),
                    max_abs_diff=None if units_changed else st.get("max_abs_diff"),
                    rel_l2_diff=None if units_changed else st.get("rel_l2_diff"),
                    before=st.get("before"),
                    after=st.get("after"),
                    comparison_scope=st["comparison_scope"],
                    selection_before=st.get("selection_before"),
                    selection_after=st.get("selection_after"),
                    source_shape_before=st.get("source_shape_before"),
                    source_shape_after=st.get("source_shape_after"),
                    units_before=ua,
                    units_after=ub,
                    difference_units=ua if not units_changed else None,
                    algorithm=st["algorithm"],
                    metric_precision=st.get("metric_precision"),
                    read_before=st.get("read_before"),
                    read_after=st.get("read_after"),
                    note=note,
                )
            )
        return out
    finally:
        session.clear()


def _read_target(
    ed: SampleEditor,
    rel: str,
    sel: Selection,
    session: LocalReadSession,
) -> tuple[ArrayReader | None, ArraySelection | None, str | None]:
    """Open numeric targets without materialization; metadata/group changes have no diff."""
    s = ed.reopen()
    prefixes = ("fields/", "cell_fields/", "static/", "signals/", "coords/")
    geometry = ("points", "points_t", "faces", "cells", "times", "cell_area", "cell_types")
    if not rel.startswith(prefixes) and rel not in geometry:
        return None, None, None
    # Missing expected arrays must fail, rather than turn corrupt input into an empty diff.
    array = s.group[rel]
    if not hasattr(array, "shape"):
        return None, None, None
    dims = array.attrs.get("dims", [])
    if rel.startswith(("fields/", "cell_fields/")):
        dims = s.dims(rel)
    time_dependent = (
        dims[:1] == ["time"]
        or rel in ("times", "points_t")
        or (rel.startswith("signals/") and array.ndim >= 2)
    )
    selection = (
        (slice(0, sel.max_steps),) + (slice(None),) * (array.ndim - 1) if time_dependent else None
    )
    units = array.attrs.get("units")
    if units is None and rel == "times" and s.meta.time is not None:
        units = s.meta.time.units
    if units is None and rel.startswith(("fields/", "cell_fields/", "static/", "signals/")):
        name = rel.split("/", 1)[1]
        units = next((f.units for f in s.meta.fields if f.name == name), None)
    return session.open(s.store_path, rel), selection, units

# SPDX-License-Identifier: Apache-2.0
"""Optional artifact renderers. Only administrator-installed entry points execute."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import sys
from collections import Counter
from importlib import metadata
from pathlib import Path

from . import (
    IncompatiblePluginError,
    load_entry_point,
    manifest_for_distribution,
    require_compatible,
)

MAX_INPUT = 64 * 1024 * 1024
MAX_OUTPUT = 32 * 1024 * 1024
IDENTIFIER = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}\Z")


def entries():
    return metadata.entry_points(group="frontier.views")


def catalog() -> list[dict]:
    """Read capabilities without importing renderer code, including disabled entries."""
    result = []
    for entry in entries():
        try:
            if entry.dist is None:
                continue
            manifest = manifest_for_distribution(entry.dist) or {}
            plugin = manifest.get("id", "")
            if not IDENTIFIER.fullmatch(plugin) or not IDENTIFIER.fullmatch(entry.name):
                continue
            problems = []
            try:
                require_compatible(entry)
            except IncompatiblePluginError as exc:
                problems.append(str(exc))
            result.append(
                {
                    "plugin": plugin,
                    "view": entry.name,
                    "name": manifest.get("name", plugin),
                    "available": not problems,
                    "problems": problems,
                }
            )
        except (OSError, ValueError, TypeError):
            continue
    counts = Counter((row["plugin"], row["view"]) for row in result)
    for row in result:
        if counts[row["plugin"], row["view"]] > 1:
            row["available"] = False
            row["problems"].append("More than one installed entry point provides this view")
    return sorted(result, key=lambda row: (row["plugin"], row["view"]))


def render(plugin: str, view: str, source: Path) -> dict:
    if not IDENTIFIER.fullmatch(plugin) or not IDENTIFIER.fullmatch(view):
        raise ValueError("Invalid plugin or view identifier")
    matches = []
    for entry in entries():
        if entry.name != view or entry.dist is None:
            continue
        manifest = manifest_for_distribution(entry.dist) or {}
        if manifest.get("id") == plugin:
            matches.append(entry)
    if len(matches) != 1:
        raise ValueError("Install and enable a unique compatible plugin view on this server")
    renderer = load_entry_point(matches[0])
    with source.open("rb") as stream:
        payload = stream.read(MAX_INPUT + 1)
    if len(payload) > MAX_INPUT:
        raise ValueError("Plugin view input exceeds 64 MiB")
    html = renderer(payload)
    if not isinstance(html, str) or len(html.encode("utf-8")) > MAX_OUTPUT:
        raise ValueError("Plugin view must return HTML smaller than 32 MiB")
    return {
        "schema": "frontier-plugin-view/1",
        "plugin": plugin,
        "view": view,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "html": html,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["list", "render"])
    parser.add_argument("--plugin")
    parser.add_argument("--view")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    try:
        # Plugin diagnostics must not corrupt the host's JSON protocol.
        with contextlib.redirect_stdout(sys.stderr):
            if args.command == "list":
                result = {"views": catalog()}
            elif args.plugin and args.view and args.source:
                result = render(args.plugin, args.view, args.source)
            else:
                raise ValueError("Rendering requires plugin, view and source")
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    except Exception as exc:
        parser.exit(2, json.dumps({"error": str(exc)}) + "\n")


if __name__ == "__main__":
    main()

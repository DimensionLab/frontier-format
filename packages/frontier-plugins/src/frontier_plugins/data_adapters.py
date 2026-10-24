# SPDX-License-Identifier: Apache-2.0
"""Discover data engines without importing their optional native runtimes."""

from __future__ import annotations

import platform
from collections import Counter
from importlib import metadata
from typing import Any

from . import load_entry_point, manifest_for_distribution, require_compatible

GROUP = "frontier.data_adapters"
STAGES = frozenset({"read", "write", "transform", "load", "visualize"})
DEVICES = frozenset({"cpu", "cuda"})
SYSTEMS = frozenset({"Linux", "Darwin", "Windows"})


def declaration(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    """Validate per-entry capabilities; installed does not mean GPU-ready."""
    table = manifest.get("data_adapters", {})
    row = table.get(name) if isinstance(table, dict) else None
    if not isinstance(row, dict):
        raise ValueError(f"data_adapters.{name} must declare its capabilities")
    for key, choices in (("stages", STAGES), ("devices", DEVICES), ("platforms", SYSTEMS)):
        values = row.get(key)
        if (not isinstance(values, list) or not values
                or any(not isinstance(v, str) or v not in choices for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"data_adapters.{name}.{key} must be a unique nonempty list from "
                             f"{sorted(choices)}")
    if row.get("determinism") not in {"exact", "fixed-runtime", "statistical"}:
        raise ValueError(f"data_adapters.{name}.determinism must state its replay contract")
    if not isinstance(row.get("summary"), str) or not row["summary"].strip():
        raise ValueError(f"data_adapters.{name}.summary is required")
    requirements = row.get("requires", [])
    if not isinstance(requirements, list) or not all(isinstance(v, str) for v in requirements):
        raise ValueError(f"data_adapters.{name}.requires must list dependency specifiers")
    return dict(row)


def catalog() -> list[dict[str, Any]]:
    """Metadata-only report for the local executor; never probes or initializes CUDA."""
    return _catalog(metadata.entry_points(group=GROUP))


def _catalog(entries: Any) -> list[dict[str, Any]]:
    from . import check_requirement

    rows = []
    for entry in entries:
        row: dict[str, Any] = {"adapter": entry.name, "plugin": None, "available": False,
                               "problems": [], "runtime_verified": False}
        try:
            if entry.dist is None:
                raise ValueError("Data adapters require installed distribution metadata")
            row["distribution"] = entry.dist.name
            row["version"] = entry.dist.version
            row["entry_point"] = entry.value
            manifest = manifest_for_distribution(entry.dist) or {}
            row["plugin"] = manifest.get("id")
            if not isinstance(row["plugin"], str) or not row["plugin"].strip():
                raise ValueError("A plugin id is required")
            capabilities = declaration(manifest, entry.name)
            row["capabilities"] = capabilities
            api = manifest.get("api", {})
            if not isinstance(api, dict) or "data_adapters" not in api:
                raise ValueError("api.data_adapters must declare a supported interface range")
            row["api"] = dict(api)
            require_compatible(entry)
            if platform.system() not in capabilities["platforms"]:
                raise ValueError(f"Unsupported executor platform: {platform.system()}")
            for required in capabilities.get("requires", []):
                checked = check_requirement(required)
                if not checked["ok"]:
                    row["problems"].append(checked["error"])
            row["available"] = not row["problems"]
        except (ValueError, OSError, ImportError, RuntimeError, TypeError) as exc:
            row["problems"].append(str(exc))
        rows.append(row)
    counts = Counter((r["plugin"], r["adapter"]) for r in rows)
    for row in rows:
        if counts[row["plugin"], row["adapter"]] > 1:
            row["available"] = False
            row["problems"].append("Ambiguous installed adapter; remove the duplicate provider")
    return sorted(rows, key=lambda r: (r["plugin"] or "", r["adapter"]))


def load(plugin: str, adapter: str, *, stage: str, device: str = "cpu") -> Any:
    """Activate one compatible provider; use activate() to also retain its provenance."""
    return activate(plugin, adapter, stage=stage, device=device)[0]


def activate(
    plugin: str, adapter: str, *, stage: str, device: str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """Import only the selected, enabled adapter after capability/dependency checks.

    Factories are trusted administrator-installed code, not a sandbox. A source URI
    never supplies an executable module path. Hardware checks belong to the factory
    on the selected executor, and must fail rather than silently change devices.
    """
    entries = [entry for entry in metadata.entry_points(group=GROUP) if entry.name == adapter]
    matches = [row for row in _catalog(entries) if row["plugin"] == plugin]
    if len(matches) != 1:
        raise ValueError("Install a unique data adapter with this plugin and adapter id")
    row = matches[0]
    if not row["available"]:
        raise ValueError("; ".join(row["problems"]))
    if stage not in row["capabilities"]["stages"] or device not in row["capabilities"]["devices"]:
        raise ValueError(f"{plugin}/{adapter} does not support {stage} on {device}")
    for entry in entries:
        if entry.name == adapter and entry.dist is not None:
            manifest = manifest_for_distribution(entry.dist) or {}
            if manifest.get("id") == plugin:
                if entry.dist.version != row["version"] or entry.value != row["entry_point"]:
                    raise ValueError("Installed adapter changed during activation; retry discovery")
                return load_entry_point(entry), {
                    key: row[key] for key in
                    ("plugin", "adapter", "distribution", "version", "entry_point", "api",
                     "capabilities")
                }
    raise ValueError("Installed adapters changed during discovery; retry discovery")

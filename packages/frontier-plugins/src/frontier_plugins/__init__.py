# SPDX-License-Identifier: Apache-2.0
"""Public compatibility contract shared by Frontier's Python extension loaders.

Discovery reads metadata and TOML without importing the extension. API versions here
describe public interfaces, not the Frontier application or the plugin's release.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import tomllib
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

__version__ = "1.3.0"

API_VERSIONS: dict[str, str] = dict.fromkeys(
    (
        "connectors",
        "importers",
        "exporters",
        "ops",
        "benchmarks",
        "geometry",
        "converters",
        "hosts",
        "models",
        "profilers",
        "views",
        "data_adapters",
    ),
    "1.0.0",
)
ENTRY_POINT_APIS = {
    "frontier.connectors": "connectors",
    "frontier.importers": "importers",
    "frontier.exporters": "exporters",
    "frontier.ops": "ops",
    "frontier_eval.benchmarks": "benchmarks",
    "frontier_data.geometry": "geometry",
    "frontier_hub.converters": "converters",
    "frontier_deploy.hosts": "hosts",
    "frontier_train.models": "models",
    "frontier.profilers": "profilers",
    "frontier.views": "views",
    "frontier.data_adapters": "data_adapters",
}
log = logging.getLogger("frontier.plugins")


@dataclass
class Compatibility:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def compatible(self) -> bool:
        return not self.errors


class IncompatiblePluginError(RuntimeError):
    pass


def check_api(
    required: Any, *, available: dict[str, str] | None = None, family: str | None = None
) -> Compatibility:
    """Check PEP 440 ranges before loading code, including unknown required interfaces."""
    result = Compatibility()
    available = API_VERSIONS if available is None else available
    if required is None or required == {}:
        result.warnings.append("No API compatibility declared; using the legacy v1 interface.")
        return result
    if not isinstance(required, dict):
        result.errors.append("api must be a table of API names and version ranges")
        return result
    if family and family not in required:
        result.errors.append(f"api.{family} must declare the interface this entry point uses")
    for name, spec in required.items():
        if not isinstance(spec, str) or not spec.strip():
            result.errors.append(f"api.{name} must be a nonempty version range")
            continue
        try:
            accepted = SpecifierSet(spec)
        except InvalidSpecifier:
            result.errors.append(f"api.{name} has an invalid version range: {spec}")
            continue
        current = available.get(name)
        if current is None:
            result.errors.append(f"Required API {name} is not available")
        elif not accepted.contains(current):
            result.errors.append(f"Requires {name} API {spec}; this host provides {current}")
    return result


def check_requirement(text: str) -> dict[str, Any]:
    """Validate an installed dependency, respecting environment markers and ranges."""
    row: dict[str, Any] = {"requirement": text, "installed": None, "ok": False}
    try:
        req = Requirement(text)
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            return row | {"ok": True, "skipped": True}
        installed = metadata.version(req.name)
        row["installed"] = installed
        row["ok"] = not req.specifier or req.specifier.contains(installed)
        if not row["ok"]:
            row["error"] = f"Requires {text}; installed {installed}"
    except metadata.PackageNotFoundError:
        row["error"] = f"Missing requirement {text}"
    except (InvalidRequirement, InvalidVersion, ValueError) as exc:
        row["error"] = f"Invalid requirement metadata {text}: {exc}"
    return row


def same_version(left: str, right: str) -> bool:
    try:
        return Version(left) == Version(right)
    except InvalidVersion:
        return False


def manifest_for_distribution(dist: Any) -> dict[str, Any] | None:
    """Find the manifest advertised by the distribution, without importing its package."""
    entries = [e for e in dist.entry_points if e.group == "frontier.plugins"]
    if not entries:
        return None
    manifests = []
    for ep in entries:
        module = ep.value.split(":", 1)[0].strip()
        # Resolve only the top-level package; find_spec on a dotted module imports parents.
        top, *tail = module.split(".")
        spec = importlib.util.find_spec(top)
        roots = list(spec.submodule_search_locations or []) if spec else []
        path = next(
            (
                Path(root).joinpath(*tail, "plugin.toml")
                for root in roots
                if Path(root).joinpath(*tail, "plugin.toml").is_file()
            ),
            None,
        )
        if path is None:
            raise ValueError(f"Manifest for {module} is missing")
        manifests.append(tomllib.loads(path.read_text()))
    if len(manifests) != 1:
        raise ValueError("A distribution must advertise one Frontier plugin manifest")
    return manifests[0]


def check_entry_point(ep: Any) -> Compatibility:
    """Shared preflight for all extension families; no plugin code is executed."""
    dist = getattr(ep, "dist", None)
    if dist is None:
        return Compatibility(warnings=["Entry point has no distribution metadata (legacy API)."])
    try:
        doc = manifest_for_distribution(dist)
        result = check_api((doc or {}).get("api"), family=ENTRY_POINT_APIS.get(ep.group))
        if doc and not same_version(str(doc.get("version", "")), dist.version):
            result.errors.append(
                "Plugin manifest version differs from its installed package version"
            )
        declared = (doc or {}).get("requires", [])
        if not isinstance(declared, list) or not all(isinstance(r, str) for r in declared):
            result.errors.append("requires must be a list of dependency specifiers")
            return result
        requirements: list[str] = [*(dist.requires or []), *declared]
        for requirement in dict.fromkeys(requirements):
            row = check_requirement(requirement)
            if not row["ok"]:
                result.errors.append(row["error"])
        return result
    except (ValueError, OSError, ImportError, AttributeError) as exc:
        return Compatibility(errors=[f"Cannot read plugin metadata: {exc}"])


def require_compatible(ep: Any) -> None:
    result = check_entry_point(ep)
    if not result.compatible:
        raise IncompatiblePluginError(f"{ep.name}: " + "; ".join(result.errors))
    for warning in result.warnings:
        log.debug("%s: %s", ep.name, warning)
    state_file = (
        Path(os.environ.get("FRONTIER_HOME", str(Path.home() / ".frontier"))) / "plugins.toml"
    )
    if state_file.is_file():
        state = tomllib.loads(state_file.read_text())
        disabled = state.get("disabled", {})
        dist = getattr(ep, "dist", None)
        doc = manifest_for_distribution(dist) if dist else None
        plugin_id = (doc or {}).get("id") or getattr(dist, "metadata", {}).get("Name", "")
        family = ENTRY_POINT_APIS.get(ep.group, "")
        if plugin_id in disabled.get("plugins", []) or ep.name in disabled.get(family, []):
            raise IncompatiblePluginError(f"{ep.name}: plugin is disabled")


def load_entry_point(ep: Any) -> Any:
    require_compatible(ep)
    return ep.load()

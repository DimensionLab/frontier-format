# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from frontier_plugins import (
    IncompatiblePluginError,
    check_api,
    check_entry_point,
    check_requirement,
    load_entry_point,
    same_version,
)


@pytest.mark.parametrize("version", ["1.0.0", "1.4.2", "1.99.0"])
def test_one_plugin_supports_multiple_host_api_releases(version):
    assert check_api({"connectors": ">=1,<2"}, available={"connectors": version}).compatible


def test_breaking_api_and_unknown_interfaces_are_rejected():
    assert not check_api({"connectors": ">=2,<3"}).compatible
    assert not check_api({"unknown": ">=1"}).compatible
    assert not check_api({"connectors": "not-a-range"}).compatible
    assert not check_api({"connectors": ""}).compatible
    assert not check_api([]).compatible


def test_declared_manifest_must_cover_the_loaded_interface():
    assert not check_api({"connectors": ">=1,<2"}, family="hosts").compatible


def test_legacy_plugins_remain_usable_with_diagnostic():
    result = check_api({})
    assert result.compatible and result.warnings


def test_dependency_range_and_environment_marker(monkeypatch):
    monkeypatch.setattr("frontier_plugins.metadata.version", lambda name: "1.4")
    assert check_requirement("some-package>=1,<2")["ok"]
    assert not check_requirement("some-package>=2")["ok"]
    assert check_requirement('some-package>=2; python_version < "2"')["skipped"]
    assert not check_requirement("bad requirement !")["ok"]


def test_package_version_normalization():
    assert same_version("1.0", "1.0.0")
    assert not same_version("1.0", "2.0")
    assert not same_version("1.0garbage", "1.0garbage")


def test_incompatible_plugin_is_never_imported(tmp_path, monkeypatch):
    pkg = tmp_path / "community_plugin"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('raise RuntimeError("must not import during discovery")')
    (pkg / "plugin.toml").write_text('version = "17.3.0"\n[api]\nconnectors = ">=2,<3"\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    dist = SimpleNamespace(
        version="17.3.0",
        requires=[],
        entry_points=[SimpleNamespace(group="frontier.plugins", value="community_plugin")],
    )
    ep = SimpleNamespace(name="community", group="frontier.connectors", dist=dist, load=Mock())
    assert not check_entry_point(ep).compatible
    assert "community_plugin" not in sys.modules
    with pytest.raises(IncompatiblePluginError, match="API"):
        load_entry_point(ep)
    ep.load.assert_not_called()
    (pkg / "plugin.toml").write_text('version = "17.3.0"\n[api]\nconnectors = ">=1,<2"\n')
    assert check_entry_point(ep).compatible
    load_entry_point(ep)
    ep.load.assert_called_once()


@pytest.mark.parametrize(
    "family,group",
    [
        ("ops", "frontier.ops"),
        ("benchmarks", "frontier_eval.benchmarks"),
        ("geometry", "frontier_data.geometry"),
        ("converters", "frontier_hub.converters"),
        ("hosts", "frontier_deploy.hosts"),
    ],
)
def test_disabled_extensions_never_activate(tmp_path, monkeypatch, family, group):
    monkeypatch.setenv("FRONTIER_HOME", str(tmp_path))
    (tmp_path / "plugins.toml").write_text(f'[disabled]\n{family} = ["community"]\n')
    ep = SimpleNamespace(name="community", group=group, dist=None, load=Mock())
    with pytest.raises(IncompatiblePluginError, match="disabled"):
        load_entry_point(ep)
    ep.load.assert_not_called()

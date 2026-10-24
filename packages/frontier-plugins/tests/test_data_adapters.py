# SPDX-License-Identifier: Apache-2.0
import platform
from importlib import metadata
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from frontier_plugins import IncompatiblePluginError, data_adapters


@pytest.fixture
def installed(monkeypatch):
    entry = SimpleNamespace(name="reader", dist=object())
    manifest = {"id": "community-data", "api": {"data_adapters": ">=1,<2"},
                "data_adapters": {"reader": {
        "summary": "Example reader", "stages": ["read"], "devices": ["cpu"],
        "platforms": ["Linux", "Darwin", "Windows"], "determinism": "exact",
    }}}
    monkeypatch.setattr(metadata, "entry_points", lambda **_: [entry])
    monkeypatch.setattr(data_adapters, "manifest_for_distribution", lambda _: manifest)
    monkeypatch.setattr(data_adapters, "require_compatible", lambda _: None)
    return entry, manifest


def test_discovery_does_not_import_and_does_not_claim_gpu_readiness(installed, monkeypatch):
    loader = Mock(side_effect=AssertionError("Code imported during discovery"))
    monkeypatch.setattr(data_adapters, "load_entry_point", loader)
    result = data_adapters.catalog()[0]
    assert result["available"] and not result["runtime_verified"]
    with pytest.raises(ValueError, match="does not support"):
        data_adapters.load("community-data", "reader", stage="read", device="cuda")
    loader.assert_not_called()


def test_explicit_selection_loads_only_matching_entry(installed, monkeypatch):
    factory = Mock()
    loader = Mock(return_value=factory)
    monkeypatch.setattr(data_adapters, "load_entry_point", loader)
    assert data_adapters.load("community-data", "reader", stage="read") is factory
    loader.assert_called_once_with(installed[0])


@pytest.mark.parametrize("failure", ["duplicate", "disabled", "dependency", "platform"])
def test_unsupported_adapters_fail_closed(installed, monkeypatch, failure):
    entry, manifest = installed
    if failure == "duplicate":
        monkeypatch.setattr(metadata, "entry_points", lambda **_: [entry, entry])
    elif failure == "disabled":
        def disabled(_):
            raise IncompatiblePluginError("plugin is disabled")
        monkeypatch.setattr(data_adapters, "require_compatible", disabled)
    elif failure == "dependency":
        manifest["data_adapters"]["reader"]["requires"] = ["missing-frontier-test-engine>=1"]
    else:
        monkeypatch.setattr(platform, "system", lambda: "Unsupported")
    loader = Mock()
    monkeypatch.setattr(data_adapters, "load_entry_point", loader)
    assert all(not row["available"] for row in data_adapters.catalog())
    with pytest.raises(ValueError):
        data_adapters.load("community-data", "reader", stage="read")
    loader.assert_not_called()


@pytest.mark.parametrize("key,value", [("stages", []), ("devices", ["cuda", "cuda"]),
                                      ("platforms", "Windows"), ("determinism", "always"),
                                      ("requires", [12])])
def test_invalid_declaration(installed, key, value):
    installed[1]["data_adapters"]["reader"][key] = value
    assert not data_adapters.catalog()[0]["available"]


def test_adapter_requires_explicit_interface_range(installed):
    installed[1].pop("api")
    row = data_adapters.catalog()[0]
    assert not row["available"]
    assert "api.data_adapters" in row["problems"][0]

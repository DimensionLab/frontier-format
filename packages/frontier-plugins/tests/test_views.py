# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from frontier_plugins import views


@pytest.fixture
def installed_view(monkeypatch):
    entry = SimpleNamespace(name="timeline", dist=object())
    monkeypatch.setattr(views, "entries", lambda: [entry])
    monkeypatch.setattr(views, "manifest_for_distribution", lambda _: {"id": "community-profiler"})
    monkeypatch.setattr(views, "require_compatible", lambda _: None)
    return entry


def test_catalog_does_not_import_renderers(installed_view, monkeypatch):
    load = Mock(side_effect=AssertionError("discovery imported executable code"))
    monkeypatch.setattr(views, "load_entry_point", load)
    assert views.catalog() == [
        {
            "plugin": "community-profiler",
            "view": "timeline",
            "name": "community-profiler",
            "available": True,
            "problems": [],
        }
    ]
    load.assert_not_called()


def test_ambiguous_views_are_unavailable_in_catalog(installed_view, monkeypatch):
    monkeypatch.setattr(views, "entries", lambda: [installed_view, installed_view])
    rows = views.catalog()
    assert len(rows) == 2
    assert all(not row["available"] and row["problems"] for row in rows)


def test_generic_view_accepts_bytes_and_returns_identity(installed_view, monkeypatch, tmp_path):
    renderer = Mock(return_value="<p>Recorded evidence</p>")
    monkeypatch.setattr(views, "load_entry_point", lambda _: renderer)
    source = tmp_path / "artifact"
    source.write_bytes(b"abc")
    result = views.render("community-profiler", "timeline", source)
    renderer.assert_called_once_with(b"abc")
    assert result["source_sha256"] == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert result["schema"] == "frontier-plugin-view/1"


@pytest.mark.parametrize("output", [b"not a string", None, "\u00e9" * 6])
def test_output_limit_counts_utf8_bytes(installed_view, monkeypatch, tmp_path, output):
    monkeypatch.setattr(views, "load_entry_point", lambda _: lambda payload: output)
    monkeypatch.setattr(views, "MAX_OUTPUT", 10)
    source = tmp_path / "artifact"
    source.write_bytes(b"abc")
    with pytest.raises(ValueError, match="must return HTML"):
        views.render("community-profiler", "timeline", source)

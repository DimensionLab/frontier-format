# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import zarr

from arrayroute.session import LocalReadSession
from frontier_data.dataset import DatasetWriter
from frontier_data.grid_export import export_predictions
from frontier_data.prediction_view import PredictionView
from frontier_data.predictions import write_transient_prediction


@pytest.fixture
def case(tmp_path):
    source, predictions = tmp_path / "data", tmp_path / "predictions"
    truth = np.arange(6 * 4 * 5 * 3 * 3, dtype=np.float32).reshape(6, 4, 5, 3, 3)
    truth[0, 0, 0, 0, 0] = -0.0
    times = np.arange(6, dtype=np.float64) * .25 + 10
    writer = DatasetWriter(source, "flow")
    writer.write_grid_sample(
        "a", fields={"u": truth, "p": truth[..., :1]}, times=times,
        spacing=[.5, 1, 2], origin=[-3, 8, 9], field_units={"u": "m/s", "p": "Pa"},
    )
    writer.finalize()
    write_transient_prediction(
        predictions, "a", times=times, pred={"u": truth + 2, "p": truth[..., :1] + 3},
        true={"u": truth, "p": truth[..., :1]}, spacing=[.5, 1, 2], origin=[-3, 8, 9],
        field_units={"u": "m/s", "p": "Pa"},
        info={"dataset_manifest_sha256": hashlib.sha256(
            (source / "manifest.json").read_bytes()).hexdigest()},
    )
    return source, predictions, truth, times


@pytest.mark.parametrize("engine", ["zarr-python", "tensorstore", "zarrista"])
def test_selected_regions_exact_across_engines_without_npz(case, engine, monkeypatch):
    _, predictions, truth, times = case
    (predictions / "a.npz").unlink()
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("NPZ access"))
    view = PredictionView(predictions, "a", session=LocalReadSession(engine))
    assert view.units("u") == "m/s" and view.time_units == "s"
    assert view.time(-1) == times[-1]
    for region in [(slice(1, 6, 2), slice(None), 2, slice(0, 3, 2), 1),
                   (0, slice(None), slice(None), slice(None), slice(None)),
                   (slice(0, 0), slice(None), slice(None), slice(None), slice(None))]:
        actual = view.read("u", "true", selection=region)
        expected = truth[region]
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()


def test_read_budget_and_cancel_precede_payload(case, monkeypatch):
    _, predictions, _, _ = case
    view = PredictionView(predictions, "a", session=LocalReadSession(buffer_bytes=64))
    reader = view.reader("u")
    monkeypatch.setattr(reader, "_execute", lambda _: pytest.fail("payload access"))
    with pytest.raises(MemoryError):
        view.read("u")
    with pytest.raises(InterruptedError):
        view.read("u", cancelled=lambda: True)
    assert view.stats.requests == 0


def test_stream_reconstructs_strided_selection_and_cancels_between_tiles(tmp_path):
    root = tmp_path / "predictions"
    root.mkdir()
    g = zarr.open_group(str(root / "a.zarr"), mode="w")
    g.attrs.update(sample_kind="grid", time_units="s")
    data = np.arange(32 * 16 * 16, dtype=np.float32).reshape(32, 16, 16, 1)
    g.create_array("times", data=np.arange(32, dtype=np.float64), chunks=(4,))
    for role in ["pred", "true"]:
        g.create_array(f"fields/u_{role}", data=data, chunks=(1, 4, 4, 1))
    view = PredictionView(root, "a", session=LocalReadSession(buffer_bytes=512))
    selection = (slice(1, 32, 3), slice(None), slice(0, 16, 2), slice(None))
    expected = data[selection]
    actual = np.empty_like(expected)
    with pytest.raises(MemoryError):
        view.read("u", selection=selection)
    for tile, values in view.stream("u", selection=selection):
        actual[tile.destination] = values
    assert actual.tobytes() == expected.tobytes()
    stopped = False
    stream = view.stream("u", cancelled=lambda: stopped)
    next(stream)
    stopped = True
    with pytest.raises(InterruptedError):
        next(stream)


def test_export_actual_vtk_values_masks_coordinates_units_and_sparse_times(case, tmp_path):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkCommonCore import vtkDataArray
    from vtkmodules.vtkIOXML import vtkXMLImageDataReader

    source, predictions, truth, times = case
    g = zarr.open_group(str(predictions / "a.zarr"), mode="a")
    mask = np.ones(truth.shape, np.uint8)
    mask[:, 1, 2, 0] = 0
    g.create_array("fields/u_valid", data=mask, chunks=g["fields/u_true"].chunks)
    (predictions / "a.npz").unlink()
    result = export_predictions(
        predictions, source, "a", tmp_path / "vtk", fields=["u"], frames=slice(1, 6, 2),
    )
    entries = ET.parse(result).findall("./Collection/DataSet")
    assert [float(e.attrib["timestep"]) for e in entries] == list(times[1::2])
    assert [e.attrib["file"] for e in entries] == [f"frame-{i:05d}.vti" for i in [1, 3, 5]]
    reader = vtkXMLImageDataReader()
    reader.SetFileName(str(result.parent / entries[0].attrib["file"]))
    reader.Update()
    image = reader.GetOutput()
    assert image.GetOrigin() == (-3, 8, 9) and image.GetSpacing() == (.5, 1, 2)
    assert image.GetPointData().GetArray("p_prediction") is None
    for suffix, expected in [("reference", truth[1]), ("prediction", truth[1] + 2),
                             ("error", np.full_like(truth[1], 2))]:
        expected = np.where(mask[1], expected, np.nan)
        array = image.GetPointData().GetArray(f"u_{suffix}")
        np.testing.assert_array_equal(vtk_to_numpy(array),
                                      expected.transpose(2, 1, 0, 3).reshape(-1, 3))
        assert array.GetInformation().Get(vtkDataArray.UNITS_LABEL()) == "m/s"


@pytest.mark.parametrize("change,match", [
    ({"dataset_manifest_sha256": "wrong"}, "dataset revision"),
    ({"validity_file": "a.validity.npz"}, "validity mask"),
])
def test_export_rejects_wrong_provenance_or_missing_mask(case, tmp_path, change, match):
    source, predictions, _, _ = case
    path = predictions / "a.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **change}))
    with pytest.raises(ValueError, match=match):
        export_predictions(predictions, source, "a", tmp_path / "vtk")
    assert not (tmp_path / "vtk").exists()
    assert not list(tmp_path.glob(".arrayroute-*"))


@pytest.mark.parametrize("failure", ["cancel", "write", "race"])
def test_export_never_publishes_partial_output(case, tmp_path, monkeypatch, failure):
    from frontier_data import grid_export

    source, predictions, _, _ = case
    target = tmp_path / "vtk"
    calls = []
    original = grid_export.write_vti

    def write(*args, **kwargs):
        original(*args, **kwargs)
        calls.append(1)
        if failure == "write":
            raise OSError("test writer failure")
        if failure == "race" and len(calls) == 1:
            target.mkdir()

    monkeypatch.setattr(grid_export, "write_vti", write)
    with pytest.raises((InterruptedError, OSError)):
        export_predictions(predictions, source, "a", target,
                           cancelled=lambda: failure == "cancel" and bool(calls))
    assert not list(tmp_path.glob(".arrayroute-*"))
    if failure == "race":
        assert target.is_dir() and not list(target.iterdir())
    else:
        assert not target.exists()


def test_export_admission_before_any_payload(case, tmp_path, monkeypatch):
    source, predictions, _, _ = case
    monkeypatch.setattr(PredictionView, "read", lambda *a, **kw: pytest.fail("payload read"))
    with pytest.raises(MemoryError):
        export_predictions(predictions, source, "a", tmp_path / "vtk", buffer_bytes=128)
    assert not (tmp_path / "vtk").exists()


@pytest.mark.parametrize("sid", ["../a", "a/b", "a\\b", "", ".", ".."])
def test_unsafe_ids_rejected(tmp_path, sid):
    with pytest.raises(ValueError, match="safe path"):
        PredictionView(tmp_path, sid)


def test_cli_frame_selection_and_actionable_failure(case, tmp_path, monkeypatch, capsys):
    import sys

    from frontier_data.grid_export import main

    source, predictions, _, times = case
    out = tmp_path / "vtk"
    argv = ["frontier-prediction-export", "--predictions", str(predictions),
            "--dataset", str(source), "--sample", "a", "--out", str(out),
            "--fields", "u", "--start", "2", "--stop", "3"]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    assert str(out / "flow.pvd") in capsys.readouterr().out
    entries = ET.parse(out / "flow.pvd").findall("./Collection/DataSet")
    assert len(entries) == 1 and float(entries[0].attrib["timestep"]) == times[2]
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "Refusing to replace" in capsys.readouterr().err


def test_prediction_writer_preserves_validity_in_both_consumers(case, tmp_path):
    source, predictions, truth, times = case
    masks = np.ones(truth.shape, bool)
    masks[:, 1] = False
    info = json.loads((predictions / "a.json").read_text())
    write_transient_prediction(
        predictions, "a", times=times, pred={"u": truth + 2}, true={"u": truth},
        validity={"u": masks}, info=info, spacing=[.5, 1, 2], origin=[-3, 8, 9],
        field_units={"u": "m/s"},
    )
    view = PredictionView(predictions, "a")
    np.testing.assert_array_equal(view.read("u", "valid"), masks)
    assert view.units("u", "valid") == "1"
    with np.load(predictions / "a.validity.npz") as old:
        np.testing.assert_array_equal(old["u"], masks)
    export_predictions(predictions, source, "a", tmp_path / "vtk")


@pytest.mark.parametrize("mask", [np.ones((2, 3)), np.full((6, 4, 5, 3, 3), 2)])
def test_invalid_mask_rejected_before_writing_predictions(tmp_path, mask):
    truth = np.ones((6, 4, 5, 3, 3), np.float32)
    with pytest.raises(ValueError, match="validity"):
        write_transient_prediction(tmp_path, "a", times=np.arange(6),
                                   pred={"u": truth}, true={"u": truth}, validity={"u": mask})
    assert not (tmp_path / "a.npz").exists()


@pytest.mark.parametrize("engine", ["zarr-python", "tensorstore", "zarrista"])
def test_rollout_error_curves_without_npz(case, engine, monkeypatch):
    _, predictions, truth, _ = case
    (predictions / 'a.npz').unlink()
    monkeypatch.setattr(np, 'load', lambda *a, **kw: pytest.fail('NPZ access'))
    view = PredictionView(predictions, 'a', session=LocalReadSession(engine))
    result = view.error_curves('u')
    np.testing.assert_allclose(result['rmse'], 2 * np.sqrt(3), rtol=1e-14)
    expected = np.linalg.norm(np.full_like(truth, 2).reshape(6, -1), axis=1) / np.linalg.norm(
        truth.reshape(6, -1), axis=1)
    np.testing.assert_allclose(result['l2_rel'], expected, rtol=1e-6)
    with pytest.raises(InterruptedError):
        view.error_curves('u', cancelled=lambda: True)


def test_rollout_scoring_requires_declared_validity(case):
    _, predictions, _, _ = case
    info = predictions / 'a.json'
    data = json.loads(info.read_text())
    data['validity_fields'] = ['u']
    info.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='validity'):
        PredictionView(predictions, 'a').error_curves('u')


def test_rollout_scoring_uses_stored_masks(tmp_path):
    truth = np.ones((3, 7, 9, 2, 3), dtype=np.float32)
    pred = truth + np.array([3, 4, 0], dtype=np.float32)
    valid = np.ones(truth.shape, bool)
    valid[:, :4] = False
    truth[~valid], pred[~valid] = np.nan, np.inf
    write_transient_prediction(tmp_path, 'mask', times=np.arange(3, dtype=float),
                               pred={'u': pred}, true={'u': truth}, validity={'u': valid},
                               field_units={'u': 'm/s'})
    (tmp_path / 'mask.npz').unlink()
    (tmp_path / 'mask.validity.npz').unlink()
    result = PredictionView(tmp_path, 'mask').error_curves('u')
    np.testing.assert_array_equal(result['rmse'], [5, 5, 5])
    assert result['finite'].all()

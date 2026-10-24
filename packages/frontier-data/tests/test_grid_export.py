# SPDX-License-Identifier: Apache-2.0
import numpy as np

from frontier_data.grid_export import write_vti


def test_vti_axis_order_vectors_units_and_coordinates(tmp_path):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkCommonCore import vtkDataArray
    from vtkmodules.vtkIOXML import vtkXMLImageDataReader

    path = tmp_path / "flow.vti"
    a = np.arange(2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 3)
    write_vti(
        path,
        {"velocity": a},
        origin=[-2, 10, 3],
        spacing=[0.5, 2, 4],
        time_s=3.25,
        units={"velocity": "m/s"},
    )
    reader = vtkXMLImageDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    image = reader.GetOutput()
    assert image.GetDimensions() == (2, 3, 4)
    assert image.GetPoint(1) == (-1.5, 10, 3)
    assert image.GetPoint(2) == (-2, 12, 3)
    assert image.GetPoint(6) == (-2, 10, 7)
    values = image.GetPointData().GetArray("velocity")
    np.testing.assert_array_equal(vtk_to_numpy(values), a.transpose(2, 1, 0, 3).reshape(-1, 3))
    assert values.GetInformation().Get(vtkDataArray.UNITS_LABEL()) == "m/s"
    assert image.GetFieldData().GetArray("TimeValue").GetValue(0) == 3.25


def test_comparison_dataset_keeps_predictions_references_and_units(tmp_path):
    import hashlib

    from frontier_data.dataset import Dataset, DatasetWriter
    from frontier_data.grid_export import comparison_dataset
    from frontier_data.predictions import write_transient_prediction

    source = tmp_path / "source"
    writer = DatasetWriter(source, "flow")
    truth = np.ones((3, 2, 3, 4, 3), np.float32)
    times = np.array([10.0, 10.05, 10.1])
    writer.write_grid_sample(
        "a",
        fields={"velocity": truth},
        times=times,
        spacing=[1, 2, 3],
        origin=[-3, 4, 8],
        field_units={"velocity": "m/s"},
        static={"fluid_geometry_mask": np.ones((2, 3, 4))},
    )
    writer.finalize()
    original = (source / "manifest.json").read_bytes()
    predictions = tmp_path / "predictions"
    write_transient_prediction(
        predictions,
        "a",
        pred={"velocity": truth + 2},
        true={"velocity": truth},
        times=times,
        info={"dataset_manifest_sha256": hashlib.sha256(original).hexdigest()},
        write_zarr=False,
    )
    result = comparison_dataset(predictions, source, tmp_path / "comparison")
    sample = Dataset(result).sample("a")
    np.testing.assert_array_equal(sample.array("velocity_error")[:], np.full_like(truth, 2))
    np.testing.assert_array_equal(sample.array("velocity_reference")[:], truth)
    np.testing.assert_array_equal(sample.times, times)
    assert sample.origin == [-3, 4, 8] and sample.spacing == [1, 2, 3]
    assert all(f.units == "m/s" for f in sample.meta.fields)
    assert (source / "manifest.json").read_bytes() == original


def test_service_forecast_exports_channels_and_absolute_physical_time(tmp_path):
    import json
    from xml.etree import ElementTree

    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOXML import vtkXMLImageDataReader

    from frontier_data.grid_export import export_service_response

    states = np.arange(2 * 2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 2, 3, 4, 4)
    meta = {
        "fields": ["velocity", "water_fraction"],
        "channels": [3, 1],
        "units": {"velocity": "m/s", "water_fraction": "1"},
        "dt": 0.05,
        "time_units": "s",
        "n_steps": 2,
        "shape": [2, 3, 4, 4],
    }
    response = tmp_path / "response.npz"
    np.savez(response, states=states, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8))
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "initial_time_s": 10.1,
                "origin_m": [-2, 3, 4],
                "spacing_m": [0.5, 1, 2],
            }
        )
    )
    pvd = export_service_response(response, contract, tmp_path / "vtk")
    frames = ElementTree.parse(pvd).findall("./Collection/DataSet")
    np.testing.assert_allclose([float(f.attrib["timestep"]) for f in frames], [10.15, 10.2])
    reader = vtkXMLImageDataReader()
    reader.SetFileName(str(pvd.parent / frames[1].attrib["file"]))
    reader.Update()
    image = reader.GetOutput()
    np.testing.assert_array_equal(
        vtk_to_numpy(image.GetPointData().GetArray("velocity")),
        states[1, ..., :3].transpose(2, 1, 0, 3).reshape(-1, 3),
    )
    assert image.GetOrigin() == (-2, 3, 4)
    assert image.GetSpacing() == (0.5, 1, 2)

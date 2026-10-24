# SPDX-License-Identifier: Apache-2.0
"""Export physical grid fields to VTK ImageData and a ParaView time collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np


def comparison_dataset(predictions: Path, dataset: Path, out: Path) -> Path:
    """Registerable viewer dataset with explicitly named model, reference and error fields."""
    from frontier_data.dataset import Dataset, DatasetWriter
    from frontier_data.predictions import list_transient_predictions, read_transient_prediction

    if out.exists():
        raise FileExistsError(f"Refusing to replace {out}")
    source = Dataset(dataset)
    with (dataset / "manifest.json").open("rb") as stream:
        manifest_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="comparison-pending-", dir=out.parent))
    try:
        writer = DatasetWriter(stage, source.manifest.name + " · model comparison")
        ids = list_transient_predictions(predictions)
        if not ids:
            raise ValueError("No prediction cases found")
        for sid in ids:
            pred = read_transient_prediction(predictions, sid)
            sample = source.sample(sid)
            if pred.kind != "grid" or sample.spacing is None:
                raise ValueError("Comparison dataset requires uniform-grid predictions")
            if pred.info.get("dataset_manifest_sha256") != manifest_hash:
                raise ValueError(f"{sid}: predictions belong to a different dataset revision")
            fields, units = {}, {}
            original_units = {f.name: f.units for f in sample.meta.fields}
            for name in pred.field_names:
                for suffix, values in (
                    ("prediction", pred.pred[name]),
                    ("reference", pred.true[name]),
                    ("error", pred.pred[name] - pred.true[name]),
                ):
                    fields[f"{name}_{suffix}"] = values
                    units[f"{name}_{suffix}"] = original_units[name]
            mask_file = pred.info.get("validity_file")
            if mask_file:
                if mask_file != sid + ".validity.npz":
                    raise ValueError("Unexpected validity sidecar path")
                with np.load(predictions / mask_file, allow_pickle=False) as masks:
                    for name in pred.field_names:
                        valid = masks[name]
                        if valid.shape != pred.pred[name].shape or not np.isin(valid, [0, 1]).all():
                            raise ValueError(f"{name}: invalid comparison mask")
                        fields[name + "_valid"] = valid.astype(np.float32)
                        units[name + "_valid"] = "1"
            meta = sample.meta.model_copy(deep=True)
            meta.extra.update(
                comparison_source_manifest_sha256=manifest_hash,
                checkpoint_sha256=pred.info.get("checkpoint_sha256"),
                role="Model/reference review; preserve separate validity fields when scoring",
            )
            static = {}
            if "fluid_geometry_mask" in sample.static_names:
                static["fluid_geometry_mask"] = np.asarray(sample.static("fluid_geometry_mask"))
            writer.write_grid_sample(
                sid,
                fields=fields,
                times=pred.times,
                spacing=sample.spacing,
                origin=sample.origin,
                static=static,
                parameters=sample.meta.parameters,
                group=source.entry(sid).group,
                meta=meta,
                field_units=units,
                time_units="s",
            )
        writer.manifest.splits = {"test": ids}
        writer.finalize()
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return out


def write_vti(
    path: Path,
    fields: Mapping[str, np.ndarray],
    *,
    origin: Sequence[float],
    spacing: Sequence[float],
    time_s: float,
    units: Mapping[str, str],
) -> None:
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonCore import vtkDataArray, vtkDoubleArray
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkIOXML import vtkXMLImageDataWriter

    if not fields:
        raise ValueError("At least one grid field is required")
    shape = next(iter(fields.values())).shape[:3]
    if len(shape) != 3 or len(origin) != 3 or len(spacing) != 3:
        raise ValueError("Three-dimensional grid coordinates are required")
    if not np.isfinite([*origin, *spacing, time_s]).all() or min(spacing) <= 0:
        raise ValueError("Finite SI coordinates/time and positive spacing are required")
    image = vtkImageData()
    image.SetDimensions(*shape)
    image.SetOrigin(*origin)
    image.SetSpacing(*spacing)
    for name, values in fields.items():
        a = np.asarray(values)
        if a.ndim != 4 or a.shape[:3] != shape:
            raise ValueError(f"{name}: fields must be (X,Y,Z,C) on the same grid")
        if not units.get(name):
            raise ValueError(f"{name}: a physical unit or '1' must be supplied")
        # VTK stores X fastest. Frontier arrays are XYZ with Z fastest.
        packed = np.ascontiguousarray(a.transpose(2, 1, 0, 3).reshape(-1, a.shape[-1]))
        vtk_array = numpy_to_vtk(packed.astype(np.float32), deep=True)
        vtk_array.SetName(name)
        information: Any = vtk_array.GetInformation()
        information.Set(vtkDataArray.UNITS_LABEL(), units[name])
        image.GetPointData().AddArray(vtk_array)
    stamp = vtkDoubleArray()
    stamp.SetName("TimeValue")
    stamp.InsertNextValue(time_s)
    image.GetFieldData().AddArray(stamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtkXMLImageDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(image)
    writer.SetDataModeToAppended()
    writer.SetCompressorTypeToZLib()
    if writer.Write() != 1:
        raise OSError(f"VTK failed to write {path}")


def export_predictions(
    predictions: Path, dataset: Path, sample: str, out: Path, *,
    fields: Sequence[str] | None = None, frames: slice | None = None,
    buffer_bytes: int = 128 << 20, cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Export selected frames with bounded array reads and atomic directory visibility.

    The allowance admits per-frame NumPy output/workspace and ArrayRoute reads. VTK's
    internal allocations, codecs, metadata and OS caches are additional. Inputs must
    stay immutable. Native validity arrays are required when masks are declared; NPZ
    masks are never silently ignored or eagerly loaded as a fallback.
    """
    from arrayroute.publication import staged_directory
    from arrayroute.session import LocalReadSession
    from frontier_data.dataset import Dataset
    from frontier_data.prediction_view import PredictionView, Role

    if out.exists():
        raise FileExistsError(f"Refusing to replace {out}")
    source = Dataset(dataset).sample(sample)
    prediction = PredictionView(
        predictions, sample, session=LocalReadSession(buffer_bytes=buffer_bytes),
    )
    if source.spacing is None or source.origin is None or prediction.kind != "grid":
        raise ValueError("A uniform-grid prediction is required")
    with (dataset / "manifest.json").open("rb") as stream:
        manifest_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if prediction.info.get("dataset_manifest_sha256") != manifest_hash:
        raise ValueError("Predictions belong to a different or unspecified dataset revision")
    if prediction.time_units != "s":
        raise ValueError("ParaView export requires prediction time in seconds")
    for key, expected in (("spacing", source.spacing), ("origin", source.origin)):
        if prediction.group.attrs.get(key) != expected:
            raise ValueError(f"Prediction {key} differs from the source grid")
    selected = prediction.selected_fields(fields)
    units = {f.name: f.units for f in source.meta.fields if f.name in selected}
    if any(not units.get(name) for name in selected):
        raise ValueError("Every selected field requires physical units")
    masks = set()
    output_bytes = 0
    scratch_bytes = 0
    for name in selected:
        shape = prediction.reader(name).backend.spec.shape
        if len(shape) != 5 or tuple(shape[1:4]) != tuple(source.shape):
            raise ValueError(f"{name}: predictions must match the three-dimensional source grid")
        stored_units = prediction.units(name)
        if stored_units is not None and stored_units != units[name]:
            raise ValueError(f"{name}: prediction and dataset units disagree")
        has_mask = f"{name}_valid" in prediction.group["fields"]
        if prediction.info.get("validity_file") and not has_mask:
            raise ValueError(f"{name}: store the validity mask as fields/{name}_valid in Zarr")
        roles: list[Role] = ["pred", "true"]
        if has_mask:
            masks.add(name)
            roles.append("valid")
        # Float64 input is allowed; VTK output is float32. Reserve returned fields,
        # p/t, subtraction, masking, packing and temporary float32 conversion.
        points = int(np.prod(shape[1:]))
        output_bytes += points * 4 * 4
        for role in roles:
            reader = prediction.reader(name, role)
            if reader.backend.spec.shape != shape or reader.backend.dtype.kind not in "fbiu":
                raise ValueError(f"{name}: invalid {role} field shape or dtype")
            request = reader.plan(prediction.frame_selection(name, 0))
            scratch_bytes = max(scratch_bytes, request.buffer_bytes_upper_bound + points * 40)
    if output_bytes + scratch_bytes > buffer_bytes:
        raise MemoryError(
            "Selected export frame exceeds allowance; select fewer fields or raise it")
    chosen = frames if frames is not None else slice(None)
    if not isinstance(chosen, slice) or (chosen.step is not None and chosen.step <= 0):
        raise ValueError("frames must be a slice with positive step")
    indices = range(*chosen.indices(prediction.n_steps))
    if not indices:
        raise ValueError("No prediction frames selected")
    root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = ET.SubElement(root, "Collection")
    with staged_directory(out, cancelled=cancelled) as stage:
        for index in indices:
            if cancelled is not None and cancelled():
                raise InterruptedError("Prediction export cancelled")
            time_s = prediction.time(index)
            frame, field_units = {}, {}
            for name in selected:
                selection = prediction.frame_selection(name, index)
                p = prediction.read(name, selection=selection, cancelled=cancelled)
                t = prediction.read(name, "true", selection=selection, cancelled=cancelled)
                valid = (prediction.read(name, "valid", selection=selection, cancelled=cancelled)
                         if name in masks else np.ones(p.shape, dtype=bool))
                if not np.isin(valid, [0, 1]).all():
                    raise ValueError(f"{name}: invalid mask")
                for suffix, a in (("prediction", p), ("reference", t), ("error", p - t)):
                    frame[f"{name}_{suffix}"] = np.where(valid, a, np.nan).astype(np.float32)
                    field_units[f"{name}_{suffix}"] = units[name]
                frame[f"{name}_valid"] = valid.astype(np.float32)
                field_units[f"{name}_valid"] = "1"
                del p, t, valid, a
            filename = f"frame-{index:05d}.vti"
            write_vti(
                stage / filename, frame, origin=source.origin, spacing=source.spacing,
                time_s=time_s, units=field_units,
            )
            del frame
            ET.SubElement(
                collection, "DataSet", timestep=repr(time_s), group="", part="0", file=filename,
            )
        ET.ElementTree(root).write(stage / "flow.pvd", encoding="utf-8", xml_declaration=True)
        (stage / "units.json").write_text(
            json.dumps({
                "coordinates": "m", "time": "s", "fields": units,
                "dataset_manifest_sha256": manifest_hash,
                "model_z_to_report_elevation_offset_m": source.meta.extra.get(
                    "report_elevation_offset_m", 0),
                "missing_values": "NaN outside per-field validity; masks included",
            }, indent=2)
        )
    return out / "flow.pvd"


def export_service_response(response: Path, contract: Path, out: Path) -> Path:
    """Open a served grid forecast in ParaView without inventing geometry or timestamps.

    The request-side contract supplies the initial physical time and grid coordinates;
    the response supplies the field/channel layout, units and temporal step.
    """
    if out.exists():
        raise FileExistsError(out)
    request = json.loads(contract.read_text())
    with np.load(response, allow_pickle=False) as arrays:
        states = arrays["states"]
        meta = json.loads(arrays["meta"].tobytes())
    names, channels = meta["fields"], meta["channels"]
    if (
        states.ndim != 5
        or not np.isfinite(states).all()
        or len(names) != len(channels)
        or len(set(names)) != len(names)
        or any(not isinstance(c, int) or isinstance(c, bool) or c < 1 for c in channels)
        or sum(channels) != states.shape[-1]
        or meta["n_steps"] != len(states)
        or list(states.shape[1:]) != meta["shape"]
    ):
        raise ValueError("Response is not a consistent three-dimensional grid forecast")
    if meta.get("time_units") != "s":
        raise ValueError("Response time must be in seconds")
    dt, initial = float(meta["dt"]), float(request["initial_time_s"])
    if not np.isfinite([dt, initial]).all() or dt <= 0:
        raise ValueError("Finite initial time and a positive time step are required")
    origin, spacing = request["origin_m"], request["spacing_m"]
    units = meta["units"]
    offsets = np.cumsum([0, *channels])
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="forecast-vtk-pending-", dir=out.parent))
    root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = ET.SubElement(root, "Collection")
    try:
        for i, state in enumerate(states):
            time_s = initial + (i + 1) * dt
            filename = f"frame-{i:05d}.vti"
            fields = {name: state[..., offsets[j] : offsets[j + 1]] for j, name in enumerate(names)}
            write_vti(
                stage / filename,
                fields,
                origin=origin,
                spacing=spacing,
                time_s=time_s,
                units=units,
            )
            ET.SubElement(
                collection,
                "DataSet",
                timestep=repr(time_s),
                group="",
                part="0",
                file=filename,
            )
        ET.ElementTree(root).write(stage / "flow.pvd", encoding="utf-8", xml_declaration=True)
        (stage / "units.json").write_text(
            json.dumps(
                {
                    "coordinates": "m",
                    "time": "s",
                    "fields": units,
                    "source": "Model inference; no reference values or future validity masks",
                    "model": meta.get("model"),
                    "request": meta.get("request"),
                },
                indent=2,
            )
        )
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return out / "flow.pvd"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fields", nargs="+", help="Selected physical fields; default: all")
    parser.add_argument("--start", type=int, help="First frame index (inclusive)")
    parser.add_argument("--stop", type=int, help="Last frame index (exclusive)")
    parser.add_argument("--stride", type=int, default=1, help="Export every Nth frame")
    parser.add_argument(
        "--buffer-mib", type=int, default=128, help="Array/workspace allowance in MiB")
    args = parser.parse_args()
    try:
        path = export_predictions(
            args.predictions, args.dataset, args.sample, args.out, fields=args.fields,
            frames=slice(args.start, args.stop, args.stride), buffer_bytes=args.buffer_mib << 20,
        )
    except (ValueError, MemoryError, OSError) as error:
        parser.exit(2, f"Prediction export failed: {error}\n")
    print(path)


if __name__ == "__main__":
    main()

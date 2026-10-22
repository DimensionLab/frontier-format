# SPDX-License-Identifier: Apache-2.0
"""Compare field readers and bind reusable settings to the measured source and region."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from arrayroute.backends import ZarrBackend
from arrayroute.compare import ReadPolicy, compare_reads
from arrayroute.settings import ReadSettings
from frontier_data.arrays import Sample
from frontier_data.dataset import Dataset
from frontier_data.query import FieldReader

MAX_REPORT_BYTES = 16 << 20


def _runtime(settings: ReadSettings) -> dict:
    engine_package = {'zarr-python': 'zarr'}.get(settings.engine, settings.engine)
    packages = ['arrayroute', 'numpy', engine_package]
    return {'system': platform.system(), 'machine': platform.machine(),
            'logical_cpus': os.cpu_count(),
            'thread_environment': {key: os.environ.get(key) for key in
                                   ('RAYON_NUM_THREADS', 'OMP_NUM_THREADS',
                                    'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
            'versions': {p: importlib.metadata.version(p) for p in packages}}


def _binding(dataset: Path, sample: str, field: str, reader: FieldReader) -> dict:
    backend = reader.reader.backend
    assert isinstance(backend, ZarrBackend)
    array_metadata = json.dumps(backend.array.metadata.to_dict(), sort_keys=True).encode()
    meta_path = Dataset(dataset).sample_dir(sample)/'meta.json'
    return {'dataset': str(dataset.resolve()), 'sample': sample, 'field': field,
            'location': reader.location, 'lod': reader.lod,
            'dimensions': list(reader.dimensions), 'units': reader.units,
            'association': reader.association,
            'sample_metadata_sha256': hashlib.sha256(meta_path.read_bytes()).hexdigest(),
            'array_metadata_sha256': hashlib.sha256(array_metadata).hexdigest()}


def tune_field(
    dataset: Path, sample: str, field: str, *, vary: str,
    fixed: Mapping[str, int | slice], settings: Sequence[ReadSettings],
    count: int = 8, worker_counts: Sequence[int] = (1, 2, 4),
    total_buffer_bytes: int = 128 << 20, max_output_bytes: int = 1 << 30,
    repeats: int = 3, time_budget_seconds: float = 60,
    location: str | None = None, lod: int = 0,
) -> dict:
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 4096:
        raise ValueError('Use 1–4096 representative regions')
    directory = Dataset(dataset).sample_dir(sample)

    def open_field(configuration, budget):
        return FieldReader(Sample.open(directory, sample_id=sample), field, location=location,
                           lod=lod, session=configuration.session(budget))

    reference = ReadSettings()
    prototype = open_field(reference, total_buffer_bytes)
    if vary not in prototype.dimensions or vary in fixed:
        raise ValueError(f'Choose --vary from {prototype.dimensions}, outside --select')
    length = prototype.reader.backend.spec.shape[prototype.dimensions.index(vary)]
    count = min(count, length)
    if not count:
        raise ValueError('The varied dimension is empty')
    indices = [i*(length-1)//(count-1) if count > 1 else 0 for i in range(count)]
    regions = [prototype.selection({**fixed, vary: i}) for i in indices]
    binding = _binding(dataset, sample, field, prototype)
    policies = [ReadPolicy(s.name, lambda budget, s=s: open_field(s, budget).reader)
                for s in settings]
    report = compare_reads(lambda budget: open_field(reference, budget).reader,
        policies, regions, worker_counts=worker_counts, total_buffer_bytes=total_buffer_bytes,
        max_output_bytes=max_output_bytes, repeats=repeats, time_budget_seconds=time_budget_seconds)
    if _binding(dataset, sample, field, open_field(reference, total_buffer_bytes)) != binding:
        raise ValueError('Source metadata changed during the comparison')
    report.update(kind='frontier-field-read-pilot', profile_version=1, binding=binding,
                  varied_dimension=vary, selected_indices=indices,
                  reader_settings={s.name: asdict(s) for s in settings},
                  native_thread_policy='TensorStore threads are per native pool per outer reader; '
                                       'other engines use their process defaults',
                  profile_scope='Performance recommendation, not a snapshot or integrity proof')
    if report['recommendation'] is not None:
        chosen = next(s for s in settings if s.name == report['recommendation']['policy'])
        report['runtime'] = _runtime(chosen)
    return report


def save_report(path: Path, report: dict) -> None:
    """Publish a bounded JSON receipt atomically without replacing an existing file."""
    payload = (json.dumps(report, indent=2, allow_nan=False)+'\n').encode()
    if len(payload) > MAX_REPORT_BYTES:
        raise ValueError('Read report exceeds 16 MiB')
    if os.path.lexists(path):
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.partial', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def profiled_field(
    path: Path, dataset: Path, sample: str, field: str,
    selectors: Mapping[str, int | slice], *, location: str | None, lod: int,
) -> tuple[FieldReader, dict]:
    """Open a measured configuration only for its bound field, region pattern and runtime."""
    with path.open('rb') as handle:
        raw = handle.read(MAX_REPORT_BYTES+1)
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError('Read profile exceeds 16 MiB')
    report = json.loads(raw)
    if (not isinstance(report, dict) or report.get('kind') != 'frontier-field-read-pilot'
            or type(report.get('profile_version')) is not int or report['profile_version'] != 1
            or report.get('state') != 'completed'
            or not isinstance(report.get('recommendation'), dict)):
        raise ValueError('Use a completed tune-field report with a recommendation')
    recommendation = report['recommendation']
    try:
        chosen = ReadSettings(**report['reader_settings'][recommendation['policy']])
        budget, workers = recommendation['buffer_bytes_per_worker'], recommendation['workers']
        if (any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (budget, workers))
                or workers > 256 or type(report.get('total_buffer_bytes')) is not int
                or budget != report['total_buffer_bytes']//workers):
            raise ValueError('Invalid read-profile buffer allowance')
        if _runtime(chosen) != report.get('runtime'):
            raise ValueError('Reader runtime changed; run tune-field again on this machine')
        source = Dataset(dataset).sample(sample)
        prototype = FieldReader(source, field, location=location, lod=lod, buffer_bytes=budget)
        if _binding(dataset, sample, field, prototype) != report.get('binding'):
            raise ValueError('Read profile belongs to another field or its metadata changed')
        normalized = prototype.plan(selectors).selection
        actual = [{'start': s.start, 'stop': s.stop, 'step': s.step}
                  if isinstance(s, slice) else s for s in normalized]
        axis = prototype.dimensions.index(report['varied_dimension'])
        measured = report['regions'][0]
        if (not isinstance(normalized[axis], int) or len(measured) != len(actual)
                or any(a != b for i, (a, b) in enumerate(zip(actual, measured, strict=True))
                       if i != axis)):
            raise ValueError('Selection differs from the measured region pattern; '
                             'run tune-field again')
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError('Malformed read profile') from exc
    reader = FieldReader(source, field, location=location, lod=lod, session=chosen.session(budget))
    return reader, {'path': str(path.resolve()), 'settings': asdict(chosen),
                    'measured_outer_workers': workers, 'buffer_bytes_per_worker': budget,
                    'scope': 'Performance recommendation; not a snapshot or integrity proof'}

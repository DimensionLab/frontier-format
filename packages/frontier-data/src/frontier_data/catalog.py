# SPDX-License-Identifier: Apache-2.0
"""Scientific metadata snapshots for bounded dataset browsing, without opening arrays."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from arrayroute.catalog import Catalog, CatalogReceipt, write_catalog
from frontier_data.dataset import Manifest, SampleMeta


def index_dataset(root: str | Path, out: str | Path) -> CatalogReceipt:
    """Index a quiescent v1 dataset once; subsequent browsing needs only the catalog.

    Conversion still loads the source JSON manifest and split membership into memory.
    Sample metadata is streamed, one file at a time. Per-file hashes identify captured
    inputs; this does not establish an atomic snapshot of concurrently changing inputs.
    Numerical payloads are neither opened nor copied.
    """
    root = Path(root).resolve()
    raw = (root / "manifest.json").read_bytes()
    source = json.loads(raw)
    manifest = Manifest.model_validate(source)
    ids = {entry.id for entry in manifest.samples}
    splits: dict[str, list[str]] = defaultdict(list)
    for name, members in manifest.splits.items():
        for member in members:
            if member not in ids:
                raise ValueError(f"split {name!r} references unknown sample {member!r}")
            splits[member].append(name)
    metadata = {
        "schema": "frontier-dataset-catalog-v1",
        "source_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest": {k: v for k, v in source.items() if k not in {"samples", "splits"}},
        "split_counts": {name: len(members) for name, members in manifest.splits.items()},
    }

    def records():
        for entry, original_entry in zip(manifest.samples, source["samples"], strict=True):
            relative = PurePosixPath(entry.path.replace("\\", "/"))
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or (relative.parts and ":" in relative.parts[0])
            ):
                raise ValueError(f"sample {entry.id!r} leaves the dataset directory")
            path = (root / relative / "meta.json").resolve()
            if root not in path.parents:
                raise ValueError(f"sample {entry.id!r} resolves outside the dataset directory")
            content = path.read_bytes()
            sample = json.loads(content)
            if SampleMeta.model_validate(sample).id != entry.id:
                raise ValueError(f"metadata id does not match sample {entry.id!r}")
            yield (
                entry.id,
                {
                    "entry": original_entry,
                    "metadata": sample,
                    "metadata_sha256": hashlib.sha256(content).hexdigest(),
                    "splits": sorted(splits.get(entry.id, [])),
                },
            )
        if (root / "manifest.json").read_bytes() != raw:
            raise ValueError(
                "dataset manifest changed during catalog creation; retry when quiescent"
            )

    return write_catalog(out, records(), metadata=metadata)


def inspect_catalog(
    path: str | Path,
    *,
    sample: str | None = None,
    start: int = 0,
    limit: int = 20,
    verify: bool = False,
) -> dict[str, Any]:
    with Catalog(path) as catalog:
        if catalog.metadata.get("schema") != "frontier-dataset-catalog-v1":
            raise ValueError("not a Frontier dataset catalog")
        verified = catalog.verify() if verify else None
        if sample is not None:
            if start:
                raise ValueError("sample and start cannot be combined")
            entries = (catalog.get(sample),)
            next_ordinal = None
        else:
            page = catalog.page(start=start, limit=limit)
            entries, next_ordinal = page.entries, page.next_ordinal
        return {
            **catalog.metadata,
            "revision": catalog.revision,
            "verified_revision": verified,
            "total_samples": catalog.count,
            "samples": [{"id": e.key, **e.value} for e in entries],
            "next_ordinal": next_ordinal,
        }

# SPDX-License-Identifier: Apache-2.0
"""Clone local Zarr v3 datasets with flat, immutable hardlinked chunk files."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from frontier_data.dataset import Dataset, write_json


def chunk_key(index: tuple[int, ...], separator: str) -> str:
    if separator not in ("/", "."):
        raise ValueError("Chunk separator must be '/' or '.'")
    return "c" + (separator + separator.join(map(str, index)) if index else "")


def chunk_inventory(directory: Path, metadata: dict[str, Any]) -> tuple[dict, int]:
    """Validate stored keys; absent chunks retain Zarr's implicit fill-value semantics."""
    encoding = metadata.get("chunk_key_encoding", {})
    separator = encoding.get("configuration", {}).get("separator", "/")
    grid = metadata.get("chunk_grid", {})
    if (
        metadata.get("zarr_format") != 3
        or encoding.get("name") != "default"
        or separator not in ("/", ".")
        or grid.get("name") != "regular"
        or metadata.get("storage_transformers")
    ):
        raise ValueError(f"{directory}: need regular Zarr v3 default chunk keys")
    shape, chunks = metadata["shape"], grid["configuration"]["chunk_shape"]
    if len(shape) != len(chunks) or any(n < 0 for n in shape) or any(n < 1 for n in chunks):
        raise ValueError(f"{directory}: invalid shape or chunk grid")
    limits = tuple((n + c - 1) // c for n, c in zip(shape, chunks, strict=True))
    found = {}
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Refusing symlink {path}")
        if path.is_dir():
            continue
        key = path.relative_to(directory).as_posix()
        if key == "zarr.json":
            continue
        parts = key.split(separator)
        if parts[0] != "c" or len(parts) != len(shape) + 1:
            raise ValueError(f"{directory}: malformed chunk key {key}")
        try:
            index = tuple(int(part) for part in parts[1:])
        except ValueError as exc:
            raise ValueError(f"{directory}: malformed chunk key {key}") from exc
        if index in found:
            raise ValueError(f"{directory}: duplicate chunk coordinate {index}")
        if key != chunk_key(index, separator) or any(
            i < 0 or i >= n for i, n in zip(index, limits, strict=True)
        ):
            raise ValueError(f"{directory}: noncanonical or out-of-range chunk {key}")
        found[index] = path
    return found, math.prod(limits) - len(found)


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _clone_node(source: Path, target: Path, inventory: list, complete: bool) -> None:
    metadata_path = source / "zarr.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("zarr_format") != 3 or metadata.get("consolidated_metadata"):
        raise ValueError(f"{source}: need unconsolidated Zarr v3 metadata")
    target.mkdir(parents=True)
    if metadata.get("node_type") == "group":
        write_json(target / "zarr.json", metadata)
        for child in source.iterdir():
            if child.name == "zarr.json":
                continue
            if not child.is_dir() or not (child / "zarr.json").is_file():
                raise ValueError(f"{child}: unsupported Zarr group entry")
            _clone_node(child, target / child.name, inventory, complete)
        return
    if metadata.get("node_type") != "array":
        raise ValueError(f"{source}: unknown Zarr node type")
    chunks, missing = chunk_inventory(source, metadata)
    if complete and missing:
        raise ValueError(f"{source}: {missing} missing chunks (implicit fill is disabled)")
    metadata["chunk_key_encoding"] = {"name": "default", "configuration": {"separator": "."}}
    write_json(target / "zarr.json", metadata)
    digests = {}
    for index, path in sorted(chunks.items()):
        key = chunk_key(index, ".")
        os.link(path, target / key)
        digests[key] = _sha(path)
    actual, target_missing = chunk_inventory(target, metadata)
    if set(actual) != set(chunks) or target_missing != missing:
        raise ValueError(f"{target}: incomplete copied chunk inventory")
    source_array, target_array = (
        zarr.open_array(str(source), mode="r"),
        zarr.open_array(str(target), mode="r"),
    )
    shape = metadata["shape"]
    sizes = metadata["chunk_grid"]["configuration"]["chunk_shape"]
    for index in chunks:
        selection = tuple(
            slice(i * size, min((i + 1) * size, n))
            for i, size, n in zip(index, sizes, shape, strict=True)
        )
        if not np.array_equal(source_array[selection], target_array[selection], equal_nan=True):
            raise ValueError(f"{target}: decoded chunk mismatch at {index}")
    inventory.append(
        {
            "array": str(target),
            "source_metadata_sha256": _sha(metadata_path),
            "chunks": len(chunks),
            "implicit_fill_chunks": missing,
            "compressed_bytes": sum(path.stat().st_size for path in chunks.values()),
            "chunk_sha256": digests,
        }
    )


def repack_dataset(source: Path, out: Path, *, require_complete_chunks: bool = False) -> dict:
    """Make a verified flat-key clone; neither source metadata nor array values change."""
    source, out = source.absolute(), out.absolute()
    if source.is_symlink() or any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Refusing a symlink in the source dataset")
    source = source.resolve()
    if out.exists() or out == source or source in out.parents:
        raise ValueError("Output must be a new directory outside the source dataset")
    ds = Dataset(source)
    if len({entry.path for entry in ds.manifest.samples}) != len(ds.manifest.samples):
        raise ValueError("Duplicate sample paths")
    out.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != out.parent.stat().st_dev:
        raise ValueError("Hardlink clone requires the same filesystem")
    partial = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    inventory = []
    original_hash = _sha(source / "manifest.json")
    try:
        for extra in source.iterdir():
            if extra.is_file() and extra.name not in ("manifest.json", "storage-layout.json"):
                shutil.copy2(extra, partial / extra.name)
        for sid in ds.sample_ids:
            logging.getLogger(__name__).info("Cloning and verifying %s", sid)
            original = ds.sample_dir(sid)
            target = partial / ds.entry(sid).path
            target.mkdir(parents=True)
            for child in original.iterdir():
                if child.is_dir():
                    _clone_node(child, target / child.name, inventory, require_complete_chunks)
                else:
                    shutil.copy2(child, target / child.name)
        for item in inventory:
            item["array"] = str(Path(item["array"]).relative_to(partial))
        write_json(partial / "storage-layout.json", {"arrays": inventory})
        manifest = ds.manifest.model_copy(deep=True)
        manifest.source["storage_layout"] = {
            "source_manifest_sha256": original_hash,
            "implementation_sha256": _sha(Path(__file__)),
            "separator": ".",
            "mode": "immutable hardlinks",
            "decoded_verification": "all stored chunks",
            "implicit_fill_chunks": sum(item["implicit_fill_chunks"] for item in inventory),
            "inventory_sha256": _sha(partial / "storage-layout.json"),
        }
        write_json(partial / "manifest.json", manifest.model_dump())
        if _sha(source / "manifest.json") != original_hash:
            raise ValueError("Source manifest changed during clone")
        for item in inventory:
            for key in item["chunk_sha256"]:
                path = partial / item["array"] / key
                path.chmod(path.stat().st_mode & ~0o222)
        partial.rename(out)
    except BaseException:
        shutil.rmtree(partial)
        raise
    return {
        "out": str(out),
        "arrays": len(inventory),
        "chunks": sum(item["chunks"] for item in inventory),
        "shared_compressed_bytes": sum(item["compressed_bytes"] for item in inventory),
        "implicit_fill_chunks": sum(item["implicit_fill_chunks"] for item in inventory),
    }

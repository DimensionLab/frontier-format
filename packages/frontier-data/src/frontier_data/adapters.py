# SPDX-License-Identifier: Apache-2.0
"""Built-in factories implementing the same entry-point contract as third parties."""

from __future__ import annotations

from typing import Any

from arrayroute.session import LocalReadSession


def _local(engine: str, source: dict[str, str], **options: Any) -> Any:
    if set(source) != {"store", "array_path"}:
        raise ValueError("Local array sources require store and array_path")
    session = LocalReadSession(engine, **options)
    return session.open(source["store"], source["array_path"]).backend


def zarr(source: dict[str, str], **options: Any) -> Any:
    return _local("zarr-python", source, **options)


def tensorstore(source: dict[str, str], **options: Any) -> Any:
    return _local("tensorstore", source, **options)


def zarrista(source: dict[str, str], **options: Any) -> Any:
    return _local("zarrista", source, **options)


def normalize(*, mean: list[float], scale: list[float], device: str = "cpu") -> Any:
    from arrayroute.torch_transforms import TorchChannelNormalizer

    return TorchChannelNormalizer(mean, scale, device=device)

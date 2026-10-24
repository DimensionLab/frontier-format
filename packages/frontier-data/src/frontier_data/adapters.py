# SPDX-License-Identifier: Apache-2.0
"""Built-in factories implementing the same entry-point contract as third parties."""

from __future__ import annotations

from typing import Any, Literal

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


class _OwnedHDF5:
    def __init__(self, source: dict[str, str], *, strided_reads: Literal["auto", "native"]):
        from arrayroute.hdf5 import HDF5ReadSession

        if set(source) != {"store", "array_path"}:
            raise ValueError("HDF5 sources require store (file) and array_path (dataset)")
        self._session = HDF5ReadSession(source["store"], strided_reads=strided_reads,
                                        max_open_arrays=1)
        try:
            self._backend = self._session.open(source["array_path"]).backend
        except BaseException:
            self._session.close()
            raise
        self.spec, self.dtype = self._backend.spec, self._backend.dtype

    def read(self, selection):
        return self._backend.read(selection)

    def close(self) -> None:
        self._session.close()

    def __getstate__(self):
        raise TypeError("Open an HDF5 adapter in each worker; native handles cannot be copied")


def hdf5(
    source: dict[str, str], *, strided_reads: Literal["auto", "native"] = "auto",
) -> Any:
    """Open one embedded fixed-size array; the returned backend owns its file handle."""
    return _OwnedHDF5(source, strided_reads=strided_reads)


def normalize(*, mean: list[float], scale: list[float], device: str = "cpu") -> Any:
    from arrayroute.torch_transforms import TorchChannelNormalizer

    return TorchChannelNormalizer(mean, scale, device=device)

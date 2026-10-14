# SPDX-License-Identifier: Apache-2.0
"""Which implementation of the hot kernels runs: the Rust ports in ``frontier_native.prep``
or the numpy reference in ``frontier_prep.ops``.

The numpy code is the reference and the fallback; the Rust kernels are validated against it
(``tests/test_kernels_native.py``) and produce the same bytes, so the choice never changes a
version hash. Selection:

- ``FRONTIER_PREP_KERNELS`` unset or ``auto``: native when ``frontier_native`` is importable
  and implements this package's spec revision, the reference otherwise;
- ``FRONTIER_PREP_KERNELS=python``: always the reference;
- ``FRONTIER_PREP_KERNELS=native``: the Rust kernels, and an error when they are missing.

A native kernel still hands single calls back to the reference for input it does not take
(arrays that are not float32/float64 ``(N, 3)`` points or ``(M, 3)`` faces, negative or
out-of-range indices, a tolerance grid that does not fit int64): the reference then decides,
including which error to raise. ``describe()`` is what the engine writes into provenance.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Literal

ENV_VAR = "FRONTIER_PREP_KERNELS"
#: revision of the port specification (``docs/spec/data-prep.md`` section 11) the reference
#: kernels implement; a native module with another revision is not used
SPEC_REVISION = 1

Backend = Literal["native", "python"]


class KernelUnavailableError(RuntimeError):
    """``FRONTIER_PREP_KERNELS=native`` was requested and the Rust kernels cannot be used."""


def requested() -> Literal["auto", "native", "python"]:
    """The value of ``FRONTIER_PREP_KERNELS`` (``auto`` when unset or empty)."""
    raw = os.environ.get(ENV_VAR, "").strip().lower()
    if raw in ("", "auto"):
        return "auto"
    if raw == "native":
        return "native"
    if raw == "python":
        return "python"
    raise ValueError(f"{ENV_VAR} must be one of auto, native, python; got {raw!r}")


@lru_cache(maxsize=1)
def _load() -> tuple[Any | None, str]:
    """``(frontier_native.prep or None, reason it is unusable)``; imported once per process."""
    try:
        mod = importlib.import_module("frontier_native")
    except ImportError as exc:
        return None, f"frontier_native is not importable ({exc})"
    prep = getattr(mod, "prep", None)
    if prep is None:
        return None, "frontier_native has no prep kernels (built before they existed)"
    rev = getattr(prep, "SPEC_REVISION", None)
    if rev != SPEC_REVISION:
        return None, (
            f"frontier_native.prep implements kernel spec revision {rev}, "
            f"frontier_prep needs {SPEC_REVISION}"
        )
    return prep, ""


def native() -> Any | None:
    """The native kernel module to use for this call, or ``None`` for the reference."""
    want = requested()
    if want == "python":
        return None
    prep, why = _load()
    if prep is None and want == "native":
        raise KernelUnavailableError(
            f"{ENV_VAR}=native but {why}; build it with `pnpm nx run frontier-py:develop`"
        )
    return prep


@contextmanager
def forced(choice: Literal["auto", "native", "python"]) -> Iterator[None]:
    """Run a block with ``FRONTIER_PREP_KERNELS`` set to ``choice`` (tests, benchmarks).
    Worker processes started inside the block inherit it."""
    old = os.environ.get(ENV_VAR)
    os.environ[ENV_VAR] = choice
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(ENV_VAR, None)
        else:
            os.environ[ENV_VAR] = old


def backend() -> Backend:
    return "native" if native() is not None else "python"


def _dist_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0+local"


def describe() -> dict[str, Any]:
    """Provenance of the kernel choice: ``backend`` (``native`` | ``python``), ``version``
    (of the Rust kernel crate, or of ``frontier-prep`` for the reference), ``spec_revision``
    and what ``FRONTIER_PREP_KERNELS`` asked for."""
    prep = native()
    if prep is None:
        version = _dist_version("frontier-prep")
    else:
        version = str(getattr(prep, "KERNELS_VERSION", "unknown"))
    return {
        "backend": "native" if prep is not None else "python",
        "version": version,
        "spec_revision": SPEC_REVISION,
        "requested": requested(),
    }


def threads() -> int | None:
    """Threads the native kernels use in this process (``RAYON_NUM_THREADS``, default all
    cores); ``None`` for the reference. Outputs do not depend on it."""
    prep = native()
    return int(prep.num_threads()) if prep is not None else None

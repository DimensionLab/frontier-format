# SPDX-License-Identifier: Apache-2.0
"""First-party operations. Importing this package registers every builtin op; ``BUILTIN`` is
what the ``frontier.ops`` entry point resolves to."""

from frontier_prep import adapt  # noqa: E402,F401 - registration side effect
from frontier_prep.op import REGISTRY
from frontier_prep.ops import (  # noqa: F401 - registration side effect
    dataset,
    fields,
    mesh,
    quality,
    smoothing,
    time,
    units,
)

BUILTIN: list = list(REGISTRY._ops.values())  # noqa: SLF001

__all__ = ["BUILTIN", "REGISTRY"]

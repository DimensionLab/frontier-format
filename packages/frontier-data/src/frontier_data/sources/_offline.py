# SPDX-License-Identifier: Apache-2.0
"""The offline guard for this package's downloads.

``frontier-data`` does not depend on the ``frontier`` SDK. When the SDK is installed, the
shared guard :func:`frontier.config.require_online` decides (setting ``network.offline``,
loopback URLs pass). Without it, the variables the Frontier command line exports to its
child processes decide: ``FRONTIER_OFFLINE`` and ``FRONTIER_NETWORK__OFFLINE``.
"""

from __future__ import annotations

import os

OfflineError: type[RuntimeError]
try:
    from frontier.config import OfflineError as _SdkOfflineError

    OfflineError = _SdkOfflineError
except ImportError:  # the SDK is optional here

    class _LocalOfflineError(RuntimeError):
        """An outbound connection was attempted while offline mode is on."""

    OfflineError = _LocalOfflineError


_TRUE = ("1", "true", "yes", "on")


def require_online(what: str, *, url: str | None = None) -> None:
    """Raise :class:`OfflineError` before any connection when offline mode is on."""
    try:
        from frontier import config
    except ImportError:
        names = ("FRONTIER_NETWORK__OFFLINE", "FRONTIER_OFFLINE")
        if any(os.environ.get(n, "").strip().lower() in _TRUE for n in names):
            raise OfflineError(
                f"offline mode is on (setting network.offline), so {what} is not possible"
            ) from None
        return
    config.require_online(what, url=url)


__all__ = ["OfflineError", "require_online"]

<!-- SPDX-License-Identifier: Apache-2.0 -->
# Connector plugin contract

## Protocol

Python connectors are discovered through the entry-point group `frontier.connectors`.
Each implements `frontier_connectors.Connector`:

```python
class Connector(Protocol):
    id: str; name: str; formats: list[str]; vendor: str; license: str
    def detect(self, path: Path) -> Detection | None: ...
    def describe(self, path: Path) -> SourceDescription: ...   # zones, fields, times, units, solver metadata
    def read(self, path: Path, *, zones=None, fields=None, times=None) -> Iterator[RawSample]: ...
```

`RawSample` is the in-memory neutral form (points, cells/faces, fields with `dims`, times,
metadata) that `frontier_data` writes to v1.1. Connectors that need vendor software
(Abaqus, STAR-CCM+, COMSOL, MATLAB live sessions) ship an *exporter script* run inside the
vendor tool plus a reader for what it writes; they must never require the vendor licence at
import time. Rust `frontier-connectors` keeps `detect` fast paths and delegates reads to
Python through `frontier-py` or a subprocess.

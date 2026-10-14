# SPDX-License-Identifier: Apache-2.0
"""Download AhmedML-style runs from the HuggingFace Hub with resume and provenance.

Files are cached under ``raw_dir`` mirroring the repository layout. ``hf_hub_download`` keeps
partial downloads in ``raw_dir/.cache/huggingface`` and resumes them, so interrupted downloads
are safe to retry. For every file we record size, sha256 and the exact ``resolve`` URL in
``raw_dir/provenance.json`` (hashes are reused while size+mtime are unchanged).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from frontier_data import ahmedml
from frontier_data.sources._offline import require_online

log = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("datasets/raw/ahmedml")
ProgressFn = Callable[[str, int, int], None]  # (message, done, total)


class ForbiddenFileError(ValueError):
    """Raised when a caller asks for a file we refuse to download (volume VTUs)."""


@dataclass(frozen=True)
class RawFile:
    role: str
    repo_path: str
    local_path: str
    url: str
    size: int
    sha256: str
    revision: str | None = None


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def resolve_url(repo: str, repo_path: str, revision: str | None, repo_type: str) -> str:
    rev = revision or "main"
    prefix = f"{repo_type}s/" if repo_type != "model" else ""
    return f"https://huggingface.co/{prefix}{repo}/resolve/{rev}/{repo_path}"


class ProvenanceStore:
    """Thread-safe cache of ``{repo_path: {size, mtime, sha256, url}}`` on disk."""

    def __init__(self, raw_dir: Path) -> None:
        self.path = raw_dir / "provenance.json"
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, object]] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                log.warning("ignoring corrupt provenance file %s", self.path)

    def record(self, repo_path: str, local: Path, url: str, revision: str | None) -> RawFile:
        st = local.stat()
        with self._lock:
            cached = self._data.get(repo_path)
        if (
            cached
            and cached.get("size") == st.st_size
            and cached.get("mtime") == st.st_mtime
            and isinstance(cached.get("sha256"), str)
        ):
            digest = str(cached["sha256"])
        else:
            digest = sha256_of(local)
        entry = {
            "size": st.st_size,
            "mtime": st.st_mtime,
            "sha256": digest,
            "url": url,
            "revision": revision,
        }
        with self._lock:
            self._data[repo_path] = entry
        return RawFile(
            role="",
            repo_path=repo_path,
            local_path=str(local),
            url=url,
            size=st.st_size,
            sha256=digest,
            revision=revision,
        )

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._data, indent=1, sort_keys=True))
            os.replace(tmp, self.path)


def download_file(
    repo_path: str,
    raw_dir: Path = DEFAULT_RAW_DIR,
    *,
    repo: str = ahmedml.HF_REPO,
    repo_type: str = ahmedml.HF_REPO_TYPE,
    revision: str | None = None,
    retries: int = 4,
) -> Path:
    """Download one repository file into ``raw_dir`` (resumable, idempotent)."""
    if ahmedml.is_forbidden(repo_path):
        raise ForbiddenFileError(f"refusing to download volume data: {repo_path}")
    require_online(f"downloading {repo}/{repo_path} from the Hugging Face Hub")
    from huggingface_hub import hf_hub_download

    delay = 2.0
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            local = hf_hub_download(
                repo_id=repo,
                filename=repo_path,
                repo_type=repo_type,
                revision=revision,
                local_dir=str(raw_dir),
            )
            return Path(local)
        except Exception as exc:  # network errors, rate limits
            last = exc
            if attempt == retries:
                break
            log.warning("download %s failed (%s), retrying in %.0fs", repo_path, exc, delay)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"failed to download {repo_path}") from last


def download_run(
    run: int,
    raw_dir: Path = DEFAULT_RAW_DIR,
    *,
    roles: Iterable[str] | None = None,
    revision: str | None = None,
    provenance: ProvenanceStore | None = None,
) -> dict[str, RawFile]:
    """Download the files of one AhmedML run and return ``{role: RawFile}``."""
    raw_dir = Path(raw_dir)
    store = provenance or ProvenanceStore(raw_dir)
    files = ahmedml.run_files(run)
    wanted = set(roles) if roles is not None else set(files)
    out: dict[str, RawFile] = {}
    for role, repo_path in files.items():
        if role not in wanted:
            continue
        try:
            local = download_file(repo_path, raw_dir, revision=revision)
        except RuntimeError:
            if role in ahmedml.OPTIONAL_ROLES:
                log.warning("optional file %s missing for run %d", repo_path, run)
                continue
            raise
        url = resolve_url(ahmedml.HF_REPO, repo_path, revision, ahmedml.HF_REPO_TYPE)
        rec = store.record(repo_path, local, url, revision)
        out[role] = RawFile(**{**asdict(rec), "role": role})
    if provenance is None:
        store.save()
    return out


def download_runs(
    runs: Iterable[int],
    raw_dir: Path = DEFAULT_RAW_DIR,
    *,
    workers: int = 4,
    revision: str | None = None,
    roles: Iterable[str] | None = None,
    progress: ProgressFn | None = None,
) -> dict[int, dict[str, RawFile]]:
    """Download several runs concurrently. Returns ``{run: {role: RawFile}}`` sorted by run."""
    runs = list(runs)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    store = ProvenanceStore(raw_dir)
    results: dict[int, dict[str, RawFile]] = {}
    role_list = list(roles) if roles is not None else None
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {
            ex.submit(
                download_run, r, raw_dir, roles=role_list, revision=revision, provenance=store
            ): r
            for r in runs
        }
        done = 0
        for fut in as_completed(futs):
            r = futs[fut]
            results[r] = fut.result()
            done += 1
            if progress:
                progress(f"downloaded run_{r}", done, len(runs))
            if done % 5 == 0:
                store.save()
    store.save()
    return dict(sorted(results.items()))


def parse_runs(spec: str) -> list[int]:
    """Parse ``"1-40"``, ``"1,3,5-8"`` into a sorted list of run numbers."""
    out: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if hi < lo:
                raise ValueError(f"bad range {part!r}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(out)

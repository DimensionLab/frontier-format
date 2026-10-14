# SPDX-License-Identifier: Apache-2.0
"""Immutable dataset versions with structural sharing, provenance, diff, revert and lineage
(``docs/spec/data-prep.md`` sections 3 to 5).

Layout inside a dataset directory::

    <dataset>/.frontier/
      objects/<aa>/<sha256>      content-addressed, read-only files (zarr chunks, json, ...)
      trees/<sha256>.json        per-sample subtrees {relative path: [hash, size]}
      versions/<id>.json         {"id","parent","created_at","message","actor","op_record",
                                  "tree": {"files": {...}, "samples": {sid: tree hash}}, "stats"}
      HEAD                       current version id
      refs/<name>                tags
      index.json                 working-tree cache {path: {hash, size, mtime_ns, ino}}
      provenance.jsonl           append-only records (version / op / assertion / approval / note)

Working-tree files are hardlinks to objects (copies where the filesystem refuses hardlinks),
so a checkout costs directory entries, not bytes, and an op that renames a field writes one
new ``zarr.json`` and no chunk. Object files are mode 0o444: any attempt to edit a chunk in
place fails loudly instead of silently corrupting history (every writer in this package and
zarr 3 itself write whole files atomically, so normal use never trips this).

The version id is content-addressed (``v_`` + 16 hex of the sha256 of the sorted
``path\\0hash`` lines): the same bytes on two machines give the same id, which is what the
determinism tests of ``frontier_prep`` assert.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

STORE_DIR = ".frontier"
IGNORED_NAMES = {".DS_Store", "Thumbs.db", STORE_DIR}
HASH_BLOCK = 4 << 20
#: paths under a sample directory are stored in a per-sample subtree
SAMPLES_DIR = "samples"
DIFF_MAX_ELEMENTS = 4_000_000

ProgressFn = Callable[[int, int, str], None]


#: the tag ``prep.quality.gate`` puts on a version that passed
GATE_TAG = "train-ready"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(HASH_BLOCK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "model_dump"):
        return o.model_dump()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


# -- records ---------------------------------------------------------------------------------


@dataclass
class FileEntry:
    hash: str
    size: int


Tree = dict[str, FileEntry]  # path -> entry (paths are posix, relative to the dataset root)


@dataclass
class Version:
    id: str
    parent: str | None
    created_at: str
    message: str
    actor: dict[str, Any]
    op_record: str | None
    files: dict[str, list]  # non-sample files: path -> [hash, size]
    samples: dict[str, str]  # sample id -> subtree hash
    stats: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent": self.parent,
            "created_at": self.created_at,
            "message": self.message,
            "actor": self.actor,
            "op_record": self.op_record,
            "tree": {"files": self.files, "samples": self.samples},
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Version:
        tree = d.get("tree", {})
        return cls(
            id=d["id"],
            parent=d.get("parent"),
            created_at=d.get("created_at", ""),
            message=d.get("message", ""),
            actor=d.get("actor", {}),
            op_record=d.get("op_record"),
            files=dict(tree.get("files", {})),
            samples=dict(tree.get("samples", {})),
            stats=dict(d.get("stats", {})),
        )


@dataclass
class Status:
    head: str | None
    added: list[str]
    removed: list[str]
    modified: list[str]

    @property
    def clean(self) -> bool:
        return not (self.added or self.removed or self.modified)

    def to_dict(self) -> dict[str, Any]:
        return {
            "head": self.head,
            "clean": self.clean,
            "added": self.added,
            "removed": self.removed,
            "modified": self.modified,
        }


# -- the store -------------------------------------------------------------------------------


class VersionStore:
    """Versions of one dataset directory. ``VersionStore(root)`` opens or lazily creates the
    ``.frontier/`` store; :meth:`init` records the baseline version of an existing dataset."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        if not (self.root / "manifest.json").exists():
            raise FileNotFoundError(f"{self.root} is not a frontier-dataset (no manifest.json)")
        self.dir = self.root / STORE_DIR
        self.objects = self.dir / "objects"
        self.trees = self.dir / "trees"
        self.versions_dir = self.dir / "versions"
        self.refs = self.dir / "refs"
        self._index: dict[str, dict[str, Any]] | None = None
        self._hardlinks: bool | None = None

    # -- state ---------------------------------------------------------------------------

    @property
    def initialized(self) -> bool:
        return (self.dir / "HEAD").exists()

    @property
    def head(self) -> str | None:
        p = self.dir / "HEAD"
        return p.read_text().strip() or None if p.exists() else None

    def _set_head(self, version_id: str) -> None:
        self.dir.mkdir(exist_ok=True)
        tmp = self.dir / "HEAD.tmp"
        tmp.write_text(version_id + "\n")
        tmp.replace(self.dir / "HEAD")

    def _ensure_dirs(self) -> None:
        for d in (self.dir, self.objects, self.trees, self.versions_dir, self.refs):
            d.mkdir(parents=True, exist_ok=True)

    # -- index -----------------------------------------------------------------------------

    @property
    def index(self) -> dict[str, dict[str, Any]]:
        if self._index is None:
            p = self.dir / "index.json"
            self._index = json.loads(p.read_text()) if p.exists() else {}
        assert self._index is not None
        return self._index

    def _save_index(self) -> None:
        self.dir.mkdir(exist_ok=True)
        tmp = self.dir / "index.json.tmp"
        tmp.write_text(json.dumps(self._index or {}, separators=(",", ":")))
        tmp.replace(self.dir / "index.json")

    # -- scanning ----------------------------------------------------------------------------

    def _walk(self) -> Iterator[tuple[str, os.stat_result]]:
        """Every regular file of the working tree (posix relative path, stat)."""
        stack = [self.root]
        while stack:
            d = stack.pop()
            try:
                with os.scandir(d) as it:
                    entries = sorted(it, key=lambda e: e.name)
            except FileNotFoundError:
                continue
            for e in entries:
                if e.name in IGNORED_NAMES:
                    continue
                if e.is_dir(follow_symlinks=False):
                    stack.append(Path(e.path))
                elif e.is_file(follow_symlinks=False):
                    st = e.stat(follow_symlinks=False)
                    yield Path(e.path).relative_to(self.root).as_posix(), st

    def scan(self, progress: ProgressFn | None = None, workers: int = 8) -> Tree:
        """Hash the working tree, reusing the index for files whose (size, mtime, inode) did
        not change. Returns ``{path: FileEntry}`` and updates the index."""
        index = self.index
        tree: Tree = {}
        todo: list[tuple[str, os.stat_result]] = []
        for rel, st in self._walk():
            cached = index.get(rel)
            if (
                cached
                and cached.get("size") == st.st_size
                and cached.get("mtime_ns") == st.st_mtime_ns
                and cached.get("ino") == st.st_ino
            ):
                tree[rel] = FileEntry(cached["hash"], st.st_size)
            else:
                todo.append((rel, st))
        if todo:
            paths = [self.root / rel for rel, _ in todo]
            with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
                hashes = list(ex.map(sha256_file, paths))
            for (rel, st), h in zip(todo, hashes, strict=True):
                tree[rel] = FileEntry(h, st.st_size)
                index[rel] = {
                    "hash": h,
                    "size": st.st_size,
                    "mtime_ns": st.st_mtime_ns,
                    "ino": st.st_ino,
                }
            if progress:
                progress(len(todo), len(todo), f"hashed {len(todo)} files")
        for rel in [r for r in index if r not in tree]:
            del index[rel]
        self._save_index()
        return tree

    # -- objects ------------------------------------------------------------------------------

    def object_path(self, h: str) -> Path:
        return self.objects / h[:2] / h

    def _can_hardlink(self) -> bool:
        if self._hardlinks is None:
            self._ensure_dirs()
            probe = self.dir / ".probe"
            link = self.dir / ".probe.link"
            try:
                probe.write_text("x")
                if link.exists():
                    link.unlink()
                os.link(probe, link)
                self._hardlinks = True
            except OSError:
                self._hardlinks = False
            finally:
                for p in (probe, link):
                    if p.exists():
                        p.unlink()
        return self._hardlinks

    def _intern(self, rel: str, entry: FileEntry) -> int:
        """Make ``rel`` share its bytes with the object store. Returns the bytes newly added
        to the store (0 when the object already existed)."""
        src = self.root / rel
        obj = self.object_path(entry.hash)
        added = 0
        if not obj.exists():
            obj.parent.mkdir(parents=True, exist_ok=True)
            if self._can_hardlink():
                try:
                    os.link(src, obj)
                except OSError:
                    shutil.copyfile(src, obj)
            else:
                shutil.copyfile(src, obj)
            os.chmod(obj, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            added = entry.size
        elif self._can_hardlink():
            try:
                same = os.stat(src).st_ino == os.stat(obj).st_ino
            except FileNotFoundError:
                same = False
            if not same:
                tmp = src.with_name(src.name + ".vtmp")
                os.link(obj, tmp)
                os.replace(tmp, src)
        st = os.stat(src)
        self.index[rel] = {
            "hash": entry.hash,
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "ino": st.st_ino,
        }
        return added

    def _place(self, rel: str, h: str) -> None:
        """Materialise object ``h`` at working-tree path ``rel`` (hardlink or copy)."""
        dst = self.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        obj = self.object_path(h)
        if self._can_hardlink():
            try:
                os.link(obj, dst)
            except OSError:
                shutil.copyfile(obj, dst)
        else:
            shutil.copyfile(obj, dst)
        st = os.stat(dst)
        self.index[rel] = {
            "hash": h,
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "ino": st.st_ino,
        }

    # -- trees ---------------------------------------------------------------------------------

    @staticmethod
    def _split(tree: Tree) -> tuple[dict[str, list], dict[str, dict[str, list]]]:
        files: dict[str, list] = {}
        samples: dict[str, dict[str, list]] = {}
        prefix = SAMPLES_DIR + "/"
        for rel in sorted(tree):
            e = tree[rel]
            if rel.startswith(prefix):
                rest = rel[len(prefix) :]
                sid, _, sub = rest.partition("/")
                if sub:
                    samples.setdefault(sid, {})[sub] = [e.hash, e.size]
                    continue
            files[rel] = [e.hash, e.size]
        return files, samples

    def _store_subtree(self, sub: dict[str, list]) -> str:
        text = _canonical_json(sub)
        h = sha256_bytes(text.encode())
        p = self.trees / f"{h}.json"
        if not p.exists():
            p.write_text(text)
        return h

    def subtree(self, h: str) -> dict[str, list]:
        return json.loads((self.trees / f"{h}.json").read_text())

    def flatten(self, v: Version) -> Tree:
        """Full ``{path: FileEntry}`` of a version."""
        out: Tree = {p: FileEntry(e[0], e[1]) for p, e in v.files.items()}
        for sid, th in v.samples.items():
            for sub, e in self.subtree(th).items():
                out[f"{SAMPLES_DIR}/{sid}/{sub}"] = FileEntry(e[0], e[1])
        return out

    @staticmethod
    def version_id(tree: Tree) -> str:
        lines = "".join(f"{p}\0{tree[p].hash}\n" for p in sorted(tree))
        return "v_" + sha256_bytes(lines.encode())[:16]

    # -- versions ----------------------------------------------------------------------------

    def version(self, version_id: str) -> Version:
        p = self.versions_dir / f"{self.resolve(version_id)}.json"
        if not p.exists():
            raise KeyError(f"unknown version {version_id!r}")
        v = Version.from_dict(json.loads(p.read_text()))
        v.tags = [n for n, t in self.tags().items() if t == v.id]
        return v

    def resolve(self, ref: str) -> str:
        """``HEAD``, a tag, a full id or a unique id prefix -> version id."""
        if ref in ("HEAD", "head", ""):
            h = self.head
            if h is None:
                raise KeyError("no HEAD: run `versions init` first")
            return h
        if ref.startswith("HEAD~"):
            n = int(ref[5:] or "1")
            v = self.version("HEAD")
            for _ in range(n):
                if v.parent is None:
                    raise KeyError(f"{ref}: history is shorter")
                v = self.version(v.parent)
            return v.id
        tags = self.tags()
        if ref in tags:
            return tags[ref]
        if (self.versions_dir / f"{ref}.json").exists():
            return ref
        cands = (
            [p.stem for p in self.versions_dir.glob(f"{ref}*.json")]
            if self.versions_dir.exists()
            else []
        )
        if len(cands) == 1:
            return cands[0]
        raise KeyError(f"unknown version {ref!r}" + (f" (ambiguous: {cands})" if cands else ""))

    def log(self, limit: int | None = None) -> list[Version]:
        out: list[Version] = []
        seen: set[str] = set()
        cur = self.head
        while cur is not None and cur not in seen and (limit is None or len(out) < limit):
            seen.add(cur)
            v = self.version(cur)
            out.append(v)
            cur = v.parent
        return out

    def all_versions(self) -> list[Version]:
        if not self.versions_dir.exists():
            return []
        vs = [
            Version.from_dict(json.loads(p.read_text())) for p in self.versions_dir.glob("*.json")
        ]
        return sorted(vs, key=lambda v: v.created_at)

    def init(
        self,
        message: str = "baseline",
        *,
        actor: dict[str, Any] | None = None,
        progress: ProgressFn | None = None,
    ) -> Version:
        """Record the baseline version of an existing dataset (migration)."""
        if self.initialized:
            return self.version("HEAD")
        self._ensure_dirs()
        manifest = json.loads((self.root / "manifest.json").read_text())
        rec = self.append_provenance(
            {
                "kind": "op",
                "op": "prep.baseline",
                "op_version": "1.0.0",
                "params": {"pipeline": manifest.get("pipeline"), "source": manifest.get("source")},
                "dataset": manifest.get("name"),
                "input_version": None,
                "summary": "baseline version of an existing dataset",
                "actor": actor or {"kind": "system", "name": "versions.init"},
            }
        )
        v = self.commit(message, op_record=rec, actor=actor, progress=progress, allow_empty=True)
        assert v is not None
        self.tag("baseline", v.id)
        self._update_record_output(rec, v.id)
        return self.version(v.id)

    def commit(
        self,
        message: str,
        *,
        op_record: str | None = None,
        actor: dict[str, Any] | None = None,
        progress: ProgressFn | None = None,
        allow_empty: bool = False,
        workers: int = 8,
    ) -> Version | None:
        """Record the working tree as a new version (``None`` when nothing changed)."""
        self._ensure_dirs()
        tree = self.scan(progress, workers=workers)
        vid = self.version_id(tree)
        parent = self.head
        if parent == vid and not allow_empty:
            return None
        if (self.versions_dir / f"{vid}.json").exists():
            # same content as an earlier version (a revert, or an idempotent op): re-point HEAD
            # to a new record so history stays linear and the parent is explicit
            pass
        new_bytes = 0
        n_new = 0
        for rel in sorted(tree):
            e = tree[rel]
            cached = self.index.get(rel)
            obj = self.object_path(e.hash)
            if cached and obj.exists():
                try:
                    if os.stat(self.root / rel).st_ino == os.stat(obj).st_ino:
                        continue
                except FileNotFoundError:
                    pass
            added = self._intern(rel, e)
            new_bytes += added
            n_new += 1 if added else 0
        self._save_index()
        files, samples = self._split(tree)
        sample_trees = {sid: self._store_subtree(sub) for sid, sub in samples.items()}
        if parent is not None and parent != vid:
            # keep ids unique per commit even if two commits have identical content but
            # different parents: fold the parent into the id
            vid_final = (
                vid if not (self.versions_dir / f"{vid}.json").exists() else _derive_id(vid, parent)
            )
        else:
            vid_final = vid
        v = Version(
            id=vid_final,
            parent=parent,
            created_at=now_iso(),
            message=message,
            actor=actor or {"kind": "human", "name": os.environ.get("USER", "")},
            op_record=op_record,
            files=files,
            samples=sample_trees,
            stats={
                "n_files": len(tree),
                "n_bytes": int(sum(e.size for e in tree.values())),
                "n_new_files": n_new,
                "n_new_bytes": new_bytes,
                "n_samples": len(sample_trees),
                "content_id": vid,
            },
        )
        (self.versions_dir / f"{v.id}.json").write_text(json.dumps(v.to_dict(), indent=1))
        self._set_head(v.id)
        self.append_provenance(
            {
                "kind": "version",
                "version": v.id,
                "parent": parent,
                "message": message,
                "op_record": op_record,
                "actor": v.actor,
                "stats": v.stats,
            }
        )
        return v

    def status(self) -> Status:
        head = self.head
        tree = self.scan()
        if head is None:
            return Status(None, sorted(tree), [], [])
        base = self.flatten(self.version(head))
        added = sorted(p for p in tree if p not in base)
        removed = sorted(p for p in base if p not in tree)
        modified = sorted(p for p in tree if p in base and base[p].hash != tree[p].hash)
        return Status(head, added, removed, modified)

    def checkout(self, ref: str) -> Version:
        """Working tree := version; HEAD moves; no new version is recorded."""
        v = self.version(ref)
        target = self.flatten(v)
        current = self.scan()
        for rel in current:
            if rel not in target:
                p = self.root / rel
                p.unlink(missing_ok=True)
                self.index.pop(rel, None)
        for rel, e in target.items():
            cur = current.get(rel)
            if cur is not None and cur.hash == e.hash:
                continue
            self._place(rel, e.hash)
        _prune_empty_dirs(self.root, exclude=self.dir)
        self._save_index()
        self._set_head(v.id)
        return v

    def revert(
        self, ref: str, message: str | None = None, *, actor: dict[str, Any] | None = None
    ) -> Version:
        """New version whose content equals ``ref`` (history is kept). Reverting to content
        equal to the head's is a no-op that returns the head (no self-parented version).

        The ``train-ready`` tag follows the revert when, and only when, the new HEAD's content
        (every file and sample subtree) is identical to the content of the version carrying
        the tag: the gate judged that content, so the judgment still holds, and a provenance
        note records the move. Otherwise HEAD stays untagged and the gate has to run again
        (:meth:`gate_status` tells where the tag is)."""
        target = self.version(ref)
        head = self.head
        if head is not None:
            current = self.version(head)
            if (current.files, current.samples) == (target.files, target.samples):
                self._gate_tag_follows(current)
                return self.version(current.id)
        self.checkout(target.id)
        self._set_head(head or target.id)
        rec = self.append_provenance(
            {
                "kind": "op",
                "op": "prep.revert",
                "op_version": "1.0.0",
                "params": {"to": target.id},
                "input_version": head,
                "summary": f"revert to {target.id}",
                "actor": actor or {"kind": "human", "name": os.environ.get("USER", "")},
            }
        )
        v = self.commit(
            message or f"revert to {target.id}", op_record=rec, actor=actor, allow_empty=True
        )
        assert v is not None
        self._update_record_output(rec, v.id)
        self._gate_tag_follows(v)
        return self.version(v.id)

    def _gate_tag_follows(self, v: Version) -> bool:
        """Move ``train-ready`` to ``v`` when its content equals the tagged version's."""
        tagged = self.tags().get(GATE_TAG)
        if tagged is None or tagged == v.id:
            return False
        try:
            gated = self.version(tagged)
        except KeyError:
            return False
        if (gated.files, gated.samples) != (v.files, v.samples):
            return False
        self.tag(GATE_TAG, v.id)
        self.append_provenance(
            {
                "kind": "note",
                "version": v.id,
                "tag": GATE_TAG,
                "from_version": gated.id,
                "note": (
                    f"{GATE_TAG} tag followed the revert: the content of {v.id} is identical "
                    f"to {gated.id}"
                ),
            }
        )
        return True

    def gate_status(self) -> dict[str, Any]:
        """``{"head", "tagged", "train_ready"}``: HEAD, the version carrying ``train-ready``
        (``None`` when no version does) and whether they are the same."""
        head = self.head
        tagged = self.tags().get(GATE_TAG)
        return {"head": head, "tagged": tagged, "train_ready": head is not None and tagged == head}

    # -- tags ------------------------------------------------------------------------------------

    def tag(self, name: str, ref: str = "HEAD") -> str:
        self._ensure_dirs()
        vid = self.resolve(ref)
        # temp file + rename: a hardlink clone of the store must not see this tag
        tmp = self.refs / f".{name}.tmp"
        tmp.write_text(vid + "\n")
        tmp.replace(self.refs / name)
        return vid

    def untag(self, name: str) -> None:
        (self.refs / name).unlink(missing_ok=True)

    def tags(self) -> dict[str, str]:
        if not self.refs.exists():
            return {}
        return {p.name: p.read_text().strip() for p in sorted(self.refs.iterdir()) if p.is_file()}

    # -- provenance -----------------------------------------------------------------------------

    @property
    def provenance_path(self) -> Path:
        return self.dir / "provenance.jsonl"

    def append_provenance(self, record: dict[str, Any]) -> str:
        self.dir.mkdir(exist_ok=True)
        rec = dict(record)
        rec.setdefault("at", now_iso())
        rec.setdefault("actor", {"kind": "system", "name": ""})
        body = _canonical_json({k: v for k, v in rec.items() if k != "id"})
        rec["id"] = sha256_bytes(body.encode())[:16]
        _unshare(self.provenance_path)
        with open(self.provenance_path, "a") as f:
            f.write(json.dumps(rec, default=_json_default, separators=(",", ":")) + "\n")
        return rec["id"]

    def _update_record_output(self, rec_id: str, version_id: str) -> None:
        """Ops record their output version in a follow-up record (the log is append-only)."""
        self.append_provenance(
            {"kind": "note", "ref": rec_id, "output_version": version_id, "note": "output"}
        )

    def provenance(self, kind: str | None = None) -> list[dict[str, Any]]:
        if not self.provenance_path.exists():
            return []
        out = []
        with open(self.provenance_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if kind is None or rec.get("kind") == kind:
                    out.append(rec)
        return out

    def record(self, rec_id: str) -> dict[str, Any] | None:
        for rec in self.provenance():
            if rec.get("id") == rec_id:
                return rec
        return None

    def assertions(self, version: str | None = None) -> list[dict[str, Any]]:
        vid = self.resolve(version or "HEAD")
        return [r for r in self.provenance("assertion") if r.get("version") == vid]

    def effective_assertions(self, version: str | None = None) -> list[dict[str, Any]]:
        """Assertions that still hold at ``version``: the latest record per (check, target)
        walking back the chain, kept only while the content it was made on is unchanged (a
        sample's subtree for ``samples/<id>`` targets; the whole sample set for dataset-level
        targets). Metadata-only versions (normalisation statistics, splits) therefore keep the
        closure evidence of earlier checks."""
        head = self.version(version or "HEAD")
        seen: set[tuple[str, str]] = set()
        out: list[dict[str, Any]] = []
        by_version: dict[str, list[dict[str, Any]]] = {}
        for rec in self.provenance("assertion"):
            by_version.setdefault(str(rec.get("version")), []).append(rec)
        for v in self.log():
            for rec in by_version.get(v.id, []):
                target = str(rec.get("target", "dataset"))
                key = (str(rec.get("check")), target)
                if key in seen:
                    continue
                if target.startswith("samples/"):
                    sid = target.split("/")[1]
                    if v.samples.get(sid) != head.samples.get(sid):
                        continue
                elif v.samples != head.samples:
                    continue
                seen.add(key)
                out.append(rec)
        return out

    def approvals(self, version: str | None = None) -> list[dict[str, Any]]:
        vid = self.resolve(version or "HEAD")
        return [r for r in self.provenance("approval") if r.get("version") == vid]

    def approve(self, ref: str, approved_by: str, decision: str = "approve", note: str = "") -> str:
        vid = self.resolve(ref)
        return self.append_provenance(
            {
                "kind": "approval",
                "version": vid,
                "approved_by": approved_by,
                "decision": decision,
                "note": note,
                "actor": {"kind": "human", "name": approved_by},
            }
        )

    def lineage(self, target: str = "manifest.json") -> list[dict[str, Any]]:
        """Versions (oldest first) in which ``target`` (a path or a path prefix such as
        ``samples/run_1`` or ``samples/run_1/surface.zarr/fields/p``) changed, with the op
        record that produced each."""
        out: list[dict[str, Any]] = []
        prev: dict[str, str] = {}
        for v in reversed(self.log()):
            cur = {
                p: e.hash
                for p, e in self.flatten(v).items()
                if p == target or p.startswith(target.rstrip("/") + "/")
            }
            if cur != prev:
                rec = self.record(v.op_record) if v.op_record else None
                out.append(
                    {
                        "version": v.id,
                        "created_at": v.created_at,
                        "message": v.message,
                        "op": rec.get("op") if rec else None,
                        "op_record": v.op_record,
                        "params": rec.get("params") if rec else None,
                        "n_files": len(cur),
                        "added": len([p for p in cur if p not in prev]),
                        "removed": len([p for p in prev if p not in cur]),
                        "modified": len([p for p in cur if p in prev and prev[p] != cur[p]]),
                    }
                )
            prev = cur
        return out

    # -- gc -----------------------------------------------------------------------------------

    def gc(self, *, dry_run: bool = False) -> dict[str, Any]:
        """Delete objects and subtrees referenced by no version and not in the working tree."""
        live: set[str] = set()
        live_trees: set[str] = set()
        for v in self.all_versions():
            live.update(e[0] for e in v.files.values())
            for th in v.samples.values():
                live_trees.add(th)
                live.update(e[0] for e in self.subtree(th).values())
        live.update(e["hash"] for e in self.index.values())
        removed, freed = 0, 0
        if self.objects.exists():
            for p in self.objects.glob("*/*"):
                if p.name not in live:
                    freed += p.stat().st_size
                    removed += 1
                    if not dry_run:
                        p.unlink()
        removed_trees = 0
        if self.trees.exists():
            for p in self.trees.glob("*.json"):
                if p.stem not in live_trees:
                    removed_trees += 1
                    if not dry_run:
                        p.unlink()
        return {
            "removed_objects": removed,
            "freed_bytes": freed,
            "removed_trees": removed_trees,
            "dry_run": dry_run,
        }

    # -- materialise ----------------------------------------------------------------------------

    def materialize(
        self, ref: str, dest: str | Path, *, samples: Iterable[str] | None = None
    ) -> Path:
        """Hardlink checkout of a version into ``dest`` (nothing is copied when hardlinks
        work). ``samples`` restricts the sample directories; the manifest is always included."""
        v = self.version(ref)
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        wanted = None if samples is None else set(samples)
        for rel, e in v.files.items():
            _link_or_copy(self.object_path(e[0]), dest / rel)
        for sid, th in v.samples.items():
            if wanted is not None and sid not in wanted:
                continue
            for sub, e in self.subtree(th).items():
                _link_or_copy(self.object_path(e[0]), dest / SAMPLES_DIR / sid / sub)
        return dest

    # -- diff -----------------------------------------------------------------------------------

    def diff(
        self,
        a: str,
        b: str = "HEAD",
        *,
        arrays: bool = True,
        max_elements: int = DIFF_MAX_ELEMENTS,
    ) -> dict[str, Any]:
        """Structural + statistical diff between two versions (section 5 of the spec)."""
        va, vb = self.version(a), self.version(b)
        ta, tb = self.flatten(va), self.flatten(vb)
        added = sorted(p for p in tb if p not in ta)
        removed = sorted(p for p in ta if p not in tb)
        modified = sorted(p for p in tb if p in ta and ta[p].hash != tb[p].hash)
        out: dict[str, Any] = {
            "a": va.id,
            "b": vb.id,
            "files": {"added": len(added), "removed": len(removed), "modified": len(modified)},
            "manifest": {},
            "samples": {"added": [], "removed": [], "modified": {}},
            "scalars": {},
        }
        if "manifest.json" in modified or "manifest.json" in added:
            out["manifest"] = _diff_manifest(
                self._read_json(ta.get("manifest.json")), self._read_json(tb.get("manifest.json"))
            )
        sa, sb = set(va.samples), set(vb.samples)
        out["samples"]["added"] = sorted(sb - sa)
        out["samples"]["removed"] = sorted(sa - sb)
        changed_sids = sorted(s for s in sa & sb if va.samples[s] != vb.samples[s])
        if not changed_sids:
            return out
        with tempfile.TemporaryDirectory(prefix="fdiff-") as tmp:
            da = self.materialize(va.id, Path(tmp) / "a", samples=changed_sids)
            db = self.materialize(vb.id, Path(tmp) / "b", samples=changed_sids)
            for sid in changed_sids:
                sub_a, sub_b = self.subtree(va.samples[sid]), self.subtree(vb.samples[sid])
                entry: dict[str, Any] = {"files": _file_delta(sub_a, sub_b)}
                for name in ("meta.json", "forces.json"):
                    if name in sub_a and name in sub_b and sub_a[name][0] != sub_b[name][0]:
                        entry[name.split(".")[0]] = _diff_json(
                            json.loads((da / SAMPLES_DIR / sid / name).read_text()),
                            json.loads((db / SAMPLES_DIR / sid / name).read_text()),
                        )
                    elif name in sub_b and name not in sub_a:
                        entry[name.split(".")[0]] = {"added": True}
                if arrays:
                    entry["arrays"] = _diff_sample_arrays(
                        da / SAMPLES_DIR / sid, db / SAMPLES_DIR / sid, sub_a, sub_b, max_elements
                    )
                out["samples"]["modified"][sid] = entry
        out["scalars"] = _scalar_summary(out["samples"]["modified"])
        return out

    def _read_json(self, e: FileEntry | None) -> dict[str, Any]:
        if e is None:
            return {}
        return json.loads(self.object_path(e.hash).read_text())


# -- helpers ---------------------------------------------------------------------------------


def _unshare(path: Path) -> None:
    """Give ``path`` its own inode before an in-place write. A hardlink copy of a dataset
    (``cp -al``, an evaluation overlay) shares the inode of every file, and an append to the
    provenance log would otherwise land in the original's log as well."""
    try:
        if os.stat(path).st_nlink <= 1:
            return
    except FileNotFoundError:
        return
    tmp = path.with_name(path.name + ".unshare")
    shutil.copyfile(path, tmp)
    os.replace(tmp, path)


#: store entries a clone links instead of copying (immutable, content-addressed)
_CLONE_LINKED = ("objects", "trees")


def clone(src: str | Path, dst: str | Path) -> Path:
    """A cheap independent copy of a versioned dataset: the working tree, the objects and the
    trees are hardlinked (immutable or replaced atomically by every writer), the store's
    mutable metadata (HEAD, refs, index, version records, provenance log, recipes) is copied,
    so nothing done to the clone can reach the original. ``dst`` must not exist."""
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(dst)

    def link(a: str, b: str) -> None:
        try:
            os.link(a, b)
        except OSError:
            shutil.copy2(a, b)

    shutil.copytree(src, dst, copy_function=link, ignore=shutil.ignore_patterns(STORE_DIR))
    store = src / STORE_DIR
    if store.is_dir():
        (dst / STORE_DIR).mkdir()
        for entry in sorted(store.iterdir()):
            target = dst / STORE_DIR / entry.name
            if entry.is_dir():
                fn = link if entry.name in _CLONE_LINKED else shutil.copy2
                shutil.copytree(entry, target, copy_function=fn)
            else:
                shutil.copy2(entry, target)
        # the index remembers inodes of the source tree; they are the same files here
    return dst


def _derive_id(content_id: str, parent: str) -> str:
    return "v_" + sha256_bytes(f"{content_id}\0{parent}".encode())[:16]


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _prune_empty_dirs(root: Path, exclude: Path) -> None:
    for d in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
        if d == exclude or exclude in d.parents:
            continue
        try:
            d.rmdir()
        except OSError:
            pass


def _file_delta(a: dict[str, list], b: dict[str, list]) -> dict[str, int]:
    return {
        "added": len([p for p in b if p not in a]),
        "removed": len([p for p in a if p not in b]),
        "modified": len([p for p in b if p in a and a[p][0] != b[p][0]]),
    }


def _diff_json(a: Any, b: Any, prefix: str = "") -> dict[str, Any]:
    """Key-level diff of two JSON documents: ``{key: [before, after]}`` for scalars, nested
    for dicts; lists are compared as wholes unless they are lists of scalars (set delta)."""
    out: dict[str, Any] = {}
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out[prefix + k] = [None, b[k]]
            elif k not in b:
                out[prefix + k] = [a[k], None]
            elif a[k] != b[k]:
                sub = _diff_json(a[k], b[k], prefix + k + ".")
                if isinstance(a[k], dict) and isinstance(b[k], dict):
                    out.update(sub)
                else:
                    out.update(sub)
        return out
    if (
        isinstance(a, list)
        and isinstance(b, list)
        and all(not isinstance(x, dict | list) for x in a + b)
    ):
        sa, sb = set(map(str, a)), set(map(str, b))
        key = prefix.rstrip(".")
        if sa != sb:
            out[key] = {"added": sorted(sb - sa), "removed": sorted(sa - sb)}
        elif a != b:
            out[key] = {"reordered": True}
        return out
    key = prefix.rstrip(".")
    if a != b:
        out[key] = [a, b] if not isinstance(a, list | dict) else {"changed": True}
    return out


def _diff_manifest(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "changed_keys": sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    }
    fa = {f["name"]: f for f in a.get("fields", [])}
    fb = {f["name"]: f for f in b.get("fields", [])}
    if fa != fb:

        def same_shape(x: dict[str, Any], y: dict[str, Any]) -> bool:
            return (x.get("channels"), x.get("location")) == (y.get("channels"), y.get("location"))

        removed = [n for n in fa if n not in fb]
        added = [n for n in fb if n not in fa]
        renamed = []
        for r in list(removed):
            for ad in list(added):
                if same_shape(fa[r], fb[ad]):
                    renamed.append([r, ad])
                    removed.remove(r)
                    added.remove(ad)
                    break
        changed = {n: _diff_json(fa[n], fb[n]) for n in fa if n in fb and fa[n] != fb[n]}
        out["fields"] = {"renamed": renamed, "added": added, "removed": removed, "changed": changed}
    if a.get("splits") != b.get("splits"):
        out["splits"] = {
            k: {
                "added": sorted(
                    set(b.get("splits", {}).get(k, [])) - set(a.get("splits", {}).get(k, []))
                ),
                "removed": sorted(
                    set(a.get("splits", {}).get(k, [])) - set(b.get("splits", {}).get(k, []))
                ),
            }
            for k in sorted(set(a.get("splits", {})) | set(b.get("splits", {})))
        }
    sa = {s["id"] for s in a.get("samples", [])}
    sb = {s["id"] for s in b.get("samples", [])}
    if sa != sb:
        out["samples"] = {"added": sorted(sb - sa), "removed": sorted(sa - sb)}
    for key in ("split_strategy", "stats", "parameters", "conditions", "geometry_params"):
        if a.get(key) != b.get(key):
            out[key] = _diff_json(a.get(key), b.get(key))
    return out


def _array_dirs(sub: dict[str, list]) -> dict[str, str]:
    """``{store/array/path: store}`` for every zarr array in a sample subtree."""
    out: dict[str, str] = {}
    for p in sub:
        if p.endswith("/zarr.json"):
            d = p[: -len("/zarr.json")]
            store = d.split("/", 1)[0]
            if "/" in d:
                out[d] = store
    return out


def array_stats(a: np.ndarray) -> dict[str, float]:
    a = np.asarray(a)
    if a.dtype.kind not in "fiub":
        return {"n": float(a.size)}
    x = a.astype(np.float64, copy=False).reshape(-1)
    finite = np.isfinite(x)
    nan_frac = float(1.0 - finite.mean()) if x.size else 0.0
    x = x[finite]
    if x.size == 0:
        return {"n": float(a.size), "nan_frac": nan_frac}
    q = np.percentile(x, [1, 50, 99])
    return {
        "n": float(a.size),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p01": float(q[0]),
        "p50": float(q[1]),
        "p99": float(q[2]),
        "nan_frac": nan_frac,
    }


def _subsample(a: np.ndarray, max_elements: int) -> np.ndarray:
    if a.size <= max_elements:
        return a
    flat = a.reshape(-1)
    idx = np.linspace(0, flat.size - 1, max_elements).astype(np.int64)
    return flat[idx]


def _read_array(path: Path) -> np.ndarray | None:
    import zarr

    try:
        arr = zarr.open_array(str(path), mode="r")
    except Exception:
        return None
    return np.asarray(arr[...])


def compare_arrays(a: np.ndarray | None, b: np.ndarray | None, max_elements: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if a is None and b is None:
        return {"note": "unreadable"}
    if a is None:
        assert b is not None
        out.update(
            {"n_before": None, "n_after": int(b.size), "after": array_stats(b), "note": "added"}
        )
        return out
    if b is None:
        out.update(
            {"n_before": int(a.size), "n_after": None, "before": array_stats(a), "note": "removed"}
        )
        return out
    out["n_before"], out["n_after"] = int(a.size), int(b.size)
    out["shape_before"], out["shape_after"] = list(a.shape), list(b.shape)
    out["before"], out["after"] = array_stats(a), array_stats(b)
    if a.shape != b.shape:
        out["note"] = "shape changed"
        return out
    if a.dtype.kind in "fiub" and b.dtype.kind in "fiub":
        xa = _subsample(a, max_elements).astype(np.float64)
        xb = _subsample(b, max_elements).astype(np.float64)
        both_nan = np.isnan(xa) & np.isnan(xb)
        changed = (xa != xb) & ~both_nan
        d = np.where(np.isfinite(xa) & np.isfinite(xb), xb - xa, 0.0)
        out["n_changed"] = int(changed.sum() * (a.size / max(xa.size, 1)))
        out["frac_changed"] = float(changed.mean()) if xa.size else 0.0
        out["max_abs_diff"] = float(np.abs(d).max()) if d.size else 0.0
        na = float(np.sqrt(np.sum(xa[np.isfinite(xa)] ** 2)))
        out["rel_l2_diff"] = float(np.sqrt(np.sum(d**2)) / na) if na > 0 else None
        if xa.size < a.size:
            out["subsampled"] = int(xa.size)
    else:
        out["n_changed"] = int(np.sum(a != b))
        out["frac_changed"] = float(np.mean(a != b))
    return out


def _diff_sample_arrays(
    da: Path, db: Path, sub_a: dict[str, list], sub_b: dict[str, list], max_elements: int
) -> dict[str, Any]:
    arrays_a, arrays_b = _array_dirs(sub_a), _array_dirs(sub_b)
    changed: set[str] = set()
    for p in set(sub_a) | set(sub_b):
        ea, eb = sub_a.get(p), sub_b.get(p)
        if ea is not None and eb is not None and ea[0] == eb[0]:
            continue
        # a changed file belongs to the deepest array directory that contains it
        parts = p.split("/")
        for k in range(len(parts) - 1, 0, -1):
            d = "/".join(parts[:k])
            if d in arrays_a or d in arrays_b:
                changed.add(d)
                break
    out: dict[str, Any] = {}
    for d in sorted(changed):
        a = _read_array(da / d) if d in arrays_a else None
        b = _read_array(db / d) if d in arrays_b else None
        out[d] = compare_arrays(a, b, max_elements)
    return out


def _scalar_summary(modified: dict[str, Any]) -> dict[str, Any]:
    keys: dict[str, int] = {}
    for entry in modified.values():
        for section in ("meta", "forces"):
            for k in entry.get(section, {}):
                keys[f"{section}.{k}"] = keys.get(f"{section}.{k}", 0) + 1
    return {"changed_keys": keys}


# -- migration ---------------------------------------------------------------------------------


def find_datasets(root: str | Path, *, skip: Iterable[str] = ("raw", "prep-work")) -> list[Path]:
    """Every ``frontier-dataset`` directory under ``root`` (a directory holding
    ``manifest.json``), depth-first, sorted; the store directory, ``raw`` download trees and
    the ``skip`` names are not descended into."""
    root = Path(root)
    skip_names = set(skip) | IGNORED_NAMES
    out: list[Path] = []
    stack = [root]
    while stack:
        d = stack.pop()
        if (d / "manifest.json").exists():
            out.append(d)
            continue
        try:
            children = sorted(p for p in d.iterdir() if p.is_dir() and not p.is_symlink())
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            continue
        stack.extend(c for c in children if c.name not in skip_names)
    return sorted(out)


def migrate(
    root: str | Path,
    *,
    message: str = "baseline (migration)",
    actor: dict[str, Any] | None = None,
    progress: ProgressFn | None = None,
    workers: int = 8,
) -> list[dict[str, Any]]:
    """Give every dataset under ``root`` a baseline version. Idempotent: an initialised store
    is reported with its HEAD and left alone (a dirty working tree is reported, not committed).
    Returns one row per dataset: ``{path, version, action: initialized|unchanged|dirty, stats}``."""
    rows: list[dict[str, Any]] = []
    for ds in find_datasets(root):
        store = VersionStore(ds)
        if store.initialized:
            st = store.status()
            rows.append(
                {
                    "path": str(ds),
                    "version": st.head,
                    "action": "unchanged" if st.clean else "dirty",
                    "stats": {} if st.clean else st.to_dict(),
                }
            )
            continue
        v = store.init(
            message,
            actor=actor or {"kind": "system", "name": "versions.migrate"},
            progress=progress,
        )
        rows.append({"path": str(ds), "version": v.id, "action": "initialized", "stats": v.stats})
    return rows


# -- PROV-JSON export --------------------------------------------------------------------------


def export_prov_json(store: VersionStore) -> dict[str, Any]:
    """W3C PROV-JSON view of a dataset's provenance: versions are ``entity``, op records are
    ``activity``, actors are ``agent``; ``wasGeneratedBy`` / ``used`` / ``wasDerivedFrom`` /
    ``wasAssociatedWith`` / ``wasAttributedTo`` carry the DAG. Auditors can load the output
    with standard PROV tooling (``prov`` on PyPI, ProvToolbox)."""
    prefix = {
        "frontier": "https://frontier.dimensionlab.org/prov#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
    }
    entity: dict[str, Any] = {}
    activity: dict[str, Any] = {}
    agent: dict[str, Any] = {}
    gen: dict[str, Any] = {}
    used: dict[str, Any] = {}
    derived: dict[str, Any] = {}
    assoc: dict[str, Any] = {}
    attributed: dict[str, Any] = {}
    outputs = {
        r["ref"]: r["output_version"]
        for r in store.provenance("note")
        if r.get("note") == "output" and r.get("ref")
    }
    n = 0

    def agent_id(actor: dict[str, Any]) -> str:
        kind, name = actor.get("kind", "system"), actor.get("name", "")
        aid = f"frontier:agent/{kind}/{name or 'anonymous'}"
        agent.setdefault(aid, {"prov:type": f"frontier:{kind}", "frontier:name": name})
        return aid

    for v in store.all_versions():
        eid = f"frontier:version/{v.id}"
        entity[eid] = {
            "prov:type": "frontier:DatasetVersion",
            "frontier:createdAt": v.created_at,
            "frontier:message": v.message,
            "frontier:nFiles": v.stats.get("n_files"),
            "frontier:nBytes": v.stats.get("n_bytes"),
            "frontier:tags": v.tags,
        }
        if v.parent:
            n += 1
            derived[f"_:d{n}"] = {
                "prov:generatedEntity": eid,
                "prov:usedEntity": f"frontier:version/{v.parent}",
            }
        if v.actor:
            n += 1
            attributed[f"_:at{n}"] = {"prov:entity": eid, "prov:agent": agent_id(v.actor)}
    for rec in store.provenance("op"):
        aid = f"frontier:op/{rec['id']}"
        activity[aid] = {
            "prov:type": "frontier:Operation",
            "frontier:op": rec.get("op"),
            "frontier:opVersion": rec.get("op_version"),
            "frontier:params": rec.get("params"),
            "frontier:summary": rec.get("summary"),
            "prov:startTime": rec.get("at"),
            "frontier:determinism": rec.get("determinism"),
        }
        if rec.get("actor"):
            n += 1
            assoc[f"_:as{n}"] = {"prov:activity": aid, "prov:agent": agent_id(rec["actor"])}
        if rec.get("input_version"):
            n += 1
            used[f"_:u{n}"] = {
                "prov:activity": aid,
                "prov:entity": f"frontier:version/{rec['input_version']}",
            }
        out_v = rec.get("output_version") or outputs.get(rec["id"])
        if out_v:
            n += 1
            gen[f"_:g{n}"] = {"prov:entity": f"frontier:version/{out_v}", "prov:activity": aid}
    for rec in store.provenance("assertion"):
        eid = f"frontier:assertion/{rec['id']}"
        entity[eid] = {
            "prov:type": "frontier:Assertion",
            "frontier:check": rec.get("check"),
            "frontier:status": rec.get("status"),
            "frontier:target": rec.get("target"),
            "frontier:detail": rec.get("detail"),
        }
        if rec.get("version"):
            n += 1
            derived[f"_:d{n}"] = {
                "prov:generatedEntity": eid,
                "prov:usedEntity": f"frontier:version/{rec['version']}",
            }
    for rec in store.provenance("approval"):
        aid = f"frontier:approval/{rec['id']}"
        activity[aid] = {
            "prov:type": "frontier:Approval",
            "frontier:decision": rec.get("decision"),
            "frontier:note": rec.get("note"),
            "prov:startTime": rec.get("at"),
        }
        n += 1
        assoc[f"_:as{n}"] = {
            "prov:activity": aid,
            "prov:agent": agent_id({"kind": "human", "name": rec.get("approved_by", "")}),
        }
        if rec.get("version"):
            n += 1
            used[f"_:u{n}"] = {
                "prov:activity": aid,
                "prov:entity": f"frontier:version/{rec['version']}",
            }
    return {
        "prefix": prefix,
        "entity": entity,
        "activity": activity,
        "agent": agent,
        "wasGeneratedBy": gen,
        "used": used,
        "wasDerivedFrom": derived,
        "wasAssociatedWith": assoc,
        "wasAttributedTo": attributed,
    }


def import_prov_json(prov: dict[str, Any]) -> list[dict[str, Any]]:
    """The op chain of a PROV-JSON export (:func:`export_prov_json`), oldest first: one row
    per ``frontier:Operation`` activity with ``op``, ``op_version``, ``params``,
    ``input_version`` and ``output_version``, ordered by following ``wasDerivedFrom`` from
    the version without a parent. This is what a replay needs to rebuild the versions
    without the original store: feed the rows to a recipe."""
    parent: dict[str, str] = {}
    for d in prov.get("wasDerivedFrom", {}).values():
        g, u = d.get("prov:generatedEntity", ""), d.get("prov:usedEntity", "")
        if g.startswith("frontier:version/") and u.startswith("frontier:version/"):
            parent[g.split("/", 1)[1]] = u.split("/", 1)[1]
    versions = [
        e.split("/", 1)[1] for e in prov.get("entity", {}) if e.startswith("frontier:version/")
    ]
    children: dict[str, list[str]] = {}
    for v in versions:
        children.setdefault(parent.get(v, ""), []).append(v)
    order: list[str] = []
    stack = list(reversed(children.get("", [])))
    while stack:
        v = stack.pop()
        order.append(v)
        stack.extend(reversed(children.get(v, [])))
    generated: dict[str, str] = {}
    for g in prov.get("wasGeneratedBy", {}).values():
        e, a = g.get("prov:entity", ""), g.get("prov:activity", "")
        if e.startswith("frontier:version/") and a.startswith("frontier:op/"):
            generated[e.split("/", 1)[1]] = a
    used: dict[str, str] = {}
    for u in prov.get("used", {}).values():
        e, a = u.get("prov:entity", ""), u.get("prov:activity", "")
        if e.startswith("frontier:version/") and a.startswith("frontier:op/"):
            used[a] = e.split("/", 1)[1]
    activity = prov.get("activity", {})
    rows: list[dict[str, Any]] = []
    for v in order:
        aid = generated.get(v)
        if aid is None:
            continue
        act = activity.get(aid, {})
        if act.get("prov:type") != "frontier:Operation":
            continue
        rows.append(
            {
                "op": act.get("frontier:op"),
                "op_version": act.get("frontier:opVersion"),
                "params": act.get("frontier:params") or {},
                "input_version": used.get(aid, parent.get(v)),
                "output_version": v,
                "record": aid.split("/", 1)[1],
            }
        )
    return rows

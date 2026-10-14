# SPDX-License-Identifier: Apache-2.0
"""Portable subprocess execution over Frontier's authenticated outbound HTTPS API.

The board stays on its owning server. Each remote tool receives immutable input blobs;
its output directory is verified and synchronized before the board advances.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

PROTOCOL = 1
MAX_UNPACKED_BYTES = 100 * 1024**3


def restore_file(source: Path, destination: Path) -> None:
    try:
        source.replace(destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        # Container /tmp and persistent volumes are different filesystems. Publish on
        # the destination filesystem only after the complete file has been copied.
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                with source.open("rb") as content:
                    shutil.copyfileobj(content, stream)
                stream.flush()
                os.fsync(stream.fileno())
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        source.unlink()


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unpack(archive: Path, root: Path) -> None:
    """Reject links, special files, duplicate members and traversal before writing."""
    with tarfile.open(archive, "r:") as tar:
        members = tar.getmembers()
        names: set[str] = set()
        size = 0
        for member in members:
            path = Path(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or not (member.isfile() or member.isdir())
                or member.name in names
            ):
                raise ValueError(f"Unsafe transfer member: {member.name}")
            names.add(member.name)
            size += member.size
            if size > MAX_UNPACKED_BYTES:
                raise ValueError("Transfer expands beyond the unpacked size limit")
        # Fresh extraction roots only: existing links must never redirect archive writes.
        root.mkdir(parents=True, exist_ok=False)
        tar.extractall(root, members=members, filter="data")


def pack(source: Path, archive: Path) -> str:
    with tarfile.open(archive, "w") as tar:
        paths = sorted(source.rglob("*")) if source.is_dir() else [source]
        for path in paths:
            if path.name == ".remote-job.json":
                continue
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise ValueError(f"Transfer requires regular files: {path}")
            name = path.relative_to(source) if source.is_dir() else Path("file")
            info = tar.gettarinfo(str(path), str(name))
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            info.mode = 0o755 if info.isdir() else 0o644
            if info.isfile():
                with path.open("rb") as stream:
                    tar.addfile(info, stream)
            else:
                tar.addfile(info)
    return digest_file(archive)


def replace(value: Any, mappings: dict[str, str]) -> Any:
    if isinstance(value, str):
        if not mappings:
            return value
        sources = "|".join(re.escape(s) for s in sorted(mappings, key=len, reverse=True))
        pattern = rf"(?<![\w/.-])(?:{sources})(?=$|[/\s\"'\],}}):])"
        return re.sub(pattern, lambda match: mappings[match.group()], value)
    if isinstance(value, list):
        return [replace(v, mappings) for v in value]
    if isinstance(value, dict):
        return {k: replace(v, mappings) for k, v in value.items()}
    return value


def paths_in(value: Any) -> set[Path]:
    if isinstance(value, str):
        if value.startswith("/") and "\n" not in value:
            try:
                path = Path(value)
                if path.exists():
                    return {path.absolute()}
            except OSError:
                pass
        try:
            parsed = json.loads(value.split("=", 1)[-1])
        except (ValueError, TypeError):
            return set()
        return paths_in(parsed) if parsed != value else set()
    if isinstance(value, list):
        return set().union(*(paths_in(v) for v in value))
    if isinstance(value, dict):
        return set().union(*(paths_in(v) for v in value.values()))
    return set()


def execute(
    request: dict[str, Any], root: Path, cancel_event: threading.Event | None = None
) -> dict[str, Any]:
    """Worker-side tool runner. Credentials and local server URLs never enter its child."""
    if request.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported cloud execution protocol")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="execute-", dir=root) as temporary:
        work = Path(temporary)
        mappings: dict[str, str] = {}
        for index, item in enumerate(request["inputs"]):
            digest = item["digest"]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Invalid input digest")
            archive = root / digest
            if digest_file(archive) != digest:
                raise ValueError("Input checksum mismatch")
            dest = work / f"input-{index}"
            unpack(archive, dest)
            mappings[item["path"]] = str(dest / "file" if item["file"] else dest)
        output = work / "output"
        output.mkdir()
        if request["output"] in mappings:
            shutil.copytree(mappings[request["output"]], output, dirs_exist_ok=True)
        mappings[request["output"]] = str(output)
        # Model and dataset manifests may themselves contain absolute source paths.
        for directory in work.glob("input-*"):
            for manifest in directory.rglob("*.json"):
                try:
                    data = json.loads(manifest.read_text())
                except (ValueError, UnicodeError):
                    continue
                manifest.write_text(json.dumps(replace(data, mappings)))
        before = {
            str(path): digest_file(path)
            for directory in work.glob("input-*")
            for path in directory.rglob("*")
            if path.is_file()
        }
        argv = replace(request["argv"], mappings)
        argv[0] = sys.executable
        if (
            len(argv) < 3
            or argv[1] != "-m"
            or argv[2]
            not in {
                "frontier_train",
                "frontier_eval",
                "frontier_data",
                "frontier_prep",
                "frontier_data.board_run",
            }
        ):
            raise ValueError("Unsupported remote tool")
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("FRONTIER_")
            and not any(part in k.upper() for part in ("TOKEN", "SECRET", "PASSWORD", "API_KEY"))
        }
        env.update(PYTHONUNBUFFERED="1", FRONTIER_PIPELINE_STEP=request["step"])
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=cancel_event is not None,
        )
        finished = threading.Event()
        if cancel_event is not None:

            def watch_cancel() -> None:
                while not finished.is_set():
                    if cancel_event.wait(0.1):
                        if proc.poll() is None:
                            if os.name == "posix":
                                with contextlib.suppress(ProcessLookupError):
                                    os.killpg(proc.pid, signal.SIGTERM)
                            else:
                                proc.terminate()
                            try:
                                proc.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                if os.name == "posix":
                                    with contextlib.suppress(ProcessLookupError):
                                        os.killpg(proc.pid, signal.SIGKILL)
                                else:
                                    proc.kill()
                        return

            threading.Thread(target=watch_cancel, daemon=True).start()
        assert proc.stdout is not None
        reverse = {v: k for k, v in mappings.items()}
        with (output / "remote-tool.log").open("w") as log:
            for line in proc.stdout:
                line = replace(line, reverse)
                print(line, end="", flush=True)
                log.write(line)
        code = proc.wait()
        finished.set()
        for index, _item in enumerate(request["inputs"]):
            directory = work / f"input-{index}"
            for path in directory.rglob("*"):
                if not path.is_file() or before.get(str(path)) == digest_file(path):
                    continue
                changed = (
                    output / ".frontier-input-changes" / str(index) / path.relative_to(directory)
                )
                changed.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, changed)
        # Restore original paths before synchronizing manifests back to the owner.
        for manifest in output.rglob("*.json"):
            try:
                data = json.loads(manifest.read_text())
            except (ValueError, UnicodeError):
                continue
            manifest.write_text(json.dumps(replace(data, reverse)))
        archive = work / "result.tar"
        digest = pack(output, archive)
        archive.replace(root / digest)
        return {"protocol": PROTOCOL, "digest": digest, "exit_code": code}


def progress(step: str, message: str, done: int = 0, total: int = 0) -> None:
    from frontier_data.board_run import emit

    data: dict[str, Any] = {"message": message}
    if total:
        data.update(done=done, total=total, fraction=min(1.0, done / total))
    emit("step.progress", step, data=data)


class Dedicated:
    def __init__(self, http: Any) -> None:
        self.http = http
        self.step = ""

    def call(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self.http.request(method, "/api/v1" + path, **kwargs)
        response.raise_for_status()
        return response.json()

    def info(self, digest: str) -> dict[str, Any]:
        return self.call("GET", f"/compute/transfers/{digest}")

    def upload(self, digest: str, path: Path) -> None:
        def chunks():
            size = path.stat().st_size
            done = 0
            with path.open("rb") as stream:
                while block := stream.read(4 * 1024**2):
                    yield block
                    done += len(block)
                    progress(self.step, "Staging inputs", done, size)

        self.call("PUT", f"/compute/transfers/{digest}", content=chunks())

    def submit(self, payload: dict[str, Any]) -> str:
        root = self.info("0" * 64)["root"]
        return self.call(
            "POST",
            "/jobs",
            json={
                "name": f"Remote: {payload['step']}",
                "backend": "local",
                "command": [
                    "python",
                    "-m",
                    "frontier_data.remote",
                    "worker",
                    "--root",
                    root,
                    "--json",
                    json.dumps(payload),
                ],
            },
        )["id"]

    def status(self, job: str) -> dict[str, Any]:
        status = self.call("GET", f"/jobs/{job}")
        if status["state"] not in {"succeeded", "failed", "cancelled"}:
            return {"finished": False, "state": status["state"]}
        response = self.http.get(f"/api/v1/jobs/{job}/log")
        response.raise_for_status()
        for line in reversed(response.text.splitlines()):
            try:
                result = json.loads(line)
            except ValueError:
                continue
            if isinstance(result, dict) and result.get("protocol") == PROTOCOL:
                return {"finished": True, "result": result}
        raise RuntimeError(f"Remote job {job} ended without results: {status['state']}")

    def cancel(self, job: str) -> None:
        self.call("POST", f"/jobs/{job}/cancel")

    def download(self, digest: str, path: Path) -> None:
        with self.http.stream("GET", f"/api/v1/compute/transfers/{digest}/content") as response:
            response.raise_for_status()
            with path.open("wb") as stream:
                for chunk in response.iter_bytes():
                    stream.write(chunk)


class Serverless:
    chunk_size = 4 * 1024**2

    def __init__(self, http: Any) -> None:
        self.http = http
        self.step = ""
        self.active: str | None = None

    def call(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self.http.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    def submit(self, payload: dict[str, Any]) -> str:
        return self.call(
            "POST", "/run", json={"input": {"execution": {"op": "execute", "request": payload}}}
        )["id"]

    def status(self, job: str) -> dict[str, Any]:
        value = self.call("GET", f"/status/{job}")
        if value["status"] not in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
            return {"finished": False, "state": value["status"]}
        if value["status"] != "COMPLETED":
            raise RuntimeError(f"Serverless job {job}: {value['status']}")
        output = value.get("output")
        if isinstance(output, list):
            output = output[-1] if output else {}
        if not isinstance(output, dict) or output.get("status") != 200:
            raise RuntimeError(f"Serverless worker rejected the request: {output}")
        return {"finished": True, "result": output["json"]}

    def cancel(self, job: str) -> None:
        self.call("POST", f"/cancel/{job}")

    def operation(self, **request: Any) -> dict[str, Any]:
        job = self.call("POST", "/run", json={"input": {"execution": request}})["id"]
        self.active = job
        try:
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                status = self.status(job)
                if status["finished"]:
                    return status["result"]
                time.sleep(1)
            raise TimeoutError("Serverless transfer timed out")
        except BaseException:
            self.cancel(job)
            raise
        finally:
            self.active = None

    def info(self, digest: str) -> dict[str, Any]:
        return self.operation(op="status", digest=digest)

    def upload(self, digest: str, path: Path) -> None:
        import base64

        with path.open("rb") as stream:
            offset = 0
            while data := stream.read(self.chunk_size):
                self.operation(
                    op="put", digest=digest, offset=offset, data=base64.b64encode(data).decode()
                )
                offset += len(data)
                progress(self.step, "Staging inputs", offset, path.stat().st_size)
        self.operation(op="commit", digest=digest, size=path.stat().st_size)

    def download(self, digest: str, path: Path) -> None:
        import base64

        size = self.info(digest)["size"]
        with path.open("wb") as stream:
            for offset in range(0, size, self.chunk_size):
                chunk = self.operation(op="get", digest=digest, offset=offset)
                stream.write(base64.b64decode(chunk["data"], validate=True))


def client(request: dict[str, Any]) -> int:
    import httpx

    target = request["target"]
    connections = json.loads(os.environ.get("FRONTIER_REMOTE_ENVIRONMENTS", "{}"))
    connection = connections.get(target)
    if not connection:
        raise ValueError(
            "Remote credentials unavailable; use a trusted worker and a ready environment"
        )
    serverless = connection["url"].startswith("https://api.runpod.ai/v2/")
    if serverless and request["argv"][1:4] != ["-m", "frontier_train", "infer"]:
        raise ValueError(
            "This serverless target supports inference. Choose a dedicated GPU for this step."
        )
    # Preserve the spelling in argv (macOS /tmp aliases /private/tmp). Resolving only
    # discovered paths would leave the original command pointing at the owner's disk.
    output = Path(request["output"]).absolute()
    output.mkdir(parents=True, exist_ok=True)
    with (
        httpx.Client(
            base_url=connection["url"].rstrip("/"),
            headers={"Authorization": f"Bearer {connection['token']}"},
            timeout=120,
        ) as http,
        tempfile.TemporaryDirectory() as temporary,
    ):
        transport = Serverless(http) if serverless else Dedicated(http)
        transport.step = request["step"]
        job: str | None = None

        def cancel(_signum: int, _frame: Any) -> None:
            if job:
                transport.cancel(job)
            raise KeyboardInterrupt("Remote execution cancelled")

        previous = {sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            candidates = paths_in([request["argv"][1:], request.get("context", {})])
            roots = sorted(p for p in candidates if p != output and not p.is_relative_to(output))
            roots = [p for p in roots if not any(p != q and p.is_relative_to(q) for q in roots)]
            if any(p.is_file() and p.name != ".remote-job.json" for p in output.rglob("*")):
                roots.append(output)
            inputs = []
            for index, path in enumerate(roots):
                archive = Path(temporary) / f"{index}.tar"
                print(f"Staging {path.name}", flush=True)
                digest = pack(path, archive)
                info = transport.info(digest)
                if not serverless and info.get("protocol") != PROTOCOL:
                    raise ValueError("Remote runtime does not support this execution protocol")
                if not info["exists"]:
                    transport.upload(digest, archive)
                inputs.append({"path": str(path), "digest": digest, "file": path.is_file()})
            payload = {
                "protocol": PROTOCOL,
                "inputs": inputs,
                "output": str(output),
                "argv": request["argv"],
                "step": request["step"],
            }
            record = output / ".remote-job.json"
            fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            prior = json.loads(record.read_text()) if record.exists() else {}
            if prior.get("fingerprint") == fingerprint and prior.get("target") == target:
                job = prior["job"]
            else:
                job = transport.submit(payload)
                record.write_text(
                    json.dumps({"job": job, "target": target, "fingerprint": fingerprint})
                )
            if not isinstance(job, str) or not job:
                raise ValueError("Remote runtime did not return a valid job identifier")
            print(f"Remote job {job}: queued on {target}", flush=True)
            deadline = time.monotonic() + connection.get("timeout_secs", 3600)
            last_state = ""
            while time.monotonic() < deadline:
                try:
                    status = transport.status(job)
                except httpx.TransportError:
                    progress(request["step"], "Reconnecting to remote job")
                    time.sleep(2)
                    continue
                if status.get("state") != last_state:
                    last_state = status.get("state", "")
                    progress(request["step"], f"Remote job: {last_state or 'synchronizing'}")
                if status["finished"]:
                    break
                time.sleep(2)
            else:
                transport.cancel(job)
                raise TimeoutError("Remote job timed out")
            result = status["result"]
            digest = result.get("digest", "")
            if (
                result.get("protocol") != PROTOCOL
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("Invalid remote result manifest")
            archive = Path(temporary) / "result.tar"
            progress(request["step"], "Synchronizing remote results")
            transport.download(digest, archive)
            if digest_file(archive) != digest:
                raise ValueError("Result checksum mismatch")
            restored = Path(temporary) / "restored"
            unpack(archive, restored)
            changes = restored / ".frontier-input-changes"
            if changes.exists():
                for group in changes.iterdir():
                    index = int(group.name)
                    if index < 0 or index >= len(inputs):
                        raise ValueError("Unexpected changed input")
                    item = inputs[index]
                    source = Path(item["path"])
                    for path in group.rglob("*"):
                        if not path.is_file():
                            continue
                        relative = path.relative_to(group)
                        destination = (
                            source
                            if item["file"] and str(relative) == "file"
                            else source / relative
                        )
                        allowed = source.parent if item["file"] else source
                        if not destination.resolve().is_relative_to(allowed.resolve()):
                            raise ValueError("Changed input leaves its source root")
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        restore_file(path, destination)
                shutil.rmtree(changes)
            for path in restored.rglob("*"):
                destination = output / path.relative_to(restored)
                if path.is_file():
                    if not destination.resolve().is_relative_to(output.resolve()):
                        raise ValueError("Result destination leaves the job output directory")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    restore_file(path, destination)
            log = output / "remote-tool.log"
            if log.exists():
                print(log.read_text(), end="", flush=True)
            record.unlink(missing_ok=True)
            return int(result["exit_code"])
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("worker", "client"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    request = json.loads(args.json)
    if args.mode == "worker":
        result = execute(request, args.root)
        print(json.dumps(result), flush=True)
        raise SystemExit(result["exit_code"])
    else:
        raise SystemExit(client(request))


if __name__ == "__main__":
    main()

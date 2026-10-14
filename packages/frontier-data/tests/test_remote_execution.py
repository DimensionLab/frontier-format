# SPDX-License-Identifier: Apache-2.0

import io
import json
import tarfile

import pytest

from frontier_data.remote import digest_file, execute, pack, paths_in, unpack


def test_content_address_changes_with_checkpoint_not_mtime(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    checkpoint = model / "last.pt"
    checkpoint.write_bytes(b"checkpoint one")
    first = pack(model, tmp_path / "one.tar")
    checkpoint.touch()
    assert pack(model, tmp_path / "two.tar") == first
    checkpoint.write_bytes(b"checkpoint two")
    assert pack(model, tmp_path / "three.tar") != first


@pytest.mark.parametrize(
    "name,kind",
    [("../outside", "file"), ("/outside", "file"), ("link", "link"), ("device", "device")],
)
def test_archive_rejects_unsafe_members_before_writing(tmp_path, name, kind):
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as tar:
        entry = tarfile.TarInfo(name)
        entry.type = {"file": tarfile.REGTYPE, "link": tarfile.SYMTYPE, "device": tarfile.CHRTYPE}[
            kind
        ]
        entry.linkname = "/tmp"
        tar.addfile(entry, io.BytesIO())
    root = tmp_path / "extract"
    with pytest.raises(ValueError, match="Unsafe transfer"):
        unpack(archive, root)
    assert not root.exists()


def test_worker_executes_tool_and_returns_verifiable_archive(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({"path": str(source)}))
    archive = tmp_path / "source.tar"
    digest = pack(source, archive)
    archive.rename(tmp_path / digest)
    request = {
        "protocol": 1,
        "inputs": [{"path": str(source), "digest": digest, "file": False}],
        "argv": ["python", "-m", "frontier_data", "--help"],
        "output": "/original/output",
        "step": "infer",
    }
    result = execute(request, tmp_path)
    assert result["exit_code"] == 0
    assert digest_file(tmp_path / result["digest"]) == result["digest"]
    unpack(tmp_path / result["digest"], tmp_path / "result")
    assert "Usage" in (tmp_path / "result/remote-tool.log").read_text()
    assert json.loads((source / "manifest.json").read_text())["path"] == str(source)


def test_input_discovery_includes_nested_json_and_overrides(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    assert paths_in(
        [json.dumps({"model": str(checkpoint)}), "model=" + json.dumps(str(checkpoint))]
    ) == {checkpoint}


def test_no_writes_to_preexisting_extraction_root(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    archive = tmp_path / "empty.tar"
    pack(source, archive)
    with pytest.raises(FileExistsError):
        unpack(archive, source)


def test_client_synchronizes_before_returning_and_cancels_on_signal(tmp_path, monkeypatch):
    import signal

    from frontier_data import remote

    result_dir = tmp_path / "remote-output"
    result_dir.mkdir()
    (result_dir / "predictions.json").write_text('{"pressure": 3.5}')
    archive = tmp_path / "result.tar"
    digest = pack(result_dir, archive)
    cancelled = []
    handlers = {}
    interrupt = False

    class Transport:
        def __init__(self, http):
            pass

        def submit(self, payload):
            assert payload["output"] == str(tmp_path / "local-output")
            return "remote-job-1"

        def status(self, job):
            if interrupt:
                handlers[signal.SIGTERM](signal.SIGTERM, None)
            return {"finished": True, "result": {"protocol": 1, "digest": digest, "exit_code": 0}}

        def cancel(self, job):
            cancelled.append(job)

        def download(self, digest, path):
            path.write_bytes(archive.read_bytes())

    monkeypatch.setattr(remote, "Dedicated", Transport)
    monkeypatch.setenv(
        "FRONTIER_REMOTE_ENVIRONMENTS",
        json.dumps({"gpu": {"url": "http://127.0.0.1:9999", "token": "test-token"}}),
    )
    monkeypatch.setattr(signal, "signal", lambda sig, fn: handlers.setdefault(sig, fn))
    request = {
        "target": "gpu",
        "output": str(tmp_path / "local-output"),
        "argv": ["python", "-m", "frontier_data", "--help"],
        "step": "infer",
    }
    assert remote.client(request) == 0
    assert (tmp_path / "local-output/predictions.json").read_text() == '{"pressure": 3.5}'
    (tmp_path / "local-output/predictions.json").unlink()
    interrupt = True
    handlers.clear()
    with pytest.raises(KeyboardInterrupt, match="cancelled"):
        remote.client(request)
    assert cancelled == ["remote-job-1"]


def test_serverless_cancellation_terminates_tool_process(tmp_path):
    import threading

    cancelled = threading.Event()
    cancelled.set()
    result = execute(
        {
            "protocol": 1,
            "inputs": [],
            "output": "/output",
            "step": "infer",
            "argv": ["python", "-m", "frontier_data", "--help"],
        },
        tmp_path,
        cancelled,
    )
    assert result["exit_code"] != 0


def test_path_mapping_respects_boundaries_and_does_not_remap_replacements():
    from frontier_data.remote import replace

    mapping = {"/local/data": "/remote/data", "/remote": "/elsewhere"}
    assert replace("/local/database/file", mapping) == "/local/database/file"
    assert replace('path="/local/data/file"', mapping) == 'path="/remote/data/file"'
    assert replace(["/local/data", "/remote/x"], mapping) == ["/remote/data", "/elsewhere/x"]


def test_input_discovery_preserves_path_alias_used_in_command(tmp_path):
    data = tmp_path / "actual"
    data.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(data, target_is_directory=True)
    assert paths_in(str(alias)) == {alias}


def test_result_restoration_crosses_filesystems_atomically(tmp_path, monkeypatch):
    import errno
    import os

    from frontier_data.remote import restore_file

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_bytes(b"verified prediction")
    destination.write_bytes(b"old prediction")
    original = os.replace

    def different_filesystem(src, dst):
        if src == source:
            raise OSError(errno.EXDEV, "different filesystem")
        return original(src, dst)

    monkeypatch.setattr(os, "replace", different_filesystem)
    restore_file(source, destination)
    assert destination.read_bytes() == b"verified prediction"
    assert not source.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["destination"]

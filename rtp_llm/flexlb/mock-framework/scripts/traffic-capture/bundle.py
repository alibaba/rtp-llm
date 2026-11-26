"""Signed capture control and sealed archives on the Frontend Pod network."""

import base64
import fcntl
import gzip
import hashlib
import hmac
import json
import os
import re
import tarfile
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 20557
CHUNK_BYTES = 3 * 1024 * 1024
DRAIN_MS = 60_000
SHARD = re.compile(r"worker-(\d+)-(\d+)-[0-9a-f]{32}\.jsonl\.gz")


def configured_key():
    value = os.environ.get("MOCK_CAPTURE_KEY")
    if value is None:
        return None
    key = bytes.fromhex(value)
    if len(key) != 32:
        raise ValueError("invalid capture key")
    return key


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        partial = Path(temp.name)
        temp.write(json.dumps(value, separators=(",", ":")).encode())
    partial.replace(path)


def signature(payload, key):
    return hmac.digest(key, b"frontend-capture-v1\n" + payload, "sha256").hex()


def vip_members(domain):
    from rtp_llm.vipserver.vip_client import get_host_list_by_domain_now

    return {(host.ip, int(host.port)) for host in get_host_list_by_domain_now(domain)}


def state(directory, workers):
    sealed = sorted(directory.glob("worker-*.jsonl.gz"))
    if sorted(int(SHARD.fullmatch(p.name)[1]) for p in sealed) == list(range(workers)):
        return "sealed", sealed
    active = []
    for path in directory.glob("worker-*.jsonl.gz.part"):
        match = SHARD.fullmatch(path.name.removesuffix(".part"))
        if not match:
            continue
        try:
            os.kill(int(match[2]), 0)
        except ProcessLookupError:
            continue
        active.append(int(match[1]))
    return ("armed" if sorted(active) == list(range(workers)) else "starting"), []


def archive_manifest(directory, start, workers):
    status, files = state(directory, workers)
    if status != "sealed" or list(directory.glob("worker-*.jsonl.gz.part")):
        raise ValueError("capture_not_sealed")
    manifest = directory / "export.json"
    if manifest.is_file():
        return json.loads(manifest.read_text())
    with (directory / ".export.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest.is_file():
            return json.loads(manifest.read_text())
        archive = directory / "pod.tar"
        with tempfile.NamedTemporaryFile(
            dir=directory, prefix=".pod-", delete=False
        ) as temp:
            partial = Path(temp.name)
        try:
            with tarfile.open(partial, "w") as output:
                for path in files:
                    if path.is_symlink() or not path.is_file():
                        raise ValueError("invalid_capture_shard")
                    with gzip.open(path, "rt") as stream:
                        if json.loads(stream.readline(4096))[:3] != [
                            "header",
                            4,
                            start,
                        ]:
                            raise ValueError("capture_batch_mismatch")
                    output.add(path, arcname=path.name, recursive=False)
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
        result = {"bytes": archive.stat().st_size, "sha256": checksum(archive)}
        write_json(manifest, result)
        return result


def arm(root, run, start, end, workers):
    now = time.time_ns() // 1_000_000
    if (
        type(run) is not int
        or not 0 < run <= start
        or type(end) is not int
        or not 0 < end - start <= 300_000
        or start > now + 300_000
    ):
        raise ValueError("invalid_capture_window")
    active = root / "plan.json"
    plan = {"run": run, "start": start, "end": end}
    with (root / ".plan.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = json.loads(active.read_text()) if active.exists() else None
        if old != plan:
            continuation = old is not None and old.get("run") == run
            allowed = (
                start == old["end"]
                if continuation
                else run == start and (old is None or now >= old["end"] + DRAIN_MS)
            )
            if start <= now or not allowed:
                raise ValueError("capture_window_unavailable")
            (root / str(start)).mkdir(mode=0o700, exist_ok=False)
            write_json(active, plan)
    return {"state": state(root / str(start), workers)[0], "workers": workers}


def read(root, command, start, workers):
    directory = root / str(start)
    data = archive_manifest(directory, start, workers)
    if "offset" not in command:
        return data
    offset, size = command["offset"], command["size"]
    if (
        type(offset) is not int
        or type(size) is not int
        or not 0 <= offset < data["bytes"]
        or not 0 < size <= CHUNK_BYTES
        or command["sha256"] != data["sha256"]
    ):
        raise ValueError("invalid_capture_range")
    with (directory / "pod.tar").open("rb") as archive:
        archive.seek(offset)
        size = min(size, data["bytes"] - offset)
        chunk = archive.read(size)
    if len(chunk) != size:
        raise OSError("archive_changed")
    return {
        **data,
        "offset": offset,
        "chunk_sha256": hashlib.sha256(chunk).hexdigest(),
        "data_base64": base64.b64encode(chunk).decode("ascii"),
    }


def serve(root, workers, key):
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    instance = uuid.uuid4().hex

    class CaptureHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                if self.path not in {"/capture/arm", "/capture/archive"}:
                    self.reply(404, {"error": "unknown_capture_operation"})
                    return
                length = int(self.headers["Content-Length"])
                if not 0 < length <= 1024:
                    raise ValueError("invalid_capture_request")
                payload = self.rfile.read(length)
                mac = self.headers.get("X-Capture-Signature", "")
                command = json.loads(payload)
                if (
                    not hmac.compare_digest(
                        signature(self.path.encode() + b"\n" + payload, key), mac
                    )
                    or not time.time() <= command["expires"] <= time.time() + 180
                ):
                    self.reply(403, {"error": "invalid_capture_signature"})
                    return
                start = command["start"]
                if type(start) is not int or start <= 0:
                    raise ValueError("invalid_capture_batch")
                if self.path == "/capture/arm":
                    data = arm(root, command["run"], start, command["end"], workers)
                else:
                    data = read(root, command, start, workers)
                self.reply(200, {**data, "start": start, "instance": instance})
            except (ValueError, KeyError, TypeError, OSError, AttributeError) as error:
                code = str(error)
                self.reply(
                    409,
                    {
                        "error": (
                            code
                            if re.fullmatch(r"[a-z_]{1,64}", code)
                            else "capture_unavailable"
                        )
                    },
                )

        def reply(self, status, doc):
            response = json.dumps(doc, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    server = ThreadingHTTPServer(("0.0.0.0", PORT), CaptureHandler)
    threading.Thread(
        target=server.serve_forever, daemon=True, name="capture-control"
    ).start()

"""Serve the finite training import plan with bounded concurrent source reads."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import stat
import struct
import tempfile
import threading
import time

from q38_checkpoint_plan import make_plan, plan_digest
from q38_checkpoint_relay import MODEL, REVISION, Range, _private_ip, _readonly, read_local, verify_identity
from q38_remote_checkpoint import Checkpoint, HISTORIC_CATALOG_SHA256, Metadata, checked_json, decode_json, load_headers


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_receipt(path, value):
    data = _json_bytes(value)
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as output:
        temporary = Path(output.name)
        output.write(data)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()
    return hashlib.sha256(data).hexdigest()


def _read_source(root, selected):
    data = bytearray(selected.length)
    view = memoryview(data)
    with open(Path(root)/selected.name, "rb", buffering=0, opener=_readonly) as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != selected.size:
            raise ValueError("checkpoint file size or type changed")
        source.seek(selected.offset)
        while view:
            count = source.readinto(view)
            if not count:
                raise ValueError("checkpoint range ended before the admitted bytes")
            view = view[count:]
        after = os.fstat(source.fileno())
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            raise ValueError("checkpoint file changed during the read")
    return data


def validate_source(root, ranges, catalog_path):
    verify_identity(root)
    catalog = checked_json(Path(catalog_path).read_bytes(), HISTORIC_CATALOG_SHA256, "header catalog")
    rows = {row["file"]: row for row in catalog["headers"]}
    for name, size in sorted(ranges):
        row = rows[name]
        if size != row["file_bytes"]:
            raise ValueError(f"planned file size differs from pinned header {name}")
        selected = Range(name, size, 0, row["header_length"]+8)
        raw = read_local(root, selected)
        if (struct.unpack("<Q", raw[:8])[0] != row["header_length"]
                or hashlib.sha256(raw).hexdigest() != row["header_sha256"]):
            raise ValueError(f"mounted header differs from the pinned source {name}")


def serve_training_plan(root, host, port, choices, catalog_path, *, seconds=3600, workers=8,
                        plan_sha256=None, ready_file=None):
    """Use the existing checkpoint Range/HTTP digest protocol on approved peers."""
    host = _private_ip(host)
    if not 1 <= workers <= 8 or not 1 <= seconds <= 3600:
        raise ValueError("invalid finite checkpoint service bounds")
    pending = {}
    objects = set()
    for peer, rows in choices.items():
        peer = _private_ip(peer)
        for row in rows:
            selected = Range(**row)
            key = (peer, selected.path, selected.request_header)
            if key in pending:
                raise ValueError("duplicate planned checkpoint request")
            pending[key] = selected
            objects.add((selected.name, selected.size))
    if not pending:
        raise ValueError("no planned reads for this mounted source")
    validate_source(root, objects, catalog_path)
    count = len(pending)
    expected_bytes = sum(item.length for item in pending.values())
    ready = {
        "host": host, "port": port, "revision": REVISION, "planned_ranges": count,
        "planned_bytes": expected_bytes, "source_plan_sha256": plan_digest(choices),
        "plan_sha256": plan_sha256, "seconds": seconds,
    }
    ready_bytes = _json_bytes(ready)
    ready_peers = frozenset(choices)
    lock = threading.Lock()
    inflight = set()
    completed = {}
    deadline = time.monotonic() + seconds

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            self.request.settimeout(12)
            super().setup()

        def log_message(self, *_):
            pass

        def do_GET(self):
            values = self.headers.get_all("Range", [])
            if (self.path == f"/{REVISION}/ready" and not values
                    and self.client_address[0] in ready_peers and time.monotonic() < deadline):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(ready_bytes)))
                self.end_headers()
                self.wfile.write(ready_bytes)
                return
            if len(values) != 1:
                self.send_error(416)
                return
            key = (self.client_address[0], self.path, values[0])
            with lock:
                selected = pending.get(key)
                allowed = selected is not None and key not in inflight and time.monotonic() < deadline
                if allowed:
                    inflight.add(key)
            if not allowed:
                self.send_error(404)
                return
            try:
                data = _read_source(root, selected)
                sha = hashlib.sha256(data).hexdigest()
                self.send_response(206)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Content-Range", selected.content_range)
                self.send_header("X-Checkpoint-SHA256", sha)
                self.end_headers()
                self.wfile.write(data)
                with lock:
                    del pending[key]
                    completed[key] = (len(data), sha)
            finally:
                with lock:
                    inflight.remove(key)

    class BoundedServer(HTTPServer):
        request_queue_size = 128

        def process_request(self, request, client_address):
            # The kernel backlog holds new connections while all allowed
            # workers have one live range. No unbounded task queue is built.
            while time.monotonic() < deadline:
                if slots.acquire(timeout=0.25):
                    pool.submit(self.handle_with_slot, request, client_address)
                    return
            self.shutdown_request(request)

        def handle_with_slot(self, request, client_address):
            try:
                self.finish_request(request, client_address)
            except (ValueError, OSError):
                # A short or failed source read leaves that exact range pending.
                pass
            finally:
                self.shutdown_request(request)
                slots.release()

    slots = threading.BoundedSemaphore(workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        with BoundedServer((host, port), Handler) as server:
            server.timeout = 0.25
            if ready_file is not None:
                _write_receipt(ready_file, ready)
            print("Q38_TRAIN_CHECKPOINT_READY " + json.dumps(ready, sort_keys=True), flush=True)
            while time.monotonic() < deadline:
                with lock:
                    if not pending:
                        break
                server.handle_request()
    if pending:
        raise TimeoutError(f"mounted source {host} still has {len(pending)} ranges pending")
    sent = sum(length for length, _ in completed.values())
    if len(completed) != count or sent != expected_bytes:
        raise ValueError("checkpoint relay completion differs from its admitted plan")
    return {**ready, "ranges": count, "bytes": sent, "coverage_sha256": plan_digest(sorted(completed.items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--peers", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("plan")
    write.add_argument("--endpoints-out", type=Path, required=True)
    write.add_argument("--ledger-out", type=Path, required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--host", required=True)
    serve.add_argument("--expected-plan-sha256", required=True)
    serve.add_argument("--ready-file", type=Path, required=True)
    serve.add_argument("--result-file", type=Path, required=True)
    serve.add_argument("--seconds", type=int, default=3600)
    serve.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    peers = decode_json(args.peers.read_bytes())
    if set(peers) != {"train", "rollout"} or any(not isinstance(rows, list) for rows in peers.values()):
        raise ValueError("expected ordered train and rollout private IPs")
    train = [_private_ip(value) for value in peers["train"]]
    rollout = [_private_ip(value) for value in peers["rollout"]]
    metadata = Metadata.from_files(args.model_root, args.objects, args.policy)
    headers = load_headers(metadata, args.catalog, HISTORIC_CATALOG_SHA256)
    checkpoint = Checkpoint(metadata, headers, read=None)
    endpoints, plans, ledger = make_plan(checkpoint, train, rollout, args.port)
    if args.command == "plan":
        ledger.update(revision=REVISION, port=args.port, catalog_sha256=HISTORIC_CATALOG_SHA256,
                      endpoint_file_sha256=_write_receipt(args.endpoints_out, endpoints))
        _write_receipt(args.ledger_out, ledger)
        print("Q38_TRAIN_CHECKPOINT_PLAN " + json.dumps(ledger, sort_keys=True), flush=True)
        return

    if (args.model_root != Path(MODEL) or args.host not in plans
            or args.expected_plan_sha256 != ledger["plan_sha256"]):
        raise ValueError("source host, mounted root or checkpoint plan differs from the released plan")
    result = serve_training_plan(args.model_root, args.host, args.port, plans[args.host], args.catalog,
                                 seconds=args.seconds, workers=args.workers, plan_sha256=ledger["plan_sha256"],
                                 ready_file=args.ready_file)
    if (result["source_plan_sha256"] != ledger["per_source_plan_sha256"][args.host]
            or result["bytes"] != ledger["per_source_bytes"][args.host]
            or result["ranges"] != ledger["per_source_ranges"][args.host]):
        raise ValueError("completed relay receipt differs from the released coverage ledger")
    _write_receipt(args.result_file, result)
    print("Q38_TRAIN_CHECKPOINT_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

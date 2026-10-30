"""One-run rendezvous for two Bazel remote test shards.

The controller keeps only bounded test metadata. It cannot execute commands or
serve source files. The random bearer token is supplied through a private file.
"""

import argparse
import copy
import hmac
import ipaddress
import json
import math
import os
import queue
import re
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

DIAGNOSTIC_BODY_BYTES = 8192
DIAGNOSTIC_EVENTS_PER_ROLE = 512
DIAGNOSTIC_ROLES = ("prefill", "decode")
STARTUP_STAGES = {
    "waiting_for_logs",
    "backend_start",
    "jit_cache",
    "weight_loading",
    "model_init",
    "cache_init",
    "cuda_graph",
    "executor_init",
    "waiting_for_listeners",
    "ready",
}
STARTUP_MARKERS = {
    "backend_started",
    "jit_scope",
    "jit_restored",
    "jit_fail_open",
    "rank_started",
    "weight_load_started",
    "weights_materialized",
    "draft_load_started",
    "model_load_finished",
    "cache_layout_planned",
    "cache_manager_ready",
    "cuda_graph_started",
    "cuda_graph_batch_done",
    "executor_ready",
    "backend_ready",
    "server_start_finished",
    "target_weights_ready",
    "draft_weights_ready",
    "max_weight_load_ms",
    "max_model_load_ms",
}
REQUEST_COUNTS = {
    "elapsed_ms",
    "case_count",
    "tokenize_calls",
    "tokenize_ms",
    "fit_calls",
    "fit_ms",
    "token_cache_hits",
}
CONSOLE_STARTUP_MARKERS = {
    "backend_ready",
    "rank_started",
    "jit_restored",
    "jit_fail_open",
    "max_weight_load_ms",
    "max_model_load_ms",
}


class ConsoleProgress:
    """Best-effort, bounded output independent of the coordination threads."""

    def __init__(self, fd, token):
        self.fd = fd
        self.token = token
        self.pending = queue.Queue(maxsize=128)
        self.stop = threading.Event()
        self.writer = threading.Thread(target=self._write, daemon=True)
        self.writer.start()

    def __call__(self, event):
        try:
            self.pending.put_nowait(event)
        except queue.Full:
            pass

    def _write(self):
        while not self.stop.is_set() or not self.pending.empty():
            try:
                event = self.pending.get(timeout=0.1)
            except queue.Empty:
                continue
            line = json.dumps(event, separators=(",", ":")).replace(
                self.token, "<redacted>"
            )
            try:
                os.write(self.fd, f"K3 PD progress: {line}\n".encode())
            except OSError:
                return

    def close(self):
        self.stop.set()
        self.writer.join(timeout=0.5)


def _diagnostic_count(value):
    return type(value) is int and 0 <= value < 2**53


def _diagnostic_endpoints(items):
    return (
        isinstance(items, list)
        and len(items) <= 26
        and all(
            isinstance(item, dict)
            and item.keys() == {"rank", "kind", "port"}
            and type(item["rank"]) is int
            and 0 <= item["rank"] < 8
            and item["kind"] in ("rpc", "cache_tcp", "cache_rdma", "http")
            and type(item["port"]) is int
            and 0 < item["port"] <= 65535
            for item in items
        )
    )


def _valid_diagnostic(phase, data):
    if not isinstance(data, dict):
        return False
    if phase in ("launcher_started", "requests_finished"):
        return data.keys() == {"elapsed_ms"} and _diagnostic_count(data["elapsed_ms"])
    if phase in ("health_wait", "ready"):
        return (
            data.keys()
            == {"elapsed_ms", "stage", "markers", "missing_listeners", "unhealthy_http"}
            and _diagnostic_count(data["elapsed_ms"])
            and isinstance(data["stage"], str)
            and data["stage"] in STARTUP_STAGES
            and isinstance(data["markers"], dict)
            and data["markers"].keys() <= STARTUP_MARKERS
            and all(_diagnostic_count(value) for value in data["markers"].values())
            and _diagnostic_endpoints(data["missing_listeners"])
            and _diagnostic_endpoints(data["unhealthy_http"])
        )
    if phase == "request_stage":
        event = data.get("event")
        keys = REQUEST_COUNTS | {"event", "passed"}
        if event in ("request_stage_start", "request_stage"):
            keys = keys | {"stage"}
            stage = data.get("stage")
            if not isinstance(stage, str) or not re.fullmatch(
                r"[A-Za-z][A-Za-z0-9_+-]{0,95}", stage
            ):
                return False
        elif event != "request_summary":
            return False
        return (
            data.keys() == keys
            and type(data["passed"]) is bool
            and all(_diagnostic_count(data[key]) for key in REQUEST_COUNTS)
        )
    if phase == "failed":
        return (
            data.keys() == {"error_type"}
            and isinstance(data["error_type"], str)
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", data["error_type"])
            is not None
        )
    if phase == "cleanup":
        return (
            data.keys() == {"elapsed_ms", "passed"}
            and _diagnostic_count(data["elapsed_ms"])
            and type(data["passed"]) is bool
        )
    return False


class Rendezvous:
    def __init__(self, run_id, timeout, output, progress=None):
        if not re.fullmatch(r"k3-pd-[a-zA-Z0-9]{1,64}", run_id):
            raise ValueError("Invalid run identity")
        if (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or not 0 < timeout <= 86400
        ):
            raise ValueError("Timeout must be positive and no more than one day")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.progress = progress
        self.lock = threading.RLock()
        self.state = {
            "run_id": run_id,
            "deadline_unix_s": time.time() + timeout,
            "roles": {},
            "error": None,
        }
        self.heartbeats = {}
        self.events = self.output / "events.jsonl"
        self.diagnostics = self.output / "diagnostics.jsonl"
        self.diagnostics_lock = threading.Lock()
        self.diagnostic_counts = dict.fromkeys(DIAGNOSTIC_ROLES, 0)

    def record_diagnostic(self, message):
        if not isinstance(message, dict) or message.keys() != {"role", "phase", "data"}:
            raise ValueError("Expected diagnostic role, phase and data")
        role, phase = message["role"], message["phase"]
        if role not in DIAGNOSTIC_ROLES or not _valid_diagnostic(
            phase, message["data"]
        ):
            raise ValueError("Invalid diagnostic metadata")
        with self.diagnostics_lock:
            if self.diagnostic_counts[role] >= DIAGNOSTIC_EVENTS_PER_ROLE:
                return False
            recorded_at = time.time()
            try:
                with self.diagnostics.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"time": recorded_at, **message}) + "\n")
            except OSError:
                return False
            self.diagnostic_counts[role] += 1
        if self.progress is not None:
            data = message["data"]
            if phase in ("health_wait", "ready"):
                data = {
                    "elapsed_ms": data["elapsed_ms"],
                    "stage": data["stage"],
                    "markers": {
                        key: value
                        for key, value in data["markers"].items()
                        if key in CONSOLE_STARTUP_MARKERS
                    },
                    "missing_listener_count": len(data["missing_listeners"]),
                    "unhealthy_http_count": len(data["unhealthy_http"]),
                }
            self.progress(
                {
                    "run_id": self.state["run_id"],
                    "time": recorded_at,
                    "role": role,
                    "phase": phase,
                    "data": data,
                }
            )
        return True

    def _error_once(self, message):
        if self.state["error"] is None:
            self.state["error"] = message[:2048]

    def _expire(self):
        records = self.state["roles"]
        complete = len(records) == 2 and all(
            record[key] is not None
            for record in records.values()
            for key in ("result", "cleanup")
        )
        if complete or self.state["error"] is not None:
            return
        now = time.time()
        if now >= self.state["deadline_unix_s"]:
            self._error_once("CI rendezvous deadline exceeded")
        for role, last in self.heartbeats.items():
            if now - last > 120 and records[role]["cleanup"] is None:
                self._error_once(f"{role} heartbeat expired")

    def snapshot(self):
        with self.lock:
            self._expire()
            return copy.deepcopy(self.state)

    def update(self, message):
        if not isinstance(message, dict) or set(message) != {"op", "role", "data"}:
            raise ValueError("Expected op, role and data")
        role, op, data = message["role"], message["op"], message["data"]
        if role not in ("prefill", "decode"):
            raise ValueError("Invalid role")
        if op not in {
            "register",
            "heartbeat",
            "prepared",
            "ready",
            "result",
            "cleanup",
        }:
            raise ValueError("Invalid operation")
        with self.lock:
            # Evaluate the previous lease before this request can refresh it.
            self._expire()
            records = self.state["roles"]
            new_milestone = False
            if op == "register":
                self._validate_registration(data)
                if role in records:
                    if records[role]["registration"] != data:
                        raise ValueError(
                            "A different worker already registered this role"
                        )
                else:
                    new_milestone = True
                    records[role] = {
                        "registration": copy.deepcopy(data),
                        "prepared": None,
                        "ready": False,
                        "result": None,
                        "cleanup": None,
                    }
                    if len(records) == 2:
                        first = records["prefill"]["registration"]
                        second = records["decode"]["registration"]
                        if (
                            first["ip"] == second["ip"]
                            or first["hostname"] == second["hostname"]
                        ):
                            self._error_once(
                                "Two role actions resolved to the same host"
                            )
            elif role not in records:
                raise ValueError("Register before publishing role state")
            elif op == "heartbeat":
                if data != {}:
                    raise ValueError("Heartbeat data must be empty")
            else:
                record = records[role]
                if op == "ready":
                    if data is not True or record["prepared"] is None:
                        raise ValueError("Only a prepared role can become ready")
                elif op in {"result", "cleanup"}:
                    if (
                        not isinstance(data, dict)
                        or set(data) != {"passed", "errors"}
                        or type(data["passed"]) is not bool
                        or not isinstance(data["errors"], list)
                        or len(data["errors"]) > 16
                        or not all(
                            isinstance(item, str) and 1 <= len(item) <= 2048
                            for item in data["errors"]
                        )
                        or data["passed"] == bool(data["errors"])
                    ):
                        raise ValueError("Invalid terminal role result")
                    if op == "result" and data["passed"] and not record["ready"]:
                        raise ValueError(
                            "Only a ready role can publish a passing result"
                        )
                elif op == "prepared":
                    self._validate_prepared(data, record["registration"])
                old = record[op]
                if old not in (None, False) and old != data:
                    raise ValueError(f"Cannot replace immutable {op} state")
                if (
                    op in {"prepared", "ready"}
                    and old != data
                    and (record["result"] is not None or record["cleanup"] is not None)
                ):
                    raise ValueError("Cannot advance a finished role")
                # Nothing above this point may partially apply the new outcome.
                new_milestone = old != data
                record[op] = copy.deepcopy(data)
                if op in {"result", "cleanup"} and not data["passed"]:
                    self._error_once(
                        f"{role} {op} failed: " + "; ".join(data["errors"])
                    )
                if op == "prepared":
                    self._check_prepared_pair()
            self.heartbeats[role] = time.time()
            if op != "heartbeat":
                recorded_at = time.time()
                with self.events.open("a") as stream:
                    stream.write(json.dumps({"time": recorded_at, **message}) + "\n")
                self.persist()
                if new_milestone and self.progress is not None:
                    milestone = {
                        "run_id": self.state["run_id"],
                        "time": recorded_at,
                        "role": role,
                        "milestone": op,
                    }
                    if op in ("result", "cleanup"):
                        milestone["passed"] = data["passed"]
                    self.progress(milestone)
            return self.snapshot()

    @staticmethod
    def _validate_registration(data):
        keys = {"hostname", "ip"}
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Invalid registration")
        if not isinstance(data["hostname"], str) or not re.fullmatch(
            r"[a-zA-Z0-9_.-]{1,253}", data["hostname"]
        ):
            raise ValueError("Invalid worker hostname")
        ip = ipaddress.IPv4Address(data["ip"])
        if (
            ip.is_loopback
            or ip.is_unspecified
            or ip.is_multicast
            or ip.is_link_local
            or ip.is_reserved
        ):
            raise ValueError("Worker needs a routable address")

    @staticmethod
    def _validate_prepared(data, registration):
        keys = {
            "hostname",
            "gpu_uuids",
            "checkpoint_identity",
            "draft_identity",
            "guard_sha256",
        }
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Invalid prepared identity")
        if data["hostname"] != registration["hostname"]:
            raise ValueError("Prepared identity differs from registration")
        uuids = data["gpu_uuids"]
        if (
            not isinstance(uuids, list)
            or len(uuids) != 8
            or not all(
                isinstance(value, str)
                and re.fullmatch(r"GPU-[A-Za-z0-9-]{1,128}", value)
                for value in uuids
            )
            or len(set(uuids)) != 8
        ):
            raise ValueError("Each role needs eight distinct GPUs")
        guard = data["guard_sha256"]
        if not isinstance(guard, str) or not re.fullmatch(r"[0-9a-f]{64}", guard):
            raise ValueError("Invalid checkpoint guard identity")
        for key in ("checkpoint_identity", "draft_identity"):
            checkpoint = data[key]
            if not isinstance(checkpoint, dict) or set(checkpoint) != {
                "config_sha256",
                "index_sha256",
                "shards",
            }:
                raise ValueError("Invalid checkpoint identity")
            for field in ("config_sha256", "index_sha256"):
                digest = checkpoint[field]
                if field == "index_sha256" and digest is None:
                    continue
                if not isinstance(digest, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", digest
                ):
                    raise ValueError("Invalid checkpoint digest")
            shards = checkpoint["shards"]
            if not isinstance(shards, list) or not 1 <= len(shards) <= 4096:
                raise ValueError("Invalid checkpoint shards")
            names = set()
            for shard in shards:
                if (
                    not isinstance(shard, list)
                    or len(shard) != 2
                    or not isinstance(shard[0], str)
                    or not re.fullmatch(r"[A-Za-z0-9_.-]{1,255}", shard[0])
                    or type(shard[1]) is not int
                    or shard[1] <= 0
                    or shard[0] in names
                ):
                    raise ValueError("Invalid checkpoint shard")
                names.add(shard[0])

    def _check_prepared_pair(self):
        records = self.state["roles"]
        if len(records) != 2 or any(
            record["prepared"] is None for record in records.values()
        ):
            return
        first, second = (records[role]["prepared"] for role in ("prefill", "decode"))
        if set(first["gpu_uuids"]) & set(second["gpu_uuids"]):
            self._error_once("PD role actions overlap physical GPUs")
        for key in ("checkpoint_identity", "draft_identity", "guard_sha256"):
            if first[key] != second[key]:
                self._error_once(f"PD role {key} mismatch")

    def persist(self):
        with self.lock:
            temporary = self.output / "state.json.part"
            temporary.write_text(json.dumps(self.snapshot(), indent=2) + "\n")
            temporary.replace(self.output / "state.json")


def make_handler(rendezvous, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, payload):
            content = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def authorized(self, suffix=""):
            expected = "Bearer " + token
            if not hmac.compare_digest(self.headers.get("Authorization", ""), expected):
                self.reply(403, {"error": "Invalid task token"})
                return False
            if (
                urlsplit(self.path).path
                != "/v1/runs/" + rendezvous.state["run_id"] + suffix
            ):
                self.reply(404, {"error": "Unknown run"})
                return False
            return True

        def do_GET(self):
            if self.authorized():
                self.reply(200, rendezvous.snapshot())

        def do_POST(self):
            diagnostic = urlsplit(self.path).path.endswith("/diagnostics")
            if not self.authorized("/diagnostics" if diagnostic else ""):
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= (DIAGNOSTIC_BODY_BYTES if diagnostic else 32768):
                    raise ValueError("Invalid metadata size")
                message = json.loads(self.rfile.read(size))
                if diagnostic:
                    self.reply(200, {"recorded": rendezvous.record_diagnostic(message)})
                else:
                    self.reply(200, rendezvous.update(message))
            except (ValueError, TypeError, KeyError, RecursionError) as error:
                self.reply(400, {"error": str(error)})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--advertise-ip", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=int, default=7000)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--progress-fd", type=int)
    args = parser.parse_args()
    token_file = Path(args.token_file)
    if token_file.stat().st_mode & 0o077:
        raise ValueError("Task token file must be private")
    token = token_file.read_text().strip()
    if not re.fullmatch(r"[a-f0-9]{64}", token):
        raise ValueError("Expected a random 256-bit task token")
    ipaddress.IPv4Address(args.advertise_ip)
    progress = (
        None if args.progress_fd is None else ConsoleProgress(args.progress_fd, token)
    )
    rendezvous = Rendezvous(args.run_id, args.timeout, args.output, progress=progress)
    server = ThreadingHTTPServer(
        ("0.0.0.0", args.port), make_handler(rendezvous, token)
    )
    server.daemon_threads = True
    ready = {
        "url": f"http://{args.advertise_ip}:{server.server_port}",
        "run_id": args.run_id,
        "pid": os.getpid(),
    }
    (Path(args.output) / "ready.json").write_text(json.dumps(ready) + "\n")

    def stop(signum, frame):
        rendezvous.persist()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever()
    finally:
        rendezvous.persist()
        server.server_close()
        if progress is not None:
            progress.close()


if __name__ == "__main__":
    main()

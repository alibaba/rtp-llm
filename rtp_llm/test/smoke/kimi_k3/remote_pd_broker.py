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
import re
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


class Rendezvous:
    def __init__(self, run_id, timeout, output):
        if not re.fullmatch(r"k3-pd-[a-zA-Z0-9]+", run_id):
            raise ValueError("Invalid run identity")
        if (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or not 0 < timeout <= 86400
        ):
            raise ValueError("Timeout must be positive and no more than one day")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.state = {
            "run_id": run_id,
            "deadline_unix_s": time.time() + timeout,
            "roles": {},
            "error": None,
        }
        self.heartbeats = {}
        self.events = self.output / "events.jsonl"

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
            if op == "register":
                self._validate_registration(data)
                if role in records:
                    if records[role]["registration"] != data:
                        raise ValueError(
                            "A different worker already registered this role"
                        )
                else:
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
                        for key in ("external_revision", "internal_revision"):
                            if first[key] != second[key]:
                                self._error_once(f"Role {key} mismatch")
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
                record[op] = copy.deepcopy(data)
                if op in {"result", "cleanup"} and not data["passed"]:
                    self._error_once(
                        f"{role} {op} failed: " + "; ".join(data["errors"])
                    )
                if op == "prepared":
                    self._check_prepared_pair()
            self.heartbeats[role] = time.time()
            if op != "heartbeat":
                with self.events.open("a") as stream:
                    stream.write(json.dumps({"time": time.time(), **message}) + "\n")
                self.persist()
            return self.snapshot()

    @staticmethod
    def _validate_registration(data):
        keys = {"hostname", "ip", "external_revision", "internal_revision"}
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
        for key in ("external_revision", "internal_revision"):
            if not isinstance(data[key], str) or not re.fullmatch(
                r"[0-9a-f]{40}", data[key]
            ):
                raise ValueError("Invalid source pin")

    @staticmethod
    def _validate_prepared(data, registration):
        keys = {
            "hostname",
            "gpu_uuids",
            "external_revision",
            "internal_revision",
            "checkpoint_identity",
            "draft_identity",
            "guard_sha256",
        }
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Invalid prepared identity")
        for key in ("hostname", "external_revision", "internal_revision"):
            if data[key] != registration[key]:
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

        def authorized(self):
            expected = "Bearer " + token
            if not hmac.compare_digest(self.headers.get("Authorization", ""), expected):
                self.reply(403, {"error": "Invalid task token"})
                return False
            if urlsplit(self.path).path != "/v1/runs/" + rendezvous.state["run_id"]:
                self.reply(404, {"error": "Unknown run"})
                return False
            return True

        def do_GET(self):
            if self.authorized():
                self.reply(200, rendezvous.snapshot())

        def do_POST(self):
            if not self.authorized():
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 32768:
                    raise ValueError("Invalid metadata size")
                message = json.loads(self.rfile.read(size))
                self.reply(200, rendezvous.update(message))
            except (ValueError, TypeError, KeyError) as error:
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
    args = parser.parse_args()
    token_file = Path(args.token_file)
    if token_file.stat().st_mode & 0o077:
        raise ValueError("Task token file must be private")
    token = token_file.read_text().strip()
    if not re.fullmatch(r"[a-f0-9]{64}", token):
        raise ValueError("Expected a random 256-bit task token")
    ipaddress.IPv4Address(args.advertise_ip)
    rendezvous = Rendezvous(args.run_id, args.timeout, args.output)
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


if __name__ == "__main__":
    main()

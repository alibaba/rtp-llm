"""Native K3.1 CI role adapter; peer traffic contains CPU execution events only."""

import gzip
import hashlib
import hmac
import io
import json
import os
import platform
import re
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rtp_llm.test.smoke.kimi_k3 import k31_full_smoke
from rtp_llm.test.smoke.kimi_k3.k31_smoke_capacity import device_pool_usable_blocks
from rtp_llm.test.smoke.kimi_k31_bazel.worker import Worker

MAX_WIRE_BYTES = 16 * 1024**2
# Eight-rank smoke traces reached 271,476,950 bytes before compression. Keep
# every event for the shared audit, with separate finite decoded/wire budgets.
MAX_EVENT_BYTES = 512 * 1024**2


def audit_service_cleanup(directory, supervisor_exitcode):
    """Verify fixture resource release, separately from graceful runtime exit."""
    cleanup = json.loads((Path(directory) / "cleanup.json").read_text())
    if (
        cleanup.get("stopped") is not True
        or cleanup.get("gpu_contexts_verified") is not True
        or cleanup.get("errors") != []
        or cleanup.get("unexpected_gpu_contexts") != []
    ):
        raise ValueError("Service cleanup or GPU resource verification failed")
    if type(supervisor_exitcode) is not int or supervisor_exitcode not in (
        0,
        -2,
        -9,
        -15,
    ):
        raise ValueError(f"Unexpected service supervisor exit: {supervisor_exitcode}")
    forced_child_cleanup = False
    for pattern in ("service.log", "service/logs/*.log", "logs/engine.log"):
        for path in Path(directory).glob(pattern):
            with path.open(errors="replace") as stream:
                for line in stream:
                    if re.search(
                        r"Fatal Python error:|Signal: (?:Aborted|Segmentation fault) \("
                        r"|DeepGEMM grid sync timeout:|CUDA error: unspecified launch failure",
                        line,
                    ):
                        raise ValueError(f"Service logged a fatal error in {path.name}")
                    forced_child_cleanup |= bool(
                        re.search(r"Force killing process \d+", line)
                    )
    return {
        "passed": True,
        "scope": "k3_service_process_group_and_gpu_cleanup",
        "supervisor_exitcode": supervisor_exitcode,
        "supervisor_force_killed": supervisor_exitcode == -9,
        "forced_child_cleanup_observed": forced_child_cleanup,
        "backend_rank_exitcodes_verified": False,
        "graceful_backend_shutdown_verified": False,
    }


def runtime_geometry(engine, launch, source_revision, stage="regular"):
    """Use the traditional smoke geometry contract without a CI-only parser."""
    from rtp_llm.test.smoke.kimi_k3.k31_smoke_capacity import cache_geometry

    geometry = cache_geometry(engine, "prefill", 8, long_case=stage == "long")
    geometry.update(source_commit=source_revision,
                    engine_sha256=hashlib.sha256(engine.read_bytes()).hexdigest(),
                    launch_sha256=hashlib.sha256(launch.read_bytes()).hexdigest())
    return geometry


def encode_events(events, run_id, role, *, engine_geometry=None):
    value = dict(run_id=run_id, role=role, events=events)
    if engine_geometry is not None:
        value["engine_geometry"] = engine_geometry
    payload = json.dumps(value, separators=(",", ":")).encode()
    if len(payload) > MAX_EVENT_BYTES:
        raise ValueError(
            "Execution event snapshot exceeds the uncompressed budget: "
            f"{len(payload)} > {MAX_EVENT_BYTES} bytes"
        )
    compressed = gzip.compress(payload)
    if len(compressed) > MAX_WIRE_BYTES:
        raise ValueError("Execution event snapshot exceeds the wire budget")
    return compressed


def decode_events(payload, run_id, role, expected_ranks=range(8)):
    from rtp_llm.test.smoke.kimi_k3.audit_k31_full_smoke import IndexedEvents

    if len(payload) > MAX_WIRE_BYTES:
        raise ValueError("Execution event response exceeds the wire budget")
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
        data = stream.read(MAX_EVENT_BYTES + 1)
    if len(data) > MAX_EVENT_BYTES:
        raise ValueError(
            "Execution event response exceeds the uncompressed budget: "
            f"more than {MAX_EVENT_BYTES} bytes"
        )
    value = json.loads(data)
    if value.get("run_id") != run_id or value.get("role") != role:
        raise ValueError("Execution events belong to a different run or role")
    events = value["events"]
    if not isinstance(events, dict) or set(events) != {
        str(rank) for rank in expected_ranks
    }:
        raise ValueError(
            "Execution evidence must include all eight ranks"
            if len(expected_ranks) == 8
            else "Execution evidence does not match this member's four ranks"
        )
    for rows in events.values():
        if (
            not isinstance(rows, list)
            or not rows
            or any(
                not isinstance(row, dict) or type(row.get("time_ns")) is not int
                for row in rows
            )
        ):
            raise ValueError("Missing or malformed rank execution events")
        if any(a["time_ns"] > b["time_ns"] for a, b in zip(rows, rows[1:])):
            raise ValueError("Rank execution events are not ordered")
    return {int(rank): IndexedEvents(rows) for rank, rows in events.items()}


class EvidenceServer:
    """One authenticated, fixed-purpose endpoint; no paths, commands or weights."""

    def __init__(self, address, token, snapshot, on_error=None):
        if not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{64}", token):
            raise ValueError("Missing per-run execution evidence token")

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                if not hmac.compare_digest(
                    self.headers.get("Authorization", ""), "Bearer " + token
                ):
                    self.send_error(403)
                    return
                if self.path != "/execution-events":
                    self.send_error(404)
                    return
                try:
                    body = snapshot()
                except Exception as error:
                    if on_error is not None:
                        on_error(error)
                    self.send_error(503, "Execution evidence unavailable")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/gzip")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(address, Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("Execution evidence server did not stop")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


class K31Worker(Worker):
    evidence_server = None
    gpu_model_tokens = ("B300", "L20D")
    worker_layout = "x86-2x8"

    def stop_service(self):
        """Let the shared Worker stop the supervisor before cleaning its group."""
        if self.service is None or self.service.poll() is not None:
            return
        pid = self.service.pid
        # Worker.start creates this supervisor with start_new_session=True.
        # Refuse a shared process group rather than signalling unrelated work.
        if os.getpgid(pid) != pid or os.getsid(pid) != pid:
            raise ValueError("Service does not own its process group and session")
        super().stop_service()
        (self.task / "termination.json").write_text(json.dumps(dict(
            policy="shared_worker_supervisor_term_then_cleanup", supervisor_pid=pid,
            supervisor_exitcode=self.service.returncode,
            initial_signal=15, supervisor_wait_timeout_s=330,
            supervisor_force_killed=self.service.returncode == -9,
            graceful_backend_shutdown_verified=False), indent=2) + "\n")

    def _lock_gpus(self, *args, **kwargs):
        layout = (self.config or {}).get("worker_layout", self.worker_layout)
        architectures = {
            "x86-2x8": ("x86_64", "amd64"),
        }
        if layout not in architectures:
            raise ValueError("Unsupported K3.1 worker layout")
        if platform.machine() not in architectures[layout]:
            raise ValueError(f"K3.1 {layout} requires a matching worker architecture")
        tokens = (self.config or {}).get("expected_gpu_names", self.gpu_model_tokens)
        if (
            not isinstance(tokens, (list, tuple))
            or not tokens
            or any(
                not isinstance(token, str) or not token or len(token) > 32
                for token in tokens
            )
        ):
            raise ValueError("Invalid expected GPU names")
        self.gpu_model_tokens = tuple(tokens)
        return super()._lock_gpus(*args, **kwargs)

    def launcher_options(self):
        options = [
            "--model-type",
            "kimi_k31",
            "--execution-evidence",
            "--test-block-num",
            str(self.host["test_block_num"]),
            "--core-dump-limit-mib",
            "64",
        ]
        for key in (
            "request_capacity", "decode_capture_batches", "max_seq_len",
            "linear_pool_blocks", "block_tree_transfer_worker_count", "prefill_token_capacity",
        ):
            if key in self.host:
                options += ["--" + key.replace("_", "-"), str(self.host[key])]
        return options

    def engine_log(self):
        # alog.conf writes relative to the service cwd (Worker.task), while
        # Python rank logs use the launcher's separate absolute LOG_PATH.
        return self.task / "logs" / "engine.log"

    def events(self):
        from rtp_llm.test.smoke.kimi_k3.audit_k31_full_smoke import read_events

        return read_events(
            self.task / "service" / "logs",
            self.engine_log(),
            getattr(self, "execution_ranks", range(8)),
        )

    def evidence_error(self, error):
        (self.task / "execution-evidence-error.json").write_text(
            json.dumps({"type": type(error).__name__, "message": str(error)}, indent=2)
            + "\n"
        )

    def start(self):
        result = super().start()
        self.projection_audit_required = True
        if self.role == "decode":
            self.evidence_server = EvidenceServer(
                (self.host["ip"], self.host["evidence_port"]),
                self.config["evidence_token"],
                lambda: encode_events(self.events(), self.task.parent.name, self.role),
                self.evidence_error,
            )
        return result

    def request_command(self):
        if self.role != "prefill":
            raise ValueError("K3.1 requests must run on Prefill")
        self.status()
        geometry = runtime_geometry(
            self.geometry_engine_log(),
            self.task / "service" / "launch.json",
            self.config["external_revision"],
            self.config.get("k31_smoke_stage", "regular"),
        )
        path = self.task / "geometry.json"
        with path.open("x") as stream:
            json.dump(geometry, stream, indent=2)
        peer = self.config["decode"]
        if self.config.get("k31_smoke_stage") == "long":
            from rtp_llm.test.smoke.kimi_k3 import k31_long_prefix_smoke

            command = [sys.executable, str(Path(k31_long_prefix_smoke.__file__)),
                       "run", "--base-url", f"http://{self.host['ip']}:{self.host['port']}",
                       "--namespace", self.task.parent.name,
                       "--geometry", str(path),
                       "--output", str(self.task / "requests" / "long-prefix")]
            for address in self.decode_addresses():
                command += ["--decode-role-addr", address]
            return command
        command = [
            sys.executable,
            str(Path(k31_full_smoke.__file__)),
            "--base-url",
            f"http://{self.host['ip']}:{self.host['port']}",
            "--decode-health-url",
            f"http://{peer['ip']}:{peer['port']}/health",
            "--output",
            str(self.task / "requests" / "accuracy.json"),
            "--namespace",
            self.task.parent.name,
            "--source-commit",
            self.config["external_revision"],
            "--geometry",
            str(path),
            "--checkpoint",
            self.host["checkpoint"],
            "--draft-checkpoint",
            self.host["draft_checkpoint"],
            "--prefill-engine-log",
            str(self.geometry_engine_log()),
            "--prefill-rpc-runfiles",
            str(self.workspace.parent),
            "--prefill-grpc-port",
            str(self.host["port"] + 1),
            "--decode-batch-profile",
            "smoke",
        ]
        for address in self.decode_addresses():
            command += ["--decode-role-addr", address]
        return command

    def geometry_engine_log(self):
        return self.engine_log()

    def decode_addresses(self):
        from rtp_llm.test.smoke.kimi_k3.worker import pd_endpoints

        peer = self.config["decode"]
        endpoints = pd_endpoints(peer["port"], self.config.get("decode_dp", 2))
        rpc = {row["rank"]: row["port"] for row in endpoints if row["kind"] == "rpc"}
        return [f"{peer['ip']}:{row['port']}:{rpc[row['rank']]}"
                for row in endpoints if row["kind"] == "http"]

    def execution_inputs(self):
        peer = self.config["decode"]
        request = urllib.request.Request(
            f"http://{peer['ip']}:{peer['evidence_port']}/execution-events",
            headers={"Authorization": "Bearer " + self.config["evidence_token"]},
        )
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirect()
        )
        with opener.open(request, timeout=60) as response:
            if response.headers.get_content_type() != "application/gzip":
                raise ValueError("Unexpected execution evidence content type")
            payload = response.read(MAX_WIRE_BYTES + 1)
        decode = decode_events(payload, self.task.parent.name, "decode")
        (self.task / "decode-execution-events.json.gz").write_bytes(payload)
        return self.events(), decode, self.engine_log()

    def requests(self, message):
        super().requests(message)
        from rtp_llm.test.smoke.kimi_k3.audit_k31_full_smoke import audit
        from rtp_llm.test.smoke.kimi_k3.result_check import (
            _raw_consistency,
            audit_raw_answers,
        )

        prefill, decode, engine = self.execution_inputs()
        accuracy = self.task / "requests" / "accuracy.json"
        if self.config.get("k31_smoke_stage") == "long":
            from rtp_llm.test.smoke.kimi_k3.k31_long_prefix_smoke import verify
            from rtp_llm.test.smoke.kimi_k3.k31_two_host_smoke import audit_long_events

            proof = verify(self.task / "requests" / "long-prefix", self.task / "geometry.json")
            proof["pd_native_route"] = audit_long_events(proof, prefill, decode)
            verdict = dict(passed=True, validation_profile="long", long_smoke_passed=True,
                           native_K31_MTP=True, errors=[], proof=proof,
                           counts=dict(formal=1, preparation=1))
            accuracy.write_text(json.dumps(dict(passed=True, validation_profile="long",
                                                source_commit=self.config["external_revision"],
                                                counts=verdict["counts"]), indent=2) + "\n")
            accuracy.with_name("execution-audit.json").write_text(json.dumps(verdict, indent=2) + "\n")
            print(json.dumps({"k31_long_execution_audit": dict(passed=True, counts=verdict["counts"])}), flush=True)
            return {"requests_passed": True}
        result = json.loads(accuracy.read_text())
        try:
            verdict = audit(
                result,
                prefill,
                decode,
                4,
                prefill_full_pages=device_pool_usable_blocks(engine.read_text(errors="replace"), "full"),
            )
            rows = {row["name"]: row for row in result["cases"]}
            verdict["errors"].extend(
                audit_raw_answers(accuracy.parent / "requests", rows)
            )
            verdict["errors"].extend(
                _raw_consistency(accuracy.parent / "requests", rows)
            )
            verdict["passed"] = not verdict["errors"]
            verdict["complete_smoke_passed"] = (
                verdict.get("complete_smoke_passed") is True and verdict["passed"]
            )
        except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
            verdict = {
                "passed": False,
                "complete_smoke_passed": False,
                "errors": [f"{type(error).__name__}: {error}"],
            }
        with accuracy.with_name("execution-audit.json").open("x") as stream:
            json.dump(verdict, stream, indent=2)
        print(
            json.dumps(
                {
                    "k31_execution_audit": {
                        key: verdict.get(key)
                        for key in (
                            "passed",
                            "execution_contract",
                            "counts",
                            "decode_batches",
                            "decode_capacity_required",
                            "decode_capacity_verified",
                            "errors",
                        )
                    }
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if not verdict["passed"]:
            raise RuntimeError(
                "K3.1 execution audit failed: " + "; ".join(verdict["errors"][:5])
            )
        return {"requests_passed": True}

    def close(self):
        errors = []
        try:
            if self.evidence_server is not None:
                self.evidence_server.close()
        except Exception as error:
            errors.append(str(error))
        try:
            super().close()
        except Exception as error:
            errors.append(str(error))
        if self.task is not None and self.service is not None:
            if getattr(self, "projection_audit_required", False):
                try:
                    from rtp_llm.test.smoke.kimi_k3.nvfp4_evidence import audit_projection_precision
                    from rtp_llm.test.smoke.kimi_k3.k31_two_host_smoke import audit_crc_workspace

                    proof = audit_projection_precision(self.task / "service", self.config)
                    workers = self.host.get("block_tree_transfer_worker_count")
                    if workers is not None:
                        proof["crc_workspace"] = audit_crc_workspace(self.task, workers)
                    (self.task / "projection-precision.json").write_text(json.dumps(proof, indent=2) + "\n")
                except Exception as error:
                    errors.append("Projection/CRC audit: " + str(error))
            try:
                verdict = audit_service_cleanup(self.task, self.service.returncode)
            except Exception as error:
                verdict = {"passed": False, "errors": [str(error)]}
                errors.append("Service cleanup audit: " + str(error))
            (self.task / "shutdown-audit.json").write_text(
                json.dumps(verdict, indent=2) + "\n"
            )
        if self.task is not None and any(self.task.glob("core-*")):
            try:
                from rtp_llm.test.smoke.kimi_k31_bazel.core_evidence import (
                    preserve_core_evidence,
                )
                from rtp_llm.test.utils.coredump_util import (
                    summarize_and_cleanup_coredumps,
                )

                preserve_core_evidence(self.task)
                summarize_and_cleanup_coredumps(str(self.task))
                if any(self.task.glob("core-*")):
                    raise RuntimeError("Core artifact cleanup did not complete")
            except Exception as error:
                errors.append(f"Core evidence: {error}")
        manifest = self.task / "crash-evidence.json" if self.task is not None else None
        if manifest is not None and manifest.is_file():
            try:
                if json.loads(manifest.read_text()).get("core_dump_detected"):
                    errors.append(
                        "Service generated core dumps; see crash-evidence.json"
                    )
            except Exception as error:
                errors.append(f"Core evidence manifest: {error}")
        if errors:
            raise RuntimeError("; ".join(errors))


def check_ci_result(accuracy_path):
    verdict = json.loads(
        Path(accuracy_path).with_name("execution-audit.json").read_text()
    )
    if (
        verdict.get("passed") is not True
        or verdict.get("long_smoke_passed" if verdict.get("validation_profile") == "long"
                       else "complete_smoke_passed") is not True
        or verdict.get("native_K31_MTP") is not True
    ):
        return {
            "passed": False,
            "errors": ["Full native K3.1 execution audit did not pass"],
        }
    return verdict

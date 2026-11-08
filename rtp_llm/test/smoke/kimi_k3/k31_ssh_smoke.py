"""SSH/container transport for the shared K3.1 smoke worker."""

import concurrent.futures
import copy
import hashlib
import json
import os
import queue
import secrets
import shlex
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from . import two_host_launcher as launcher
from .k31_smoke import K31Worker, check_ci_result, smoke_config


class SSHWorker(K31Worker):
    def runtime_paths(self):
        workspace = Path(self.runtime["repo_root"]).resolve(strict=True)
        server = Path(self.runtime["server"]).resolve(strict=True)
        if not workspace.is_dir() or not server.is_file():
            raise ValueError("Missing source/runtime closure")
        return workspace, server

    def runtime_runfiles(self):
        return Path(str(self.server) + ".runfiles")

    def launcher_options(self):
        return super().launcher_options() + [
            "--traditional-artifact-root",
            str(Path(self.host["artifact_root"]).resolve()),
            "--draft-allow-checkpoint-root",
            self.host["draft_checkpoint_root"],
        ]

    @staticmethod
    def _require_native_artifact_root(root):
        if not root.is_absolute() or not root.is_dir() or root.is_symlink():
            raise ValueError(
                "Traditional smoke requires an existing private artifact root"
            )
        if root.stat().st_uid != os.geteuid():
            raise ValueError("Artifact root must belong to the runtime user")


def worker_main():
    disconnected = threading.Event()
    messages = queue.Queue()

    def receive():
        try:
            pending = b""
            while True:
                chunk = os.read(sys.stdin.fileno(), 65536)
                if not chunk:
                    break
                pending += chunk
                if len(pending) > 4 * 1024**2:
                    raise ValueError("Smoke control message exceeds 4 MiB")
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    messages.put(line.decode())
        finally:
            disconnected.set()
            messages.put(None)

    threading.Thread(target=receive, daemon=True).start()
    worker = SSHWorker(disconnected)
    output = sys.stdout
    # Worker diagnostics must not corrupt the line-delimited control channel.
    sys.stdout = sys.stderr
    try:
        for line in iter(messages.get, None):
            message = json.loads(line)
            operation = message["operation"]
            try:
                if operation == "prepare":
                    worker.runtime = message["runtime"]
                    if str(
                        Path(worker.runtime["repo_root"]).resolve()
                    ) not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
                        raise ValueError(
                            "Use the selected runtime bootstrap before preparing a role"
                        )
                    result = worker.prepare_native(
                        message["config"], message["role"], message["run_id"]
                    )
                    launcher.checkpoint_preflight(
                        message["traditional_config"], message["role"], worker.task
                    )
                    for name, expected in (
                        message["traditional_config"].get("source_files", {}).items()
                    ):
                        path = (worker.workspace / name).resolve(strict=True)
                        if (
                            not path.is_relative_to(worker.workspace)
                            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
                        ):
                            raise ValueError(f"Source manifest mismatch: {name}")
                    result.update(
                        execution_mode="traditional-ssh",
                        source_identity_kind="declared-pinned-runtime",
                    )
                    (worker.task / "identity.json").write_text(
                        json.dumps(result, indent=2) + "\n"
                    )
                elif operation in ("start", "status"):
                    result = getattr(worker, operation)()
                elif operation == "requests":
                    result = worker.requests(message)
                    result["audit"] = check_ci_result(
                        worker.task / "requests" / "accuracy.json",
                        required_phases=message.get("phases", ("regular", "long")),
                    )
                elif operation == "close":
                    worker.close()
                    result = {"closed": True, "artifact_dir": str(worker.task)}
                else:
                    raise ValueError("Unknown smoke control operation")
                reply = {"passed": True, "result": result}
            except Exception as error:
                traceback.print_exc()
                reply = {"passed": False, "error": f"{type(error).__name__}: {error}"}
            output.write(json.dumps(reply) + "\n")
            output.flush()
            if operation == "close":
                worker = None
                break
    finally:
        if worker is not None:
            worker.close()


class RoleConnection:
    def __init__(self, config, role, output):
        host = config[role]
        command = launcher.runtime_command(
            host,
            [
                "exec",
                "-i",
                "-u",
                config.get("container_user", "luohaocheng.lhc"),
                host["container"],
                "python3",
                str(
                    Path(host["repo_root"])
                    / Path(launcher.SCRIPT).with_name("k31_runtime_bootstrap.py")
                ),
                "--source",
                host["repo_root"],
                "--server",
                host["server"],
                "--artifact-root",
                host["artifact_root"],
            ],
        )
        options = ["-F", config["ssh_config"]] if config.get("ssh_config") else []
        self.log = (output / (role + "-control.log")).open("x")
        self.process = subprocess.Popen(
            [
                "ssh",
                *options,
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                host["ssh_target"],
                shlex.join(command),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
        )
        self.reader = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.broken = False

    def call(self, operation, timeout=90, **fields):
        if self.broken:
            raise RuntimeError("Smoke control channel is no longer usable")
        self.process.stdin.write(json.dumps(dict(operation=operation, **fields)) + "\n")
        self.process.stdin.flush()
        try:
            line = self.reader.submit(self.process.stdout.readline).result(
                timeout=timeout
            )
        except concurrent.futures.TimeoutError:
            self.broken = True
            self.process.stdin.close()
            raise TimeoutError(f"Smoke control operation timed out: {operation}")
        if not line:
            raise RuntimeError(
                "Smoke worker control channel closed; see role control log"
            )
        reply = json.loads(line)
        if reply.get("passed") is not True:
            raise RuntimeError(reply.get("error", "Smoke worker failed"))
        return reply["result"]

    def close(self):
        if not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            # EOF sets the remote worker's disconnect event and runs its
            # scoped service cleanup, even after a request/control failure.
            self.process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=15)
        self.reader.shutdown(wait=False, cancel_futures=True)
        self.log.close()
        if self.process.returncode != 0:
            raise RuntimeError(
                f"Smoke worker control process exited {self.process.returncode}"
            )


def runtime_config(base):
    """Remove only transport fields before the common configuration gate."""
    config = smoke_config(copy.deepcopy(base))
    config["external_revision"] = base.get(
        "external_revision", base.get("source_commit")
    )
    config["evidence_token"] = secrets.token_hex(32)
    for role in launcher.ROLES:
        host = config[role]
        for key in (
            "ssh_target",
            "runtime",
            "runtime_group",
            "container",
            "repo_root",
            "server",
            "cluster",
        ):
            host.pop(key, None)
        # Each existing traditional configuration pins the allowed checkpoint
        # root. Native workers independently audit both checkpoint manifests.
        host.setdefault("allow_checkpoint_root", host["checkpoint_root"])
        host.setdefault("draft_checkpoint_root", host["checkpoint_root"])
        host.setdefault("evidence_port", host["port"] + 90)
    return config


def run(base, output, startup_timeout, stages=("regular", "long")):
    base = copy.deepcopy(base)
    for role in launcher.ROLES:
        base[role].setdefault(
            "server",
            str(Path(base[role]["repo_root"]) / "bazel-bin/rtp_llm/rtp_llm_server"),
        )
    launcher.validate(base)
    config = runtime_config(base)
    run_id = "k31-pd-" + hashlib.sha256(base["run_id"].encode()).hexdigest()[:24]
    output.mkdir(parents=True, exist_ok=False)
    launcher.atomic(output / "config.json", base)
    connections, identities = {}, {}
    result = dict(passed=False, server_lifecycles=1, decode_dp=2, decode_tp=4)
    try:
        for role in launcher.ROLES:
            connections[role] = RoleConnection(base, role, output)
            identities[role] = connections[role].call(
                "prepare",
                config=config,
                role=role,
                run_id=run_id,
                runtime={key: base[role][key] for key in ("repo_root", "server")},
                traditional_config=base,
            )
        for role in ("decode", "prefill"):
            connections[role].call("start")
        deadline = time.monotonic() + startup_timeout
        while True:
            ready = {
                role: connection.call("status")["healthy"]
                for role, connection in connections.items()
            }
            launcher.atomic(output / "readiness.json", ready)
            if all(ready.values()):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("K3.1 services did not become ready")
            time.sleep(5)
        timeout = config.get("request_timeout_s", 14400)
        requests = connections["prefill"].call(
            "requests", timeout=timeout + 120, timeout_s=timeout, phases=stages
        )
        result["audit"] = requests["audit"]
        if requests["audit"]["passed"] is not True:
            raise RuntimeError("K3.1 smoke execution audit failed")
        result["passed"] = True
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        result["roles"] = identities
        result["cleanup"] = {}
        for role, connection in connections.items():
            try:
                result["cleanup"][role] = connection.call("close", timeout=600)
            except Exception as error:
                result["cleanup"][role] = {"error": str(error)}
                result["passed"] = False
            finally:
                try:
                    connection.close()
                except Exception as error:
                    result["cleanup"][role]["transport_error"] = str(error)
                    result["passed"] = False
        launcher.atomic(output / "RESULT.json", result)
    print(
        f"K3.1 smoke {'PASS' if result['passed'] else 'FAIL'}; evidence {output}",
        flush=True,
    )
    return 0 if result["passed"] else 1

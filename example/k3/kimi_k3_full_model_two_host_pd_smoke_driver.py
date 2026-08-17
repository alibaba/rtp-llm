#!/usr/bin/env python3
"""Launch both sides of the Kimi K3 full-model PD smoke concurrently over SSH."""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import TextIO


ROLE_SCRIPT = "./example/k3/kimi_k3_full_model_two_host_pd_smoke.sh"


def env_default(name: str, fallback: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start Kimi K3 Prefill and Decode smoke roles concurrently on two "
            "SSH hosts, stream both logs, and return a combined exit status."
        )
    )
    parser.add_argument("--prefill-ssh-target", default=env_default("PREFILL_SSH_TARGET"))
    parser.add_argument("--decode-ssh-target", default=env_default("DECODE_SSH_TARGET"))
    parser.add_argument("--prefill-repo-root", default=env_default("PREFILL_REPO_ROOT"))
    parser.add_argument("--decode-repo-root", default=env_default("DECODE_REPO_ROOT"))
    parser.add_argument(
        "--prefill-checkpoint-path",
        default=env_default("PREFILL_CHECKPOINT_PATH", env_default("CHECKPOINT_PATH")),
    )
    parser.add_argument(
        "--decode-checkpoint-path",
        default=env_default("DECODE_CHECKPOINT_PATH", env_default("CHECKPOINT_PATH")),
    )
    parser.add_argument("--prefill-endpoint", default=env_default("PREFILL_ENDPOINT"))
    parser.add_argument("--decode-endpoint", default=env_default("DECODE_ENDPOINT"))
    parser.add_argument("--run-id", default=env_default("SMOKE_RUN_ID"))
    parser.add_argument(
        "--suite",
        choices=("all",),
        default=env_default("SMOKE_SUITE", "all"),
        help="complete accuracy suite; all is required before merge",
    )
    parser.add_argument(
        "--result-endpoint", default=env_default("SMOKE_RESULT_ENDPOINT")
    )
    parser.add_argument(
        "--container", default=env_default("SMOKE_CONTAINER", "lhc_GPU")
    )
    parser.add_argument(
        "--container-user",
        default=env_default("SMOKE_CONTAINER_USER", "luohaocheng.lhc"),
    )
    parser.add_argument(
        "--prefill-container-runtime",
        choices=("docker", "pouch"),
        default=env_default("PREFILL_CONTAINER_RUNTIME", "docker"),
    )
    parser.add_argument(
        "--decode-container-runtime",
        choices=("docker", "pouch"),
        default=env_default("DECODE_CONTAINER_RUNTIME", "docker"),
    )
    parser.add_argument("--ssh-bin", default=env_default("SMOKE_SSH_BIN", "ssh"))
    parser.add_argument(
        "--artifact-root",
        type=pathlib.Path,
        default=pathlib.Path(
            env_default("SMOKE_CONTROLLER_ARTIFACT_ROOT", "/tmp/kimi-k3-pd-smoke-controller")
        ),
    )
    parser.add_argument(
        "--overall-timeout",
        type=int,
        default=int(env_default("SMOKE_CONTROLLER_TIMEOUT_S", "32400")),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    required = (
        "prefill_ssh_target",
        "decode_ssh_target",
        "prefill_repo_root",
        "decode_repo_root",
        "prefill_checkpoint_path",
        "decode_checkpoint_path",
        "prefill_endpoint",
        "decode_endpoint",
        "run_id",
    )
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error("missing required settings: " + ", ".join(missing))
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        parser.error("--run-id may contain only letters, digits, dot, underscore and dash")
    endpoint_pattern = r"[^:]+:[0-9]+"
    for name in ("prefill_endpoint", "decode_endpoint"):
        if re.fullmatch(endpoint_pattern, getattr(args, name)) is None:
            parser.error(f"--{name.replace('_', '-')} must have host:port form")
    if args.result_endpoint and re.fullmatch(endpoint_pattern, args.result_endpoint) is None:
        parser.error("--result-endpoint must have host:port form")
    if args.overall_timeout <= 0:
        parser.error("--overall-timeout must be positive")
    return args


def forwarded_optional_environment(role: str) -> dict[str, str]:
    result: dict[str, str] = {}
    names = (
        "SMOKE_ARTIFACT_ROOT",
        "SMOKE_STARTUP_TIMEOUT_S",
        "SMOKE_REQUEST_TIMEOUT_S",
        "SMOKE_RESULT_TIMEOUT_S",
        "SMOKE_MAX_TOKENS",
        "RTP_LLM_SKIP_BUILD",
    )
    for name in names:
        value = env_default(name)
        if value is not None:
            result[name] = value
    binary = env_default(
        f"{role.upper()}_RTP_LLM_SERVER_BINARY",
        env_default("RTP_LLM_SERVER_BINARY"),
    )
    if binary is not None:
        result["RTP_LLM_SERVER_BINARY"] = binary
    return result


def build_remote_command(args: argparse.Namespace, role: str) -> str:
    is_prefill = role == "prefill"
    repo_root = args.prefill_repo_root if is_prefill else args.decode_repo_root
    checkpoint = (
        args.prefill_checkpoint_path if is_prefill else args.decode_checkpoint_path
    )
    runtime = (
        args.prefill_container_runtime if is_prefill else args.decode_container_runtime
    )
    role_environment = {
        "CHECKPOINT_PATH": checkpoint,
        "PREFILL_ENDPOINT": args.prefill_endpoint,
        "DECODE_ENDPOINT": args.decode_endpoint,
        "SMOKE_RUN_ID": args.run_id,
        "SMOKE_SUITE": args.suite,
        **forwarded_optional_environment(role),
    }
    if args.result_endpoint:
        role_environment["SMOKE_RESULT_ENDPOINT"] = args.result_endpoint

    env_command = ["env"]
    env_command.extend(f"{key}={value}" for key, value in role_environment.items())
    env_command.extend((ROLE_SCRIPT, role))
    inner = f"cd {shlex.quote(repo_root)} && exec {shlex.join(env_command)}"
    return shlex.join(
        (
            runtime,
            "exec",
            "-u",
            args.container_user,
            args.container,
            "bash",
            "-lc",
            inner,
        )
    )


def build_ssh_command(args: argparse.Namespace, role: str) -> list[str]:
    target = args.prefill_ssh_target if role == "prefill" else args.decode_ssh_target
    return [
        args.ssh_bin,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        target,
        build_remote_command(args, role),
    ]


@dataclass
class RemoteRole:
    role: str
    command: list[str]
    log_path: pathlib.Path
    process: subprocess.Popen[str] | None = None
    reader: threading.Thread | None = None
    log_file: TextIO | None = None

    def start(self) -> None:
        self.log_file = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        def stream() -> None:
            assert self.process is not None and self.process.stdout is not None
            assert self.log_file is not None
            for line in self.process.stdout:
                self.log_file.write(line)
                self.log_file.flush()
                print(f"[{self.role}] {line}", end="", flush=True)

        self.reader = threading.Thread(target=stream, daemon=True)
        self.reader.start()

    def poll(self) -> int | None:
        assert self.process is not None
        return self.process.poll()

    def terminate(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()

    def finish(self) -> int:
        if self.process is None:
            if self.log_file is not None:
                self.log_file.close()
            return 127
        rc = self.process.wait()
        if self.reader is not None:
            self.reader.join(timeout=5)
        if self.log_file is not None:
            self.log_file.close()
        return rc


def show_tail(path: pathlib.Path, line_count: int = 80) -> None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    print(f"--- tail {path} ---", file=sys.stderr)
    for line in lines[-line_count:]:
        print(line, file=sys.stderr)


def main() -> int:
    args = parse_args()
    commands = {role: build_ssh_command(args, role) for role in ("decode", "prefill")}
    if args.dry_run:
        for role in ("decode", "prefill"):
            print(f"{role}: {shlex.join(commands[role])}")
        return 0

    run_dir = args.artifact_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    roles = {
        role: RemoteRole(role, commands[role], run_dir / f"{role}.log")
        for role in ("decode", "prefill")
    }
    started_at = time.monotonic()
    controller_error: BaseException | None = None
    print(
        f"starting Decode and Prefill concurrently; run_id={args.run_id} "
        f"suite={args.suite} controller_artifacts={run_dir}"
    )
    try:
        # Launch back-to-back without waiting for either model to load. The role
        # scripts perform model/result-channel readiness handshakes themselves.
        roles["decode"].start()
        roles["prefill"].start()
        while True:
            statuses = {role: remote.poll() for role, remote in roles.items()}
            if all(status is not None for status in statuses.values()):
                break
            failed = [role for role, status in statuses.items() if status not in (None, 0)]
            if failed:
                for role, remote in roles.items():
                    if statuses[role] is None:
                        remote.terminate()
                break
            if time.monotonic() - started_at > args.overall_timeout:
                raise TimeoutError(
                    f"smoke exceeded controller timeout {args.overall_timeout}s"
                )
            time.sleep(0.5)
    except (KeyboardInterrupt, TimeoutError, OSError) as exc:
        controller_error = exc
        print(f"controller abort: {exc}", file=sys.stderr)
        for remote in roles.values():
            remote.terminate()
    finally:
        return_codes = {role: remote.finish() for role, remote in roles.items()}

    if controller_error is not None or any(code != 0 for code in return_codes.values()):
        print(
            f"FAIL: controller_error={controller_error!r} "
            f"remote role exit codes={return_codes}",
            file=sys.stderr,
        )
        for remote in roles.values():
            show_tail(remote.log_path)
        return 1
    print(f"PASS: both remote roles completed successfully; artifacts={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

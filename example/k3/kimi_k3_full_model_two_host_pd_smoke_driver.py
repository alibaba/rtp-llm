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
            "Start the Kimi K3 Decode smoke role first, wait for readiness, then "
            "start Prefill on the other SSH host and return a combined exit status."
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
    parser.add_argument(
        "--prefill-sp-checkpoint-path",
        default=env_default(
            "PREFILL_SP_CHECKPOINT_PATH", env_default("SP_CHECKPOINT_PATH")
        ),
    )
    parser.add_argument(
        "--decode-sp-checkpoint-path",
        default=env_default(
            "DECODE_SP_CHECKPOINT_PATH", env_default("SP_CHECKPOINT_PATH")
        ),
    )
    parser.add_argument("--prefill-endpoint", default=env_default("PREFILL_ENDPOINT"))
    parser.add_argument("--decode-endpoint", default=env_default("DECODE_ENDPOINT"))
    parser.add_argument("--run-id", default=env_default("SMOKE_RUN_ID"))
    parser.add_argument(
        "--suite",
        choices=("flow", "all"),
        default=env_default("SMOKE_SUITE", "all"),
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
        "--prefill-ssh-control-path",
        default=env_default("PREFILL_SSH_CONTROL_PATH"),
    )
    parser.add_argument(
        "--decode-ssh-control-path",
        default=env_default("DECODE_SSH_CONTROL_PATH"),
    )
    parser.add_argument(
        "--remote-detached",
        action="store_true",
        help=(
            "start both role scripts as detached container execs and poll short-lived "
            "status commands; use this when a WebTerminal relay cannot carry a long SSH session"
        ),
    )
    parser.add_argument(
        "--remote-control-root",
        default=env_default(
            "SMOKE_REMOTE_CONTROL_ROOT",
            env_default("SMOKE_ARTIFACT_ROOT", "/tmp/kimi-k3-two-host-pd-smoke"),
        ),
    )
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
    parser.add_argument(
        "--prefill-start-delay-s",
        type=float,
        default=float(env_default("SMOKE_PREFILL_START_DELAY_S", "15")),
        help="start Decode first, then wait this many seconds before checking readiness",
    )
    parser.add_argument(
        "--decode-ready-timeout-s",
        type=int,
        default=int(env_default("SMOKE_STARTUP_TIMEOUT_S", "14400")),
        help="wait at most this many seconds for Decode /health before starting Prefill",
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
        "prefill_sp_checkpoint_path",
        "decode_sp_checkpoint_path",
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
    if args.prefill_start_delay_s < 0:
        parser.error("--prefill-start-delay-s must be non-negative")
    if args.decode_ready_timeout_s <= 0:
        parser.error("--decode-ready-timeout-s must be positive")
    if not args.remote_control_root.startswith("/"):
        parser.error("--remote-control-root must be an absolute path")
    return args


def forwarded_optional_environment(role: str) -> dict[str, str]:
    result: dict[str, str] = {}
    names = (
        "SMOKE_ARTIFACT_ROOT",
        "SMOKE_STARTUP_TIMEOUT_S",
        "SMOKE_REQUEST_TIMEOUT_S",
        "SMOKE_RESULT_TIMEOUT_S",
        "SMOKE_MAX_TOKENS",
        "SMOKE_IDENTITY_MAX_TOKENS",
        "SMOKE_SINGLE_EXACT_MAX_TOKENS",
        "SMOKE_MTP_CHUNK_MAX_TOKENS",
        "SMOKE_RDMA_PREWARM_ATTEMPTS",
        "SMOKE_RDMA_PREWARM_BACKOFF_S",
        "SMOKE_RDMA_PREWARM_SETTLE_S",
        "SMOKE_ACCL_USE_NICS",
        "SMOKE_EXPECTED_LAYERS",
        "SMOKE_BLOCK_SIZE",
        "SMOKE_KERNEL_BLOCK_SIZE",
        "SMOKE_CHUNK_TOKENS",
        "SMOKE_LINEAR_STEP",
        "SMOKE_CHUNKWISE_RDMA",
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


def role_launch_parts(
    args: argparse.Namespace, role: str
) -> tuple[str, str, str, list[str]]:
    is_prefill = role == "prefill"
    repo_root = args.prefill_repo_root if is_prefill else args.decode_repo_root
    checkpoint = (
        args.prefill_checkpoint_path if is_prefill else args.decode_checkpoint_path
    )
    sp_checkpoint = (
        args.prefill_sp_checkpoint_path
        if is_prefill
        else args.decode_sp_checkpoint_path
    )
    runtime = (
        args.prefill_container_runtime if is_prefill else args.decode_container_runtime
    )
    role_environment = {
        "CHECKPOINT_PATH": checkpoint,
        "SP_CHECKPOINT_PATH": sp_checkpoint,
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
    return repo_root, runtime, args.container, env_command


def build_remote_command(args: argparse.Namespace, role: str) -> str:
    repo_root, runtime, container, env_command = role_launch_parts(args, role)
    inner = f"cd {shlex.quote(repo_root)} && exec {shlex.join(env_command)}"
    return shlex.join(
        (
            runtime,
            "exec",
            "-u",
            args.container_user,
            container,
            "bash",
            "-lc",
            inner,
        )
    )


def detached_control_paths(args: argparse.Namespace, role: str) -> dict[str, str]:
    base = pathlib.PurePosixPath(args.remote_control_root) / args.run_id / "controller"
    return {
        "dir": str(base),
        "log": str(base / f"{role}.log"),
        "pid": str(base / f"{role}.pid"),
        "status": str(base / f"{role}.status"),
    }


def build_detached_remote_command(args: argparse.Namespace, role: str) -> str:
    repo_root, runtime, container, env_command = role_launch_parts(args, role)
    paths = detached_control_paths(args, role)
    role_inner = f"cd {shlex.quote(repo_root)} && exec {shlex.join(env_command)}"
    wrapper = "\n".join(
        (
            "set -u",
            f"mkdir -p {shlex.quote(paths['dir'])}",
            f"test ! -e {shlex.quote(paths['status'])}",
            (
                f"setsid bash -lc {shlex.quote(role_inner)} "
                f">{shlex.quote(paths['log'])} 2>&1 &"
            ),
            "child=$!",
            f"printf '%s\\n' \"$child\" >{shlex.quote(paths['pid'])}",
            "set +e",
            "wait \"$child\"",
            "rc=$?",
            f"printf '%s\\n' \"$rc\" >{shlex.quote(paths['status'])}",
            "exit 0",
        )
    )
    return shlex.join(
        (
            runtime,
            "exec",
            "-d",
            "-u",
            args.container_user,
            container,
            "bash",
            "-lc",
            wrapper,
        )
    )


def build_ssh_command(args: argparse.Namespace, role: str) -> list[str]:
    target = args.prefill_ssh_target if role == "prefill" else args.decode_ssh_target
    control_path = (
        args.prefill_ssh_control_path
        if role == "prefill"
        else args.decode_ssh_control_path
    )
    command = [
        args.ssh_bin,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=120",
    ]
    if control_path:
        command.extend(("-S", control_path))
    command.extend((target, build_remote_command(args, role)))
    return command


def build_short_ssh_command(
    args: argparse.Namespace, role: str, remote_command: str
) -> list[str]:
    target = args.prefill_ssh_target if role == "prefill" else args.decode_ssh_target
    control_path = (
        args.prefill_ssh_control_path
        if role == "prefill"
        else args.decode_ssh_control_path
    )
    command = [
        args.ssh_bin,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
    ]
    if control_path:
        command.extend(("-S", control_path))
    command.extend((target, remote_command))
    return command


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


def run_short_ssh(
    args: argparse.Namespace,
    role: str,
    remote_command: str,
    *,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        build_short_ssh_command(args, role, remote_command),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def fetch_detached_log(
    args: argparse.Namespace, role: str, destination: pathlib.Path
) -> None:
    result = run_short_ssh(
        args,
        role,
        f"cat {shlex.quote(detached_control_paths(args, role)['log'])}",
    )
    destination.write_text(
        result.stdout + (result.stderr if result.returncode else ""),
        encoding="utf-8",
    )


def stop_detached_role(args: argparse.Namespace, role: str) -> None:
    paths = detached_control_paths(args, role)
    _, runtime, container, _ = role_launch_parts(args, role)
    stop_inner = "\n".join(
        (
            f"test -f {shlex.quote(paths['pid'])} || exit 0",
            f"read -r pid <{shlex.quote(paths['pid'])}",
            "case \"$pid\" in ''|*[!0-9]*) exit 2;; esac",
            "kill -TERM -- \"-$pid\" 2>/dev/null || true",
        )
    )
    remote_command = shlex.join(
        (
            runtime,
            "exec",
            "-u",
            args.container_user,
            container,
            "bash",
            "-lc",
            stop_inner,
        )
    )
    try:
        run_short_ssh(args, role, remote_command)
    except (OSError, subprocess.TimeoutExpired):
        pass


def wait_for_decode_ready(args: argparse.Namespace) -> None:
    decode_port = args.decode_endpoint.rsplit(":", 1)[1]
    deadline = time.monotonic() + args.decode_ready_timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            result = run_short_ssh(
                args,
                "decode",
                (
                    "curl -fsS --max-time 2 "
                    f"http://127.0.0.1:{decode_port}/health >/dev/null"
                ),
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            last_error = str(exc)
        else:
            if result.returncode == 0:
                print("Decode /health is ready; starting Prefill", flush=True)
                return
            last_error = result.stderr.strip() or f"curl rc={result.returncode}"
        time.sleep(2)
    raise TimeoutError(
        "Decode did not become ready before Prefill launch "
        f"({args.decode_ready_timeout_s}s): {last_error}"
    )


def run_detached(args: argparse.Namespace, run_dir: pathlib.Path) -> int:
    roles = ("decode", "prefill")
    launch_errors = []
    for role in roles:
        command = build_short_ssh_command(
            args, role, build_detached_remote_command(args, role)
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            output, _ = process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
        if process.returncode != 0:
            launch_errors.append(
                f"{role} detached launch rc={process.returncode}: {output.strip()}"
            )
            break
        if role == "decode":
            if args.prefill_start_delay_s:
                print(
                    f"Decode launch accepted; waiting {args.prefill_start_delay_s:g}s "
                    "before readiness check",
                    flush=True,
                )
                time.sleep(args.prefill_start_delay_s)
            try:
                wait_for_decode_ready(args)
            except (OSError, TimeoutError) as exc:
                launch_errors.append(f"Decode readiness failed: {exc}")
                break
    if launch_errors:
        for role in roles:
            stop_detached_role(args, role)
        print("FAIL: " + "; ".join(launch_errors), file=sys.stderr)
        return 1

    print("detached Decode and Prefill role scripts accepted; polling status files")
    started_at = time.monotonic()
    statuses: dict[str, int | None] = {role: None for role in roles}
    poll_failures: dict[str, int] = {role: 0 for role in roles}
    try:
        while any(status is None for status in statuses.values()):
            if time.monotonic() - started_at > args.overall_timeout:
                raise TimeoutError(
                    f"smoke exceeded controller timeout {args.overall_timeout}s"
                )
            for role in roles:
                if statuses[role] is not None:
                    continue
                status_path = detached_control_paths(args, role)["status"]
                try:
                    result = run_short_ssh(
                        args,
                        role,
                        (
                            f"if test -f {shlex.quote(status_path)}; then "
                            f"cat {shlex.quote(status_path)}; else exit 3; fi"
                        ),
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    poll_failures[role] += 1
                    print(
                        f"[{role}] transient status poll failure "
                        f"{poll_failures[role]}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                if result.returncode == 0:
                    value = result.stdout.strip()
                    if re.fullmatch(r"[0-9]+", value) is None:
                        raise RuntimeError(
                            f"{role} returned invalid detached status {value!r}"
                        )
                    statuses[role] = int(value)
                    print(f"[{role}] detached role completed rc={value}")
                    continue
                if result.returncode == 3:
                    poll_failures[role] = 0
                    continue
                poll_failures[role] += 1
                print(
                    f"[{role}] transient status poll rc={result.returncode}: "
                    f"{result.stderr.strip()}",
                    file=sys.stderr,
                )
                if poll_failures[role] >= 12:
                    raise RuntimeError(
                        f"{role} status polling failed {poll_failures[role]} times"
                    )
            time.sleep(2)
    except (KeyboardInterrupt, TimeoutError, OSError, RuntimeError) as exc:
        print(f"controller abort: {exc}", file=sys.stderr)
        for role in roles:
            if statuses[role] is None:
                stop_detached_role(args, role)
        for role in roles:
            fetch_detached_log(args, role, run_dir / f"{role}.log")
            show_tail(run_dir / f"{role}.log")
        return 1

    for role in roles:
        fetch_detached_log(args, role, run_dir / f"{role}.log")
    if any(status != 0 for status in statuses.values()):
        print(f"FAIL: detached remote role exit codes={statuses}", file=sys.stderr)
        for role in roles:
            show_tail(run_dir / f"{role}.log")
        return 1
    print(f"PASS: both detached remote roles completed; artifacts={run_dir}")
    return 0


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
    if args.remote_detached:
        commands = {
            role: build_short_ssh_command(
                args, role, build_detached_remote_command(args, role)
            )
            for role in ("decode", "prefill")
        }
    else:
        commands = {
            role: build_ssh_command(args, role) for role in ("decode", "prefill")
        }
    if args.dry_run:
        for role in ("decode", "prefill"):
            print(f"{role}: {shlex.join(commands[role])}")
        return 0

    run_dir = args.artifact_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    print(
        f"starting Decode before Prefill; prefill_start_delay_s="
        f"{args.prefill_start_delay_s:g}; run_id={args.run_id} "
        f"suite={args.suite} controller_artifacts={run_dir} "
        f"remote_detached={args.remote_detached}"
    )
    if args.remote_detached:
        return run_detached(args, run_dir)
    roles = {
        role: RemoteRole(role, commands[role], run_dir / f"{role}.log")
        for role in ("decode", "prefill")
    }
    started_at = time.monotonic()
    controller_error: BaseException | None = None
    try:
        # HTTP readiness is only the launch-order gate. The Prefill role runs a
        # batch-sized RDMA prewarm barrier before the formal accuracy suite.
        roles["decode"].start()
        if args.prefill_start_delay_s:
            time.sleep(args.prefill_start_delay_s)
        wait_for_decode_ready(args)
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

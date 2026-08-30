#!/usr/bin/env python3
"""Launch one Prefill and a two-node DP16 Decode gang over SSH."""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time

try:
    from example.k3 import kimi_k3_full_model_two_host_pd_smoke_driver as common
except ModuleNotFoundError:  # direct execution from example/k3
    import kimi_k3_full_model_two_host_pd_smoke_driver as common


ROLE_SCRIPT = "./example/k3/kimi_k3_full_model_two_host_pd_smoke.sh"
ROLES = ("decode0", "decode1", "prefill")


def env_default(name: str, fallback: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start Kimi K3 Prefill TP8/EP8 plus Decode DP16/KTP16/EP16."
    )
    for role, prefix in (("prefill", "PREFILL"), ("decode0", "DECODE0"), ("decode1", "DECODE1")):
        parser.add_argument(
            f"--{role}-ssh-target", default=env_default(f"{prefix}_SSH_TARGET")
        )
        parser.add_argument(
            f"--{role}-repo-root", default=env_default(f"{prefix}_REPO_ROOT")
        )
        parser.add_argument(
            f"--{role}-checkpoint-path",
            default=env_default(f"{prefix}_CHECKPOINT_PATH", env_default("CHECKPOINT_PATH")),
        )
        parser.add_argument(
            f"--{role}-container-runtime",
            choices=("docker", "pouch"),
            default=env_default(f"{prefix}_CONTAINER_RUNTIME", "docker"),
        )
        parser.add_argument(
            f"--{role}-ssh-control-path",
            default=env_default(f"{prefix}_SSH_CONTROL_PATH"),
        )
    parser.add_argument("--prefill-endpoint", default=env_default("PREFILL_ENDPOINT"))
    parser.add_argument("--decode0-endpoint", default=env_default("DECODE0_ENDPOINT"))
    parser.add_argument("--decode1-endpoint", default=env_default("DECODE1_ENDPOINT"))
    parser.add_argument("--run-id", default=env_default("SMOKE_RUN_ID"))
    parser.add_argument(
        "--suite", choices=("flow", "all"), default=env_default("SMOKE_SUITE", "flow")
    )
    parser.add_argument("--result-endpoint", default=env_default("SMOKE_RESULT_ENDPOINT"))
    parser.add_argument("--container", default=env_default("SMOKE_CONTAINER", "lhc_GPU"))
    parser.add_argument(
        "--container-user",
        default=env_default("SMOKE_CONTAINER_USER", "luohaocheng.lhc"),
    )
    parser.add_argument("--ssh-bin", default=env_default("SMOKE_SSH_BIN", "ssh"))
    parser.add_argument("--remote-detached", action="store_true")
    parser.add_argument(
        "--remote-control-root",
        default=env_default("SMOKE_REMOTE_CONTROL_ROOT", "/tmp/kimi-k3-three-host-pd-smoke"),
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
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    required = [
        f"{role}_{field}"
        for role in ("prefill", "decode0", "decode1")
        for field in ("ssh_target", "repo_root", "checkpoint_path")
    ] + ["prefill_endpoint", "decode0_endpoint", "decode1_endpoint", "run_id"]
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error("missing required settings: " + ", ".join(missing))
    if len({args.prefill_ssh_target, args.decode0_ssh_target, args.decode1_ssh_target}) != 3:
        parser.error("Prefill, Decode0 and Decode1 must use three distinct SSH targets")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        parser.error("--run-id may contain only letters, digits, dot, underscore and dash")
    for name in ("prefill_endpoint", "decode0_endpoint", "decode1_endpoint"):
        if re.fullmatch(r"[^:]+:[1-9][0-9]*", getattr(args, name)) is None:
            parser.error(f"--{name.replace('_', '-')} must have host:port form")
    if args.result_endpoint and re.fullmatch(r"[^:]+:[1-9][0-9]*", args.result_endpoint) is None:
        parser.error("--result-endpoint must have host:port form")
    if not args.remote_control_root.startswith("/"):
        parser.error("--remote-control-root must be absolute")
    if args.overall_timeout <= 0 or args.prefill_start_delay_s < 0:
        parser.error("timeouts must be positive and start delay non-negative")
    if args.result_endpoint is None:
        host, port = split_endpoint(args.decode0_endpoint)
        result_port = port + 100
        if result_port > 65535:
            parser.error("Decode0 port is too high to derive the result endpoint")
        args.result_endpoint = f"{host}:{result_port}"
    # Ordinary Projection-KTP explicitly excludes speculative checkpoints.
    if env_default("SP_CHECKPOINT_PATH") or any(
        env_default(f"{prefix}_SP_CHECKPOINT_PATH")
        for prefix in ("PREFILL", "DECODE0", "DECODE1")
    ):
        parser.error("Projection-KTP ordinary Decode requires SP_CHECKPOINT_PATH unset")
    decode_role_addresses(args)  # validate the full ordered port plan early
    return args


def split_endpoint(endpoint: str) -> tuple[str, int]:
    host, port = endpoint.rsplit(":", 1)
    return host, int(port)


def decode_role_addresses(args: argparse.Namespace) -> list[str]:
    addresses: list[str] = []
    for endpoint in (args.decode0_endpoint, args.decode1_endpoint):
        host, base_port = split_endpoint(endpoint)
        for local_rank in range(8):
            http_port = base_port + local_rank * 9
            grpc_port = http_port + 1
            if grpc_port > 65535:
                raise ValueError(f"Decode worker port exceeds 65535 for {endpoint}")
            addresses.append(f"{host}:{http_port}:{grpc_port}")
    return addresses


def gang_config(args: argparse.Namespace) -> str:
    host0, port0 = split_endpoint(args.decode0_endpoint)
    host1, port1 = split_endpoint(args.decode1_endpoint)
    return (
        f"name:k3_part0,ip:{host0},port:{port0};"
        f"name:k3_part1,ip:{host1},port:{port1}"
    )


def handshake_file(args: argparse.Namespace, name: str) -> str:
    return str(
        pathlib.PurePosixPath(args.remote_control_root)
        / args.run_id
        / "controller"
        / name
    )


def secondary_completion_file(args: argparse.Namespace) -> str:
    return handshake_file(args, "decode1.success")


def primary_ready_file(args: argparse.Namespace) -> str:
    return handshake_file(args, "decode0.ready")


def primary_completion_file(args: argparse.Namespace) -> str:
    return handshake_file(args, "decode0.success")


def forwarded_optional_environment(role: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in (
        "SMOKE_ARTIFACT_ROOT",
        "SMOKE_STARTUP_TIMEOUT_S",
        "SMOKE_REQUEST_TIMEOUT_S",
        "SMOKE_RESULT_TIMEOUT_S",
        "SMOKE_MAX_TOKENS",
        "SMOKE_IDENTITY_MAX_TOKENS",
        "SMOKE_SINGLE_EXACT_MAX_TOKENS",
        "SMOKE_DECODE_KV_CACHE_MEM_MB",
        "SMOKE_RDMA_PREWARM_ATTEMPTS",
        "SMOKE_RDMA_PREWARM_TIMEOUT_S",
        "SMOKE_RDMA_PREWARM_BACKOFF_S",
        "SMOKE_RDMA_PREWARM_SETTLE_S",
        "SMOKE_EXPECTED_LAYERS",
        "SMOKE_BLOCK_SIZE",
        "SMOKE_KERNEL_BLOCK_SIZE",
        "SMOKE_CHUNK_TOKENS",
        "SMOKE_LINEAR_STEP",
        "SMOKE_CHUNKWISE_RDMA",
        "FT_CORE_DUMP_ON_EXCEPTION",
        "RTP_LLM_SKIP_BUILD",
    ):
        value = env_default(name)
        if value is not None:
            result[name] = value
    binary = env_default(
        f"{role.upper()}_RTP_LLM_SERVER_BINARY", env_default("RTP_LLM_SERVER_BINARY")
    )
    if binary:
        result["RTP_LLM_SERVER_BINARY"] = binary
    return result


def role_value(args: argparse.Namespace, role: str, suffix: str) -> str:
    return getattr(args, f"{role}_{suffix}")


def role_launch_parts(args: argparse.Namespace, role: str) -> tuple[str, str, str, list[str]]:
    script_role = "prefill" if role == "prefill" else "decode"
    environment = {
        "CHECKPOINT_PATH": role_value(args, role, "checkpoint_path"),
        "PREFILL_ENDPOINT": args.prefill_endpoint,
        "DECODE_ENDPOINT": args.decode0_endpoint,
        "SMOKE_RESULT_ENDPOINT": args.result_endpoint,
        "SMOKE_RUN_ID": args.run_id,
        "SMOKE_SUITE": args.suite,
        "SMOKE_DECODE_TOPOLOGY": "dp16_ktp16_ep16",
        "SMOKE_DECODE_ROLE_ADDRS": ",".join(decode_role_addresses(args)),
        "WORLD_RANK": "0",
        **forwarded_optional_environment(role),
    }
    if role != "prefill":
        node_index = 0 if role == "decode0" else 1
        environment.update(
            {
                "SMOKE_DECODE_NODE_INDEX": str(node_index),
                "WORLD_RANK": str(node_index * 8),
                "GANG_CONFIG_STRING": gang_config(args),
                # L20-a RoCE rails expose link-local and routed GIDs. NCCL's
                # automatic choice can select index 0 (link-local), which is
                # not routable between the two Decode hosts. Keep this
                # overrideable for other fabrics while making the supported
                # three-host topology select the routed RoCE v2 GID.
                "NCCL_IB_GID_INDEX": env_default("NCCL_IB_GID_INDEX", "3"),
            }
        )
    if role == "decode1":
        environment["SMOKE_SECONDARY_COMPLETION_FILE"] = secondary_completion_file(args)
    elif role == "decode0":
        environment["SMOKE_PRIMARY_READY_FILE"] = primary_ready_file(args)
        environment["SMOKE_PRIMARY_COMPLETION_FILE"] = primary_completion_file(args)
    command = ["env", *(f"{key}={value}" for key, value in environment.items()), ROLE_SCRIPT, script_role]
    return (
        role_value(args, role, "repo_root"),
        role_value(args, role, "container_runtime"),
        args.container,
        command,
    )


def role_remote_command(args: argparse.Namespace, role: str) -> str:
    repo, runtime, container, command = role_launch_parts(args, role)
    inner = f"cd {shlex.quote(repo)} && exec {shlex.join(command)}"
    return shlex.join((runtime, "exec", "-u", args.container_user, container, "bash", "-lc", inner))


def ssh_command(args: argparse.Namespace, role: str, remote_command: str, *, long: bool) -> list[str]:
    command = [
        args.ssh_bin,
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
    ]
    if long:
        command += ["-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=120"]
    control_path = role_value(args, role, "ssh_control_path")
    if control_path:
        command += ["-S", control_path]
    return command + [role_value(args, role, "ssh_target"), remote_command]


def control_paths(args: argparse.Namespace, role: str) -> dict[str, str]:
    base = pathlib.PurePosixPath(args.remote_control_root) / args.run_id / "controller"
    return {name: str(base / f"{role}.{name}") for name in ("log", "pid", "status")}


def container_control_command(args: argparse.Namespace, role: str, inner: str) -> str:
    _, runtime, container, _ = role_launch_parts(args, role)
    return shlex.join((runtime, "exec", "-u", args.container_user, container, "bash", "-lc", inner))


def detached_remote_command(args: argparse.Namespace, role: str) -> str:
    repo, runtime, container, command = role_launch_parts(args, role)
    paths = control_paths(args, role)
    base = str(pathlib.PurePosixPath(paths["pid"]).parent)
    inner = f"cd {shlex.quote(repo)} && exec {shlex.join(command)}"
    wrapper = "\n".join(
        (
            "set -u",
            f"mkdir -p {shlex.quote(base)}",
            f"test ! -e {shlex.quote(paths['status'])}",
            f"setsid bash -lc {shlex.quote(inner)} >{shlex.quote(paths['log'])} 2>&1 &",
            "child=$!",
            f"printf '%s\\n' \"$child\" >{shlex.quote(paths['pid'])}",
            "set +e",
            "wait \"$child\"",
            "rc=$?",
            f"printf '%s\\n' \"$rc\" >{shlex.quote(paths['status'])}",
            "exit 0",
        )
    )
    return shlex.join((runtime, "exec", "-d", "-u", args.container_user, container, "bash", "-lc", wrapper))


def run_short(args: argparse.Namespace, role: str, remote_command: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ssh_command(args, role, remote_command, long=False),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def signal_completion(args: argparse.Namespace, role: str) -> None:
    marker = secondary_completion_file(args) if role == "decode1" else primary_completion_file(args)
    inner = f"test ! -e {shlex.quote(marker)} && touch {shlex.quote(marker)}"
    result = run_short(args, role, container_control_command(args, role, inner))
    if result.returncode != 0:
        raise RuntimeError(f"failed to release {role}: {result.stderr.strip()}")


def primary_is_ready(args: argparse.Namespace) -> bool:
    marker = primary_ready_file(args)
    result = run_short(
        args,
        "decode0",
        container_control_command(args, "decode0", f"test -f {shlex.quote(marker)}"),
    )
    return result.returncode == 0


def stop_detached(args: argparse.Namespace, role: str) -> None:
    paths = control_paths(args, role)
    inner = "\n".join(
        (
            f"test -f {shlex.quote(paths['pid'])} || exit 0",
            f"read -r pid <{shlex.quote(paths['pid'])}",
            "case \"$pid\" in ''|*[!0-9]*) exit 2;; esac",
            "kill -TERM -- \"-$pid\" 2>/dev/null || true",
        )
    )
    try:
        run_short(args, role, container_control_command(args, role, inner))
    except (OSError, subprocess.TimeoutExpired):
        pass


def fetch_log(args: argparse.Namespace, role: str, destination: pathlib.Path) -> None:
    inner = f"cat {shlex.quote(control_paths(args, role)['log'])}"
    result = run_short(args, role, container_control_command(args, role, inner))
    destination.write_text(result.stdout + (result.stderr if result.returncode else ""), encoding="utf-8")


def launch_detached(args: argparse.Namespace, role: str) -> None:
    result = run_short(args, role, detached_remote_command(args, role), timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"{role} detached launch rc={result.returncode}: {result.stderr.strip()}")


def run_detached(args: argparse.Namespace, run_dir: pathlib.Path) -> int:
    statuses: dict[str, int | None] = {role: None for role in ROLES}
    secondary_released = False
    primary_released = False
    try:
        launch_detached(args, "decode0")
        launch_detached(args, "decode1")
        if args.prefill_start_delay_s:
            time.sleep(args.prefill_start_delay_s)
        launch_detached(args, "prefill")
        started = time.monotonic()
        while any(value is None for value in statuses.values()):
            if time.monotonic() - started > args.overall_timeout:
                raise TimeoutError(f"smoke exceeded {args.overall_timeout}s")
            for role in ROLES:
                if statuses[role] is not None:
                    continue
                path = control_paths(args, role)["status"]
                inner = f"if test -f {shlex.quote(path)}; then cat {shlex.quote(path)}; else exit 3; fi"
                result = run_short(args, role, container_control_command(args, role, inner))
                if result.returncode == 0:
                    value = result.stdout.strip()
                    if not value.isdigit():
                        raise RuntimeError(f"invalid {role} status {value!r}")
                    statuses[role] = int(value)
                    print(f"[{role}] completed rc={value}", flush=True)
                elif result.returncode != 3:
                    raise RuntimeError(f"{role} status poll failed: {result.stderr.strip()}")
            failed = {role: rc for role, rc in statuses.items() if rc not in (None, 0)}
            if failed:
                raise RuntimeError(f"role failed: {failed}")
            if (
                not secondary_released
                and statuses["prefill"] == 0
                and statuses["decode0"] is None
                and primary_is_ready(args)
            ):
                signal_completion(args, "decode1")
                secondary_released = True
                print("[controller] released successful Decode1 shutdown", flush=True)
            if secondary_released and not primary_released and statuses["decode1"] == 0:
                signal_completion(args, "decode0")
                primary_released = True
                print("[controller] released successful Decode0 shutdown", flush=True)
            time.sleep(2)
    except (KeyboardInterrupt, OSError, RuntimeError, TimeoutError) as exc:
        print(f"controller abort: {exc}", file=sys.stderr)
        for role in ROLES:
            if statuses[role] is None:
                stop_detached(args, role)
        rc = 1
    else:
        rc = 0 if all(value == 0 for value in statuses.values()) else 1
    for role in ROLES:
        fetch_log(args, role, run_dir / f"{role}.log")
    if rc:
        for role in ROLES:
            common.show_tail(run_dir / f"{role}.log")
    return rc


def start_roles(args: argparse.Namespace, roles: dict[str, common.RemoteRole]) -> None:
    roles["decode0"].start()
    roles["decode1"].start()
    if args.prefill_start_delay_s:
        time.sleep(args.prefill_start_delay_s)
    roles["prefill"].start()


def run_attached(args: argparse.Namespace, run_dir: pathlib.Path, commands: dict[str, list[str]]) -> int:
    roles = {
        role: common.RemoteRole(role, commands[role], run_dir / f"{role}.log")
        for role in ROLES
    }
    secondary_released = False
    primary_released = False
    error: BaseException | None = None
    started = time.monotonic()
    try:
        start_roles(args, roles)
        while True:
            statuses = {role: remote.poll() for role, remote in roles.items()}
            if any(rc not in (None, 0) for rc in statuses.values()):
                raise RuntimeError(f"role failed early: {statuses}")
            if (
                not secondary_released
                and statuses["prefill"] == 0
                and statuses["decode0"] is None
                and primary_is_ready(args)
            ):
                signal_completion(args, "decode1")
                secondary_released = True
            if secondary_released and not primary_released and statuses["decode1"] == 0:
                signal_completion(args, "decode0")
                primary_released = True
            if all(rc is not None for rc in statuses.values()):
                break
            if time.monotonic() - started > args.overall_timeout:
                raise TimeoutError(f"smoke exceeded {args.overall_timeout}s")
            time.sleep(0.5)
    except (KeyboardInterrupt, OSError, RuntimeError, TimeoutError) as exc:
        error = exc
        print(f"controller abort: {exc}", file=sys.stderr)
        for remote in roles.values():
            remote.terminate()
    return_codes = {role: remote.finish() for role, remote in roles.items()}
    if error is not None or any(rc != 0 for rc in return_codes.values()):
        print(f"FAIL: error={error!r} role exit codes={return_codes}", file=sys.stderr)
        for remote in roles.values():
            common.show_tail(remote.log_path)
        return 1
    return 0


def main() -> int:
    args = parse_args()
    if args.remote_detached:
        commands = {
            role: ssh_command(args, role, detached_remote_command(args, role), long=False)
            for role in ROLES
        }
    else:
        commands = {
            role: ssh_command(args, role, role_remote_command(args, role), long=True)
            for role in ROLES
        }
    if args.dry_run:
        for role in ROLES:
            print(f"{role}: {shlex.join(commands[role])}")
        for role, marker in (
            ("decode1", secondary_completion_file(args)),
            ("decode0", primary_completion_file(args)),
        ):
            print(
                f"{role}-success: "
                f"{shlex.join(ssh_command(args, role, container_control_command(args, role, 'touch ' + shlex.quote(marker)), long=False))}"
            )
        return 0

    run_dir = args.artifact_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    print(
        f"starting three-host DP16 Projection-KTP smoke run_id={args.run_id} "
        f"suite={args.suite} detached={args.remote_detached} artifacts={run_dir}",
        flush=True,
    )
    rc = run_detached(args, run_dir) if args.remote_detached else run_attached(args, run_dir, commands)
    if rc == 0:
        print(f"PASS: all three remote roles completed successfully; artifacts={run_dir}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

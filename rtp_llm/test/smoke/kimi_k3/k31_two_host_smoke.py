#!/usr/bin/env python3
"""Run the default K3.1 smoke and audit both stages after collecting evidence."""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import subprocess
import sys
import tarfile
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path, PurePosixPath

if __package__:
    from . import two_host_launcher as launcher
    from .k31_smoke_capacity import (
        LONG_INPUT_TOKENS,
        MAX_OUTPUT_TOKENS,
        MTP_RESERVE_TOKENS,
        fastsafetensors_proof,
    )
else:
    source_package = Path(__file__).resolve().parents[3]
    sys.path.append(str(source_package.parent))
    import rtp_llm

    # The compiled package supplies RPC stubs and the source tree supplies smoke clients.
    if str(source_package) not in rtp_llm.__path__:
        rtp_llm.__path__.append(str(source_package))
    import two_host_launcher as launcher
    from k31_smoke_capacity import (
        LONG_INPUT_TOKENS,
        MAX_OUTPUT_TOKENS,
        MTP_RESERVE_TOKENS,
        fastsafetensors_proof,
    )

SCRIPT = Path(launcher.SCRIPT).with_name("k31_two_host_smoke.py")
STAGES = ("regular", "long")
EXPECTED_COUNTS = dict(total=306, formal=268, preparation=38)
REQUEST_EVIDENCE = "request-evidence"
EVIDENCE_NAMES = (
    "control.json",
    "instance.json",
    "exit.json",
    "readiness.json",
    "runtime-geometry.json",
    "requests.log",
    "launcher.log",
    "server.log",
    "checkpoint-preflight.json",
    "draft_checkpoint-preflight.json",
    "rdma-preflight.json",
    "logs",
    REQUEST_EVIDENCE,
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stage_configs(base):
    require(
        base.get("target_only", False) is False, "default smoke requires native MTP"
    )
    configs = []
    for stage, decode_dp, block_count in (("regular", 2, 96), ("long", 1, 140)):
        config = copy.deepcopy(base)
        # The public cache-store implementation uses TCP. An internally
        # linked RDMA implementation can still be selected explicitly.
        config.setdefault("cache_store_rdma_mode", False)
        config.setdefault("prefill_kv_workspace_budget_gib", launcher.DEFAULT_KV_WORKSPACE_GIB)
        config.update(
            run_id=f"{base['run_id']}-{stage}",
            k31_smoke_stage=stage,
            decode_dp=decode_dp,
            target_only=False,
        )
        config["prefill"]["test_block_num"] = block_count
        if stage == "long":
            config["decode"].update(
                test_block_num=block_count,
                request_capacity=1,
                decode_capture_batches="1",
            )
            for role in launcher.ROLES:
                limit = config[role].get("max_seq_len", 2_097_152)
                require(
                    limit >= LONG_INPUT_TOKENS + MAX_OUTPUT_TOKENS + MTP_RESERVE_TOKENS,
                    f"{role} max_seq_len must admit the exact 1M continuation and generation reserve",
                )
        control = launcher.run_dir(config, "prefill") / "control.json"
        config["request_argv"] = [
            "python3",
            "-u",
            str(Path(config["prefill"]["repo_root"]) / SCRIPT),
            "--request-control",
            str(control),
        ]
        configs.append(launcher.validate(config))
    return configs


def request_args(config, root):
    # Reuse the launcher's endpoint inventory so DP owners follow the same rank stride.
    if __package__:
        from .worker import pd_endpoints
    else:
        from worker import pd_endpoints

    prefill, decode = (config[role] for role in launcher.ROLES)
    dp = launcher.decode_dp_size(config)
    endpoints = pd_endpoints(decode["port"], dp)
    rpc_ports = {
        item["rank"]: item["port"] for item in endpoints if item["kind"] == "rpc"
    }
    args = [
        "--base-url",
        f"http://127.0.0.1:{prefill['port']}",
        "--namespace",
        config["run_id"],
        "--geometry",
        str(root / "runtime-geometry.json"),
    ]
    for item in endpoints:
        if item["kind"] == "http":
            args += [
                "--decode-role-addr",
                f"{decode['ip']}:{item['port']}:{rpc_ports[item['rank']]}",
            ]
    return args


def run_requests(control):
    config = launcher.validate(read_json(control))
    root = launcher.run_dir(config, "prefill")
    require(
        control.resolve() == (root / "control.json").resolve(),
        "unexpected Prefill control path",
    )
    evidence = root / REQUEST_EVIDENCE
    args = request_args(config, root)
    if config["k31_smoke_stage"] == "long":
        if __package__:
            from .k31_long_prefix_smoke import main
        else:
            from k31_long_prefix_smoke import main

        return main(["run", *args, "--output", str(evidence)])
    require(config["k31_smoke_stage"] == "regular", "unsupported K3.1 smoke stage")
    if __package__:
        from .k31_full_smoke import main
    else:
        from k31_full_smoke import main

    prefill, decode = config["prefill"], config["decode"]
    source = Path(prefill["repo_root"])
    binary = (source / "bazel-bin/rtp_llm/rtp_llm_server").resolve(strict=True)
    commit = (
        config.get("source_commit")
        or subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
    )
    return main(
        [
            *args,
            "--output",
            str(evidence / "RESULT.json"),
            "--source-commit",
            commit,
            "--checkpoint",
            prefill["checkpoint"],
            "--draft-checkpoint",
            prefill["draft_checkpoint"],
            "--decode-health-url",
            f"http://{decode['ip']}:{decode['port']}/health",
            "--prefill-engine-log",
            str(root / "logs" / "engine.log"),
            "--prefill-grpc-port",
            str(prefill["port"] + 1),
            "--prefill-rpc-runfiles",
            str(binary) + ".runfiles",
        ]
    )


# The personal host account can read the same data-disk bind mounted by the
# runtime. Stream only run-owned evidence and require no remote script update.
COLLECT_HOST_EVIDENCE = r"""
set -euo pipefail
cd -- "$1"
shift
physical=$(pwd -P)
if [[ ! $physical =~ ^/(data[0-9]*|ssd)/ || ! -O . ]]; then
    printf 'Not an owned data-disk evidence directory: %s\n' "$physical" >&2
    exit 1
fi
case $(findmnt -T . -n -o FSTYPE) in
    ext4|xfs|btrfs) ;;
    *) printf 'Evidence directory is not on a local data filesystem\n' >&2; exit 1 ;;
esac
if [[ ! -f control.json || ! -s control.json || ! -O control.json || -L control.json ]]; then
    printf 'Missing owned control.json in %s\n' "$physical" >&2
    exit 1
fi
entries=()
for name; do
    if [[ ! -L $name && ( -f $name || -d $name ) ]]; then
        entries+=("$name")
    fi
done
find -P "${entries[@]}" -type f -uid "$EUID" -print0 |
    tar --hard-dereference --no-recursion --null --files-from=- --gzip --create --file=-
"""


def extract_evidence(archive_path, output):
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            require(
                (member.isfile() or member.isdir())
                and not name.is_absolute()
                and ".." not in name.parts
                and name.parts
                and name.parts[0] in EVIDENCE_NAMES,
                f"unsafe evidence archive entry: {member.name}",
            )
        controls = [member for member in members if member.name == "control.json"]
        require(
            len(controls) == 1 and controls[0].isfile() and controls[0].size > 0,
            "evidence archive lacks a unique nonempty control.json",
        )
        output.mkdir(exist_ok=False)
        archive.extractall(output, members=members)


def transfer_evidence(config, role, output):
    host = config[role]
    command = [
        "bash",
        "-c",
        COLLECT_HOST_EVIDENCE,
        "collect-k31-evidence",
        str(launcher.run_dir(config, role)),
        *EVIDENCE_NAMES,
    ]
    options = ["-F", config["ssh_config"]] if config.get("ssh_config") else []
    archive = output / f"{role}-evidence.tar.gz"
    with archive.open("xb") as data, (output / f"{role}-transfer.log").open(
        "xb"
    ) as errors:
        subprocess.run(
            [
                "ssh",
                *options,
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                host["ssh_target"],
                shlex.join(command),
            ],
            stdin=subprocess.DEVNULL,
            stdout=data,
            stderr=errors,
            check=True,
            timeout=1800,
        )
    extract_evidence(archive, output / role)
    require(
        read_json(output / role / "control.json") == config,
        f"{role}: downloaded evidence belongs to a different control receipt",
    )
    archive.unlink()


def wait_for_cleanup(config, timeout=120):
    deadline = time.monotonic() + timeout
    receipts = {}
    pending = set(launcher.ROLES)
    while pending:
        for role in sorted(pending):
            try:
                receipts[role] = launcher.remote(config, role, "status")
                if receipts[role].get("live_group_members"):
                    launcher.remote(config, role, "stop-role")
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                receipts[role] = {"error": f"{type(error).__name__}: {error}"}
            if "exited" in receipts[role] and not receipts[role].get("live_group_members"):
                pending.remove(role)
        if pending and time.monotonic() >= deadline:
            return False, receipts
        if pending:
            time.sleep(2)
    return True, receipts


def audit_startup(config, output):
    proof = {}
    readiness = read_json(output / "launch" / "readiness.json")
    for role in launcher.ROLES:
        directory = output / role
        require(
            read_json(directory / "control.json") == config,
            f"{role}: control receipt differs",
        )
        ready = readiness[role]
        require(ready.get("healthy") is True, f"{role}: service never became ready")
        geometry = read_json(directory / "runtime-geometry.json")
        require(
            geometry.get("runtime_verified") is True,
            f"{role}: missing runtime cache geometry",
        )
        require(
            ready.get("runtime_geometry") == geometry,
            f"{role}: readiness geometry differs",
        )
        receipts = ready.get("rank_prefill_kv_workspace_budget", {})
        require(
            set(receipts) == {str(rank) for rank in range(8)}
            and all(float(row["gib"]) == 10 for row in receipts.values()),
            f"{role}: expected all eight 10 GiB workspace receipts",
        )
        checkpoints = {}
        for checkpoint in launcher.checkpoint_fields(config):
            report = read_json(directory / f"{checkpoint}-preflight.json")
            require(
                report.get("passed") is True
                and report.get("load_method") == "fastsafetensors",
                f"{role}: {checkpoint} FastSafetensors preflight was not verified",
            )
            checkpoints[checkpoint] = {
                key: report[key] for key in ("config_sha256", "index_sha256")
            }
        proof[role] = dict(
            fastsafetensors_lines=fastsafetensors_proof(directory / "logs"),
            geometry=geometry,
            checkpoints=checkpoints,
        )
    require(
        proof["prefill"]["checkpoints"] == proof["decode"]["checkpoints"],
        "PD checkpoints differ",
    )
    return proof


def audit_long_native_route(proof, prefill_logs, decode_logs):
    if __package__:
        from .audit_k31_full_smoke import group_evidence, read_events, request_events
    else:
        from audit_k31_full_smoke import group_evidence, read_events, request_events

    prefill = read_events(prefill_logs, prefill_logs / "engine.log")
    decode = read_events(decode_logs, decode_logs / "engine.log")
    groups = {}
    for event in prefill[0]:
        if event.get("event") != "mtp_target_prefill_forward" or event.get("fake"):
            continue
        ids = group_evidence(prefill, event, 8)
        require(len(ids) == 1, "long smoke Prefill must execute BS1")
        groups.setdefault(ids[0], []).append(
            (event["execution_step"], event["context_lengths"][0], event["reuse_lengths"][0])
        )
    require(len(groups) == 2, "expected two native Prefill requests in the long stage")
    groups = sorted(groups.items(), key=lambda group: group[1][0][0])
    result = {}
    ranks = set(range(8))
    for name, (request_id, steps) in zip(("seed", "continuation"), groups):
        case = proof["cases"][name]
        cursor = case["prefill_total_reuse_len"]
        for _, query, prefix in steps:
            require(prefix == cursor and query > 0, f"{name}: native Prefill coverage gap")
            cursor += query
        require(cursor == case["input_len"], f"{name}: native and HTTP input lengths differ")

        per_rank, bytes_by_source = {}, dict.fromkeys(ranks, 0)
        for rank in range(8):
            loaded = request_events(decode[rank], "pd_cache_loaded", request_id)
            transfers = request_events(decode[rank], "pd_transfer_submitted", request_id)
            require(len(loaded) == 1 and loaded[0]["loaded_blocks"] > 0
                    and loaded[0]["dp_rank"] == 0 and loaded[0]["attn_tp_rank"] == rank,
                    f"{name}: rank {rank} did not load the remote cache")
            require({event["source_tp_rank"] for event in transfers} == ranks,
                    f"{name}: rank {rank} has an incomplete source plan")
            for event in transfers:
                require(event["source_tp"] == event["destination_tp"] == 8
                        and event["dp_rank"] == 0 and event["attn_tp_rank"] == rank,
                        f"{name}: rank {rank} has incorrect PD routing dimensions")
                bytes_by_source[event["source_tp_rank"]] += event["transfer_bytes"]
            per_rank[str(rank)] = {"loaded_blocks": loaded[0]["loaded_blocks"]}
        require(all(value > 0 for value in bytes_by_source.values()), f"{name}: missing source KV payload")
        result[name] = dict(request_id=request_id, prefill_steps=steps,
                            decode_ranks=per_rank, submitted_bytes_by_source_rank=bytes_by_source)
    return result


def audit_stage(config, output):
    summary = read_json(output / "launch" / "summary.json")
    require(
        summary.get("request_runner_passed") is True,
        "request runner did not complete successfully",
    )
    evidence = output / "prefill" / REQUEST_EVIDENCE
    prefill_logs, decode_logs = (output / role / "logs" for role in launcher.ROLES)
    if config["k31_smoke_stage"] == "regular":
        audit_path = output / "audit.json"
        with (output / "audit.log").open("xb") as log:
            audit = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rtp_llm.test.smoke.kimi_k3.audit_k31_full_smoke",
                    "--result",
                    str(evidence / "RESULT.json"),
                    "--output",
                    str(audit_path),
                    "--prefill-logs",
                    str(prefill_logs),
                    "--prefill-engine",
                    str(prefill_logs / "engine.log"),
                    "--decode-logs",
                    str(decode_logs),
                    "--decode-engine",
                    str(decode_logs / "engine.log"),
                    "--decode-tp",
                    "4",
                ],
                cwd=Path(__file__).resolve().parents[4],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        proof = read_json(audit_path)
        require(
            audit.returncode == 0
            and proof.get("complete_smoke_passed") is True
            and proof.get("native_K31_MTP") is True,
            f"regular smoke audit failed: {proof.get('errors')}",
        )
        return dict(total=304, **proof["counts"])

    if __package__:
        from .k31_long_prefix_smoke import verify
    else:
        from k31_long_prefix_smoke import verify

    proof = verify(evidence, output / "prefill" / "runtime-geometry.json")
    require(
        set(proof["cases"]) == {"seed", "continuation"},
        "long smoke must contain exactly two requests",
    )
    proof["pd_native_route"] = audit_long_native_route(proof, prefill_logs, decode_logs)
    launcher.atomic(output / "audit.json", proof)
    return dict(total=2, formal=1, preparation=1)


def run_stage(config, output, startup_timeout):
    output.mkdir(exist_ok=False)
    stage = config["k31_smoke_stage"]
    print(f"{stage}: starting {config['run_id']}", flush=True)
    record = {"passed": False, "run_id": config["run_id"], "errors": []}
    args = argparse.Namespace(
        dry_run=False,
        operation="launch",
        output=output / "launch",
        startup_timeout=startup_timeout,
        keep_services=False,
    )
    try:
        with (output / "launch.log").open("x") as log, redirect_stdout(
            log
        ), redirect_stderr(log):
            require(
                launcher.controller(args, config) == 0,
                "launcher did not return success",
            )
    except Exception as error:
        record["errors"].append(f"launch: {type(error).__name__}: {error}")
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            with (output / "launch.log").open("a") as log:
                log.write(error.stderr)
    finally:
        clean, receipts = wait_for_cleanup(config)
        record["cleanup_complete"] = clean
        launcher.atomic(output / "cleanup.json", receipts)
        if not clean:
            record["errors"].append("cleanup: role exit could not be confirmed")
        for role in launcher.ROLES:
            try:
                transfer_evidence(config, role, output)
            except Exception as error:
                record["errors"].append(
                    f"{role} evidence: {type(error).__name__}: {error}"
                )
        try:
            startup = audit_startup(config, output)
            launcher.atomic(output / "startup-proof.json", startup)
            record["checkpoints"] = startup["prefill"]["checkpoints"]
            record["request_counts"] = audit_stage(config, output)
        except Exception as error:
            record["errors"].append(f"audit: {type(error).__name__}: {error}")
        record["passed"] = not record["errors"]
        launcher.atomic(output / "RESULT.json", record)
    print(
        f"{stage}: {'PASS' if record['passed'] else 'FAIL'}; evidence {output}",
        flush=True,
    )
    return record


def run(base, output, startup_timeout):
    configs = stage_configs(base)
    output.mkdir(parents=True, exist_ok=False)
    launcher.atomic(output / "config.json", base)
    result = {"passed": False, "stages": {}, "expected_counts": EXPECTED_COUNTS}
    try:
        for stage, config in zip(STAGES, configs):
            record = run_stage(config, output / stage, startup_timeout)
            result["stages"][stage] = record
            if not record["cleanup_complete"]:
                break
        counts = {
            key: sum(
                record.get("request_counts", {}).get(key, 0)
                for record in result["stages"].values()
            )
            for key in EXPECTED_COUNTS
        }
        result["request_counts"] = counts
        checkpoints = [
            record.get("checkpoints") for record in result["stages"].values()
        ]
        if len(checkpoints) == 2 and checkpoints[0] != checkpoints[1]:
            result["error"] = (
                "the two stages did not use the same target and native MTP checkpoints"
            )
        result["passed"] = (
            len(result["stages"]) == len(STAGES)
            and counts == EXPECTED_COUNTS
            and "error" not in result
            and all(record["passed"] for record in result["stages"].values())
        )
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        launcher.atomic(output / "RESULT.json", result)
    print(
        f"Default K3.1 smoke {'PASS' if result['passed'] else 'FAIL'}; evidence {output}",
        flush=True,
    )
    return 0 if result["passed"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    entry = parser.add_mutually_exclusive_group(required=True)
    entry.add_argument("--config", type=Path)
    entry.add_argument("--request-control", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--startup-timeout", type=int, default=14400)
    args = parser.parse_args(argv)
    if args.request_control:
        return run_requests(args.request_control)
    if args.output is None or args.startup_timeout < 1:
        parser.error("--config requires --output and a positive --startup-timeout")
    return run(read_json(args.config), args.output.resolve(), args.startup_timeout)


if __name__ == "__main__":
    raise SystemExit(main())

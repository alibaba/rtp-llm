#!/usr/bin/env python3
"""Run the default K3.1 smoke and audit both stages after collecting evidence."""

from __future__ import annotations

import argparse
import copy
import json
import re
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
        long_prefill_linear_minimum,
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
        long_prefill_linear_minimum,
        fastsafetensors_proof,
    )

SCRIPT = Path(launcher.SCRIPT).with_name("k31_two_host_smoke.py")
STAGES = ("regular", "long")
COUNT_KEYS = ("total", "formal", "preparation")
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
    "port-preflight.json",
    "logs",
    "logits",
    REQUEST_EVIDENCE,
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stage_config(base, stage):
    """Common model topology for the traditional and Bazel stage runners."""
    require(
        base.get("target_only", False) is False, "default smoke requires native MTP"
    )
    last_n = base.get("last_n_checkpoints")
    page_tokens = base.get("last_n_page_tokens", 128)
    require(page_tokens in (64, 128) and ("last_n_page_tokens" not in base or last_n is not None),
            "Last-N page size requires an explicit Last-N smoke and 64 or 128 tokens")
    calibration = base.get('last_n_calibration', False)
    tier_probe = base.get('last_n_tier_probe')
    require(tier_probe is None or (tier_probe in ('DEVICE', 'HOST', 'DISK')
            and last_n == 5 and not calibration and page_tokens == 128
            and base['prefill'].get('prefill_token_capacity') == 2048),
            'isolated tier probe requires Last-5, 2K chunks and no calibration mode')
    require(type(calibration) is bool and (not calibration or last_n == 5),
            'unbounded calibration requires the explicit Last-5 geometry')
    require(last_n is None or type(last_n) is int and last_n == 5,
            "full Last-N smoke uses Last-5; smaller N is covered by focused tests")
    if last_n is not None:
        require(stage == "regular",
                "Last-5 uses the existing 304 regular slots; select --stages regular")
        require(base.get("decode_dp", 1) == 1, "Last-5 smoke requires Decode TP8/EP8")
    require(stage in STAGES, "unsupported K3.1 smoke stage")
    decode_dp, block_count = {
        "regular": (2, 96),
        "long": (1, 140),
    }[stage]
    if stage == "regular":
        decode_dp = 1 if last_n is not None else base.get("decode_dp", decode_dp)
    if last_n is not None:
        long_pool_blocks = LONG_INPUT_TOKENS // (8 * page_tokens) + 8
        block_count = max(256 if calibration else long_pool_blocks, base["prefill"].get("test_block_num", 0))
    config = copy.deepcopy(base)
    config.setdefault("cache_store_rdma_mode", True)
    config.setdefault("fp8_gemm", True)
    config.setdefault("fp8_kv_cache", True)
    config.setdefault(
        "prefill_kv_workspace_budget_gib", launcher.DEFAULT_KV_WORKSPACE_GIB
    )
    config.update(
        run_id=f"{base['run_id']}-{stage}",
        k31_smoke_stage=stage,
        decode_dp=decode_dp,
        target_only=False,
    )
    config["prefill"]["test_block_num"] = block_count
    if last_n is not None:
        config["decode"]["test_block_num"] = max(64 if calibration else long_pool_blocks, base["decode"].get("test_block_num", 0))
        # FULL covers 1M; LINEAR needs independent pressure for HOST/SSD hits.
        config["prefill"].setdefault("linear_pool_blocks", 256 if calibration else 96)
        config["decode"].setdefault("linear_pool_blocks", 64 if calibration else 512)
        config["prefill"].setdefault("memory_cache_size_mb", 8192)
        config["prefill"].setdefault("disk_cache_size_mb", 65536)
        for role in launcher.ROLES:
            require(config[role].get("max_seq_len", 2_097_152) >=
                    LONG_INPUT_TOKENS + MAX_OUTPUT_TOKENS + MTP_RESERVE_TOKENS,
                    f"{role}: Last-5 smoke must admit the existing 1M cold seed")
    if stage == "long":
        config["prefill"].setdefault(
            "linear_pool_blocks", long_prefill_linear_minimum() + 1
        )
        config["prefill"].setdefault("request_capacity", 1)
        # Keep the traditional long-stage CRC and MoE staging budgets.
        config["prefill"].setdefault("block_tree_transfer_worker_count", 1)
        config["prefill"].setdefault("prefill_token_capacity", 8192)
        config["decode"].update(
            test_block_num=block_count,
            request_capacity=1,
            decode_capture_batches="1",
        )
        for role in launcher.ROLES:
            limit = config[role].setdefault("max_seq_len", LONG_INPUT_TOKENS + 4096)
            require(
                limit >= LONG_INPUT_TOKENS + MAX_OUTPUT_TOKENS + MTP_RESERVE_TOKENS,
                f"{role} max_seq_len must admit the exact 1M continuation and generation reserve",
            )
    return config


def stage_configs(base, stages=STAGES):
    require(stages and len(set(stages)) == len(stages) and set(stages) <= set(STAGES),
            "select regular and/or long exactly once")
    stages = tuple(stage for stage in STAGES if stage in stages)
    if base.get("last_n_checkpoints") is not None:
        require(tuple(stages) == ("regular",),
                "Last-5 uses the existing 304 regular slots; select --stages regular")
    configs = ([stage_config(base, stage) for stage in stages]
               if base.get("last_n_checkpoints") is not None else [smoke_config(base)])
    for config in configs:
        if base.get("last_n_checkpoints") is None:
            config.update(k31_smoke_stage=stages[0], request_stages=list(stages))
        control = launcher.run_dir(config, "prefill") / "control.json"
        config["request_argv"] = [
            "python3",
            "-u",
            str(Path(config["prefill"]["repo_root"]) / SCRIPT),
            "--request-control",
            str(control),
        ]
        launcher.validate(config)
    return configs


def smoke_config(base):
    """One P8/D8 lifecycle for the regular and 1M request programs."""
    config = stage_config(base, "regular")
    config.pop("k31_smoke_stage")
    config.update(run_id=base["run_id"], smoke_profile="k31")
    require(config["decode_dp"] == 2, "smoke requires Decode DP2/TP4")
    prefill, decode = config["prefill"], config["decode"]
    prefill["test_block_num"] = 140
    prefill.setdefault("prefill_token_capacity", 65536)
    require(prefill["prefill_token_capacity"] == 65536, "smoke retains the regular 64K Prefill forward budget")
    prefill.setdefault("reuse_cache_tail_blocks", 5)
    require(prefill["reuse_cache_tail_blocks"] == 5, "smoke uses Prefill Last-5 retention")
    prefill.setdefault("linear_pool_blocks", 128)
    decode["test_block_num"] = max(272, decode.get("test_block_num", 0))
    decode.setdefault("linear_pool_blocks", 192)
    for role in launcher.ROLES:
        limit = config[role].setdefault("max_seq_len", LONG_INPUT_TOKENS + 4096)
        require(limit >= LONG_INPUT_TOKENS + MAX_OUTPUT_TOKENS + MTP_RESERVE_TOKENS,
                f"{role}: smoke must admit the exact 1M continuation")
    return config


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
        config.get("case_namespace", config["run_id"]),
        "--geometry",
        str(root / "runtime-geometry.json"),
    ]
    if config.get("last_n_checkpoints") is not None:
        args += ["--last-n-checkpoints", str(config["last_n_checkpoints"])]
    for item in endpoints:
        if item["kind"] == "http":
            args += [
                "--decode-role-addr",
                f"{decode['ip']}:{item['port']}:{rpc_ports[item['rank']]}",
            ]
            if config["k31_smoke_stage"] == "long":
                break
    return args


def run_requests(control, stage=None):
    config = launcher.validate(read_json(control))
    root = launcher.run_dir(config, "prefill")
    require(
        control.resolve() == (root / "control.json").resolve(),
        "unexpected Prefill control path",
    )
    evidence = root / REQUEST_EVIDENCE
    if config.get("request_stages"):
        if stage is None:
            from rtp_llm.test.smoke.kimi_k3.audit_k31_full_smoke import read_events

            for phase in config["request_stages"]:
                code = run_requests(control, phase)
                events = read_events(root / "logs", root / "logs/engine.log")
                launcher.atomic(evidence / phase / "event-end.json", {
                    str(rank): dict(count=len(rows), last=rows[-1] if rows else None)
                    for rank, rows in events.items()
                })
                if code:
                    return code
            return 0
        require(stage in config["request_stages"], "unexpected smoke request phase")
        config = {**config, "k31_smoke_stage": stage}
        evidence /= stage
    args = request_args(config, root)
    if config["k31_smoke_stage"] == "long":
        if __package__:
            from .k31_long_prefix_smoke import main
        else:
            from k31_long_prefix_smoke import main

        return main(["run", *args, "--output", str(evidence)])
    require(config["k31_smoke_stage"] == "regular", "unsupported K3.1 smoke stage")
    if config.get('last_n_calibration'):
        if __package__:
            from .k31_last_n_calibration import main
        else:
            from k31_last_n_calibration import main
    elif config.get('last_n_tier_probe'):
        if __package__:
            from .k31_last_n_tier_probe import main
        else:
            from k31_last_n_tier_probe import main
        args += ['--isolated-tier', config['last_n_tier_probe']]
    else:
        if __package__:
            from .k31_full_smoke import main
        else:
            from k31_full_smoke import main

    prefill, decode = config["prefill"], config["decode"]
    source = Path(prefill["repo_root"])
    binary = (source / prefill.get("server", "bazel-bin/rtp_llm/rtp_llm_server")).resolve(strict=True)
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


def audit_crc_workspace(directory, workers, expected_batch=None):
    engine_log = (directory / "logs/engine.log").read_text(errors="replace")
    crc = re.findall(
        r"\[RANK (\d+)\][^\n]*BlockTree CRC32C enabled: "
        r"device=\d+ batch=(\d+) slots=(\d+) device_total=([\d.]+) MiB",
        engine_log,
    )
    require(
        len(crc) == 8 and {int(rank) for rank, *_ in crc} == set(range(8))
        and all(int(slots) == workers * 2 for _, _, slots, _ in crc),
        f"{directory.name}: expected eight enabled CRC receipts with {workers * 2} slots",
    )
    require(
        expected_batch is None or all(int(batch) == expected_batch for _, batch, _, _ in crc),
        f"{directory.name}: CRC transfer batch differs from the configured limit",
    )
    return {
        rank: dict(batch=int(batch), slots=int(slots), device_total_mib=float(mib))
        for rank, batch, slots, mib in crc
    }


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
        transport = read_json(directory / "rdma-preflight.json")
        require(transport.get("cache_store_transport") == "rdma",
                f"{role}: smoke requires RDMA preflight")
        logs = directory / "logs"
        with (logs / "engine.log").open(encoding="utf-8", errors="replace") as engine:
            rdma_ranks = sorted({
                int(match.group(1))
                for line in engine
                if (match := re.search(r"\[RANK (\d+)\].*rdma messager init success", line))
            })
        require(rdma_ranks == list(range(8)), f"{role}: expected native RDMA on all eight ranks")
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
        budget = float(config.get("prefill_kv_workspace_budget_gib", launcher.DEFAULT_KV_WORKSPACE_GIB))
        require(
            set(receipts) == {str(rank) for rank in range(8)}
            and all(float(row["gib"]) == budget for row in receipts.values()),
            f"{role}: expected all eight {budget:g} GiB workspace receipts",
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
            fastsafetensors_lines=fastsafetensors_proof(logs),
            rdma_ranks=rdma_ranks,
            geometry=geometry,
            checkpoints=checkpoints,
        )
        workers = config[role].get("block_tree_transfer_worker_count")
        if workers is not None:
            proof[role]["crc_workspace"] = audit_crc_workspace(
                directory, workers, config[role].get("memory_cache_max_descriptors_per_transfer_batch")
            )
    require(
        proof["prefill"]["checkpoints"] == proof["decode"]["checkpoints"],
        "PD checkpoints differ",
    )
    return proof


def audit_long_execution(proof, prefill, decode, geometry, *, decode_tp, max_chunk_tokens):
    if __package__:
        from .audit_k31_last_n import audit_long_retention, audit_prefill_retention
    else:
        from audit_k31_last_n import audit_long_retention, audit_prefill_retention

    proof["pd_native_route"] = audit_long_events(proof, prefill, decode, decode_tp=decode_tp)
    proof["prefill_retention"] = audit_prefill_retention(
        prefill, geometry, max_chunk_tokens=max_chunk_tokens,
    )
    proof["retention_requests"] = audit_long_retention(proof, proof["prefill_retention"])
    return proof


def audit_long_events(proof, prefill, decode, *, decode_tp=8, decode_owner=0):
    """Same native route proof for files or authenticated CI event snapshots."""
    if __package__:
        from .audit_k31_full_smoke import group_evidence, request_events
    else:
        from audit_k31_full_smoke import group_evidence, request_events

    require(
        decode_tp in (4, 8) and 0 <= decode_owner < 8 // decode_tp,
        "invalid long smoke Decode owner geometry",
    )
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
        selected_ranks = range(decode_owner * decode_tp, (decode_owner + 1) * decode_tp)
        for rank in range(8):
            if rank not in selected_ranks:
                require(
                    not request_events(decode[rank], "pd_cache_loaded", request_id),
                    f"{name}: unexpected cache load on Decode owner rank {rank}",
                )
                continue
            loaded = request_events(decode[rank], "pd_cache_loaded", request_id)
            transfers = request_events(decode[rank], "pd_transfer_submitted", request_id)
            require(
                len(loaded) == 1
                and loaded[0]["loaded_blocks"] > 0
                and loaded[0]["dp_rank"] == decode_owner
                and loaded[0]["attn_tp_rank"] == rank % decode_tp,
                f"{name}: rank {rank} did not load the remote cache",
            )
            require({event["source_tp_rank"] for event in transfers} == ranks,
                    f"{name}: rank {rank} has an incomplete source plan")
            for event in transfers:
                require(
                    event["source_tp"] == 8
                    and event["destination_tp"] == decode_tp
                    and event["dp_rank"] == decode_owner
                    and event["attn_tp_rank"] == rank % decode_tp,
                    f"{name}: rank {rank} has incorrect PD routing dimensions",
                )
                bytes_by_source[event["source_tp_rank"]] += event["transfer_bytes"]
            per_rank[str(rank)] = {"loaded_blocks": loaded[0]["loaded_blocks"]}
        require(all(value > 0 for value in bytes_by_source.values()), f"{name}: missing source KV payload")
        result[name] = dict(request_id=request_id, prefill_steps=steps,
                            decode_ranks=per_rank, submitted_bytes_by_source_rank=bytes_by_source)
    return result


def audit_stage(config, output):
    summary = read_json(output / "launch" / "summary.json")
    require(
        summary.get("request_runner_passed") is True or bool(config.get("request_stages")),
        "request runner did not complete successfully",
    )
    evidence = output / "prefill" / REQUEST_EVIDENCE
    phase = config["k31_smoke_stage"]
    start = end = None
    if config.get("request_stages"):
        if phase == "long" and "regular" in config["request_stages"]:
            start = read_json(evidence / "regular/event-end.json")
        evidence /= phase
        end = read_json(evidence / "event-end.json")
    audit_path = output / (f"{phase}-audit.json" if end is not None else "audit.json")
    prefill_logs, decode_logs = (output / role / "logs" for role in launcher.ROLES)
    if config.get('last_n_tier_probe'):
        if __package__:
            from .k31_last_n_tier_probe import audit_probe
            from .audit_k31_full_smoke import read_events
            from . import result_check
        else:
            from k31_last_n_tier_probe import audit_probe
            from audit_k31_full_smoke import read_events
            import result_check
        result = read_json(evidence/'RESULT.json')
        rows = {row['name']: row for row in result['cases']}
        errors = (result_check.audit_raw_answers(evidence/'requests', rows)
                  + result_check._raw_consistency(evidence/'requests', rows))
        for row in rows.values():
            errors += result_check.independent_answer_check(row) + result_check._cache_errors(row, 128)
        require(not errors, f'isolated tier raw response audit failed: {errors}')
        proof = audit_probe(result, read_events(prefill_logs, prefill_logs/'engine.log'),
                            read_events(decode_logs, decode_logs/'engine.log'), output, config)
        launcher.atomic(output/'audit.json', proof)
        return dict(total=2, formal=1, preparation=1)
    if config.get('last_n_calibration'):
        if __package__:
            from .k31_last_n_calibration import audit_calibration
            from .audit_k31_full_smoke import read_events
            from . import result_check
        else:
            from k31_last_n_calibration import audit_calibration
            from audit_k31_full_smoke import read_events
            import result_check
        result = read_json(evidence/'RESULT.json')
        rows = {row['name']: row for row in result['cases']}
        raw_errors = (result_check.audit_raw_answers(evidence/'requests', rows)
                      + result_check._raw_consistency(evidence/'requests', rows))
        require(not raw_errors, f'calibration raw response audit failed: {raw_errors}')
        proof = audit_calibration(result,
            read_events(prefill_logs, prefill_logs/'engine.log'),
            read_events(decode_logs, decode_logs/'engine.log'), output,
            prefill_chunk_tokens=config['prefill'].get('prefill_token_capacity', 65536))
        launcher.atomic(output/'audit.json', proof)
        return dict(total=6, formal=2, preparation=4)
    if config["k31_smoke_stage"] == "regular":
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
                    str(8 // launcher.decode_dp_size(config)),
                ] + (["--prefill-event-end", str(evidence / "event-end.json")]
                     if end is not None else []) + precision_audit_args(config, output),
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
        return proof["request_budget"]

    if __package__:
        from .k31_long_prefix_smoke import verify
        from .audit_k31_full_smoke import read_events
    else:
        from k31_long_prefix_smoke import verify
        from audit_k31_full_smoke import read_events

    geometry = output / "prefill" / "runtime-geometry.json"
    proof = verify(evidence, geometry)
    require(
        set(proof["cases"]) == {"seed", "continuation"},
        "long smoke must contain exactly two requests",
    )
    audit_long_execution(
        proof,
        read_events(prefill_logs, prefill_logs / "engine.log", start=start, end=end),
        read_events(decode_logs, decode_logs / "engine.log"),
        read_json(geometry),
        decode_tp=8 // launcher.decode_dp_size(config),
        max_chunk_tokens=config["prefill"]["prefill_token_capacity"],
    )
    launcher.atomic(audit_path, proof)
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
            phases = config.get("request_stages", (stage,))
            record["stages"] = {}
            for phase in phases:
                try:
                    counts = audit_stage({**config, "k31_smoke_stage": phase}, output)
                    record["stages"][phase] = dict(passed=True, request_counts=counts)
                except Exception as error:
                    record["stages"][phase] = dict(passed=False, errors=[f"audit: {type(error).__name__}: {error}"])
                    record["errors"].extend(record["stages"][phase]["errors"])
            if "request_stages" not in config and record["stages"][stage]["passed"]:
                record["request_counts"] = record["stages"][stage]["request_counts"]
            if __package__:
                from .nvfp4_evidence import audit_projection_precision
            else:
                from nvfp4_evidence import audit_projection_precision
            record["projection_precision"] = {
                role: audit_projection_precision(output / role, config) for role in launcher.ROLES
            }
        except Exception as error:
            record["errors"].append(f"audit: {type(error).__name__}: {error}")
        record["passed"] = not record["errors"]
        launcher.atomic(output / "RESULT.json", record)
    print(
        f"{stage}: {'PASS' if record['passed'] else 'FAIL'}; evidence {output}",
        flush=True,
    )
    return record


def precision_audit_args(config, output):
    if config.get('last_n_checkpoints') is None:
        return []
    for key in ('precision_reference', 'precision_contract'):
        require(isinstance(config.get(key), str) and Path(config[key]).is_absolute()
                and Path(config[key]).is_file(), f'Last-5 needs an existing absolute {key} before launch')
    return ['--prefill-logits', str(output/'prefill'/'logits'),
            '--decode-logits', str(output/'decode'/'logits'),
            '--precision-reference', config['precision_reference'],
            '--precision-contract', config['precision_contract']]


def run(base, output, startup_timeout, stages=STAGES):
    configs = stage_configs(base, stages)
    if base.get('last_n_checkpoints') is not None and not base.get('last_n_calibration'):
        precision_audit_args(base, output)
        contract = read_json(Path(base['precision_contract']))
        require(contract.get('schema') == 1 and type(contract.get('created_ns')) is int
                and 0 < contract['created_ns'] < time.time_ns()
                and contract.get('source_commit') == base.get('source_commit'),
                'Last-5 precision limits must be locked for this source SHA before starting services')
        require(contract.get('prefill_chunk_tokens', 65536) ==
                base['prefill'].get('prefill_token_capacity', 65536),
                'Last-5 calibration and candidate must use the same Prefill chunk capacity')
    output.mkdir(parents=True, exist_ok=False)
    launcher.atomic(output / "config.json", base)
    result = {"passed": False, "stages": {},
              "server_lifecycles": len(configs)}
    try:
        for config in configs:
            stage = config["k31_smoke_stage"]
            record = run_stage(config, output / stage, startup_timeout)
            if "request_stages" in config:
                for phase in config["request_stages"]:
                    result["stages"][phase] = {
                        **record, **record.get("stages", {}).get(phase, dict(passed=False))}
                    result["stages"][phase]["passed"] &= record["passed"]
            else:
                result["stages"][stage] = record
            if not record["cleanup_complete"]:
                break
        # Each phase audit checks its runtime-derived budget, including all
        # cache-pressure probes; a larger shared pool can require more probes.
        counts = {
            key: sum(
                record.get("request_counts", {}).get(key, 0)
                for record in result["stages"].values()
            )
            for key in COUNT_KEYS
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
            len(result["stages"]) == len(stages)
            and "error" not in result
            and all(record["passed"] for record in result["stages"].values())
        )
        if base.get('last_n_calibration'):
            result.update(last_n_calibration=True, complete_smoke_passed=False)
        if base.get('last_n_tier_probe'):
            result.update(last_n_tier_probe=base['last_n_tier_probe'], complete_smoke_passed=False)
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        launcher.atomic(output / "RESULT.json", result)
    label = 'Last-N calibration' if base.get('last_n_calibration') else 'Default K3.1 smoke'
    print(
        f"{label} {'PASS' if result['passed'] else 'FAIL'}; evidence {output}",
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
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--kda-nvfp4", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dsa-nvfp4", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    if args.request_control:
        return run_requests(args.request_control)
    if args.output is None or args.startup_timeout < 1:
        parser.error("--config requires --output and a positive --startup-timeout")
    config = read_json(args.config)
    for key in ("kda_nvfp4", "dsa_nvfp4"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    return run(config, args.output.resolve(), args.startup_timeout, tuple(dict.fromkeys(args.stages)))


if __name__ == "__main__":
    raise SystemExit(main())

"""Launch real NormalCacheStore ranks without loading a model."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def collect_ranks(root, ranks, processes, source_tp, target_tp, target_dp):
    reports = []
    for index, (role, rank, _) in enumerate(ranks):
        role_size = source_tp if role == "sender" else target_tp * target_dp
        stem = role if role_size == 1 else f"{role}.{rank}"
        result = root / f"{stem}.result.json"
        log = root / f"{role}.{rank}.log"
        proc = processes[index] if index < len(processes) else None
        report = {}
        errors = []
        try:
            report = json.loads(result.read_text())
            if not isinstance(report, dict):
                raise ValueError("native result must be an object")
        except (OSError, ValueError) as error:
            report = {}
            errors.append(f"{type(error).__name__}: {error}")
        if not log.is_file():
            errors.append("native log missing")
        if proc is None or report.get("pid") != proc.pid:
            errors.append("native PID missing or differs from owned process")
        report.update(
            role=role,
            owned_pid=proc.pid if proc else None,
            exit_code=proc.returncode if proc else None,
            started=proc is not None,
            raw_result=str(result.relative_to(root.parent)),
            raw_log=str(log.relative_to(root.parent)),
            collection_errors=errors,
        )
        if errors:
            report["passed"] = False
        reports.append(report)
    return reports


def rank_summary(report):
    # Keep bulk per-byte/unit evidence only in the referenced native JSON.
    summary = {
        key: value
        for key, value in report.items()
        if key not in ("payload_regions", "logical_units", "request_runs")
    }
    if "request_runs" in report:
        summary["request_runs"] = [
            rank_summary(part) for part in report["request_runs"]
        ]
    return summary


def run_single(binary, path, output):
    # Control files and incidental native output belong to the test sandbox,
    # not the CI artifact archive. Logs are streamed directly to declared output.
    with TemporaryDirectory(
        prefix="cache-store-", dir=os.environ["TEST_TMPDIR"]
    ) as temporary:
        return run_native(binary.resolve(), path, output.resolve(), Path(temporary))


def run_native(binary, path, output, root):
    config = json.loads(path.read_text())
    registered_types = {
        "mha": ("bf16", "int8"),
        "mla": ("bf16", "fp8"),
        "next": ("bf16",),
        "dsv4": ("fp8",),
    }
    if config.get("dtype") not in registered_types.get(config.get("layout"), ()):
        raise ValueError("unregistered Cache smoke layout/data type")
    source_tp, target_tp = config["source_tp"], config["target_tp"]
    source_cp = config.get("source_cp", 1)
    target_dp = config.get("target_dp", 1)
    if (source_tp, target_tp) not in ((1, 1), (1, 2), (2, 1), (2, 2)):
        raise ValueError("unconfirmed NormalCacheStore topology")
    if source_cp == 2:
        if (
            source_tp != 2
            or target_tp != 1
            or (
                (config["layout"] == "mla" and target_dp != 1)
                or (config["layout"] == "dsv4" and target_dp != 2)
                or config["layout"] not in ("mla", "dsv4")
            )
        ):
            raise ValueError(
                "only traced CP2-to-TP1/DP2 service topology is registered"
            )
    elif source_tp != target_tp and config["layout"] != "mha":
        raise ValueError("asymmetric TP requires pure MHA")
    if config["layout"] == "next" and (source_tp, target_tp) not in ((1, 1), (2, 2)):
        raise ValueError("Next FULL+Linear only permits symmetric TP1/TP2")
    if (
        config["layout"] == "dsv4"
        and source_cp == 1
        and ((source_tp, target_tp) not in ((1, 1), (2, 2)) or target_dp != 1)
    ):
        raise ValueError("DSV4 only permits symmetric TP1/TP2 or CP2-to-DP2")
    if config["tokens"] % config["block_tokens"]:
        raise ValueError("tokens must align to full cache blocks")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) < source_tp + target_tp * target_dp:
        raise ValueError("not enough isolated GPU ranks")
    ranks = [("sender", rank, source_tp) for rank in range(source_tp)] + [
        ("receiver", rank, target_tp) for rank in range(target_tp * target_dp)
    ]
    output.mkdir(parents=True, exist_ok=True)
    # Native logs/results survive even if supervision or JSON decoding fails.
    artifacts = output / "native"
    artifacts.mkdir()
    reports = []
    lifecycle = config.get("request_lifecycle", False)
    if lifecycle and not (
        config["layout"] == "mha"
        and source_tp == target_tp
        and source_cp == 1
        and target_dp == 1
        and config["tokens"] == 1024
    ):
        raise ValueError("request lifecycle requires symmetric MHA at 1K")
    processes = []
    logs = []
    primary_error = None
    cleanup_errors = []
    try:
        for index, (role, rank, size) in enumerate(ranks):
            log_path = artifacts / f"{role}.{rank}.log"
            log = log_path.open("w")
            logs.append(log)
            command = [
                str(binary),
                role,
                str(root),
                "0",
                "auto",
                *[
                    str(config[key])
                    for key in (
                        "seed",
                        "layers",
                        "heads",
                        "kv_heads",
                        "head_dim",
                        "block_tokens",
                        "tokens",
                        "pool_blocks",
                        "deadline_ms",
                    )
                ],
                config["dtype"],
                str(config.get("fault_mode", 0)),
                str(size),
                str(rank if role == "sender" or target_dp == 1 else 0),
                str(target_tp if role == "sender" else source_tp),
                config["layout"],
                str(config.get("latent_dim", 0)),
                str(config.get("rope_dim", 0)),
            ]
            if source_cp == 2 and config["layout"] != "dsv4":
                command.append("2")
            if config["layout"] == "next":
                command.extend(
                    str(config[key])
                    for key in (
                        "linear_key_heads",
                        "linear_value_heads",
                        "linear_head_dim",
                        "conv_kernel",
                    )
                )
                command.extend(
                    (
                        ",".join(config["layer_tags"]),
                        str(config["linear_pool_blocks"]),
                    )
                )
            if config["layout"] == "dsv4":
                command.extend(
                    (
                        str(config["indexer_head_dim"]),
                        str(config["sliding_window"]),
                        config["descriptor_projection"],
                        str(int(config["fixed_pool_use_host_memory"])),
                        str(config["fixed_pool_blocks"]),
                        str(config["hca_state_pool_blocks"]),
                        ",".join(map(str, config["layer_compress_ratios"])),
                    )
                )
                if source_cp == 2:
                    command.extend(
                        (
                            "2",
                            str(target_dp),
                            str(rank if role == "receiver" else 0),
                        )
                    )
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=devices[index],
                CACHE_SMOKE_REQUEST_LIFECYCLE=str(int(lifecycle)),
            )
            processes.append(
                subprocess.Popen(
                    command, env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT
                )
            )
        limit = time.monotonic() + config["process_timeout_seconds"]
        while any(proc.poll() is None for proc in processes):
            if time.monotonic() > limit:
                raise TimeoutError("native rank deadline")
            time.sleep(0.05)
    except BaseException as error:
        primary_error = error
    finally:
        for proc in processes:
            try:
                if proc.poll() is None:
                    proc.terminate()
            except OSError as error:
                cleanup_errors.append(f"terminate {proc.pid}: {error}")
        for proc in processes:
            try:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired) as error:
                cleanup_errors.append(f"reap {proc.pid}: {error}")
        for handle in logs:
            try:
                handle.close()
            except OSError as error:
                cleanup_errors.append(f"close: {error}")
        # Allowlist native result files, including partial/malformed diagnostics.
        # Never archive synchronization files, cache backing, or arbitrary cwd files.
        for result in root.rglob("*.result.json"):
            try:
                destination = artifacts / result.relative_to(root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(result, destination)
            except OSError as error:
                cleanup_errors.append(f"save native result {result.name}: {error}")
        reports = collect_ranks(
            artifacts, ranks, processes, source_tp, target_tp, target_dp
        )
        summaries = [rank_summary(report) for report in reports]
        try:
            write_json(output / "native-ranks.json", summaries)
            write_json(
                output / "supervisor.json",
                {
                    "primary_error": (
                        f"{type(primary_error).__name__}: {primary_error}"
                        if primary_error
                        else None
                    ),
                    "cleanup_errors": cleanup_errors,
                    "all_started_processes_reaped": all(
                        proc.returncode is not None for proc in processes
                    ),
                },
            )
        except OSError as error:
            if primary_error is None:
                raise
            print(
                f"Could not save summary; raw files remain at {artifacts}: {error}",
                file=sys.stderr,
            )
    if primary_error is not None:
        raise primary_error
    if cleanup_errors:
        raise RuntimeError(f"rank cleanup failed: {cleanup_errors}")
    listener_ports = [report.get("listen_port") for report in reports]
    if not all(
        type(port) is int and 0 < port <= 65535 for port in listener_ports
    ) or len(set(listener_ports)) != len(ranks):
        raise AssertionError("missing or overlapping production listener ports")
    if not all(report.get("port_selection") == "kernel" for report in reports):
        raise AssertionError("production listeners must use kernel-assigned ports")
    comparisons = validate_reports(reports, config)
    request_checks = []
    if lifecycle:
        if not all(
            report.get("lifecycle") is True
            and report.get("all_requests_passed") is True
            and report.get("store_initializations") == 1
            and len(report.get("request_runs", [])) == 2
            for report in reports
        ):
            raise AssertionError("missing same-instance request lifecycle evidence")
        for index in range(2):
            request_reports = []
            for report in reports:
                part = report["request_runs"][index]
                if not (
                    part["pid"] == report["owned_pid"]
                    and part["request_number"] == index + 1
                    and part["request_key"] == f"cache-smoke-{index + 1}"
                    and part["seed"] == config["seed"] + index
                    and part["request_end_checked"] is True
                    and part["free_before"] == part["free_after"]
                ):
                    raise AssertionError(
                        "lifecycle request identity or resource release differs"
                    )
                if report["role"] == "receiver" and not (
                    part["expired_rejected"] is True
                    and part["expired_bytes_unchanged"] is True
                    and part["expired_error"] == "CACHE_STORE_LOAD_BUFFER_TIMEOUT"
                ):
                    raise AssertionError(
                        "expired request behavior differs from production"
                    )
                request_reports.append(
                    {**part, "role": report["role"], "exit_code": report["exit_code"]}
                )
            effective = {
                **config,
                "seed": config["seed"] + index,
                "fault_mode": config.get("fault_mode", 0) if index == 0 else 0,
            }
            request_checks.append(
                {
                    "request_number": index + 1,
                    "seed": effective["seed"],
                    "comparisons": validate_reports(request_reports, effective),
                }
            )
    result = {
        "status": "PASS",
        "case_id": config["case_id"],
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "entry": "RemoteRpcServer::initCacheStore -> NormalCacheStore; DecodeRpcServer::loadCache -> CacheStore::loadBuffers",
        "ranks": summaries,
        "comparisons": comparisons,
        "request_checks": request_checks,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def validate_reports(reports, config):
    source_tp = config["source_tp"]
    target_dp = config.get("target_dp", 1)
    source_cp = config.get("source_cp", 1)
    unit_fields = (
        "layer",
        "group",
        "logical_block",
        "global_head",
        "component",
        "scale",
    )
    cp_dsv4 = config["layout"] == "dsv4" and source_cp == 2
    replicated_dsv4 = config["layout"] == "dsv4" and source_tp == 2 and not cp_dsv4
    by_role = {"sender": {}, "receiver": {}}
    receiver_replicas = {}
    for report in reports:
        for unit in report.get("logical_units", []):
            key = tuple(unit[field] for field in unit_fields)
            if replicated_dsv4:
                key = (report["tp_rank"],) + key
            role_units = (
                receiver_replicas.setdefault(report["dp_rank"], {})
                if cp_dsv4 and report["role"] == "receiver"
                else by_role[report["role"]]
            )
            if key in role_units:
                raise AssertionError(f"duplicate {report['role']} unit: {key}")
            role_units[key] = unit
    expected = {
        (layer, "default", block, head, component, scale)
        for layer in range(config["layers"])
        for block in range(config["tokens"] // config["block_tokens"])
        for head in range(config["kv_heads"] if config["layout"] == "mha" else 1)
        for component in (
            ("K", "V")
            if config["layout"] == "mha"
            else (
                ("latent", "inline_scale", "RoPE")
                if config["dtype"] == "fp8"
                else ("latent", "RoPE")
            )
        )
        for scale in (
            (False, True)
            if config["layout"] == "mha" and config["dtype"] == "int8"
            else ((True,) if component == "inline_scale" else (False,))
        )
    }
    target_maps = (
        [receiver_replicas.get(rank, {}) for rank in range(target_dp)]
        if cp_dsv4
        else [by_role["receiver"]]
    )
    if config["layout"] == "next":
        expected = set()
        for layer, tag in enumerate(config["layer_tags"]):
            components = (
                (("K", config["kv_heads"]), ("V", config["kv_heads"]))
                if tag == "full"
                else (
                    ("SSM", config["linear_value_heads"]),
                    ("conv_Q", config["linear_key_heads"]),
                    ("conv_K", config["linear_key_heads"]),
                    ("conv_V", config["linear_value_heads"]),
                )
            )
            pages = (
                range(config["tokens"] // config["block_tokens"])
                if tag == "full"
                else (config["tokens"] // config["block_tokens"] - 1,)
            )
            expected.update(
                (layer, tag, page, head, component, False)
                for page in pages
                for component, count in components
                for head in range(count)
            )
    if config["layout"] == "dsv4":
        expected = set()
        total_pages = config["tokens"] // config["block_tokens"]
        for layer, tags in enumerate(config["expected_layer_tags"]):
            for tag in tags:
                group = config["independent_group_layout"][tag]
                component = (
                    "FP32_state"
                    if tag.endswith("state")
                    else (
                        "FP8_indexer_packed" if tag == "indexer_kv" else "FP8_KV_packed"
                    )
                )
                tail = group["tail_blocks"]
                fixed = tag == "swa_kv" or tag.endswith("state")
                page_count = (
                    (total_pages + 1) // 2 if cp_dsv4 and fixed else total_pages
                )
                for ordinal in range(page_count - tail if tail else 0, page_count):
                    page = (
                        min(2 * ordinal + 1, total_pages - 1)
                        if cp_dsv4 and fixed
                        else ordinal
                    )
                    if cp_dsv4 and fixed:
                        for peer in range(2):
                            expected.add(
                                (layer, tag, page, 0, f"{component}.cp{peer}", False)
                            )
                            if (
                                peer == 1
                                and group["stride_bytes"] > group["payload_bytes"]
                            ):
                                expected.add(
                                    (layer, tag, page, 0, "stride_padding.cp1", False)
                                )
                    else:
                        expected.add((layer, tag, page, 0, component, False))
                        if group["stride_bytes"] > group["payload_bytes"]:
                            expected.add((layer, tag, page, 0, "stride_padding", False))
        if replicated_dsv4:
            expected = {
                (rank,) + unit for rank in range(source_tp) for unit in expected
            }
    if by_role["sender"].keys() != expected or any(
        units.keys() != expected for units in target_maps
    ):
        raise AssertionError("missing or extra source/receiver logical payload units")
    comparisons = []
    for rank, target_units in enumerate(target_maps):
        for key in sorted(expected):
            source, target = by_role["sender"][key], target_units[key]
            if not (
                source["matches"]
                and target["matches"]
                and source["expected"]
                == source["observed"]
                == target["expected"]
                == target["observed"]
            ):
                raise AssertionError(
                    f"source/receiver/independent oracle mismatch: {key}"
                )
            comparisons.append(
                {
                    "unit": key,
                    "receiver_rank": rank,
                    "expected": source["expected"],
                    "sender_before": source["observed"],
                    "receiver_after": target["observed"],
                }
            )
    if not all(report.get("passed") and report["exit_code"] == 0 for report in reports):
        raise AssertionError("native rank failed; see native-ranks.json")
    if config.get("fault_mode", 0):
        receivers = [report for report in reports if report["role"] == "receiver"]
        if not all(
            report["fault_rejected"]
            and report["fault_bytes_unchanged"]
            and report["fault_error"] == config["expected_fault_error"]
            for report in receivers
        ):
            raise AssertionError(
                "production fault behavior differs; see native-ranks.json"
            )
    return comparisons


def run_cases(binary, path, output):
    config = json.loads(path.read_text())
    sequence_cases = config.get("sequence_cases")
    if not sequence_cases:
        return run_single(binary, path, output)
    if config.get("fault_mode", 0):
        raise ValueError("sequence_cases are only for normal Cache transfers")

    allowed_overrides = {
        "tokens",
        "layers",
        "heads",
        "kv_heads",
        "head_dim",
        "pool_blocks",
        "linear_pool_blocks",
        "deadline_ms",
        "process_timeout_seconds",
    }
    lengths = [config["tokens"]] + [case["tokens"] for case in sequence_cases]
    if lengths != sorted(set(lengths)) or any(length <= 0 for length in lengths):
        raise ValueError("sequence lengths must be positive, distinct, and increasing")
    if any(set(case) - allowed_overrides for case in sequence_cases):
        raise ValueError("sequence case changes the cache layout or transfer topology")

    output.mkdir(parents=True, exist_ok=True)
    base = {key: value for key, value in config.items() if key != "sequence_cases"}
    runs = []
    for overrides in [{}] + sequence_cases:
        effective = {**base, **overrides}
        tokens = effective["tokens"]
        if tokens % effective["block_tokens"]:
            raise ValueError("sequence length must align to full cache blocks")
        if "logical_blocks" in effective:
            effective["logical_blocks"] = tokens // effective["block_tokens"]
        effective["case_id"] = f"{config['case_id']}-tokens-{tokens}"
        case_output = output / f"tokens-{tokens}"
        case_output.mkdir()
        case_config = case_output / "effective-config.json"
        case_config.write_text(json.dumps(effective, indent=2) + "\n")
        run_single(binary, case_config, case_output)
        result = json.loads((case_output / "result.json").read_text())
        runs.append(
            {
                "tokens": tokens,
                "status": result["status"],
                "result": str((case_output / "result.json").relative_to(output)),
                "config_sha256": result["config_sha256"],
                "rank_count": len(result["ranks"]),
                "logical_unit_comparisons": len(result["comparisons"]),
            }
        )
        (output / "sequence-progress.json").write_text(
            json.dumps(runs, indent=2) + "\n"
        )

    (output / "result.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "case_id": config["case_id"],
                "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "entry": "RemoteRpcServer::initCacheStore -> NormalCacheStore; DecodeRpcServer::loadCache -> CacheStore::loadBuffers",
                "sequence_runs": runs,
            },
            indent=2,
        )
        + "\n"
    )


def run(binary, path, output):
    output.mkdir(parents=True, exist_ok=True)
    try:
        run_cases(binary, path, output)
    except BaseException as error:
        try:
            write_json(
                output / "result.json",
                {
                    "status": "FAIL",
                    "error": f"{type(error).__name__}: {error}",
                    "config": str(path),
                },
            )
        except OSError as archive_error:
            print(f"Could not save failure summary: {archive_error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))

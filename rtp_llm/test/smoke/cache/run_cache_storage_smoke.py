"""Model-free storage smoke: real pools, transfer engine, files and full payloads."""

import argparse
import json
import os
import pathlib
import subprocess
from collections import Counter
from tempfile import TemporaryDirectory

SCENES = {
    "staging_full",
    "corruption",
    "truncate",
    "unlink",
    "init_existing",
    "init_enospc",
    "write_io_error",
    "pool_exhaustion",
}


def resolve(config):
    if config.get("scene") not in SCENES or config.get("dtype") != "INT8":
        raise ValueError("storage smoke requires a supported scene and MHA INT8")
    for name in (
        "layers",
        "kv_heads",
        "head_dim",
        "tokens",
        "block_tokens",
        "staging_blocks",
        "max_descriptors_per_batch",
        "repeat",
    ):
        if type(config.get(name)) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(config.get("seed")) is not int or not 0 <= config["seed"] < 2**64 - 2:
        raise ValueError("seed must allow two subsequent uint64 seeds")
    if (
        config["tokens"] % config["block_tokens"]
        or config["tokens"] // config["block_tokens"] < 4
    ):
        raise ValueError("storage requires at least four complete blocks")
    if config["staging_blocks"] < 2 or config["staging_blocks"] % 2:
        raise ValueError("production staging requires an even count of at least two")
    if config["tokens"] // config["block_tokens"] <= config["staging_blocks"] // 2:
        raise ValueError("descriptor count must exceed production lane capacity")
    payload = (
        config["layers"]
        * 2
        * config["kv_heads"]
        * config["block_tokens"]
        * (config["head_dim"] + 4)
    )
    if payload * (config["tokens"] // config["block_tokens"] + 3) > 256 * 1024**2:
        raise ValueError("storage fixture exceeds per-tier 256 MiB safety budget")
    return config


def check(report, config):
    failures = []

    def expect(value, message):
        if not value:
            failures.append(message)

    expect(
        report.get("scene") == config["scene"] and report.get("seed") == config["seed"],
        "native identity differs",
    )
    expect(report.get("phase") == "complete", "native scenario incomplete")
    resolved = report["resolved"]
    pages = config["tokens"] // config["block_tokens"]
    payload = (
        config["layers"]
        * 2
        * config["kv_heads"]
        * config["block_tokens"]
        * (config["head_dim"] + 4)
    )
    expect(
        resolved["payload_bytes_per_block"] == payload and resolved["pages"] == pages,
        "resolved independent geometry differs",
    )
    expect(
        resolved.get("device_address_oracle") == "raw_base_mha_int8",
        "storage DEVICE oracle must use independent raw-base addressing",
    )
    address_checks = report["address_self_checks"]
    expect(
        address_checks["restored"] is True, "address self-check did not restore data"
    )
    for kind in ("block_swap", "layer_swap"):
        sample = address_checks[kind]
        if kind == "layer_swap" and config["layers"] < 2:
            expect(sample is None, "single-layer fixture cannot swap layers")
            continue
        expect(
            sample["equal"] is False
            and sample["mismatch_bytes"] > 0
            and len(sample["units"]) == pages * config["layers"] * 2
            and sum(unit["mismatch_bytes"] for unit in sample["units"])
            == sample["mismatch_bytes"],
            f"{kind} must be detected by the full-byte checker",
        )
        changed = {
            (unit["logical_block"], unit["layer"], unit["component"])
            for unit in sample["units"]
            if not unit["equal"] and unit["expected_hash"] != unit["actual_hash"]
        }
        expected_changed = {
            (block, layer, component)
            for block in (range(2) if kind == "block_swap" else (0,))
            for layer in (range(config["layers"]) if kind == "block_swap" else range(2))
            for component in ("KV", "KV_scale")
        }
        expect(changed == expected_changed, f"{kind} affected unexpected units")
    expect(
        resolved["actual_lane_capacity"] == config["staging_blocks"] // 2,
        "constructor lane formula differs",
    )
    expect(
        all(
            resolved[name] == config[name]
            for name in (
                "tokens",
                "block_tokens",
                "layers",
                "kv_heads",
                "head_dim",
                "max_descriptors_per_batch",
            )
        )
        and resolved["staging_blocks"] == config["staging_blocks"]
        and resolved["dtype"] == "INT8"
        and resolved["scale_dtype"] == "FP32"
        and resolved["physical_blocks"] == pages + 3
        and resolved["group_type"] == "FULL",
        "native resolved configuration differs",
    )
    required = [("source_before", "DEVICE")]
    for phase, tier in (
        ("baseline_device_host", "HOST"),
        ("baseline_host_disk", "DISK"),
        ("baseline_disk_host", "HOST"),
        ("baseline_host_device", "DEVICE"),
        ("baseline_disk_device", "DEVICE"),
        ("recovery_disk", "DISK"),
        ("recovery_target", "DEVICE"),
        ("recovery_source", "DEVICE"),
    ):
        required.append((phase, tier))
    api_phases = {
        "baseline_device_host",
        "baseline_host_disk",
        "baseline_disk_host",
        "baseline_host_device",
        "baseline_device_disk",
        "baseline_disk_device",
        "recovery_device_disk",
        "recovery_disk_device",
    }
    scene = config["scene"]
    if scene in ("corruption", "truncate", "unlink"):
        api_phases |= {"fault_disk_device", "after_explicit_fixture_repair"}
        required += [
            ("fault_target", "DEVICE"),
            ("after_explicit_fixture_repair", "DEVICE"),
        ]
    if scene == "corruption":
        required.append(("fault_disk_source", "DISK"))
    if scene == "write_io_error":
        api_phases.add("fault_device_disk")
        required += [
            ("fault_disk_unchanged", "DISK"),
            ("fault_source_retained", "DEVICE"),
        ]
    required += [("guards", tier) for tier in ("DEVICE", "HOST", "DISK")]
    actual_snapshots = [
        (item["phase"], item.get("tier"))
        for item in report["observations"]
        if "units" in item
    ]
    expect(
        Counter(actual_snapshots) == Counter(required),
        "required full snapshot phases/tier coverage differs",
    )
    expect(
        Counter(item["phase"] for item in report["observations"] if "api" in item)
        == Counter(api_phases),
        "required real transfer phase coverage differs",
    )
    by_phase = {}
    for observation in report["observations"]:
        by_phase.setdefault(observation["phase"], []).append(observation)
        if "api" in observation:
            expected = not (
                observation["phase"] == "fault_device_disk"
                and scene == "write_io_error"
                or observation["phase"] == "fault_disk_device"
                and scene == "truncate"
            )
            expect(
                observation["expected_success"] == expected,
                "native expected API differs from scene contract",
            )
            expected_calls = (
                pages
                if observation["phase"]
                in ("baseline_device_disk", "recovery_device_disk")
                else 1
            )
            expect(
                len(observation["api"]) == expected_calls,
                "production API call count differs",
            )
            expect(
                observation["success"] == expected,
                f"{observation['phase']} API success differs",
            )
            expect(
                all((item["code"] == 0) == expected for item in observation["api"]),
                f"{observation['phase']} API codes differ",
            )
            if not expected:
                # All current storage transport failures aggregate to the existing
                # generic execution error; preserve each raw message and leaf I/O.
                expect(
                    all(item["code"] == 606 for item in observation["api"]),
                    "upper error is not existing execution contract",
                )
            if observation["phase"] in ("baseline_disk_device", "recovery_disk_device"):
                batches = [
                    item["count"]
                    for item in observation["io_events"]
                    if item["operation"] == "read_batch"
                ]
                capacity = min(
                    config["staging_blocks"] // 2, config["max_descriptors_per_batch"]
                )
                expect(
                    len(batches) > 1
                    and sum(batches) == pages
                    and max(batches) <= capacity,
                    "actual DISK read batch boundary not exercised",
                )
        else:
            logical_blocks = (
                [-1, pages] if observation["phase"] == "guards" else range(pages)
            )
            expected_units = [
                (block, layer, component)
                for block in logical_blocks
                for layer in range(config["layers"])
                for component in ("KV", "KV_scale")
            ]
            actual_units = [
                (unit["logical_block"], unit["layer"], unit["component"])
                for unit in observation["units"]
            ]
            expect(
                actual_units == expected_units,
                "complete ordered logical unit identity differs",
            )
            for component, width in (("KV", config["head_dim"]), ("KV_scale", 4)):
                expect(
                    all(
                        unit["bytes"]
                        == 2 * config["kv_heads"] * config["block_tokens"] * width
                        for unit in observation["units"]
                        if unit["component"] == component
                    ),
                    "logical component byte geometry differs",
                )
            physical_by_logical = {}
            for unit in observation["units"]:
                physical_by_logical.setdefault(unit["logical_block"], set()).add(
                    unit["physical_block"]
                )
            expect(
                all(
                    len(blocks) == 1 and 1 <= next(iter(blocks)) <= pages + 2
                    for blocks in physical_by_logical.values()
                )
                and len({next(iter(blocks)) for blocks in physical_by_logical.values()})
                == len(physical_by_logical),
                "physical blocks overlap or differ per logical block",
            )
            intentional = observation["phase"] in (
                "fault_disk_source",
                "fault_target",
            ) and config["scene"] in ("corruption", "truncate")
            expect(
                observation["equal"] != intentional,
                f"{observation['phase']} full-byte result differs",
            )
            expect(
                sum(unit["mismatch_bytes"] for unit in observation["units"])
                == observation["mismatch_bytes"],
                "unit mismatch aggregation differs",
            )
            expect(
                all(
                    unit["equal"] == (unit["mismatch_bytes"] == 0)
                    for unit in observation["units"]
                ),
                "unit byte predicate differs",
            )
            if not intentional:
                expect(
                    all(
                        unit["equal"] and unit["actual_hash"] == unit["expected_hash"]
                        for unit in observation["units"]
                    ),
                    "complete expected/source/target signature differs",
                )
            expect(
                len(observation["units"])
                == (2 if observation["phase"] == "guards" else pages)
                * config["layers"]
                * 2,
                "logical payload coverage differs",
            )
    injection = report["injection"]
    scene = config["scene"]
    if scene == "corruption":
        source = by_phase["fault_disk_source"][0]
        target = by_phase["fault_target"][0]
        expect(
            source["mismatch_bytes"] == target["mismatch_bytes"] == 1,
            "one actual corrupt byte was not independently detected",
        )
        expect(
            [unit["actual_hash"] for unit in source["units"]]
            == [unit["actual_hash"] for unit in target["units"]],
            "accepted corrupted image differs source/target",
        )
        expect(
            injection["original_byte"] ^ injection["changed_byte"] == 0x5A,
            "physical corruption differs",
        )
    elif scene == "truncate":
        statuses = [
            event.get("status")
            for item in by_phase["fault_disk_device"]
            for event in item["io_events"]
        ]
        expect("PARTIAL_FAILURE" in statuses, "real EOF leaf PARTIAL_FAILURE absent")
    elif scene == "unlink":
        expect(
            injection["path_absent"] and injection["open_fd_remains"],
            "unlink open-fd state differs",
        )
    elif scene.startswith("init_"):
        expect(injection["init_rejected"], "production init did not reject")
        expect(
            any(item.get("status") == "IO_ERROR" for item in injection["io_events"]),
            "real init IO_ERROR absent",
        )
        expect(
            injection["enospc_syscall_count"] == (1 if scene == "init_enospc" else 0),
            "scoped ENOSPC syscall count differs",
        )
    elif scene == "write_io_error":
        statuses = [
            event
            for item in by_phase["fault_device_disk"]
            for event in item["io_events"]
        ]
        expect(
            any(
                event.get("injected") and event.get("status") == "IO_ERROR"
                for event in statuses
            ),
            "write interface seam absent",
        )
    elif scene == "pool_exhaustion":
        expect(
            len(injection["pools"]) == 3
            and all(
                item["extra_allocation_rejected"]
                and item["free_before"] == item["free_after"] == 0
                for item in injection["pools"]
            ),
            "three real tier pool exhaustion differs",
        )
    expect(
        len(report["capacity"]) == 3
        and all(
            item["full_capacity_allocated"]
            and item["extra_allocation_rejected"]
            and item["free_after_engine_destroy"]
            == item["free_after_probe"]
            == pages + 2
            for item in report["capacity"]
        ),
        "real full-capacity release/reallocation differs",
    )
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    args.binary = str(pathlib.Path(args.binary).resolve())
    config = resolve(json.loads(pathlib.Path(args.config).read_text()))
    output = pathlib.Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    runs = []
    for repetition in range(config["repeat"]):
        directory = output / f"run{repetition}"
        directory.mkdir()
        with TemporaryDirectory(
            prefix="cache-storage-", dir=os.environ["TEST_TMPDIR"]
        ) as temporary:
            backing = pathlib.Path(temporary)
            report_path = directory / "native.json"
            command = [args.binary, config["scene"], str(backing), str(report_path)] + [
                str(config[name])
                for name in (
                    "seed",
                    "layers",
                    "kv_heads",
                    "head_dim",
                    "tokens",
                    "block_tokens",
                    "staging_blocks",
                    "max_descriptors_per_batch",
                )
            ]
            with (directory / "native.log").open("w") as log:
                process = subprocess.Popen(
                    command, cwd=backing, stdout=log, stderr=subprocess.STDOUT
                )
                try:
                    returncode = process.wait(
                        timeout=config.get("process_timeout_seconds", 120)
                    )
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    returncode = -9
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
            failures = []
            report = None
            if returncode == 0 and report_path.exists():
                report = json.loads(report_path.read_text())
                failures = check(report, config)
                if report["pid"] != process.pid:
                    failures.append("native PID differs from owned Popen")
            else:
                failures.append(
                    f"native return code {returncode}, report exists {report_path.exists()}"
                )
            files = list(backing.iterdir())
        runs.append(
            {
                "pid": process.pid,
                "returncode": returncode,
                "failures": failures,
                "native_report": str(report_path),
                "backing_files_removed": len(files),
                "process_reaped": True,
                "backing_directory_removed": not backing.exists(),
            }
        )
    summary = {
        "name": config["name"],
        "test_type": "smoke",
        "backend": "production_PerRankBlockTransferEngine_PosixDiskIO",
        "config": config,
        "runs": runs,
        "status": "PASS" if all(not run["failures"] for run in runs) else "FAIL",
    }
    (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

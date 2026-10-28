"""Manual ACU collection inside the outer Bazel GPU lock.

The parent imports only stdlib. ACU starts this same file with --worker before
the child imports torch. Native PPU reports are not NVIDIA NCU reports. Unknown
CSV/kernel evidence fails closed until a measured native schema is reviewed.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ACU = "/usr/local/PPU_SDK/asight/bin/acu"
BENCHMARK = "test_cascade_kernel_benchmark.py"
DEFAULT_SECTIONS = "LaunchStats,SpeedOfLight,Occupancy"
NATIVE_CSV_HEADER = (
    "ID",
    "Process ID",
    "Process Name",
    "Host Name",
    "Kernel Name",
    "Kernel Mangled Name",
    "Context",
    "Stream",
    "Block Size",
    "Grid Size",
    "Device",
    "CC",
    "Section Name",
    "Metric Name",
    "Metric Unit",
    "Metric Value",
    "Rule Name",
    "Rule Type",
    "Rule Description",
)
ACCURACY_SCOPE = {
    "acceptance": "NOT_ASSESSED_BY_PROFILING",
    "baseline": "cascade_baseline_real_01",
    "known_status": "REGRESSION_FAIL",
    "known_scope": (
        "real 665-token layer27 on/off exceeds legacy output tolerance: "
        "39 elements in original state, 183 in derived perturbation; "
        "not a numerical acceptance of either path"
    ),
    "normal_replay": (
        "captured layers0/13/27 matched normal off output bitwise in three modes"
    ),
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_info(path):
    exists = path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else None,
        "sha256": digest_file(path) if exists else None,
    }


def discover_native_report(requested, manifest):
    # Measured ACU 2.1 appends .acurep even to an explicit .asysrep export.
    # Keep its real filename and record both candidates; never rename a report.
    candidates = [requested, Path(str(requested) + ".acurep")]
    observed = [file_info(path) for path in candidates]
    manifest["native_export"] = {
        "requested_path": str(requested),
        "candidates": observed,
    }
    nonempty = [path for path in candidates if path.is_file() and path.stat().st_size]
    require(len(nonempty) <= 1, "ambiguous nonempty native export files")
    selected = nonempty[0] if nonempty else requested
    if selected.is_file():
        # Native reports embed the target environment; keep the original local.
        selected.chmod(0o600)
    manifest["native_export"]["selected_path"] = str(selected)
    return selected


def parse_options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixture", help="real B500 L665 layer27 fixture")
    source.add_argument("--synthetic", action="store_true", help="explicit B500 L665")
    parser.add_argument("--backend", choices=("off", "on"), default="on")
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--kernel-regex", default="BatchPrefillWithPagedKVCacheKernel")
    parser.add_argument("--launch-count", type=int, default=1)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sections", help="comma-separated section identifiers")
    selection.add_argument("--metrics", help="explicit comma-separated native metrics")
    parser.add_argument("--timeout-s", type=float, default=300)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--workspace-mib", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    require(Path(options.profile_dir).is_absolute(), "profile-dir must be absolute")
    if options.fixture:
        require(Path(options.fixture).is_absolute(), "fixture must be absolute")
    for name in ("launch_count", "timeout_s", "warmup", "workspace_mib"):
        require(getattr(options, name) > 0, f"{name} must be positive")
    require(bool(options.kernel_regex), "kernel-regex cannot be empty")
    require(not options.kernel_regex.startswith("regex:"), "omit regex: prefix")
    re.compile(options.kernel_regex)
    if options.metrics is None and options.sections is None:
        options.sections = DEFAULT_SECTIONS
    if options.metrics is not None:
        metrics = options.metrics.split(",")
        require(
            all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_.]*", value) for value in metrics),
            "metrics must be explicit comma-separated native identifiers",
        )
        require(len(set(metrics)) == len(metrics), "duplicate metrics")
    else:
        sections = options.sections.split(",")
        require(
            all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value) for value in sections),
            "sections must be comma-separated native ACU identifiers",
        )
        require(len(set(sections)) == len(sections), "duplicate sections")
    return options


def common_arguments(options):
    result = ["--profile-dir", options.profile_dir]
    result += ["--fixture", options.fixture] if options.fixture else ["--synthetic"]
    for name in (
        "backend",
        "mode",
        "kernel_regex",
        "launch_count",
        "timeout_s",
        "warmup",
        "workspace_mib",
        "seed",
    ):
        result += ["--" + name.replace("_", "-"), str(getattr(options, name))]
    result += (
        ["--metrics", options.metrics]
        if options.metrics is not None
        else ["--sections", options.sections]
    )
    return result


def process_identity(pid):
    """Read one explicitly named PID only; never enumerate other processes."""
    root = Path("/proc") / str(pid)
    try:
        status = (root / "status").read_text()
        uid_line = next(line for line in status.splitlines() if line.startswith("Uid:"))
        uids = [int(value) for value in uid_line.split()[1:]]
        stat = (root / "stat").read_text()
        fields = stat[stat.rfind(")") + 2 :].split()
        return {
            "pid": pid,
            "uids": uids,
            "start_ticks": int(fields[19]),
            "state": fields[0],
            "pgid": os.getpgid(pid),
            "argv": [
                os.fsdecode(value)
                for value in (root / "cmdline").read_bytes().split(b"\0")
                if value
            ],
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def matches_worker(identity, expected_script, profile_dir):
    if identity is None or identity["uids"] != [os.getuid()] * 4:
        return False
    argv = identity["argv"]
    if len(argv) < 3 or argv[1] != expected_script or argv.count("--worker") != 1:
        return False
    # The frozen child command uses a separate exact value, never substring matching.
    indices = [i for i, value in enumerate(argv) if value == "--profile-dir"]
    return (
        len(indices) == 1
        and indices[0] + 1 < len(argv)
        and argv[indices[0] + 1] == profile_dir
    )


def target_alive(pid, group):
    try:
        (os.killpg if group else os.kill)(pid, 0)
        return True
    except ProcessLookupError:
        return False


def cleanup_worker(manifest):
    """Bounded cleanup of this run's verified worker/private group only."""
    result = {
        "status": "UNVERIFIED_INCOMPLETE",
        "signals": [],
        "scope_limit": "no claim about processes outside the verified target/group",
    }
    profile_dir = manifest["options"]["profile_dir"]
    record_path = Path(profile_dir) / "analysis" / "worker.json"
    result["worker_record"] = str(record_path)
    if not record_path.is_file():
        result["reason"] = "worker record absent; no PID guessing or process scan"
        return result
    pid = json.loads(record_path.read_text()).get("pid")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 1
        or pid == os.getpid()
    ):
        result["reason"] = "invalid recorded worker PID"
        return result
    result["pid"] = pid
    expected_script = manifest["child_argv"][1]
    initial = process_identity(pid)
    if not matches_worker(initial, expected_script, profile_dir):
        result["reason"] = (
            "worker absent or UID/exact command identity rejected; no signal sent"
        )
        result["worker_present"] = initial is not None
        return result
    group = initial["pgid"] == pid
    result.update(
        {
            "identity_verified": True,
            "uid": initial["uids"][0],
            "start_ticks": initial["start_ticks"],
            "initial_pgid": initial["pgid"],
            "scope": "verified_private_process_group" if group else "verified_pid_only",
        }
    )
    for sig in (signal.SIGTERM, signal.SIGKILL):
        current = process_identity(pid)
        if current is not None and (
            current["start_ticks"] != initial["start_ticks"]
            or (
                current["state"] != "Z"
                and not matches_worker(current, expected_script, profile_dir)
            )
            or current["uids"] != initial["uids"]
            or (group and current["pgid"] != pid)
        ):
            result["reason"] = "identity changed before signal; cleanup stopped"
            return result
        if not target_alive(pid, group):
            break
        # If the original leader has exited, a still-existing verified group
        # retains its identity. A reused leader PID is rejected by start_ticks.
        try:
            (os.killpg if group else os.kill)(pid, sig)
            result["signals"].append(signal.Signals(sig).name)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and target_alive(pid, group):
            time.sleep(0.1)
    result["target_still_exists"] = target_alive(pid, group)
    result["status"] = (
        "VERIFIED_PRIVATE_GROUP_GONE"
        if group and not result["target_still_exists"]
        else (
            "INCOMPLETE_TARGET_STILL_EXISTS"
            if result["target_still_exists"]
            else "INCOMPLETE_PID_ONLY_SCOPE"
        )
    )
    return result


def no_kernel_evidence(reports):
    """Only recognize negative evidence measured in native ACU smoke02."""
    evidence = []
    stdout = reports / "acu_import.stdout.txt"
    if stdout.is_file() and "No kernel in report." in [
        line.strip() for line in stdout.read_text(errors="replace").splitlines()
    ]:
        evidence.append("native_import_explicit_no_kernel")
    path = reports / "report.csv"
    raw = reports / "report.raw.csv"
    if path.is_file():
        with path.open(newline="") as handle:
            rows = csv.reader(handle)
            if (
                tuple(next(rows, ())) == NATIVE_CSV_HEADER
                and not any(any(value.strip() for value in row) for row in rows)
                and not (raw.is_file() and raw.stat().st_size)
            ):
                evidence.append("measured_native_csv_header_without_data_rows")
    return evidence


def run_process(argv, stdout_path, stderr_path, timeout_s, manifest, label):
    result = {
        "argv": argv,
        "started_unix": time.time(),
        "returncode": None,
        "timed_out": False,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    manifest["processes"][label] = result
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                argv, stdout=stdout, stderr=stderr, start_new_session=True
            )
            result["pid"] = process.pid
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                result["timed_out"] = True
                # ACU creates a separate target PGID: the parent group alone
                # does not establish cleanup of the worker or other descendants.
                if label == "collect":
                    try:
                        result["worker_cleanup"] = cleanup_worker(manifest)
                    except Exception:
                        result["worker_cleanup"] = {
                            "status": "CLEANUP_EXCEPTION_INCOMPLETE",
                            "exception": traceback.format_exc(),
                        }
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                # ACU can exit before a child that ignores TERM; kill remaining
                # members of this dedicated group even if the leader has exited.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    result["acu_cleanup_incomplete"] = True
                result["acu_process_still_alive"] = process.poll() is None
                result["acu_group_still_exists"] = target_alive(process.pid, True)
            result["returncode"] = process.returncode
    except Exception:
        result["exception"] = traceback.format_exc()
        raise
    finally:
        result["finished_unix"] = time.time()
    return result


def worker(options):
    root = Path(options.profile_dir)
    path = root / "analysis" / "worker.json"
    record = {
        "schema_version": 1,
        "status": "STARTING",
        "argv": sys.argv,
        "pid": os.getpid(),
        "returncode": None,
        "profiler_start_returned": False,
        "profiler_stop_returned": False,
        "profiled_calls": 0,
        "accuracy": ACCURACY_SCOPE,
    }
    exit_code = 1
    try:
        require(root.is_dir() and path.parent.is_dir(), "parent did not initialize run")
        require(
            os.environ.get("PPU_CASCADE_ACU_WORKER_DIR") == str(root),
            "worker must be launched by this parent under ACU",
        )
        write_json(path, record)
        # No accelerator-dependent import occurs in the parent process.
        import torch

        from rtp_llm.device.device_type import DeviceType, get_device_type

        source = root / "harness" / BENCHMARK
        specification = importlib.util.spec_from_file_location(
            "cascade_acu_frozen_benchmark", source
        )
        require(
            specification is not None and specification.loader is not None,
            "cannot load frozen benchmark",
        )
        benchmark = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(benchmark)
        detected = get_device_type()
        record["device_detection"] = {
            "device_type": detected.name,
            "torch_version": str(torch.__version__),
            "ppu_home": os.environ.get("PPU_HOME"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_compatibility_available": torch.cuda.is_available(),
        }
        write_json(path, record)
        require(detected == DeviceType.Ppu, "requires PPU; never skip as PASS")
        require(torch.cuda.is_available(), "PPU CUDA compatibility runtime unavailable")
        torch.set_num_threads(4)
        torch.manual_seed(options.seed)
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        record["device_detection"].update(
            {
                "current_device": device_index,
                "name": torch.cuda.get_device_name(device_index),
                "properties": str(properties),
                "cpu_threads": torch.get_num_threads(),
            }
        )
        record["benchmark_source"] = file_info(source)
        if options.fixture:
            payload = torch.load(options.fixture, map_location="cpu", weights_only=True)
            require(payload.get("schema_version") == 1, "fixture schema must be1")
            require(
                "cases" not in payload, "this worker requires one snapshot per file"
            )
            state = benchmark.validate_state(payload)
            metadata = state.get("metadata", {})
            require(metadata.get("layer") == 27, "first real profile targets layer27")
            require(
                metadata.get("exact_prefill_token_ids") is True,
                "real fixture must carry exact request identity",
            )
            require(
                metadata.get("current_step_kv_already_written") is True,
                "real fixture must include current-step KV",
            )
            record["input"] = {
                "source": "real_fixture",
                "fixture": file_info(Path(options.fixture)),
                "layer": metadata["layer"],
                "request_file_sha256": metadata.get("request_file_sha256"),
                "capture_run_id": metadata.get("run_id"),
                "is_real_model_step": True,
            }
        else:
            state = benchmark.synthetic_state(500, 665, 1, options.seed)
            record["input"] = {
                "source": "explicit_synthetic",
                "seed": options.seed,
                "is_real_model_step": False,
            }
        require(tuple(state["q"].shape) == (500, 16, 128), "requires exact B500 Q")
        require(bool((state["sequence_lengths"] == 665).all()), "requires history L665")
        record["input"]["tensors"] = {
            key: {
                "shape": list(state[key].shape),
                "dtype": str(state[key].dtype),
                "device": str(state[key].device),
            }
            for key in ("q", "kv_cache", "block_table", "sequence_lengths")
        }
        record["input"].update(
            {
                "history_min": 665,
                "history_max": 665,
                "valid_kv_tokens": 666,
                "layout": "HND",
                "scale": 128**-0.5,
                "bucket": 500,
            }
        )
        factory = (
            benchmark.DecodeBackend
            if options.backend == "off"
            else benchmark.CascadeBackend
        )
        runner = benchmark.Runner(
            factory,
            500,
            state["block_table"].shape[1],
            state["kv_cache"].shape[0] + 2,
            1,
            options.mode == "graph",
            SimpleNamespace(workspace_mib=options.workspace_mib),
        )
        runner.update(state)
        if options.mode == "graph":
            runner.capture(repeats=1, warmup=options.warmup)
        else:
            for _ in range(options.warmup):
                runner.run()
        torch.cuda.synchronize()
        record.update(
            {
                "status": "READY_FOR_PROFILER_START",
                "backend": options.backend,
                "mode": options.mode,
                "runner": runner.describe(),
                "warmup_calls": options.warmup,
                "graph_repeat": 1,
                "profile_region": "one full attention call, filtered by ACU kernel regex",
                "numeric_acceptance_performed": False,
            }
        )
        write_json(path, record)
        primary_error = None
        try:
            torch.cuda.profiler.start()
            record["profiler_start_returned"] = True
            if options.mode == "graph":
                runner.graph.replay()
            else:
                runner.run()
            record["profiled_calls"] = 1
            torch.cuda.synchronize()
        except Exception:
            primary_error = traceback.format_exc()
            record["profile_region_exception"] = primary_error
        finally:
            if record["profiler_start_returned"]:
                try:
                    torch.cuda.profiler.stop()
                    record["profiler_stop_returned"] = True
                except Exception:
                    record["profiler_stop_exception"] = traceback.format_exc()
        require(
            primary_error is None,
            "profiler start/run/synchronize failed; see worker.json",
        )
        require(
            record["profiler_stop_returned"], "profiler stop failed; see worker.json"
        )
        record["status"] = "WORKER_COMPLETED_NOT_NUMERICAL_ACCEPTANCE"
        exit_code = 0
    except Exception:
        record["status"] = "WORKER_FAILED"
        record["exception"] = traceback.format_exc()
        print(record["exception"], file=sys.stderr, flush=True)
    finally:
        record["returncode"] = exit_code
        if path.parent.is_dir():
            write_json(path, record)
    return exit_code


def report_text(manifest):
    return "\n".join(
        [
            "# PPU Cascade ACU collection",
            "",
            f"Status: {manifest['status']}",
            "",
            "This is a profiling artifact, not numerical acceptance or clean latency.",
            "The known layer27 Cascade regression remains open; see manifest accuracy scope.",
            "",
            "Native tool: ACU/PPU. Reports retain .asysrep and native metric names/units.",
            "No ncu_report.load_report compatibility or NVIDIA metric mapping is claimed.",
            "ACU cache-control=none still flushes L1/L2; it only preserves LLC.",
            "Profiler replay durations must not be merged into uninstrumented timings.",
            "",
            "## Evidence and next diagnosis",
            "",
            "See analysis/manifest.json, analysis/worker.json, and reports/ raw outputs.",
            "Kernel count/schema is not inferred from a nonempty report or CSV. Review the",
            "first native export before adding a measured kernel-evidence acceptance rule.",
            "",
            "## Six dimensions (NOT_ASSESSED until native metrics are verified)",
            "",
            "1. Occupancy and launch geometry.",
            "2. Block balance and tail effects.",
            "3. Instruction stalls and source correlation.",
            "4. Tensor-core usage, preserving BF16 metric identity.",
            "5. Utilization timeline.",
            "6. Memory and cache behavior.",
            "Additional: roofline/efficiency, with verified PPU peaks and units.",
            "",
            "Plan: verify matched prefix/tail/merge kernel and supported counters first;",
            "then diagnose bottlenecks and propose changes using separate clean timing.",
            "",
        ]
    )


def parent(options):
    require("torch" not in sys.modules, "parent must start before torch import")
    root = Path(options.profile_dir)
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name in ("harness", "reports", "analysis"):
        (root / name).mkdir()
    manifest = {
        "schema_version": 1,
        "status": "INITIALIZING",
        "tool": "ACU/PPU",
        "parent_argv": sys.argv,
        "parent_pid": os.getpid(),
        "options": vars(options),
        "processes": {},
        "accuracy": ACCURACY_SCOPE,
        "kernel_evidence": {"status": "NOT_ASSESSED", "matched_kernel_count": None},
        "cache_control": "none: PPU ACU still flushes L1/L2, preserves LLC",
        "timing_scope": "profiling/replay only; exclude from clean latency/QPS",
        "gpu_lock": "inherited from outer Bazel test; no independent GPU allocation",
        "tool_environment": {
            name: os.environ.get(name)
            for name in (
                "PATH",
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "PYTHONPATH",
                "PPU_HOME",
                "PPU_SDK",
                "CUDA_HOME",
                "CUDA_PATH",
                "ASIGHT_HOME",
                "CUDA_VISIBLE_DEVICES",
                "GPU_COUNT",
            )
        },
    }
    manifest_path = root / "analysis" / "manifest.json"
    exit_code = 1
    previous_worker_dir = os.environ.get("PPU_CASCADE_ACU_WORKER_DIR")
    try:
        require(previous_worker_dir is None, "recursive profiler parent is forbidden")
        require(
            os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"), "requires outer Bazel test"
        )
        source = Path(__file__).resolve()
        worker_source = root / "harness" / source.name
        benchmark_source = source.with_name(BENCHMARK)
        shutil.copy2(source, worker_source)
        shutil.copy2(benchmark_source, root / "harness" / BENCHMARK)
        manifest["sources"] = [
            file_info(worker_source),
            file_info(root / "harness" / BENCHMARK),
        ]
        manifest["acu_binary"] = file_info(Path(ACU))
        release = Path("/usr/local/PPU_SDK/release.yaml")
        manifest["ppu_sdk_release"] = {
            **file_info(release),
            "text": release.read_text() if release.is_file() else None,
        }
        require(os.access(ACU, os.X_OK), "ACU binary is unavailable or not executable")
        if options.fixture:
            manifest["fixture"] = file_info(Path(options.fixture))
            require(manifest["fixture"]["exists"], "fixture does not exist")
        reports = root / "reports"
        native, csv_path = reports / "report.asysrep", reports / "report.csv"
        child_argv = [
            sys.executable,
            str(worker_source),
            "--worker",
        ] + common_arguments(options)
        argv = [
            ACU,
            "--profile-from-start",
            "no",
            "--replay-mode",
            "kernel",
            "--graph-profiling",
            "node",
            "--kernel-name-base",
            "demangled",
            "-k",
            "regex:" + options.kernel_regex,
            "--launch-count",
            str(options.launch_count),
            "--kill",
            "no",
            "--check-exit-code",
            "yes",
            "--cache-control",
            "none",
            "--export",
            str(native),
            "--csv-file",
            str(csv_path),
        ]
        if options.metrics is not None:
            argv += ["--metrics", options.metrics]
        else:
            for section in options.sections.split(","):
                argv += ["--section", section]
        argv += child_argv
        manifest.update(
            {"status": "COLLECTING", "child_argv": child_argv, "acu_argv": argv}
        )
        write_json(manifest_path, manifest)
        (root / "REPORT.md").write_text(report_text(manifest))
        os.environ["PPU_CASCADE_ACU_WORKER_DIR"] = str(root)
        collection = run_process(
            argv,
            reports / "acu.stdout.txt",
            reports / "acu.stderr.txt",
            options.timeout_s,
            manifest,
            "collect",
        )
        native = discover_native_report(native, manifest)
        manifest["artifacts"] = {
            "native": file_info(native),
            "csv": file_info(csv_path),
        }
        raw_path = reports / "report.raw.csv"
        if native.is_file() and native.stat().st_size:
            run_process(
                [
                    ACU,
                    "--import",
                    str(native),
                    "--page",
                    "raw",
                    "--csv-file",
                    str(raw_path),
                ],
                reports / "acu_import.stdout.txt",
                reports / "acu_import.stderr.txt",
                min(options.timeout_s, 120),
                manifest,
                "import_raw",
            )
        manifest["artifacts"]["raw_csv"] = file_info(raw_path)
        worker_path = root / "analysis" / "worker.json"
        child = json.loads(worker_path.read_text()) if worker_path.is_file() else {}
        manifest["worker_returncode"] = child.get("returncode")
        manifest["worker_status"] = child.get("status", "NO_WORKER_RECORD")
        failures = []
        if collection["timed_out"] or collection["returncode"] != 0:
            failures.append("ACU collection failed or timed out")
        if child.get("returncode") != 0 or child.get("profiled_calls") != 1:
            failures.append("worker did not complete exactly one profiled call")
        if not child.get("profiler_start_returned") or not child.get(
            "profiler_stop_returned"
        ):
            failures.append("explicit profiler start/stop was not confirmed")
        for key in ("native", "csv", "raw_csv"):
            if not manifest["artifacts"][key]["bytes"]:
                failures.append(f"missing/empty {key} report")
        imported = manifest["processes"].get("import_raw", {})
        if imported.get("returncode") != 0 or imported.get("timed_out"):
            failures.append("native raw import did not complete successfully")
        negative_evidence = no_kernel_evidence(reports)
        if negative_evidence:
            manifest["kernel_evidence"] = {
                "status": "NO_KERNEL",
                "matched_kernel_count": 0,
                "evidence": negative_evidence,
            }
            failures.append("native report contains no kernel")
        manifest["collection_failures"] = failures
        if failures:
            manifest["status"] = "COLLECTION_FAILED"
        else:
            # First native CSV schema is unknown. Do not invent a kernel counter
            # or accept headers/diagnostics as proof of a matched kernel.
            manifest["status"] = "KERNEL_EVIDENCE_UNVERIFIED"
            manifest["kernel_evidence"]["status"] = "NEEDS_NATIVE_REPORT_REVIEW"
            exit_code = 2
    except Exception:
        manifest["status"] = "HARNESS_FAILED"
        manifest["exception"] = traceback.format_exc()
        print(manifest["exception"], file=sys.stderr, flush=True)
    finally:
        if previous_worker_dir is None:
            os.environ.pop("PPU_CASCADE_ACU_WORKER_DIR", None)
        else:
            os.environ["PPU_CASCADE_ACU_WORKER_DIR"] = previous_worker_dir
        manifest["harness_returncode"] = exit_code
        write_json(manifest_path, manifest)
        (root / "REPORT.md").write_text(report_text(manifest))
        archive_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if archive_dir:
            write_json(
                Path(archive_dir) / "cascade_acu_artifacts.json",
                {
                    "schema_version": 1,
                    "status": manifest["status"],
                    "profile_dir": str(root),
                    "manifest": str(manifest_path),
                    "report": str(root / "REPORT.md"),
                    "harness_returncode": exit_code,
                },
            )
        print(
            f"CASCADE_ACU status={manifest['status']} manifest={manifest_path}",
            flush=True,
        )
    return exit_code


if __name__ == "__main__":
    arguments = parse_options()
    sys.exit(worker(arguments) if arguments.worker else parent(arguments))

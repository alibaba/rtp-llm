#!/usr/bin/env python3
"""
BlockTreeCache GPU Benchmark Python Driver.

Orchestrates native process execution for smoke and profile benchmark suites.
Handles case selection, repetitions, watchdog, timeout, and perf collection.
"""

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from benchmark_cases import CASE_REGISTRY, BenchmarkCase, get_suite_cases


def parse_args():
    parser = argparse.ArgumentParser(description="BlockTreeCache GPU Benchmark Driver")
    parser.add_argument(
        "--suite",
        default="all",
        choices=["all", "smoke", "profile"],
        help="Benchmark suite to run",
    )
    parser.add_argument(
        "--case",
        default="all",
        help="Comma-separated case names, or 'all' for all cases in suite",
    )
    parser.add_argument(
        "--binary",
        default=None,
        help="Path to block_tree_cache_gpu_benchmark binary "
        "(default: auto-detect from Bazel runfiles)",
    )
    parser.add_argument(
        "--model-profile",
        default=None,
        help="Path to model profile JSON " "(default: auto-detect from Bazel runfiles)",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/block_tree_cache_benchmark",
        help="Output directory for results",
    )
    parser.add_argument("--disk-root", help="Root directory for disk benchmark cases")
    parser.add_argument(
        "--cuda-device", type=int, default=0, help="CUDA device ordinal"
    )
    parser.add_argument(
        "--max-device-memory-fraction",
        type=float,
        default=0.8,
        help="Fraction of currently free CUDA memory usable by preflight (default: 0.8)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Base seed; repetition N uses seed+N"
    )
    parser.add_argument(
        "--task-pool-size",
        type=int,
        action="append",
        dest="task_pool_sizes",
        help="Override Tree task-pool size; repeat for a paired matrix (for example 4 and 8)",
    )
    parser.add_argument(
        "--process-repetitions",
        type=int,
        default=1,
        help="Number of native process repetitions (default: 1)",
    )
    parser.add_argument(
        "--case-timeout-seconds",
        type=int,
        default=600,
        help="Per-case timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--termination-grace-seconds",
        type=int,
        default=30,
        help="Grace period for process termination (default: 30)",
    )
    parser.add_argument(
        "--perf",
        default="record",
        choices=["record", "stat", "off"],
        help="Perf collection mode (default: record)",
    )
    parser.add_argument(
        "--perf-frequency",
        type=int,
        default=999,
        help="Perf sampling frequency in Hz (default: 999)",
    )
    parser.add_argument(
        "--flamegraph-tools-dir",
        help="Directory containing stackcollapse-perf.pl and flamegraph.pl "
        "(default: auto-detect FLAMEGRAPH_DIR, ~/FlameGraph, or ~/FlameGraph-master)",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Return success even when a case is skipped or partial",
    )
    return parser.parse_args()


def build_native_command(
    binary: str,
    subcommand: str,
    params: Dict[str, str],
    output_json: str,
    cuda_device: int,
    max_device_memory_fraction: float,
    seed: int,
    repetition_id: int,
    model_profile: str,
) -> List[str]:
    cmd = [binary, subcommand]
    cmd.append("--cuda-device")
    cmd.append(str(cuda_device))
    cmd.append("--max-device-memory-fraction")
    cmd.append(str(max_device_memory_fraction))
    cmd.append("--model-profile")
    cmd.append(model_profile)
    cmd.append("--seed")
    cmd.append(str(seed))
    cmd.append("--repetition-id")
    cmd.append(str(repetition_id))
    if output_json:
        cmd.extend(["--output-json", output_json])

    for key, value in params.items():
        cmd.append(key)
        cmd.append(value)
    return cmd


def run_native_process(
    binary: str,
    subcommand: str,
    params: Dict[str, str],
    output_json: str,
    cuda_device: int,
    max_device_memory_fraction: float,
    seed: int,
    repetition_id: int,
    model_profile: str,
    timeout_seconds: int,
    grace_seconds: int,
) -> Tuple[int, str, str]:
    """Run a single native benchmark process and return (exit_code, stdout, stderr)."""
    cmd = build_native_command(
        binary,
        subcommand,
        params,
        output_json,
        cuda_device,
        max_device_memory_fraction,
        seed,
        repetition_id,
        model_profile,
    )

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
        return proc.returncode, stdout.decode(), stderr.decode()
    except subprocess.TimeoutExpired:
        # Kill process group
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=grace_seconds)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                pass
        return -1, "", "TIMEOUT"
    except Exception as e:
        return -1, "", str(e)


def resolve_flamegraph_tools(
    configured_dir: Optional[str],
) -> Optional[Tuple[str, str]]:
    """Locate Brendan Gregg's stackcollapse and flamegraph scripts."""
    candidates = [
        configured_dir,
        os.environ.get("FLAMEGRAPH_DIR"),
        os.path.expanduser("~/FlameGraph"),
        os.path.expanduser("~/FlameGraph-master"),
    ]
    stackcollapse_on_path = shutil.which("stackcollapse-perf.pl")
    flamegraph_on_path = shutil.which("flamegraph.pl")
    if stackcollapse_on_path and flamegraph_on_path:
        return stackcollapse_on_path, flamegraph_on_path
    for directory in candidates:
        if not directory:
            continue
        stackcollapse = os.path.join(directory, "stackcollapse-perf.pl")
        flamegraph = os.path.join(directory, "flamegraph.pl")
        if os.path.isfile(stackcollapse) and os.path.isfile(flamegraph):
            return stackcollapse, flamegraph
    return None


def generate_flamegraph(
    perf_data: str,
    output_dir: str,
    flamegraph_tools: Tuple[str, str],
) -> Tuple[bool, str]:
    """Convert perf.data into folded stacks and a self-contained SVG."""
    stackcollapse, flamegraph = flamegraph_tools
    perf_script_path = os.path.join(output_dir, "perf_script.txt")
    folded_path = os.path.join(output_dir, "perf.folded")
    svg_path = os.path.join(output_dir, "flamegraph.svg")
    try:
        with open(perf_script_path, "w") as perf_script:
            scripted = subprocess.run(
                ["perf", "script", "--input", perf_data],
                stdout=perf_script,
                stderr=subprocess.PIPE,
                text=True,
                timeout=180,
            )
        if scripted.returncode != 0:
            return False, f"perf script failed: {scripted.stderr[:300]}"

        with open(perf_script_path) as perf_script, open(folded_path, "w") as folded:
            collapsed = subprocess.run(
                ["perl", stackcollapse],
                stdin=perf_script,
                stdout=folded,
                stderr=subprocess.PIPE,
                text=True,
                timeout=180,
            )
        if (
            collapsed.returncode != 0
            or not os.path.exists(folded_path)
            or os.path.getsize(folded_path) == 0
        ):
            return False, f"stackcollapse-perf failed: {collapsed.stderr[:300]}"

        with open(folded_path) as folded, open(svg_path, "w") as svg:
            rendered = subprocess.run(
                [
                    "perl",
                    flamegraph,
                    "--title",
                    "BlockTreeCache CPU Flame Graph",
                    "--countname",
                    "samples",
                ],
                stdin=folded,
                stdout=svg,
                stderr=subprocess.PIPE,
                text=True,
                timeout=180,
            )
        if (
            rendered.returncode != 0
            or not os.path.exists(svg_path)
            or os.path.getsize(svg_path) == 0
        ):
            return False, f"flamegraph render failed: {rendered.stderr[:300]}"
        total_samples = 0
        unknown_samples = 0
        weighted_depth = 0
        with open(folded_path) as folded:
            for line in folded:
                stack, separator, count_text = line.rstrip().rpartition(" ")
                if not separator:
                    continue
                try:
                    count = int(count_text)
                except ValueError:
                    continue
                total_samples += count
                weighted_depth += count * (stack.count(";") + 1)
                if "[unknown]" in stack:
                    unknown_samples += count
        quality_path = os.path.join(output_dir, "stack_quality.txt")
        with open(quality_path, "w") as quality:
            quality.write(f"unwind_mode=dwarf,16384\n")
            quality.write(f"samples={total_samples}\n")
            quality.write(
                "unknown_sample_ratio="
                f"{unknown_samples / total_samples if total_samples else 1.0:.6f}\n"
            )
            quality.write(
                "average_stack_depth="
                f"{weighted_depth / total_samples if total_samples else 0.0:.3f}\n"
            )
        return True, "flamegraph.svg generated"
    except (OSError, subprocess.SubprocessError) as error:
        return False, str(error)
    finally:
        if os.path.exists(perf_script_path):
            os.unlink(perf_script_path)


def run_perf_record(
    binary: str,
    subcommand: str,
    params: Dict[str, str],
    output_dir: str,
    cuda_device: int,
    max_device_memory_fraction: float,
    seed: int,
    model_profile: str,
    frequency: int,
    flamegraph_tools: Tuple[str, str],
    process_timeout_seconds: int = 180,
) -> Tuple[bool, str]:
    """Attach perf to the native process only during its measured window.

    Every runner prints PROFILE_ATTACH_READY and then leaves a 2s attach
    window before MEASURE_START, keeping profiler startup outside measurement.
    """
    perf_data = os.path.join(output_dir, "perf.data")
    native_cmd = build_native_command(
        binary,
        subcommand,
        params,
        "",
        cuda_device,
        max_device_memory_fraction,
        seed,
        0,
        model_profile,
    )

    # Save command
    perf_cmd_path = os.path.join(output_dir, "perf_command.txt")
    with open(perf_cmd_path, "w") as f:
        f.write(" ".join(native_cmd) + "\n")

    native = None
    perf_proc = None
    try:
        native = subprocess.Popen(
            native_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Wait for the common attach marker with a generous setup+warmup timeout.
        ready = False
        deadline = time.time() + 900
        for raw_line in native.stdout:
            if time.time() > deadline:
                break
            if b"PROFILE_ATTACH_READY" in raw_line:
                ready = True
                break
        if not ready:
            return False, "native process did not announce a profiler attach marker"

        # Attach perf to the native pid; it samples until we terminate it.
        perf_cmd = [
            "perf",
            "record",
            "-F",
            str(frequency),
            "-g",
            "--call-graph",
            "dwarf,16384",
            "-p",
            str(native.pid),
            "-o",
            perf_data,
        ]
        perf_proc = subprocess.Popen(
            perf_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Wait for the native process to finish its measured window and exit.
        native.wait(timeout=process_timeout_seconds)
        native_rc = native.returncode
        native_stderr = native.stderr.read().decode()[:500] if native.stderr else ""

        # Stop perf; SIGTERM makes perf write out perf.data.
        if perf_proc.poll() is None:
            perf_proc.terminate()
            perf_proc.wait(timeout=15)
        _, perf_err = perf_proc.communicate() if perf_proc.stdout else (b"", b"")
        if native_rc != 0:
            return False, f"native benchmark failed rc={native_rc}: {native_stderr}"

        # Verify perf.data
        if not os.path.exists(perf_data) or os.path.getsize(perf_data) == 0:
            return False, "perf.data is empty (attach may have failed)"

        # Generate summary
        summary = subprocess.run(
            ["perf", "report", "--stdio", "--no-children", "--input", perf_data],
            capture_output=True,
            text=True,
            timeout=60,
        )
        summary_path = os.path.join(output_dir, "perf_summary.txt")
        lines = summary.stdout.split("\n")[:40]
        with open(summary_path, "w") as f:
            f.write("\n".join(lines))

        flamegraph_ok, flamegraph_summary = generate_flamegraph(
            perf_data, output_dir, flamegraph_tools
        )
        if not flamegraph_ok:
            return False, flamegraph_summary
        quality_path = os.path.join(output_dir, "stack_quality.txt")
        if os.path.exists(quality_path):
            with open(quality_path) as quality, open(summary_path, "a") as summary_file:
                summary_file.write("\n\nStack quality\n")
                summary_file.write(quality.read())
        return True, "\n".join(lines)
    except subprocess.TimeoutExpired:
        if native is not None:
            try:
                native.kill()
            except Exception:
                pass
        return False, "native benchmark timed out before perf attach completed"
    except FileNotFoundError:
        return False, "perf not found"
    except Exception as e:
        return False, str(e)
    finally:
        if perf_proc is not None and perf_proc.poll() is None:
            try:
                perf_proc.kill()
            except Exception:
                pass
        if native is not None and native.poll() is None:
            try:
                native.kill()
            except Exception:
                pass


def run_perf_stat(
    binary: str,
    subcommand: str,
    params: Dict[str, str],
    output_dir: str,
    cuda_device: int,
    max_device_memory_fraction: float,
    seed: int,
    model_profile: str,
) -> Tuple[bool, str]:
    """Run a single perf stat session."""
    perf_stat_path = os.path.join(output_dir, "perf_stat.txt")
    native_cmd = build_native_command(
        binary,
        subcommand,
        params,
        "",
        cuda_device,
        max_device_memory_fraction,
        seed,
        0,
        model_profile,
    )
    cmd = [
        "perf",
        "stat",
        "--output",
        perf_stat_path,
        "--",
    ] + native_cmd

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.communicate(timeout=600)
        if os.path.exists(perf_stat_path) and os.path.getsize(perf_stat_path) > 0:
            with open(perf_stat_path) as f:
                return True, f.read()[:500]
        return False, "perf stat produced no output"
    except Exception as e:
        return False, str(e)


def sample_nvidia_smi(output_dir: str, label: str) -> Optional[str]:
    """Sample nvidia-smi metrics and save to file."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            smi_path = os.path.join(output_dir, f"nvidia_smi_{label}.txt")
            with open(smi_path, "w") as f:
                f.write(result.stdout)
            return result.stdout
    except Exception:
        pass
    return None


def sample_vmstat(output_dir: str, label: str) -> Optional[str]:
    """Sample /proc/vmstat dirty/writeback water levels and save to file."""
    keys = [
        "nr_dirty",
        "nr_writeback",
        "nr_file_pages",
        "nr_dirty_threshold",
        "nr_dirty_background_threshold",
        "pgpgin",
        "pgpgout",
    ]
    try:
        stats = {}
        with open("/proc/vmstat") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[0] in keys:
                    stats[parts[0]] = parts[1]
        if not stats:
            return None
        vm_path = os.path.join(output_dir, f"vmstat_{label}.txt")
        with open(vm_path, "w") as f:
            for k in keys:
                if k in stats:
                    f.write(f"{k} {stats[k]}\n")
        return str(stats)
    except Exception:
        return None


def validate_result_json(
    json_path: str, process_start_ns: int, expected_runner: str
) -> Tuple[bool, str]:
    """Validate only driver-owned result envelope and freshness invariants."""
    if not os.path.exists(json_path):
        return False, "result.json is missing"
    if os.stat(json_path).st_mtime_ns < process_start_ns:
        return False, "result.json predates this repetition"
    try:
        with open(json_path) as source:
            data = json.load(source)
        expected = {
            "schema_version": 1,
            "component": "BlockTreeCache",
            "binary": "block_tree_cache_gpu_benchmark",
            "runner": expected_runner,
            "status": "completed",
        }
        for field, value in expected.items():
            if data.get(field) != value:
                return (
                    False,
                    f"result.json {field} is {data.get(field)!r}, expected {value!r}",
                )
        measured_ns = data.get("phases_ns", {}).get("measured")
        if not isinstance(measured_ns, (int, float)) or measured_ns <= 0:
            return False, "result.json has no non-empty measured window"
        return True, ""
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        return False, str(error)


def sync_disk_path(path: str) -> float:
    """Drain buffered writes for one repetition and return elapsed seconds."""
    start = time.monotonic()
    subprocess.run(["sync", "-f", path], check=True, timeout=600)
    return time.monotonic() - start


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_environment(
    binary: str, model_profile: str, disk_root: Optional[str], cuda_device: int
) -> Dict:
    environment = {
        "binary_sha256": sha256_file(binary),
        "model_profile_sha256": sha256_file(model_profile),
        "collected_at": datetime.now().astimezone().isoformat(),
    }
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            str(cuda_device),
            "--query-gpu=name,memory.total,driver_version,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    if gpu_query.returncode == 0:
        environment["gpu"] = gpu_query.stdout.strip()

    cpu_query = subprocess.run(["lscpu"], capture_output=True, text=True)
    if cpu_query.returncode == 0:
        cpu_fields = {}
        for line in cpu_query.stdout.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                cpu_fields[key.strip()] = value.strip()
        cpu_parts = [
            cpu_fields.get("Model name"),
            (
                f"logical_cpus={cpu_fields.get('CPU(s)')}"
                if cpu_fields.get("CPU(s)")
                else None
            ),
            (
                f"sockets={cpu_fields.get('Socket(s)')}"
                if cpu_fields.get("Socket(s)")
                else None
            ),
            (
                f"cores_per_socket={cpu_fields.get('Core(s) per socket')}"
                if cpu_fields.get("Core(s) per socket")
                else None
            ),
        ]
        environment["cpu"] = ", ".join(part for part in cpu_parts if part)

    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemTotal:"):
                    environment["memory"] = " ".join(line.split()[1:])
                    break
    except OSError:
        pass

    uname_query = subprocess.run(["uname", "-srmo"], capture_output=True, text=True)
    if uname_query.returncode == 0:
        environment["kernel"] = uname_query.stdout.strip()
    if disk_root:
        # Record the mount as seen by the benchmark process. In Docker this is
        # the container mount namespace; do not expose the huge overlay
        # lowerdir/upperdir option list or pretend that it identifies the host
        # physical block device.
        environment["disk_scope"] = (
            "benchmark process mount namespace (container-visible)"
            if os.path.exists("/.dockerenv")
            else "benchmark process mount namespace"
        )
        environment["disk_target"] = disk_root
        disk_query = subprocess.run(
            ["findmnt", "-no", "TARGET,SOURCE,FSTYPE", "--target", disk_root],
            capture_output=True,
            text=True,
        )
        if disk_query.returncode == 0:
            mount_fields = disk_query.stdout.split()
            if len(mount_fields) >= 3:
                environment["disk_mount"] = (
                    f"target={mount_fields[0]}, source={mount_fields[1]}, "
                    f"fstype={mount_fields[2]}"
                )
        else:
            disk_query = subprocess.run(
                ["df", "--output=source,fstype", disk_root],
                capture_output=True,
                text=True,
            )
            if disk_query.returncode == 0:
                environment["disk_mount"] = "source/fstype=" + " ".join(
                    disk_query.stdout.splitlines()[-1].split()
                )
        capacity_query = subprocess.run(
            ["df", "-h", "--output=size,used,avail,pcent", disk_root],
            capture_output=True,
            text=True,
        )
        if capacity_query.returncode == 0:
            capacity_fields = capacity_query.stdout.splitlines()[-1].split()
            if len(capacity_fields) == 4:
                environment["disk_capacity"] = (
                    f"size={capacity_fields[0]}, used={capacity_fields[1]}, "
                    f"available={capacity_fields[2]}, use={capacity_fields[3]}"
                )

    # The driver normally runs from Bazel runfiles, where __file__ resolves
    # under the Bazel execroot rather than the source checkout. Locate the
    # lexical checkout containing bazel-bin (or use the caller's cwd) so the
    # recorded commit describes the code that produced the binary.
    binary_path = Path(os.path.abspath(binary))
    git_candidates = [Path(os.getcwd())]
    git_candidates.extend(binary_path.parents)
    seen_candidates = set()
    for candidate in git_candidates:
        candidate_text = str(candidate)
        if candidate_text in seen_candidates or not (candidate / ".git").exists():
            continue
        seen_candidates.add(candidate_text)
        git_query = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=candidate_text,
            capture_output=True,
            text=True,
        )
        if git_query.returncode != 0:
            continue
        environment["code_commit"] = git_query.stdout.strip()
        dirty_query = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=candidate_text,
            capture_output=True,
            text=True,
        )
        if dirty_query.returncode == 0:
            environment["code_dirty"] = "true" if dirty_query.stdout else "false"
        break
    return environment


def load_profile_group_set_payloads(model_profile: str) -> Dict[str, int]:
    """Load group set payload bytes from the model profile JSON."""
    try:
        with open(model_profile) as f:
            data = json.load(f)
        payloads = {}
        for gs in data.get("group_sets", []):
            payloads[gs["name"]] = int(gs.get("payload_bytes", 0))
        return payloads
    except Exception:
        return {}


def run_case(
    case: BenchmarkCase,
    binary: str,
    model_profile: str,
    output_dir: str,
    cuda_device: int,
    max_device_memory_fraction: float,
    base_seed: int,
    repetitions: int,
    case_timeout: int,
    grace_seconds: int,
    perf_mode: str,
    perf_frequency: int,
    flamegraph_tools: Optional[Tuple[str, str]],
    disk_root: Optional[str],
    profile_payloads: Optional[Dict[str, int]] = None,
    force_perf: bool = False,
) -> Dict:
    """Run a single benchmark case with all repetitions."""
    case_dir = os.path.join(output_dir, case.suite, case.name)
    os.makedirs(case_dir, exist_ok=True)

    # Native process timeout: case metadata overrides the driver default so
    # the online Tree case covers setup + 15s warmup + 60s measured + drain
    # + profiler teardown.
    process_timeout = (
        case.expected_process_timeout_seconds
        if case.expected_process_timeout_seconds
        else case_timeout
    )

    manifest = {
        "case": case.name,
        "suite": case.suite,
        "subcommand": case.subcommand,
        "status": "unknown",
        "repetitions": [],
        "perf": {"status": "skipped"},
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "process_timeout_seconds": process_timeout,
    }

    # Check disk requirement
    if case.requires_disk and not disk_root:
        manifest["status"] = "skipped_no_disk"
        manifest["end_time"] = datetime.now().isoformat()
        with open(os.path.join(case_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest

    # Build resolved params. Each repetition receives its own disk path below.
    params = dict(case.params)

    # Scale transfer operation count to meet the minimum logical bytes target
    if case.min_logical_bytes > 0 and params.get("--transfer-operation-count"):
        group_set = params.get("--group-set", "full_context")
        payload = (profile_payloads or {}).get(group_set, 0)
        if payload > 0:
            min_ops = math.ceil(case.min_logical_bytes / payload)
            requested = int(params["--transfer-operation-count"])
            scaled = max(requested, min_ops)
            params["--transfer-operation-count"] = str(scaled)
            manifest["resolved_transfer_operation_count"] = scaled
            manifest["min_logical_bytes_target"] = case.min_logical_bytes
            manifest["group_payload_bytes"] = payload

    # Run repetitions
    for rep in range(repetitions if case.suite == "profile" else 1):
        rep_dir = os.path.join(case_dir, f"rep_{rep:04d}")
        os.makedirs(rep_dir, exist_ok=True)

        output_json = os.path.join(rep_dir, "result.json")
        if os.path.exists(output_json):
            os.unlink(output_json)

        rep_params = dict(params)
        rep_disk_dir = None
        if case.requires_disk:
            rep_disk_dir = os.path.join(
                disk_root, f"benchmark_{case.name}", f"rep_{rep:04d}"
            )
            if os.path.isdir(rep_disk_dir):
                shutil.rmtree(rep_disk_dir)
            os.makedirs(rep_disk_dir)
            rep_params["--disk-path"] = rep_disk_dir

        sample_nvidia_smi(rep_dir, "before")
        sample_vmstat(rep_dir, "before")
        process_start_ns = time.time_ns()
        native_cmd = build_native_command(
            binary,
            case.subcommand,
            rep_params,
            output_json,
            cuda_device,
            max_device_memory_fraction,
            base_seed + rep,
            rep,
            model_profile,
        )
        command_path = os.path.join(rep_dir, "command.txt")
        with open(command_path, "w") as command_file:
            command_file.write(" ".join(map(shlex.quote, native_cmd)) + "\n")
        exit_code, stdout, stderr = run_native_process(
            binary,
            case.subcommand,
            rep_params,
            output_json,
            cuda_device,
            max_device_memory_fraction,
            base_seed + rep,
            rep,
            model_profile,
            process_timeout,
            grace_seconds,
        )

        drain_seconds = 0.0
        drain_error = ""
        if rep_disk_dir and params.get("--disk-io-mode") == "buffered":
            try:
                drain_seconds = sync_disk_path(rep_disk_dir)
            except (OSError, subprocess.SubprocessError) as error:
                drain_error = str(error)
        sample_nvidia_smi(rep_dir, "after")
        sample_vmstat(rep_dir, "after")

        rep_manifest = {
            "repetition": rep,
            "seed": base_seed + rep,
            "command_file": command_path,
            "exit_code": exit_code,
            "stdout_file": os.path.join(rep_dir, "stdout.txt"),
            "stderr_file": os.path.join(rep_dir, "stderr.txt"),
            "result_json": output_json if os.path.exists(output_json) else None,
            "valid": False,
            "status": "failed",
            "disk_dir": rep_disk_dir,
            "drain_seconds": drain_seconds,
        }

        # Save stdout/stderr
        with open(os.path.join(rep_dir, "stdout.txt"), "w") as f:
            f.write(stdout)
        with open(os.path.join(rep_dir, "stderr.txt"), "w") as f:
            f.write(stderr)

        # Validate result
        result_valid, validation_error = validate_result_json(
            output_json, process_start_ns, case.subcommand
        )
        if exit_code == 0 and result_valid and not drain_error:
            rep_manifest["valid"] = True
            rep_manifest["status"] = "completed"
        else:
            # Write failure manifest
            fail_manifest = {
                "status": "failed",
                "exit_code": exit_code,
                "error": drain_error
                or validation_error
                or (stderr[:500] if stderr else "unknown"),
            }
            with open(os.path.join(rep_dir, "manifest.json"), "w") as f:
                json.dump(fail_manifest, f, indent=2)

        manifest["repetitions"].append(rep_manifest)
        if rep_disk_dir and os.path.isdir(rep_disk_dir):
            shutil.rmtree(rep_disk_dir)
            rep_manifest["disk_dir_cleaned"] = True

    # Perf collection: representative cases in full suite mode, or any case in
    # single-case mode (design 9.1). Smoke never collects perf by default.
    perf_mode_actual = perf_mode
    if case.suite == "smoke":
        perf_mode_actual = "off"  # No perf for smoke by default

    if perf_mode_actual != "off" and (force_perf or case.is_representative_perf):
        perf_dir = os.path.join(case_dir, "perf")
        os.makedirs(perf_dir, exist_ok=True)
        perf_params = dict(params)
        perf_disk_dir = None
        if case.requires_disk:
            perf_disk_dir = os.path.join(disk_root, f"benchmark_{case.name}", "perf")
            if os.path.isdir(perf_disk_dir):
                shutil.rmtree(perf_disk_dir)
            os.makedirs(perf_disk_dir)
            perf_params["--disk-path"] = perf_disk_dir

        if perf_mode_actual == "record":
            if flamegraph_tools is None:
                perf_ok, perf_summary = False, "FlameGraph tools not found"
            else:
                perf_ok, perf_summary = run_perf_record(
                    binary,
                    case.subcommand,
                    perf_params,
                    perf_dir,
                    cuda_device,
                    max_device_memory_fraction,
                    base_seed,
                    model_profile,
                    perf_frequency,
                    flamegraph_tools,
                    process_timeout,
                )
        else:
            perf_ok, perf_summary = run_perf_stat(
                binary,
                case.subcommand,
                perf_params,
                perf_dir,
                cuda_device,
                max_device_memory_fraction,
                base_seed,
                model_profile,
            )

        manifest["perf"] = {
            "status": "ok" if perf_ok else "failed",
            "mode": perf_mode_actual,
            "summary": perf_summary[:200] if perf_summary else "",
        }
        artifact_names = {
            "perf_data": "perf.data",
            "summary": "perf_summary.txt",
            "folded": "perf.folded",
            "flamegraph": "flamegraph.svg",
            "stack_quality": "stack_quality.txt",
        }
        manifest["perf"]["artifacts"] = {
            key: os.path.join("perf", filename)
            for key, filename in artifact_names.items()
            if os.path.exists(os.path.join(perf_dir, filename))
        }
        if perf_disk_dir and os.path.isdir(perf_disk_dir):
            try:
                if params.get("--disk-io-mode") == "buffered":
                    sync_disk_path(perf_disk_dir)
            except (OSError, subprocess.SubprocessError) as error:
                manifest["perf"]["status"] = "failed"
                manifest["perf"]["summary"] = str(error)[:200]
            shutil.rmtree(perf_disk_dir)

    # Compute overall status
    all_valid = all(r["valid"] for r in manifest["repetitions"])
    any_valid = any(r["valid"] for r in manifest["repetitions"])
    perf_required_failed = manifest["perf"]["status"] == "failed"
    if perf_required_failed:
        manifest["status"] = "failed"
    elif all_valid:
        manifest["status"] = "completed"
    elif any_valid:
        manifest["status"] = "partial"
    else:
        manifest["status"] = "failed"

    manifest["end_time"] = datetime.now().isoformat()

    # Write case manifest
    manifest_path = os.path.join(case_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def apply_task_pool_overrides(
    cases: List[BenchmarkCase], task_pool_sizes: Optional[List[int]]
) -> List[BenchmarkCase]:
    if not task_pool_sizes:
        return cases
    expanded_cases = []
    for case in cases:
        if case.subcommand != "tree":
            expanded_cases.append(case)
            continue
        for task_pool_size in task_pool_sizes:
            params = dict(case.params)
            params["--task-pool-size"] = str(task_pool_size)
            expanded_cases.append(
                replace(
                    case,
                    name=f"{case.name}_tp{task_pool_size}",
                    params=params,
                )
            )
    return expanded_cases


def run_suite(
    suite: str,
    case_name: str,
    binary: str,
    model_profile: str,
    output_dir: str,
    cuda_device: int,
    max_device_memory_fraction: float,
    base_seed: int,
    task_pool_sizes: Optional[List[int]],
    repetitions: int,
    case_timeout: int,
    grace_seconds: int,
    perf_mode: str,
    perf_frequency: int,
    flamegraph_tools: Optional[Tuple[str, str]],
    disk_root: Optional[str],
    allow_incomplete: bool = False,
):
    """Run all cases in a suite."""
    if case_name == "all":
        cases = get_suite_cases(suite)
    else:
        case_names = [name.strip() for name in case_name.split(",") if name.strip()]
        unknown = [name for name in case_names if name not in CASE_REGISTRY]
        if unknown:
            print(f"Error: unknown case(s): {', '.join(unknown)}")
            sys.exit(1)
        cases = [CASE_REGISTRY[name] for name in case_names]
        wrong_suite = [
            case.name for case in cases if suite != "all" and case.suite != suite
        ]
        if wrong_suite:
            print(f"Error: case(s) not in suite '{suite}': {', '.join(wrong_suite)}")
            sys.exit(1)

    cases = apply_task_pool_overrides(cases, task_pool_sizes)

    print(f"Running {len(cases)} case(s) in suite '{suite}'")
    print(f"Output directory: {output_dir}")

    # If single case mode, perf collects on the specified case fully
    force_perf = case_name != "all"

    profile_payloads = load_profile_group_set_payloads(model_profile)
    all_manifests = []
    for case in cases:
        rep_count = 1 if case.suite == "smoke" else repetitions
        print(f"  Running: {case.name} ({rep_count} rep(s))...", end=" ", flush=True)

        case_manifest = run_case(
            case=case,
            binary=binary,
            model_profile=model_profile,
            output_dir=output_dir,
            cuda_device=cuda_device,
            max_device_memory_fraction=max_device_memory_fraction,
            base_seed=base_seed,
            repetitions=rep_count,
            case_timeout=case_timeout,
            grace_seconds=grace_seconds,
            perf_mode=perf_mode,
            perf_frequency=perf_frequency,
            flamegraph_tools=flamegraph_tools,
            disk_root=disk_root,
            profile_payloads=profile_payloads,
            force_perf=force_perf,
        )
        all_manifests.append(case_manifest)
        print(f"[{case_manifest['status']}]")

    # Write suite manifest
    suite_manifest = {
        "suite": suite,
        "binary": binary,
        "model_profile": model_profile,
        "cuda_device": cuda_device,
        "max_device_memory_fraction": max_device_memory_fraction,
        "environment": collect_environment(
            binary, model_profile, disk_root, cuda_device
        ),
        "invocation": {
            "suite": suite,
            "case": case_name,
            "process_repetitions": repetitions,
            "base_seed": base_seed,
            "task_pool_sizes": task_pool_sizes,
            "case_timeout_seconds": case_timeout,
            "termination_grace_seconds": grace_seconds,
            "perf": perf_mode,
            "perf_frequency": perf_frequency,
            "flamegraph_tools_dir": (
                os.path.dirname(flamegraph_tools[0]) if flamegraph_tools else None
            ),
            "output_dir": output_dir,
            "disk_root": disk_root,
            "allow_incomplete": allow_incomplete,
        },
        "total_cases": len(cases),
        "canonical_total_cases": len(get_suite_cases(suite)),
        "completed": sum(1 for m in all_manifests if m["status"] == "completed"),
        "partial": sum(1 for m in all_manifests if m["status"] == "partial"),
        "failed": sum(1 for m in all_manifests if m["status"] == "failed"),
        "skipped": sum(1 for m in all_manifests if m["status"] == "skipped_no_disk"),
        "cases": all_manifests,
    }

    suite_dir = os.path.join(output_dir, suite)
    os.makedirs(suite_dir, exist_ok=True)
    suite_manifest_path = os.path.join(suite_dir, "suite_manifest.json")
    with open(suite_manifest_path, "w") as f:
        json.dump(suite_manifest, f, indent=2)

    print(
        f"\nSuite '{suite}' complete: "
        f"{suite_manifest['completed']} completed, "
        f"{suite_manifest['partial']} partial, "
        f"{suite_manifest['failed']} failed, "
        f"{suite_manifest['skipped']} skipped"
    )

    # Canonical suites are strict by default: failed, partial and skipped are
    # all incomplete outcomes.
    if suite_manifest["failed"] > 0:
        return 1
    if not allow_incomplete and (
        suite_manifest["partial"] > 0 or suite_manifest["skipped"] > 0
    ):
        return 1
    return 0


def resolve_runfile_path(name: str) -> Optional[str]:
    """Locate a data file inside the Bazel runfiles tree."""
    runfiles_dir = os.environ.get("RUNFILES_DIR") or os.environ.get("TEST_SRCDIR")
    if runfiles_dir:
        for workspace in ("rtp_llm", "github-opensource"):
            candidate = os.path.join(
                runfiles_dir,
                workspace,
                "rtp_llm/cpp/cache/block_tree_cache/benchmark",
                name,
            )
            if os.path.exists(candidate):
                return candidate
    # Fallback: same directory as this script
    candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    if os.path.exists(candidate):
        return candidate
    return None


def main():
    args = parse_args()
    args.output_dir = os.path.abspath(args.output_dir)
    if args.disk_root:
        args.disk_root = os.path.abspath(args.disk_root)

    for name, value in [
        ("--process-repetitions", args.process_repetitions),
        ("--case-timeout-seconds", args.case_timeout_seconds),
        ("--termination-grace-seconds", args.termination_grace_seconds),
        ("--perf-frequency", args.perf_frequency),
    ]:
        if value <= 0:
            raise SystemExit(f"Error: {name} must be positive")
    if args.cuda_device < 0:
        raise SystemExit("Error: --cuda-device must be non-negative")
    if not math.isfinite(args.max_device_memory_fraction) or not (
        0.0 < args.max_device_memory_fraction <= 1.0
    ):
        raise SystemExit("Error: --max-device-memory-fraction must be in (0, 1]")
    if args.seed < 0:
        raise SystemExit("Error: --seed must be non-negative")
    if args.task_pool_sizes:
        if any(size <= 0 for size in args.task_pool_sizes):
            raise SystemExit("Error: --task-pool-size must be positive")
        args.task_pool_sizes = list(dict.fromkeys(args.task_pool_sizes))

    # Resolve binary path (explicit arg wins, else runfiles auto-detect)
    binary = args.binary
    if binary is None:
        binary = resolve_runfile_path("block_tree_cache_gpu_benchmark")
        if binary:
            print(f"Auto-detected benchmark binary: {binary}")

    # Resolve model profile path
    model_profile = args.model_profile
    if model_profile is None:
        model_profile = resolve_runfile_path(
            "profiles/deepseek_v4_pro_fp8_tp1_cp1.json"
        )
        if model_profile:
            print(f"Auto-detected model profile: {model_profile}")

    # Check binary exists
    if not binary or not os.path.exists(binary):
        print(f"Error: benchmark binary not found (pass --binary explicitly): {binary}")
        sys.exit(1)

    # Check model profile exists
    if not model_profile or not os.path.exists(model_profile):
        print(
            f"Error: model profile not found (pass --model-profile explicitly): {model_profile}"
        )
        sys.exit(1)

    # Check perf availability
    flamegraph_tools = None
    if args.perf != "off":
        perf_check = subprocess.run(["which", "perf"], capture_output=True)
        if perf_check.returncode != 0:
            print("Warning: 'perf' not found, disabling perf collection")
            args.perf = "off"
        elif args.perf == "record":
            flamegraph_tools = resolve_flamegraph_tools(args.flamegraph_tools_dir)
            if flamegraph_tools is None:
                print(
                    "Error: perf record requires stackcollapse-perf.pl and flamegraph.pl; "
                    "pass --flamegraph-tools-dir",
                    file=sys.stderr,
                )
                sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run the suite
    exit_code = run_suite(
        suite=args.suite,
        case_name=args.case,
        binary=binary,
        model_profile=model_profile,
        output_dir=args.output_dir,
        cuda_device=args.cuda_device,
        max_device_memory_fraction=args.max_device_memory_fraction,
        base_seed=args.seed,
        task_pool_sizes=args.task_pool_sizes,
        repetitions=args.process_repetitions,
        case_timeout=args.case_timeout_seconds,
        grace_seconds=args.termination_grace_seconds,
        perf_mode=args.perf,
        perf_frequency=args.perf_frequency,
        flamegraph_tools=flamegraph_tools,
        disk_root=args.disk_root,
        allow_incomplete=args.allow_incomplete,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

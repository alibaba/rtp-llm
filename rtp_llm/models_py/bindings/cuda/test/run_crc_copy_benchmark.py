"""Manual Bazel test entry point; preserve raw data even when a child fails."""

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from crc_copy_benchmark_stats import analyze, write_results


DIAGNOSTIC_VARIANTS = (
    "integrated_crc",
    "copy1d_batch",
    "copy3d_batch",
    "staged_no_crc",
    "gather_control",
)


def validate_diagnostic(path, args, repeat):
    """Keep diagnostic records out of the formal performance analyzer."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not records or any(
        not str(record.get("type", "")).startswith("diagnostic_")
        for record in records
    ):
        raise ValueError(f"{path.name}: contains non-diagnostic records")
    metadata = [r for r in records if r["type"] == "diagnostic_metadata"]
    complete = [r for r in records if r["type"] == "diagnostic_complete"]
    if (
        len(metadata) != 1
        or metadata[0].get("formal_performance_data") is not False
        or len(complete) != 1
        or complete[0].get("success") is not True
        or records[-1] != complete[0]
    ):
        raise ValueError(f"{path.name}: missing successful diagnostic completion")
    expected_metadata = {
        "layout": args.diagnostic_layout,
        "direction": args.diagnostic_direction,
        "variant": args.diagnostic_variant,
        "blocks": args.diagnostic_blocks,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "repeat": repeat,
        "seed": args.seed + repeat,
        "requested_main_cpu": (
            args.diagnostic_main_cpu
            if args.diagnostic_main_cpu is not None
            else -1
        ),
    }
    if any(metadata[0].get(key) != value for key, value in expected_metadata.items()):
        raise ValueError(f"{path.name}: diagnostic metadata differs from request")
    variants = [
        variant
        for variant in DIAGNOSTIC_VARIANTS
        if args.diagnostic_variant in ("all", variant)
        and not (
            args.exclude_1d_h2d
            and args.diagnostic_direction == "h2d"
            and variant == "copy1d_batch"
        )
    ]
    checks = [r for r in records if r["type"] == "diagnostic_correctness"]
    if (
        len(checks) != len(variants)
        or {r.get("variant") for r in checks} != set(variants)
        or any(
            r.get("success") is not True or r.get("rotations_checked") != [0, 1]
            for r in checks
        )
    ):
        raise ValueError(f"{path.name}: incomplete diagnostic correctness checks")
    if "integrated_crc" in variants and not any(
        r["type"] == "diagnostic_crc_corruption" and r.get("success") is True
        for r in records
    ):
        raise ValueError(f"{path.name}: missing CRC corruption check")
    samples = [r for r in records if r["type"] == "diagnostic_sample"]
    expected_count = 0 if args.correctness_only else args.iterations * len(variants)
    if len(samples) != expected_count:
        raise ValueError(
            f"{path.name}: expected {expected_count} diagnostic samples, got {len(samples)}"
        )
    if args.correctness_only:
        if complete[0].get("correctness_only") is not True:
            raise ValueError(f"{path.name}: incorrect correctness-only completion")
    elif complete[0].get("samples") != expected_count:
        raise ValueError(f"{path.name}: diagnostic completion sample count differs")
    seen = set()
    positions = {}
    expected_layout = (
        "full" if args.diagnostic_layout == "full" else "prefill_cp8_no_spec_swa"
    )
    for sample in samples:
        key = (sample["round"], sample["variant"])
        if (
            key in seen
            or not 0 <= sample["round"] < args.iterations
            or sample["variant"] not in variants
            or sample["repeat"] != repeat
            or sample["blocks"] != args.diagnostic_blocks
            or sample["layout"] != expected_layout
            or sample["direction"] != args.diagnostic_direction
            or not 0 <= sample["position"] < len(variants)
        ):
            raise ValueError(f"{path.name}: invalid/duplicate diagnostic sample key")
        seen.add(key)
        positions.setdefault(sample["round"], set()).add(sample["position"])
        if (
            any(
                not math.isfinite(sample[field]) or sample[field] < 0
                for field in ("wall_us", "thread_cpu_us")
            )
            or sample["end_monotonic_ns"] < sample["begin_monotonic_ns"]
            or sample["voluntary_context_switches"] < 0
            or sample["involuntary_context_switches"] < 0
        ):
            raise ValueError(f"{path.name}: invalid diagnostic timing")
    if any(len(values) != len(variants) for values in positions.values()):
        raise ValueError(f"{path.name}: duplicate candidate position within a round")
    return {"repeat": repeat, "samples": len(samples), "variants": variants}


def profile_artifacts(output, stem):
    result = []
    for path in sorted(output.glob(f"{stem}.profile*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            result.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
    return result


def select_cpu(request):
    if request == "none":
        return None
    allowed = os.sched_getaffinity(0)
    if request == "auto":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        if len(visible) != 1 or not visible[0].strip():
            raise ValueError(
                "Expose exactly one GPU, normally with --run_under=//rtp_llm/test/utils:gpu_lock"
            )
        bus = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    visible[0],
                    "--query-gpu=pci.bus_id",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            .strip()
            .lower()
        )
        domain, rest = bus.split(":", 1)
        ranges = (
            (
                Path("/sys/bus/pci/devices")
                / (f"{int(domain, 16):04x}:" + rest)
                / "local_cpulist"
            )
            .read_text()
            .strip()
            .split(",")
        )
        near = set()
        for item in ranges:
            limits = item.split("-")
            near.update(range(int(limits[0]), int(limits[-1]) + 1))
        candidates = allowed & near
        if not candidates:
            raise ValueError(
                "No allowed CPU local to the selected GPU; use --cpu=<index> or --cpu=none explicitly"
            )
        first = ranges[0].split("-")
        first_range = set(range(int(first[0]), int(first[-1]) + 1))
        cpu = max(candidates & first_range or candidates)
    else:
        cpu = int(request)
    if cpu not in allowed:
        raise ValueError(f"CPU {cpu} is outside this process's allowed affinity")
    os.sched_setaffinity(0, {cpu})
    return cpu


def source_provenance():
    result = {}
    for field, variable in (
        ("source_commit", "CRC_BENCH_SOURCE_COMMIT"),
        ("base_commit", "CRC_BENCH_BASE_COMMIT"),
    ):
        value = os.environ.get(variable, "")
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError(
                f"Set --test_env={variable}=<full 40-character commit SHA>; "
                "the benchmark must identify the tested source and main baseline"
            )
        result[field] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument(
        "--cpu", default="auto", help="NUMA-local CPU (auto), a CPU index, or none"
    )
    parser.add_argument(
        "--exclude-1d-h2d",
        action="store_true",
        help="Explicitly mark 1D H2D unavailable; never treats it as passing",
    )
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--diagnostic-blocks", type=int, default=8)
    parser.add_argument("--diagnostic-layout", choices=("full", "swa"), default="full")
    parser.add_argument(
        "--diagnostic-direction", choices=("h2d", "d2h"), default="h2d"
    )
    parser.add_argument("--diagnostic-main-cpu", type=int)
    parser.add_argument(
        "--diagnostic-variant", choices=("all", *DIAGNOSTIC_VARIANTS), default="all"
    )
    parser.add_argument(
        "--diagnostic-nsys",
        action="store_true",
        help="Capture diagnostic repeats with Nsight Systems; keep reports with raw records",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "crc_copy_benchmark_results")
        ),
    )
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or not 0 <= args.seed <= 2**31 - 82:
        parser.error(
            "iterations must be positive, warmup nonnegative, seed in [0, 2147483566]"
        )
    if not 1 <= args.diagnostic_blocks <= 32:
        parser.error("diagnostic-blocks must be in [1, 32]")
    if args.diagnostic_main_cpu is not None and args.diagnostic_main_cpu < 0:
        parser.error("diagnostic-main-cpu must be nonnegative")
    if args.diagnostic_nsys and (not args.diagnostic or args.correctness_only):
        parser.error("diagnostic-nsys requires diagnostic timing (not correctness-only)")
    if (
        args.diagnostic
        and args.exclude_1d_h2d
        and args.diagnostic_direction == "h2d"
        and args.diagnostic_variant == "copy1d_batch"
    ):
        parser.error("diagnostic-variant copy1d_batch conflicts with exclude-1d-h2d")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(
        (output / name).exists()
        for name in (
            "run_manifest.json",
            "analysis.json",
            "copy_80.jsonl",
            "copy_81.jsonl",
            "diagnostic_80.jsonl",
            "diagnostic_81.jsonl",
        )
    ):
        parser.error(
            "Output directory contains an earlier run; choose an empty directory"
        )
    manifest = {
        "success": False,
        "status": "starting",
        "command": sys.argv,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "exclusions": ["copy1d_batch/h2d"] if args.exclude_1d_h2d else [],
    }
    settings = {
        key: getattr(args, key)
        for key in (
            "iterations",
            "warmup",
            "seed",
            "exclude_1d_h2d",
            "correctness_only",
        )
    }
    manifest["settings"] = settings
    if args.diagnostic:
        manifest.update(
            formal_performance_data=False,
            diagnostic=True,
            profiled=args.diagnostic_nsys,
            diagnostic_runs=[],
        )
        settings.update(
            {
                key: getattr(args, key)
                for key in (
                    "diagnostic_blocks",
                    "diagnostic_layout",
                    "diagnostic_direction",
                    "diagnostic_main_cpu",
                    "diagnostic_variant",
                    "diagnostic_nsys",
                )
            }
        )
    manifest_path = output / "run_manifest.json"
    try:
        binary = args.binary.resolve(strict=True)
        manifest["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        provenance = source_provenance()
        manifest.update(provenance)
        settings.update(provenance, binary_sha256=manifest["binary_sha256"])
        child_environment = dict(os.environ)
        child_environment["CRC_BENCH_BINARY_SHA256"] = manifest["binary_sha256"]
        manifest["cpu"] = select_cpu(args.cpu)
        manifest["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        nsys = None
        if args.diagnostic_nsys:
            nsys = shutil.which("nsys")
            if nsys is None:
                raise RuntimeError("--diagnostic-nsys requires nsys in PATH")
            version = subprocess.check_output(
                [nsys, "--version"], text=True, stderr=subprocess.STDOUT
            ).strip()
            manifest["nsys"] = {
                "executable": nsys,
                "version": version,
                "trace": ["cuda-sw", "osrt"],
                "sample": "none",
                "cpuctxsw": "process-tree",
                "capture_range": "cudaProfilerApi",
                "capture_range_end": "stop",
            }
        manifest["status"] = "running"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"CRC copy benchmark artifacts: {output}", flush=True)
        if args.exclude_1d_h2d:
            print(
                "EXPLICIT EXCLUSION: copy1d_batch/h2d will be unavailable, not reported as passing.",
                flush=True,
            )
        paths = []
        for repeat in (80, 81):
            stem = f"{'diagnostic' if args.diagnostic else 'copy'}_{repeat}"
            raw = output / f"{stem}.jsonl"
            command = [
                str(binary),
                "--iterations",
                str(args.iterations),
                "--warmup",
                str(args.warmup),
                "--repeat",
                str(repeat),
                "--seed",
                str(args.seed + repeat),
                "--output",
                str(raw),
            ]
            if args.exclude_1d_h2d:
                command.append("--exclude-1d-h2d")
            if args.correctness_only:
                command.append("--correctness-only")
            if args.diagnostic:
                command.append("--diagnostic")
                for key in ("blocks", "layout", "direction", "variant", "main_cpu"):
                    value = getattr(args, f"diagnostic_{key}")
                    if value is not None:
                        command += [f"--diagnostic-{key.replace('_', '-')}", str(value)]
                run = {"repeat": repeat, "raw": str(raw), "binary_command": command[:]}
                if nsys is not None:
                    # Force software CUDA tracing: hardware tracing on the
                    # tested Blackwell stack dropped all kernel timestamps.
                    command = [
                        nsys,
                        "profile",
                        "--trace=cuda-sw,osrt",
                        "--sample=none",
                        "--cpuctxsw=process-tree",
                        "--capture-range=cudaProfilerApi",
                        "--capture-range-end=stop",
                        "--force-overwrite=true",
                        f"--output={output / (stem + '.profile')}",
                        *command,
                    ]
                run.update(command=command[:], started_wall_ns=time.time_ns())
                manifest["diagnostic_runs"].append(run)
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(shlex.join(command), flush=True)
            stdout_path, stderr_path = (
                output / f"{stem}.stdout",
                output / f"{stem}.stderr",
            )
            with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                completed = subprocess.run(
                    command,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                    env=child_environment,
                )
            (output / f"{stem}.exitcode").write_text(
                str(completed.returncode) + "\n"
            )
            if args.diagnostic:
                run.update(
                    ended_wall_ns=time.time_ns(),
                    exitcode=completed.returncode,
                    stdout=str(stdout_path),
                    stderr=str(stderr_path),
                    profile_artifacts=profile_artifacts(output, stem),
                )
            if completed.returncode:
                print(stderr_path.read_text()[-12000:], file=sys.stderr)
                raise RuntimeError(
                    f"repeat {repeat} failed (exit {completed.returncode}); raw evidence retained, no performance summary produced"
                )
            paths.append(raw)
            if args.diagnostic:
                run["validation"] = validate_diagnostic(raw, args, repeat)
                if nsys is not None and not any(
                    item["path"].endswith(".nsys-rep")
                    for item in run["profile_artifacts"]
                ):
                    raise RuntimeError(
                        f"repeat {repeat}: nsys completed without an .nsys-rep artifact"
                    )
            print(f"repeat {repeat}: completed", flush=True)
        if args.diagnostic:
            manifest.update(
                success=True,
                status="passed",
                samples=sum(
                    run["validation"]["samples"] for run in manifest["diagnostic_runs"]
                ),
                formal_performance_data=False,
            )
            print(
                json.dumps(
                    {
                        key: manifest[key]
                        for key in ("status", "samples", "formal_performance_data", "profiled")
                    }
                ),
                flush=True,
            )
            print(f"Diagnostic manifest: {manifest_path}", flush=True)
            return 0
        result = analyze(paths, settings)
        write_results(result, output)
        manifest.update(
            success=True,
            status="passed",
            samples=result["sample_count"],
            paired_rounds=result["paired_round_count"],
            drift_warnings=len(result["warnings"]),
        )
        print(
            json.dumps(
                {
                    key: manifest[key]
                    for key in (
                        "status",
                        "samples",
                        "paired_rounds",
                        "drift_warnings",
                        "exclusions",
                    )
                }
            ),
            flush=True,
        )
        print(f"Summary: {output / 'summary.md'}", flush=True)
        return 0
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        IndexError,
        subprocess.SubprocessError,
    ) as error:
        manifest.update(status="failed", error=str(error))
        print(f"CRC copy benchmark FAILED: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())

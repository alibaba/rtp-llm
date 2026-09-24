"""Manual Bazel test entry point; preserve raw data even when a child fails."""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from crc_copy_benchmark_stats import analyze, write_results


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
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(
        (output / name).exists()
        for name in (
            "run_manifest.json",
            "analysis.json",
            "copy_80.jsonl",
            "copy_81.jsonl",
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
    manifest_path = output / "run_manifest.json"
    try:
        binary = args.binary.resolve(strict=True)
        manifest["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        manifest["cpu"] = select_cpu(args.cpu)
        manifest["cpu_affinity"] = sorted(os.sched_getaffinity(0))
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
            raw = output / f"copy_{repeat}.jsonl"
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
            print(shlex.join(command), flush=True)
            stdout_path, stderr_path = (
                output / f"copy_{repeat}.stdout",
                output / f"copy_{repeat}.stderr",
            )
            with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                completed = subprocess.run(
                    command, stdout=stdout, stderr=stderr, check=False
                )
            (output / f"copy_{repeat}.exitcode").write_text(
                str(completed.returncode) + "\n"
            )
            if completed.returncode:
                print(stderr_path.read_text()[-12000:], file=sys.stderr)
                raise RuntimeError(
                    f"repeat {repeat} failed (exit {completed.returncode}); raw evidence retained, no performance summary produced"
                )
            paths.append(raw)
            print(f"repeat {repeat}: completed", flush=True)
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

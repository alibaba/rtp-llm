"""Run two independent CRC segmentation A/B processes and retain their evidence."""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from crc_segmentation_benchmark_stats import REPEAT_IDS, analyze, write_results


def select_cpu(request):
    # Match the existing copy benchmark's NUMA policy without importing its
    # five-path analyzer or changing its command-line interface.
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
    parser.add_argument("--backend-artifact", type=Path, action="append", default=[])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument(
        "--repeats",
        type=int,
        choices=(2,),
        default=2,
        help="Independent processes (the A/B protocol requires two)",
    )
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument(
        "--cpu", default="auto", help="NUMA-local CPU (auto), a CPU index, or none"
    )
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "TEST_UNDECLARED_OUTPUTS_DIR", "crc_segmentation_benchmark_results"
            )
        ),
    )
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or not 0 <= args.seed <= 2**31 - 82:
        parser.error(
            "iterations must be positive, warmup nonnegative, seed in [0, 2147483566]"
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(
            "Output directory is not empty; choose an empty directory to retain earlier evidence"
        )
    manifest = {
        "success": False,
        "status": "starting",
        "protocol": "crc_segmentation_ab_v1",
        "command": sys.argv,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "processes": [],
        "backend_artifacts": [],
    }
    settings = {
        key: getattr(args, key)
        for key in ("iterations", "warmup", "repeats", "seed", "correctness_only")
    }
    manifest["settings"] = settings
    manifest_path = output / "run_manifest.json"

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    try:
        save_manifest()
        binary = args.binary.resolve(strict=True)
        manifest["binary"] = str(binary)
        manifest["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        provenance = source_provenance()
        manifest.update(provenance)
        settings.update(provenance, binary_sha256=manifest["binary_sha256"])
        names = [artifact.name for artifact in args.backend_artifact]
        if len(set(names)) != len(names):
            raise ValueError("Backend artifacts must have distinct filenames")
        if args.backend_artifact:
            source_output = output / "backend_source"
            source_output.mkdir()
            for artifact in args.backend_artifact:
                source = artifact.resolve(strict=True)
                destination = source_output / artifact.name
                shutil.copyfile(source, destination)
                manifest["backend_artifacts"].append(
                    {
                        "source": str(source),
                        "path": str(destination),
                        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    }
                )
        environment = dict(os.environ)
        environment["CRC_BENCH_BINARY_SHA256"] = manifest["binary_sha256"]
        manifest["cpu"] = select_cpu(args.cpu)
        manifest["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        manifest["status"] = "running"
        save_manifest()
        print(f"CRC segmentation A/B artifacts: {output}", flush=True)
        paths = []
        for repeat in REPEAT_IDS:
            raw = output / f"segmentation_{repeat}.jsonl"
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
            if args.correctness_only:
                command.append("--correctness-only")
            stdout_path = output / f"segmentation_{repeat}.stdout"
            stderr_path = output / f"segmentation_{repeat}.stderr"
            process = {
                "repeat": repeat,
                "seed": args.seed + repeat,
                "command": command,
                "raw": str(raw),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "status": "running",
            }
            manifest["processes"].append(process)
            save_manifest()
            print(shlex.join(command), flush=True)
            with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                completed = subprocess.run(
                    command, stdout=stdout, stderr=stderr, check=False, env=environment
                )
            (output / f"segmentation_{repeat}.exitcode").write_text(
                str(completed.returncode) + "\n"
            )
            process.update(
                exitcode=completed.returncode,
                status="completed" if completed.returncode == 0 else "failed",
            )
            if raw.is_file():
                process["raw_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
            save_manifest()
            if completed.returncode:
                print(stderr_path.read_text()[-12000:], file=sys.stderr)
                raise RuntimeError(
                    f"repeat {repeat} failed (exit {completed.returncode}); "
                    "raw evidence retained, no performance summary produced"
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
                    for key in ("status", "samples", "paired_rounds", "drift_warnings")
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
        print(f"CRC segmentation A/B FAILED: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        save_manifest()


if __name__ == "__main__":
    sys.exit(main())

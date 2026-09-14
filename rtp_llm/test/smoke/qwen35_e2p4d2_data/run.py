"""Prepare or execute the E2/P4/D2 best-configuration Bazel smoke/benchmark."""

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


def check_gpu_idle(out):
    """Read driver telemetry only; never initialize CUDA or clean up processes."""
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader",
        ],
        text=True,
        timeout=15,
    )
    memory = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    )
    (out / "gpu-precheck.txt").write_text(gpu)
    (out / "gpu-memory-precheck.txt").write_text(memory)
    if gpu.strip():
        raise SystemExit("GPU processes exist; benchmark was not launched.")
    try:
        rows = [
            tuple(int(value.strip()) for value in line.split(","))
            for line in memory.splitlines()
            if line.strip()
        ]
        if sorted(row[0] for row in rows) != list(range(8)) or any(
            len(row) != 3 for row in rows
        ):
            raise ValueError("expected telemetry for all eight GPUs")
    except (ValueError, IndexError) as error:
        raise SystemExit("Cannot verify GPU idleness: " + str(error)) from error
    # Compute-process visibility may be restricted by the container's PID namespace.
    busy = [row for row in rows if row[1] > 1024 or row[2] > 0]
    if busy:
        raise SystemExit(
            "GPUs are not idle (index, used MiB, utilization %): "
            + str(busy)
            + "; benchmark was not launched. Idle guard allows at most 1024 MiB per GPU."
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--kernel-repository", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--bazel", default="bazelisk")
    p.add_argument("--bazel-option", action="append", default=[])
    p.add_argument("--mode", choices=["smoke", "benchmark"], default="smoke")
    p.add_argument(
        "--e-batch",
        type=int,
        default=os.environ.get("VIT_GPU_MAX_BATCH_SIZE"),
        help="Override the smoke target batch argument; also accepts VIT_GPU_MAX_BATCH_SIZE",
    )
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if a.e_batch is not None and a.e_batch <= 0:
        p.error("--e-batch / VIT_GPU_MAX_BATCH_SIZE must be positive")
    data = Path(__file__).resolve().parent
    repo = data.parents[3]
    out = a.output.resolve()
    command = [a.bazel, "--batch"]
    if a.cache_root:
        command.append("--output_user_root=" + str(a.cache_root.resolve()))
    command += [
        "test",
        "//rtp_llm/test/smoke:qwen35_e2p4d2_repro",
        "--config=cuda13",
        "--config=sm10x",
        "--action_env=TF_CUDA_COMPUTE_CAPABILITIES=10.3",
        "--host_action_env=TF_CUDA_COMPUTE_CAPABILITIES=10.3",
        "--override_repository=pip_gpu_cuda13_torch_rtp_kernel="
        + str(a.kernel_repository.resolve()),
        "--run_under=//rtp_llm/test/utils:gpu_lock",
        "--test_output=all",
        "--nocache_test_results",
        "--test_timeout=14400",
        "--test_env=CUDA_LAUNCH_BLOCKING=0",
        "--test_env=WORLD_SIZE=8",
        "--test_env=GPU_COUNT=8",
        "--test_env=CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
        "--test_env=OMP_NUM_THREADS=8",
    ] + a.bazel_option
    flags = [
        "--model-dir=" + str(a.model_dir.resolve()),
        "--data-dir=" + str(data),
        "--output=" + str(out / "gpu-preflight"),
        "--reserve-mb=24576",
        "--p-token-budget=20000",
        "--d-reserve-mb=8192",
        "--d-kv-mb=49152",
        "--d-seq-limit=96",
        "--concurrencies=96",
        "--min-seconds=180",
        "--max-seconds=900",
        "--long-repeats=3",
    ]
    if a.e_batch is not None:
        flags.append("--e-batch=" + str(a.e_batch))
    if a.mode == "benchmark":
        flags.append("--benchmark")
    command += ["--test_arg=" + x for x in flags]
    public_command = [
        "--remote_header=<redacted>" if x.startswith("--remote_header=") else x
        for x in command
    ]
    print(shlex.join(public_command), flush=True)
    if not a.execute:
        return
    for directory in [a.model_dir, a.kernel_repository]:
        if not directory.is_dir():
            raise SystemExit("Missing directory: " + str(directory))
    subprocess.run(["python3", str(data / "verify.py")], check=True)
    precheck = repo / "internal_source/.cursor/skills/test-execution/pre_build_check.sh"
    if not precheck.is_file():
        raise SystemExit("Historical build precheck is missing; see README.")
    out.mkdir(parents=True, exist_ok=False)
    (out / "command.redacted.json").write_text(json.dumps(public_command, indent=2))
    evidence = {
        name: subprocess.check_output(cmd, cwd=repo, text=True)
        for name, cmd in {
            "head": ["git", "rev-parse", "HEAD"],
            "status": ["git", "status", "--short"],
        }.items()
    }
    (out / "source-evidence.json").write_text(json.dumps(evidence, indent=2))
    check = ["bash", str(precheck), "local", str(repo.parent)]
    if a.cache_root:
        check.append("--output-user-root=" + str(a.cache_root.resolve()))
    with (out / "precheck.log").open("w") as log:
        subprocess.run(
            check, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    check_gpu_idle(out)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", PYTHONNOUSERSITE="1")
    with (out / "bazel.log").open("w") as log:
        result = subprocess.run(
            command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    (out / "exit.json").write_text(json.dumps({"exit_code": result.returncode}))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

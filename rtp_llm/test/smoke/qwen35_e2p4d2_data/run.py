"""Prepare or execute the E2/P4/D2 best-configuration Bazel smoke/benchmark."""

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--kernel-repository", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--bazel", default="bazelisk")
    p.add_argument("--bazel-option", action="append", default=[])
    p.add_argument("--mode", choices=["smoke", "benchmark"], default="smoke")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
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
        "--e-batch=1",
        "--d-reserve-mb=8192",
        "--d-kv-mb=49152",
        "--d-seq-limit=96",
        "--concurrencies=96",
        "--min-seconds=180",
        "--max-seconds=900",
        "--long-repeats=3",
    ]
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
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader",
        ],
        text=True,
    )
    (out / "gpu-precheck.txt").write_text(gpu)
    if gpu.strip():
        raise SystemExit("GPU processes exist; benchmark was not launched.")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", PYTHONNOUSERSITE="1")
    with (out / "bazel.log").open("w") as log:
        result = subprocess.run(
            command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    (out / "exit.json").write_text(json.dumps({"exit_code": result.returncode}))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

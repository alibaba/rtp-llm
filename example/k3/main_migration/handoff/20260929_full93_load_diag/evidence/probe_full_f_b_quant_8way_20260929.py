"""Bounded eight-GPU reproduction of the first full-model FP8 tensor conversion.

This diagnostic starts one independent worker per GPU, as model TP loading does.
It is not a latency benchmark.
"""

import json
import os
import pathlib
import signal
import subprocess
import sys
import time


ROOT = pathlib.Path("/data0/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927")
SHARD = pathlib.Path("/mnt/hf3fs/3fs/models/kimi/kimi-k3/model-00001-of-000096.safetensors")
WORKER = ROOT / "probe_full_f_b_quant_20260929.py"
OUT = ROOT / f"probe-full-fb-quant-8way-exact-{os.environ.get('K3_DIAG_HOST_ID', '115')}-20260929"
RUNFILES = pathlib.Path(os.environ.get("K3_DIAG_RUNFILES") or (
    "/data0/luohaocheng.lhc/.cache/bazel/k3-integrated-perf-20260929/"
    "72fe907babae969d3cf28e3db89db495/execroot/rtp_llm/"
    "bazel-out/k8-opt/bin/rtp_llm/rtp_llm_server.runfiles"
))


def main():
    OUT.mkdir(mode=0o700, exist_ok=False)
    base_env = os.environ.copy()
    extra = [
        str(RUNFILES / f"pip_gpu_cuda13_torch_{name}" / "site-packages")
        for name in ("deep_gemm", "safetensors")
    ]
    base_env["PYTHONPATH"] = os.pathsep.join(extra + [base_env.get("PYTHONPATH", "")])
    base_env["CUDA_LAUNCH_BLOCKING"] = "1"
    base_env["DG_JIT_CACHE_DIR"] = str(ROOT / "jit-cache-integrated-114115-115" / "deepgemm")
    workers = []
    start = time.monotonic()
    try:
        for gpu in range(8):
            env = base_env.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_path = OUT / f"gpu{gpu}.log"
            output_path = OUT / f"gpu{gpu}.json"
            log = open(log_path, "w")
            proc = subprocess.Popen(
                [sys.executable, str(WORKER), "--shard", str(SHARD), "--output", str(output_path), "--exact-chunk-loop"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            workers.append((gpu, proc, log, output_path))
        deadline = start + 90
        while time.monotonic() < deadline and any(p.poll() is None for _, p, _, _ in workers):
            time.sleep(0.2)
    finally:
        for _, proc, _, _ in workers:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for _, proc, log, _ in workers:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            log.close()
    result = {
        "purpose": "concurrent functional diagnostic; timings are not benchmark results",
        "elapsed_wall_s": round(time.monotonic() - start, 3),
        "workers": [
            {"gpu": gpu, "exit_code": proc.returncode, "result_exists": path.exists()}
            for gpu, proc, _, path in workers
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if all(w["exit_code"] == 0 and w["result_exists"] for w in result["workers"]) else 1


if __name__ == "__main__":
    sys.exit(main())

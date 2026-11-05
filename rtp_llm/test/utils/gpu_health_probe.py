"""Node/GPU health probe for the CI GPU pools (scratch diagnostic, not a contract test).

Prints hostname, the pinned CUDA_VISIBLE_DEVICES, nvidia-smi GPU/UUID/occupancy and
compute-app list, then probes each visible GPU with a *time-bounded* child process so a
wedged device shows up as WEDGED instead of hanging the probe itself.
"""
import json
import os
import socket
import subprocess
import sys
import time

import torch


def _run(cmd, timeout=20):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as exc:  # noqa: BLE001 - probe must never raise
        return f"<{type(exc).__name__}: {exc}>"


def main():
    host = socket.gethostname()
    print(f"[probe] host={host} pid={os.getpid()}")
    print(f"[probe] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} GPU_COUNT={os.environ.get('GPU_COUNT', '<unset>')}")
    print("[probe] gpu list:\n" + _run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.used,memory.total,compute_cap", "--format=csv,noheader"]))
    print("[probe] compute apps:\n" + _run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"]))

    n = torch.cuda.device_count()
    print(f"[probe] torch.cuda.device_count={n}")
    wedged = []
    for i in range(n):
        code = (
            "import torch;"
            f"torch.cuda.set_device({i});"
            "x=torch.zeros(1024,1024,device='cuda');torch.cuda.synchronize();"
            f"print('init-ok', torch.cuda.get_device_name({i}))"
        )
        t0 = time.time()
        try:
            proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
            state = "OK" if proc.returncode == 0 else f"FAIL rc={proc.returncode}"
            detail = (proc.stdout or proc.stderr).strip().splitlines()[-1][:160] if (proc.stdout or proc.stderr).strip() else ""
        except subprocess.TimeoutExpired:
            state, detail = "WEDGED", "child did not finish within 60s"
            wedged.append(i)
        print(f"[probe] gpu{i}: {state} in {time.time() - t0:.1f}s | {detail}")
    summary = {"host": host, "visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpus": n, "wedged": wedged}
    print("[probe] summary=" + json.dumps(summary, sort_keys=True))
    if wedged:
        print(f"[probe] {len(wedged)} wedged device(s) on {host}: {wedged}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

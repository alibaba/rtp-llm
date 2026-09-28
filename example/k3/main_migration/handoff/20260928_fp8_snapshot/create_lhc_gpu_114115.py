#!/usr/bin/env python3
"""Create a personal CUDA 13/RDMA lhc_GPU on 114 or 115 with approved ddev."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


IMAGE = "hub.docker.alibaba-inc.com/isearch/rtp_llm_base_gpu_cuda13"
TAG = "2026_04_30_00_05_eba6d8a"
DDEV = Path("/data0/luohaocheng.lhc/env/work/alibaba/docker/ddev/ddev")
DDEV_SHA256 = "91f8fe5ac526f4e0b4ec17dd4b0689de348df850b3031b670ab2764ec28e7032"
CHECKPOINT_ROOT = Path("/mnt/hf3fs/3fs")


def output(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if output("id", "-un") != "luohaocheng.lhc":
        raise SystemExit("run under the personal account")
    if not DDEV.is_file() or not os.access(DDEV, os.X_OK):
        raise SystemExit("approved ddev is unavailable")
    if hashlib.sha256(DDEV.read_bytes()).hexdigest() != DDEV_SHA256:
        raise SystemExit("approved ddev checksum differs")
    if output("findmnt", "-T", str(CHECKPOINT_ROOT), "-n", "-o", "FSTYPE") != "fuse.hf3fs":
        raise SystemExit("3FS checkpoint mount is unavailable")
    if output("findmnt", "/mnt/ram", "-n", "-o", "FSTYPE") != "ramfs":
        raise SystemExit("ddev ramfs prerequisite is unavailable")
    if "lhc_GPU" in output("docker", "ps", "-a", "--format", "{{.Names}}").splitlines():
        raise SystemExit("lhc_GPU already exists; refusing to replace it")
    output("docker", "image", "inspect", f"{IMAGE}:{TAG}", "--format", "{{.Id}}")

    devices = [Path(f"/dev/infiniband/uverbs{index}") for index in range(12)]
    devices += [Path("/dev/infiniband/rdma_cm")]
    devices += [Path(f"/dev/nvidia{index}") for index in range(8)]
    if any(not device.exists() for device in devices):
        raise SystemExit("GPU or RDMA device set is incomplete")
    extra = [f"-v {CHECKPOINT_ROOT}:{CHECKPOINT_ROOT}:ro"]
    extra += [f"--device={device}" for device in devices[:12] if device.name != "uverbs0"]
    command = [
        "python2", str(DDEV), "create", "lhc_GPU", "--gpu", "--rdma",
        "--image", IMAGE, "--tag", TAG, "--docker_args", " ".join(extra),
    ]
    if args.dry_run:
        print(json.dumps({"host": output("hostname"), "command": command}))
        return
    subprocess.run(command, check=True)
    print(json.dumps({"host": output("hostname"), "container": "lhc_GPU", "status": "created"}))


if __name__ == "__main__":
    main()

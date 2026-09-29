#!/usr/bin/env python3
"""Restore the user's 112 development container after its prior removal."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


BASE = Path("/data1/luohaocheng.lhc")
DDEV = BASE / "artifacts/k3-fp8-opt-20260927/ddev"
DDEV_SHA256 = "91f8fe5ac526f4e0b4ec17dd4b0689de348df850b3031b670ab2764ec28e7032"
IMAGE = "hub.docker.alibaba-inc.com/isearch/rtp_llm_base_gpu_cuda13:2026_04_30_00_05_eba6d8a"
IMAGE_ID = "sha256:85085895ba19b24f19d0ba0f49cb3d32b644a688b2cc519aadd023d07c34016e"
NAME = "lhc_GPU"


def output(*argv: str) -> str:
    return subprocess.check_output(argv, universal_newlines=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if output("id", "-un") != "luohaocheng.lhc":
        raise SystemExit("the personal account is required")
    if not DDEV.is_file() or hashlib.sha256(DDEV.read_bytes()).hexdigest() != DDEV_SHA256:
        raise SystemExit("the approved ddev script is missing or changed")
    if NAME in output("docker", "ps", "-a", "--format", "{{.Names}}").splitlines():
        raise SystemExit(f"{NAME} already exists; refusing to replace it")
    if output("docker", "image", "inspect", IMAGE, "--format", "{{.Id}}") != IMAGE_ID:
        raise SystemExit("the local CUDA13 image differs from the approved image")
    for path, expected in (
        (BASE, "ext4"),
        (Path("/data4/luohaocheng.lhc/.home"), "ext4"),
        (Path("/mnt/hf3fs/3fs"), "fuse.hf3fs"),
        (Path("/mnt/ram"), "ramfs"),
    ):
        if output("findmnt", "-T", str(path), "-n", "-o", "FSTYPE") != expected:
            raise SystemExit(f"unexpected filesystem for {path}")
    devices = [Path(f"/dev/infiniband/uverbs{i}") for i in range(12)]
    devices += [Path("/dev/infiniband/rdma_cm")]
    devices += [Path(f"/dev/nvidia{i}") for i in range(8)]
    if any(not path.exists() for path in devices):
        raise SystemExit("GPU or RDMA device set is incomplete")

    extra = [
        "--cap-add=IPC_LOCK", "--ipc=host", "--ulimit=memlock=-1:-1",
        "-v /mnt/hf3fs/3fs:/mnt/hf3fs/3fs:ro", "-v /data3:/data3", "-v /data4:/data4",
    ]
    extra += [f"--device={path}" for path in devices[1:12]]
    command = [
        "python2", str(DDEV), "create", NAME, "--gpu", "--rdma",
        "--image", IMAGE.split(":", 1)[0], "--tag", IMAGE.rsplit(":", 1)[1],
        "--docker_args", " ".join(extra),
    ]
    if args.dry_run:
        print(json.dumps({"host": output("hostname"), "command": command}))
        return
    (DDEV.parent / "mnt/fuse").mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    memlock = output("docker", "exec", "-u", "luohaocheng.lhc", NAME,
                     "bash", "-lc", "ulimit -l")
    if memlock != "unlimited":
        raise SystemExit(f"{NAME} memlock is {memlock}, expected unlimited")
    inside = output("docker", "exec", "-u", "luohaocheng.lhc", NAME,
                    "bash", "-lc", "id -un; findmnt -T /data1/luohaocheng.lhc -n -o FSTYPE")
    if inside.splitlines() != ["luohaocheng.lhc", "ext4"]:
        raise SystemExit(f"container user or personal disk differs: {inside}")
    print(json.dumps({"host": output("hostname"), "container": NAME,
                      "image_id": IMAGE_ID, "memlock": memlock, "status": "created"}))


if __name__ == "__main__":
    main()

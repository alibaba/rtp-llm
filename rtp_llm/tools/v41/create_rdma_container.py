"""Create an additional PPU/RDMA container using the existing Pouch daemon."""

import argparse
import json
import stat
import subprocess
from pathlib import Path


def pouch(*args):
    result = subprocess.run(
        ["pouch", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if result.returncode:
        raise RuntimeError(f"pouch {args[0]} failed: {result.stderr.strip()}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="existing PPU container")
    parser.add_argument("--name", required=True, help="new RDMA container name")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    existing = subprocess.run(
        ["pouch", "inspect", args.name], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if existing.returncode == 0:
        parser.error(f"container already exists: {args.name}")
    source = json.loads(pouch("inspect", args.source))[0]
    host = source["HostConfig"]
    image = source.get("Config", {}).get("Image") or source["Image"]
    command = [
        "create",
        "--name",
        args.name,
        "--net",
        "host",
        "--shm-size",
        "8g",
        "--ulimit",
        "memlock=-1:-1",
        "--entrypoint",
        "/bin/bash",
        "--label",
        "v41.purpose=native-pd-rdma",
        "--label",
        f"v41.source={args.source}",
    ]
    for field, flag in (
        ("CgroupParent", "--cgroup-parent"),
        ("Memory", "--memory"),
        ("MemorySwap", "--memory-swap"),
        ("CpusetCpus", "--cpuset-cpus"),
        ("CpusetMems", "--cpuset-mems"),
        ("CpuQuota", "--cpu-quota"),
        ("CpuPeriod", "--cpu-period"),
        ("CpuShares", "--cpu-shares"),
    ):
        value = host.get(field)
        if value:
            command.extend([flag, str(value)])
    # Preserve declared mounts, including the propagated model FUSE mount.
    for bind in host.get("Binds", []):
        command.extend(["--volume", bind])
    for capability in host.get("CapAdd", []):
        command.extend(["--cap-add", capability])
    devices = {
        (
            item["PathOnHost"],
            item["PathInContainer"],
            item.get("CgroupPermissions") or "rwm",
        )
        for item in host.get("Devices", [])
    }
    rdma = []
    for path in sorted(Path("/dev/infiniband").glob("*")):
        if stat.S_ISCHR(path.stat().st_mode):
            devices.add((str(path), str(path), "rwm"))
            rdma.append(str(path))
    if not any(Path(path).name.startswith("uverbs") for path in rdma):
        raise RuntimeError("host has no RDMA uverbs devices")
    for source_device, destination, access in sorted(devices):
        command.extend(["--device", f"{source_device}:{destination}:{access}"])
    command.extend([image, "-lc", "exec sleep infinity"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    created = pouch(*command).strip()
    report = {
        "source_container": args.source,
        "container": args.name,
        "container_id": created,
        "image": image,
        "rdma_devices": rdma,
        "existing_container_restarted": False,
        "state": "created",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    pouch("start", args.name)
    report["state"] = "running"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Conservative compatibility detection for legacy DeepEP installations."""

import re
import subprocess
from pathlib import Path
from uuid import UUID

import torch
import torch.distributed as dist


def _local_topology(device, probe=True):
    import torch.distributed._symmetric_memory as symm_mem

    if not all(
        callable(getattr(symm_mem, name, None)) for name in ("empty", "rendezvous")
    ):
        raise RuntimeError("symmetric memory API unavailable")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if not boot:
        raise RuntimeError("missing host boot identity")
    uuid = "GPU-" + str(
        UUID(str(torch.cuda.get_device_properties(device).uuid).removeprefix("GPU-"))
    )
    if not probe:
        return {"boot": boot, "uuid": uuid}
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=5,
    )
    devices = {}
    for line in inventory.splitlines():
        index, gpu_uuid = line.split(",", 1)
        devices[gpu_uuid.strip()] = "GPU" + index.strip()
    if uuid not in devices:
        raise RuntimeError("CUDA GPU UUID not in topology inventory")
    topology = subprocess.check_output(
        ["nvidia-smi", "topo", "-m"], text=True, timeout=30
    )
    topology = re.sub(r"\x1b\[[0-9;]*m", "", topology)
    lines = [line.split() for line in topology.splitlines() if line.strip()]
    header = lines[0]
    rows = {
        line[0]: dict(zip(header, line[1:]))
        for line in lines[1:]
        if line[0] in devices.values()
    }
    adjacency = {
        gpu: [
            peer
            for peer, label in devices.items()
            if re.fullmatch(r"NV[1-9][0-9]*", rows.get(row, {}).get(label, ""))
        ]
        for gpu, row in devices.items()
    }
    return {"boot": boot, "uuid": uuid, "adjacency": adjacency}


def verified_nvlink_domain(records):
    if not records or any("error" in r for r in records):
        return False
    if len({r["boot"] for r in records}) != 1:
        return False
    uuids = {r["uuid"] for r in records}
    if len(uuids) != len(records):
        return False
    return all(uuids - {r["uuid"]} <= set(r["links"]) for r in records)


def legacy_nvlink_domain(group, device):
    """All ranks agree on host identity, unique GPUs and every NVLink edge.

    Missing tools, MIG identities and PCIe-only pairs fail closed. This only
    selects the candidate transport; rendezvous failures remain fatal rather
    than allowing ranks to fall back independently during a collective.
    """
    try:
        local = _local_topology(device, probe=group.rank() == 0)
    except (
        OSError,
        ImportError,
        RuntimeError,
        AttributeError,
        ValueError,
        IndexError,
        subprocess.SubprocessError,
    ) as error:
        local = {"error": str(error)}
    records = [None] * group.size()
    dist.all_gather_object(records, local, group=group)
    adjacency = records[0].get("adjacency", {})
    for record in records:
        record["links"] = adjacency.get(record.get("uuid"), [])
    return verified_nvlink_domain(records), records

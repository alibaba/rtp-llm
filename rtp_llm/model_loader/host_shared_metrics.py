"""Actual resident-memory and backing identity observations for Engram probes."""

import os
import resource
from pathlib import Path

import torch


def memory_accounting(lookup):
    tensors = {}
    paths = set()
    for name, item in lookup.shared.manifest["tensors"].items():
        path = lookup.shared.directory / item["file"]
        stat = path.stat()
        paths.add(str(path.resolve()))
        tensors[name] = {
            "path": str(path),
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "file_bytes": stat.st_size,
            "shape": item["shape"],
            "sha256": item["sha256"],
        }
    fields = (
        "Rss",
        "Pss",
        "Shared_Clean",
        "Shared_Dirty",
        "Private_Clean",
        "Private_Dirty",
        "Locked",
        "AnonHugePages",
    )
    totals = {name + "_bytes": 0 for name in fields}
    selected = False
    mapping_count = 0
    page_sizes = set()
    permissions = set()
    for line in Path("/proc/self/smaps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) >= 5 and "-" in parts[0] and len(parts[1]) == 4:
            selected = len(parts) == 6 and parts[5] in paths
            mapping_count += int(selected)
            if selected:
                permissions.add(parts[1])
        elif selected and ":" in line:
            key, value = line.split(":", 1)
            if key in fields:
                totals[key + "_bytes"] += int(value.split()[0]) * 1024
            elif key in ("KernelPageSize", "MMUPageSize"):
                page_sizes.add(int(value.split()[0]) * 1024)
    numa = []
    for line in Path("/proc/self/numa_maps").read_text().splitlines():
        if any("file=" + path in line for path in paths):
            numa.append(line)
    cgroup = {}
    errors = []
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        hierarchy, controllers, relative = line.split(":", 2)
        if hierarchy == "0" and not controllers:
            candidates = [
                Path("/sys/fs/cgroup") / relative.lstrip("/"),
                Path("/sys/fs/cgroup"),
            ]
            for directory in candidates:
                if (directory / "memory.current").is_file():
                    for name in (
                        "memory.current",
                        "memory.peak",
                        "memory.max",
                        "memory.swap.current",
                    ):
                        path = directory / name
                        if path.is_file():
                            raw = path.read_text().strip()
                            cgroup[name] = int(raw) if raw.isdecimal() else raw
                    break
    if not cgroup:
        errors.append("cgroup_v2_memory_accounting_unavailable")
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    props = torch.cuda.get_device_properties(lookup.device)
    free, total = torch.cuda.mem_get_info(lookup.device)
    return {
        "pid": os.getpid(),
        "uid": os.geteuid(),
        "kernel_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "backing_identity": lookup.shared.manifest["identity"],
        "model_revision": lookup.shared.manifest["revision"],
        "backing_file_bytes": lookup.total_bytes,
        "tensors": tensors,
        "mapping_count": mapping_count,
        "resident_mappings": totals,
        "page_sizes_bytes": sorted(page_sizes),
        "mapping_permissions": sorted(permissions),
        "numa_maps": numa,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
        "cgroup": cgroup,
        "memlock_soft": soft,
        "memlock_hard": hard,
        "mode": lookup.mode,
        "cpu_mapping_write_protected": not lookup.writable_mapping,
        "query_api_read_only": True,
        "registration_flags": lookup.flags,
        "capabilities": lookup.capabilities,
        "gpu_index": lookup.device,
        "gpu_uuid": str(getattr(props, "uuid", "unavailable")),
        "gpu_capability": list(torch.cuda.get_device_capability(lookup.device)),
        "hbm_allocated_bytes": torch.cuda.memory_allocated(lookup.device),
        "hbm_reserved_bytes": torch.cuda.memory_reserved(lookup.device),
        "hbm_peak_allocated_bytes": torch.cuda.max_memory_allocated(lookup.device),
        "cuda_free_bytes": free,
        "cuda_total_bytes": total,
        "accounting_errors": errors,
    }

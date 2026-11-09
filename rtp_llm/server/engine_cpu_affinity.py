import logging
import os
import re
from pathlib import Path
from typing import Mapping, NamedTuple


class EngineCpuPlacement(NamedTuple):
    device_id: int
    pci_bdf: str
    numa_node: int
    cpu: int


def _device_pci_bdf(device_id: int) -> str:
    from rtp_llm.ops.compute_ops import get_device_pci_bus_id

    raw_bdf = get_device_pci_bus_id(device_id).lower()
    match = re.fullmatch(
        r"([0-9a-f]{1,8}):([0-9a-f]{2}:[0-9a-f]{2}\.[0-7])", raw_bdf
    )
    if match is None:
        raise ValueError(f"invalid PCI address for device {device_id}: {raw_bdf!r}")
    domain, bus_device_function = match.groups()
    return f"{int(domain, 16):04x}:{bus_device_function}"


def _read_cpus(path: Path) -> set[int]:
    cpus = set()
    text = path.read_text().strip()
    if not text:
        return cpus
    for item in text.split(","):
        first, separator, last = item.partition("-")
        start = int(first)
        end = int(last) if separator else start
        if not 0 <= start <= end:
            raise ValueError(f"invalid CPU range {item!r} in {path}")
        cpus.update(range(start, end + 1))
    return cpus


def plan_engine_cpu_affinity(
    rank_devices: Mapping[int, int], sysfs: Path = Path("/sys")
) -> dict[int, EngineCpuPlacement]:
    """Plan all local ranks together, using one distinct physical core per rank."""
    try:
        allowed = os.sched_getaffinity(0)
        used_cores = set()
        plan = {}
        for rank, device_id in sorted(rank_devices.items()):
            pci_bdf = _device_pci_bdf(device_id)
            node = int((sysfs / "bus/pci/devices" / pci_bdf / "numa_node").read_text())
            if node < 0:
                raise ValueError(f"NUMA node is unknown for PCI device {pci_bdf}")
            candidates = allowed & _read_cpus(
                sysfs / f"devices/system/node/node{node}/cpulist"
            )
            for cpu in sorted(candidates):
                siblings = _read_cpus(
                    sysfs / f"devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
                )
                if cpu not in siblings:
                    raise ValueError(f"CPU {cpu} is missing from its SMT sibling list")
                core = min(siblings)
                if core not in used_cores:
                    used_cores.add(core)
                    plan[rank] = EngineCpuPlacement(device_id, pci_bdf, node, cpu)
                    break
            else:
                raise ValueError(f"rank {rank} has no free physical core in NUMA {node}")
        return plan
    except (AttributeError, ImportError, OSError, RuntimeError, ValueError) as error:
        logging.warning("Engine CPU affinity skipped for local ranks: %s", error)
        return {}


def resolve_engine_cpu(rank: int, device_id: int, placement: EngineCpuPlacement) -> int:
    """Check the plan against the rank's actual device and inherited CPU mask."""
    try:
        if (
            device_id != placement.device_id
            or _device_pci_bdf(device_id) != placement.pci_bdf
        ):
            raise ValueError(
                f"device {device_id} does not match planned PCI device {placement.pci_bdf}"
            )
        if placement.cpu not in os.sched_getaffinity(0):
            raise ValueError(f"planned CPU {placement.cpu} is outside the rank's CPU mask")
    except (AttributeError, ImportError, OSError, RuntimeError, ValueError) as error:
        logging.warning("Engine CPU affinity skipped for rank %d: %s", rank, error)
        return -1
    logging.info(
        "Engine CPU affinity plan: rank=%d device=%d pci=%s numa=%d cpu=%d",
        rank,
        device_id,
        placement.pci_bdf,
        placement.numa_node,
        placement.cpu,
    )
    return placement.cpu

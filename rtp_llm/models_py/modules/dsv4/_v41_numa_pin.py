"""#21(a) NUMA-local host affinity for V4.1 prefill ranks (R18/B).

``DSV41_NUMA_PIN=1`` (default 0) pins each rank's calling thread — and
every thread created afterwards, since Linux children inherit the CPU
mask — to the CPUs of the NUMA node local to the rank's GPU. Host-side
allocations after the hook land on local memory by first-touch (the
engine thread pools, weight-loading staging buffers and NCCL host
buffers are all created after the hook in ``_initialize_impl``); no
libnuma/mbind dependency is taken, and GPU-side numerics are untouched
(the parity sha gate is the backstop).

Node map (verified via ``nvidia-smi topo -m`` on the reference 8-GPU host): GPU0-3 ->
node0 (cpus 0-83,168-251), GPU4-7 -> node1 (84-167,252-335). Resolution
is pure sysfs: PCI bus id (pynvml, nvidia-smi fallback) ->
``/sys/bus/pci/devices/<bus>/numa_node`` ->
``/sys/devices/system/node/node<N>/cpulist``, intersected with the
process's current ``sched_getaffinity`` so cgroup/taskset restrictions
are respected. ANY resolution failure skips pinning for that rank
(announced, never fatal).
"""

import os
import sys

_NUMA_FLAG = "DSV41_NUMA_PIN"
_NUMA_ANNOUNCED = False


def numa_pin_enabled() -> bool:
    """#21(a): NUMA-local rank affinity (default 0)."""
    return os.environ.get(_NUMA_FLAG, "0") == "1"


def _announce(detail: str) -> None:
    global _NUMA_ANNOUNCED
    if not _NUMA_ANNOUNCED:
        _NUMA_ANNOUNCED = True
        print(f"[NUMAPIN] {detail}", file=sys.stderr, flush=True)


def cpulist_to_set(text: str) -> set:
    """Parse a kernel cpulist ("0-83,168-251") into a set of ints."""
    cpus = set()
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi) + 1 if hi else int(lo) + 1))
    return cpus


def normalize_pci_bus_id(bus: str) -> str:
    """nvidia-smi/pynvml bus ids ("00000000:1B:00.0") -> sysfs form
    ("0000:1b:00.0"): 4-hex domain, lowercase."""
    parts = bus.strip().split(":")
    if len(parts) != 3:
        return bus.strip().lower()
    domain = int(parts[0], 16) & 0xFFFF
    return f"{domain:04x}:{parts[1].lower()}:{parts[2].lower()}"


def _gpu_bus_id(device_index: int):
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            info = pynvml.nvmlDeviceGetPciInfo(handle)
            bus = info.busId
            return bus.decode() if isinstance(bus, bytes) else str(bus)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    try:
        import subprocess

        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=pci.bus_id",
                "--format=csv,noheader",
                "-i",
                str(device_index),
            ],
            text=True,
            timeout=10,
        )
        line = out.strip().splitlines()[0].strip()
        return line or None
    except Exception:
        return None


def gpu_numa_node(device_index: int):
    """NUMA node of a GPU via sysfs, or None when unresolvable."""
    bus = _gpu_bus_id(device_index)
    if not bus:
        return None
    key = normalize_pci_bus_id(bus)
    try:
        with open(f"/sys/bus/pci/devices/{key}/numa_node") as handle:
            node = int(handle.read().strip())
    except (OSError, ValueError):
        return None
    return node if node >= 0 else None


def pin_current_thread(device_index: int):
    """Pin the calling thread to the GPU-local NUMA node's CPUs.

    Returns ``(node, ncpus)`` on success, ``None`` when skipped (flag off,
    unresolvable node, or empty mask after the cgroup intersection).
    """
    if not numa_pin_enabled():
        return None
    node = gpu_numa_node(device_index)
    if node is None:
        _announce(f"gpu={device_index} state=skipped (numa node unresolved)")
        return None
    try:
        with open(f"/sys/devices/system/node/node{node}/cpulist") as handle:
            cpus = cpulist_to_set(handle.read())
    except OSError:
        _announce(f"gpu={device_index} numa={node} state=skipped (cpulist unreadable)")
        return None
    cpus &= os.sched_getaffinity(0)
    if not cpus:
        _announce(
            f"gpu={device_index} numa={node} state=skipped (empty after cgroup intersect)"
        )
        return None
    os.sched_setaffinity(0, cpus)
    _announce(
        f"gpu={device_index} numa={node} cpus={len(cpus)} state=pinned pid={os.getpid()}"
    )
    return (node, len(cpus))

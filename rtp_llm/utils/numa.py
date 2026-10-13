"""Configure host-memory placement before the server imports native runtimes."""

import ctypes
import os
import sys
from pathlib import Path
from typing import Set


def _parse_nodes(value: str) -> Set[int]:
    nodes = set()
    for part in value.strip().split(","):
        bounds = part.split("-", 1)
        first = int(bounds[0])
        last = int(bounds[-1])
        nodes.update(range(first, last + 1))
    if not nodes:
        raise ValueError("NUMA node list is empty")
    return nodes


def _available_memory_nodes() -> Set[int]:
    memory_nodes = _parse_nodes(Path("/sys/devices/system/node/has_memory").read_text())
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("Mems_allowed_list:"):
            return memory_nodes & _parse_nodes(line.split(":", 1)[1])
    raise RuntimeError("Mems_allowed_list is missing from /proc/self/status")


def _set_interleave(nodes: Set[int]) -> None:
    libnuma = ctypes.CDLL("libnuma.so.1", use_errno=True)
    set_mempolicy = libnuma.set_mempolicy
    set_mempolicy.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_ulong,
    ]
    set_mempolicy.restype = ctypes.c_long
    maxnode = max(nodes) + 1
    word_bits = ctypes.sizeof(ctypes.c_ulong) * 8
    mask = (ctypes.c_ulong * ((maxnode + word_bits - 1) // word_bits))()
    for node in nodes:
        mask[node // word_bits] |= 1 << (node % word_bits)
    # Match libnuma's size + 1: Linux get_nodes() subtracts one from maxnode.
    # MPOL_INTERLEAVE applies to this thread and is inherited across fork/exec.
    if set_mempolicy(3, mask, len(mask) * word_bits + 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def configure_numa_interleave() -> None:
    """Unset: auto-enable for two allowed memory nodes; empty: leave unchanged."""
    configured = os.environ.get("NUMA_INTERLEAVE_NODES")
    if configured is not None and not configured.strip():
        return
    automatic = configured is None
    if automatic and sys.platform != "linux":
        return
    try:
        if configured is None:
            nodes = _available_memory_nodes()
            if len(nodes) != 2:
                return
        else:
            nodes = _parse_nodes(configured)
        _set_interleave(nodes)
    except (OSError, ValueError, RuntimeError) as error:
        if not automatic:
            raise RuntimeError(
                f"Failed to apply NUMA_INTERLEAVE_NODES={configured!r}"
            ) from error
        print(f"[NUMA] Automatic interleave unavailable: {error}", file=sys.stderr)
        return
    value = ",".join(str(node) for node in sorted(nodes))
    os.environ["NUMA_INTERLEAVE_NODES"] = value
    print(f"[NUMA] Enabled memory interleave on nodes {value}", file=sys.stderr)

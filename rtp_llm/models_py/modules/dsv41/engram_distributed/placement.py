# SPDX-License-Identifier: Apache-2.0
"""Shard geometry, NUMA/GPU placement and native backing allocation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from math import gcd
from pathlib import Path

from rtp_llm.utils.numa import available_memory_nodes, parse_nodes

DEFAULT_GRANULARITY = 2 * 1024 * 1024
DEFAULT_BLOCK = 32
DEFAULT_SHARD_COUNT = 4


def _lcm(a: int, b: int) -> int:
    return a // gcd(a, b) * b


def _round_up(value: int, unit: int) -> int:
    if unit <= 0:
        raise ValueError("alignment unit must be positive")
    return -(-value // unit) * unit


@dataclass(frozen=True)
class ShardPlan:
    """Immutable geometry of one sharded Engram tensor."""

    total_rows: int
    shard_count: int
    shard_rows: int
    dim: int
    block: int
    granularity: int

    @property
    def weight_bytes(self) -> int:
        return self.shard_rows * self.dim

    @property
    def scale_bytes(self) -> int:
        return self.shard_rows * (self.dim // self.block)

    @property
    def shard_bytes(self) -> int:
        return self.weight_bytes + self.scale_bytes

    @property
    def padded_rows(self) -> int:
        return self.shard_rows * self.shard_count

    def row_range(self, shard: int) -> tuple[int, int]:
        """Half-open *real* row range owned by ``shard`` (padding excluded)."""
        if type(shard) is not int or not 0 <= shard < self.shard_count:
            raise IndexError(f"shard {shard} outside 0..{self.shard_count - 1}")
        start = shard * self.shard_rows
        end = min(start + self.shard_rows, self.total_rows)
        return start, end


def plan_shards(
    total_rows: int,
    dim: int,
    shard_count: int = DEFAULT_SHARD_COUNT,
    *,
    block: int = DEFAULT_BLOCK,
    granularity: int = DEFAULT_GRANULARITY,
) -> ShardPlan:
    """Compute the shard geometry, padding rows so every shard is mappable.

    Raises ``ValueError`` for geometries that cannot be mapped at a granularity
    boundary (for example a ``dim`` that is not a multiple of ``block``).
    """
    for name, value in (
        ("total_rows", total_rows),
        ("dim", dim),
        ("shard_count", shard_count),
        ("block", block),
        ("granularity", granularity),
    ):
        if type(value) is not int or not 0 < value < (1 << 63):
            raise ValueError(f"{name} must be a positive integer below 2**63")
    if total_rows <= 0:
        raise ValueError("total_rows must be positive")
    if dim <= 0 or dim % block:
        raise ValueError("dim must be a positive multiple of block")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if granularity <= 0 or granularity & (granularity - 1):
        raise ValueError("granularity must be a positive power of two")

    weight_unit = granularity // gcd(granularity, dim)
    scale_unit = granularity // gcd(granularity, dim // block)
    align_rows = _lcm(weight_unit, scale_unit)

    rows_per_shard = -(-total_rows // shard_count)
    shard_rows = _round_up(rows_per_shard, align_rows)
    if (shard_count - 1) * shard_rows >= total_rows:
        raise ValueError("shard_count too large: alignment would produce empty shards")
    if shard_rows * shard_count * dim >= (1 << 63):
        raise ValueError("shard address span exceeds 63-bit limit")
    plan = ShardPlan(
        total_rows=total_rows,
        shard_count=shard_count,
        shard_rows=shard_rows,
        dim=dim,
        block=block,
        granularity=granularity,
    )
    # Post-conditions the native mapping relies on.
    assert plan.weight_bytes % granularity == 0
    assert plan.scale_bytes % granularity == 0
    assert plan.shard_bytes % granularity == 0
    assert plan.padded_rows >= total_rows
    return plan


MAX_LOCAL_SEGMENTS = 8


def parse_numa(value):
    """Keep scalar placement compatible; lists/auto opt into row segmentation."""
    if value is None:
        return 0
    if value == "auto":
        return "auto"
    if type(value) is not str or not value:
        raise ValueError("INVALID_ENGRAM_NUMA_PLACEMENT")
    nodes = tuple(sorted(parse_nodes(value)))
    if len(nodes) > MAX_LOCAL_SEGMENTS:
        raise ValueError("INVALID_ENGRAM_NUMA_PLACEMENT")
    return nodes[0] if len(nodes) == 1 else nodes


def resolve_numas(
    value,
    *,
    native=None,
    sysfs=Path("/sys/devices/system/node"),
    status=Path("/proc/self/status"),
):
    if type(value) is int:
        if not 0 <= value <= 65535:
            raise ValueError("INVALID_ENGRAM_NUMA_PLACEMENT")
        return (value,)
    affinity = None
    if value == "auto":
        if not hasattr(native, "device_numa_nodes"):
            raise RuntimeError("PREBUILT_ENGRAM_NUMA_NATIVE_REQUIRED")
        try:
            affinity = dict(enumerate(native.device_numa_nodes()))
        except RuntimeError as error:
            raise RuntimeError("ENGRAM_CUDA_NUMA_QUERY_FAILED") from error
        if not affinity:
            raise ValueError("ENGRAM_NO_VISIBLE_GPU")
        if any(node < 0 for node in affinity.values()):
            raise ValueError("ENGRAM_HOST_NUMA_UNAVAILABLE")
    nodes = tuple(sorted(set(affinity.values()))) if affinity is not None else value
    if (
        type(nodes) is not tuple
        or not 1 <= len(nodes) <= MAX_LOCAL_SEGMENTS
        or any(type(n) is not int or not 0 <= n <= 65535 for n in nodes)
        or len(set(nodes)) != len(nodes)
    ):
        raise ValueError("INVALID_ENGRAM_NUMA_PLACEMENT")
    if not set(nodes) <= available_memory_nodes(sysfs, status):
        raise ValueError("ENGRAM_NUMA_OUTSIDE_CPUSET")
    for node in nodes:
        root = sysfs / f"node{node}"
        total = next(
            (
                int(line.split()[-2])
                for line in (root / "meminfo").read_text().splitlines()
                if "MemTotal:" in line
            ),
            0,
        )
        if total <= 0 or not (root / "cpulist").read_text().strip():
            raise ValueError("ENGRAM_NUMA_REQUIRES_CPU_MEMORY")
    logging.getLogger(__name__).info(
        "engram event=numa_placement mode=single_copy_rows nodes=%s gpu_numa=%s",
        nodes,
        affinity if affinity is not None else "explicit",
    )
    return nodes


def segment_rows(plan, shard, nodes):
    """(location, global begin, global end, padded rows), without extra backing."""
    begin, end = plan.row_range(shard)
    # Weight row width is a multiple of scale row width, so scale alignment
    # also aligns weight. Split whole alignment units, keeping total unchanged.
    unit = plan.granularity // gcd(plan.granularity, plan.dim // plan.block)
    units = plan.shard_rows // unit
    if not nodes or len(nodes) > MAX_LOCAL_SEGMENTS or units < len(nodes):
        raise ValueError("ENGRAM_SHARD_TOO_SMALL_FOR_LOCAL_SPLIT")
    result = []
    for index, node in enumerate(nodes):
        offset = (units * index // len(nodes)) * unit
        stop = (units * (index + 1) // len(nodes)) * unit
        start, finish = begin + offset, min(begin + stop, end)
        if start >= finish:
            raise ValueError("ENGRAM_SHARD_TOO_SMALL_FOR_LOCAL_SPLIT")
        result.append((node, start, finish, stop - offset))
    assert sum(part[3] for part in result) == plan.shard_rows
    return tuple(result)


@dataclass(frozen=True)
class PhysicalAllocation:
    """One actual native allocation, including alignment padding."""

    layer_id: int
    tensor_name: str
    location: int
    row_begin: int
    row_end: int
    padded_rows: int
    columns: int

    @property
    def size_bytes(self):
        return self.padded_rows * self.columns


@dataclass(frozen=True)
class PlacementPlan:
    """Immutable plan shared by admission, allocation, loading and publication."""

    allocations: tuple[PhysicalAllocation, ...]

    @classmethod
    def build(cls, checkpoint, shard, shards, granularity, locations):

        allocations = []
        for layer in checkpoint.layers:
            weight = next(t for t in layer.tensors if t.name.endswith(".weight"))
            rows = segment_rows(
                plan_shards(*weight.shape, shards, granularity=granularity),
                shard,
                locations[layer.layer_id],
            )
            for tensor in layer.tensors:
                for location, begin, end, padded in rows:
                    allocations.append(
                        PhysicalAllocation(
                            layer.layer_id,
                            tensor.name,
                            location,
                            begin,
                            end,
                            padded,
                            tensor.shape[1],
                        )
                    )
        return cls(tuple(allocations))

    @property
    def total_bytes(self):
        return sum(part.size_bytes for part in self.allocations)

    def for_tensor(self, name):
        return tuple(part for part in self.allocations if part.tensor_name == name)

    def bytes_by_location(self):
        required = {}
        for part in self.allocations:
            required[part.location] = required.get(part.location, 0) + part.size_bytes
        return required

    def validate_capacity(self, allocation_limit, budget):
        if any(part.size_bytes > allocation_limit for part in self.allocations):
            raise ValueError("local shard exceeds allocation limit")
        if self.total_bytes > budget:
            raise ValueError("local shards exceed host budget")


def host_groups(tensor):
    """Consecutive physical pieces belonging to each logical storage host."""
    groups = []
    for part in tensor.shards:
        if not groups or part.host_id != groups[-1][0].host_id:
            groups.append([])
        groups[-1].append(part)
    return tuple(tuple(group) for group in groups)


def _check_gpu_devices(native, devices):
    """Check the selected local CUDA ordinals before allocating backing."""
    if (
        type(devices) is not tuple
        or not 1 <= len(devices) <= MAX_LOCAL_SEGMENTS
        or any(type(d) is not int or d < 0 for d in devices)
        or len(set(devices)) != len(devices)
    ):
        raise ValueError("INVALID_ENGRAM_GPU_DEVICES")

    if max(devices) >= 2**31:
        raise ValueError("ENGRAM_GPU_OUTSIDE_VISIBLE_DEVICES")
    if not hasattr(native, "check_fabric_devices") or not hasattr(
        native, "device_granularity"
    ):
        raise RuntimeError("PREBUILT_ENGRAM_GPU_NATIVE_REQUIRED")
    try:
        native.check_fabric_devices(devices)
    except IndexError:
        raise ValueError("ENGRAM_GPU_OUTSIDE_VISIBLE_DEVICES") from None
    except ValueError:
        raise ValueError("ENGRAM_GPU_FABRIC_UNSUPPORTED") from None
    except RuntimeError as error:
        raise RuntimeError("ENGRAM_GPU_PLACEMENT_QUERY_FAILED") from error


def check_gpu_capacity(required, reserve_bytes):
    """Preflight ALL selected GPUs before acquiring any persistent backing.

    This is current free memory, not a prediction of later model/KV allocations.
    The caller must retain the model's normal KV/runtime budget independently.
    """
    import torch

    if type(reserve_bytes) is not int or not 0 < reserve_bytes < 2**63:
        raise ValueError("INVALID_ENGRAM_GPU_RESERVE")
    for device, needed in sorted(required.items()):
        free, total = torch.cuda.mem_get_info(device)
        logging.getLogger(__name__).info(
            "engram event=gpu_capacity device=%d required_bytes=%d free_bytes=%d total_bytes=%d reserve_bytes=%d",
            device,
            needed,
            free,
            total,
            reserve_bytes,
        )
        if needed > free - reserve_bytes:
            raise ValueError(
                f"ENGRAM_GPU_CAPACITY_INSUFFICIENT device={device} required_bytes={needed} free_bytes={free} reserve_bytes={reserve_bytes}"
            )


@dataclass(frozen=True)
class FabricBacking:
    native: object
    memory_kind: str
    device: int
    nodes: tuple
    granularity: int

    @classmethod
    def resolve(
        cls,
        native,
        layers,
        *,
        memory_kind,
        device,
        numa,
        gpu_devices,
    ):
        if memory_kind not in ("host", "gpu"):
            raise ValueError("ENGRAM_MEMORY_MUST_BE_HOST_OR_GPU")
        if not layers:
            raise ValueError("Fabric backing requires at least one Engram layer")
        gpu = memory_kind == "gpu"
        if gpu:
            _check_gpu_devices(native, gpu_devices)
            nodes = gpu_devices
            granularities = {native.device_granularity(device) for device in nodes}
        else:
            nodes = resolve_numas(numa, native=native)
            granularities = {
                native.host_numa_granularity(node, device) for node in nodes
            }
        if len(granularities) != 1:
            raise ValueError(
                "selected backing locations must share the same allocation granularity"
            )
        return cls(
            native,
            memory_kind,
            device,
            nodes,
            granularities.pop(),
        )

    def target_device(self, allocation):
        return allocation.location if self.memory_kind == "gpu" else self.device

    def validate_capacity(self, plan, reserve):
        if self.memory_kind == "gpu":
            check_gpu_capacity(plan.bytes_by_location(), reserve)

    def allocate(self, allocation):
        gpu = self.memory_kind == "gpu"
        device = self.target_device(allocation)
        return self.native.FabricAllocation(
            allocation.size_bytes,
            -1 if gpu else allocation.location,
            device,
            **({"memory_kind": "gpu"} if gpu else {}),
        )

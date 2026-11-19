# SPDX-License-Identifier: Apache-2.0
"""Startup-only FABRIC mappings for immutable DP8 Prefill Engram shards."""

import json
import logging
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import timedelta
from itertools import count
from pathlib import Path

import torch
import torch.distributed as dist

from .engram import (
    FP8_BLOCK_SIZE,
    _capture_local_engram_bindings,
    _engram_host_tensor_key,
    _triton_kernels,
)


ENV = "RTP_V41_PPU_ENGRAM_IN_HBM"
_STARTUP_EPOCH = count()
# A local module destructor cannot know whether other ranks still read its
# tables. Keep device mappings and their descriptors until process teardown.
_LIVE_TABLES = []


def _exchange(store, prefix, rank, payload):
    store.set(f"{prefix}/{rank}", json.dumps(payload))
    keys = [f"{prefix}/{peer}" for peer in range(8)]
    store.wait(keys, timedelta(seconds=120))
    return [json.loads(store.get(key)) for key in keys]


def _check(result, name):
    status, *values = result
    if int(status):
        raise RuntimeError(f"Engram peer shards: {name} returned {status}")
    return values[0] if values else None


def _map_allocation(driver, owner, handle, size, granularity, device):
    base = int(_check(driver.cuMemAddressReserve(size, granularity, 0, 0), "reserve"))
    owner.callback(lambda: _check(driver.cuMemAddressFree(base, size), "free address"))
    _check(driver.cuMemMap(base, size, 0, handle, 0), "map")
    owner.callback(lambda: _check(driver.cuMemUnmap(base, size), "unmap"))
    desc = driver.CUmemAccessDesc()
    desc.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = device.index
    desc.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    _check(driver.cuMemSetAccess(base, size, [desc], 1), "set access")
    return base


def _allocate_local(driver, owner, shard, device):
    fabric = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    prop = driver.CUmemAllocationProp()
    prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device.index
    prop.requestedHandleTypes = fabric
    granularity = int(_check(driver.cuMemGetAllocationGranularity(
        prop, driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
    ), "allocation granularity"))
    size = -(-shard.bytes_estimate // granularity) * granularity
    handle = _check(driver.cuMemCreate(size, prop, 0), "create FABRIC allocation")
    owner.callback(lambda: _check(driver.cuMemRelease(handle), "release local handle"))
    base = _map_allocation(driver, owner, handle, size, granularity, device)
    storage = torch._C._construct_storage_from_data_pointer(base, device, size)
    raw = torch.empty(0, dtype=torch.uint8, device=device).set_(storage, 0, (size,))
    split = shard.payload.numel()
    try:
        raw[:split].copy_(shard.payload.view(torch.uint8).reshape(-1))
        raw[split : split + shard.scale.numel()].copy_(shard.scale.reshape(-1))
    finally:
        torch.cuda.current_stream(device).synchronize()
    exported = _check(driver.cuMemExportToShareableHandle(handle, fabric, 0), "export FABRIC handle")
    return base, {
        "handle": bytes(exported.data).hex(), "size": size, "granularity": granularity,
        "weight_bytes": split,
    }


def _import_peer(driver, owner, info, device):
    fabric = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    handle = _check(driver.cuMemImportFromShareableHandle(bytes.fromhex(info["handle"]), fabric), "import FABRIC handle")
    try:
        return _map_allocation(driver, owner, handle, info["size"], info["granularity"], device)
    finally:
        _check(driver.cuMemRelease(handle), "release imported handle")


@dataclass(frozen=True)
class _PeerTable:
    weight_ptrs: torch.Tensor
    scale_ptrs: torch.Tensor
    table_rows: int
    rows_per_shard: int
    kernels: object
    host_keys: tuple

    def lookup(self, ids, shard, layout):
        if (layout.num_embeddings, layout.num_per_part) != (self.table_rows, self.rows_per_shard) or (
            self.host_keys != tuple(_engram_host_tensor_key(t) for t in (shard.payload, shard.scale))
        ):
            raise RuntimeError("Engram peer shard changed after startup")
        return self.kernels.engram_gather_peer(
            ids, self.weight_ptrs, self.scale_ptrs, self.table_rows, self.rows_per_shard,
        )


def _errors(votes, phase):
    errors = [(rank, vote["error"]) for rank, vote in enumerate(votes) if vote["error"]]
    if errors:
        raise RuntimeError(f"Engram peer shards {phase} failed: {errors}")


def prepare_peer_shards(bindings, device, *, flag, eligible):
    """Publish both layer mappings together after all WORLD8 owners are ready.

    Every member of the Prefill world calls this once, even when disabled.
    A requested but unsupported configuration fails collectively at startup.
    Host shards also serve the existing unadmitted lookup paths.
    """
    world = dist.group.WORLD
    rank = dist.get_rank(world)
    store = dist.distributed_c10d._get_default_store()
    epoch = next(_STARTUP_EPOCH)

    def exchange(phase, payload):
        return _exchange(store, f"rtp_llm/ppu_engram_peer/{epoch}/{phase}", rank, payload)

    flags = [vote["flag"] for vote in exchange("flag", {"flag": flag})]
    if flags == ["0"] * 8:
        return
    if flags != ["1"] * 8:
        raise ValueError(f"{ENV} must be uniformly 0 or 1 across the Prefill world")

    error = None
    geometry = None
    try:
        from cuda.bindings import driver

        from ._quant import is_ppu

        captured, geometry, bad = _capture_local_engram_bindings(bindings, rank)
        if not eligible or bad or not is_ppu() or torch.cuda.get_device_capability(device) != (8, 9):
            raise RuntimeError("requires PPU M890 non-speculative TP1/DP8/EP8/CP1 eager Prefill with local rounds")
        if any(not shard.pinned or shard.bytes_estimate == 0 for _, shard, *_ in captured):
            raise RuntimeError("requires pinned, nonempty Engram host shards on every rank")
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        properties = torch.cuda.get_device_properties(device)
        uuid = str(properties.uuid)
        kernels = _triton_kernels()
        if not boot_id or not uuid or kernels is None:
            raise RuntimeError("missing physical device identity or Triton Engram kernels")
        required = sum(shard.bytes_estimate for _, shard, *_ in captured)
        free, _ = torch.cuda.mem_get_info(device)
        if free < required:
            raise RuntimeError(f"needs {required} bytes of device memory for both tables, found {free}")
    except Exception as exc:
        error = repr(exc)
        boot_id = uuid = None
    votes = exchange("eligibility", {"geometry": geometry, "boot_id": boot_id, "uuid": uuid, "error": error})
    _errors(votes, "eligibility")
    if (len({tuple(v["geometry"]) for v in votes}) != 1
            or len({v["boot_id"] for v in votes}) != 1
            or len({v["uuid"] for v in votes}) != 8):
        raise RuntimeError("Engram peer shards require matching tables and eight distinct devices on one host")

    with torch.cuda.device(device), ExitStack() as owner:
        tables = []
        total_bytes = 0
        for layer, shard, layout, _, _, host_keys, _ in captured:
            info = base = error = None
            try:
                base, info = _allocate_local(driver, owner, shard, device)
            except Exception as exc:
                error = repr(exc)
            votes = exchange(f"allocate/{layer}", {"allocation": info, "error": error})
            _errors(votes, f"layer {layer} allocation")

            error = None
            try:
                weights, scales = [], []
                for peer, vote in enumerate(votes):
                    allocation = vote["allocation"]
                    weight_bytes = layout.valid_rows(peer) * shard.dim
                    if (allocation["weight_bytes"] != weight_bytes
                            or allocation["size"] < weight_bytes + weight_bytes // FP8_BLOCK_SIZE):
                        raise RuntimeError(f"incompatible Engram allocation geometry from rank {peer}")
                    address = base if peer == rank else _import_peer(driver, owner, allocation, device)
                    weights.append(address)
                    scales.append(address + weight_bytes)
                tables.append((shard, _PeerTable(
                    torch.tensor(weights, dtype=torch.int64, device=device),
                    torch.tensor(scales, dtype=torch.int64, device=device),
                    layout.num_embeddings, layout.num_per_part, kernels, host_keys,
                )))
                torch.cuda.current_stream(device).synchronize()
            except Exception as exc:
                error = repr(exc)
            _errors(exchange(f"mapped/{layer}", {"error": error}), f"layer {layer} mapping")
            total_bytes += info["size"]
        _LIVE_TABLES.append((owner.pop_all(), tuple(tables)))
        for shard, table in tables:
            shard._peer_table = table
        logging.info("DP8 Prefill Engram peer shards ready: layers 1/14, FABRIC, %d local bytes", total_bytes)

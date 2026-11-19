"""Storage adapters for the unchanged public EPLB runtime.

No planning, statistics lifetime or shuffle policy lives here. Communication
views borrow executor allocations, including their per-expert layout padding.
"""

import logging
import math
import os
import struct
from contextlib import contextmanager
from functools import lru_cache

from rtp_llm.eplb.runtime import EplbLayer


class _StaticPrefillR32Layer(EplbLayer):
    def map_ids(self, ids):
        from .prefill_r32_remap import try_prefill_r32_remap

        physical = try_prefill_r32_remap(ids, self.logical_to_physical, self.logical_count, self.replica_offset)
        return super().map_ids(ids) if physical is None else physical


def _static_prefill_r32_enabled(engine):
    flags = {
        "RTP_V41_PPU_PREFILL_R32_FUSED_REMAP": "1",
        "ROLE_TYPE": "PREFILL",
        "EPLB_MODE": "NONE",
        "TP_SIZE": "1",
        "DP_SIZE": "8",
        "EP_SIZE": "8",
        "PREFILL_CP_SIZE": "1",
    }
    return (
        all(os.environ.get(key) == value for key, value in flags.items())
        and (engine.num_routed_experts, engine.num_physical_experts, engine.num_local_experts) == (384, 416, 52)
        and (engine.backend, engine.precision, engine.ep_size) == ("megamoe", "w4a4", 8)
    )


def physical_expert_geometry(logical, physical, ep_size):
    logical, physical, ep_size = int(logical), int(physical), int(ep_size)
    if logical <= 0 or ep_size <= 0 or not logical <= physical <= 2 * logical:
        raise ValueError("physical experts must cover logical experts with R <= E")
    if physical % ep_size:
        raise ValueError("physical expert slots must divide evenly over EP")
    return physical, physical // ep_size


def expert_storage_geometry(shape, strides, itemsize, offset, storage_bytes, experts):
    """Return a byte-row geometry, rejecting overlapping/non-expert layouts.

    The inner dimensions may be permuted or padded. An expert's entire
    stride is meaningful storage, so copy that stride rather than flattening
    only visible elements. Never include a following tensor's allocation.
    """
    if not shape or shape[0] != experts or experts <= 0 or itemsize <= 0:
        raise ValueError("EPLB tensor must have local experts on dimension zero")
    if any(n <= 0 for n in shape) or any(s <= 0 for s in strides):
        raise ValueError("EPLB requires positive, non-overlapping expert strides")
    inner = sorted(zip(strides[1:], shape[1:]))
    extent = 1
    for stride, size in inner:
        if size > 1 and stride < extent:
            raise ValueError("EPLB expert's inner dimensions overlap")
        extent += (size - 1) * stride
    span = strides[0]
    if span < extent or (offset + experts * span) * itemsize > storage_bytes:
        raise ValueError("EPLB requires complete disjoint expert storage rows")
    return experts, span * itemsize, offset * itemsize


def expert_storage_view(tensor, experts):
    """A stable uint8 [experts, bytes] alias; no conversion or allocation."""
    import torch

    count, width, offset = expert_storage_geometry(
        tuple(tensor.shape), tuple(tensor.stride()), tensor.element_size(),
        tensor.storage_offset(), tensor.untyped_storage().nbytes(), experts,
    )
    return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
        tensor.untyped_storage(), offset, (count, width), (width, 1)
    )


def scalar_identity(value):
    """Native GEMM's inv_scale ABI is one finite positive Python scalar."""
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("PPU inverse weight scale must be a finite positive scalar")
    return struct.pack("!d", float(value)).hex()


@lru_cache(maxsize=1)
def _sail_dynamic_remap():
    import torch

    @torch.compile(fullgraph=True, dynamic=False)
    def remap(ids, mapping, counts, slot=None):
        logical = ids.clamp_min(0)
        if slot is None:
            slot = torch.randint(0, 65536, ids.shape, dtype=torch.int32, device=ids.device)
        physical = mapping[logical, slot % counts[logical]]
        return torch.where(ids >= 0, physical, -1).to(ids.dtype)

    return remap


@contextmanager
def batch_sail_replica_draws(engine, mode, rows, device):
    """Generate independent draws for the same dense E448 replica policy once per replay."""
    if (
        os.environ.get("RTP_V41_PPU_SAIL_BATCHED_REPLICA_RNG", "1") != "1"
        or mode != "dense" or rows != 32
        or (engine.num_routed_experts, engine.num_physical_experts, engine.ep_size) != (384, 448, 8)
        or len(engine._eplb_layers) != 40
        or not all(hasattr(layer, "map_sail_draws") for layer in engine._eplb_layers.values())
    ):
        yield
        return

    import torch

    bank = torch.randint(0, 65536, (40, rows, 6), dtype=torch.int32, device=device)
    previous = engine._sail_replica_draws
    engine._sail_replica_draws = dict(zip(sorted(engine._eplb_layers), bank.unbind(0)))
    try:
        yield
    finally:
        engine._sail_replica_draws = previous


def _configure_sail_dynamic(layer):
    """Match Sail's single-node candidate pruning and random replica choice."""
    import torch

    if (layer.num_logical_experts, layer.num_physical_experts, layer.num_local_experts) != (384, 448, 56):
        raise ValueError("Sail dynamic replica benchmark requires target 384+64 experts on EP8")
    expected = torch.arange(448, dtype=torch.int32) % 384
    if not torch.equal(layer.physical_to_logical.cpu(), expected):
        raise ValueError("Sail dynamic replica benchmark requires the Sail trivial physical map")
    first = torch.arange(384, dtype=torch.int32)
    other = first + 384
    has_replica = first < 64
    local_first = first // 56 == layer.ep_rank
    local_other = has_replica & (other // 56 == layer.ep_rank)
    has_remote_choice = has_replica & ~local_first & ~local_other
    mapping = torch.stack((torch.where(local_other, other, first),
                           torch.where(has_remote_choice, other, -1)), dim=1)
    device = layer.logical_to_physical.device
    mapping = mapping.to(device)
    counts = (1 + has_remote_choice.to(torch.int32)).to(device)
    remap = _sail_dynamic_remap()

    def map_ids(ids, draws=None):
        if draws is not None and (draws.shape != ids.shape or draws.device != ids.device):
            draws = None
        return remap(ids, mapping, counts, draws)

    layer.map_ids = layer.map_sail_draws = map_ids
    layer._fuse_sail_replica_routes = (
        os.environ.get("RTP_V41_PPU_SAIL_FUSED_BALANCED_REPLICA_ROUTES", "1") == "1"
    )
    if layer.layer_id == 0:
        logging.info("V4.1 Sail replica benchmark EP rank=%d physical=448 local=56 remote_choices=%d",
                     layer.ep_rank, int(has_remote_choice.sum()))


def bind_layer(engine, layer_id, mapping, counts, *, raw_tensors=()):
    """Freeze raw AND processed consumers before RuntimeExpertBalancer/Graph."""
    processed = engine._processed_weights(layer_id, mapping.device)
    # W4A8 preprocessing may choose a projection-wide exponent compensation.
    # The scalar GEMM ABI cannot express heterogeneous compensation after a
    # cross-rank move. Prove each projection is globally invariant at startup;
    # no rescaling, extra conversion or changed kernel is used to make it fit.
    invariants = tuple(scalar_identity(value) for value in processed[4:])
    if engine.ep_size > 1:
        import torch.distributed as dist

        gathered = [None] * engine.ep_size
        dist.all_gather_object(gathered, invariants, group=engine.group)
        if any(value != invariants for value in gathered):
            raise ValueError(
                f"L{layer_id} native inv13/inv2 differ across EP ranks: {gathered}; "
                "scalar GEMM ABI cannot migrate this representation losslessly"
            )
    named = [
        ("raw.w13", engine._weights[layer_id][0]),
        ("raw.s13", engine._weights[layer_id][1]),
        ("raw.w2", engine._weights[layer_id][2]),
        ("raw.s2", engine._weights[layer_id][3]),
        *raw_tensors,
        *zip(("processed.w13", "processed.s13", "processed.w2", "processed.s2"), processed[:4]),
    ]
    inventory, unique, regions = [], [], set()
    for name, tensor in named:
        view = expert_storage_view(tensor, engine.num_local_experts)
        region = (view.data_ptr(), tuple(view.shape))
        inventory.append((name, tensor, view))
        # E.g. grouped W4A4 packed weights alias raw weights. Shuffling an
        # allocation twice would apply a permutation twice; borrow it once.
        if region not in regions:
            start, shape = region
            end = start + shape[0] * shape[1]
            if any(start < other_start + other_shape[0] * other_shape[1] and other_start < end
                   for other_start, other_shape in regions):
                raise ValueError("partially overlapping EPLB views would shuffle executor bytes twice")
            regions.add(region)
            unique.append(view)
    layer_type = _StaticPrefillR32Layer if _static_prefill_r32_enabled(engine) else EplbLayer
    layer = layer_type(
        layer_id, mapping, counts, unique,
        ep_rank=engine.ep_rank, replica_offset=engine.ep_rank,
    )
    if (layer.num_logical_experts, layer.num_physical_experts) != (
        engine.num_routed_experts, engine.num_physical_experts,
    ):
        raise ValueError("loader maps differ from executor logical/physical geometry")
    if (os.environ.get("RTP_V41_PPU_SAIL_DYNAMIC_REPLICAS") == "1"
            and layer.num_physical_experts > layer.num_logical_experts):
        if (os.environ.get("EPLB_MODE", "NONE").upper() != "NONE"
                or engine.backend != "megamoe"):
            raise ValueError("Sail dynamic replica benchmark requires static MegaMoE placement")
        _configure_sail_dynamic(layer)
    engine._eplb_inventory[layer_id] = tuple(inventory)
    engine._eplb_invariants[layer_id] = invariants
    engine._eplb_layers[layer_id] = layer
    return layer

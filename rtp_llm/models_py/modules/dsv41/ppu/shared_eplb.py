"""Storage adapters for the unchanged public EPLB runtime.

No planning, statistics lifetime or shuffle policy lives here. Communication
views borrow executor allocations, including their per-expert layout padding.
"""

import math
import struct


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


def bind_layer(engine, layer_id, mapping, counts, *, raw_tensors=()):
    """Freeze raw AND processed consumers before RuntimeExpertBalancer/Graph."""
    from rtp_llm.eplb.runtime import EplbLayer

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
    layer = EplbLayer(
        layer_id, mapping, counts, unique,
        ep_rank=engine.ep_rank, replica_offset=engine.ep_rank,
    )
    if (layer.num_logical_experts, layer.num_physical_experts) != (
        engine.num_routed_experts, engine.num_physical_experts,
    ):
        raise ValueError("loader maps differ from executor logical/physical geometry")
    engine._eplb_inventory[layer_id] = tuple(inventory)
    engine._eplb_invariants[layer_id] = invariants
    engine._eplb_layers[layer_id] = layer
    return layer

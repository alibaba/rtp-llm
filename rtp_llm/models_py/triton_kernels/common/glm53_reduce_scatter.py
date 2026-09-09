"""Publish BF16 TP partials into the existing GLM GEMM/RS workspace."""

import triton
import triton.language as tl


@triton.jit
def publish_glm53_partials(
    source,
    peer_pointers,
    LOCAL_NUMEL: tl.constexpr,
    NUM_RANKS: tl.constexpr,
    SOURCE_RANK: tl.constexpr,
    DATA_OFFSET_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Interleave peers and rotate by source rank to avoid simultaneous incast
    # into rank 0. Each destination retains source-rank-major partials, so the
    # following FP32 reduction has an identical order on every destination.
    pid = tl.program_id(0).to(tl.int64)
    destination = (pid % NUM_RANKS + SOURCE_RANK) % NUM_RANKS
    offsets = pid // NUM_RANKS * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < LOCAL_NUMEL
    pointer = tl.multiple_of(tl.load(peer_pointers + destination), 16)
    target = (pointer + DATA_OFFSET_BYTES + 2 * SOURCE_RANK * LOCAL_NUMEL).to(
        tl.pointer_type(tl.bfloat16)
    )
    values = tl.load(source + destination * LOCAL_NUMEL + offsets, mask, other=0)
    tl.store(target + offsets, values, mask)

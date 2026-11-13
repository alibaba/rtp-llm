import triton
import triton.language as tl


@triton.jit
def map_local_routes(
    global_ids,
    local_ids,
    count: tl.constexpr,
    offset: tl.constexpr,
    local_experts: tl.constexpr,
    block: tl.constexpr,
):
    index = tl.program_id(0) * block + tl.arange(0, block)
    valid = index < count
    expert = tl.load(global_ids + index, valid, other=-1).to(tl.int32) - offset
    local = tl.where((expert >= 0) & (expert < local_experts), expert, -1)
    tl.store(local_ids + index, local, valid)

"""Experimental original Torch BF16 top-k8 set, values and unstable slot order."""

import torch
import triton
import triton.language as tl


@triton.jit
def _router_topk_sort_kernel(P, VALUES, IDS, COUNT):
    row = tl.program_id(0).to(tl.int64)
    column = tl.arange(0, 128)
    values = tl.load(P + row * 128 + column, row < COUNT, other=0)
    bits = values.to(tl.uint16, bitcast=True).to(tl.uint32)
    mask = tl.where((bits & 0x8000) != 0, 0xFFFF, 0x8000)
    radix = tl.where(values == values, bits ^ mask, 0xFFFF).to(tl.int32)
    # Unique index priority chooses the same first-seen pivot ties. It does
    # not define final slot order: original gather phases + bitonic do that.
    encoded = (radix << 7) | (127 - column)
    ordered = tl.sort(encoded, descending=True)
    pivot = tl.sum(tl.where(column == 7, ordered >> 7, 0), axis=0)
    lane = tl.arange(0, 32)
    selected = tl.gather(ordered, lane, axis=0)
    chosen_id = 127 - (selected & 127)
    phase_index = chosen_id + tl.where((selected >> 7) == pivot, 128, 0)
    gather_order = tl.sort(
        tl.where(lane < 8, phase_index, 0x7FFFFFFF), descending=False
    )
    ids = (gather_order & 127).to(tl.int64)
    valid = lane < 8
    raw = tl.load(P + row * 128 + ids, valid & (row < COUNT), other=0)
    raw_bits = raw.to(tl.uint16, bitcast=True)
    keys = raw.to(tl.float32)
    for level in tl.static_range(1, 6):
        for step in tl.static_range(level - 1, -1, -1):
            stride = 1 << step
            peer = lane ^ stride
            peer_key = tl.gather(keys, peer, axis=0)
            peer_id = tl.gather(ids, peer, axis=0)
            peer_bits = tl.gather(raw_bits, peer, axis=0)
            peer_valid = tl.gather(valid, peer, axis=0)
            lower = (lane & stride) == 0
            a = tl.where(lower, keys, peer_key)
            b = tl.where(lower, peer_key, keys)
            va = tl.where(lower, valid, peer_valid)
            vb = tl.where(lower, peer_valid, valid)
            greater = (a > b) | ((a != a) & (b == b))
            comparison = (greater & va) | ~vb
            if level < 5:
                reverse = (lane & (1 << level)) != 0
            else:
                reverse = tl.full((32,), False, tl.int1)
            exchange = comparison == reverse
            keys = tl.where(exchange, peer_key, keys)
            ids = tl.where(exchange, peer_id, ids)
            raw_bits = tl.where(exchange, peer_bits, raw_bits)
            valid = tl.where(exchange, peer_valid, valid)
    tl.store(VALUES + row * 8 + lane, raw_bits.to(tl.bfloat16, bitcast=True), lane < 8)
    tl.store(IDS + row * 8 + lane, ids, lane < 8)


def router_topk(probabilities, return_kernel=False):
    if (
        not probabilities.is_cuda
        or probabilities.dtype != torch.bfloat16
        or probabilities.dim() != 2
        or probabilities.shape[1] != 128
        or not 1 <= probabilities.shape[0] <= 131072
        or not probabilities.is_contiguous()
    ):
        return None
    shape = (probabilities.shape[0], 8)
    values = torch.empty(shape, device=probabilities.device, dtype=torch.bfloat16)
    indices = torch.empty(shape, device=probabilities.device, dtype=torch.int64)
    kernel = _router_topk_sort_kernel[(probabilities.shape[0],)](
        probabilities, values, indices, probabilities.shape[0], num_warps=4,
    )
    return (values, indices, kernel) if return_kernel else (values, indices)

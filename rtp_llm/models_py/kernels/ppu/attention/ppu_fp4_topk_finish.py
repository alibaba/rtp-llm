"""Q3 native TopK finalization with the original short-row sorting order."""

import torch
import triton
import triton.language as tl


@triton.jit
def _finish_selected(
    S,
    L,
    O,
    SELECTED,
    STARTS,
    IS: tl.constexpr,
    WIDTH: tl.constexpr,
    SS: tl.constexpr,
    OS: tl.constexpr,
    K: tl.constexpr,
    STORE: tl.constexpr,
):
    row = tl.program_id(0)
    n = tl.load(L + row)
    if (n >= 0) & (n <= 32) & (n <= WIDTH):
        i = tl.arange(0, 32)
        value = tl.load(S + row * SS + i, i < n, other=0)
        bits = value.to(tl.uint16, bitcast=True).to(tl.int32)
        # Ascending BF16 radix key distinguishes -0 from +0. The network
        # below compares numbers, so signed zero ties there.
        key = tl.where((bits & 0x8000) != 0, bits ^ 0xFFFF, bits ^ 0x8000)
        floor = tl.min(tl.where(i < n, key, 0x10000), 0)
        first = (i < n) & (key > floor)
        last = (i < n) & (key == floor)
        dst = tl.where(
            first,
            tl.cumsum(first.to(tl.int32)) - 1,
            tl.sum(first.to(tl.int32), 0) + tl.cumsum(last.to(tl.int32)) - 1,
        )
        # Invert the stable gather permutation. Mask invalid ranks explicitly.
        ids = tl.min(
            tl.where((i[:, None] < n) & (dst[:, None] == i[None, :]), i[:, None], 32),
            axis=0,
        )
        vals = tl.gather(value.to(tl.float32), ids % 32, 0)
        for log_stage in tl.static_range(1, 6):
            for log_stride in tl.static_range(log_stage - 1, -1, -1):
                stride = 1 << log_stride
                other = i ^ stride
                peer_ids = tl.gather(ids, other, 0)
                peer_vals = tl.gather(vals, other, 0)
                lower = (i & stride) == 0
                a = tl.where(lower, ids, peer_ids)
                b = tl.where(lower, peer_ids, ids)
                av = tl.where(lower, vals, peer_vals)
                bv = tl.where(lower, peer_vals, vals)
                lane = ((i // (2 * stride)) * stride) + i % stride
                desc = (b == 32) | ((a != 32) & (av > bv))
                swap = desc == ((lane & (1 << (log_stage - 1))) != 0)
                ids = tl.where(swap, peer_ids, ids)
                vals = tl.where(swap, peer_vals, vals)
        col = tl.arange(0, STORE)
        tl.store(O + row * OS + col, -1, (col >= n) & (col < K))
        tl.store(O + row * OS + i, ids, i < n)
    else:
        selected_col = tl.arange(0, STORE)
        index = tl.load(SELECTED + row * IS + selected_col, selected_col < K, other=-1)
        start = tl.load(STARTS + row)
        selected_value = tl.where((index >= start) & (index < n), index, -1).to(
            tl.int32
        )
        tl.store(O + row * OS + selected_col, selected_value, selected_col < K)


def native_topk_restore(scores, starts, ends, out, *, topk=512):
    assert scores.is_cuda and torch.cuda.get_device_name(scores.device) == "ZW-M890P"
    rows = scores.shape[0]
    assert (
        scores.dtype == torch.bfloat16
        and rows in (3, 24)
        and scores.shape == (rows, 12288)
    )
    assert starts.dtype == ends.dtype == torch.int32 and starts.shape == ends.shape == (
        rows,
    )
    assert starts.stride(0) == ends.stride(0) == 1
    assert scores.stride(1) == out.stride(1) == 1
    assert out.dtype == torch.int32 and out.shape == (rows, 512) and topk == 512
    assert scores.device == out.device == starts.device == ends.device
    # Preserve the complete original native selection, including ties and rank shape.
    selected = scores.unsqueeze(0).topk(topk, dim=-1).indices[0]
    _finish_selected[(rows,)](
        scores,
        ends,
        out,
        selected,
        starts,
        selected.stride(0),
        12288,
        scores.stride(0),
        out.stride(0),
        512,
        512,
        num_warps=4,
    )
    return out

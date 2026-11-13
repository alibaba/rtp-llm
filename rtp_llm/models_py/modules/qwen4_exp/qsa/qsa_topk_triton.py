"""Fuse QSA selection bookkeeping while preserving native TopK tie ordering."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mask_scores(
    scores,
    lengths,
    masked,
    N: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, BN)
    length = tl.load(lengths + row * LENGTH_STRIDE).to(tl.int64)
    complete = (length - tl.where(length < 0, 3, 0)) // 4
    values = tl.load(
        scores + row * ROW_STRIDE + column * COL_STRIDE, column < N, other=0
    )
    tl.store(
        masked + row * N + column,
        tl.where(column < complete, values, float("-inf")),
        column < N,
    )


@triton.jit
def _expand_tokens(
    values,
    blocks,
    lengths,
    output,
    K: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    FINITE_ONLY: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.arange(0, 4096)
    index = slot // 4
    value = tl.load(values + row * K + index, index < K, other=float("-inf"))
    block = tl.load(blocks + row * K + index, index < K, other=0)
    if FINITE_ONLY:
        finite = (value == value) & (tl.abs(value) != float("inf"))
    else:
        finite = value > float("-inf")
    token = tl.where((index < K) & finite, block * 4 + slot % 4, -1)
    length = tl.load(lengths + row * LENGTH_STRIDE).to(tl.int64)
    complete = (length - tl.where(length < 0, 3, 0)) // 4
    tail_offset = slot - 2048
    tail = tl.where(tail_offset < length - complete * 4, complete * 4 + tail_offset, -1)
    token = tl.where(slot >= 2048, tail, token)
    tl.store(output + row * 2051 + slot, token.to(tl.int32), slot < 2051)


@triton.jit
def _packed_qsa_tokens(
    S,
    L,
    O,
    N: tl.constexpr,
    B: tl.constexpr,
    FINITE_ONLY: tl.constexpr,
    PACKED_COUNT: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    length = tl.load(L + row).to(tl.int64)
    complete = (length - tl.where(length < 0, 3, 0)) // 4
    raw = S.to(tl.pointer_type(tl.uint32))
    bits = tl.load(raw + row * N + col, col < N, other=0xFF800000)
    bits = tl.where(col < complete, bits, 0xFF800000).to(tl.uint32)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
    is_nan = ((bits & 0x7F800000) == 0x7F800000) & ((bits & 0x007FFFFF) != 0)
    # Native CUDA selects with all NaNs at the high sentinel, then sorts the
    # chosen values by their raw radix bits. Negative NaNs therefore occupy
    # the tail of the selected set. Signed zeros retain their original bits.
    selection_order = tl.where(is_nan, 0xFFFFFFFF, ordered).to(tl.uint32)
    key = (selection_order.to(tl.uint64) << 32) | (0xFFFFFFFF - col.to(tl.uint32)).to(
        tl.uint64
    )
    key = tl.where(col < N, key, tl.full((B,), 0, tl.uint64))
    selected = tl.topk(key, 512)
    if tl.max(selected >> 32, 0) == 0xFFFFFFFF:
        chosen = (0xFFFFFFFF - (selected & 0xFFFFFFFF).to(tl.uint32)).to(tl.int32)
        chosen_bits = tl.load(raw + row * N + chosen)
        raw_order = tl.where(
            (chosen_bits & 0x80000000) != 0, ~chosen_bits, chosen_bits ^ 0x80000000
        ).to(tl.uint32)
        raw_key = (raw_order.to(tl.uint64) << 32) | (
            0xFFFFFFFF - chosen.to(tl.uint32)
        ).to(tl.uint64)
        selected = tl.sort(raw_key, descending=True)
    index = (0xFFFFFFFF - (selected & 0xFFFFFFFF).to(tl.uint32)).to(tl.int32)
    value_bits = tl.load(raw + row * N + index)
    finite = ((value_bits & 0x7F800000) != 0x7F800000) & (index < complete)
    if not FINITE_ONLY:
        finite = (finite | (value_bits == 0x7F800000)) & (index < complete)
    token = tl.where(finite[:, None], index[:, None] * 4 + tl.arange(0, 4)[None, :], -1)
    slot = tl.arange(0, 512)[:, None] * 4 + tl.arange(0, 4)[None, :]
    if PACKED_COUNT:
        block_count = tl.sum(finite.to(tl.int32), 0)
        ordered_slot = tl.arange(0, 512)
        first_invalid = tl.min(tl.where(finite, 512, ordered_slot), 0)
        if first_invalid == block_count:
            block_rank = ordered_slot
        else:
            # Non-finite selected values may introduce holes. Stable scan is
            # needed only for those rows; ordinary finite prefixes keep order.
            block_rank = tl.cumsum(finite.to(tl.int32), 0) - 1
        count = block_count * 4 + tl.minimum(tl.maximum(length - complete * 4, 0), 3)
        packed_slot = block_rank[:, None] * 4 + tl.arange(0, 4)[None, :]
        tl.store(O + row * 2052 + packed_slot, token.to(tl.int32), finite[:, None])
        # Disjoint stores: no reader observes unused slots, but initialize them
        # for diagnostics and stable allocator-independent snapshots.
        tl.store(O + row * 2052 + slot, -1, slot >= count)
        tl.store(O + row * 2052 + 2051, count)
    else:
        tl.store(O + row * 2051 + slot, token.to(tl.int32))
    tail = tl.arange(0, 4)
    tail_token = tl.where(tail < length - complete * 4, complete * 4 + tail, -1)
    if PACKED_COUNT:
        tl.store(
            O + row * 2052 + block_count * 4 + tail,
            tail_token.to(tl.int32),
            (tail < 3) & (tail < length - complete * 4),
        )
        tl.store(O + row * 2052 + 2048 + tail, -1, (tail < 3) & (2048 + tail >= count))
    else:
        tl.store(O + row * 2051 + 2048 + tail, tail_token.to(tl.int32), tail < 3)


def try_select_qsa_tokens(
    scores,
    lengths,
    *,
    compress_ratio,
    token_budget,
    scores_are_masked=False,
    finite_only=True,
    packed_count=False,
):
    if not (
        scores.is_cuda
        and torch.version.hip is None
        and scores.dtype == torch.float32
        and scores.ndim == 2
        and 0 < scores.shape[0] <= 32768
        and 0 < scores.shape[1] <= 8192
        and scores.stride(0) > 0
        and scores.stride(1) > 0
        and lengths.shape == (scores.shape[0],)
        and lengths.dtype == torch.int32
        and lengths.device == scores.device
        and lengths.stride(0) > 0
        and compress_ratio == 4
        and token_budget == 2048
        and torch.cuda.get_device_capability(scores.device)[0] >= 8
    ):
        return None
    rows, columns = scores.shape
    if scores_are_masked and not packed_count:
        from rtp_llm.models_py.modules.qwen4_exp.qsa.cub_topk import try_cub_qsa_tokens

        output = try_cub_qsa_tokens(scores, lengths, finite_only=finite_only)
        if output is not None:
            return output
    if (
        scores_are_masked
        and columns in (1024, 2048)
        and rows <= 32
        and scores.is_contiguous()
        and lengths.is_contiguous()
        and torch.__version__.split("+", 1)[0] == "2.11.0"
        and torch.version.git_version == "70d99e998b4955e0049d13a98d77ae1b14db1f45"
        and torch.cuda.get_device_capability(scores.device) == (12, 0)
    ):
        # This bounded, bit-exact radix-key order is validated against the
        # frozen native CUDA TopK, including mixed ties, signed zero and NaNs.
        # Larger score buffers were slower in full-consumer A/B; retain native.
        output = torch.empty(
            (rows, 2052 if packed_count else 2051),
            device=scores.device,
            dtype=torch.int32,
        )
        warps = 8 if columns == 1024 and rows < 16 else 16
        _packed_qsa_tokens[(rows,)](
            scores,
            lengths,
            output,
            N=columns,
            B=columns,
            FINITE_ONLY=finite_only,
            PACKED_COUNT=packed_count,
            num_warps=warps,
        )
        return output
    # Paged scoring has already stored -inf past each completed-block limit.
    # Only callers owning that producer contract may skip the duplicate mask.
    if scores_are_masked and scores.is_contiguous():
        masked = scores
    else:
        masked = torch.empty((rows, columns), device=scores.device, dtype=scores.dtype)
        _mask_scores[(rows,)](
            scores,
            lengths,
            masked,
            N=columns,
            ROW_STRIDE=scores.stride(0),
            COL_STRIDE=scores.stride(1),
            LENGTH_STRIDE=lengths.stride(0),
            BN=triton.next_power_of_2(columns),
            num_warps=4,
        )
    # Keep the original primitive and its sorting/tie behavior unchanged.
    values, blocks = torch.topk(masked, min(512, columns), dim=-1)
    output = torch.empty((rows, 2051), device=scores.device, dtype=torch.int32)
    _expand_tokens[(rows,)](
        values,
        blocks,
        lengths,
        output,
        K=values.shape[1],
        LENGTH_STRIDE=lengths.stride(0),
        FINITE_ONLY=finite_only,
        num_warps=4,
    )
    if packed_count:
        output = compact_qsa_selection(output)
    return output


@triton.jit
def _compact_selected(I, O, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    token = tl.load(I + row * WIDTH + col, col < WIDTH, other=-1)
    valid = (col < WIDTH) & (token >= 0)
    rank = tl.cumsum(valid.to(tl.int32), 0) - 1
    count = tl.sum(valid.to(tl.int32), 0)
    tl.store(O + row * (WIDTH + 1) + rank, token, valid)
    tl.store(O + row * (WIDTH + 1) + col, -1, (col < WIDTH) & (col >= count))
    tl.store(O + row * (WIDTH + 1) + WIDTH, count)


def compact_qsa_selection(selected):
    """Internal stable 2052-column format: 2051 tokens plus device valid_count.

    The public 2051-slot selection/undo contract is unchanged. Consumers must
    explicitly opt into this format; count is never treated as a token index.
    """
    if not (
        selected.is_cuda
        and selected.dtype == torch.int32
        and selected.ndim in (2, 3)
        and selected.shape[-1] == 2051
        and selected.is_contiguous()
    ):
        raise ValueError("packed QSA requires contiguous CUDA int32 2051-slot rows")
    output = torch.empty(
        (*selected.shape[:-1], 2052), device=selected.device, dtype=torch.int32
    )
    _compact_selected[(selected.numel() // 2051,)](
        selected, output, WIDTH=2051, BLOCK=4096, num_warps=8
    )
    return output

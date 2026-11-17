"""NTC dilated PLE convolution with explicit BF16 consumer boundaries."""

import torch
import triton
import triton.language as tl


@triton.jit
def _store_ple_output(
    acc, Gated, Output, Residual, offset, valid, HAS_RESIDUAL: tl.constexpr
):
    conv = acc.to(tl.bfloat16).to(tl.float32)
    activated = (conv * tl.sigmoid(conv)).to(tl.bfloat16).to(tl.float32)
    gated = tl.load(Gated + offset, valid, other=0).to(tl.float32)
    output = gated + activated
    if HAS_RESIDUAL:
        # Preserve the original BF16 PLE output before the outer residual add.
        output = output.to(tl.bfloat16).to(tl.float32)
        residual = tl.load(Residual + offset, valid, other=0).to(tl.float32)
        output = residual + output
    tl.store(Output + offset, output, valid)


@triton.jit(do_not_specialize=["LENGTH"])
def _ple_conv(
    Gated,
    Normed,
    Initial,
    Weight,
    Output,
    Residual,
    NextState,
    LENGTH,
    WIDTH: tl.constexpr,
    KERNEL: tl.constexpr,
    DILATION: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    RETURN_STATE: tl.constexpr,
    INITIAL_BATCH_STRIDE: tl.constexpr,
    INITIAL_ROW_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # The linear offset covers gated.numel() elements; past ~2^31/10240 rows
    # (reachable with max_seq_len=262144 in one unchunked prefill step) an
    # int32 program offset would wrap, so promote before multiplying.
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    if RETURN_STATE:
        # Single-token decode has one full-width output per batch slot.
        token, col = 0, offset
        valid = tl.full((BLOCK,), True, tl.int1)
        length = 1
    else:
        token, col = offset // WIDTH, offset % WIDTH
        valid = token < LENGTH
        length = LENGTH
    # A separate batch axis avoids division by the runtime sequence length.
    batch = tl.program_id(1).to(tl.int64)
    offset += batch * length * WIDTH
    state_len: tl.constexpr = (KERNEL - 1) * DILATION
    if RETURN_STATE:
        state_offset = batch * state_len * WIDTH + col
    acc = tl.full((BLOCK,), 0, tl.float32)
    for k in tl.static_range(KERNEL):
        position = token - state_len + k * DILATION
        current = tl.load(
            Normed + (batch * length + position) * WIDTH + col,
            valid & (position >= 0),
            other=0,
        )
        value = current.to(tl.float32)
        if HAS_INITIAL:
            previous = tl.load(
                Initial
                + batch * INITIAL_BATCH_STRIDE
                + (position + state_len) * INITIAL_ROW_STRIDE
                + col,
                valid & (position < 0),
                other=0,
            )
            value = tl.where(position < 0, previous.to(tl.float32), value)
            if RETURN_STATE and k > 0 and k < KERNEL - 1:
                tl.store(
                    NextState + state_offset + (k * DILATION - 1) * WIDTH,
                    previous,
                    valid,
                )
        if RETURN_STATE and k == KERNEL - 1:
            tl.store(NextState + state_offset + (state_len - 1) * WIDTH, current, valid)
        weight = tl.load(Weight + col * KERNEL + k, valid, other=0).to(tl.float32)
        acc = tl.fma(value, weight, acc)
    _store_ple_output(acc, Gated, Output, Residual, offset, valid, HAS_RESIDUAL)
    if RETURN_STATE:
        # The convolution loads taps 0, 3 and 6 of the previous state. Preserve
        # the intervening rows as well because later tokens will sample them.
        for row in tl.static_range(1, state_len):
            if not HAS_INITIAL or row % DILATION != 0:
                if HAS_INITIAL:
                    history = tl.load(
                        Initial
                        + batch * INITIAL_BATCH_STRIDE
                        + row * INITIAL_ROW_STRIDE
                        + col,
                        valid,
                        other=0,
                    )
                else:
                    history = tl.full((BLOCK,), 0, tl.bfloat16)
                tl.store(NextState + state_offset + (row - 1) * WIDTH, history, valid)


@triton.jit
def _prefill_conv_input(
    Normed,
    Initial,
    batch,
    position,
    col,
    length,
    HAS_INITIAL: tl.constexpr,
    INITIAL_BATCH_STRIDE: tl.constexpr,
    INITIAL_ROW_STRIDE: tl.constexpr,
):
    value = tl.load(
        Normed + (batch * length + position) * 32768 + col,
        (position >= 0) & (position < length),
        other=0,
    ).to(tl.float32)
    if HAS_INITIAL:
        previous = tl.load(
            Initial
            + batch * INITIAL_BATCH_STRIDE
            + (position + 9) * INITIAL_ROW_STRIDE
            + col,
            position < 0,
            other=0,
        ).to(tl.float32)
        value = tl.where(position < 0, previous, value)
    return value


@triton.jit(do_not_specialize=["LENGTH"])
def _ple_conv_prefill_tiled(
    Gated,
    Normed,
    Initial,
    Weight,
    Output,
    Residual,
    LENGTH,
    HAS_INITIAL: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    INITIAL_BATCH_STRIDE: tl.constexpr,
    INITIAL_ROW_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    TIME: tl.constexpr,
    StatePool=None,
    CtxPool=None,
    StateBlocks=None,
    CtxBlocks=None,
    History=None,
    PAGE_SIZE: tl.constexpr = 0,
    STATE_POOL_STRIDE: tl.constexpr = 0,
    CTX_POOL_STRIDE: tl.constexpr = 0,
    CTX_SIZE: tl.constexpr = 0,
    CTX_BLOCK: tl.constexpr = 0,
    HISTORY_STRIDE: tl.constexpr = 0,
    CACHE_BLOCK: tl.constexpr = 2048,
):
    tile = tl.program_id(0).to(tl.int64)
    if PAGE_SIZE:
        groups = (LENGTH + 3 * TIME - 1) // (3 * TIME) * 3
        conv_tiles = groups * (32768 // BLOCK)
        # Separate CTAs keep page metadata and stores out of the convolution loop.
        is_cache_tile = tile >= conv_tiles
    else:
        is_cache_tile = False
    if not is_cache_tile:
        group = tile // (32768 // BLOCK)
        position = group // 3 * (3 * TIME) + group % 3
        col_tile = tile % (32768 // BLOCK)
        col = col_tile * BLOCK + tl.arange(0, BLOCK)
        batch = tl.program_id(1).to(tl.int64)
        w0 = tl.load(Weight + col * 4).to(tl.float32)
        w1 = tl.load(Weight + col * 4 + 1).to(tl.float32)
        w2 = tl.load(Weight + col * 4 + 2).to(tl.float32)
        w3 = tl.load(Weight + col * 4 + 3).to(tl.float32)
        a = _prefill_conv_input(
            Normed,
            Initial,
            batch,
            position - 9,
            col,
            LENGTH,
            HAS_INITIAL,
            INITIAL_BATCH_STRIDE,
            INITIAL_ROW_STRIDE,
        )
        b = _prefill_conv_input(
            Normed,
            Initial,
            batch,
            position - 6,
            col,
            LENGTH,
            HAS_INITIAL,
            INITIAL_BATCH_STRIDE,
            INITIAL_ROW_STRIDE,
        )
        c = _prefill_conv_input(
            Normed,
            Initial,
            batch,
            position - 3,
            col,
            LENGTH,
            HAS_INITIAL,
            INITIAL_BATCH_STRIDE,
            INITIAL_ROW_STRIDE,
        )
        # Follow one dilation phase so the next output reuses three input taps.
        for t in tl.static_range(TIME):
            row = position + 3 * t
            offset = (batch * LENGTH + row) * 32768 + col
            valid = row < LENGTH
            d = tl.load(Normed + offset, valid, other=0).to(tl.float32)
            acc = tl.fma(a, w0, 0.0)
            acc = tl.fma(b, w1, acc)
            acc = tl.fma(c, w2, acc)
            acc = tl.fma(d, w3, acc)
            _store_ple_output(acc, Gated, Output, Residual, offset, valid, HAS_RESIDUAL)
            a, b, c = b, c, d
    elif PAGE_SIZE:
        cache_tile = tile - conv_tiles
        page = cache_tile // (9 * 32768 // CACHE_BLOCK)
        page_tile = cache_tile % (9 * 32768 // CACHE_BLOCK)
        end = tl.minimum((page + 1) * PAGE_SIZE, LENGTH)
        state_block = tl.load(StateBlocks + page).to(tl.int64)
        ctx_block = tl.load(CtxBlocks + page).to(tl.int64)
        if (state_block > 0) & (ctx_block > 0):
            col = page_tile * CACHE_BLOCK + tl.arange(0, CACHE_BLOCK)
            value = tl.load(Normed + (end - 9) * 32768 + col)
            tl.store(StatePool + state_block * STATE_POOL_STRIDE + col, value)
            if page_tile == 0:
                ctx_col = tl.arange(0, CTX_BLOCK)
                mask = ctx_col < CTX_SIZE
                token = tl.load(
                    History + (end + ctx_col) * HISTORY_STRIDE, mask, other=0
                )
                tl.store(CtxPool + ctx_block * CTX_POOL_STRIDE + ctx_col, token, mask)


def try_ple_conv(
    gated,
    normed,
    initial,
    weight,
    dilation,
    *,
    residual=None,
    return_state=False,
    out=None,
    cache_commit=None,
    history=None,
    page_size=128,
):
    """Apply PLE convolution with optional state and destination.

    The destination may be the gated buffer itself. Each element consumes the
    gated value at its own offset before replacing it. Other inputs stay disjoint
    from the destination.
    Prefill cache destinations must have validated IDs and cannot alias inputs.
    """
    if not (
        gated.is_cuda
        and torch.version.hip is None
        and gated.dtype == normed.dtype == weight.dtype == torch.bfloat16
        and gated.ndim == 3
        and gated.shape == normed.shape
        and gated.shape[1] > 0
        and gated.is_contiguous()
        and normed.is_contiguous()
        and weight.shape == (gated.shape[-1], 1, 4)
        and weight.is_contiguous()
        and gated.device == normed.device == weight.device
        and dilation == 3
        and (
            initial is None
            or (
                initial.shape == (gated.shape[0], 9, gated.shape[-1])
                and initial.dtype == gated.dtype
                and initial.device == gated.device
                and initial.stride(-1) == 1
            )
        )
        and not any(x.requires_grad for x in (gated, normed, weight))
        and (
            out is None
            or (
                out.shape == gated.shape
                and out.dtype == gated.dtype
                and out.device == gated.device
                and out.is_contiguous()
                and not out.requires_grad
            )
        )
        and (
            residual is None
            or (
                residual.shape == gated.shape
                and residual.dtype == gated.dtype
                and residual.device == gated.device
                and residual.is_contiguous()
                and not residual.requires_grad
            )
        )
    ):
        return None
    batch, length, width = gated.shape
    capability = torch.cuda.get_device_capability(gated.device)
    warps = 4
    if capability == (12, 0) and width == 10240 and not return_state:
        block = 1024
    elif capability in ((10, 0), (10, 3)) and width == 32768:
        if return_state and length != 1:
            return None
        block = 256
        if return_state and batch >= 128:
            warps = 2
    else:
        return None
    state_pool = ctx_pool = state_blocks = ctx_blocks = None
    if cache_commit is not None:
        state_pool, ctx_pool, state_blocks, ctx_blocks = cache_commit
        if not (
            width == 32768
            and length >= 192
            and batch == 1
            and not return_state
            and page_size == 128
            and history is not None
            and all(
                x.device == gated.device
                for x in (state_pool, ctx_pool, history, state_blocks, ctx_blocks)
            )
            and state_pool.ndim == 3
            and state_pool.shape[1:] == (9, width)
            and state_pool.dtype == gated.dtype
            and state_pool.stride()[1:] == (width, 1)
            and (
                initial is None
                or initial.untyped_storage().data_ptr()
                != state_pool.untyped_storage().data_ptr()
            )
            and ctx_pool.ndim == history.ndim == 2
            and ctx_pool.dtype == history.dtype == torch.int64
            and ctx_pool.stride(1) == 1
            and 0 < ctx_pool.shape[1] <= 32
            and history.shape == (1, length + ctx_pool.shape[1])
            and all(
                x.shape == (triton.cdiv(length, page_size),)
                and x.dtype in (torch.int32, torch.int64)
                and x.is_contiguous()
                for x in (state_blocks, ctx_blocks)
            )
        ):
            return None
    output = torch.empty_like(gated) if out is None else out
    initial_batch_stride = initial.stride(0) if initial is not None else 0
    initial_row_stride = initial.stride(1) if initial is not None else 0
    if width == 32768 and length >= 192:
        block, time = 128, 8
        groups = triton.cdiv(length, 3 * time) * 3
        tiles = groups * (width // block)
        cache_block = 2048
        if cache_commit is not None:
            tiles += triton.cdiv(length, page_size) * (9 * width // cache_block)
        _ple_conv_prefill_tiled[(tiles, batch)](
            gated,
            normed,
            initial if initial is not None else normed,
            weight,
            output,
            residual if residual is not None else gated,
            LENGTH=length,
            HAS_INITIAL=initial is not None,
            HAS_RESIDUAL=residual is not None,
            INITIAL_BATCH_STRIDE=initial_batch_stride,
            INITIAL_ROW_STRIDE=initial_row_stride,
            BLOCK=block,
            TIME=time,
            StatePool=state_pool,
            CtxPool=ctx_pool,
            StateBlocks=state_blocks,
            CtxBlocks=ctx_blocks,
            History=history if cache_commit is not None else None,
            PAGE_SIZE=page_size if cache_commit is not None else 0,
            STATE_POOL_STRIDE=state_pool.stride(0) if state_pool is not None else 0,
            CTX_POOL_STRIDE=ctx_pool.stride(0) if ctx_pool is not None else 0,
            CTX_SIZE=ctx_pool.shape[1] if ctx_pool is not None else 0,
            CTX_BLOCK=(
                triton.next_power_of_2(ctx_pool.shape[1]) if ctx_pool is not None else 0
            ),
            HISTORY_STRIDE=history.stride(1) if cache_commit is not None else 0,
            CACHE_BLOCK=cache_block,
            num_warps=2,
        )
        return output
    next_state = gated.new_empty((batch, 9, width)) if return_state else None
    _ple_conv[(triton.cdiv(length * width, block), batch)](
        gated,
        normed,
        initial if initial is not None else normed,
        weight,
        output,
        residual if residual is not None else gated,
        next_state if return_state else output,
        LENGTH=length,
        WIDTH=width,
        KERNEL=4,
        DILATION=3,
        HAS_INITIAL=initial is not None,
        HAS_RESIDUAL=residual is not None,
        RETURN_STATE=return_state,
        INITIAL_BATCH_STRIDE=initial_batch_stride,
        INITIAL_ROW_STRIDE=initial_row_stride,
        BLOCK=block,
        num_warps=warps,
    )
    return (output, next_state) if return_state else output

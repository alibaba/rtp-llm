"""Accepted-prefix replay shared by KDA and gated delta attention.

State pages contain the committed anchor. A verify window only publishes its
normalized keys, corrected values, activated gates, and raw convolution inputs.
All request metadata remains on the device, including the acceptance length.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.fla.op import exp, softplus


@triton.jit
def _replay_trap():
    # An invalid lease must fail execution even when TRITON_DEBUG is disabled.
    tl.inline_asm_elementwise(
        "trap; mov.u32 $0, 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _replay_row(
    Slot,
    Generation,
    Source,
    Destination,
    Accept,
    HistoryCount,
    HistoryEpoch,
    InitKind,
    PoolGeneration,
    PoolEpoch,
    PoolCount,
    Error,
    row,
    NSLOT: tl.constexpr,
    NBLOCK: tl.constexpr,
    CAPACITY: tl.constexpr,
):
    slot = tl.load(Slot + row).to(tl.int64)
    active = slot != -1
    slot_ok = (slot >= 0) & (slot < NSLOT)
    if active & ~slot_ok:
        tl.atomic_or(Error, 1)
        _replay_trap()
    safe_slot = tl.where(slot_ok, slot, 0)
    source = tl.load(Source + row).to(tl.int64)
    destination = tl.load(Destination + row).to(tl.int64)
    accepted = tl.load(Accept + row)
    count = tl.load(HistoryCount + row)
    init = tl.load(InitKind + row)
    generation = tl.load(Generation + row)
    epoch = tl.load(HistoryEpoch + row)
    old_generation = tl.load(PoolGeneration + safe_slot, mask=slot_ok, other=0)
    old_epoch = tl.load(PoolEpoch + safe_slot, mask=slot_ok, other=0)
    old_count = tl.load(PoolCount + safe_slot, mask=slot_ok, other=0)
    old_error = tl.load(Error + safe_slot, mask=slot_ok, other=0)
    normal = (
        (init == 0)
        & (generation == old_generation)
        & (epoch == old_epoch)
        & (count == old_count)
        & (accepted >= 1)
        & (accepted <= count)
        & (count <= CAPACITY)
    )
    cold = ((init == 1) | (init == 2)) & (accepted == 0) & (count == 0)
    valid = (
        (generation > 0)
        & (old_error == 0)
        & (destination > 0)
        & (destination < NBLOCK)
        & (((source > 0) & (source < NBLOCK)) | (init == 2))
        & (normal | cold)
    )
    if active & ~valid:
        tl.atomic_or(Error + safe_slot, 1)
        _replay_trap()
    return active, safe_slot, source, destination, accepted, init


@triton.jit
def _rebuild_state_kernel(
    State,
    KLog,
    ULog,
    GLog,
    Slot,
    Generation,
    Source,
    Destination,
    Accept,
    HistoryCount,
    HistoryEpoch,
    InitKind,
    PoolGeneration,
    PoolEpoch,
    PoolCount,
    Error,
    STATE_STRIDE: tl.constexpr,
    NSLOT: tl.constexpr,
    NBLOCK: tl.constexpr,
    CAPACITY: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    GD: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    row, hv = tl.program_id(0) // HV, tl.program_id(0) % HV
    tile_v = tl.program_id(1)
    active, slot, source, destination, accepted, init = _replay_row(
        Slot,
        Generation,
        Source,
        Destination,
        Accept,
        HistoryCount,
        HistoryEpoch,
        InitKind,
        PoolGeneration,
        PoolEpoch,
        PoolCount,
        Error,
        row,
        NSLOT,
        NBLOCK,
        CAPACITY,
    )
    if not active:
        return
    hk = hv // (HV // H)
    ik = tl.arange(0, BK)
    iv = tile_v * BV + tl.arange(0, BV)
    state_offset = hv * K * V + ik[None, :] * V + iv[:, None]
    state_mask = (ik[None, :] < K) & (iv[:, None] < V)
    state = tl.load(
        State + source * STATE_STRIDE + state_offset,
        mask=state_mask & (init != 2),
        other=0,
    ).to(tl.float32)
    for t in range(accepted):
        key = tl.load(
            KLog + ((slot * CAPACITY + t) * H + hk) * K + ik,
            mask=ik < K,
            other=0,
        )
        residual = tl.load(
            ULog + ((slot * CAPACITY + t) * HV + hv) * V + iv,
            mask=iv < V,
            other=0,
        )
        if GD == 1:
            gate = tl.load(GLog + (slot * CAPACITY + t) * HV + hv)
            state *= exp(gate)
        else:
            gate = tl.load(
                GLog + ((slot * CAPACITY + t) * HV + hv) * GD + ik,
                mask=ik < K,
                other=0,
            )
            state *= exp(gate[None, :])
        state += residual[:, None] * key[None, :]
    tl.store(State + destination * STATE_STRIDE + state_offset, state, state_mask)


@triton.jit
def _replay_conv_kernel(
    Q,
    KInput,
    VInput,
    Weight,
    State,
    RawLog,
    Output,
    Slot,
    Generation,
    Source,
    Destination,
    Accept,
    HistoryCount,
    HistoryEpoch,
    InitKind,
    PoolGeneration,
    PoolEpoch,
    PoolCount,
    Error,
    Q_STRIDE: tl.constexpr,
    K_STRIDE: tl.constexpr,
    V_STRIDE: tl.constexpr,
    W_STRIDE_D: tl.constexpr,
    W_STRIDE_T: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    STATE_STRIDE_T: tl.constexpr,
    NSLOT: tl.constexpr,
    NBLOCK: tl.constexpr,
    CAPACITY: tl.constexpr,
    T: tl.constexpr,
    QD: tl.constexpr,
    VD: tl.constexpr,
    C: tl.constexpr,
    WIDTH: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    KDA_ARITHMETIC: tl.constexpr,
):
    row = tl.program_id(0)
    active, slot, source, destination, accepted, init = _replay_row(
        Slot,
        Generation,
        Source,
        Destination,
        Accept,
        HistoryCount,
        HistoryEpoch,
        InitKind,
        PoolGeneration,
        PoolEpoch,
        PoolCount,
        Error,
        row,
        NSLOT,
        NBLOCK,
        CAPACITY,
    )
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    w = tl.arange(0, BW)
    if not active:
        for t in tl.static_range(T):
            tl.store(Output + (row * T + t) * C + d, 0, d < C)
        return
    # Read the complete accepted history before overwriting either aliased pool.
    history_index = accepted + w
    previous_t = history_index - (WIDTH - 1)
    history = tl.load(
        State
        + source * STATE_STRIDE
        + history_index[None, :] * STATE_STRIDE_T
        + d[:, None],
        mask=(d[:, None] < C) & (history_index[None, :] < WIDTH - 1) & (init != 2),
        other=0,
    ).to(tl.float32)
    history += tl.load(
        RawLog + (slot * CAPACITY + previous_t[None, :]) * C + d[:, None],
        mask=(d[:, None] < C)
        & (w[None, :] < WIDTH - 1)
        & (previous_t[None, :] >= 0)
        & (previous_t[None, :] < accepted),
        other=0,
    ).to(tl.float32)
    weights = tl.load(
        Weight + d[:, None] * W_STRIDE_D + w[None, :] * W_STRIDE_T,
        mask=(d[:, None] < C) & (w[None, :] < WIDTH),
        other=0,
    ).to(tl.float32)
    tl.debug_barrier()
    tl.store(
        State + destination * STATE_STRIDE + w[None, :] * STATE_STRIDE_T + d[:, None],
        history,
        mask=(d[:, None] < C) & (w[None, :] < WIDTH - 1),
    )
    for t in tl.static_range(T):
        token = row * T + t
        q = tl.load(Q + token * Q_STRIDE + d, mask=d < QD, other=0)
        k = tl.load(
            KInput + token * K_STRIDE + d - QD, mask=(d >= QD) & (d < 2 * QD), other=0
        )
        v = tl.load(
            VInput + token * V_STRIDE + d - 2 * QD,
            mask=(d >= 2 * QD) & (d < C),
            other=0,
        )
        raw = q.to(tl.float32) + k.to(tl.float32) + v.to(tl.float32)
        window = tl.where(w[None, :] == WIDTH - 1, raw[:, None], history)
        if KDA_ARITHMETIC:
            y = tl.sum(window * weights, 1)
            y *= tl.sigmoid(y)
        else:
            # Qwen's causal_conv1d_update accumulates taps chronologically.
            y = tl.full((BD,), 0, tl.float32)
            for tap in tl.static_range(WIDTH):
                sample = tl.sum(tl.where(w[None, :] == tap, window, 0), 1)
                weight = tl.sum(tl.where(w[None, :] == tap, weights, 0), 1)
                y += sample * weight
            y = y / (1 + tl.exp(-y))
        tl.store(Output + token * C + d, y, d < C)
        tl.store(RawLog + (slot * CAPACITY + t) * C + d, raw, d < C)
        shift = tl.broadcast_to(tl.minimum(w + 1, BW - 1)[None, :], (BD, BW))
        history = tl.gather(window, shift, 1)


@triton.jit
def _serial_verify_kernel(
    QKV,
    RawGate,
    RawBeta,
    ALog,
    DtBias,
    State,
    Output,
    KLog,
    ULog,
    GLog,
    Slot,
    Generation,
    Source,
    Destination,
    Accept,
    HistoryCount,
    HistoryEpoch,
    InitKind,
    PoolGeneration,
    PoolEpoch,
    PoolCount,
    Error,
    GATE_STRIDE: tl.constexpr,
    BETA_STRIDE: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    NSLOT: tl.constexpr,
    NBLOCK: tl.constexpr,
    CAPACITY: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    GD: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
):
    row, hv = tl.program_id(0) // HV, tl.program_id(0) % HV
    tile_v = tl.program_id(1)
    active, slot, source, destination, accepted, init = _replay_row(
        Slot,
        Generation,
        Source,
        Destination,
        Accept,
        HistoryCount,
        HistoryEpoch,
        InitKind,
        PoolGeneration,
        PoolEpoch,
        PoolCount,
        Error,
        row,
        NSLOT,
        NBLOCK,
        CAPACITY,
    )
    ik = tl.arange(0, BK)
    iv = tile_v * BV + tl.arange(0, BV)
    if not active:
        for t in tl.static_range(T):
            tl.store(Output + ((row * T + t) * HV + hv) * V + iv, 0, iv < V)
        return
    hk = hv // (HV // H)
    c = 2 * H * K + HV * V
    state = tl.load(
        State + destination * STATE_STRIDE + hv * K * V + ik[None, :] * V + iv[:, None],
        mask=(ik[None, :] < K) & (iv[:, None] < V),
        other=0,
    ).to(tl.float32)
    alog = tl.load(ALog + hv).to(tl.float32)
    if GD == 1:
        bias = tl.load(DtBias + hv).to(tl.float32)
    else:
        bias = tl.load(DtBias + hv * K + ik, ik < K, other=0).to(tl.float32)
    for t in tl.static_range(T):
        token = row * T + t
        q = tl.load(QKV + token * c + hk * K + ik, ik < K, other=0).to(tl.float32)
        k = tl.load(QKV + token * c + H * K + hk * K + ik, ik < K, other=0).to(
            tl.float32
        )
        v = tl.load(QKV + token * c + 2 * H * K + hv * V + iv, iv < V, other=0).to(
            tl.float32
        )
        q /= tl.sqrt(tl.sum(q * q) + 1e-6)
        k /= tl.sqrt(tl.sum(k * k) + 1e-6)
        q *= K**-0.5
        if GD == 1:
            raw_gate = tl.load(RawGate + token * GATE_STRIDE + hv).to(tl.float32)
            x = raw_gate + bias
            gate = -tl.exp(alog) * tl.where(x <= 20, tl.log(1 + tl.exp(x)), x)
            state *= exp(gate)
        else:
            raw_gate = tl.load(
                RawGate + token * GATE_STRIDE + hv * K + ik, ik < K, other=0
            ).to(tl.float32)
            x = raw_gate + bias
            if LOWER_BOUND is None:
                gate = -exp(alog) * softplus(x)
            else:
                gate = LOWER_BOUND * tl.sigmoid(exp(alog) * x)
            state *= exp(gate[None, :])
        beta = tl.sigmoid(tl.load(RawBeta + token * BETA_STRIDE + hv).to(tl.float32))
        if GD == 1:
            beta = beta.to(RawBeta.dtype.element_ty).to(tl.float32)
        residual = (v - tl.sum(state * k[None, :], 1)) * beta
        state += residual[:, None] * k[None, :]
        output = tl.sum(state * q[None, :], 1)
        tl.store(Output + (token * HV + hv) * V + iv, output, iv < V)
        tl.store(ULog + ((slot * CAPACITY + t) * HV + hv) * V + iv, residual, iv < V)
        # Grouped value heads share one key log; each address has one writer.
        if (tile_v == 0) & (hv % (HV // H) == 0):
            tl.store(KLog + ((slot * CAPACITY + t) * H + hk) * K + ik, k, ik < K)
        if tile_v == 0:
            if GD == 1:
                tl.store(GLog + (slot * CAPACITY + t) * HV + hv, gate)
            else:
                tl.store(
                    GLog + ((slot * CAPACITY + t) * HV + hv) * K + ik, gate, ik < K
                )


@triton.jit
def _finalize_kernel(
    Slot,
    Generation,
    Epoch,
    PoolGeneration,
    PoolEpoch,
    PoolCount,
    Error,
    B: tl.constexpr,
    NSLOT: tl.constexpr,
    T: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(Slot + row, row < B, other=-1).to(tl.int64)
    valid = (row < B) & (slot >= 0) & (slot < NSLOT)
    generation = tl.load(Generation + row, valid, other=0)
    epoch = tl.load(Epoch + row, valid, other=0)
    error = tl.load(Error + slot, valid, other=0)
    bad = ((row < B) & (slot != -1) & ~valid) | (
        valid & ((generation <= 0) | (epoch <= 0) | (error != 0))
    )
    if tl.sum(bad.to(tl.int32), 0) > 0:
        _replay_trap()
    tl.store(PoolGeneration + slot, generation, valid)
    tl.store(PoolEpoch + slot, epoch, valid)
    tl.store(PoolCount + slot, T, valid)


def finalize_linear_replay(layer_cache: Any, replay_inputs: Any, steps: int) -> None:
    """Publish shared log headers once, after every target layer has completed."""
    if steps <= 0 or steps > layer_cache.k.shape[1]:
        raise ValueError("LINEAR replay finalize length exceeds the log capacity")
    batch = replay_inputs.slot_ids.numel()
    _finalize_kernel[(triton.cdiv(batch, 128),)](
        replay_inputs.slot_ids,
        replay_inputs.slot_generations,
        replay_inputs.verify_epochs,
        layer_cache.slot_generations,
        layer_cache.log_epochs,
        layer_cache.valid_counts,
        layer_cache.error_flags,
        batch,
        layer_cache.k.shape[0],
        steps,
        128,
        num_warps=4,
    )


def _require_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype | None = None,
    *,
    contiguous: bool = True,
) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device != device
        or tuple(tensor.shape) != shape
        or (dtype is not None and tensor.dtype != dtype)
        or (contiguous and not tensor.is_contiguous())
    ):
        raise ValueError(
            f"invalid LINEAR replay {name}; expected {shape}, {dtype}, {device}"
        )


def linear_serial_replay(
    q_projected: torch.Tensor,
    k_projected: torch.Tensor,
    v_projected: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    conv_weights: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_pool: torch.Tensor,
    conv_pool: torch.Tensor,
    layer_cache: Any,
    replay_inputs: Any,
    *,
    group_id: int,
    vector_gate: bool,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """Rebuild the accepted anchor and verify a dense, short token window."""
    # TODO(decode-cache-return): Save additional accepted LINEAR checkpoints
    # at LINEAR block-size boundaries before enabling decode-cache return.
    # Each checkpoint needs matching SSM/conv state and processed length;
    # the mutable tail anchor and one-round replay log are not sufficient.
    if layer_cache is None or replay_inputs is None:
        raise ValueError("target LINEAR verification requires replay cache and inputs")
    if not q_projected.is_cuda:
        raise ValueError("LINEAR replay requires CUDA inputs")
    device = q_projected.device
    batch = replay_inputs.slot_ids.numel()
    tokens = q_projected.shape[0]
    if batch <= 0 or tokens % batch:
        raise ValueError("LINEAR replay requires an integral verify length per request")
    steps = tokens // batch
    slots, capacity, h, k = layer_cache.k.shape
    _, _, hv, v = layer_cache.u.shape
    gate_dim = k if vector_gate else 1
    channels = 2 * h * k + hv * v
    if steps <= 0 or steps > capacity or hv % h:
        raise ValueError("invalid LINEAR replay window capacity or grouped head counts")
    for name, tensor, shape in (
        ("q", q_projected, (tokens, h * k)),
        ("k", k_projected, (tokens, h * k)),
        ("v", v_projected, (tokens, hv * v)),
        ("gate", raw_gate, (tokens, hv * gate_dim)),
        ("beta", raw_beta, (tokens, hv)),
    ):
        _require_tensor(name, tensor, shape, device, contiguous=False)
        if tensor.stride(-1) != 1:
            raise ValueError(f"LINEAR replay {name} requires unit innermost stride")
    for name, shape, dtype in (
        ("k", (slots, capacity, h, k), torch.float32),
        ("u", (slots, capacity, hv, v), torch.float32),
        ("g", (slots, capacity, hv, gate_dim), torch.float32),
        ("conv_inputs", (slots, capacity, channels), q_projected.dtype),
        ("slot_generations", (slots,), torch.int64),
        ("log_epochs", (slots,), torch.int64),
        ("valid_counts", (slots,), torch.int32),
        ("error_flags", (slots,), torch.int32),
    ):
        _require_tensor(name, getattr(layer_cache, name), shape, device, dtype)
    for name in (
        "slot_ids",
        "slot_generations",
        "prev_accept_lengths",
        "history_valid_lengths",
        "history_epochs",
        "verify_epochs",
        "init_kinds",
    ):
        dtype = (
            torch.int64
            if name in ("slot_generations", "history_epochs", "verify_epochs")
            else torch.int32
        )
        _require_tensor(name, getattr(replay_inputs, name), (batch,), device, dtype)
    for name in ("state_read_block_ids", "active_block_ids"):
        tensor = getattr(replay_inputs, name)
        if tensor.ndim != 2 or group_id < 0 or tensor.shape[0] <= group_id:
            raise ValueError(
                f"LINEAR replay {name} does not contain the layer's cache group"
            )
        _require_tensor(
            name,
            tensor,
            (tensor.shape[0], batch),
            device,
            torch.int32,
            contiguous=False,
        )
        if tensor.stride(-1) != 1:
            raise ValueError(f"LINEAR replay {name} requires contiguous request rows")
    blocks = ssm_pool.shape[0]
    width = conv_pool.shape[1] + 1
    _require_tensor(
        "SSM", ssm_pool, (blocks, hv, k, v), device, torch.float32, contiguous=False
    )
    _require_tensor(
        "conv",
        conv_pool,
        (blocks, width - 1, channels),
        device,
        q_projected.dtype,
        contiguous=False,
    )
    if ssm_pool.stride()[1:] != (k * v, v, 1) or conv_pool.stride()[1:] != (
        channels,
        1,
    ):
        raise ValueError(
            "LINEAR replay requires canonical contiguous state within each page"
        )
    _require_tensor(
        "conv weights", conv_weights, (channels, width), device, contiguous=False
    )
    _require_tensor("A_log", a_log, (hv,), device)
    if (
        dt_bias.numel() != hv * gate_dim
        or dt_bias.device != device
        or not dt_bias.is_contiguous()
    ):
        raise ValueError("LINEAR replay dt_bias shape does not match its gate")
    source = replay_inputs.state_read_block_ids[group_id]
    destination = replay_inputs.active_block_ids[group_id]
    row_args = (
        replay_inputs.slot_ids,
        replay_inputs.slot_generations,
        source,
        destination,
        replay_inputs.prev_accept_lengths,
        replay_inputs.history_valid_lengths,
        replay_inputs.history_epochs,
        replay_inputs.init_kinds,
        layer_cache.slot_generations,
        layer_cache.log_epochs,
        layer_cache.valid_counts,
        layer_cache.error_flags,
    )
    tile_k = triton.next_power_of_2(k)
    tile_v = 32 if vector_gate else 8
    grid = (batch * hv, triton.cdiv(v, tile_v))
    warps = 4 if vector_gate else 1
    _rebuild_state_kernel[grid](
        ssm_pool,
        layer_cache.k,
        layer_cache.u,
        layer_cache.g,
        *row_args,
        ssm_pool.stride(0),
        slots,
        blocks,
        capacity,
        h,
        hv,
        k,
        v,
        gate_dim,
        tile_k,
        tile_v,
        num_warps=warps,
    )
    convolved = torch.empty((tokens, channels), device=device, dtype=q_projected.dtype)
    _replay_conv_kernel[(batch, triton.cdiv(channels, 128))](
        q_projected,
        k_projected,
        v_projected,
        conv_weights,
        conv_pool,
        layer_cache.conv_inputs,
        convolved,
        *row_args,
        q_projected.stride(0),
        k_projected.stride(0),
        v_projected.stride(0),
        conv_weights.stride(0),
        conv_weights.stride(1),
        conv_pool.stride(0),
        conv_pool.stride(1),
        slots,
        blocks,
        capacity,
        steps,
        h * k,
        hv * v,
        channels,
        width,
        triton.next_power_of_2(width),
        128,
        vector_gate,
        num_warps=4,
    )
    output = torch.empty((tokens, hv, v), device=device, dtype=q_projected.dtype)
    _serial_verify_kernel[grid](
        convolved,
        raw_gate,
        raw_beta,
        a_log,
        dt_bias,
        ssm_pool,
        output,
        layer_cache.k,
        layer_cache.u,
        layer_cache.g,
        *row_args,
        raw_gate.stride(0),
        raw_beta.stride(0),
        ssm_pool.stride(0),
        slots,
        blocks,
        capacity,
        steps,
        h,
        hv,
        k,
        v,
        gate_dim,
        tile_k,
        tile_v,
        lower_bound,
        num_warps=warps,
    )
    return output


__all__ = ["finalize_linear_replay", "linear_serial_replay"]

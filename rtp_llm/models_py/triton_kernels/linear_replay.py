"""Accepted-prefix replay for CUDA LINEAR attention implementations.

State pages contain the committed anchor. Each leased request slot retains the
last verify's keys, corrected values, gates, and raw convolution inputs until the
next verify replays its accepted prefix before overwriting those records.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.fla.op import exp, softplus


@triton.jit
def _replay_trap():
    # Invalid replay metadata must fail even when TRITON_DEBUG is disabled.
    tl.inline_asm_elementwise(
        "trap; mov.u32 $0, 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _replay_row(
    Slot,
    Source,
    Destination,
    Accept,
    HistoryCount,
    InitKind,
    row,
    NSLOT: tl.constexpr,
    NBLOCK: tl.constexpr,
    CAPACITY: tl.constexpr,
):
    slot = tl.load(Slot + row).to(tl.int64)
    active = slot != -1
    slot_ok = (slot >= 0) & (slot < NSLOT)
    if active & ~slot_ok:
        _replay_trap()
    safe_slot = tl.where(slot_ok, slot, 0)
    source = tl.load(Source + row).to(tl.int64)
    destination = tl.load(Destination + row).to(tl.int64)
    accepted = tl.load(Accept + row)
    count = tl.load(HistoryCount + row)
    init = tl.load(InitKind + row)
    normal = (init == 0) & (accepted >= 1) & (accepted <= count) & (count <= CAPACITY)
    cold = ((init == 1) | (init == 2)) & (accepted == 0) & (count == 0)
    valid = (
        (destination > 0)
        & (destination < NBLOCK)
        & (((source > 0) & (source < NBLOCK)) | (init == 2))
        & (normal | cold)
    )
    if active & ~valid:
        _replay_trap()
    return active, safe_slot, source, destination, accepted, init


@triton.jit
def _rebuild_state_tile(
    State,
    KLog,
    ULog,
    GLog,
    slot,
    kg_slot,
    source,
    destination,
    accepted,
    init,
    hv,
    hk,
    ik,
    iv,
    state_offset,
    state_mask,
    STATE_STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    GD: tl.constexpr,
):
    state = tl.load(
        State + source * STATE_STRIDE + state_offset,
        mask=state_mask & (init != 2),
        other=0,
    ).to(tl.float32)
    for t in range(accepted):
        key = tl.load(
            KLog + ((kg_slot * CAPACITY + t) * H + hk) * K + ik,
            mask=ik < K,
            other=0,
        )
        residual = tl.load(
            ULog + ((slot * CAPACITY + t) * HV + hv) * V + iv,
            mask=iv < V,
            other=0,
        )
        if GD == 1:
            gate = tl.load(GLog + (kg_slot * CAPACITY + t) * HV + hv)
            state *= exp(gate)
        else:
            gate = tl.load(
                GLog + ((kg_slot * CAPACITY + t) * HV + hv) * GD + ik,
                mask=ik < K,
                other=0,
            )
            state *= exp(gate[None, :])
        state = tl.fma(residual[:, None], key[None, :], state)
    tl.store(State + destination * STATE_STRIDE + state_offset, state, state_mask)
    # Keep the same rounding as a separate load of the committed page.
    return state.to(State.dtype.element_ty).to(tl.float32)


@triton.jit
def _rebuild_state_kernel(
    State,
    KLog,
    ULog,
    GLog,
    Slot,
    Source,
    Destination,
    Accept,
    HistoryCount,
    InitKind,
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
    STATE_V_FIRST: tl.constexpr,
):
    row, hv = tl.program_id(0) // HV, tl.program_id(0) % HV
    tile_v = tl.program_id(1)
    active, slot, source, destination, accepted, init = _replay_row(
        Slot,
        Source,
        Destination,
        Accept,
        HistoryCount,
        InitKind,
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
    # The recurrence register is [value, key]. Its cache address layout is
    # selected by the production attention operator for this layer.
    if STATE_V_FIRST:
        state_offset = hv * V * K + iv[:, None] * K + ik[None, :]
    else:
        state_offset = hv * V * K + ik[None, :] * V + iv[:, None]
    state_mask = (ik[None, :] < K) & (iv[:, None] < V)
    _rebuild_state_tile(
        State,
        KLog,
        ULog,
        GLog,
        slot,
        slot,
        source,
        destination,
        accepted,
        init,
        hv,
        hk,
        ik,
        iv,
        state_offset,
        state_mask,
        STATE_STRIDE,
        CAPACITY,
        H,
        HV,
        K,
        V,
        GD,
    )


@triton.jit
def _load_conv_replay_history(
    State,
    RawLog,
    channel,
    tap,
    source,
    slot,
    accepted,
    init,
    STATE_STRIDE: tl.constexpr,
    STATE_STRIDE_T: tl.constexpr,
    CAPACITY: tl.constexpr,
    C: tl.constexpr,
    WIDTH: tl.constexpr,
):
    history_index = accepted + tap
    previous_t = history_index - (WIDTH - 1)
    mask = (channel < C) & (tap < WIDTH - 1)
    history = tl.load(
        State + source * STATE_STRIDE + history_index * STATE_STRIDE_T + channel,
        mask & (history_index < WIDTH - 1) & (init != 2),
        other=0,
    ).to(tl.float32)
    history += tl.load(
        RawLog + (slot * CAPACITY + previous_t) * C + channel,
        mask & (previous_t >= 0) & (previous_t < accepted),
        other=0,
    ).to(tl.float32)
    return history


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
    Source,
    Destination,
    Accept,
    HistoryCount,
    InitKind,
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
    C: tl.constexpr,
    WIDTH: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    DIRECT_TAPS: tl.constexpr,
):
    row = tl.program_id(0)
    active, slot, source, destination, accepted, init = _replay_row(
        Slot,
        Source,
        Destination,
        Accept,
        HistoryCount,
        InitKind,
        row,
        NSLOT,
        NBLOCK,
        CAPACITY,
    )
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    if not active:
        for t in tl.static_range(T):
            tl.store(Output + (row * T + t) * C + d, 0, d < C)
        return

    state_out = State + destination * STATE_STRIDE
    weight_base = Weight + d * W_STRIDE_D
    # Read the complete accepted history before overwriting either aliased pool.
    if DIRECT_TAPS:
        # Width 4 uses a three-sample delay line per channel. Keep each tap
        # in a vector of channels so shifting it needs no cross-lane gather.
        h0 = _load_conv_replay_history(
            State,
            RawLog,
            d,
            0,
            source,
            slot,
            accepted,
            init,
            STATE_STRIDE,
            STATE_STRIDE_T,
            CAPACITY,
            C,
            WIDTH,
        )
        h1 = _load_conv_replay_history(
            State,
            RawLog,
            d,
            1,
            source,
            slot,
            accepted,
            init,
            STATE_STRIDE,
            STATE_STRIDE_T,
            CAPACITY,
            C,
            WIDTH,
        )
        h2 = _load_conv_replay_history(
            State,
            RawLog,
            d,
            2,
            source,
            slot,
            accepted,
            init,
            STATE_STRIDE,
            STATE_STRIDE_T,
            CAPACITY,
            C,
            WIDTH,
        )
        w0 = tl.load(weight_base, d < C, other=0).to(tl.float32)
        w1 = tl.load(weight_base + W_STRIDE_T, d < C, other=0).to(tl.float32)
        w2 = tl.load(weight_base + 2 * W_STRIDE_T, d < C, other=0).to(tl.float32)
        w3 = tl.load(weight_base + 3 * W_STRIDE_T, d < C, other=0).to(tl.float32)
        tl.debug_barrier()
        tl.store(state_out + d, h0, d < C)
        tl.store(state_out + STATE_STRIDE_T + d, h1, d < C)
        tl.store(state_out + 2 * STATE_STRIDE_T + d, h2, d < C)
    else:
        w = tl.arange(0, BW)
        history = _load_conv_replay_history(
            State,
            RawLog,
            d[:, None],
            w[None, :],
            source,
            slot,
            accepted,
            init,
            STATE_STRIDE,
            STATE_STRIDE_T,
            CAPACITY,
            C,
            WIDTH,
        )
        weights = tl.load(
            weight_base[:, None] + w[None, :] * W_STRIDE_T,
            mask=(d[:, None] < C) & (w[None, :] < WIDTH),
            other=0,
        ).to(tl.float32)
        tl.debug_barrier()
        tl.store(
            state_out + w[None, :] * STATE_STRIDE_T + d[:, None],
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
        # Match causal_conv1d_update's tap order and SiLU arithmetic.
        y = tl.full((BD,), 0, tl.float32)
        if DIRECT_TAPS:
            y += h0.to(State.dtype.element_ty) * w0.to(Weight.dtype.element_ty)
            y += h1.to(State.dtype.element_ty) * w1.to(Weight.dtype.element_ty)
            y += h2.to(State.dtype.element_ty) * w2.to(Weight.dtype.element_ty)
            y += raw.to(State.dtype.element_ty) * w3.to(Weight.dtype.element_ty)
            h0, h1, h2 = h1, h2, raw
        else:
            window = tl.where(w[None, :] == WIDTH - 1, raw[:, None], history)
            for tap in tl.static_range(WIDTH):
                sample = tl.sum(tl.where(w[None, :] == tap, window, 0), 1).to(
                    State.dtype.element_ty
                )
                weight = tl.sum(tl.where(w[None, :] == tap, weights, 0), 1).to(
                    Weight.dtype.element_ty
                )
                y += sample * weight
            shift = tl.broadcast_to(tl.minimum(w + 1, BW - 1)[None, :], (BD, BW))
            history = tl.gather(window, shift, 1)
        y = y / (1 + tl.exp(-y))
        tl.store(Output + token * C + d, y, d < C)
        tl.store(RawLog + (slot * CAPACITY + t) * C + d, raw, d < C)


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
    Source,
    Destination,
    Accept,
    InitKind,
    HistoryLogBanks,
    GATE_STRIDE: tl.constexpr,
    BETA_STRIDE: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    NSLOT: tl.constexpr,
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
    STATE_V_FIRST: tl.constexpr,
    FUSE_REBUILD: tl.constexpr,
):
    row, hv = tl.program_id(0) // HV, tl.program_id(0) % HV
    tile_v = tl.program_id(1)
    # Convolution validated this immutable metadata earlier on the same stream.
    slot = tl.load(Slot + row).to(tl.int64)
    ik = tl.arange(0, BK)
    iv = tile_v * BV + tl.arange(0, BV)
    if slot == -1:
        for t in tl.static_range(T):
            tl.store(Output + ((row * T + t) * HV + hv) * V + iv, 0, iv < V)
        return
    destination = tl.load(Destination + row).to(tl.int64)
    hk = hv // (HV // H)
    c = 2 * H * K + HV * V
    if STATE_V_FIRST:
        state_offset = hv * V * K + iv[:, None] * K + ik[None, :]
    else:
        state_offset = hv * V * K + ik[None, :] * V + iv[:, None]
    state_mask = (ik[None, :] < K) & (iv[:, None] < V)
    if FUSE_REBUILD:
        bank = tl.load(HistoryLogBanks + row).to(tl.int64)
        if (bank < 0) | (bank > 1):
            _replay_trap()
        source = tl.load(Source + row).to(tl.int64)
        accepted = tl.load(Accept + row)
        init = tl.load(InitKind + row)
        history_kg_slot = slot + bank * NSLOT
        output_kg_slot = slot + (1 - bank) * NSLOT
        # ULog is private to this value tile. Shared K/G writes use the other
        # bank because sibling CTAs can still be replaying the old window.
        state = _rebuild_state_tile(
            State,
            KLog,
            ULog,
            GLog,
            slot,
            history_kg_slot,
            source,
            destination,
            accepted,
            init,
            hv,
            hk,
            ik,
            iv,
            state_offset,
            state_mask,
            STATE_STRIDE,
            CAPACITY,
            H,
            HV,
            K,
            V,
            GD,
        )
    else:
        output_kg_slot = slot
        state = tl.load(
            State + destination * STATE_STRIDE + state_offset, state_mask, other=0
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
        state = tl.fma(residual[:, None], k[None, :], state)
        output = tl.sum(state * q[None, :], 1)
        tl.store(Output + (token * HV + hv) * V + iv, output, iv < V)
        tl.store(ULog + ((slot * CAPACITY + t) * HV + hv) * V + iv, residual, iv < V)
        # Grouped value heads share one key log; each address has one writer.
        if (tile_v == 0) & (hv % (HV // H) == 0):
            tl.store(
                KLog + ((output_kg_slot * CAPACITY + t) * H + hk) * K + ik, k, ik < K
            )
        if tile_v == 0:
            if GD == 1:
                tl.store(GLog + (output_kg_slot * CAPACITY + t) * HV + hv, gate)
            else:
                tl.store(
                    GLog + ((output_kg_slot * CAPACITY + t) * HV + hv) * K + ik,
                    gate,
                    ik < K,
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


class LinearReplay:
    """Validate and bind immutable layer resources before replaying requests."""

    def __init__(
        self,
        conv_weights: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_pool: torch.Tensor,
        conv_pool: torch.Tensor,
        layer_cache: Any,
        *,
        group_id: int,
        vector_gate: bool,
        state_v_first: bool = True,
        lower_bound: float | None = None,
    ):
        if layer_cache is None:
            raise ValueError(
                "target LINEAR verification requires replay cache and inputs"
            )
        device = ssm_pool.device
        kg_slots, capacity, h, k = layer_cache.k.shape
        slots, _, hv, v = layer_cache.u.shape
        banked_kg = vector_gate and kg_slots == 2 * slots
        if not banked_kg and kg_slots != slots:
            raise ValueError("invalid LINEAR replay K/G bank geometry")
        gate_dim = k if vector_gate else 1
        channels = 2 * h * k + hv * v
        if min(slots, capacity, h, k, hv, v) <= 0 or hv % h:
            raise ValueError(
                "invalid LINEAR replay window capacity or grouped head counts"
            )
        for name, shape, dtype in (
            ("k", (kg_slots, capacity, h, k), torch.float32),
            ("u", (slots, capacity, hv, v), torch.float32),
            ("g", (kg_slots, capacity, hv, gate_dim), torch.float32),
            ("conv_inputs", (slots, capacity, channels), conv_pool.dtype),
        ):
            _require_tensor(name, getattr(layer_cache, name), shape, device, dtype)
        blocks = ssm_pool.shape[0]
        width = conv_pool.shape[1] + 1
        state_shape = (blocks, hv, v, k) if state_v_first else (blocks, hv, k, v)
        _require_tensor("SSM", ssm_pool, state_shape, device, contiguous=False)
        if ssm_pool.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError("LINEAR replay supports FP32, BF16 or FP16 SSM state")
        _require_tensor(
            "conv", conv_pool, (blocks, width - 1, channels), device, contiguous=False
        )
        state_strides = (v * k, k, 1) if state_v_first else (v * k, v, 1)
        if ssm_pool.stride()[1:] != state_strides or conv_pool.stride()[1:] != (
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
        self.layer_cache = layer_cache
        self.base: torch.Tensor | None = None
        self.group_id = group_id
        self.qkv_sizes = (h * k, h * k, hv * v)
        self.projected = partial(
            _linear_serial_replay,
            conv_weights=conv_weights,
            a_log=a_log,
            dt_bias=dt_bias,
            ssm_pool=ssm_pool,
            conv_pool=conv_pool,
            layer_cache=layer_cache,
            group_id=group_id,
            vector_gate=vector_gate,
            state_v_first=state_v_first,
            lower_bound=lower_bound,
        )

    @classmethod
    def bind(
        cls,
        previous: LinearReplay | None,
        attention: Any,
        kv_cache: Any,
        **options: Any,
    ) -> LinearReplay:
        if (
            previous is not None
            and previous.base is kv_cache.kv_cache_base
            and previous.group_id == kv_cache.group_id
            and kv_cache.linear_replay is not None
            and previous.layer_cache.k is kv_cache.linear_replay.k
        ):
            return previous
        base = kv_cache.kv_cache_base
        cache = base.reshape(base.shape[0], -1)
        replay = cls(
            attention.conv_weights,
            attention.alog,
            attention.dt_bias,
            attention._get_ssm_states(cache),
            attention._get_conv_states(cache),
            kv_cache.linear_replay,
            group_id=kv_cache.group_id,
            **options,
        )
        replay.base = base
        return replay

    def __call__(
        self,
        mixed_qkv: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        replay_inputs: Any,
    ) -> torch.Tensor:
        q, k, v = torch.split(mixed_qkv, self.qkv_sizes, dim=-1)
        return self.projected(q, k, v, raw_gate, raw_beta, replay_inputs=replay_inputs)


def _linear_serial_replay(
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
    state_v_first: bool = True,
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
    kg_slots, capacity, h, k = layer_cache.k.shape
    slots, _, hv, v = layer_cache.u.shape
    banked_kg = vector_gate and kg_slots == 2 * slots
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
    if q_projected.device != ssm_pool.device or q_projected.dtype != conv_pool.dtype:
        raise ValueError("LINEAR replay inputs must match the bound state pools")
    for name in (
        "slot_ids",
        "prev_accept_lengths",
        "history_valid_lengths",
        "history_log_banks",
        "init_kinds",
    ):
        _require_tensor(
            name, getattr(replay_inputs, name), (batch,), device, torch.int32
        )
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
    source = replay_inputs.state_read_block_ids[group_id]
    destination = replay_inputs.active_block_ids[group_id]
    row_args = (
        replay_inputs.slot_ids,
        source,
        destination,
        replay_inputs.prev_accept_lengths,
        replay_inputs.history_valid_lengths,
        replay_inputs.init_kinds,
    )
    tile_k = triton.next_power_of_2(k)
    tile_v = 32 if vector_gate else 8
    grid = (batch * hv, triton.cdiv(v, tile_v))
    warps = 4 if vector_gate else 1
    if not banked_kg:
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
            state_v_first,
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
        channels,
        width,
        triton.next_power_of_2(width),
        128,
        banked_kg and width == 4,
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
        replay_inputs.slot_ids,
        source,
        destination,
        replay_inputs.prev_accept_lengths,
        replay_inputs.init_kinds,
        replay_inputs.history_log_banks,
        raw_gate.stride(0),
        raw_beta.stride(0),
        ssm_pool.stride(0),
        slots,
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
        state_v_first,
        banked_kg,
        num_warps=warps,
    )
    return output


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
    state_v_first: bool = True,
    lower_bound: float | None = None,
) -> torch.Tensor:
    replay = LinearReplay(
        conv_weights,
        a_log,
        dt_bias,
        ssm_pool,
        conv_pool,
        layer_cache,
        group_id=group_id,
        vector_gate=vector_gate,
        state_v_first=state_v_first,
        lower_bound=lower_bound,
    )
    return replay.projected(
        q_projected,
        k_projected,
        v_projected,
        raw_gate,
        raw_beta,
        replay_inputs=replay_inputs,
    )


__all__ = ["LinearReplay", "linear_serial_replay"]

"""Read-only QSA target snapshot and norm/RoPE preparation, then original writer.

Every snapshot finishes in the first launch. The independent second launch
retains the writer's launch options and BF16/state/undo boundaries.
"""

import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .indexer import is_qsa_rope_style
from .indexer_compressor import IndexerCacheUndo
from .target_writer_triton import _snapshot_target_window, _write_target_window


@triton.jit
def _prepare_target_window(
    X,
    QG,
    QP,
    KP,
    FREQ,
    ROTATED,
    COS,
    SIN,
    PREFIX,
    KV,
    KT,
    STATE,
    ST,
    SS,
    OS,
    KS,
    OK,
    Q: tl.constexpr,
    H: tl.constexpr,
    XS: tl.constexpr,
    PS: tl.constexpr,
    INV_SCALE: tl.constexpr,
    EPS: tl.constexpr,
    KTW: tl.constexpr,
    STW: tl.constexpr,
    KB: tl.constexpr,
    SB: tl.constexpr,
    BQ: tl.constexpr,
):
    request = tl.program_id(0)
    local = tl.program_id(1)
    step = local // H
    head = local % H
    row = request * Q + step
    if local == 0:
        _snapshot_target_window(
            PREFIX,
            KV,
            KT,
            STATE,
            ST,
            SS,
            OS,
            KS,
            OK,
            Q=Q,
            KTW=KTW,
            STW=STW,
            KB=KB,
            SB=SB,
            BQ=BQ,
        )
    lane = tl.arange(0, 32)
    col = lane * 4
    offset = row * XS + head * 128
    a = tl.load(X + offset + col).to(tl.float32)
    b = tl.load(X + offset + col + 1).to(tl.float32)
    c = tl.load(X + offset + col + 2).to(tl.float32)
    d = tl.load(X + offset + col + 3).to(tl.float32)
    inv = tl.rsqrt(tl.sum(((a * a + b * b) + c * c) + d * d, 0) / 128.0 + EPS)
    position = tl.load(QP + row * PS).to(tl.float32)
    frequency = tl.load(FREQ + lane)
    angle = (position * INV_SCALE) * frequency
    cosine = libdevice.cos(angle).to(tl.bfloat16).to(tl.float32)
    sine = libdevice.sin(angle).to(tl.bfloat16).to(tl.float32)
    for j in tl.static_range(4):
        i = col + j
        partner = tl.where(i < 32, i + 32, i - 32)
        partner = tl.where(i < 64, partner, i)
        gain = 1.0 + tl.load(QG + i).to(tl.float32)
        other_gain = 1.0 + tl.load(QG + partner).to(tl.float32)
        value = tl.load(X + offset + i).to(tl.float32)
        other = tl.load(X + offset + partner).to(tl.float32)
        normal = ((value * inv) * gain).to(tl.bfloat16).to(tl.float32)
        rotated = ((other * inv) * other_gain).to(tl.bfloat16).to(tl.float32)
        fi = i % 32
        cv = tl.gather(cosine, fi, 0)
        sv = tl.gather(sine, fi, 0)
        left = (normal * cv).to(tl.bfloat16).to(tl.float32)
        right = (
            (rotated * tl.where(i < 32, -1.0, 1.0) * sv).to(tl.bfloat16).to(tl.float32)
        )
        result = tl.where(
            i < 64, (left + right).to(tl.bfloat16), normal.to(tl.bfloat16)
        )
        tl.store(ROTATED + (row * H + head) * 128 + i, result)
    if head == 0:
        kp = tl.load(KP + row * PS).to(tl.float32)
        ka = (kp * INV_SCALE) * frequency
        kc = libdevice.cos(ka)
        ks = libdevice.sin(ka)
        tl.store(COS + row * 64 + lane, kc)
        tl.store(COS + row * 64 + lane + 32, kc)
        tl.store(SIN + row * 64 + lane, ks)
        tl.store(SIN + row * 64 + lane + 32, ks)


def try_prepare_target_window(
    q,
    raw,
    prefixes,
    query_positions,
    block_starts,
    frequency,
    q_gamma,
    k_gamma,
    kv,
    kt,
    state,
    st,
    *,
    rope_config,
    norm_eps,
):
    """Return None before writes unless all bounded target layouts qualify.

    The model's existing target plan validates logical/physical ownership before
    this call. These shape/layout gates also permit standalone replay tests.
    """
    if not (
        q.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(q.device) == (12, 0)
        and q.ndim == 3
        and q.shape[1:] == (4, 128)
        and 1 <= q.shape[0] <= 32
        and q.dtype == torch.bfloat16
        and q.stride(-1) == 1
        and q.stride(-2) == 128
        and q.stride(0) >= 512
        and raw.shape == (q.shape[0], 128)
        and raw.dtype == torch.bfloat16
        and raw.stride(1) == 1
        and raw.stride(0) >= 128
        and prefixes.ndim == 1
        and prefixes.numel() > 0
        and q.shape[0] % prefixes.numel() == 0
        and 1 <= q.shape[0] // prefixes.numel() <= 4
        and prefixes.dtype == torch.int32
        and query_positions.shape == block_starts.shape == (q.shape[0],)
        and query_positions.dtype == block_starts.dtype
        and query_positions.dtype in (torch.int32, torch.int64)
        and isinstance(frequency, torch.Tensor)
        and frequency.shape == (32,)
        and frequency.dtype == torch.float32
        and q_gamma.shape == k_gamma.shape == (128,)
        and q_gamma.dtype in (torch.bfloat16, torch.float32)
        and k_gamma.dtype == torch.bfloat16
        and kv.ndim == state.ndim == 3
        and kv.shape[1:] == (32, 128)
        and kv.dtype == torch.bfloat16
        and state.shape[1:] == (8, 128)
        and state.dtype == torch.float32
        and kt.ndim == st.ndim == 2
        and kt.shape[0] == st.shape[0] == prefixes.numel()
        and kt.dtype == st.dtype == torch.int32
        and min(kv.shape[0], state.shape[0], kt.shape[1], st.shape[1]) > 0
        and all(
            t.device == q.device and t.is_contiguous()
            for t in (
                prefixes,
                query_positions,
                block_starts,
                frequency,
                q_gamma,
                k_gamma,
                kv,
                kt,
                state,
                st,
            )
        )
        and not any(t.requires_grad for t in (q, raw, q_gamma, k_gamma, kv, state))
        and isinstance(norm_eps, (float, int))
        and math.isfinite(norm_eps)
        and norm_eps >= 0
        and int(rope_config.dim) == 64
        and bool(getattr(rope_config, "indexer_is_neox_style", True))
    ):
        return None
    factor = int(rope_config.index_factor)
    if is_qsa_rope_style(rope_config, "Base"):
        if factor not in (1, 3):
            return None
    elif is_qsa_rope_style(rope_config, "Mrope"):
        sections = (
            int(rope_config.mrope_dim1),
            int(rope_config.mrope_dim2),
            int(rope_config.mrope_dim3),
        )
        if not (
            factor == 3
            and bool(rope_config.mrope_interleaved)
            and min(sections) >= 0
            and sum(sections) == 32
            and sections[1] <= 11
            and sections[2] <= 10
        ):
            return None
    else:
        return None
    scale = float(getattr(rope_config, "scale", 1.0))
    base = float(rope_config.base)
    if not (math.isfinite(scale) and scale > 0 and math.isfinite(base) and base > 0):
        return None
    batch = prefixes.numel()
    query = q.shape[0] // batch
    rotated = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    cosine = torch.empty((q.shape[0], 64), device=q.device, dtype=q.dtype)
    sine = torch.empty_like(cosine)
    state_slots = torch.empty(q.shape[0], device=q.device, dtype=torch.int64)
    old_state = torch.empty((q.shape[0], 128), device=q.device, dtype=state.dtype)
    kv_slots = torch.empty(batch, device=q.device, dtype=torch.int64)
    old_kv = torch.empty((batch, 128), device=q.device, dtype=kv.dtype)
    parameters = dict(
        Q=query,
        KTW=kt.shape[1],
        STW=st.shape[1],
        KB=kv.shape[0],
        SB=state.shape[0],
        BQ=triton.next_power_of_2(query),
    )
    _prepare_target_window[(batch, query * 4)](
        q,
        q_gamma,
        query_positions,
        block_starts,
        frequency,
        rotated,
        cosine,
        sine,
        prefixes,
        kv,
        kt,
        state,
        st,
        state_slots,
        old_state,
        kv_slots,
        old_kv,
        H=4,
        XS=q.stride(0),
        PS=1,
        INV_SCALE=1.0 / scale,
        EPS=norm_eps,
        num_warps=1,
        enable_fp_fusion=False,
        **parameters,
    )
    # Keep precisely the original writer configuration, including fusion.
    _write_target_window[(batch,)](
        raw,
        prefixes,
        cosine,
        sine,
        k_gamma,
        kv,
        kt,
        state,
        st,
        RAW_STRIDE=raw.stride(0),
        EPS=norm_eps,
        num_warps=4,
        **parameters,
    )
    return rotated, IndexerCacheUndo(
        state_pool=state,
        state_slots=state_slots,
        original_state=old_state,
        kv_pool=kv,
        kv_slots=kv_slots,
        original_kv=old_kv,
        padded_slots=True,
    )

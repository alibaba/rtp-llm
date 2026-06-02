# Gluon implementation of chunk_gated_delta_rule_fwd_h for AMD CDNA4 (MI355X/gfx950)
# Optimized for TP2 (H≤32) with T≤65536. Falls back to Triton for other configs.
# Key optimizations: convert_layout for narrow tiles, k_width=8, manual buffer_load prefetch.

from typing import Optional, Tuple

import torch
import triton

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    GLUON_AVAILABLE = True
except ImportError:
    GLUON_AVAILABLE = False

from rtp_llm.models_py.triton_kernels.fla.index import prepare_chunk_offsets

# Maximum H and T for Gluon advantage (empirically determined)
GLUON_MAX_H = 32
# Block tile width hardcoded inside the kernel (two 64-wide tiles when K>64).
# K must be exactly 64 or 128, otherwise the kernel will read past the K
# dimension and segfault / corrupt memory.
GLUON_K_ALLOWED = (64, 128)
# BV is hardcoded to 16 in the kernel and there are no per-element V masks
# on the buffer_load/store ops, so V must be a multiple of BV.
GLUON_BV = 16
# All shared-memory layouts and dot-operand casts inside the kernel are
# pinned to bfloat16. Feeding fp16 (or anything else) would silently
# bit-reinterpret the data and produce scrambled outputs without raising.
GLUON_SUPPORTED_DTYPES = (torch.bfloat16,)
# Chunk size baked into the kernel: every per-iteration buffer_load /
# buffer_store along the T axis (prologue + main-loop prefetch + v_new
# store) walks a full BT-sized tile *without* a per-row mask. If T is
# not a multiple of BT, the tail iteration reads past the end of
# k/v/w/v_new and — most dangerously — writes past the end of v_new,
# corrupting whatever tensor was allocated next.
GLUON_BT = 64


def _is_gluon_beneficial(
    H: int,
    T: int,
    K: int = 128,
    *,
    k_dtype: Optional[torch.dtype] = None,
    v_dim: Optional[int] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> bool:
    """Check if Gluon V2 fwd_h is both *correct* and faster than Triton.

    .. note::

       Gluon auto-dispatch is **temporarily disabled** because there is no
       gfx950/MI355X CI runner to cover this path. All calls fall back to
       the generic Triton kernel until automated test coverage is added.
       To re-enable, remove the early ``return False`` below.

    The Gluon kernel makes several hardcoded assumptions that the dispatcher
    must enforce, otherwise correctness is silently broken:
      * K ∈ {64, 128} — the kernel allocates exactly one or two 64-wide
        tiles and uses ``if K > 64`` to add the second tile; any K not in
        this set will read/write past the actual K dim.
      * V % 16 == 0 — h/v offsets step by ``i_v * BV`` (BV=16) without
        per-element masks.
      * inputs are bf16 — smem is allocated as gl.bfloat16 and every
        ``.to()`` cast inside the kernel targets bfloat16.
      * every per-sequence T is a multiple of BT (=64) — the prologue and
        the main-loop prefetch issue full-tile buffer_load/buffer_store on
        k/v/w/v_new with no row mask. A non-aligned tail therefore reads
        garbage into b_v_pre (which feeds the last mfma into b_h, i.e.
        the *output*) and writes the bf16 v_new tile past the real T,
        clobbering neighboring allocations.

    Performance side: K<=128, H<=32, gfx950. Both bf16 and fp32 state win
    at all T (fp32: +6-14% across T=4K-64K on MI355X, measured 2026-04-30).
    """
    # DISABLED: no gfx950 CI coverage yet — fall back to Triton unconditionally.
    # Remove this guard once MI355X test infrastructure is available.
    return False

    if not GLUON_AVAILABLE:
        return False
    try:
        target = triton.runtime.driver.active.get_current_target()
        if target.backend != "hip" or "gfx950" not in target.arch:
            return False
    except AttributeError:
        return False
    # Performance gate
    if H > GLUON_MAX_H:
        return False
    # Correctness gates (kernel hardcoded assumptions)
    if K not in GLUON_K_ALLOWED:
        return False
    if v_dim is not None and v_dim % GLUON_BV != 0:
        return False
    if k_dtype is not None and k_dtype not in GLUON_SUPPORTED_DTYPES:
        return False
    # T-alignment gate: avoid the tail-iteration OOB read/write described
    # above. For varlen, every individual sequence length must be aligned
    # because the kernel re-derives T = eos - bos per program.
    if cu_seqlens is None:
        if T % GLUON_BT != 0:
            return False
    else:
        # One device->host sync per dispatcher call. This runs once per
        # prefill (not per token), so the cost is dwarfed by the kernel
        # itself, and it is strictly cheaper than the OOB it prevents.
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        if bool((seqlens % GLUON_BT != 0).any().item()):
            return False
    return True


if GLUON_AVAILABLE:

    @triton.heuristics(
        {
            "USE_G": lambda args: args["g"] is not None,
            "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
            "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
            "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
            "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        }
    )
    @gluon.jit
    def gluon_chunk_fwd_kernel_h(
        k,
        v,
        w,
        v_new,
        g,
        h,
        h0,
        ht,
        cu_seqlens,
        chunk_offsets,
        T,
        H: gl.constexpr,
        Hg: gl.constexpr,
        K: gl.constexpr,
        V: gl.constexpr,
        BT: gl.constexpr,
        BV: gl.constexpr,
        USE_G: gl.constexpr,
        USE_INITIAL_STATE: gl.constexpr,
        STORE_FINAL_STATE: gl.constexpr,
        SAVE_NEW_VALUE: gl.constexpr,
        H_FP32: gl.constexpr,
        IS_VARLEN: gl.constexpr,
        NUM_WARPS: gl.constexpr,
    ):
        i_v = gl.program_id(0)
        i_nh = gl.program_id(1)
        i_n = i_nh // H
        i_h = i_nh % H

        if IS_VARLEN:
            bos = gl.load(cu_seqlens + i_n).to(gl.int32)
            eos = gl.load(cu_seqlens + i_n + 1).to(gl.int32)
            T = eos - bos
            NT = gl.cdiv(T, BT)
            boh = gl.load(chunk_offsets + i_n).to(gl.int32)
        else:
            bos = i_n * T
            NT = gl.cdiv(T, BT)
            boh = i_n * NT

        mma: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16],
            transposed=True,
            warps_per_cta=[NUM_WARPS, 1],
        )
        dot_op0: gl.constexpr = gl.DotOperandLayout(
            operand_index=0, parent=mma, k_width=8
        )
        dot_op1: gl.constexpr = gl.DotOperandLayout(
            operand_index=1, parent=mma, k_width=8
        )

        shared_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[1, 0])
        shared_layout_t: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[0, 1])

        blocked_wk: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[8, 8],
            warps_per_cta=[NUM_WARPS, 1],
            order=[1, 0],
        )
        blocked_kt: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[8, 1],
            threads_per_warp=[8, 8],
            warps_per_cta=[1, NUM_WARPS],
            order=[0, 1],
        )

        h_base = h + (boh * H + i_h) * K * V
        v_base = v + (bos * H + i_h) * V
        k_base = k + (bos * Hg + i_h // (H // Hg)) * K
        w_base = w + (bos * H + i_h) * K

        stride_v = H * V
        stride_h = H * K * V
        stride_k = Hg * K
        stride_w = H * K

        b_h1 = gl.zeros((64, BV), dtype=gl.float32, layout=mma)
        if K > 64:
            b_h2 = gl.zeros((64, BV), dtype=gl.float32, layout=mma)

        if USE_INITIAL_STATE:
            h0_base = h0 + i_nh * K * V
            h0_row = gl.arange(0, 64, layout=gl.SliceLayout(1, mma))
            h0_col = gl.arange(0, BV, layout=gl.SliceLayout(0, mma))
            h0_offs = gl.cast(
                h0_row[:, None] * V + (i_v * BV + h0_col[None, :]), gl.int32
            )
            b_h1 = b_h1 + gl.amd.cdna4.buffer_load(ptr=h0_base, offsets=h0_offs).to(
                gl.float32
            )
            if K > 64:
                h0_offs2 = (
                    (64 + h0_row[:, None]) * V + (i_v * BV + h0_col[None, :])
                ).to(gl.int32)
                b_h2 = b_h2 + gl.amd.cdna4.buffer_load(
                    ptr=h0_base, offsets=h0_offs2
                ).to(gl.float32)

        # 4 independent smem buffers for full double-buffer pipeline
        smem_w1 = gl.allocate_shared_memory(gl.bfloat16, [64, 64], shared_layout)
        smem_w2 = gl.allocate_shared_memory(gl.bfloat16, [64, 64], shared_layout)
        smem_k1 = gl.allocate_shared_memory(gl.bfloat16, [64, 64], shared_layout_t)
        smem_k2 = gl.allocate_shared_memory(gl.bfloat16, [64, 64], shared_layout_t)

        m_row = gl.arange(0, 64, layout=gl.SliceLayout(1, mma))
        m_col_bv = gl.arange(0, BV, layout=gl.SliceLayout(0, mma))
        m_row_bt = gl.arange(0, BT, layout=gl.SliceLayout(1, mma))
        blocked_v: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 4],
            threads_per_warp=[16, 4],
            warps_per_cta=[NUM_WARPS, 1],
            order=[1, 0],
        )
        v_row_b2 = gl.arange(0, BT, layout=gl.SliceLayout(1, blocked_v))
        v_col_b2 = gl.arange(0, BV, layout=gl.SliceLayout(0, blocked_v))

        w_row = gl.arange(0, BT, layout=gl.SliceLayout(1, blocked_wk))
        w_col = gl.arange(0, 64, layout=gl.SliceLayout(0, blocked_wk))
        kt_row = gl.arange(0, 64, layout=gl.SliceLayout(1, blocked_kt))
        kt_col = gl.arange(0, BT, layout=gl.SliceLayout(0, blocked_kt))

        h_offs_base1 = gl.cast(
            m_row[:, None] * V + (i_v * BV + m_col_bv[None, :]), gl.int32
        )
        if K > 64:
            h_offs_base2 = gl.cast(
                64 * V + m_row[:, None] * V + (i_v * BV + m_col_bv[None, :]), gl.int32
            )
        w_offs_base1 = gl.cast(w_row[:, None] * stride_w + w_col[None, :], gl.int32)
        if K > 64:
            w_offs_base2 = gl.cast(
                w_row[:, None] * stride_w + (64 + w_col[None, :]), gl.int32
            )
        v_offs_base = gl.cast(
            v_row_b2[:, None] * stride_v + (i_v * BV + v_col_b2[None, :]), gl.int32
        )
        kt_offs_base1 = gl.cast(kt_row[:, None] + kt_col[None, :] * stride_k, gl.int32)
        if K > 64:
            kt_offs_base2 = gl.cast(
                64 + kt_row[:, None] + kt_col[None, :] * stride_k, gl.int32
            )

        if USE_G:
            g_base = g + bos * H + i_h
            g_offs_base = (m_row_bt * H).to(gl.int32)
        if SAVE_NEW_VALUE:
            vn_base = v_new + (bos * H + i_h) * V

        # Prologue: load iter 0 data → smem
        b_w1_buf = gl.amd.cdna4.buffer_load(ptr=w_base, offsets=w_offs_base1)
        if K > 64:
            b_w2_buf = gl.amd.cdna4.buffer_load(ptr=w_base, offsets=w_offs_base2)
        b_v_pre = gl.amd.cdna4.buffer_load(ptr=v_base, offsets=v_offs_base)
        b_kt1_buf = gl.amd.cdna4.buffer_load(ptr=k_base, offsets=kt_offs_base1)
        if K > 64:
            b_kt2_buf = gl.amd.cdna4.buffer_load(ptr=k_base, offsets=kt_offs_base2)

        smem_w1.store(b_w1_buf)
        if K > 64:
            smem_w2.store(b_w2_buf)
        smem_k1.store(b_kt1_buf)
        if K > 64:
            smem_k2.store(b_kt2_buf)

        # Main loop: N-1 iterations with full pipeline
        for i_t in range(NT - 1):
            i_t_bt = (i_t * BT).to(gl.int32)
            next_bt = i_t_bt + BT

            h_off_iter = (i_t * stride_h).to(gl.int32)
            h_val1 = b_h1 if H_FP32 else b_h1.to(gl.bfloat16)
            gl.amd.cdna4.buffer_store(
                stored_value=h_val1, ptr=h_base, offsets=h_offs_base1 + h_off_iter
            )
            if K > 64:
                h_val2 = b_h2 if H_FP32 else b_h2.to(gl.bfloat16)
                gl.amd.cdna4.buffer_store(
                    stored_value=h_val2, ptr=h_base, offsets=h_offs_base2 + h_off_iter
                )

            b_w1_buf = gl.amd.cdna4.buffer_load(
                ptr=w_base, offsets=w_offs_base1 + next_bt * stride_w
            )
            if K > 64:
                b_w2_buf = gl.amd.cdna4.buffer_load(
                    ptr=w_base, offsets=w_offs_base2 + next_bt * stride_w
                )

            w_dot = smem_w1.load(dot_op0)
            h_dot = gl.convert_layout(b_h1.to(gl.bfloat16), dot_op1)
            b_wh = gl.zeros((BT, BV), dtype=gl.float32, layout=mma)
            b_wh = gl.amd.cdna4.mfma(w_dot, h_dot, b_wh)
            if K > 64:
                w_dot2 = smem_w2.load(dot_op0)
                h_dot2 = gl.convert_layout(b_h2.to(gl.bfloat16), dot_op1)
                b_wh = gl.amd.cdna4.mfma(w_dot2, h_dot2, b_wh)

            b_v_pre_next = gl.amd.cdna4.buffer_load(
                ptr=v_base, offsets=v_offs_base + next_bt * stride_v
            )
            b_v = gl.convert_layout(b_v_pre, mma).to(gl.float32) - b_wh

            if SAVE_NEW_VALUE:
                vn_offs = (
                    (i_t * BT + m_row_bt[:, None]) * stride_v
                    + (i_v * BV + m_col_bv[None, :])
                ).to(gl.int32)
                gl.amd.cdna4.buffer_store(
                    stored_value=b_v.to(gl.bfloat16), ptr=vn_base, offsets=vn_offs
                )

            if USE_G:
                last_idx = min((i_t + 1) * BT, T) - 1
                m_t = (i_t * BT + m_row_bt) < T
                b_g_last = gl.load(g_base + last_idx * H).to(gl.float32)
                b_g = gl.amd.cdna4.buffer_load(
                    ptr=g_base, offsets=g_offs_base + i_t_bt * H, mask=m_t
                ).to(gl.float32)
                b_scale = gl.where(m_t, gl.exp2(b_g_last - b_g), 0.0)
                b_g_last_val = gl.exp2(b_g_last)
                b_v = b_v * b_scale[:, None]
                b_h1 = b_h1 * b_g_last_val
                if K > 64:
                    b_h2 = b_h2 * b_g_last_val

            b_kt1_buf = gl.amd.cdna4.buffer_load(
                ptr=k_base, offsets=kt_offs_base1 + next_bt * stride_k
            )
            if K > 64:
                b_kt2_buf = gl.amd.cdna4.buffer_load(
                    ptr=k_base, offsets=kt_offs_base2 + next_bt * stride_k
                )

            v_dot = gl.convert_layout(b_v.to(gl.bfloat16), dot_op1)
            k_dot = smem_k1.load(dot_op0)
            b_h1 = gl.amd.cdna4.mfma(k_dot, v_dot, b_h1)
            if K > 64:
                k_dot2 = smem_k2.load(dot_op0)
                b_h2 = gl.amd.cdna4.mfma(k_dot2, v_dot, b_h2)

            smem_w1.store(b_w1_buf)
            if K > 64:
                smem_w2.store(b_w2_buf)
            smem_k1.store(b_kt1_buf)
            if K > 64:
                smem_k2.store(b_kt2_buf)
            b_v_pre = b_v_pre_next

        # Epilogue: last iteration (no prefetch)
        if NT > 0:
            i_t_last = NT - 1
            i_t_bt = (i_t_last * BT).to(gl.int32)

            h_off_iter = (i_t_last * stride_h).to(gl.int32)
            h_val1 = b_h1 if H_FP32 else b_h1.to(gl.bfloat16)
            gl.amd.cdna4.buffer_store(
                stored_value=h_val1, ptr=h_base, offsets=h_offs_base1 + h_off_iter
            )
            if K > 64:
                h_val2 = b_h2 if H_FP32 else b_h2.to(gl.bfloat16)
                gl.amd.cdna4.buffer_store(
                    stored_value=h_val2, ptr=h_base, offsets=h_offs_base2 + h_off_iter
                )

            w_dot = smem_w1.load(dot_op0)
            h_dot = gl.convert_layout(b_h1.to(gl.bfloat16), dot_op1)
            b_wh = gl.zeros((BT, BV), dtype=gl.float32, layout=mma)
            b_wh = gl.amd.cdna4.mfma(w_dot, h_dot, b_wh)
            if K > 64:
                w_dot2 = smem_w2.load(dot_op0)
                h_dot2 = gl.convert_layout(b_h2.to(gl.bfloat16), dot_op1)
                b_wh = gl.amd.cdna4.mfma(w_dot2, h_dot2, b_wh)

            b_v = gl.convert_layout(b_v_pre, mma).to(gl.float32) - b_wh

            if SAVE_NEW_VALUE:
                vn_offs = (
                    (i_t_last * BT + m_row_bt[:, None]) * stride_v
                    + (i_v * BV + m_col_bv[None, :])
                ).to(gl.int32)
                gl.amd.cdna4.buffer_store(
                    stored_value=b_v.to(gl.bfloat16), ptr=vn_base, offsets=vn_offs
                )

            if USE_G:
                last_idx = min((i_t_last + 1) * BT, T) - 1
                m_t = (i_t_last * BT + m_row_bt) < T
                b_g_last = gl.load(g_base + last_idx * H).to(gl.float32)
                b_g = gl.amd.cdna4.buffer_load(
                    ptr=g_base, offsets=g_offs_base + i_t_bt * H, mask=m_t
                ).to(gl.float32)
                b_scale = gl.where(m_t, gl.exp2(b_g_last - b_g), 0.0)
                b_g_last_val = gl.exp2(b_g_last)
                b_v = b_v * b_scale[:, None]
                b_h1 = b_h1 * b_g_last_val
                if K > 64:
                    b_h2 = b_h2 * b_g_last_val

            v_dot = gl.convert_layout(b_v.to(gl.bfloat16), dot_op1)
            k_dot = smem_k1.load(dot_op0)
            b_h1 = gl.amd.cdna4.mfma(k_dot, v_dot, b_h1)
            if K > 64:
                k_dot2 = smem_k2.load(dot_op0)
                b_h2 = gl.amd.cdna4.mfma(k_dot2, v_dot, b_h2)

        if STORE_FINAL_STATE:
            ht_base = ht + i_nh * K * V
            ht_offs1 = (m_row[:, None] * V + (i_v * BV + m_col_bv[None, :])).to(
                gl.int32
            )
            gl.amd.cdna4.buffer_store(stored_value=b_h1, ptr=ht_base, offsets=ht_offs1)
            if K > 64:
                ht_offs2 = (
                    (64 + m_row[:, None]) * V + (i_v * BV + m_col_bv[None, :])
                ).to(gl.int32)
                gl.amd.cdna4.buffer_store(
                    stored_value=b_h2, ptr=ht_base, offsets=ht_offs2
                )


def chunk_gated_delta_rule_fwd_h_gluon(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    state_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Gluon fwd_h — drop-in replacement for chunk_gated_delta_rule_fwd_h.

    Optimized for AMD MI355X (gfx950/CDNA4), TP2 (H≤32), T≤65536.

    Args:
        state_dtype: dtype of the per-chunk hidden state buffer ``h``.
            If explicitly provided, it is honored as-is. If ``None``,
            defaults to fp32 to preserve precision and stay consistent
            with the generic Triton fallback path.
    """
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    assert K <= 128, f"Gluon fwd_h only supports K<=128 (two 64-wide tiles), got K={K}"
    BT = chunk_size
    BV = 16
    NUM_WARPS = 4

    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(cu_seqlens) - 1
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
        NT = int(chunk_offsets[-1])  # exact total chunk count across all sequences

    if state_dtype is None:
        state_dtype = torch.float32
    # NOTE: still K-first internally; gated off in chunk_delta_h.py while the
    # Triton fallback runs V-first. See _GLUON_PATH_VFIRST_DONE there. When
    # this kernel is ported to V-first the alloc shape below also flips to
    # (B, NT, H, V, K).
    h = k.new_empty(B, NT, H, K, V, dtype=state_dtype)
    final_state = (
        k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    )
    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    gluon_chunk_fwd_kernel_h[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BV=BV,
        H_FP32=(state_dtype == torch.float32),
        NUM_WARPS=NUM_WARPS,
        num_warps=NUM_WARPS,
        num_stages=1,
    )
    return h, v_new, final_state

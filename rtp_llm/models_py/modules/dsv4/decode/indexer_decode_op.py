"""DeepSeek-V4 decode-only Indexer op.

Mirrors the BF16 reference at ``rtp_llm/models_py/modules/dsv4/indexer.py``
(prefill+decode forward, lines 81-152) but specializes the decode path
(``q_len`` is 1 in pure decode) and exposes a faster FP8 fast path
following the V3.2 DSA pattern in ``base/cuda/indexer_op.py``
(``_get_topk_paged``).

Two implementations are gated by the env flag ``RTP_LLM_DSV4_INDEXER_DECODE_FAST_PATH``
(default ``1``):

  * **Reference path** — pure PyTorch einsum + ReLU + weighted sum + topk.
    Bit-equivalent to the BF16 reference in ``indexer.py`` for valid range.
  * **Fast path** — FP8-quantizes Q + K via ``sgl_per_token_group_quant_fp8``
    (group=128) and calls ``deep_gemm.fp8_paged_mqa_logits`` with a
    one-block-per-request paged layout, then ``score.topk(...)`` to fill
    ``out_buffer``. Note: V4 KV is BF16 in cache; we FP8-quantize K at the
    op boundary so the kernel can consume it (V4-official inference also
    FP8-quantizes Q before scoring).

Decision (2026-04-26): the fast-path FP8-paged-MQA wiring needs careful
construction of the (kv_data || per-128 fp32 scale) packed layout that
``fp8_paged_mqa_logits`` expects from its kv_cache argument. Getting the
exact byte-level packing (head_dim_with_sf = 132 with scales placed AFTER
the FP8 data within each block, per ``base/cuda/indexer_op.py:380-403``)
is non-trivial without an existing helper. To unblock the model
integration, we ship the fast path as a scaffold but mark its IoU test as
WIP. The reference-path test must pass exactly. Once the integration site
in ``deepseek_v4_model.py`` is ready to call this op, we'll wire the
proper packing helper (likely a ported ``indexer_k_quant_and_cache`` view
that targets a per-batch ephemeral cache rather than the persistent KV
buffer, since V4 keeps KV in BF16).
"""

from __future__ import annotations

import os

import torch

from rtp_llm.models_py.modules.dsv4.fp8._indexer_score import (
    deepgemm_supports_block_kv,
)
from rtp_llm.models_py.modules.dsv4.indexer_topk import (
    TorchIndexerTopKBackend,
    select_indexer_topk,
)

try:
    import deep_gemm
except Exception:  # pragma: no cover - non-cuda env
    deep_gemm = None  # type: ignore[assignment]

try:
    from rtp_llm.ops.compute_ops import rtp_llm_ops as _rtp_llm_ops

    _HAS_TOPK_V3 = hasattr(_rtp_llm_ops, "topk_v3")
except Exception:  # pragma: no cover - CPU/fallback environments
    _rtp_llm_ops = None  # type: ignore[assignment]
    _HAS_TOPK_V3 = False

_TOPK_V3_WORKSPACE_CACHE: dict[int, torch.Tensor] = {}

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - non-triton env
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _indexer_q_quant_fold_kernel(
        q_ptr,
        w_ptr,
        q_fp8_ptr,
        w_out_ptr,
        q_stride_b,
        q_stride_s,
        q_stride_h,
        q_stride_d,
        w_stride_b,
        w_stride_s,
        w_stride_h,
        qf_stride_row,
        softmax_scale,
        Q_LEN: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        UE8M0: tl.constexpr,
    ):
        row_id = tl.program_id(0).to(tl.int64)
        b = row_id // Q_LEN
        s = row_id - b * Q_LEN
        h = tl.program_id(1).to(tl.int64)
        offs_d = tl.arange(0, D)

        q_base = b * q_stride_b + s * q_stride_s + h * q_stride_h
        q_vals = tl.load(q_ptr + q_base + offs_d * q_stride_d).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(q_vals), axis=0), 1e-4)
        scale = amax * (1.0 / 448.0)
        if UE8M0:
            bits = scale.to(tl.uint32, bitcast=True)
            exp_bits = (bits + 0x007FFFFF) & 0x7F800000
            scale = exp_bits.to(tl.float32, bitcast=True)

        q_scaled = tl.clamp(q_vals / scale, -448.0, 448.0).to(tl.float8e4nv)
        out_off = (row_id * H + h) * qf_stride_row
        tl.store(q_fp8_ptr + out_off + offs_d, q_scaled)

        w_val = tl.load(w_ptr + b * w_stride_b + s * w_stride_s + h * w_stride_h).to(
            tl.float32
        )
        tl.store(w_out_ptr + row_id * H + h, w_val * scale * softmax_scale)

    @triton.jit
    def _indexer_k_quant_pack_kernel(
        k_ptr,
        kv_cache_u8_ptr,
        k_stride_b,
        k_stride_t,
        k_stride_d,
        T_MAX,
        T_PADDED,
        OUT_ROW_STRIDE,
        BLOCK_KV: tl.constexpr,
        D: tl.constexpr,
        UE8M0: tl.constexpr,
    ):
        b = tl.program_id(0).to(tl.int64)
        t = tl.program_id(1).to(tl.int64)
        offs_d = tl.arange(0, D)
        offs_s = tl.arange(0, 4)
        blk = t // BLOCK_KV
        blk_off = t - blk * BLOCK_KV
        num_blocks_per_req = T_PADDED // BLOCK_KV
        block_base = kv_cache_u8_ptr + (b * num_blocks_per_req + blk) * (
            BLOCK_KV * OUT_ROW_STRIDE
        )
        k_dst = block_base + blk_off * D + offs_d
        scale_dst = block_base + BLOCK_KV * D + blk_off * 4 + offs_s

        if t < T_MAX:
            k_base = b * k_stride_b + t * k_stride_t
            k_vals = tl.load(k_ptr + k_base + offs_d * k_stride_d).to(tl.float32)
            amax = tl.maximum(tl.max(tl.abs(k_vals), axis=0), 1e-4)
            scale = amax * (1.0 / 448.0)
            if UE8M0:
                bits = scale.to(tl.uint32, bitcast=True)
                exp_bits = (bits + 0x007FFFFF) & 0x7F800000
                scale = exp_bits.to(tl.float32, bitcast=True)

            k_fp8 = tl.clamp(k_vals / scale, -448.0, 448.0).to(tl.float8e4nv)
            tl.store(k_dst, k_fp8.to(tl.uint8, bitcast=True))

            scale_u32 = scale.to(tl.uint32, bitcast=True)
            scale_bytes = ((scale_u32 >> (8 * offs_s)) & 0xFF).to(tl.uint8)
            tl.store(scale_dst, scale_bytes)
        else:
            tl.store(k_dst, tl.zeros((D,), dtype=tl.uint8))
            tl.store(scale_dst, tl.zeros((4,), dtype=tl.uint8))


def _fast_path_available(device=None, block_kv=None) -> bool:
    """Whether the DeepGEMM paged fast path can serve this device/block.

    Shares the block-size contract with ``fp8_paged_indexer_score`` (SM90/SM100
    accept ``block_kv == 64``; SM100 additionally accepts 32), so the decode op
    and the indexer score wrapper cannot drift apart.
    """
    if not (
        torch.cuda.is_available()
        and deep_gemm is not None
        and hasattr(deep_gemm, "fp8_paged_mqa_logits")
        and hasattr(deep_gemm, "get_paged_mqa_logits_metadata")
    ):
        return False
    return block_kv is None or deepgemm_supports_block_kv(device, block_kv)


_USE_DEEP_GEMM_FAST_PATH = (
    os.environ.get("RTP_LLM_DSV4_INDEXER_DECODE_FAST_PATH", "1") != "0"
)


def _fast_path_unavailable_reason(q_indexer: torch.Tensor, block_kv=None) -> str:
    if not q_indexer.is_cuda:
        return f"q_indexer is on {q_indexer.device}, expected CUDA"
    if not torch.cuda.is_available():
        return "torch.cuda.is_available() is False"
    if deep_gemm is None:
        return "deep_gemm import failed"
    missing = [
        name
        for name in ("fp8_paged_mqa_logits", "get_paged_mqa_logits_metadata")
        if not hasattr(deep_gemm, name)
    ]
    if missing:
        return f"deep_gemm missing required symbols: {missing}"
    if block_kv is not None and not deepgemm_supports_block_kv(
        q_indexer.device, block_kv
    ):
        return (
            f"deep_gemm paged MQA logits does not accept block_kv={block_kv} on "
            f"capability {torch.cuda.get_device_capability(q_indexer.device)}"
        )
    return "unknown"


class IndexerDecodeV4Op:
    """Decode-time Indexer top-K op for DeepSeek-V4 CSA layers.

    Computes ``topk_idxs[b, s] = argtopk_t( sum_h ReLU(q[b,s,h] . kv[b,t]) * weights[b,s,h] )``
    over the valid compressed-K range per request, writing results
    in-place into ``out_buffer``.

    The op is *only* used by ``compress_ratio == 4`` (CSA) layers; HCA /
    SWA-only layers don't carry an indexer.
    """

    def __init__(
        self,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        softmax_scale: float,
        # FP8 quant config (matches V3.2 DSA defaults).
        block_size: int = 128,
        scale_fmt: str = "ue8m0",
        page_block_size: int = 64,
    ):
        self.index_n_heads = int(index_n_heads)
        self.index_head_dim = int(index_head_dim)
        self.index_topk = int(index_topk)
        self.softmax_scale = float(softmax_scale)
        self.block_size = int(block_size)
        self.scale_fmt = str(scale_fmt)
        self.page_block_size = int(page_block_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(
        self,
        q_indexer: torch.Tensor,  # [B, q_len, H_idx, D_idx] bf16
        kv_indexer: torch.Tensor,  # [B, T_max,         D_idx] bf16
        weights: torch.Tensor,  # [B, q_len, H_idx]         bf16
        compressed_len_per_req: torch.Tensor,  # [B] int32
        out_buffer: torch.Tensor,  # [B, q_len, K] int32
        force_reference: bool = False,
    ) -> torch.Tensor:
        """Fill ``out_buffer`` with topk indices and return it.

        Args:
            q_indexer: BF16 query, post-RoPE.
            kv_indexer: BF16 KV view; only ``[:, :T_max]`` is read but
                only entries with ``t < compressed_len_per_req[b]`` count;
                indices >= that bound are masked to -inf before topk.
            weights: BF16 per-(req, q-token, head) weights, already
                pre-multiplied by ``softmax_scale * H_idx ** -0.5`` by
                the caller (matches ``indexer.py:104``).
            compressed_len_per_req: int32 valid lengths per request.
            out_buffer: pre-allocated int32 buffer to write topk into.
            force_reference: if True, always take the PyTorch path (used
                by tests that need bit-equivalence with the einsum oracle).

        Returns:
            ``out_buffer`` (same tensor; in-place fill).
        """
        assert q_indexer.dim() == 4, f"expected [B,q_len,H,D], got {q_indexer.shape}"
        B, q_len, H, D = q_indexer.shape
        assert kv_indexer.dim() == 3, f"expected [B,T,D], got {kv_indexer.shape}"
        Bk, T_max, Dk = kv_indexer.shape
        assert Bk == B and Dk == D, "q/kv batch & D must match"
        assert weights.shape == (
            B,
            q_len,
            H,
        ), f"weights shape mismatch: {weights.shape}"
        assert compressed_len_per_req.shape == (
            B,
        ), f"len shape mismatch: {compressed_len_per_req.shape}"
        assert out_buffer.shape == (
            B,
            q_len,
            self.index_topk,
        ), f"out_buffer shape mismatch: {out_buffer.shape}"
        assert out_buffer.dtype == torch.int32, "out_buffer must be int32"

        if (not force_reference) and _USE_DEEP_GEMM_FAST_PATH:
            if (
                not _fast_path_available(q_indexer.device, self.page_block_size)
                or not q_indexer.is_cuda
            ):
                raise RuntimeError(
                    "DSV4 indexer decode fast path is required by default but "
                    "is unavailable: "
                    f"{_fast_path_unavailable_reason(q_indexer, self.page_block_size)}. "
                    "Set RTP_LLM_DSV4_INDEXER_DECODE_FAST_PATH=0 or pass "
                    "force_reference=True for explicit reference execution."
                )
            return self._forward_fast(
                q_indexer,
                kv_indexer,
                weights,
                compressed_len_per_req,
                out_buffer,
            )
        return self._forward_reference(
            q_indexer,
            kv_indexer,
            weights,
            compressed_len_per_req,
            out_buffer,
        )

    __call__ = forward

    # ------------------------------------------------------------------
    # Reference path — pure PyTorch
    # ------------------------------------------------------------------
    def _forward_reference(
        self,
        q_indexer: torch.Tensor,
        kv_indexer: torch.Tensor,
        weights: torch.Tensor,
        compressed_len_per_req: torch.Tensor,
        out_buffer: torch.Tensor,
    ) -> torch.Tensor:
        B, q_len, H, D = q_indexer.shape
        T_max = kv_indexer.shape[1]
        device = q_indexer.device

        # Fused einsum + ReLU + weighted-sum via Triton kernel — never
        # materializes [B, q_len, H, T] fp32. No causal mask in decode
        # (the per-request compressed_len mask is applied below).
        from rtp_llm.models_py.modules.dsv4._indexer_score_triton import (
            v4_indexer_score,
        )

        score = v4_indexer_score(
            q_indexer.contiguous(),
            kv_indexer.contiguous(),
            weights,
            q_pos=None,
            compress_ratio=1,
        )

        # topk; if K > T_max we still pick top-T_max valid then -1-pad.
        k_eff = min(self.index_topk, T_max)
        ref_lengths = (
            compressed_len_per_req.view(B, 1).expand(B, q_len).contiguous().reshape(-1)
            if q_len > 1
            else compressed_len_per_req.reshape(-1).contiguous()
        )
        idxs = TorchIndexerTopKBackend().select(
            score,
            self.index_topk,
            lengths=ref_lengths,
        )
        if k_eff < self.index_topk and idxs.shape[-1] != self.index_topk:
            n_valid = idxs.shape[-1]
            out_buffer[..., :n_valid].copy_(idxs)
            out_buffer[..., n_valid:].fill_(-1)
        else:
            out_buffer.copy_(idxs)
        return out_buffer

    # ------------------------------------------------------------------
    # Fast path — FP8-quantize Q+K and call deep_gemm.fp8_paged_mqa_logits.
    # ------------------------------------------------------------------
    def _forward_fast(
        self,
        q_indexer: torch.Tensor,
        kv_indexer: torch.Tensor,
        weights: torch.Tensor,
        compressed_len_per_req: torch.Tensor,
        out_buffer: torch.Tensor,
    ) -> torch.Tensor:
        B, q_len, H, D = q_indexer.shape
        T_max = kv_indexer.shape[1]
        device = q_indexer.device
        block_kv = self.page_block_size  # paged blocksize
        # Pad T_max up to a multiple of block_kv (kernel expects block-aligned KV pool).
        num_blocks_per_req = (T_max + block_kv - 1) // block_kv
        T_padded = num_blocks_per_req * block_kv
        num_blocks = B * num_blocks_per_req
        scales_per_token = D // self.block_size  # 1 for D=128, group=128
        head_dim_with_sf = D + scales_per_token * 4  # bytes per token
        is_ue8m0 = self.scale_fmt == "ue8m0"

        if (
            _HAS_TRITON
            and D == 128
            and self.block_size == 128
            and q_indexer.is_cuda
            and kv_indexer.is_cuda
        ):
            q_fp8_in = torch.empty(
                (B, q_len, H, D), dtype=torch.float8_e4m3fn, device=device
            )
            scaled_weights = torch.empty(
                (B * q_len, H), dtype=torch.float32, device=device
            )
            if B * q_len > 0 and H > 0:
                _indexer_q_quant_fold_kernel[(B * q_len, H)](
                    q_indexer,
                    weights,
                    q_fp8_in,
                    scaled_weights,
                    q_indexer.stride(0),
                    q_indexer.stride(1),
                    q_indexer.stride(2),
                    q_indexer.stride(3),
                    weights.stride(0),
                    weights.stride(1),
                    weights.stride(2),
                    D,
                    float(self.softmax_scale),
                    Q_LEN=q_len,
                    H=H,
                    D=D,
                    UE8M0=is_ue8m0,
                    num_warps=4,
                )

            kv_cache_u8 = torch.empty(
                (num_blocks, block_kv, 1, head_dim_with_sf),
                dtype=torch.uint8,
                device=device,
            )
            if B > 0 and T_padded > 0:
                _indexer_k_quant_pack_kernel[(B, T_padded)](
                    kv_indexer,
                    kv_cache_u8,
                    kv_indexer.stride(0),
                    kv_indexer.stride(1),
                    kv_indexer.stride(2),
                    T_max,
                    T_padded,
                    head_dim_with_sf,
                    BLOCK_KV=block_kv,
                    D=D,
                    UE8M0=is_ue8m0,
                    num_warps=4,
                )
        else:
            # Fallback for non-128 head/group configurations.
            from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
                sgl_per_token_group_quant_fp8,
            )

            q_flat = q_indexer.contiguous().view(-1, D)  # [B*q_len*H, D]
            q_fp8, q_scale = sgl_per_token_group_quant_fp8(
                q_flat,
                group_size=self.block_size,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=is_ue8m0,
            )
            q_fp8_in = q_fp8.view(B, q_len, H, D)
            if is_ue8m0:
                from rtp_llm.models_py.modules.base.cuda.indexer_op import (
                    _unpack_ue8m0_scale,
                )

                q_scale = _unpack_ue8m0_scale(q_scale)
            q_scale = q_scale.view(B * q_len, H, 1)
            scaled_weights = (
                weights.contiguous().view(B * q_len, H, 1).float()
                * q_scale.float()
                * self.softmax_scale
            ).view(B * q_len, H)

            if T_padded != T_max:
                kv_padded = torch.zeros(
                    (B, T_padded, D),
                    dtype=kv_indexer.dtype,
                    device=device,
                )
                kv_padded[:, :T_max].copy_(kv_indexer)
            else:
                kv_padded = kv_indexer.contiguous()

            k_flat = kv_padded.view(-1, D)  # [B*T_padded, D]
            k_fp8, k_scale = sgl_per_token_group_quant_fp8(
                k_flat,
                group_size=self.block_size,
                eps=1e-4,
                column_major_scales=is_ue8m0,
                scale_tma_aligned=is_ue8m0,
                scale_ue8m0=is_ue8m0,
            )
            if is_ue8m0:
                from rtp_llm.models_py.modules.base.cuda.indexer_op import (
                    _unpack_ue8m0_scale,
                )

                k_scale = _unpack_ue8m0_scale(k_scale)

            kv_cache_u8 = torch.empty(
                (num_blocks, block_kv, 1, head_dim_with_sf),
                dtype=torch.uint8,
                device=device,
            )
            kv_cache_blocks = kv_cache_u8.view(num_blocks, block_kv * head_dim_with_sf)
            kv_cache_blocks[:, : block_kv * D].copy_(
                k_fp8.view(num_blocks, block_kv * D).view(torch.uint8)
            )
            k_scale_3d = k_scale.view(num_blocks, block_kv, scales_per_token)
            if not k_scale_3d.is_contiguous() or k_scale_3d.stride(-1) != 1:
                k_scale_3d = k_scale_3d.clone(memory_format=torch.contiguous_format)
            kv_cache_blocks[:, block_kv * D :].copy_(
                k_scale_3d.view(torch.uint8).view(
                    num_blocks, block_kv * scales_per_token * 4
                ),
            )

        # ---- block_table: one block-row per request, listing its blocks --
        block_table = torch.arange(
            num_blocks,
            dtype=torch.int32,
            device=device,
        ).view(B, num_blocks_per_req)

        # ---- context_lens ([B, q_len] int32 expected by DeepGEMM) ------
        # Indexer's "context_lens" are the valid compressed K count per req.
        kv_lens_1d = (
            compressed_len_per_req
            if (
                compressed_len_per_req.dtype == torch.int32
                and compressed_len_per_req.is_contiguous()
            )
            else compressed_len_per_req.to(torch.int32).contiguous()
        )
        context_lens = (
            kv_lens_1d.view(B, 1)
            if q_len == 1
            else kv_lens_1d.view(B, 1).expand(B, q_len).contiguous()
        )

        max_seq_len = T_padded
        schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
            context_lens,
            block_kv,
            deep_gemm.get_num_sms(),
        )

        logits = deep_gemm.fp8_paged_mqa_logits(
            q_fp8_in,
            kv_cache_u8,
            scaled_weights,
            context_lens,
            block_table,
            schedule_meta,
            max_seq_len,
            clean_logits=False,
        )  # [B*q_len, max_seq_len] fp32

        # ---- TopK + length mask (in-place into out_buffer when eligible) ----
        if (
            _HAS_TOPK_V3
            and self.index_topk in (512, 1024, 2048)
            and max_seq_len > 0
            and logits.is_cuda
            and out_buffer.is_cuda
            and out_buffer.is_contiguous()
        ):
            dev_idx = (
                int(device.index)
                if device.index is not None
                else torch.cuda.current_device()
            )
            ws = _TOPK_V3_WORKSPACE_CACHE.get(dev_idx)
            if ws is None:
                ws = torch.empty((1 << 20,), dtype=torch.uint8, device=device)
                _TOPK_V3_WORKSPACE_CACHE[dev_idx] = ws
            _rtp_llm_ops.topk_v3(
                logits,
                context_lens.reshape(-1),
                out_buffer.view(B * q_len, self.index_topk),
                ws,
                self.index_topk,
                max_seq_len,
            )
            return out_buffer

        logits_3d = logits.view(B, q_len, max_seq_len)
        idxs = select_indexer_topk(
            logits_3d,
            self.index_topk,
            lengths=context_lens.reshape(-1),
        )

        out_buffer.copy_(idxs)
        return out_buffer

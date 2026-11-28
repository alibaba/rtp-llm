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

from contextlib import nullcontext
import os

import torch

from rtp_llm.models_py.modules.dsv4.fp8._indexer_score import (
    deepgemm_supports_block_kv,
    fp8_paged_mqa_uses_2d_context_lens,
    get_fp8_paged_mqa_logits_metadata,
    prepare_fp8_paged_mqa_inputs,
    supports_paged_mqa_logits,
)
from rtp_llm.models_py.modules.dsv4.indexer_topk import (
    TorchIndexerTopKBackend,
    select_indexer_topk,
)

try:
    import deep_gemm
except Exception:  # pragma: no cover - non-cuda env
    deep_gemm = None  # type: ignore[assignment]


def _fast_path_available(device=None, block_kv=64) -> bool:
    """Whether the DeepGEMM paged fast path can serve this device/block.

    Shares the ABI + block-size contract with ``fp8_paged_indexer_score`` so
    the decode op and the indexer score wrapper cannot drift apart.
    """
    if not (
        torch.cuda.is_available()
        and deep_gemm is not None
        and hasattr(deep_gemm, "fp8_paged_mqa_logits")
        and (
            hasattr(deep_gemm, "get_legacy_paged_mqa_logits_metadata")
            or hasattr(deep_gemm, "get_paged_mqa_logits_metadata")
        )
    ):
        return False
    return supports_paged_mqa_logits(device, block_kv, backend=deep_gemm)


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
    if not hasattr(deep_gemm, "fp8_paged_mqa_logits"):
        return "deep_gemm missing required symbol: fp8_paged_mqa_logits"
    if not (
        hasattr(deep_gemm, "get_legacy_paged_mqa_logits_metadata")
        or hasattr(deep_gemm, "get_paged_mqa_logits_metadata")
    ):
        return "deep_gemm missing required metadata symbol"
    if block_kv is not None and not deepgemm_supports_block_kv(
        q_indexer.device, block_kv, backend=deep_gemm
    ):
        return (
            f"deep_gemm paged MQA logits does not accept block_kv={block_kv} on "
            f"capability {torch.cuda.get_device_capability(q_indexer.device)}"
        )
    return (
        "deep_gemm paged MQA logits is not supported on capability "
        f"{torch.cuda.get_device_capability(q_indexer.device)}"
    )


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
            if not q_indexer.is_cuda or not _fast_path_available(
                q_indexer.device, self.page_block_size
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
        idxs = TorchIndexerTopKBackend().select(
            score,
            self.index_topk,
            lengths=compressed_len_per_req.view(B, 1).expand(B, q_len).reshape(-1),
        )
        if k_eff < self.index_topk and idxs.shape[-1] != self.index_topk:
            pad = torch.full((B, q_len, self.index_topk - k_eff), -1, dtype=torch.int32, device=device)
            idxs = torch.cat([idxs, pad], dim=-1)

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
        # Lazy import — only paid when fast path runs.
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )

        B, q_len, H, D = q_indexer.shape
        if B == 0 or q_len == 0 or out_buffer.numel() == 0:
            return out_buffer
        if kv_indexer.shape[1] == 0:
            out_buffer.fill_(-1)
            return out_buffer
        if D != self.block_size:
            raise ValueError(
                f"DSV4 indexer decode fast path requires head dim {D} == "
                f"block_size {self.block_size}"
            )
        T_max = kv_indexer.shape[1]
        device = q_indexer.device
        block_kv = self.page_block_size  # paged blocksize
        # Pad T_max up to a multiple of block_kv (kernel expects block-aligned KV pool).
        num_blocks_per_req = (T_max + block_kv - 1) // block_kv
        T_padded = num_blocks_per_req * block_kv

        # ---- Quantize Q. Matches V3.2 quant_q_k flow -------------------
        q_flat = q_indexer.contiguous().view(-1, D)  # [B*q_len*H, D]
        q_fp8, q_scale = sgl_per_token_group_quant_fp8(
            q_flat,
            group_size=self.block_size,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=(self.scale_fmt == "ue8m0"),
        )
        q_fp8 = q_fp8.view(B * q_len, H, D)

        # Fold per-token Q scale into ``weights`` (kernel expects scaled weights).
        unpack_ue8m0_scale = None
        if self.scale_fmt == "ue8m0":
            from rtp_llm.models_py.modules.base.cuda.indexer_op import (
                _unpack_ue8m0_scale,
            )

            unpack_ue8m0_scale = _unpack_ue8m0_scale
            q_scale = unpack_ue8m0_scale(q_scale)
        q_scale = q_scale.view(B * q_len, H, 1)
        scaled_weights = (
            (
                weights.contiguous().view(B * q_len, H, 1).float()
                * q_scale.float()
                * self.softmax_scale
            )
            .view(B * q_len, H)
            .contiguous()
        )

        # ---- Quantize K and pack into per-block striped (K || scale) ---
        # Pad KV to block-aligned T_padded.
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
        use_ue8m0 = self.scale_fmt == "ue8m0"
        k_fp8, k_scale = sgl_per_token_group_quant_fp8(
            k_flat,
            group_size=self.block_size,
            eps=1e-4,
            column_major_scales=use_ue8m0,
            scale_tma_aligned=use_ue8m0,
            scale_ue8m0=use_ue8m0,
        )
        # k_scale is [B*T_padded, D//group_size] either packed int32 (ue8m0) or fp32.
        # DeepGEMM expects a per-block striped layout matching _indexer_quant_triton:
        # [0 : block_kv*D) = all K bytes in the block,
        # [block_kv*D : block_kv*(D+4)) = all fp32 K scales in the block.
        scales_per_token = D // self.block_size  # 1 for D=128, group=128
        head_dim_with_sf = D + scales_per_token * 4  # bytes per token

        num_blocks = B * num_blocks_per_req
        kv_cache_u8 = torch.zeros(
            (num_blocks, block_kv, 1, head_dim_with_sf),
            dtype=torch.uint8,
            device=device,
        )
        kv_block_bytes = kv_cache_u8.view(num_blocks, block_kv * head_dim_with_sf)
        k_data_bytes = block_kv * D
        kv_block_bytes[:, :k_data_bytes].copy_(
            k_fp8.view(num_blocks, block_kv, D)
            .contiguous()
            .view(torch.uint8)
            .reshape(num_blocks, k_data_bytes)
        )
        k_scale_f32 = (
            unpack_ue8m0_scale(k_scale)
            if unpack_ue8m0_scale is not None
            else k_scale.to(torch.float32)
        )
        k_scale_contig = k_scale_f32.contiguous().reshape(
            num_blocks, block_kv * scales_per_token
        )
        if k_scale_contig.stride(-1) != 1:
            k_scale_contig = k_scale_contig.clone(memory_format=torch.contiguous_format)
        kv_block_bytes[:, k_data_bytes:].copy_(
            k_scale_contig.view(torch.uint8).reshape(
                num_blocks, block_kv * scales_per_token * 4
            )
        )

        # ---- block_table: one block-row per request, listing its blocks --
        block_table = torch.arange(
            num_blocks,
            dtype=torch.int32,
            device=device,
        ).view(B, num_blocks_per_req)

        # ---- kv_lens: 2D [B, next_n] on >=2.8 wheels, 1D [B] on <=2.1 wheels ----
        kv_lens_req = compressed_len_per_req.to(
            device=device, dtype=torch.int32
        ).contiguous()
        kv_lens_2d = kv_lens_req.view(B, 1).expand(B, q_len).contiguous()
        kv_lens_raw = (
            kv_lens_2d
            if fp8_paged_mqa_uses_2d_context_lens(deep_gemm)
            else kv_lens_req
        )

        max_seq_len = T_padded
        dev_ctx = (
            torch.cuda.device(device)
            if q_indexer.is_cuda
            else nullcontext()
        )
        with dev_ctx:
            q_fp8_in, kv_lens_in, block_table_in = prepare_fp8_paged_mqa_inputs(
                q_fp8.view(B, q_len, H, D), kv_lens_raw, block_table, backend=deep_gemm
            )
            schedule_meta = get_fp8_paged_mqa_logits_metadata(
                deep_gemm, kv_lens_in, block_kv, deep_gemm.get_num_sms()
            )

            logits = deep_gemm.fp8_paged_mqa_logits(
                q_fp8_in,
                kv_cache_u8,
                scaled_weights,
                kv_lens_in,
                block_table_in,
                schedule_meta,
                max_seq_len,
                clean_logits=False,
            )  # [B*q_len, max_seq_len] fp32

            # ---- TopK + length mask ---------------------------------------
            logits = logits.view(B, q_len, max_seq_len)
            idxs = select_indexer_topk(
                logits,
                self.index_topk,
                lengths=kv_lens_2d.reshape(-1),
            )

            out_buffer.copy_(idxs)
            return out_buffer

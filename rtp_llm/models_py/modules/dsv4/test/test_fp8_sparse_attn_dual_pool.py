"""End-to-end GPU test: FlashMLA FP8 sparse decode (single + dual pool).

Validates that ``SparseAttnV4DecodeFp8Op.forward(...)`` produces
attention output close to a BF16 reference (dequant both pools ->
gather into BF16 packed buffer -> Python ``_sparse_attn``) within
the FP8-e4m3 quantization noise envelope.

Three scenarios:
  * Single-pool: only SWA pool, no ``extra_k_cache`` arg.
  * Dual-pool: SWA + CMP pools fed via ``extra_k_cache`` /
    ``extra_indices_in_kvcache``. Compares against BF16 ref of
    SWA-cat-CMP packed.
  * Dual-pool > single-pool attention: dual-pool over the same SWA
    slots PLUS extra CMP slots must be different from single-pool
    over SWA only (basic plumbing check that ``extra_k_cache`` is
    actually consumed).

Requires CUDA + flash_mla wheel. Skipped on dev boxes without these.
"""

from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.attention import _sparse_attn
from rtp_llm.models_py.modules.dsv4.decode.fp8_sparse_attn_decode_op import (
    _FLASH_MLA_AVAILABLE,
    SparseAttnV4DecodeFp8Op,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
    dequantize_slots_to_bf16,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
    quantize_and_insert_k_cache,
)

HEAD_DIM = 512
HEAD_BYTES = 584
# FlashMLA MODEL1 layout requires the per-block stride to be a multiple of
# TMA_K_STRIDE (= D_NOPE + 2*D_ROPE = 448 + 128 = 576) so the kernel's TMA
# tensormap can address the cache contiguously. vLLM allocates with
# ``alignment=576`` for the same reason; see
# vllm/v1/kv_cache_interface.py::_apply_alignment_padding.
_TMA_K_STRIDE = 576
# FlashMLA wheel constraint: page_block_size must be 64.
_FLASH_MLA_PAGE_BLOCK_SIZE = 64


def _populate_pool(
    num_blocks: int, block_size: int, num_tokens: int, device: torch.device
) -> tuple:
    """Populate ``num_tokens`` slots with random BF16, return (pool_3d, kv_bf16).

    The pool is allocated with TMA-aligned per-block stride: each block
    occupies ``round_up(block_size * 584, 576)`` bytes, so trailing padding
    (typically 64B for block_size=64) lives between consecutive blocks.
    The 3D view exposes the unpadded ``[num_blocks, block_size, 584]``
    shape with ``stride(0) = padded_per_block`` so FlashMLA's ``k_batch_stride
    % TMA_K_STRIDE == 0`` assertion is satisfied.
    """
    real_per_block = block_size * HEAD_BYTES
    padded_per_block = (
        (real_per_block + _TMA_K_STRIDE - 1) // _TMA_K_STRIDE
    ) * _TMA_K_STRIDE
    raw = torch.zeros(num_blocks * padded_per_block, dtype=torch.uint8, device=device)
    pool = torch.as_strided(
        raw,
        size=(num_blocks, block_size, HEAD_BYTES),
        stride=(padded_per_block, HEAD_BYTES, 1),
    )
    kv = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.1
    slots = torch.arange(num_tokens, dtype=torch.int64, device=device)
    quantize_and_insert_k_cache(kv, pool, slots)
    # Hold a reference to the underlying storage so the strided view stays alive.
    pool._raw_storage = raw  # type: ignore[attr-defined]
    return pool, kv


class TestFp8SparseAttnSinglePool(unittest.TestCase):

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        if not _FLASH_MLA_AVAILABLE:
            self.skipTest("flash_mla wheel not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)

    def test_single_pool_close_to_ref(self):
        B, q_len, H, D = 2, 1, 64, HEAD_DIM
        block_size = _FLASH_MLA_PAGE_BLOCK_SIZE  # FlashMLA wheel: must be 64
        win = 64  # SWA window — must be a multiple of FlashMLA B_TOPK tile

        # Allocate enough blocks: 1 block per request.
        swa_pool, swa_kv = _populate_pool(B, block_size, B * win, self.device)
        # Per-request top-K = window. Slot ids local to each block.
        # Each request uses block r; absolute slot = r * block_size + j.
        topk_idxs_global = (
            torch.arange(win, dtype=torch.int32, device=self.device)
            .view(1, 1, win)
            .expand(B, q_len, win)
            .contiguous()
            .clone()
        )
        # Bake in per-request offset: req r -> slots [r*block_size, r*block_size+win)
        topk_idxs_global = topk_idxs_global + (
            torch.arange(B, device=self.device, dtype=torch.int32).view(B, 1, 1)
            * block_size
        )

        q = torch.randn(B, q_len, H, D, dtype=torch.bfloat16, device=self.device) * 0.1
        sink = torch.zeros(H, dtype=torch.float32, device=self.device)
        sm_scale = D**-0.5

        block_table = (
            torch.arange(B, dtype=torch.int32, device=self.device)
            .view(B, 1)
            .contiguous()
        )
        cache_seqlens = torch.full(
            (B,), block_size, dtype=torch.int32, device=self.device
        )

        op = SparseAttnV4DecodeFp8Op(n_heads=H, head_dim=D, softmax_scale=sm_scale)
        out_fp8 = op.forward(
            q,
            swa_pool,
            sink,
            topk_idxs_global,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
        )
        self.assertEqual(tuple(out_fp8.shape), (B, q_len, H, D))

        # BF16 reference: dequant the SWA pool back, build per-request KV view,
        # call _sparse_attn with local-slot topk.
        swa_bf16_flat = dequantize_slots_to_bf16(
            swa_pool, topk_idxs_global.view(-1).to(torch.int64)
        ).view(B, win, D)
        topk_local = (
            torch.arange(win, dtype=torch.int64, device=self.device)
            .view(1, 1, win)
            .expand(B, q_len, win)
            .contiguous()
        )
        out_ref = _sparse_attn(q, swa_bf16_flat, sink, topk_local, sm_scale)

        diff = (out_fp8.float() - out_ref.float()).abs()
        ref_mag = out_ref.float().abs().mean().item() + 1e-9
        rel_mean = diff.mean().item() / ref_mag
        self.assertLess(
            rel_mean,
            0.20,
            f"single-pool fp8 vs bf16 rel_mean={rel_mean:.3e}",
        )


class TestFp8SparseAttnDualPool(unittest.TestCase):

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        if not _FLASH_MLA_AVAILABLE:
            self.skipTest("flash_mla wheel not available")
        self.device = torch.device("cuda")
        torch.manual_seed(7)

    def _build_dual_pool_inputs(self, B: int, win: int, K_cmp: int, block_size: int):
        """Build SWA + CMP FP8 pools with B requests, ``win`` SWA slots /
        ``K_cmp`` CMP slots per request. Returns:
          q, swa_pool, cmp_pool, swa_kv_bf16, cmp_kv_bf16,
          swa_topk_global, cmp_topk_global, swa_block_table, sink, sm_scale
        """
        H, D = 64, HEAD_DIM
        # 1 block per request per pool.
        swa_pool, swa_kv = _populate_pool(B, block_size, B * win, self.device)
        cmp_pool, cmp_kv = _populate_pool(B, block_size, B * K_cmp, self.device)

        # Per-request global slot ids: req r uses block r, slots [0, win) /
        # [0, K_cmp).
        swa_topk_global = (
            torch.arange(win, dtype=torch.int32, device=self.device)
            .view(1, 1, win)
            .expand(B, 1, win)
            .contiguous()
            .clone()
        ) + (
            torch.arange(B, device=self.device, dtype=torch.int32).view(B, 1, 1)
            * block_size
        )
        cmp_topk_global = (
            torch.arange(K_cmp, dtype=torch.int32, device=self.device)
            .view(1, 1, K_cmp)
            .expand(B, 1, K_cmp)
            .contiguous()
            .clone()
        ) + (
            torch.arange(B, device=self.device, dtype=torch.int32).view(B, 1, 1)
            * block_size
        )

        q = torch.randn(B, 1, H, D, dtype=torch.bfloat16, device=self.device) * 0.1
        sink = torch.zeros(H, dtype=torch.float32, device=self.device)
        sm_scale = D**-0.5
        swa_block_table = (
            torch.arange(B, dtype=torch.int32, device=self.device)
            .view(B, 1)
            .contiguous()
        )
        return (
            q,
            swa_pool,
            cmp_pool,
            swa_kv,
            cmp_kv,
            swa_topk_global,
            cmp_topk_global,
            swa_block_table,
            sink,
            sm_scale,
        )

    def test_dual_pool_close_to_bf16_cat_ref(self):
        """FlashMLA dual-pool ~= BF16(dequant SWA + dequant CMP, cat) sparse_attn."""
        # win + K_cmp must align with FlashMLA B_TOPK tile (mult of 64)
        B, win, K_cmp = 2, 64, 64
        block_size = _FLASH_MLA_PAGE_BLOCK_SIZE  # FlashMLA wheel: must be 64
        (
            q,
            swa_pool,
            cmp_pool,
            swa_kv,
            cmp_kv,
            swa_topk_global,
            cmp_topk_global,
            swa_block_table,
            sink,
            sm_scale,
        ) = self._build_dual_pool_inputs(B, win, K_cmp, block_size)
        H, D = q.shape[2], q.shape[3]

        op = SparseAttnV4DecodeFp8Op(n_heads=H, head_dim=D, softmax_scale=sm_scale)
        cache_seqlens = torch.full(
            (B,), block_size, dtype=torch.int32, device=self.device
        )
        out_fp8 = op.forward(
            q,
            swa_pool,
            sink,
            swa_topk_global,
            cache_seqlens=cache_seqlens,
            block_table=swa_block_table,
            extra_k_cache=cmp_pool,
            extra_topk_idxs=cmp_topk_global,
        )
        self.assertEqual(tuple(out_fp8.shape), (B, 1, H, D))

        # BF16 ref: dequant both pools per-request, concat along T axis,
        # build identity local topk = [0, win+K_cmp).
        swa_bf16 = dequantize_slots_to_bf16(
            swa_pool, swa_topk_global.view(-1).to(torch.int64)
        ).view(B, win, D)
        cmp_bf16 = dequantize_slots_to_bf16(
            cmp_pool, cmp_topk_global.view(-1).to(torch.int64)
        ).view(B, K_cmp, D)
        kv_packed = torch.cat([swa_bf16, cmp_bf16], dim=1)  # [B, win+K_cmp, D]
        T_total = win + K_cmp
        topk_local = (
            torch.arange(T_total, dtype=torch.int64, device=self.device)
            .view(1, 1, T_total)
            .expand(B, 1, T_total)
            .contiguous()
        )
        out_ref = _sparse_attn(q, kv_packed, sink, topk_local, sm_scale)

        diff = (out_fp8.float() - out_ref.float()).abs()
        ref_mag = out_ref.float().abs().mean().item() + 1e-9
        rel_mean = diff.mean().item() / ref_mag
        # FP8 quant noise on 2 pools' worth of cat vs FlashMLA's in-kernel
        # softmax merging — ~10-20% rel diff is expected on small shapes.
        self.assertLess(
            rel_mean,
            0.25,
            f"dual-pool fp8 vs bf16-cat ref rel_mean={rel_mean:.3e}",
        )

    def test_dual_pool_attends_more_than_single(self):
        """Sanity: dual-pool reads BOTH pools — attention output should
        differ from single-pool over just SWA when CMP has weight."""
        # win + K_cmp must align with FlashMLA B_TOPK tile (mult of 64)
        B, win, K_cmp = 2, 64, 64
        block_size = _FLASH_MLA_PAGE_BLOCK_SIZE  # FlashMLA wheel: must be 64
        (
            q,
            swa_pool,
            cmp_pool,
            swa_kv,
            cmp_kv,
            swa_topk_global,
            cmp_topk_global,
            swa_block_table,
            sink,
            sm_scale,
        ) = self._build_dual_pool_inputs(B, win, K_cmp, block_size)
        H, D = q.shape[2], q.shape[3]

        op = SparseAttnV4DecodeFp8Op(n_heads=H, head_dim=D, softmax_scale=sm_scale)
        cache_seqlens = torch.full(
            (B,), block_size, dtype=torch.int32, device=self.device
        )

        out_single = op.forward(
            q,
            swa_pool,
            sink,
            swa_topk_global,
            cache_seqlens=cache_seqlens,
            block_table=swa_block_table,
        )
        out_dual = op.forward(
            q,
            swa_pool,
            sink,
            swa_topk_global,
            cache_seqlens=cache_seqlens,
            block_table=swa_block_table,
            extra_k_cache=cmp_pool,
            extra_topk_idxs=cmp_topk_global,
        )

        diff = (out_single.float() - out_dual.float()).abs().mean().item()
        # CMP pool data is independent random — its contribution must
        # measurably shift the attention output. If dual-pool kwarg were
        # silently dropped, out_dual would equal out_single.
        self.assertGreater(
            diff,
            1e-3,
            f"dual-pool output should differ from single-pool; got mean diff={diff:.3e}",
        )


if __name__ == "__main__":
    unittest.main()

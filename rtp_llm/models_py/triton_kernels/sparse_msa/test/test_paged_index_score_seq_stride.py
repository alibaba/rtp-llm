import unittest
from types import SimpleNamespace

import torch


def _op_available() -> bool:
    try:
        from rtp_llm.ops.compute_ops import rtp_llm_ops
    except Exception:
        return False
    return hasattr(rtp_llm_ops, "minimax_decode_topk")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(_op_available(), "minimax_decode_topk op is unavailable")
class PagedIndexScoreSeqStrideTest(unittest.TestCase):
    @staticmethod
    def _scaled_side_region(
        blocks: int, page_size: int, head_dim: int
    ) -> tuple[object, torch.Tensor, torch.Tensor, torch.Tensor]:
        from rtp_llm.models_py.modules.hybrid.msa_attention import MSAAttention

        attn = MSAAttention.__new__(MSAAttention)
        torch.nn.Module.__init__(attn)
        attn.page_size = page_size
        attn.idx_head_dim = head_dim
        attn.idx_k_fp8_mode = 1
        attn._idx_k_persistent_dtype = torch.float8_e4m3fn

        carrier = torch.zeros(
            blocks,
            page_size * (head_dim + 4) // 4,
            dtype=torch.float32,
            device="cuda",
        )
        cache = SimpleNamespace(kv_scale_base=carrier)
        values, scales = attn._idx_k_paged_storage(cache)
        return attn, carrier, values, scales

    def test_scaled_side_region_decode_writer_write_through_roundtrip(self) -> None:
        page_size, head_dim, blocks = 128, 128, 3
        attn, carrier, values, scales = self._scaled_side_region(
            blocks, page_size, head_dim
        )
        base = torch.zeros(
            blocks,
            2,
            1,
            page_size,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        attn._paged_kv_base_view = lambda _cache: base
        cache = SimpleNamespace(kv_scale_base=carrier)

        # One decode token per physical block. The selected slots cross both
        # interleaved per-block scale trailers: 0, page+1, and 2*page+2.
        seq_lens = torch.tensor(
            [1, page_size + 2, 2 * page_size + 3],
            dtype=torch.int32,
            device="cuda",
        )
        block_table = torch.arange(blocks, dtype=torch.int32, device="cuda").repeat(
            blocks, 1
        )
        k = torch.zeros(blocks, 1, head_dim, dtype=torch.bfloat16, device="cuda")
        v = torch.zeros_like(k)
        idx_k = torch.stack(
            [
                torch.linspace(-1.0, 1.0, head_dim, device="cuda"),
                torch.linspace(-3.0, 2.0, head_dim, device="cuda"),
                torch.linspace(-0.25, 4.0, head_dim, device="cuda"),
            ]
        ).to(torch.bfloat16)

        result = attn._write_kv_cache_and_idx_k_for_decode(
            cache, k, v, idx_k, seq_lens, block_table
        )
        self.assertIsNotNone(result)
        torch.cuda.synchronize()

        offsets = (0, 1, 2)
        for token, offset in enumerate(offsets):
            with self.subTest(token=token):
                expected_scale = (
                    idx_k[token].float().abs().amax().div(448.0).clamp_min(1.0e-12)
                )
                expected_value = (idx_k[token].float() / expected_scale).to(
                    torch.float8_e4m3fn
                )
                self.assertTrue(
                    torch.equal(
                        values[token, offset].view(torch.uint8),
                        expected_value.view(torch.uint8),
                    )
                )
                torch.testing.assert_close(
                    scales[token, offset], expected_scale, rtol=1e-6, atol=0
                )
                restored = values[token, offset].float() * scales[token, offset]
                torch.testing.assert_close(
                    restored, idx_k[token].float(), rtol=0.07, atol=1e-4
                )

    def test_scaled_interleaved_side_region_paged_decode_topk(self) -> None:
        from rtp_llm.models_py.triton_kernels.sparse_msa.decode.flash_with_topk_idx import (
            flash_decode_with_topk_idx_paged,
        )

        page_size, head_dim, blocks = 128, 128, 3
        _, _, values, scales = self._scaled_side_region(blocks, page_size, head_dim)

        # Stored FP8 values are identical. Only the per-block scales distinguish
        # scores, so ignoring the real interleaved scale stride changes top-1.
        values.fill_(1.0)
        scales[0].fill_(1.0)
        scales[1].fill_(3.0)
        scales[2].fill_(2.0)
        q = torch.ones(1, 1, head_dim, dtype=torch.bfloat16, device="cuda")
        # Logical blocks [0, 1, 2] map to physical blocks [2, 0, 1]. Correct
        # dequantization therefore selects logical block 2 (physical scale 3).
        block_table = torch.tensor([[2, 0, 1]], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([blocks * page_size], dtype=torch.int32, device="cuda")

        _, actual = flash_decode_with_topk_idx_paged(
            q=q,
            k_paged=values,
            k_scale=scales,
            block_table=block_table,
            seq_lens=seq_lens,
            max_seqlen=blocks * page_size,
            block_size=page_size,
            topk=1,
            init_blocks=0,
            local_blocks=0,
            score_type="max",
        )
        torch.cuda.synchronize()
        self.assertEqual(actual.numel(), 1)
        self.assertEqual(actual.item(), 2)

    def test_scaled_fp8_paged_idx_matches_bf16_topk(self) -> None:
        from rtp_llm.models_py.modules.hybrid.msa_attention import _write_idx_rows
        from rtp_llm.models_py.triton_kernels.sparse_msa.decode.flash_with_topk_idx import (
            flash_decode_with_topk_idx_paged,
        )

        torch.manual_seed(20260918)
        page_size, head_dim, pages = 128, 128, 8
        q = torch.randn(2, 4, head_dim, dtype=torch.bfloat16, device="cuda")
        bf16_k = torch.randn(
            pages, page_size, head_dim, dtype=torch.bfloat16, device="cuda"
        )
        _, _, fp8_k, scales = self._scaled_side_region(pages, page_size, head_dim)
        slots = torch.arange(pages * page_size, dtype=torch.int64, device="cuda")
        _write_idx_rows(bf16_k.reshape(-1, head_dim), slots, fp8_k, scales)
        block_table = torch.arange(pages, dtype=torch.int32, device="cuda").repeat(2, 1)
        seq_lens = torch.tensor(
            [pages * page_size - 1, pages * page_size], device="cuda"
        )
        common = dict(
            q=q,
            block_table=block_table,
            seq_lens=seq_lens,
            max_seqlen=pages * page_size,
            block_size=page_size,
            topk=4,
            init_blocks=0,
            local_blocks=0,
        )
        bf16_topk = flash_decode_with_topk_idx_paged(k_paged=bf16_k, **common)[1]
        fp8_topk = flash_decode_with_topk_idx_paged(
            k_paged=fp8_k,
            k_scale=scales,
            **common,
        )[1]
        torch.cuda.synchronize()
        overlap = (fp8_topk[..., :, None] == bf16_topk[..., None, :]).any(-1).float()
        # Random scores around the top-k boundary are intentionally sensitive
        # to E4M3 quantization. This unit test guards against gross scale/layout
        # errors; matched end-to-end accuracy is covered separately.
        self.assertGreater(overlap.mean().item(), 0.90)

    def test_fused_norm_rope_write_persists_scaled_fp8_idx_k(self) -> None:
        from rtp_llm.models_py.modules.hybrid.msa_attention import (
            _fused_qk_idx_norm_rope_write_paged_decode,
            _write_decode_kv_idx_to_paged,
        )

        torch.manual_seed(20260919)
        page_size, head_dim, rotary_dim, blocks = 128, 128, 64, 4
        num_q_heads, num_kv_heads, num_idx_heads = 2, 1, 1
        fused_heads = num_q_heads + 2 * num_kv_heads + num_idx_heads + 1
        device = torch.device("cuda")
        tokens = 5
        fused = torch.randn(
            tokens, fused_heads * head_dim, dtype=torch.bfloat16, device=device
        )
        norm_weights = [
            torch.rand(head_dim, dtype=torch.bfloat16, device=device) + 0.5
            for _ in range(4)
        ]
        cos_sin = torch.randn(
            tokens + 1, rotary_dim, dtype=torch.float32, device=device
        )
        pos_ids = torch.arange(tokens, dtype=torch.int64, device=device)
        # One decode token per request; block starts and non-zero in-page
        # offsets across physical blocks 0/1/2/1/0, with REM != 0 (rotary_dim
        # < head_dim) so the non-RoPE tail joins the absmax quantization.
        seq_lens = torch.tensor([1, 129, 257, 130, 3], dtype=torch.int32, device=device)
        block_table = torch.tensor(
            [[0], [1], [2], [1], [0]], dtype=torch.int32, device=device
        )
        expected_slots = [(0, 0), (1, 0), (2, 0), (1, 1), (0, 2)]

        def run_fused(idx_values, idx_scales, kv_pool):
            _fused_qk_idx_norm_rope_write_paged_decode(
                fused,
                torch.empty(
                    tokens, num_q_heads, head_dim, dtype=torch.bfloat16, device=device
                ),
                torch.empty(
                    tokens, num_idx_heads, head_dim, dtype=torch.bfloat16, device=device
                ),
                *norm_weights,
                cos_sin,
                pos_ids,
                seq_lens,
                block_table,
                kv_pool,
                idx_values,
                idx_scales,
                page_size,
                head_dim,
                rotary_dim,
                num_q_heads,
                num_kv_heads,
                num_idx_heads,
                1.0e-5,
            )

        # Mode-0 baseline: the fused kernel persists BF16 idx_K, which is the
        # exact materialized input the unfused writer quantizes.
        bf16_pool = torch.zeros(
            blocks, page_size, head_dim, dtype=torch.bfloat16, device=device
        )
        kv_bf16_run = torch.zeros(
            blocks,
            2,
            num_kv_heads,
            page_size,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        run_fused(bf16_pool, None, kv_bf16_run)
        torch.cuda.synchronize()
        bf16_idx = torch.stack([bf16_pool[b, o] for b, o in expected_slots])
        self.assertGreater(bf16_idx.abs().max().item(), 0.0)

        # Reference: quantize the materialized BF16 idx_K with the unfused
        # stride-aware writer into one scaled side region.
        _, _, ref_values, ref_scales = self._scaled_side_region(
            blocks, page_size, head_dim
        )
        _write_decode_kv_idx_to_paged(
            torch.randn(
                tokens, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
            ),
            torch.randn(
                tokens, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
            ),
            bf16_idx,
            seq_lens,
            block_table,
            torch.zeros(
                blocks,
                2,
                num_kv_heads,
                page_size,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            ref_values,
            ref_scales,
            page_size,
            head_dim,
        )
        torch.cuda.synchronize()

        # Fused mode 1/2 path: same fused inputs, scaled-E4M3 side region.
        _, _, fused_values, fused_scales = self._scaled_side_region(
            blocks, page_size, head_dim
        )
        kv_fp8_run = torch.zeros(
            blocks,
            2,
            num_kv_heads,
            page_size,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        run_fused(fused_values, fused_scales, kv_fp8_run)
        torch.cuda.synchronize()

        self.assertTrue(
            torch.equal(fused_values.view(torch.uint8), ref_values.view(torch.uint8))
        )
        torch.testing.assert_close(fused_scales, ref_scales, rtol=1e-6, atol=0)
        # K/V persistence must be untouched by the idx_K quantization branch.
        self.assertTrue(torch.equal(kv_fp8_run, kv_bf16_run))

    def test_strided_request_final_lengths_match_contiguous(self) -> None:
        from rtp_llm.models_py.triton_kernels.sparse_msa.decode.flash_with_topk_idx import (
            flash_decode_with_topk_idx_paged,
        )

        block_size = 128
        head_dim = 64
        batch_size = 4
        max_blocks = 8
        topk = 2

        # Later logical blocks have strictly larger scores, so reading the
        # wrong request length changes the selected top-k set deterministically.
        k_paged = torch.stack(
            [
                torch.full(
                    (block_size, head_dim),
                    float(block + 1),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                for block in range(max_blocks)
            ]
        )
        block_table = torch.arange(max_blocks, device="cuda", dtype=torch.int32).repeat(
            batch_size, 1
        )

        for verify_width in (2, 3, 4):
            with self.subTest(verify_width=verify_width):
                q = torch.ones(
                    batch_size * verify_width,
                    1,
                    head_dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                final_lens = (
                    torch.tensor([3, 4, 5, 6], device="cuda", dtype=torch.int32)
                    * block_size
                )
                offsets = torch.arange(
                    verify_width - 1, -1, -1, device="cuda", dtype=torch.int32
                )
                token_seq_lens = (final_lens[:, None] - offsets[None, :]).reshape(-1)
                request_final_lens = token_seq_lens.view(batch_size, verify_width)[
                    :, -1
                ]
                self.assertEqual(request_final_lens.stride(), (verify_width,))

                kwargs = dict(
                    q=q,
                    k_paged=k_paged,
                    block_table=block_table,
                    max_seqlen=int(final_lens.max().item()),
                    block_size=block_size,
                    topk=topk,
                    init_blocks=0,
                    local_blocks=0,
                    score_type="max",
                    decode_query_len=verify_width,
                    token_seq_lens=token_seq_lens,
                )
                _, actual = flash_decode_with_topk_idx_paged(
                    seq_lens=request_final_lens, **kwargs
                )
                _, control = flash_decode_with_topk_idx_paged(
                    seq_lens=request_final_lens.contiguous(), **kwargs
                )
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.equal(actual.cpu(), control.cpu()),
                    f"strided seq_lens mismatch for verify_width={verify_width}",
                )


if __name__ == "__main__":
    unittest.main()

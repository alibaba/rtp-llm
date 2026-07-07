"""UT for the FP8 sparse prefill dispatch helper used by
SparseMlaFp8Op._forward_gather / SparseMlaFp8CPOp._attend_gather when
RTP_LLM_GLM5_SPARSE_ATTN_DTYPE=fp8.

Requires Blackwell SM100 (skips otherwise). Covers:
  - env resolution and SM guard
  - pack_q_656 shape/scale
  - paged_fp8_gather_pack roundtrip (paged FP8 with per-group scales →
    per-tensor FP8 ragged packed layout)
  - sparse_prefill_fp8_from_paged_cache matches flash_mla_sparse_fwd (bf16)
    within fp8 tolerance
"""

import os
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl._sparse_prefill_fp8_pack import (
    resolve_sparse_attn_dtype,
    sparse_prefill_fp8_from_paged_cache,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.sparse_fp8_prefill_pack import (
    BYTES_PER_TOKEN,
    FP8_MAX,
    GROUP_SIZE,
    KV_LORA,
    NUM_GROUPS,
    ROPE_OFFSET_BF16,
    SCALE_OFFSET_FP32,
    pack_q_656,
    paged_fp8_gather_pack,
)


def _build_paged_from_bf16(
    kv_bf16: torch.Tensor,
    tokens_per_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate the FP8 write side: quantize per 128-group, lay out 656B/token.

    Returns (paged_u8[num_blocks, block_size, 656], block_table[batch, blocks]).
    kv_bf16 is [total, 576] treated as a single batch.
    """
    assert kv_bf16.shape[-1] == 576
    total, _ = kv_bf16.shape
    num_blocks = (total + tokens_per_block - 1) // tokens_per_block
    device = kv_bf16.device

    paged = torch.zeros(
        (num_blocks, tokens_per_block, BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device=device,
    )
    paged_fp8 = paged.view(torch.float8_e4m3fn)
    paged_fp32 = paged.view(torch.float32)
    paged_bf16 = paged.view(torch.bfloat16)

    for t in range(total):
        blk = t // tokens_per_block
        slot = t % tokens_per_block
        nope = kv_bf16[t, :KV_LORA].to(torch.float32)
        rope = kv_bf16[t, KV_LORA:]
        for g in range(NUM_GROUPS):
            tile = nope[g * GROUP_SIZE : (g + 1) * GROUP_SIZE]
            scale = max(tile.abs().max().item() / FP8_MAX, 1e-6)
            paged_fp32[blk, slot, SCALE_OFFSET_FP32 + g] = scale
            fp8_vals = (tile / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            paged_fp8[blk, slot, g * GROUP_SIZE : (g + 1) * GROUP_SIZE] = fp8_vals
        paged_bf16[blk, slot, ROPE_OFFSET_BF16 : ROPE_OFFSET_BF16 + 64] = rope

    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(
        0
    )
    return paged, block_table


class SparsePrefillFP8PackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        cls.device = torch.device("cuda:0")
        torch.cuda.set_device(cls.device)
        cls.cc = torch.cuda.get_device_capability(cls.device)

    def setUp(self):
        resolve_sparse_attn_dtype.cache_clear()

    def tearDown(self):
        resolve_sparse_attn_dtype.cache_clear()

    def _require_sm100(self):
        if self.cc[0] != 10:
            self.skipTest(f"requires SM100; got {self.cc}")

    def test_env_default_bf16(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RTP_LLM_GLM5_SPARSE_ATTN_DTYPE", None)
            self.assertEqual(resolve_sparse_attn_dtype(), "bf16")

    def test_env_fp8_on_sm100(self):
        self._require_sm100()
        with mock.patch.dict(os.environ, {"RTP_LLM_GLM5_SPARSE_ATTN_DTYPE": "fp8"}):
            self.assertEqual(resolve_sparse_attn_dtype(), "fp8")

    def test_env_invalid_raises(self):
        with mock.patch.dict(os.environ, {"RTP_LLM_GLM5_SPARSE_ATTN_DTYPE": "int8"}):
            with self.assertRaises(ValueError):
                resolve_sparse_attn_dtype()

    def test_pack_q_shape_and_scale(self):
        """Triton pack_q_656 emits the expected 656B/token layout."""
        self._require_sm100()
        s_q, h_q = 32, 64
        q = torch.randn(s_q, h_q, 576, device=self.device, dtype=torch.bfloat16)
        pkg, q_scale = pack_q_656(q, k_scale=0.01)
        self.assertEqual(pkg.shape, (s_q, h_q, BYTES_PER_TOKEN))
        self.assertEqual(pkg.dtype, torch.uint8)
        self.assertGreater(q_scale, 0.0)
        # 16B pad region should be zero
        pad = pkg[:, :, KV_LORA : KV_LORA + 16]
        self.assertTrue(torch.equal(pad, torch.zeros_like(pad)))

    def test_paged_gather_pack_roundtrip(self):
        """Round-trip: bf16 KV → paged (per-group fp8) → gather+repack (per-tensor fp8).

        Dequantizing the packed output should reproduce the original bf16 KV within
        fp8 tolerance (values pass through two fp8 casts: per-group then rescale).
        """
        self._require_sm100()
        tokens_per_block = 64
        total = 256
        g = torch.Generator(device=self.device).manual_seed(2028)
        kv_bf16 = torch.randn(
            total, 576, device=self.device, dtype=torch.bfloat16, generator=g
        )

        paged, block_table = _build_paged_from_bf16(kv_bf16, tokens_per_block)
        workspace_starts = torch.tensor([0], dtype=torch.int32, device=self.device)
        seq_lens = torch.tensor([total], dtype=torch.int32, device=self.device)

        packed, k_scale = paged_fp8_gather_pack(
            paged,
            block_table,
            workspace_starts,
            seq_lens,
            batch_size=1,
            total_kv_len=total,
            tokens_per_block=tokens_per_block,
        )
        self.assertEqual(packed.shape, (total, BYTES_PER_TOKEN))
        self.assertGreater(k_scale, 0.0)

        # Dequant nope: packed[:, :512] fp8 * k_scale (per-tensor) → bf16 target
        packed_fp8 = packed.view(torch.float8_e4m3fn)
        nope_deq = packed_fp8[:, :KV_LORA].to(torch.float32) * k_scale
        max_abs = kv_bf16[:, :KV_LORA].float().abs().max().item()
        # fp8_e4m3 has ~2-4% relative error per cast; two casts double this
        atol = 0.06 * max_abs
        self.assertTrue(
            torch.allclose(nope_deq, kv_bf16[:, :KV_LORA].float(), atol=atol),
            f"nope roundtrip failed: max_diff={(nope_deq - kv_bf16[:, :KV_LORA].float()).abs().max().item()} "
            f"atol={atol}",
        )

        # RoPE bytes copied verbatim
        packed_bf16 = packed.view(torch.bfloat16)
        rope_out = packed_bf16[:, ROPE_OFFSET_BF16 : ROPE_OFFSET_BF16 + 64]
        self.assertTrue(torch.equal(rope_out, kv_bf16[:, KV_LORA:].contiguous()))

        # 16B pad region is zeroed
        pad = packed[:, KV_LORA : KV_LORA + 16]
        self.assertTrue(torch.equal(pad, torch.zeros_like(pad)))

    def test_dispatch_matches_bf16_reference(self):
        """sparse_prefill_fp8_from_paged_cache output should track flash_mla_sparse_fwd
        (bf16 kernel run on the same input) within fp8 tolerance."""
        self._require_sm100()
        from flash_mla import flash_mla_sparse_fwd

        tokens_per_block = 64
        s_q, s_kv, topk = 128, 1024, 128
        g = torch.Generator(device=self.device).manual_seed(2027)
        q = torch.randn(
            s_q, 64, 576, device=self.device, dtype=torch.bfloat16, generator=g
        )
        kv = torch.randn(
            s_kv, 576, device=self.device, dtype=torch.bfloat16, generator=g
        )

        # Build paged FP8 buffer from the same bf16 KV (single-request batch)
        paged, block_table = _build_paged_from_bf16(kv, tokens_per_block)
        workspace_starts = torch.tensor([0], dtype=torch.int32, device=self.device)
        seq_lens = torch.tensor([s_kv], dtype=torch.int32, device=self.device)

        indices_rows = []
        for _ in range(s_q):
            perm = torch.randperm(
                s_kv, device=self.device, generator=g, dtype=torch.int32
            )
            indices_rows.append(perm[:topk].clone())
        indices = torch.stack(indices_rows).view(s_q, 1, topk)

        sm_scale = 576**-0.5
        out_bf16, _, _ = flash_mla_sparse_fwd(
            q,
            kv.unsqueeze(1),
            indices,
            sm_scale,
            d_v=KV_LORA,
        )
        packed_ws = torch.empty(
            (s_kv, BYTES_PER_TOKEN), dtype=torch.uint8, device=self.device
        )
        out_fp8 = sparse_prefill_fp8_from_paged_cache(
            q_bf16=q,
            paged_u8=paged,
            block_table=block_table,
            workspace_starts=workspace_starts,
            seq_lens=seq_lens,
            batch_size=1,
            total_kv_len=s_kv,
            tokens_per_block=tokens_per_block,
            global_indices=indices,
            sm_scale=sm_scale,
            d_v=KV_LORA,
            packed_kv_workspace=packed_ws,
        )
        rel = (
            out_fp8.float() - out_bf16.float()
        ).abs().mean().item() / out_bf16.float().abs().mean().clamp_min(1e-9).item()
        # ~5-8% is intrinsic fp8_e4m3 vs bf16 quantization noise (per-tensor
        # rescale on Q + K passes each through one fp8 cast); use a generous
        # bound — golden refresh handles tighter validation.
        self.assertLess(rel, 0.10, f"rel_diff={rel:.4f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

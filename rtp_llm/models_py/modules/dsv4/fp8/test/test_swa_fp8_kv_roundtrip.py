"""UT: SWA FP8 KV cache quantize-and-insert ↔ dequantize-and-gather round-trip.

Covers two Triton kernels:
  * ``_swa_fp8_kv_insert_triton.quantize_and_insert_k_cache``
    (vendored from vLLM ``cache_utils.py:quantize_and_insert_k_kernel``)
  * ``_swa_fp8_dequant_triton.dequantize_and_gather_k_cache``
    (vendored from vLLM ``cache_utils.py:_dequantize_and_gather_k_kernel``)

Mirrors vLLM ``tests/kernels/test_compressor_kv_cache.py``:
``test_deepseek_v4_attention_quant_cache_roundtrip`` +
``test_deepseek_v4_quant_magnitude_range``, restructured as
``unittest.TestCase``.

End-to-end validation:
  1. quantize + insert random BF16 K into the FP8 SWA cache (584B/token)
  2. dequantize + gather back into a BF16 workspace
  3. NoPE (first 448 elements): per-token UE8M0 quant noise bound
     (``<= 16 * tile_scale``)
  4. RoPE (last 64 elements): byte-exact (passthrough, no quant)

Run:
  CUDA_VISIBLE_DEVICES=7 /opt/conda310/bin/python3 -m unittest \\
    rtp_llm.models_py.modules.dsv4.test.test_swa_fp8_kv_roundtrip
"""

from __future__ import annotations

import math
import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
    dequantize_and_gather_k_cache,
    dequantize_packed_k_cache_flat,
    gather_k_cache_packed,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
    quantize_and_insert_k_cache,
)

HEAD_DIM = 512
NOPE_DIM = 448
HEAD_BYTES = 584  # 448 fp8 + 128 bf16 + 8 uint8 scale
FP8_MAX = 448.0
QUANT_BLOCK = 64


def _ue8m0_reference_max_scale(token_nope_bf16: torch.Tensor) -> float:
    """Compute the max UE8M0 tile-scale a reference impl would assign
    across the 7 NoPE quant tiles (each 64 elements). Used to bound
    expected post-roundtrip error (FP8 e4m3 worst-case = 16 * scale).
    """
    assert token_nope_bf16.dim() == 1 and token_nope_bf16.numel() == NOPE_DIM
    n_tiles = NOPE_DIM // QUANT_BLOCK
    max_scale = 0.0
    for i in range(n_tiles):
        tile = token_nope_bf16[i * QUANT_BLOCK : (i + 1) * QUANT_BLOCK].float()
        amax = max(tile.abs().max().item(), 1e-4)
        exponent = math.ceil(math.log2(amax / FP8_MAX))
        scale = 2.0**exponent
        max_scale = max(max_scale, scale)
    return max_scale


class SwaFp8KvRoundtripTest(unittest.TestCase):

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _alloc_cache(self, num_blocks: int, block_size: int) -> torch.Tensor:
        """[num_blocks, block_size, 584] uint8 — matches RTP-LLM SWA pool."""
        return torch.zeros(
            num_blocks, block_size, HEAD_BYTES, dtype=torch.uint8, device=self.device
        )

    def _roundtrip(
        self,
        compressed_kv: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """quantize+insert sequential slots → dequant+gather → return [T, 512] bf16."""
        num_tokens = compressed_kv.shape[0]
        data_blocks = (num_tokens + block_size - 1) // block_size
        num_blocks = data_blocks + 1

        k_cache = self._alloc_cache(num_blocks, block_size)
        # Production block id 0 is invalid; valid physical blocks are positive.
        slot_mapping = (
            torch.arange(num_tokens, dtype=torch.int64, device=self.device)
            + block_size
        )
        quantize_and_insert_k_cache(compressed_kv, k_cache, slot_mapping)

        out = torch.zeros(
            1, num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )
        seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=self.device)
        block_table = torch.arange(
            1, data_blocks + 1, dtype=torch.int32, device=self.device
        ).unsqueeze(0)
        dequantize_and_gather_k_cache(
            out=out,
            k_cache=k_cache,
            seq_lens=seq_lens,
            gather_lens=None,
            block_table=block_table,
            block_size=block_size,
            offset=0,
        )
        return out[0, :num_tokens]

    def _roundtrip_packed_gather(
        self,
        compressed_kv: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """quantize+insert → packed gather → flat dequant."""
        num_tokens = compressed_kv.shape[0]
        num_blocks = (num_tokens + block_size - 1) // block_size + 1

        k_cache = self._alloc_cache(num_blocks, block_size)
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=self.device)
        quantize_and_insert_k_cache(compressed_kv, k_cache, slot_mapping)

        seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=self.device)
        block_table = torch.arange(
            num_blocks, dtype=torch.int32, device=self.device
        ).unsqueeze(0)

        packed = torch.zeros(
            1, num_tokens, HEAD_BYTES, dtype=torch.uint8, device=self.device
        )
        gather_k_cache_packed(
            out=packed,
            k_cache=k_cache,
            seq_lens=seq_lens,
            gather_lens=None,
            block_table=block_table,
            block_size=block_size,
            offset=0,
        )
        out = torch.empty(
            num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )
        dequantize_packed_k_cache_flat(out, packed[0])
        return out

    def _assert_nope_within_ue8m0_bound(
        self, original: torch.Tensor, recovered: torch.Tensor
    ):
        """Per-token NoPE diff must stay within 16 * max_tile_scale."""
        nope_orig = original[:, :NOPE_DIM]
        nope_recv = recovered[:, :NOPE_DIM]
        diff = (nope_recv.float() - nope_orig.float()).abs()
        for t in range(original.shape[0]):
            scale = _ue8m0_reference_max_scale(nope_orig[t])
            max_allowed = 16.0 * scale
            token_diff = diff[t].max().item()
            self.assertLessEqual(
                token_diff,
                max_allowed,
                msg=(
                    f"NoPE token {t}: diff={token_diff:.4g} exceeds "
                    f"max_allowed={max_allowed:.4g} (tile_scale={scale:.4g})"
                ),
            )

    def _assert_rope_exact(self, original: torch.Tensor, recovered: torch.Tensor):
        """RoPE region is BF16 passthrough — byte-exact."""
        rope_orig = original[:, NOPE_DIM:]
        rope_recv = recovered[:, NOPE_DIM:]
        diff = (rope_recv - rope_orig).abs().max().item()
        self.assertEqual(
            diff,
            0.0,
            msg=f"RoPE should be byte-exact, got max diff {diff}",
        )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_random_roundtrip_block_64(self):
        """Sweep token counts at block_size=64 (vLLM default page size)."""
        for num_tokens in [1, 4, 8, 17, 64, 100]:
            with self.subTest(num_tokens=num_tokens):
                compressed_kv = torch.randn(
                    num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=self.device
                )
                recovered = self._roundtrip(compressed_kv, block_size=64)
                self._assert_nope_within_ue8m0_bound(compressed_kv, recovered)
                self._assert_rope_exact(compressed_kv, recovered)

    def test_random_roundtrip_block_256(self):
        """Sweep token counts at block_size=256 (RTP-LLM eb=256)."""
        for num_tokens in [1, 32, 256, 257, 600]:
            with self.subTest(num_tokens=num_tokens):
                compressed_kv = torch.randn(
                    num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=self.device
                )
                recovered = self._roundtrip(compressed_kv, block_size=256)
                self._assert_nope_within_ue8m0_bound(compressed_kv, recovered)
                self._assert_rope_exact(compressed_kv, recovered)

    def test_packed_gather_matches_direct_dequant(self):
        """Packed-FP8 gather + local dequant must be bitwise-equivalent to
        the original gather+dequant path. This pins the CP all_gather payload
        optimization's byte layout.
        """
        for block_size, num_tokens in [(64, 117), (256, 513)]:
            with self.subTest(block_size=block_size, num_tokens=num_tokens):
                compressed_kv = torch.randn(
                    num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=self.device
                )
                direct = self._roundtrip(compressed_kv, block_size=block_size)
                packed = self._roundtrip_packed_gather(
                    compressed_kv, block_size=block_size
                )
                self.assertTrue(torch.equal(direct, packed))

    def test_magnitude_range(self):
        """Per-token NoPE quant scale must adapt to token magnitude.

        Mirrors vLLM ``test_deepseek_v4_quant_magnitude_range``.
        """
        block_size = 16
        num_tokens = 4
        compressed_kv = torch.zeros(
            num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )
        compressed_kv[0] = 0.001  # very small
        compressed_kv[1] = 1.0  # unit scale
        compressed_kv[2] = 100.0  # large
        compressed_kv[3] = torch.randn(
            HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )

        recovered = self._roundtrip(compressed_kv, block_size=block_size)
        self._assert_nope_within_ue8m0_bound(compressed_kv, recovered)
        self._assert_rope_exact(compressed_kv, recovered)

    def test_skipped_slots_not_overwritten(self):
        """slot_mapping with -1 sentinels must skip those tokens
        without polluting the cache (insert kernel contract)."""
        block_size = 16
        num_tokens = 6
        compressed_kv = torch.randn(
            num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )
        # Token 1 and 4 marked for skip; their slots stay zero.
        slot_mapping = torch.tensor(
            [16, -1, 17, 18, -1, 19], dtype=torch.int64, device=self.device
        )
        num_blocks = 2
        k_cache = self._alloc_cache(num_blocks, block_size)
        # Sentinel pre-fill so we can detect any unexpected write.
        sentinel = 0xAB
        k_cache.fill_(sentinel)

        quantize_and_insert_k_cache(compressed_kv, k_cache, slot_mapping)

        # Read back valid slots via gather and confirm they decode close
        # to original; valid slot positions are 0,1,2,3 in physical block 1.
        out = torch.zeros(1, 4, HEAD_DIM, dtype=torch.bfloat16, device=self.device)
        seq_lens = torch.tensor([4], dtype=torch.int32, device=self.device)
        block_table = torch.tensor([[1, -1]], dtype=torch.int32, device=self.device)
        dequantize_and_gather_k_cache(
            out=out,
            k_cache=k_cache,
            seq_lens=seq_lens,
            gather_lens=None,
            block_table=block_table,
            block_size=block_size,
            offset=0,
        )
        # Map back: out[0,0] ↔ compressed_kv[0]; out[0,1]↔kv[2]; etc.
        kv_valid = compressed_kv[[0, 2, 3, 5]]
        self._assert_rope_exact(kv_valid, out[0])
        self._assert_nope_within_ue8m0_bound(kv_valid, out[0])

        # Slots 4..15 of block 1 (untouched) plus all of block 0 must
        # still be sentinel — confirming -1 skipped tokens didn't
        # accidentally write somewhere.
        # The kernel uses a packed-per-block layout: each block holds
        #   [block_size * 576 token-data bytes || block_size * 8 scale bytes],
        # NOT a per-token contiguous [block_size, 584] view. So inspect the
        # block as a flat byte buffer and check the kernel-aligned untouched
        # regions: data bytes [4*576, 9216) and scales bytes [9216+4*8, 9344).
        TOKEN_DATA_SIZE = 576  # 448 fp8 + 128 bf16 (RoPE)
        TOKEN_SCALE_SIZE = 8
        block1_flat = k_cache[1].reshape(-1)
        data_region_end = block_size * TOKEN_DATA_SIZE  # 9216 for block_size=16
        scales_touched_end = data_region_end + 4 * TOKEN_SCALE_SIZE  # 9248
        untouched_data = block1_flat[4 * TOKEN_DATA_SIZE : data_region_end]
        untouched_scales = block1_flat[scales_touched_end:]
        self.assertTrue(
            torch.all(untouched_data == sentinel),
            msg="Untouched slot-data bytes in block 1 should remain sentinel.",
        )
        self.assertTrue(
            torch.all(untouched_scales == sentinel),
            msg="Untouched scale bytes in block 1 should remain sentinel.",
        )
        self.assertTrue(
            torch.all(k_cache[0] == sentinel),
            msg="Block 0 should remain sentinel (not targeted by this test).",
        )

    def test_sparse_paged_block_table(self):
        """Non-sequential block_table: physical block IDs jumbled relative
        to logical position. Verifies the dequant kernel resolves
        physical block via ``block_table[req, pos // block_size]`` and
        the insert kernel respects the paged formula
        ``slot = block_id * block_size + pos_in_block``.
        """
        block_size = 64
        num_tokens = 200  # 4 logical blocks (last partial)
        n_logical = (num_tokens + block_size - 1) // block_size  # 4
        # Physical blocks intentionally permuted.
        physical_ids = [3, 1, 7, 5]
        num_blocks = max(physical_ids) + 1

        compressed_kv = torch.randn(
            num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )
        k_cache = self._alloc_cache(num_blocks, block_size)

        # Build slot_mapping: token i → physical_ids[i // block_size] * block_size + (i % block_size)
        slot_mapping = torch.tensor(
            [
                physical_ids[i // block_size] * block_size + (i % block_size)
                for i in range(num_tokens)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        quantize_and_insert_k_cache(compressed_kv, k_cache, slot_mapping)

        # Gather back through the same block_table layout.
        out = torch.zeros(
            1, num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )
        seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=self.device)
        block_table = torch.tensor(
            [physical_ids], dtype=torch.int32, device=self.device
        )
        # Pad the block_table to the max ``ceil(seq_len/block_size)`` via the
        # logical layout — kernel reads ``block_table[req, pos // block_size]``.
        self.assertEqual(block_table.shape, (1, n_logical))
        dequantize_and_gather_k_cache(
            out=out,
            k_cache=k_cache,
            seq_lens=seq_lens,
            gather_lens=None,
            block_table=block_table,
            block_size=block_size,
            offset=0,
        )
        recovered = out[0, :num_tokens]
        self._assert_nope_within_ue8m0_bound(compressed_kv, recovered)
        self._assert_rope_exact(compressed_kv, recovered)


if __name__ == "__main__":
    unittest.main()

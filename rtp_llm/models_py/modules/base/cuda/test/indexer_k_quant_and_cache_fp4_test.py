"""FP4 indexer K quant + cache round-trip tests (Blackwell SM100+).

Verifies that our CUDA kernels
  - ``rtp_llm_ops.indexer_k_quant_and_cache_fp4``
  - ``rtp_llm_ops.cp_gather_indexer_k_quant_cache_fp4``
together produce a (k_fp4, scale) pair byte-identical to
``deep_gemm.utils.per_token_cast_to_fp4(..., use_ue8m0=True, gran_k=32,
use_packed_ue8m0=True)`` after the round-trip through a paged cache.
"""

from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops


def _is_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() >= (10, 0)


def _per_token_cast_to_fp4_ref(k_bf16: torch.Tensor):
    """deep_gemm reference: ``per_token_cast_to_fp4(use_ue8m0=True, gran_k=32,
    use_packed_ue8m0=True)``."""
    from deep_gemm.utils import per_token_cast_to_fp4

    return per_token_cast_to_fp4(
        k_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )


def _cast_back(packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    from deep_gemm.utils import cast_back_from_fp4

    return cast_back_from_fp4(packed, sf, gran_k=32, use_packed_ue8m0=True)


class IndexerKQuantAndCacheFp4Test(TestCase):
    HD = 128
    GRAN_K = 32
    BLOCK_SIZE = 64
    CACHE_STRIDE = 64 + 4  # HD/2 (FP4 data) + HD/gran_k (UE8M0 byte)

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        if not _is_blackwell():
            raise SkipTest("FP4 indexer kernels require Blackwell SM100+")
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        torch.manual_seed(0)

    # ------------------------------------------------------------------
    # quant_and_cache: exercise the write side and compare to deep_gemm ref
    # ------------------------------------------------------------------
    def test_quant_and_cache_matches_deep_gemm_reference(self):
        num_tokens = 17
        num_blocks = 4
        k = torch.randn(num_tokens, self.HD, dtype=torch.bfloat16, device=self.device)

        kv_cache = torch.zeros(
            num_blocks,
            self.BLOCK_SIZE,
            self.CACHE_STRIDE,
            dtype=torch.uint8,
            device=self.device,
        )
        # Spread tokens across two blocks at non-contiguous slots so the
        # paged-write logic is non-trivial.
        slot_mapping = torch.tensor(
            [i * 3 + 5 for i in range(num_tokens)],
            dtype=torch.int64,
            device=self.device,
        )

        rtp_llm_ops.indexer_k_quant_and_cache_fp4(
            k, kv_cache, slot_mapping, self.GRAN_K
        )
        torch.cuda.synchronize()

        # Reference: deep_gemm per-token cast.
        ref_packed, ref_sf = _per_token_cast_to_fp4_ref(k)

        # Read back what the kernel wrote at the configured slots and compare
        # byte-for-byte with the reference. The on-device layout is per-block:
        # ``BLOCK_SIZE * (HD/2)`` data bytes first, then ``BLOCK_SIZE *
        # (HD/gran_k)`` scale bytes (see mla_quant_kernel.cu:275-289). The 3D
        # view ``(num_blocks, BLOCK_SIZE, CACHE_STRIDE)`` does NOT correspond
        # to that layout, so we work on a flat byte view per block.
        scale_section_start = self.BLOCK_SIZE * (self.HD // 2)
        for tok_i in range(num_tokens):
            slot = int(slot_mapping[tok_i].item())
            blk = slot // self.BLOCK_SIZE
            off = slot % self.BLOCK_SIZE
            block_flat = kv_cache[blk].view(-1)

            data_start = off * (self.HD // 2)
            actual_data = block_flat[data_start : data_start + self.HD // 2]
            # ref_packed is int8 (m, HD/2); compare unsigned.
            expected_data = ref_packed[tok_i].view(torch.uint8)
            self.assertTrue(
                torch.equal(actual_data, expected_data),
                f"FP4 data mismatch at token {tok_i} (slot {slot})",
            )

            # Scale section: 4 UE8M0 bytes/token, packed as one int32 in ref.
            scale_off = scale_section_start + off * (self.HD // self.GRAN_K)
            actual_scale = block_flat[scale_off : scale_off + self.HD // self.GRAN_K]
            expected_scale_bytes = ref_sf[tok_i].view(torch.uint8)
            self.assertTrue(
                torch.equal(actual_scale, expected_scale_bytes),
                f"UE8M0 scale mismatch at token {tok_i} (slot {slot})",
            )

    # ------------------------------------------------------------------
    # gather: round-trip via paged cache and rebuild (k_fp4, scale)
    # ------------------------------------------------------------------
    def test_cp_gather_round_trip_matches_reference(self):
        batch_size = 3
        seq_lens = [5, 9, 7]
        total_tokens = sum(seq_lens)
        cu_seq_lens = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
            dtype=torch.int32,
            device=self.device,
        )

        max_pages_per_req = 2
        num_blocks = batch_size * max_pages_per_req
        block_table = torch.arange(
            num_blocks, dtype=torch.int32, device=self.device
        ).view(batch_size, max_pages_per_req)

        # Build slot_mapping that lays each request's tokens contiguously into
        # its first block (all seq_lens <= BLOCK_SIZE).
        slots = []
        for req_i, n in enumerate(seq_lens):
            base_block = int(block_table[req_i, 0].item())
            slots.extend(base_block * self.BLOCK_SIZE + j for j in range(n))
        slot_mapping = torch.tensor(slots, dtype=torch.int64, device=self.device)

        k = torch.randn(total_tokens, self.HD, dtype=torch.bfloat16, device=self.device)
        kv_cache = torch.zeros(
            num_blocks,
            self.BLOCK_SIZE,
            self.CACHE_STRIDE,
            dtype=torch.uint8,
            device=self.device,
        )

        rtp_llm_ops.indexer_k_quant_and_cache_fp4(
            k, kv_cache, slot_mapping, self.GRAN_K
        )

        dst_k = torch.empty(
            total_tokens, self.HD // 2, dtype=torch.uint8, device=self.device
        )
        dst_scale = torch.empty(
            total_tokens,
            self.HD // self.GRAN_K // 4,
            dtype=torch.int32,
            device=self.device,
        )
        rtp_llm_ops.cp_gather_indexer_k_quant_cache_fp4(
            kv_cache, dst_k, dst_scale, block_table, cu_seq_lens
        )
        torch.cuda.synchronize()

        ref_packed, ref_sf = _per_token_cast_to_fp4_ref(k)
        self.assertTrue(
            torch.equal(dst_k, ref_packed.view(torch.uint8)),
            "gathered FP4 data does not match deep_gemm reference",
        )
        self.assertTrue(
            torch.equal(dst_scale.view(torch.uint8), ref_sf.view(torch.uint8)),
            "gathered UE8M0 scale does not match deep_gemm reference",
        )

        # Dequantize both sides and check element-wise reconstruction error
        # stays in the FP4 quantization noise band.
        actual_recon = _cast_back(dst_k.view(torch.int8), dst_scale)
        ref_recon = _cast_back(ref_packed, ref_sf)
        self.assertTrue(
            torch.equal(actual_recon, ref_recon),
            "round-trip reconstruction differs from reference",
        )

    # ------------------------------------------------------------------
    # input-validation guards
    # ------------------------------------------------------------------
    def test_rejects_short_slot_mapping(self):
        k = torch.randn(1, self.HD, dtype=torch.bfloat16, device=self.device)
        kv_cache = torch.empty(
            1,
            self.BLOCK_SIZE,
            self.CACHE_STRIDE,
            dtype=torch.uint8,
            device=self.device,
        )
        slot_mapping = torch.empty(0, dtype=torch.int64, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "slot_mapping size"):
            rtp_llm_ops.indexer_k_quant_and_cache_fp4(
                k, kv_cache, slot_mapping, self.GRAN_K
            )

    def test_rejects_non_blackwell_layout_stride(self):
        # cache_stride below the minimum (HD/2 + HD/gran_k = 68) must be
        # rejected at the C++ entry point.
        k = torch.randn(1, self.HD, dtype=torch.bfloat16, device=self.device)
        bad_stride = (self.HD // 2 + self.HD // self.GRAN_K) - 1
        kv_cache = torch.empty(
            1,
            self.BLOCK_SIZE,
            bad_stride,
            dtype=torch.uint8,
            device=self.device,
        )
        slot_mapping = torch.tensor([0], dtype=torch.int64, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "cache_stride"):
            rtp_llm_ops.indexer_k_quant_and_cache_fp4(
                k, kv_cache, slot_mapping, self.GRAN_K
            )


if __name__ == "__main__":
    main()

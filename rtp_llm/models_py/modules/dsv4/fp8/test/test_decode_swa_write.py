"""UT for the decode SWA FP8 write dispatch.

The decode path uses the Triton ``quantize_and_insert_k_cache`` writer
directly.  This test covers the batch sizes that showed up in the
timeline copy-kernel investigation.
"""

from __future__ import annotations

import math
import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
    dequantize_and_gather_k_cache,
    dequantize_slots_to_bf16,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_kv_quant_decode_op import (
    read_model1_kv_slot_bytes,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.write_swa import decode_write_swa_fp8

HEAD_DIM = 512
NOPE_DIM = 448
ENTRY_BYTES = 584
BLOCK_SIZE = 256
FP8_MAX = 448.0
QUANT_BLOCK = 64


class DecodeSwaWriteTest(unittest.TestCase):

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(11)

    def _alloc_cache(self, max_slot: int) -> torch.Tensor:
        num_blocks = max(1, max_slot // BLOCK_SIZE + 1)
        return torch.empty(
            num_blocks,
            BLOCK_SIZE,
            ENTRY_BYTES,
            dtype=torch.uint8,
            device=self.device,
        )

    def _write(self, kv: torch.Tensor, slot_mapping: torch.Tensor):
        cache = self._alloc_cache(int(slot_mapping.clamp(min=0).max().item()))
        cache.fill_(0x5A)
        decode_write_swa_fp8(
            kv,
            slot_mapping,
            cache,
            bsz=int(kv.shape[0]),
            q_len=int(kv.shape[1]),
            head_dim=HEAD_DIM,
        )
        torch.cuda.synchronize()
        return cache

    def _dequant_slots(self, cache: torch.Tensor, max_slot: int) -> torch.Tensor:
        # Keep a gather-based helper for callers that need dense logical
        # positions. Block id 0 is invalid, so logical block 0 maps to phys 1.
        out = torch.zeros(
            1,
            max_slot + 1,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        )
        seq_lens = torch.tensor([max_slot + 1], dtype=torch.int32, device=self.device)
        logical_blocks = (max_slot // BLOCK_SIZE) + 1
        block_table = torch.ones(
            1, logical_blocks, dtype=torch.int32, device=self.device
        )
        dequantize_and_gather_k_cache(
            out=out,
            k_cache=cache,
            seq_lens=seq_lens,
            gather_lens=None,
            block_table=block_table,
            block_size=BLOCK_SIZE,
            offset=0,
        )
        return out[0]

    def _assert_nope_within_ue8m0_bound(
        self, original: torch.Tensor, recovered: torch.Tensor
    ) -> None:
        diff = (recovered[:, :NOPE_DIM].float() - original[:, :NOPE_DIM].float()).abs()
        for t in range(original.shape[0]):
            max_scale = 0.0
            for start in range(0, NOPE_DIM, QUANT_BLOCK):
                tile = original[t, start : start + QUANT_BLOCK].float()
                amax = max(tile.abs().max().item(), 1e-4)
                exponent = math.ceil(math.log2(amax / FP8_MAX))
                max_scale = max(max_scale, 2.0**exponent)
            self.assertLessEqual(diff[t].max().item(), 16.0 * max_scale)

    def _assert_valid_slots_roundtrip(
        self,
        kv: torch.Tensor,
        slot_mapping: torch.Tensor,
        cache: torch.Tensor,
    ) -> None:
        valid = slot_mapping >= 0
        valid_slots = slot_mapping[valid].long()
        expected = kv.reshape(-1, HEAD_DIM)[valid]
        actual = dequantize_slots_to_bf16(cache, valid_slots)
        self._assert_nope_within_ue8m0_bound(expected, actual)
        self.assertEqual(
            (actual[:, NOPE_DIM:] - expected[:, NOPE_DIM:]).abs().max().item(),
            0.0,
            msg="RoPE region should be BF16 passthrough after decode SWA write",
        )

    def test_triton_roundtrip_for_decode_batch_sizes(self):
        for bsz in (1, 8, 64, 256):
            with self.subTest(bsz=bsz):
                kv = (
                    torch.randn(
                        bsz,
                        1,
                        HEAD_DIM,
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                    * 0.1
                )
                slot_mapping = torch.arange(
                    bsz, dtype=torch.int64, device=self.device
                ) + BLOCK_SIZE
                if bsz >= 8:
                    slot_mapping[1] = -1
                    slot_mapping[-2] = -1

                triton = self._write(kv, slot_mapping)
                self._assert_valid_slots_roundtrip(kv, slot_mapping, triton)

                if bsz >= 8:
                    skipped_slot = BLOCK_SIZE + 1
                    skipped_bytes = read_model1_kv_slot_bytes(
                        triton,
                        block_idx=skipped_slot // BLOCK_SIZE,
                        block_offset=skipped_slot % BLOCK_SIZE,
                        block_size=BLOCK_SIZE,
                    )
                    self.assertTrue(
                        torch.all(skipped_bytes == 0x5A),
                        msg="slot_mapping=-1 should leave the skipped slot untouched",
                    )


if __name__ == "__main__":
    unittest.main()

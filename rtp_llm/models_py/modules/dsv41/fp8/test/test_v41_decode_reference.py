"""SM80 dual-pool sparse decode reference: layout, masking and sink semantics."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv41.fp8._v41_fp4_triton import (
    _dequantize_fp4_rows_torch,
    dequantize_k_cache_slots_fp4,
)
from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_triton import (
    _dequantize_rows_torch,
    _quantize_rows_torch,
    dequantize_swa_k_cache,
)
from rtp_llm.models_py.modules.dsv41.fp8.decode.fp8_sparse_attn_decode_op import (
    SparseAttnV4DecodeFp8Op,
)


class V41DecodeReferenceTest(unittest.TestCase):
    def test_fp4_row_layout_and_negative_slot(self):
        raw = torch.zeros(8, 288, dtype=torch.uint8)
        raw[:, 0] = 0x21  # even/odd E2M1 codes 1, 2 -> 0.5, 1.0
        raw[:, 1] = 0x43  # codes 3, 4 -> 1.5, 2.0
        raw[:, 256:288] = 0x38  # E4M3 scale 1.0 for every group
        decoded = _dequantize_fp4_rows_torch(raw, torch.bfloat16).float()
        torch.testing.assert_close(
            decoded[0, :4], torch.tensor([0.5, 1.0, 1.5, 2.0])
        )
        pool = raw.view(1, 8, 288)
        got = dequantize_k_cache_slots_fp4(
            pool, torch.tensor([0, 3, -1], dtype=torch.int32)
        )
        torch.testing.assert_close(got[0].float(), decoded[0], atol=1e-3, rtol=0)
        torch.testing.assert_close(
            got[2], torch.zeros(512, dtype=torch.bfloat16), atol=0, rtol=0
        )

    def test_swa_528_layout_and_negative_slot(self):
        keys = torch.randn(8, 512, dtype=torch.bfloat16)
        payload, scales = _quantize_rows_torch(keys)
        pool = torch.zeros(1, 8, 528, dtype=torch.uint8)
        pool[0, :, :512] = payload
        pool[0, :, 512:] = scales
        want = _dequantize_rows_torch(payload, scales, torch.bfloat16)
        got = dequantize_swa_k_cache(
            pool, torch.tensor([0, 2, -1], dtype=torch.int32)
        )
        torch.testing.assert_close(got[:2].float(), want[[0, 2]].float())
        torch.testing.assert_close(
            got[2], torch.zeros(512, dtype=torch.bfloat16), atol=0, rtol=0
        )

    def test_dual_pool_attention_matches_direct_softmax(self):
        torch.manual_seed(3)
        keys = torch.randn(8, 512, dtype=torch.bfloat16)
        payload, scales = _quantize_rows_torch(keys)
        swa_pool = torch.zeros(1, 8, 528, dtype=torch.uint8)
        swa_pool[0, :, :512] = payload
        swa_pool[0, :, 512:] = scales
        fp4_raw = torch.zeros(8, 288, dtype=torch.uint8)
        fp4_raw[:, 0] = 0x21
        fp4_raw[:, 256:288] = 0x38
        fp4_pool = fp4_raw.view(1, 8, 288)
        swa_rows = _dequantize_rows_torch(payload, scales, torch.bfloat16)
        fp4_rows = _dequantize_fp4_rows_torch(fp4_raw, torch.bfloat16)

        q = torch.randn(1, 1, 2, 512, dtype=torch.bfloat16)
        sink = torch.randn(2)
        op = SparseAttnV4DecodeFp8Op(
            n_heads=2, head_dim=512, softmax_scale=512 ** -0.5
        )
        out, lse = op._forward_reference(
            q,
            swa_pool,
            sink,
            torch.tensor([[[0, 1, -1, 3]]], dtype=torch.int32),
            torch.tensor([3], dtype=torch.int32),
            fp4_pool,
            torch.tensor([[[0, 1, 2]]], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
            True,
        )
        selected = torch.cat(
            (swa_rows[[0, 1]].float(), fp4_rows[[0, 1]].float()), dim=0
        )
        logits = q[0, 0].float() @ selected.T * (512 ** -0.5)
        peak = torch.maximum(logits.amax(-1), sink)
        weights = torch.exp(logits - peak[:, None])
        want = (weights @ selected) / (
            weights.sum(-1) + torch.exp(sink - peak)
        )[:, None]
        torch.testing.assert_close(out[0, 0].float(), want, atol=2e-2, rtol=2e-2)
        self.assertEqual(lse.shape, (1, 2, 1))


if __name__ == "__main__":
    unittest.main()

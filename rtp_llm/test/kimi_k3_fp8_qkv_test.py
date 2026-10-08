import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_fp8_kernels import (
    gather_fp8_prefix,
    quantize_fp8,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_qkv_fp8_quant import (
    quantize_qkv_fp8,
)


class KimiK3Fp8QkvTest(unittest.TestCase):
    def test_strided_qkv_quantizes_to_ordinary_e4m3(self):
        if not torch.cuda.is_available():
            self.fail("K3 FP8 QKV test requires a GPU")
        torch.manual_seed(53)
        q = torch.randn(7, 12, 192, device="cuda", dtype=torch.bfloat16) * 0.4
        k = torch.randn(9, 12, 192, device="cuda", dtype=torch.bfloat16) * 0.4
        v = (torch.randn(9, 12, 256, device="cuda", dtype=torch.bfloat16) * 0.4)[:, :, 128:]
        self.assertFalse(v.is_contiguous())
        for original, quantized in zip((q, k, v), quantize_qkv_fp8(q, k, v)):
            self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
            self.assertTrue(quantized.is_contiguous())
            torch.testing.assert_close(quantized.float(), original.to(torch.float8_e4m3fn).float(), rtol=0, atol=0)

    def test_decode_quantizer_reuses_supplied_buffer(self):
        if not torch.cuda.is_available():
            self.fail("K3 FP8 query test requires a GPU")
        query = torch.arange(96, device="cuda", dtype=torch.bfloat16).reshape(2, 3, 16) / 10
        buffer = torch.empty_like(query, dtype=torch.float8_e4m3fn)
        result = quantize_fp8(query, 1.0, buffer)
        self.assertEqual(result.data_ptr(), buffer.data_ptr())
        torch.testing.assert_close(result.float(), query.to(torch.float8_e4m3fn).float(), rtol=0, atol=0)

    def test_prefix_gather_reads_plain_e4m3_pages_and_new_bf16_rows(self):
        if not torch.cuda.is_available():
            self.fail("K3 FP8 prefix test requires a GPU")
        device = "cuda"
        page_size = 4
        cache = torch.zeros((2, page_size, 576), dtype=torch.float8_e4m3fn, device=device)
        cached_rows = torch.arange(3 * 576, dtype=torch.float32, device=device).reshape(3, 576) / 100
        cache[0, :2] = cached_rows[:2].to(torch.float8_e4m3fn)
        cache[1, 0] = cached_rows[2].to(torch.float8_e4m3fn)
        ckv = torch.arange(3 * 512, dtype=torch.float32, device=device).reshape(3, 512).to(torch.bfloat16) / 50
        rope = torch.arange(3 * 64, dtype=torch.float32, device=device).reshape(3, 64).to(torch.bfloat16) / 25
        out_ckv = torch.empty((6, 512), dtype=torch.bfloat16, device=device)
        out_rope = torch.empty((6, 64), dtype=torch.bfloat16, device=device)
        gather_fp8_prefix(
            out_ckv, out_rope, ckv, rope, cache,
            torch.tensor([0, 1], dtype=torch.int32, device=device),
            torch.tensor([[0, 2, 0, 1], [1, 1, 1, 1]], dtype=torch.int32, device=device),
            torch.tensor([0, 2, 3], dtype=torch.int32, device=device),
            page_size, scale=1.0,
        )
        expected = torch.cat((
            cache[0, :2].to(torch.bfloat16),
            torch.cat((ckv[:2], rope[:2]), dim=1),
            cache[1, :1].to(torch.bfloat16),
            torch.cat((ckv[2:], rope[2:]), dim=1),
        ))
        torch.testing.assert_close(out_ckv, expected[:, :512], rtol=0, atol=0)
        torch.testing.assert_close(out_rope, expected[:, 512:], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

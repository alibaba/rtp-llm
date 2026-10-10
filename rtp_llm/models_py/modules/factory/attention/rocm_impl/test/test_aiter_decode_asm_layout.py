"""Guard the zero-copy KV view required by AITER's ASM PA interface."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import aiter
import torch

from rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter import (
    AiterDecodeAttnOpAsm,
)


class AiterDecodeAsmLayoutTest(unittest.TestCase):
    def test_key_page_dimension_and_storage(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float8_e4m3fnuz):
            for packed in (False, True):
                with self.subTest(dtype=dtype, packed=packed):
                    self._check_key_view(dtype, packed)

    def _check_key_view(self, dtype, packed):
        op = AiterDecodeAttnOpAsm.__new__(AiterDecodeAttnOpAsm)
        op.head_num_kv = 4
        op.head_dim = 128
        op.tokens_per_block = 16
        op.enable_cuda_graph = False

        if packed:
            raw = torch.empty((3, 2 * 4 * 16 * 128 + 32), dtype=dtype)
            cache = raw[:, : 2 * 4 * 16 * 128].reshape(3, 2, 4, 16, 128)
            cache_base = raw
        else:
            cache = torch.empty((3, 2, 4, 16, 128), dtype=dtype)
            cache_base = cache
        raw_key = cache.select(1, 0)
        kv_cache = SimpleNamespace(
            kv_cache_base=cache_base,
            kv_scale_base=torch.ones((3, 2, 4)),
        )
        params = SimpleNamespace(
            seq_lens=torch.full((2,), 16, dtype=torch.int32),
            kv_cache_block_id_device=torch.zeros((2, 1), dtype=torch.int32),
        )
        query = torch.empty((2, 16, 128), dtype=torch.bfloat16)

        def check_asm_call(q, k, v, *args):
            vector_width = 16 // dtype.itemsize
            self.assertEqual(k.shape, (3, 4, 128 // vector_width, 16, vector_width))
            self.assertEqual(k.data_ptr(), raw_key.data_ptr())
            self.assertEqual(k.stride()[:2], raw_key.stride()[:2])
            self.assertEqual(v.data_ptr(), cache.select(1, 1).data_ptr())
            return args[6]

        with patch.object(aiter, "pa_fwd_asm", side_effect=check_asm_call):
            self.assertEqual(op.forward(query, kv_cache, params).shape, (2, 2048))

    @unittest.skipUnless(torch.cuda.is_available(), "requires a ROCm GPU")
    def test_asm_kernel_reads_16_token_page(self):
        cases = (
            (torch.bfloat16, torch.bfloat16, 4),
            (torch.bfloat16, torch.float8_e4m3fnuz, 4),
            (torch.bfloat16, torch.bfloat16, 5),
            (torch.bfloat16, torch.bfloat16, 8),
            (torch.bfloat16, torch.float8_e4m3fnuz, 8),
            (torch.float16, torch.float16, 5),
        )
        for query_dtype, kv_dtype, gqa in cases:
            with self.subTest(query_dtype=query_dtype, kv_dtype=kv_dtype, gqa=gqa):
                op = AiterDecodeAttnOpAsm.__new__(AiterDecodeAttnOpAsm)
                op.head_num_kv = 4
                op.head_dim = 128
                op.tokens_per_block = 16
                op.enable_cuda_graph = False

                cache = torch.empty((1, 2, 4, 16, 128), dtype=kv_dtype, device="cuda")
                cache[:, 0].zero_()
                cache[:, 1].fill_(1)
                scale = (
                    torch.ones((1, 2, 4, 16), dtype=torch.float32, device="cuda")
                    if kv_dtype == torch.float8_e4m3fnuz
                    else None
                )
                kv_cache = SimpleNamespace(kv_cache_base=cache, kv_scale_base=scale)
                params = SimpleNamespace(
                    seq_lens=torch.tensor([16], dtype=torch.int32, device="cuda"),
                    kv_cache_block_id_device=torch.zeros(
                        (1, 1), dtype=torch.int32, device="cuda"
                    ),
                )
                query = torch.ones(
                    (1, 4 * gqa, 128), dtype=query_dtype, device="cuda"
                )

                actual = op.forward(query, kv_cache, params)
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    actual, torch.ones_like(actual), rtol=0, atol=0.01
                )


if __name__ == "__main__":
    unittest.main()

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
    MlaKVCacheWriteOp,
)
from rtp_llm.ops import KvCacheDataType


class KimiK3Fp8CacheWriteTest(unittest.TestCase):
    def test_ordinary_e4m3_write_and_padding_slot(self):
        if not torch.cuda.is_available():
            self.fail("K3 FP8 cache write test requires a GPU")
        device = "cuda"
        latent = torch.linspace(-1, 1, 3 * 512, dtype=torch.bfloat16, device=device).reshape(3, 512)
        suffix = torch.linspace(-0.5, 0.5, 3 * 64, dtype=torch.bfloat16, device=device).reshape(3, 64)
        cache = torch.zeros((1, 128, 576), dtype=torch.float8_e4m3fn, device=device)
        slots = torch.tensor([-1, 7, 8], dtype=torch.long, device=device)
        op = MlaKVCacheWriteOp(KvCacheDataType.FP8, fp8_compute=True, kv_scale=1.0)
        self.assertEqual(op.kv_cache_type, "fp8")
        op.forward(latent, suffix, SimpleNamespace(kv_cache_base=cache),
                   SimpleNamespace(slot_mapping=slots))
        torch.cuda.synchronize()
        self.assertEqual(float(cache[0, 0].float().abs().max()), 0.0)
        for source, slot in ((1, 7), (2, 8)):
            expected = torch.cat((latent[source], suffix[source])).to(torch.float8_e4m3fn)
            torch.testing.assert_close(cache[0, slot].float(), expected.float(), rtol=0, atol=0)

    def test_fp8_compute_rejects_bf16_cache(self):
        with self.assertRaisesRegex(ValueError, "requires ordinary FP8 cache"):
            MlaKVCacheWriteOp(KvCacheDataType.BASE, fp8_compute=True)


if __name__ == "__main__":
    unittest.main()

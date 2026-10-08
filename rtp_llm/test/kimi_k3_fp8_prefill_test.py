import unittest

import torch

from rtp_llm.models_py.modules.kimi_k3.native_mla_prefill import KimiK3TokenspeedPrefill


class KimiK3Fp8PrefillTest(unittest.TestCase):
    def test_tokenspeed_consumes_e4m3_qkv_and_returns_bf16(self):
        if not torch.cuda.is_available():
            self.fail("K3 FP8 Prefill test requires a GPU")
        torch.manual_seed(61)
        q = (torch.randn(4, 12, 192, device="cuda", dtype=torch.bfloat16) * 0.2).to(torch.float8_e4m3fn)
        k = (torch.randn(4, 12, 192, device="cuda", dtype=torch.bfloat16) * 0.2).to(torch.float8_e4m3fn)
        v = (torch.randn(4, 12, 128, device="cuda", dtype=torch.bfloat16) * 0.2).to(torch.float8_e4m3fn)
        offsets = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
        op = KimiK3TokenspeedPrefill(fp8_compute=True)
        op.plan(offsets, offsets, 12, 12, 192, 128,
                sm_scale=192 ** -0.5, causal=True,
                q_data_type=torch.float8_e4m3fn,
                kv_data_type=torch.float8_e4m3fn)
        actual = op.run(q, k, v)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(tuple(actual.shape), (4, 12, 128))
        self.assertTrue(torch.isfinite(actual.float()).all().item())
        scores = torch.einsum("thd,shd->hts", q.float(), k.float()) * (192 ** -0.5)
        scores.masked_fill_(torch.ones(4, 4, device="cuda", dtype=torch.bool).triu(1), float("-inf"))
        expected = torch.einsum("hts,shd->thd", scores.softmax(-1), v.float())
        torch.testing.assert_close(actual.float(), expected, rtol=0.2, atol=0.05)


if __name__ == "__main__":
    unittest.main()

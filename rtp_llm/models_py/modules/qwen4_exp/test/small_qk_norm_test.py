import unittest
from unittest.mock import patch

import torch
from rtp_llm.models_py.modules.qwen4_exp.norm import Qwen4ExpFusedQKRMSNorm


class SmallQKNormTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(617)
        self.q = (torch.randn(256, device="cuda") * 0.1).bfloat16()
        self.k = (torch.randn(256, device="cuda") * 0.1).bfloat16()
        self.model = Qwen4ExpFusedQKRMSNorm(self.q, self.k, 3, 1, 256)

    def run_norm(self, value, enabled):
        with patch.dict(
            "os.environ",
            {
                "RTP_LLM_QWEN4_SMALL_QK_NORM": str(int(enabled)),
                "RTP_LLM_QWEN4_FUSED_QK_NORM": "0",
            },
        ):
            return self.model(value.clone())

    def test_prime_shapes_and_unchanged_v(self):
        for rows in (1, 3, 4, 7, 8, 17, 31, 32, 33, 4093):
            x = torch.randn(rows, 1280, device="cuda", dtype=torch.bfloat16)
            expected, actual = self.run_norm(x, False), self.run_norm(x, True)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(actual[:, 1024:], x[:, 1024:]))
            if rows > 32:
                self.assertTrue(torch.equal(actual, expected))

    def test_graph_reads_fresh_input_and_checkpoint_gains(self):
        x = torch.randn(17, 1280, device="cuda", dtype=torch.bfloat16)
        with patch.dict(
            "os.environ",
            {"RTP_LLM_QWEN4_SMALL_QK_NORM": "1", "RTP_LLM_QWEN4_FUSED_QK_NORM": "0"},
        ):
            self.model(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.model(x)
            for _ in range(3):
                source = torch.randn_like(x)
                self.q.normal_(0, 0.1)
                self.k.normal_(0, 0.1)
                expected = self.run_norm(source, False)
                x.copy_(source)
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(x, expected))
                self.assertTrue(torch.equal(x[:, 1024:], source[:, 1024:]))

    def test_unsupported_gamma_and_dtype_use_original(self):
        self.model.q_weight = torch.randn(512, device="cuda", dtype=torch.bfloat16)[::2]
        for dtype in (torch.bfloat16, torch.float32):
            x = torch.randn(7, 1280, device="cuda", dtype=dtype)
            self.assertTrue(
                torch.equal(self.run_norm(x, False), self.run_norm(x, True))
            )

    def test_fp32_checkpoint_gains(self):
        self.model.q_weight = self.q.float()
        self.model.k_weight = self.k.float()
        for rows in (1, 4, 7, 8, 17, 28, 32):
            value = torch.randn(rows, 1280, device="cuda", dtype=torch.bfloat16)
            self.assertTrue(
                torch.equal(self.run_norm(value, False), self.run_norm(value, True))
            )

    def test_fp32_plus_one(self):
        self.q.fill_(0.003)
        self.k.fill_(0.003)
        x = torch.randn(3, 1280, device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(
            self.run_norm(x, True).float(),
            self.run_norm(x, False).float(),
            atol=0.016,
            rtol=0.01,
        )


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_desc.qwen4_exp_mtp import Qwen4ExpMTPInputProjection
from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear

MODULE = "rtp_llm.models_py.kernels.cuda.qwen4_mtp_norm"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class MtpEmbeddingNormTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 pinned Torch reduction")
        torch.manual_seed(112201)

    def fixture(self):
        def projection():
            weight = (torch.randn(2560, 2560, device="cuda") * 0.1).to(
                torch.float8_e4m3fn
            )
            return Fp8Linear(weight, torch.tensor([0.119713], device="cuda"))

        return Qwen4ExpMTPInputProjection(
            torch.randn(2560, device="cuda", dtype=torch.bfloat16) * 0.07,
            torch.randn(10240, device="cuda", dtype=torch.bfloat16) * 0.07,
            projection(),
            projection(),
            hidden_size=2560,
            norm_eps=1e-6,
        )

    def test_complete_two_projection_consumer_and_live_graph(self):
        for rows in (1, 4, 32, 4096):
            model = self.fixture()
            embedding = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
            hidden = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
            for _ in range(3):
                model(embedding, hidden)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = model(embedding, hidden)
            for gain in (0.01, 0.2, 2.0):
                embedding.normal_()
                hidden.normal_()
                model.embedding_gamma.normal_(0, gain)
                graph.replay()
                with patch(MODULE + ".try_mtp_embedding_rms_norm", return_value=None):
                    expected = model(embedding, hidden)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            graph.reset()


if __name__ == "__main__":
    unittest.main()

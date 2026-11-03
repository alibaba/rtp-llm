import unittest

import torch
from torch import nn

from rtp_llm.models_py.kernels.cuda.mtp_frontend import native, try_mtp_frontend
from rtp_llm.models_py.modules import RMSNorm


class Hy4MtpFrontendTest(unittest.TestCase):
    def test_hy4_norm_epsilon_and_position_mask_match_during_graph_replay(self):
        self.assertTrue(torch.cuda.is_available())
        library = native()
        self.assertIsNotNone(
            library, "MTP frontend extension must compile for this GPU test"
        )
        torch.manual_seed(238)
        width, batch = 6144, 17
        embedding = nn.Embedding.from_pretrained(
            torch.randn(32, width, dtype=torch.bfloat16, device="cuda")
        )
        embedding.tp_size = 1
        e_norm = RMSNorm(
            torch.randn(width, dtype=torch.bfloat16, device="cuda"), eps=1e-6
        )
        h_norm = RMSNorm(
            torch.randn(width, dtype=torch.bfloat16, device="cuda"), eps=1e-6
        )
        ids = torch.arange(batch, dtype=torch.int32, device="cuda")
        positions = torch.arange(batch, dtype=torch.int32, device="cuda")
        hidden = torch.randn(batch, width, dtype=torch.bfloat16, device="cuda")

        def reference():
            embedded = embedding(ids).masked_fill(positions.reshape(-1, 1) == 0, 0)
            return torch.cat(
                (
                    e_norm(embedded),
                    h_norm(hidden),
                ),
                dim=-1,
            )

        def fused():
            result = try_mtp_frontend(
                library, embedding, hidden, e_norm, h_norm, ids, positions
            )
            self.assertIsNotNone(
                result, "HY4 H6144 BF16 inputs must use the fused path"
            )
            return result

        for _ in range(3):
            output, residual = fused()
        torch.testing.assert_close(output, reference(), rtol=0, atol=0)
        torch.testing.assert_close(residual, torch.zeros_like(hidden), rtol=0, atol=0)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output, graph_residual = fused()

        for seed in (12, 48):
            torch.manual_seed(seed)
            ids.copy_(torch.randint(32, (batch,), dtype=torch.int32, device="cuda"))
            positions.copy_(
                torch.randint(3, (batch,), dtype=torch.int32, device="cuda")
            )
            hidden.copy_(torch.randn_like(hidden))
            graph.replay()
            torch.testing.assert_close(graph_output, reference(), rtol=0, atol=0)
            torch.testing.assert_close(
                graph_residual, torch.zeros_like(hidden), rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()

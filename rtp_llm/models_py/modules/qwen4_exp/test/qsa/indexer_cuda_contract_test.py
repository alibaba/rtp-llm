import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import (
    build_qsa_rope,
    build_qsa_rope_from_logical_positions,
)


class IndexerCudaContractTest(unittest.TestCase):
    def test_generated_text_positions_capture_and_replay(self):
        self.assertTrue(torch.cuda.is_available(), "CUDA replay target requires a GPU")
        device = torch.device("cuda")
        logical = torch.tensor([2, 5], dtype=torch.int32, device=device)
        configs = (
            SimpleNamespace(
                style="Mrope",
                index_factor=3,
                dim=8,
                mrope_dim1=2,
                mrope_dim2=1,
                mrope_dim3=1,
                mrope_interleaved=True,
                base=10000,
                scale=2.0,
            ),
            SimpleNamespace(style="Base", index_factor=3, dim=8, base=10000, scale=2.0),
        )
        for config in configs:
            with self.subTest(style=config.style):
                generated = logical[:, None].expand(-1, 3).reshape(-1)
                expected = build_qsa_rope(
                    generated,
                    config,
                    token_count=2,
                    dtype=torch.float32,
                    device=device,
                    logical_positions=logical,
                )
                build_qsa_rope_from_logical_positions(
                    logical, config, token_count=2, dtype=torch.float32, device=device
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    cosine, sine = build_qsa_rope_from_logical_positions(
                        logical,
                        config,
                        token_count=2,
                        dtype=torch.float32,
                        device=device,
                    )
                graph.replay()
                torch.testing.assert_close(cosine, expected[0])
                torch.testing.assert_close(sine, expected[1])
                logical.add_(7)
                graph.replay()
                expected = build_qsa_rope(
                    logical[:, None].expand(-1, 3).reshape(-1),
                    config,
                    token_count=2,
                    dtype=torch.float32,
                    device=device,
                    logical_positions=logical,
                )
                torch.testing.assert_close(cosine, expected[0])
                torch.testing.assert_close(sine, expected[1])
                logical.sub_(7)


if __name__ == "__main__":
    unittest.main()

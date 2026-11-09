import os
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.kernels.cuda.hy4_ihc_ops import can_fuse_head
from rtp_llm.models_py.modules.hy_v4.ihc import Hy4IHCHead
from rtp_llm.models_py.modules.hy_v4.ihc_triton import maybe_fused_ihc_head
from rtp_llm.utils.model_weight import W


class Hy4IhcAotTest(unittest.TestCase):
    def test_production_dispatch_and_graph_replay(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.get_device_capability(), (10, 3))
        torch.manual_seed(20261108)
        with torch.inference_mode():
            for hidden in (4096, 6144):
                weights = {
                    W.hy4_ihc_head_fn: torch.randn(4, 4 * hidden, device="cuda")
                    / (4 * hidden) ** 0.5,
                    W.hy4_ihc_head_scale: torch.tensor([0.2], device="cuda"),
                    W.hy4_ihc_head_base: torch.randn(4, device="cuda"),
                }
                head = Hy4IHCHead(weights, hidden, 4, 1e-6, 1e-5, chunk_size=32)
                for tokens in (1, 16, 64):
                    with self.subTest(hidden=hidden, tokens=tokens):
                        x = torch.randn(
                            tokens, 4, hidden, device="cuda", dtype=torch.bfloat16
                        )
                        with mock.patch.dict(os.environ, {"RTP_LLM_HY4_IHC_HPC": "0"}):
                            self.assertFalse(
                                can_fuse_head(x, head.fn_weight, head.scale, head.base)
                            )
                            triton = head(x)
                        with mock.patch.dict(os.environ, {"RTP_LLM_HY4_IHC_HPC": "1"}):
                            self.assertTrue(
                                can_fuse_head(x, head.fn_weight, head.scale, head.base)
                            )
                            with mock.patch(
                                "rtp_llm.models_py.modules.hy_v4.ihc.maybe_fused_ihc_head",
                                side_effect=AssertionError(
                                    "unexpected Triton fallback"
                                ),
                            ):
                                actual = head(x)
                                torch.testing.assert_close(
                                    actual, triton, atol=0.02, rtol=0.02
                                )
                                for _ in range(3):
                                    head(x)
                                torch.cuda.synchronize()
                                graph = torch.cuda.CUDAGraph()
                                with torch.cuda.graph(graph):
                                    replay_out = head(x)
                                x.normal_()
                                head.fn_weight.normal_(std=(4 * hidden) ** -0.5)
                                head.scale.fill_(0.35)
                                head.base.normal_()
                                expected = head(x)
                                graph.replay()
                                torch.cuda.synchronize()
                                torch.testing.assert_close(
                                    replay_out, expected, atol=0, rtol=0
                                )
                                with mock.patch.dict(
                                    os.environ, {"RTP_LLM_HY4_IHC_HPC": "0"}
                                ):
                                    with mock.patch(
                                        "rtp_llm.models_py.modules.hy_v4.ihc.maybe_fused_ihc_head",
                                        wraps=maybe_fused_ihc_head,
                                    ):
                                        reference = head(x)
                                torch.testing.assert_close(
                                    replay_out, reference, atol=0.02, rtol=0.02
                                )

    def test_unsupported_shape_returns_to_triton(self):
        with torch.inference_mode(), mock.patch.dict(
            os.environ, {"RTP_LLM_HY4_IHC_HPC": "1"}
        ):
            hidden = 64
            weights = {
                W.hy4_ihc_head_fn: torch.randn(4, 4 * hidden, device="cuda"),
                W.hy4_ihc_head_scale: torch.tensor([0.2], device="cuda"),
                W.hy4_ihc_head_base: torch.randn(4, device="cuda"),
            }
            head = Hy4IHCHead(weights, hidden, 4, 1e-6, 1e-5)
            x = torch.randn(3, 4, hidden, device="cuda", dtype=torch.bfloat16)
            self.assertFalse(can_fuse_head(x, head.fn_weight, head.scale, head.base))
            with mock.patch(
                "rtp_llm.models_py.modules.hy_v4.ihc.maybe_fused_ihc_head",
                wraps=maybe_fused_ihc_head,
            ) as fallback:
                actual = head(x)
                fallback.assert_called_once()
                self.assertTrue(torch.isfinite(actual).all())


if __name__ == "__main__":
    unittest.main()

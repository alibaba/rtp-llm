import unittest

import torch
import torch.nn.functional as F

_HC = 4


_HIDDEN = 16


_LOWRANK = 6


_EPS = 1e-6


class GatedResidualTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_merged_down_epilogue_preserves_bf16_rounding_and_graph_replay(self):
        from rtp_llm.models_py.triton_kernels.gated_residual.down import (
            down_inject_epilogue,
        )

        source = torch.randn(17, 336, device="cuda", dtype=torch.bfloat16) * 4

        def reference():
            return (
                F.silu(source[:, :320] / 4),
                2 * torch.sigmoid(source[:, 320:324] / 4),
            )

        expected = reference()
        actual = down_inject_epilogue(source, 320, 4)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = down_inject_epilogue(source, 320, 4)
        source.copy_(
            torch.linspace(-128, 128, source.numel(), device="cuda").reshape_as(source)
        )
        graph.replay()
        for got, want in zip(actual, reference()):
            torch.testing.assert_close(got, want, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_fused_mix_reduce_matches_torch_and_replays(self):
        from rtp_llm.models_py.triton_kernels.gated_residual.mix import fused_mix_reduce

        width = 4 * 2560
        # Cover the default tile and both SM120 prefill tile configurations.
        for rows in (1, 17, 4093, 4096, 12288):
            with self.subTest(rows=rows):
                logits = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
                normed = torch.randn_like(logits)

                def reference():
                    return (
                        torch.sigmoid(logits).reshape(rows, 4, 2560)
                        * normed.reshape(rows, 4, 2560)
                    ).mean(1)

                actual = fused_mix_reduce(logits, normed, 4)
                torch.testing.assert_close(
                    actual.float(), reference().float(), atol=0.016, rtol=0.01
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replayed = fused_mix_reduce(logits, normed, 4)
                logits.normal_()
                normed.normal_()
                graph.replay()
                torch.testing.assert_close(
                    replayed.float(), reference().float(), atol=0.016, rtol=0.01
                )
                graph.reset()


if __name__ == "__main__":
    unittest.main()

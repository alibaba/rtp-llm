import itertools
import unittest

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.utils.arch import is_sm120


class Sm120Fp8QuantInvarianceTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or not is_sm120():
            self.skipTest("requires SM120")

    def test_explicit_v2_across_batch_threshold(self):
        generator = torch.Generator(device="cuda").manual_seed(42)
        layouts = (
            (False, False, False),
            (True, False, False),
            (True, True, False),
            (True, True, True),
        )
        for width, group, layout, dtype in itertools.product(
            (1024, 4096),
            (16, 32, 64, 128),
            layouts,
            (torch.float16, torch.bfloat16, torch.float32),
        ):
            values = torch.randn(
                (11, width), generator=generator, device="cuda", dtype=dtype
            )
            values[0].zero_()
            options = dict(
                group_size=group,
                eps=1e-4,
                column_major_scales=layout[0],
                scale_tma_aligned=layout[1],
                scale_ue8m0=layout[2],
                enable_v2=True,
            )
            threshold = 4 * 1024 * 1024 // width
            expected_q, expected_s = sgl_per_token_group_quant_fp8(values, **options)
            for rows in (1, 17, threshold - 1, threshold, threshold + 1):
                with self.subTest(
                    width=width, group=group, layout=layout, rows=rows, dtype=dtype
                ):
                    order = torch.arange(rows, device="cuda") % len(values)
                    x = values[order].contiguous()
                    actual_q, actual_s = sgl_per_token_group_quant_fp8(x, **options)
                    self.assertTrue(
                        torch.equal(
                            actual_q.view(torch.uint8),
                            expected_q.view(torch.uint8)[order],
                        )
                    )
                    self.assertTrue(torch.equal(actual_s, expected_s[order]))

    def test_explicit_v2_graph_replay(self):
        generator = torch.Generator(device="cuda").manual_seed(43)
        values = torch.randn(
            (17, 4096), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        options = dict(
            group_size=128, column_major_scales=True, eps=1e-4, enable_v2=True
        )
        expected_q, expected_s = sgl_per_token_group_quant_fp8(values, **options)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual_q, actual_s = sgl_per_token_group_quant_fp8(values, **options)
        for _ in range(20):
            graph.replay()
            self.assertTrue(
                torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
            )
            self.assertTrue(torch.equal(actual_s, expected_s))


if __name__ == "__main__":
    unittest.main()

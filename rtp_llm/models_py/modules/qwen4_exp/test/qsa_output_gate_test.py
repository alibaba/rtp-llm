import unittest

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_paged_fmha import (
    sparse_paged_gqa_attn,
)
from rtp_llm.models_py.modules.qwen4_exp.test import grouped_fp8_test as grouped_fixture


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QsaOutputGateTest(unittest.TestCase):
    def fixture(self, batch, queries):
        q, cache, scale, table, lengths, selected = (
            grouped_fixture.GroupedFp8Test().fixture(12, 1, queries)
        )
        q, table, lengths, selected = [
            t[:batch].contiguous() for t in (q, table, lengths, selected)
        ]
        gate = torch.randn(batch, queries, 12, 256, device="cuda", dtype=torch.bfloat16)
        return q, cache, scale, table, lengths, selected, gate

    def run_core(self, tensors, gated, graph=False):
        q, cache, scale, table, lengths, selected, gate = tensors
        return sparse_paged_gqa_attn(
            q,
            cache,
            table,
            lengths,
            selected,
            page_size=128,
            kv_head_num=1,
            kv_scale=scale,
            graph_capture=graph,
            output_gate=gate if gated else None,
        )

    def test_exact_gate_rounding_and_owner_layout(self):
        for batch in (1, 4):
            for queries in (1, 4):
                tensors = self.fixture(batch, queries)
                gate = tensors[-1]
                gate.flatten()[:3] = torch.tensor(
                    [float("-inf"), float("inf"), float("nan")],
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                expected = self.run_core(tensors, False).transpose(
                    1, 2
                ) * torch.sigmoid(gate)
                actual = self.run_core(tensors, True).transpose(1, 2)
                torch.testing.assert_close(
                    actual, expected, rtol=0, atol=0, equal_nan=True
                )
                self.assertTrue(actual.is_contiguous())

    def test_graph_refreshes_gates_and_fp32_page_scales(self):
        tensors = self.fixture(4, 4)
        for _ in range(3):
            self.run_core(tensors, True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = self.run_core(tensors, True, graph=True)
        for seed in (7370, 7371, 7372):
            torch.manual_seed(seed)
            tensors[-1].normal_()
            tensors[2].uniform_(0.2, 1)
            expected = (
                self.run_core(tensors, False).transpose(1, 2)
                * torch.sigmoid(tensors[-1])
            ).transpose(1, 2)
            graph.replay()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

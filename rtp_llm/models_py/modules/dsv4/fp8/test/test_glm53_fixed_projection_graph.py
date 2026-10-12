"""Fixed-row graph must preserve native FP32 GEMM shapes and owned outputs."""

import os
import unittest
from unittest import mock

import torch
from rtp_llm.models_py.modules.dsv4.fp8.indexer import _linear_fixed_rows
from rtp_llm.models_py.modules.factory.linear.fixed_m_linear import fixed_m_linear
from rtp_llm.models_py.modules.glm53_fixed_projection_graph import (
    fixed_projection_graph,
    projection_graph_stats,
)
from rtp_llm.models_py.modules.glm53_router import Glm53FP32Router


class FixedProjectionGraphTest(unittest.TestCase):
    def test_exact_indexer_router_and_changed_weights(self):
        self.assertTrue(torch.cuda.is_available(), "GPU validation must not skip")
        torch.manual_seed(531013)
        torch.backends.cuda.matmul.allow_tf32 = False
        for outputs, hidden in ((32, 4096), (256, 4096), (7, 129)):
            w = torch.randn(outputs, hidden, device="cuda") * 0.01
            for n in (1, 17, 509, 512, 513, 1601, 7711, 16383):
                x = torch.randn(n, hidden, device="cuda", dtype=torch.bfloat16)
                with mock.patch.dict(os.environ, {"GLM53_FIXED_PROJECTION_GRAPH": "0"}):
                    expected = _linear_fixed_rows(x, w, 512)
                with mock.patch.dict(os.environ, {"GLM53_FIXED_PROJECTION_GRAPH": "1"}):
                    actual = _linear_fixed_rows(x, w, 512)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    frozen = actual.clone()
                    w.mul_(0.99)
                    changed = fixed_projection_graph(x, w, 512)
                    self.assertIsNotNone(changed)
                    with mock.patch.dict(
                        os.environ, {"GLM53_FIXED_PROJECTION_GRAPH": "0"}
                    ):
                        new_expected = _linear_fixed_rows(x, w, 512)
                    torch.testing.assert_close(changed, new_expected, rtol=0, atol=0)
                    torch.testing.assert_close(actual, frozen, rtol=0, atol=0)
        router = Glm53FP32Router(torch.randn(4096, 256, device="cuda") * 0.01)
        x = torch.randn(13931, 4096, device="cuda", dtype=torch.bfloat16)
        with mock.patch.dict(os.environ, {"GLM53_FIXED_PROJECTION_GRAPH": "0"}):
            old = fixed_m_linear(router, x, 512)
        with mock.patch.dict(os.environ, {"GLM53_FIXED_PROJECTION_GRAPH": "1"}):
            new = fixed_m_linear(router, x, 512)
        torch.testing.assert_close(new, old, rtol=0, atol=0)
        print("Projection graph stats:", projection_graph_stats())


if __name__ == "__main__":
    unittest.main()

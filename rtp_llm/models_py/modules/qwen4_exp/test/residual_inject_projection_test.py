import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.test import residual_norm_fp8_test as fixture
from rtp_llm.models_py.triton_kernels.fla.utils import assert_close


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class InjectProjectionContractTest(unittest.TestCase):
    def module(self):
        return fixture.ResidualNormFp8Test().module(True)

    def reference_gate(self, module, x):
        normalized = module._norm(x)
        projected = torch.nn.functional.linear(
            normalized.double(), module.inject_projection.compute_weight.double()
        ).to(torch.bfloat16)
        return 2 * torch.sigmoid(projected / 4)

    def test_complete_consumer_and_independent_fp64_gate(self):
        module = self.module()
        self.assertFalse(hasattr(module, "inject_compute_padded"))
        for rows in (1, 4, 8, 16, 32, 4096, 12288):
            torch.manual_seed(8120 + rows)
            x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
            with patch(
                "rtp_llm.models_py.triton_kernels.common.gated_residual_splitk_inject.try_splitk_down_inject",
                return_value=None,
            ):
                reference = module(x)
            actual = module(x)
            native_gate = 2 * torch.sigmoid(
                torch.nn.functional.linear(
                    module._norm(x), module.inject_projection.compute_weight
                )
                / 4
            )
            torch.testing.assert_close(actual[2], native_gate, rtol=0, atol=0)
            torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=0)
            self.assertIs(actual[1], x)
            # Preserve the independent FP64 quality check in addition to
            # strict native split-K parity.
            assert_close(
                "HC independent FP64 inject gate",
                self.reference_gate(module, x),
                actual[2],
                0.005,
            )

    def test_graph_refreshes_norm_input_and_live_bf16_weight_alias(self):
        module = self.module()
        x = torch.randn(16, 10240, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            module(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = module(x)
        for seed in (8121, 8122):
            torch.manual_seed(seed)
            x.normal_()
            module.inject_projection.compute_weight.normal_(0, 0.01)
            reference = self.reference_gate(module, x)
            native_gate = 2 * torch.sigmoid(
                torch.nn.functional.linear(
                    module._norm(x), module.inject_projection.compute_weight
                )
                / 4
            )
            graph.replay()
            torch.testing.assert_close(actual[2], native_gate, rtol=0, atol=0)
            assert_close("HC live Graph inject gate", reference, actual[2], 0.005)


if __name__ == "__main__":
    unittest.main()

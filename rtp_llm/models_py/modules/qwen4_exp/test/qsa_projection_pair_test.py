import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpAttention
from rtp_llm.models_py.modules.qwen4_exp.fp8_dense import (
    Qwen4Fp8Linear,
    project_fp8_pair,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import Qwen4ExpQSAIndexer
from rtp_llm.models_py.modules.qwen4_exp.qsa.norm import exact_head_rms_norm


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QsaProjectionPairTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 complete-consumer domain")
        torch.manual_seed(107201)

    def fixture(self):
        def weight(shape):
            return (torch.randn(shape, device="cuda") * 0.1).to(torch.float8_e4m3fn)

        gate = Qwen4Fp8Linear(
            weight((3072, 2560)), torch.tensor([0.119713], device="cuda")
        )
        indexer = Qwen4ExpQSAIndexer(
            weight((1152, 2560)),
            torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.07,
            torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.07,
            n_heads=8,
            kv_heads=1,
            head_dim=128,
            token_budget=2048,
            compress_ratio=4,
            norm_eps=1e-6,
            projection_scale=torch.tensor([0.03125], device="cuda"),
        )
        return gate, indexer

    def test_independent_weight_scales_and_original_gemm_shapes(self):
        gate, indexer = self.fixture()
        for rows in (1, 3, 4, 8, 16, 20, 24, 28, 32, 4093, 4096, 12288, 32768):
            x = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
            expected_gate = gate(x)
            expected_q, expected_k = indexer.project_raw(x)
            with patch.object(torch, "_scaled_mm", wraps=torch._scaled_mm) as gemm:
                actual_gate, projected = project_fp8_pair(
                    x, gate, indexer.qk_projection
                )
                self.assertEqual(gemm.call_count, 2)
                self.assertEqual(
                    [tuple(call.args[1].shape) for call in gemm.call_args_list],
                    [(2560, 3072), (2560, 1152)],
                )
            actual_q, actual_k = indexer.split_projected(projected)
            for actual, expected in (
                (actual_gate, expected_gate),
                (actual_q, expected_q),
                (actual_k, expected_k),
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_graph_reads_live_input_raw_gamma_and_separate_weight_scales(self):
        for rows in (4, 16, 4096):
            gate, indexer = self.fixture()
            x = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)

            def run():
                g, qk = project_fp8_pair(x, gate, indexer.qk_projection)
                q, k = indexer.split_projected(qk)
                return (
                    g,
                    exact_head_rms_norm(q, indexer.q_norm_gamma, indexer.norm_eps),
                    k,
                )

            for _ in range(3):
                run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = run()
            for gain in (0.01, 0.2, 2.0):
                x.normal_(0, gain)
                indexer.q_norm_gamma.normal_(0, gain)
                gate.scale.fill_(gain)
                indexer.qk_projection.scale.fill_(gain * 0.23)
                graph.replay()
                q, k = indexer.project(x)
                expected = gate(x), q, k
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            graph.reset()

    def test_overrides_hooks_bias_and_float_inputs_keep_original_dispatch(self):
        gate, indexer = self.fixture()
        other = indexer.qk_projection
        x = torch.randn(4, 2560, device="cuda", dtype=torch.bfloat16)
        self.assertIsNone(project_fp8_pair(x.float(), gate, other))
        with patch.object(gate, "forward", wraps=gate.forward):
            self.assertIsNone(project_fp8_pair(x, gate, other))
        handle = other.register_forward_hook(lambda module, args, result: result)
        try:
            self.assertIsNone(project_fp8_pair(x, gate, other))
        finally:
            handle.remove()
        gate.bias = torch.zeros(3072, device="cuda", dtype=torch.bfloat16)
        self.assertIsNone(project_fp8_pair(x, gate, other))

    def test_owner_preserves_raw_target_and_normalized_decode_contracts(self):
        gate, indexer = self.fixture()
        attention = Qwen4ExpAttention.__new__(Qwen4ExpAttention)
        torch.nn.Module.__init__(attention)
        attention.qsa_indexer = indexer
        attention.qsa_rope_config = SimpleNamespace()
        x = torch.randn(4, 2560, device="cuda", dtype=torch.bfloat16)
        _, projected = project_fp8_pair(x, gate, indexer.qk_projection)
        qk = indexer.split_projected(projected)
        for target in (False, True):

            def capture(q, k, **kwargs):
                return q, k, kwargs.get("raw_query", False)

            runtime = SimpleNamespace(
                main_inputs=SimpleNamespace(is_target_verify=target, is_prefill=False),
                select_target_verify_tokens=capture,
                select_decode_tokens=capture,
            )
            expected = attention._qsa_selected_indices(x, runtime)
            actual = attention._qsa_selected_indices(x, runtime, projected=qk)
            for a, b in zip(actual[:2], expected[:2]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(actual[2], expected[2])


if __name__ == "__main__":
    unittest.main()

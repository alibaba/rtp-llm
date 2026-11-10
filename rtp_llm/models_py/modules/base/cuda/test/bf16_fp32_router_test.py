import json
import os
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.base.cuda.bf16_fp32_router import (
    bf16_fp32_router_logits,
    prepare_bf16_fp32_router_weight,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


class Bf16FP32RouterTest(TestCase):
    def setUp(self) -> None:
        self.assertTrue(torch.cuda.is_available(), "Requires an NVIDIA GPU")
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32, "Requires IEEE FP32")
        env = patch.dict(os.environ, {"RTP_LLM_HY4_ROUTER_BF16_FP32": "1"})
        env.start()
        self.addCleanup(env.stop)
        torch.manual_seed(173)

    def weight(self) -> torch.Tensor:
        return torch.randn(6144, 256, device="cuda", dtype=torch.bfloat16).float()

    def test_default_on_and_explicit_disable(self) -> None:
        weight = self.weight()
        with patch.dict(
            os.environ,
            {
                "RTP_LLM_HY4_FP32_ROUTER_REDUCE": "1",
                "RTP_LLM_HY4_FP32_ROUTER_LAYOUT": "1",
            },
        ):
            os.environ.pop("RTP_LLM_HY4_ROUTER_BF16_FP32")
            self.assertIsNotNone(prepare_bf16_fp32_router_weight(weight))
            os.environ["RTP_LLM_HY4_ROUTER_BF16_FP32"] = "0"
            self.assertIsNone(prepare_bf16_fp32_router_weight(weight))

    def test_packing_preserves_values_and_sign_zero(self) -> None:
        weight = self.weight()
        weight[0, :2] = torch.tensor([0.0, -0.0], device="cuda")
        for source in (weight, weight.bfloat16()):
            packed = prepare_bf16_fp32_router_weight(source)
            assert packed is not None
            self.assertEqual(packed.dtype, torch.bfloat16)
            self.assertEqual(tuple(packed.shape), (256, 6144))
            self.assertTrue(packed.is_contiguous())
            self.assertTrue(torch.equal(source, packed.t()))
            self.assertTrue(torch.equal(source.signbit(), packed.t().signbit()))

    def test_unrepresentable_and_dense_near_tie_weights_fall_back(self) -> None:
        hidden = torch.randn(16, 6144, device="cuda", dtype=torch.bfloat16)
        sparse = self.weight()
        sparse[0, 0] = 1 + 2**-20
        dense = torch.randn(6144, 1, device="cuda").expand(-1, 256).clone()
        dense[0] += torch.arange(256, device="cuda") * 2**-12
        for weight in (sparse, dense):
            packed = prepare_bf16_fp32_router_weight(weight)
            self.assertIsNone(packed)
            expected = hidden.float() @ weight
            actual = bf16_fp32_router_logits(hidden, weight, packed)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertTrue(
                torch.equal(actual.topk(8).indices, expected.topk(8).indices)
            )

    def test_invalid_preparation_and_nonfinite_values_fall_back(self) -> None:
        weight = self.weight()
        for unsupported in (
            weight.cpu(),
            weight.half(),
            weight.t(),
            weight.detach().requires_grad_(),
        ):
            self.assertIsNone(prepare_bf16_fp32_router_weight(unsupported))
        for value in (float("nan"), float("inf"), -float("inf")):
            source = weight.clone()
            source[0, 0] = value
            self.assertIsNone(prepare_bf16_fp32_router_weight(source))

    def test_native_output_matches_direct_op_for_measured_rows(self) -> None:
        weight = self.weight()
        packed = prepare_bf16_fp32_router_weight(weight)
        assert packed is not None
        observed = []
        for rows in (1, 16, 32, 48, 64, 128):
            hidden = torch.randn(rows, 6144, device="cuda", dtype=torch.bfloat16)
            actual = bf16_fp32_router_logits(hidden, weight, packed)
            expected = rtp_llm_ops.cublas_gemm_bf16_bf16_fp32(hidden, packed)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertTrue(torch.isfinite(actual).all().item())
            observed.append({"rows": rows, "direct_native_exact": True})
        output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "bf16_fp32_router_helper.json").write_text(
            json.dumps({"status": "HELPER_CORRECTNESS_ONLY", "cases": observed})
        )

    def test_unsupported_inputs_keep_fp32_math(self) -> None:
        weight = self.weight()
        packed = prepare_bf16_fp32_router_weight(weight)
        assert packed is not None
        for hidden in (
            torch.randn(17, 6144, device="cuda"),
            torch.randn(17, 6144, device="cuda", dtype=torch.float16),
            torch.randn(17, 12288, device="cuda", dtype=torch.bfloat16)[:, ::2],
            torch.randn(129, 6144, device="cuda", dtype=torch.bfloat16),
            torch.empty(0, 6144, device="cuda", dtype=torch.bfloat16),
        ):
            expected = hidden.float() @ weight
            actual = bf16_fp32_router_logits(hidden, weight, packed)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        hidden = torch.randn(17, 6144)
        actual = bf16_fp32_router_logits(hidden, weight.cpu(), packed)
        torch.testing.assert_close(actual, hidden @ weight.cpu(), atol=0, rtol=0)

    def test_native_weight_storage_fallback_and_gradients(self) -> None:
        weight = self.weight().bfloat16()
        packed = prepare_bf16_fp32_router_weight(weight)
        assert packed is not None
        hidden = torch.randn(3, 6144, device="cuda", dtype=torch.bfloat16)
        hidden.requires_grad_()
        actual = bf16_fp32_router_logits(hidden, weight, packed)
        expected = hidden.float() @ weight.float()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual_grad = torch.autograd.grad(actual.sum(), hidden)[0]
        expected_grad = torch.autograd.grad(expected.sum(), hidden)[0]
        torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)

    def test_replacement_weight_requires_new_preparation(self) -> None:
        first = self.weight()
        replacement = -first
        packed = prepare_bf16_fp32_router_weight(replacement)
        assert packed is not None
        hidden = torch.randn(32, 6144, device="cuda", dtype=torch.bfloat16)
        actual = bf16_fp32_router_logits(hidden, replacement, packed)
        expected = rtp_llm_ops.cublas_gemm_bf16_bf16_fp32(hidden, packed)
        original = prepare_bf16_fp32_router_weight(first)
        assert original is not None
        self.assertFalse(
            torch.equal(actual, bf16_fp32_router_logits(hidden, first, original))
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_graph_replay_uses_changed_input_on_nondefault_stream(self) -> None:
        weight = self.weight()
        packed = prepare_bf16_fp32_router_weight(weight)
        assert packed is not None
        hidden = torch.randn(32, 6144, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                bf16_fp32_router_logits(hidden, weight, packed)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                actual = bf16_fp32_router_logits(hidden, weight, packed)
            previous = actual.clone()
            hidden.neg_()
            graph.replay()
            expected = rtp_llm_ops.cublas_gemm_bf16_bf16_fp32(hidden, packed)
        torch.cuda.current_stream().wait_stream(stream)
        self.assertFalse(torch.equal(actual, previous))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    main()

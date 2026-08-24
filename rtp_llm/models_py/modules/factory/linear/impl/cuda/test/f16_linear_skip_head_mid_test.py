import builtins
import importlib
import os
import sys
import unittest
from unittest import mock

_LOCAL_DEEP_GEMM_PATH = os.environ.get("RTP_LOCAL_DEEP_GEMM_PATH")
if _LOCAL_DEEP_GEMM_PATH:
    sys.path.insert(0, _LOCAL_DEEP_GEMM_PATH)

import torch

from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear


class CudaF16LinearSkipHeadMidContractTest(unittest.TestCase):
    def test_cpu_forward_import_does_not_require_deep_gemm_wrapper(self) -> None:
        module_name = "rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear"
        parent_name = "rtp_llm.models_py.modules.factory.linear.impl.cuda"
        wrapper_name = "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper"
        original_module = sys.modules.pop(module_name)
        parent_module = sys.modules[parent_name]
        original_parent_attribute = parent_module.f16_linear
        original_import = builtins.__import__

        def reject_wrapper_import(name: str, *args: object, **kwargs: object) -> object:
            if name == wrapper_name:
                raise ModuleNotFoundError(
                    "DeepGEMM wrapper must not be imported by CPU F16 Linear"
                )
            return original_import(name, *args, **kwargs)

        try:
            with mock.patch("builtins.__import__", side_effect=reject_wrapper_import):
                module = importlib.import_module(module_name)
                checkpoint_weight = torch.randn((8, 16), dtype=torch.bfloat16)
                inputs = torch.randn((4, 8), dtype=torch.bfloat16)
                linear = module.CudaF16Linear(checkpoint_weight)
                actual = linear(inputs)
            expected = torch.nn.functional.linear(inputs, checkpoint_weight.T)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            sys.modules[module_name] = original_module
            parent_module.f16_linear = original_parent_attribute

    def test_invalid_split_is_rejected_before_deep_gemm_dispatch(self) -> None:
        inputs = torch.empty((1, 512), dtype=torch.bfloat16)
        checkpoint_weight = torch.empty((512, 12 * 256), dtype=torch.bfloat16)
        linear = CudaF16Linear(checkpoint_weight)

        with self.assertRaises(ValueError):
            linear.forward_skip_head_mid(inputs, (96, 64, 160))


class CudaF16LinearSkipHeadMidTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("bf16_gemm_nt_skip_head_mid requires SM100")
        import deep_gemm

        self.assertTrue(callable(deep_gemm.bf16_gemm_nt_skip_head_mid))
        if _LOCAL_DEEP_GEMM_PATH:
            self.assertTrue(deep_gemm.__file__.startswith(_LOCAL_DEEP_GEMM_PATH))

    def test_k3_projection_layout_and_values(self) -> None:
        torch.manual_seed(123)
        tokens = 257
        heads = 12
        k_dim = 512
        k_nope_dim = 128
        k_pe_dim = 64
        v_dim = 128
        logical_head_dim = k_nope_dim + v_dim
        physical_head_dim = k_nope_dim + k_pe_dim + v_dim

        inputs = torch.randn((tokens, k_dim), device="cuda", dtype=torch.bfloat16)
        checkpoint_weight = torch.randn(
            (k_dim, heads * logical_head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        linear = CudaF16Linear(checkpoint_weight)

        actual = linear.forward_skip_head_mid(
            inputs, (k_nope_dim, k_pe_dim, v_dim)
        ).view(tokens, heads, physical_head_dim)
        expected = linear(inputs).view(tokens, heads, logical_head_dim)

        torch.testing.assert_close(
            actual[..., :k_nope_dim],
            expected[..., :k_nope_dim],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual[..., k_nope_dim + k_pe_dim :],
            expected[..., k_nope_dim:],
            rtol=0,
            atol=0,
        )
        self.assertTrue(actual.is_contiguous())
        self.assertEqual(linear.weight.stride(), (1, heads * logical_head_dim))

    def test_k3_projection_reuses_preallocated_output(self) -> None:
        torch.manual_seed(456)
        tokens = 257
        heads = 12
        k_dim = 512
        head_splits = (128, 64, 128)
        physical_features = heads * sum(head_splits)
        inputs = torch.randn((tokens, k_dim), device="cuda", dtype=torch.bfloat16)
        checkpoint_weight = torch.randn(
            (k_dim, heads * (head_splits[0] + head_splits[2])),
            device="cuda",
            dtype=torch.bfloat16,
        )
        linear = CudaF16Linear(checkpoint_weight)
        self.assertTrue(linear.supports_skip_head_mid(inputs, head_splits))
        output = torch.full(
            (tokens, physical_features),
            7.0,
            device="cuda",
            dtype=torch.bfloat16,
        )

        actual = linear.forward_skip_head_mid(
            inputs,
            head_splits,
            output=output,
        )

        self.assertIs(actual, output)
        actual_heads = actual.view(tokens, heads, sum(head_splits))
        expected = linear(inputs).view(tokens, heads, -1)
        torch.testing.assert_close(
            actual_heads[..., : head_splits[0]],
            expected[..., : head_splits[0]],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual_heads[..., -head_splits[2] :],
            expected[..., head_splits[0] :],
            rtol=0,
            atol=0,
        )
        self.assertTrue(
            torch.all(
                actual_heads[..., head_splits[0] : head_splits[0] + head_splits[1]]
                == 7.0
            )
        )

    def test_k3_projection_uses_caller_output_and_preserves_gap(self) -> None:
        torch.manual_seed(124)
        tokens = 129
        heads = 12
        k_dim = 512
        head_splits = (128, 64, 128)
        physical_head_dim = sum(head_splits)
        logical_head_dim = head_splits[0] + head_splits[2]
        inputs = torch.randn((tokens, k_dim), device="cuda", dtype=torch.bfloat16)
        checkpoint_weight = torch.randn(
            (k_dim, heads * logical_head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        linear = CudaF16Linear(checkpoint_weight)
        sentinel = torch.tensor(3.25, device="cuda", dtype=torch.bfloat16)
        output_storage = torch.full(
            (tokens + 2, heads * physical_head_dim),
            sentinel,
            device="cuda",
            dtype=torch.bfloat16,
        )
        output = output_storage[1:-1]
        self.assertTrue(output.is_contiguous())
        self.assertGreater(output.storage_offset(), 0)

        returned = linear.forward_skip_head_mid_out(inputs, output, head_splits).view(
            tokens, heads, physical_head_dim
        )
        expected = linear(inputs).view(tokens, heads, logical_head_dim)

        self.assertEqual(returned.data_ptr(), output.data_ptr())
        self.assertEqual(
            returned.untyped_storage().data_ptr(),
            output.untyped_storage().data_ptr(),
        )
        torch.testing.assert_close(
            returned[..., : head_splits[0]],
            expected[..., : head_splits[0]],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            returned[..., head_splits[0] + head_splits[1] :],
            expected[..., head_splits[0] :],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            returned[..., head_splits[0] : head_splits[0] + head_splits[1]],
            sentinel.expand(tokens, heads, head_splits[1]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            output_storage[[0, -1]],
            sentinel.expand(2, heads * physical_head_dim),
            rtol=0,
            atol=0,
        )

    def test_zero_tokens_supports_preallocated_and_allocated_output(self) -> None:
        heads = 12
        k_dim = 512
        head_splits = (128, 64, 128)
        physical_features = heads * sum(head_splits)
        inputs = torch.empty((0, k_dim), device="cuda", dtype=torch.bfloat16)
        checkpoint_weight = torch.empty(
            (k_dim, heads * (head_splits[0] + head_splits[2])),
            device="cuda",
            dtype=torch.bfloat16,
        )
        linear = CudaF16Linear(checkpoint_weight)
        output = torch.empty(
            (0, physical_features), device="cuda", dtype=torch.bfloat16
        )
        self.assertEqual(inputs.untyped_storage().data_ptr(), 0)
        self.assertEqual(output.untyped_storage().data_ptr(), 0)
        self.assertTrue(linear.supports_skip_head_mid(inputs, head_splits))

        preallocated = linear.forward_skip_head_mid(
            inputs,
            head_splits,
            output=output,
        )
        allocated = linear.forward_skip_head_mid(inputs, head_splits)

        self.assertIs(preallocated, output)
        self.assertEqual(tuple(preallocated.shape), (0, physical_features))
        self.assertEqual(tuple(allocated.shape), (0, physical_features))
        self.assertEqual(preallocated.numel(), 0)
        self.assertEqual(allocated.numel(), 0)

    def test_preallocated_output_requires_16_byte_alignment(self) -> None:
        tokens = 257
        heads = 12
        k_dim = 512
        head_splits = (128, 64, 128)
        output_numel = tokens * heads * sum(head_splits)
        inputs = torch.empty((tokens, k_dim), device="cuda", dtype=torch.bfloat16)
        checkpoint_weight = torch.empty(
            (k_dim, heads * (head_splits[0] + head_splits[2])),
            device="cuda",
            dtype=torch.bfloat16,
        )
        linear = CudaF16Linear(checkpoint_weight)
        backing = torch.empty(output_numel + 1, device="cuda", dtype=torch.bfloat16)
        output = backing.narrow(0, 1, output_numel).view(
            tokens, heads * sum(head_splits)
        )
        self.assertTrue(output.is_contiguous())
        self.assertNotEqual(output.data_ptr() % 16, 0)

        with self.assertRaisesRegex(RuntimeError, "16-byte aligned"):
            linear.forward_skip_head_mid(inputs, head_splits, output=output)

    def test_preallocated_output_must_not_share_input_storage(self) -> None:
        tokens = 257
        heads = 12
        k_dim = 512
        head_splits = (128, 64, 128)
        input_numel = tokens * k_dim
        output_numel = tokens * heads * sum(head_splits)
        backing = torch.empty(
            max(input_numel, output_numel),
            device="cuda",
            dtype=torch.bfloat16,
        )
        inputs = backing.narrow(0, 0, input_numel).view(tokens, k_dim)
        output = backing.narrow(0, 0, output_numel).view(
            tokens, heads * sum(head_splits)
        )
        checkpoint_weight = torch.empty(
            (k_dim, heads * (head_splits[0] + head_splits[2])),
            device="cuda",
            dtype=torch.bfloat16,
        )
        linear = CudaF16Linear(checkpoint_weight)

        with self.assertRaisesRegex(RuntimeError, "storage independent"):
            linear.forward_skip_head_mid(inputs, head_splits, output=output)

    def test_preallocated_output_must_not_share_weight_storage(self) -> None:
        tokens = 257
        heads = 12
        k_dim = 512
        head_splits = (128, 64, 128)
        output_numel = tokens * heads * sum(head_splits)
        inputs = torch.empty((tokens, k_dim), device="cuda", dtype=torch.bfloat16)
        checkpoint_weight = torch.empty(
            (k_dim, heads * (head_splits[0] + head_splits[2])),
            device="cuda",
            dtype=torch.bfloat16,
        )
        linear = CudaF16Linear(checkpoint_weight)
        output = (
            checkpoint_weight.view(-1)
            .narrow(0, 0, output_numel)
            .view(tokens, heads * sum(head_splits))
        )

        with self.assertRaisesRegex(RuntimeError, "storage independent"):
            linear.forward_skip_head_mid(inputs, head_splits, output=output)

    def test_pitched_input_is_rejected_before_deep_gemm_dispatch(self) -> None:
        tokens = 17
        heads = 12
        k_dim = 512
        head_splits = (128, 64, 128)
        input_backing = torch.empty(
            (tokens, k_dim + 1), device="cuda", dtype=torch.bfloat16
        )
        inputs = input_backing[:, :k_dim]
        checkpoint_weight = torch.empty(
            (k_dim, heads * (head_splits[0] + head_splits[2])),
            device="cuda",
            dtype=torch.bfloat16,
        )
        linear = CudaF16Linear(checkpoint_weight)
        self.assertEqual(inputs.stride(), (k_dim + 1, 1))
        self.assertEqual(inputs.data_ptr() % 16, 0)

        self.assertFalse(linear.supports_skip_head_mid(inputs, head_splits))
        with self.assertRaisesRegex(RuntimeError, "canonical"):
            linear.forward_skip_head_mid(inputs, head_splits)

    def test_pitched_checkpoint_weight_is_rejected_before_dispatch(self) -> None:
        tokens = 17
        heads = 12
        k_dim = 512
        head_splits = (128, 64, 128)
        output_features = heads * (head_splits[0] + head_splits[2])
        inputs = torch.empty((tokens, k_dim), device="cuda", dtype=torch.bfloat16)
        weight_backing = torch.empty(
            (k_dim, output_features + 1),
            device="cuda",
            dtype=torch.bfloat16,
        )
        checkpoint_weight = weight_backing[:, :output_features]
        linear = CudaF16Linear(checkpoint_weight)
        self.assertEqual(linear.weight.stride(), (1, output_features + 1))
        self.assertEqual(linear.weight.data_ptr() % 16, 0)

        self.assertFalse(linear.supports_skip_head_mid(inputs, head_splits))
        with self.assertRaisesRegex(RuntimeError, "canonical"):
            linear.forward_skip_head_mid(inputs, head_splits)


if __name__ == "__main__":
    unittest.main()

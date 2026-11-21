"""Integer ABI / backend-preflight tests; accelerator execution is opt-in."""

import ast
import inspect
import os
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.quant_methods import groupwise_runtime as runtime


def words(values, bits):
    """Independent scalar checkpoint serializer, not the runtime repacker."""
    packed = []
    for row in values.tolist():
        output = []
        for start in range(0, len(row), 32 // bits):
            value = sum(
                int(item) << (bits * offset)
                for offset, item in enumerate(row[start : start + 32 // bits])
            )
            output.append(value if value < 2**31 else value - 2**32)
        packed.append(output)
    return torch.tensor(packed, dtype=torch.int32)


def checkpoint(method="gptq", bits=4, maximum_zero=False, expert=0, k=32, n=16):
    group = 8
    maximum = 2**bits - 1
    values = ((torch.arange(k * n).reshape(k, n) * 7 + expert * 3) % (maximum + 1)).to(
        torch.int32
    )
    stored_zero = torch.full(
        (k // group, n), maximum if maximum_zero else 2 + expert, dtype=torch.int32
    )
    scales = torch.arange(1, k // group * n + 1).reshape(k // group, n).float() / 1024
    if method == "awq":
        order = [0, 2, 4, 6, 1, 3, 5, 7]
        qweight = words(values.reshape(k, -1, 8)[:, :, order].reshape(k, n), bits)
        qzeros = words(
            stored_zero.reshape(k // group, -1, 8)[:, :, order].reshape(k // group, n),
            bits,
        )
        zero = stored_zero
    else:
        qweight = words(values.T, bits).T.contiguous()
        qzeros = words(stored_zero, bits)
        zero = stored_zero + 1
    layout = dict(
        method=method, bits=bits, group_size=group, input_size=k, output_size=n
    )
    return qweight, qzeros, scales, layout, values, zero


def decode_native(state):
    if state.runtime_bits == 4:
        weight = torch.stack((state.weight & 15, state.weight >> 4), dim=-1).flatten(-2)
        zero = torch.stack((state.zero & 15, state.zero >> 4), dim=2).flatten(1, 2)
    else:
        weight, zero = state.weight, state.zero
    return weight, zero


class GroupwiseRuntimeCPUTest(unittest.TestCase):
    def tearDown(self):
        runtime._load_backend.cache_clear()

    def test_real_integer_checkpoint_repacking_awq_gptq_4_8(self):
        for method in ("awq", "gptq"):
            for bits in (4, 8):
                with self.subTest(method=method, bits=bits):
                    q, z, s, layout, expected, zero = checkpoint(method, bits)
                    state = runtime.prepare_groupwise_state(q, z, s, layout)
                    weight, actual_zero = decode_native(state)
                    torch.testing.assert_close(
                        weight[0].int(), expected.T, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        actual_zero[0].int(), zero.T, rtol=0, atol=0
                    )
                    torch.testing.assert_close(state.scale[0], s.T, rtol=0, atol=0)
                    self.assertEqual(state.runtime_bits, bits)
                    self.assertIsNone(state.promotion_reason)

    def test_maximum_gptq_zero_is_never_wrapped_or_clipped(self):
        for bits in (4, 8):
            q, z, s, layout, expected, zero = checkpoint("gptq", bits, True)
            state = runtime.prepare_groupwise_state(q, z, s, layout)
            weight, actual_zero = decode_native(state)
            self.assertEqual(state.runtime_bits, 8)
            self.assertEqual(state.zero.dtype, torch.int16)
            torch.testing.assert_close(weight[0].int(), expected.T, rtol=0, atol=0)
            torch.testing.assert_close(actual_zero[0].int(), zero.T, rtol=0, atol=0)
            self.assertEqual(actual_zero.max().item(), 2**bits)
            if bits == 4:
                self.assertEqual(
                    state.metadata["promotion_reason"], "effective_zero_exceeds_uint4"
                )
                self.assertEqual(state.weight.numel(), q.numel() * 8)

    def test_bounded_unpack_crosses_chunk_boundary_without_reordering(self):
        for method in ("awq", "gptq"):
            for bits in (4, 8):
                q, z, s, layout, expected, zero = checkpoint(method, bits, k=160, n=24)
                state = runtime.prepare_groupwise_state(q, z, s, layout)
                weight, actual_zero = decode_native(state)
                torch.testing.assert_close(weight[0].int(), expected.T, rtol=0, atol=0)
                torch.testing.assert_close(actual_zero[0].int(), zero.T, rtol=0, atol=0)

    def test_stacked_experts_preserve_order_and_individual_scales(self):
        fixtures = [checkpoint("awq", 4, expert=i) for i in range(3)]
        state = runtime.prepare_groupwise_state(
            *(torch.stack([f[i] for f in fixtures]) for i in range(3)), fixtures[0][3]
        )
        weight, zero = decode_native(state)
        for i, fixture in enumerate(fixtures):
            torch.testing.assert_close(weight[i].int(), fixture[4].T, rtol=0, atol=0)
            torch.testing.assert_close(zero[i].int(), fixture[5].T, rtol=0, atol=0)

    def test_native_state_reuses_only_module_parameter_storage(self):
        q, z, s, layout, _, _ = checkpoint()
        layer = torch.nn.Module()
        for name, value in (("qweight", q), ("qzeros", z), ("scales", s)):
            layer.register_parameter(
                name, torch.nn.Parameter(value, requires_grad=False)
            )
        identities = {name: id(p) for name, p in layer.named_parameters()}
        state = runtime.prepare_groupwise_state(q, z, s, layout)
        installed = runtime._install_state(
            layer, ("qweight", "qzeros", "scales"), state
        )
        self.assertEqual(
            identities, {name: id(p) for name, p in layer.named_parameters()}
        )
        self.assertIs(installed.weight, layer.qweight)
        self.assertIs(installed.scale, layer.scales)
        self.assertIs(installed.zero, layer.qzeros)
        self.assertEqual(dict(layer.named_buffers()), {})

    def test_cpu_finalization_remains_unfinalized_checkpoint_staging(self):
        q, z, s, layout, _, _ = checkpoint()
        layer = torch.nn.Module()
        layer.qweight = torch.nn.Parameter(q, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(z, requires_grad=False)
        layer.scales = torch.nn.Parameter(s, requires_grad=False)
        layer._groupwise_layout = layout
        runtime.finalize_linear(layer)
        self.assertFalse(hasattr(layer, "_groupwise_runtime"))
        with self.assertRaisesRegex(RuntimeError, "staging"):
            runtime.validate_runtime_device("cpu", "gptq", 4)
        with self.assertRaisesRegex(RuntimeError, "finalized"):
            runtime.linear(layer, torch.ones(1, 32))

    def test_wrong_layout_and_scales_are_rejected(self):
        q, z, s, layout, _, _ = checkpoint()
        cases = [
            (q.float(), z, s, layout),
            (q, z[:, :1], s, layout),
            (q, z, s * 0, layout),
            (q, z, s * float("nan"), layout),
            (q, z, s, dict(layout, input_size=33)),
        ]
        for values in cases:
            with self.subTest(shape=tuple(values[0].shape)), self.assertRaises(
                (ValueError, TypeError)
            ):
                runtime.prepare_groupwise_state(*values)

    def test_cuda_nvfp4_does_not_masquerade_as_integer_groupwise(self):
        with mock.patch.object(torch.version, "hip", None):
            with self.assertRaisesRegex(RuntimeError, "NVFP4 Marlin.*not equivalent"):
                runtime.validate_runtime_device("cuda", "awq", 4)

    def test_backend_import_and_symbol_failures_are_explicit(self):
        with mock.patch.object(
            runtime.importlib, "import_module", side_effect=ImportError("missing ABI")
        ):
            with self.assertRaisesRegex(RuntimeError, "importable Aiter"):
                runtime._load_backend()
        with mock.patch.object(
            runtime.importlib, "import_module", return_value=object()
        ):
            with self.assertRaisesRegex(RuntimeError, "integer ABI"):
                runtime._load_backend()

    def test_no_torch_float_matmul_execution_path(self):
        tree = ast.parse(inspect.getsource(runtime))
        forbidden = {"matmul", "mm", "bmm", "einsum", "dequantize"}
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in forbidden
                for node in ast.walk(tree)
            )
        )


@unittest.skipUnless(
    os.environ.get("RUN_GROUPWISE_NATIVE_GPU_TESTS") == "1",
    "requires explicitly allocated ROCm GPU",
)
class GroupwiseNativeGPUTest(unittest.TestCase):
    def test_native_dense_and_promoted_zero(self):
        runtime.validate_runtime_device("cuda", "gptq", 4)
        for method in ("awq", "gptq"):
            for maximum_zero in (False, True):
                q, z, s, layout, values, zero = checkpoint(
                    method, maximum_zero=maximum_zero
                )
                layer = torch.nn.Module()
                for name, value in (
                    ("qweight", q),
                    ("qzeros", z),
                    ("scales", s.half()),
                ):
                    layer.register_parameter(
                        name, torch.nn.Parameter(value.cuda(), requires_grad=False)
                    )
                layer._groupwise_layout = layout
                runtime.finalize_linear(layer)
                pointer = layer.qweight.data_ptr()
                runtime.finalize_linear(layer)
                self.assertEqual(pointer, layer.qweight.data_ptr())
                x = torch.randn(17, 32, dtype=torch.float16, device="cuda") / 8
                expected_w = (
                    values.float() - zero.repeat_interleave(8, 0)
                ) * s.half().float().repeat_interleave(8, 0)
                expected = x @ expected_w.cuda().half()
                actual = runtime.linear(layer, x)
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)

    def test_native_routed_projection_with_inactive_slots(self):
        runtime.validate_runtime_device("cuda", "gptq", 4)
        fixtures = [checkpoint(expert=i) for i in range(2)]
        state = runtime.prepare_groupwise_state(
            *(torch.stack([f[i] for f in fixtures]).cuda() for i in range(3)),
            fixtures[0][3]
        )
        x = torch.randn(3, 32, dtype=torch.bfloat16, device="cuda") / 8
        ids = torch.tensor([[0, 1], [1, -1], [-1, 0]], dtype=torch.int32, device="cuda")
        weights = torch.tensor([[0.25, 0.75], [1, 0], [0, 1]], device="cuda")
        result = runtime._projection(
            state,
            x,
            ids,
            weights,
            *runtime._route(ids, 2),
            input_top_k=2,
            multiply_weights=True
        )
        expected = torch.zeros_like(result)
        for token, routes in enumerate(ids.cpu().tolist()):
            for slot, expert in enumerate(routes):
                if expert >= 0:
                    _, _, s, _, q, z = fixtures[expert]
                    w = (
                        (
                            (q.float() - z.repeat_interleave(8, 0))
                            * s.repeat_interleave(8, 0)
                        )
                        .cuda()
                        .bfloat16()
                    )
                    expected[token, slot] = (x[token] @ w) * weights[token, slot]
        torch.testing.assert_close(result, expected, rtol=0.03, atol=0.003)


if __name__ == "__main__":
    unittest.main()

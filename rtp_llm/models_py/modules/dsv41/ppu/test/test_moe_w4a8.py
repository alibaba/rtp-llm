"""CPU contracts for split W4A8; device execution requires explicit opt-in.

CPU kernels below stand in for communication/permutation and native GEMM.
They check the production call boundary against independent dense FP4 math;
they do not establish PPU kernel accuracy or distributed execution evidence.
Set RTP_V41_TEST_W4A8_DEVICE=1 to additionally exercise the real PPU kernels.
"""

import math
import os
import sys
import types
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41.ppu import moe as moe_module
from rtp_llm.models_py.modules.dsv41.ppu.moe_precision import (
    MOE_PRECISION_ENV,
    moe_kernel_name,
    moe_precision,
)


def _runner(precision="w4a8", hidden=128, inter=128):
    with mock.patch.dict(os.environ, {MOE_PRECISION_ENV: precision}):
        return moe_module.PPUMoE(
            num_routed_experts=2,
            num_local_experts=2,
            ep_size=1,
            hidden_size=hidden,
            inter_dim=inter,
        )


def _expert_pairs(seed=741, hidden=128, inter=128):
    generator = torch.Generator().manual_seed(seed)
    w13 = torch.randint(
        0, 256, (2, 2 * inter, hidden // 2), generator=generator, dtype=torch.uint8
    )
    w2 = torch.randint(
        0, 256, (2, hidden, inter // 2), generator=generator, dtype=torch.uint8
    )
    # Both matrices require nontrivial, different exponent compensation.
    s13 = torch.randint(
        109, 114, (2, 2 * inter, hidden // 32), generator=generator, dtype=torch.uint8
    )
    s2 = torch.randint(
        141, 146, (2, hidden, inter // 32), generator=generator, dtype=torch.uint8
    )
    s13.flatten()[0] = 109
    s2.flatten()[0] = 145
    return (
        (w13.view(torch.int8), s13.view(torch.float8_e8m0fnu)),
        (w2.view(torch.int8), s2.view(torch.float8_e8m0fnu)),
    )


def _dense_fp4(pair):
    """Natural E2M1 bytes, even K in the low nibble, and group-32 E8M0."""
    packed, scales = (value.contiguous().view(torch.uint8) for value in pair)
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
    )
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    return lut[codes.long()] * torch.exp2(scales.float() - 127).repeat_interleave(
        32, dim=-1
    )


def _reference_quant(values):
    """Scalar row maxima make the reference independent of the production helper."""
    rows = values.float().reshape(-1, values.shape[-1])
    scales = torch.tensor(
        [max(max(abs(value) for value in row), 1e-10) / 448 for row in rows.tolist()],
        dtype=torch.float32,
    ).unsqueeze(-1)
    quant = (rows / scales).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quant, scales


def _reference_forward(x, ids, weights, pairs):
    """Per-slot reference: independently route, quantize, project, and sum."""
    w13, w2 = (_dense_fp4(pair) for pair in pairs)
    result = torch.zeros(x.shape, dtype=torch.float32)
    for token in range(x.shape[0]):
        a1, s1 = _reference_quant(x[token : token + 1])
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            if expert < 0:
                continue
            gateup = ((a1.float() @ w13[expert].T) * s1).to(torch.bfloat16)
            gate, up = gateup.float().chunk(2, dim=-1)
            activation = F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
            activation *= weights[token, slot].float()
            a2, s2 = _reference_quant(activation)
            contribution = ((a2.float() @ w2[expert].T) * s2).to(torch.bfloat16)
            result[token] += contribution[0].float()
    return result.to(torch.bfloat16)


class _CPUGroupedKernels:
    """A strict native-call double, with a deliberately non-token-order scatter."""

    def __init__(self, case, pairs):
        self.case = case
        self.pairs = pairs
        self.preprocessed = []
        self.gemm_calls = []
        self.gather_weights = []
        self.downcast_inputs = []
        self.module = types.ModuleType("deep_gemm")
        self.mode = object()
        self.module.PreprocessMode = types.SimpleNamespace(W4A8_MXFP4_MMA=self.mode)
        self.module.preprocess_weights_and_scales = self.preprocess
        self.module.m_grouped_gemm_fp8_fp4_bf16_nt_nopad = self.gemm_w4a8
        self.module.m_grouped_gemm_fp4_fp4_bf16_nt_nopad = self.gemm_w4a4

    def preprocess(self, mode, fuse_activation, packed, scales):
        self.case.assertIs(mode, self.mode)
        self.case.assertFalse(fuse_activation)
        self.case.assertEqual(packed.dtype, torch.uint8)
        self.case.assertEqual(scales.dtype, torch.uint8)
        minimum, maximum = int(scales.min()), int(scales.max())
        shift = 114 - minimum if minimum < 114 else min(140 - maximum, 0)
        inverse = math.ldexp(1.0, -shift)
        e, n, packed_k = packed.shape
        prepared_weight = packed.clone()
        # The native layout permutation is external to this integration test.
        prepared_scale = torch.empty(e, n // 64, 4 * packed_k, dtype=torch.uint8)
        record = {
            "input_weight": packed.clone(),
            "input_scale": scales.clone(),
            "weight": prepared_weight,
            "scale": prepared_scale,
            "dense": _dense_fp4((packed, scales)) / inverse,
            "inverse": inverse,
        }
        self.preprocessed.append(record)
        return prepared_weight, prepared_scale, inverse

    def scatter(
        self, x, ids, counts, starts, grouped, indices, output_index, block_e=1
    ):
        self.case.assertEqual(block_e, 1)
        self.case.assertEqual(ids.dtype, torch.int32)
        # Poison padding to catch attempts to gather/index padded slots.
        output_index.fill_(torch.iinfo(torch.int32).max)
        cursor = 0
        for expert in range(counts.numel()):
            starts[expert] = cursor
            begin = cursor
            for token in reversed(range(x.shape[0])):
                for slot in range(ids.shape[1]):
                    if int(ids[token, slot]) == expert:
                        grouped[cursor] = x[token]
                        indices[cursor] = expert
                        output_index[token, slot] = cursor
                        cursor += 1
            self.case.assertEqual(cursor - begin, int(counts[expert]))
        self.case.assertEqual(cursor, grouped.shape[0])

    def gather(self, grouped, ids, weights, indices, out, routed_scaling_factor=1.0):
        self.gather_weights.append(weights.clone())
        result = torch.zeros(out.shape, dtype=torch.float32)
        for token in range(ids.shape[0]):
            for slot in range(ids.shape[1]):
                if int(ids[token, slot]) >= 0:
                    result[token] += (
                        grouped[int(indices[token, slot])].float()
                        * weights[token, slot]
                    )
        out.copy_((result * routed_scaling_factor).to(out.dtype))

    def gemm_w4a8(
        self, lhs, rhs, out, m_indices, m_rows=None, configs=None, inv_scale=1.0
    ):
        # A bias argument or a legacy config position would violate this ABI.
        record = next(item for item in self.preprocessed if item["scale"] is rhs[1])
        self.case.assertIs(rhs[0], record["weight"])
        self.case.assertEqual(lhs[0].dtype, torch.float8_e4m3fn)
        self.case.assertEqual(lhs[1].dtype, torch.float32)
        self.case.assertEqual(tuple(lhs[1].shape), (out.shape[0], 1))
        self.case.assertEqual(out.dtype, torch.bfloat16)
        self.case.assertEqual(m_rows.dtype, torch.int32)
        self.case.assertEqual(m_indices.dtype, torch.int32)
        self.case.assertIsNone(configs)
        self.case.assertEqual(inv_scale, record["inverse"])
        for value in (*lhs, *rhs, out, m_indices, m_rows):
            self.case.assertTrue(value.is_contiguous())
        expected_indices = torch.repeat_interleave(
            torch.arange(m_rows.numel(), dtype=torch.int32), m_rows.long()
        )
        self.case.assertTrue(torch.equal(m_indices, expected_indices))
        self.gemm_calls.append(
            {
                "values": lhs[0].float().clone(),
                "scale": lhs[1].clone(),
                "inv": inv_scale,
            }
        )
        start = 0
        for expert, count in enumerate(m_rows.tolist()):
            stop = start + count
            value = lhs[0][start:stop].float() @ record["dense"][expert].T
            out[start:stop] = (value * lhs[1][start:stop] * inv_scale).to(out.dtype)
            start = stop

    def downcast_identity(self, values, axis=1):
        # W4A4 quantization is separately tested; this double isolates weighting.
        self.case.assertEqual(axis, 1)
        self.downcast_inputs.append(values.clone())
        return values.clone(), torch.ones(values.shape[0], 1)

    def gemm_w4a4(self, lhs, rhs, bias, out, m_indices, m_rows):
        self.case.assertIsNone(bias)
        pair = next(
            pair for pair in self.pairs if pair[0].data_ptr() == rhs[0].data_ptr()
        )
        dense = _dense_fp4(pair)
        self.gemm_calls.append({"values": lhs[0].clone()})
        start = 0
        for expert, count in enumerate(m_rows.tolist()):
            stop = start + count
            out[start:stop] = (lhs[0][start:stop].float() @ dense[expert].T).to(
                out.dtype
            )
            start = stop


class PrecisionSelectionTest(unittest.TestCase):
    def test_capability_probe_tracks_selected_precision_and_preprocessing_mode(self):
        from rtp_llm.models_py.modules.dsv41.ppu import backend

        self.assertEqual(
            dict(backend.kernel_symbols({}))["deep_gemm"],
            (moe_kernel_name("w4a8"), "preprocess_weights_and_scales"),
        )
        self.assertEqual(
            dict(backend.kernel_symbols({MOE_PRECISION_ENV: "w4a4"}))["deep_gemm"],
            (moe_kernel_name("w4a4"),),
        )
        with self.assertRaisesRegex(RuntimeError, "W4A8_MXFP4_MMA"):
            backend.check_moe_preprocess(types.SimpleNamespace(), {})
        backend.check_moe_preprocess(
            types.SimpleNamespace(
                PreprocessMode=types.SimpleNamespace(W4A8_MXFP4_MMA=object())
            ),
            {},
        )
        backend.check_moe_preprocess(
            types.SimpleNamespace(), {MOE_PRECISION_ENV: "w4a4"}
        )

    def test_default_and_explicit_overrides(self):
        self.assertEqual(moe_precision({}), "w4a8")
        for value, expected in (("w4a8", "w4a8"), ("w4a4", "w4a4"), (" W4A4 ", "w4a4")):
            with self.subTest(value=value):
                self.assertEqual(moe_precision({MOE_PRECISION_ENV: value}), expected)
                self.assertEqual(
                    moe_kernel_name(expected),
                    f"m_grouped_gemm_{'fp8' if expected == 'w4a8' else 'fp4'}_fp4_bf16_nt_nopad",
                )

    def test_invalid_precision_is_rejected(self):
        for value in ("", "fp8", "w4a16", "auto"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, MOE_PRECISION_ENV):
                    moe_precision({MOE_PRECISION_ENV: value})

    def test_existing_instance_keeps_its_precision_when_environment_changes(self):
        runner = _runner("w4a8")
        with mock.patch.dict(os.environ, {MOE_PRECISION_ENV: "w4a4"}):
            self.assertEqual(runner.precision, "w4a8")
            self.assertEqual(runner.execution_stats()["precision"], "w4a8")
            self.assertEqual(_runner("w4a4").precision, "w4a4")

    def test_missing_w4a8_dependency_never_silently_falls_back_on_retry(self):
        runner = _runner("w4a8")
        fake_device_tensor = types.SimpleNamespace(is_cuda=True)
        with mock.patch.object(
            moe_module.importlib.util, "find_spec", return_value=None
        ):
            for attempt in range(2):
                with self.subTest(attempt=attempt):
                    with self.assertRaisesRegex(RuntimeError, "W4A8"):
                        runner._use_grouped_gemm(fake_device_tensor)


class ActivationAndRoutingTest(unittest.TestCase):
    def test_per_row_scales_zero_rows_and_contiguous_outputs(self):
        values = torch.zeros(2, 128, 3).transpose(1, 2)
        values[0, 1, 0], values[0, 1, 1] = 3.5, -7.0
        values[1, 0, 17] = 448.0
        self.assertFalse(values.is_contiguous())
        quant, scales = moe_module.quantize_per_row_fp8(values)
        self.assertEqual(quant.dtype, torch.float8_e4m3fn)
        self.assertEqual(scales.dtype, torch.float32)
        self.assertEqual(tuple(scales.shape), (2, 3, 1))
        self.assertTrue(quant.is_contiguous() and scales.is_contiguous())
        self.assertTrue(torch.isfinite(quant.float()).all())
        self.assertTrue(torch.isfinite(scales).all() and (scales > 0).all())
        self.assertTrue(torch.equal(quant[0, 0].float(), torch.zeros(128)))
        self.assertAlmostEqual(float(scales[0, 0, 0]), 1e-10 / 448, delta=1e-19)
        self.assertAlmostEqual(float(scales[0, 1, 0]), 7 / 448, delta=1e-9)
        self.assertEqual(float(scales[1, 0, 0]), 1.0)
        self.assertEqual(float(quant[0, 1, 1].float()), -448.0)

    def test_route_slots_map_to_distinct_grouped_rows_and_ignore_poisoned_padding(self):
        ids = torch.tensor([[1, 0, -1, -1], [0, 1, -1, -1]], dtype=torch.int32)
        weights = torch.tensor(
            [[0.3, 0.2, float("nan"), 9.0], [0.7, 0.0, 8.0, float("nan")]]
        )
        poison = torch.iinfo(torch.int32).max
        indices = torch.tensor(
            [[3, 1, poison, -poison], [0, 2, -poison, poison]], dtype=torch.int32
        )
        actual = moe_module.grouped_route_weights(ids, weights, indices, 4)
        torch.testing.assert_close(
            actual, torch.tensor([0.7, 0.2, 0.0, 0.3]), rtol=0, atol=0
        )
        self.assertEqual(actual.dtype, torch.float32)
        empty = moe_module.grouped_route_weights(
            ids[:, 2:], weights[:, 2:], indices[:, 2:], 0
        )
        self.assertEqual(tuple(empty.shape), (0,))

    def test_route_weight_multiplies_fp32_swiglu_before_quantization_without_bf16_round(
        self,
    ):
        gateup = torch.tensor(
            [
                [1.0, 12.0, -12.0, 1.0078125, -12.0, 4.0],
                [0.125, -2.0, 2.5, 1.5, 0.75, -1.25],
            ],
            dtype=torch.bfloat16,
        )
        weights = torch.tensor([0.3, 0.7], dtype=torch.float32)
        original = gateup.clone()
        quantizer = moe_module.quantize_per_row_fp8
        for limit in (10.0, 0.0):
            expected = []
            for row, weight in zip(gateup.float().tolist(), weights.tolist()):
                gate, up = row[:3], row[3:]
                if limit > 0:
                    gate = [min(value, limit) for value in gate]
                    up = [max(-limit, min(value, limit)) for value in up]
                expected.append(
                    [g / (1 + math.exp(-g)) * u * weight for g, u in zip(gate, up)]
                )
            expected = torch.tensor(expected, dtype=torch.float32)
            with self.subTest(limit=limit), mock.patch.object(
                moe_module, "quantize_per_row_fp8", wraps=quantizer
            ) as quantize:
                quant, scale = moe_module.routed_swiglu_fp8(gateup, weights, limit)
                quantize.assert_called_once()
                weighted = quantize.call_args.args[0]
                self.assertEqual(weighted.dtype, torch.float32)
                torch.testing.assert_close(weighted, expected, rtol=2e-6, atol=1e-8)
                self.assertFalse(torch.equal(weighted, expected.bfloat16().float()))
                reference_q, reference_s = _reference_quant(expected)
                torch.testing.assert_close(scale, reference_s, rtol=2e-6, atol=1e-12)
                torch.testing.assert_close(
                    quant.float(), reference_q.float(), rtol=0, atol=0
                )
        self.assertTrue(torch.equal(gateup, original))


class GroupedIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.pairs = _expert_pairs()
        self.generator = torch.Generator().manual_seed(742)
        self.x = torch.randn(3, 128, generator=self.generator).to(torch.bfloat16)
        self.ids = torch.tensor([[1, 0, -1, -1], [0, -1, -1, -1], [1, 0, -1, -1]])
        self.weights = torch.tensor(
            [[0.2, 0.8, 0, 0], [0.35, 0, 0, 0], [0.65, 0, 0, 0]]
        )
        self.counts = [3, 2]

    def test_w4a8_native_abi_byte_views_compensation_once_and_unweighted_gather(self):
        runner = _runner("w4a8")
        runner.bind_expert_weights(7, *self.pairs)
        kernels = _CPUGroupedKernels(self, self.pairs)
        with mock.patch.dict(
            sys.modules, {"deep_gemm": kernels.module}
        ), mock.patch.object(
            moe_module, "ep_scatter_sail", side_effect=kernels.scatter
        ), mock.patch.object(
            moe_module, "ep_gather", side_effect=kernels.gather
        ):
            actual = runner._local_compute_grouped(
                self.x, self.ids, self.weights, self.counts, 7
            )
        self.assertEqual(len(kernels.preprocessed), 2)
        self.assertEqual([call["inv"] for call in kernels.gemm_calls], [1 / 32, 32.0])
        for record, pair in zip(kernels.preprocessed, self.pairs):
            self.assertTrue(
                torch.equal(record["input_weight"], pair[0].view(torch.uint8))
            )
            self.assertTrue(
                torch.equal(record["input_scale"], pair[1].view(torch.uint8))
            )
        # An already-folded scale plus inv_scale would compensate twice.
        _, first_scale = _reference_quant(self.x[2:3])
        torch.testing.assert_close(
            kernels.gemm_calls[0]["scale"][0:1], first_scale, rtol=0, atol=0
        )
        self.assertEqual(len(kernels.gather_weights), 1)
        self.assertTrue(
            torch.equal(kernels.gather_weights[0], torch.ones_like(self.weights))
        )
        expected = _reference_forward(self.x, self.ids, self.weights, self.pairs)
        torch.testing.assert_close(actual, expected, rtol=0.008, atol=1e-5)
        stats = runner.execution_stats()
        self.assertEqual(stats["grouped_forward_calls"], 1)
        self.assertEqual(stats["grouped_gemm_calls"], 2)
        self.assertEqual(
            stats["route_weight_stage"], "before_gemm2_activation_quantization"
        )

    def test_w4a4_override_preserves_post_gemm_route_weights_and_legacy_abi(self):
        runner = _runner("w4a4")
        runner.bind_expert_weights(7, *self.pairs)
        kernels = _CPUGroupedKernels(self, self.pairs)
        with mock.patch.dict(
            sys.modules, {"deep_gemm": kernels.module}
        ), mock.patch.object(
            moe_module, "ep_scatter_sail", side_effect=kernels.scatter
        ), mock.patch.object(
            moe_module, "ep_gather", side_effect=kernels.gather
        ), mock.patch.object(
            moe_module, "downcast_to_mxfp4", side_effect=kernels.downcast_identity
        ):
            actual = runner._local_compute_grouped(
                self.x, self.ids, self.weights, self.counts, 7
            )
        self.assertEqual(len(kernels.preprocessed), 0)
        self.assertEqual(len(kernels.gemm_calls), 2)
        self.assertTrue(torch.equal(kernels.gather_weights[0], self.weights))
        self.assertEqual(kernels.downcast_inputs[1].dtype, torch.bfloat16)
        # Only the boundary under test is mocked: expert projections still use
        # independently decoded weights, followed by the legacy BF16 activation.
        expected = torch.zeros_like(self.x, dtype=torch.float32)
        w13, w2 = (_dense_fp4(pair) for pair in self.pairs)
        for token in range(self.x.shape[0]):
            for slot in range(self.ids.shape[1]):
                expert = int(self.ids[token, slot])
                if expert < 0:
                    continue
                gateup = (self.x[token].float() @ w13[expert].T).to(torch.bfloat16)
                gate, up = gateup.float().chunk(2)
                hidden = (F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)).to(
                    torch.bfloat16
                )
                contribution = (hidden.float() @ w2[expert].T).to(torch.bfloat16)
                expected[token] += contribution.float() * self.weights[token, slot]
        torch.testing.assert_close(actual, expected.bfloat16(), rtol=0.008, atol=1e-5)
        self.assertEqual(runner.execution_stats()["route_weight_stage"], "after_gemm2")

    def test_weight_rebinding_invalidates_all_devices_only_for_rebound_layer(self):
        runner = _runner("w4a8")
        runner.bind_expert_weights(7, *self.pairs)
        runner.bind_expert_weights(8, *self.pairs)
        kernels = _CPUGroupedKernels(self, self.pairs)
        device = torch.device("cpu")
        with mock.patch.dict(sys.modules, {"deep_gemm": kernels.module}):
            before = runner._processed_weights(7, device)
            other_layer = runner._processed_weights(8, device)
            self.assertIs(runner._processed_weights(7, device), before)
            self.assertEqual(len(kernels.preprocessed), 4)
            # A second cached device must also be invalidated without probing it.
            runner._processed[(7, torch.device("cuda:7"))] = before
            replacement = _expert_pairs(seed=743)
            runner.bind_expert_weights(7, *replacement)
            self.assertFalse(any(key[0] == 7 for key in runner._processed))
            self.assertIs(runner._processed_weights(8, device), other_layer)
            after = runner._processed_weights(7, device)
            self.assertIsNot(after, before)
            self.assertEqual(len(kernels.preprocessed), 6)
            self.assertTrue(
                torch.equal(
                    kernels.preprocessed[-2]["input_weight"],
                    replacement[0][0].view(torch.uint8),
                )
            )


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_W4A8_DEVICE") == "1",
    "real PPU execution requires RTP_V41_TEST_W4A8_DEVICE=1",
)
class W4A8DeviceTest(unittest.TestCase):
    def test_w4a4_device_override_with_an_empty_expert_group(self):
        if not torch.cuda.is_available():
            self.skipTest("no PPU device available")
        if "M890" not in torch.cuda.get_device_name(0).upper():
            self.skipTest("requires PPU1.5")
        runner = _runner("w4a4", hidden=5120, inter=2304)
        pairs = _expert_pairs(seed=745, hidden=5120, inter=2304)
        pairs = tuple(tuple(value.to("cuda:0") for value in pair) for pair in pairs)
        runner.bind_expert_weights(0, *pairs)
        x = torch.randn(5, 5120, device="cuda:0", dtype=torch.bfloat16)
        ids = torch.full((5, 8), -1, device="cuda:0", dtype=torch.int64)
        ids[:, 0] = 1
        weights = torch.zeros(5, 8, device="cuda:0", dtype=torch.float32)
        weights[:, 0] = 0.375
        result = runner._local_compute_grouped(x, ids, weights, [0, 5], 0)
        torch.cuda.synchronize()
        self.assertEqual(tuple(result.shape), (5, 5120))
        self.assertTrue(torch.isfinite(result).all().item())
        self.assertEqual(runner.execution_stats()["precision"], "w4a4")
        self.assertEqual(runner.execution_stats()["grouped_gemm_calls"], 2)

    def test_two_native_gemms_with_1017_tokens_and_v41_projection_dimensions(self):
        # This is the first accelerator probe in this module, after env gating.
        if not torch.cuda.is_available():
            self.skipTest("no PPU device available")
        device_name = torch.cuda.get_device_name(0).upper()
        if "PPU" not in device_name and "M890" not in device_name:
            self.skipTest(f"requires PPU1.5, found {device_name}")
        runner = _runner("w4a8", hidden=5120, inter=2304)
        # Two local experts bound memory use; full EP8/top-6 is an E2E run.
        pairs = _expert_pairs(seed=744, hidden=5120, inter=2304)
        pairs = tuple(tuple(value.to("cuda:0") for value in pair) for pair in pairs)
        runner.bind_expert_weights(0, *pairs)
        x = torch.randn(1017, 5120, device="cuda:0", dtype=torch.bfloat16)
        ids = torch.full((1017, 8), -1, device="cuda:0", dtype=torch.int64)
        ids[:, 0], ids[:, 1] = 0, 1
        weights = torch.zeros(1017, 8, device="cuda:0", dtype=torch.float32)
        weights[:, 0], weights[:, 1] = 0.2, 0.8
        result = runner._local_compute_grouped(x, ids, weights, [1017, 1017], 0)
        torch.cuda.synchronize()
        self.assertEqual(tuple(result.shape), (1017, 5120))
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(result).all().item())
        self.assertEqual(runner.execution_stats()["grouped_gemm_calls"], 2)


if __name__ == "__main__":
    unittest.main()

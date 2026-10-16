"""SM90 MXFP4 routing, scale-preservation and grouped expert contracts."""

import json
import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.moe._sm90_humming_linear import (
    SM90HummingGroupedLinear,
    decode_quantized_activation,
    unpack_activation_scales,
    validate_lossless_humming_scales,
)
from rtp_llm.models_py.modules.dsv41.moe.strategies.base import MoeCfg, select_strategy
from rtp_llm.models_py.modules.dsv41.moe.strategies.sm90_humming import (
    SM90HummingStrategy,
    _GroupedLocalExperts,
    group_local_routes,
)


def _cfg(ep_size=1):
    return MoeCfg(
        layer_id=0,
        dim=128,
        moe_inter_dim=128,
        n_routed_experts=8,
        n_activated_experts=6,
        swiglu_limit=10.0,
        ep_size=ep_size,
        ep_rank=0,
        n_local_experts=8 // ep_size,
        local_expert_start=0,
        local_expert_end=8 // ep_size,
        max_tokens_per_rank=32,
        shared_fp8_block_size=32,
    )


class SM90RoutingContractTest(unittest.TestCase):
    def test_packed_activation_scale_expansion_preserves_ue8m0_values(self):
        raw = torch.tensor([[0, 100, 127, 254], [1, 126, 128, 0]], dtype=torch.uint8)
        packed = raw.view(torch.int32)
        result = unpack_activation_scales(packed, 3)
        expected = torch.tensor(
            [[2.0**-127, 2.0**-27, 1], [2.0**-126, 0.5, 2]], dtype=torch.float32
        )
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self.assertTrue(result.is_contiguous())
        tma_scale = torch.empty((1, 4), dtype=torch.int32).T[:2]
        tma_scale.copy_(packed)
        torch.testing.assert_close(
            unpack_activation_scales(tma_scale, 3), expected, rtol=0, atol=0
        )
        torch.testing.assert_close(
            unpack_activation_scales(tma_scale[:1], 3), expected[:1], rtol=0, atol=0
        )
        nan_scale = torch.tensor([[255, 127, 127, 127]], dtype=torch.uint8)
        self.assertTrue(
            torch.isnan(unpack_activation_scales(nan_scale.view(torch.int32), 1)).all()
        )

    def test_local_permutation_excludes_padding_and_preserves_slot_order(self):
        routes = torch.tensor([[5, 4, -1], [6, 4, 99], [-1, -1, -1]])
        slots, offsets = group_local_routes(routes, 4, 8)
        self.assertEqual(slots.tolist(), [1, 4, 0, 3])
        self.assertEqual(offsets.tolist(), [0, 2, 3, 4, 4])
        self.assertEqual(offsets.dtype, torch.int32)
        empty_slots, empty_offsets = group_local_routes(routes, 10, 12)
        self.assertEqual(empty_slots.numel(), 0)
        self.assertEqual(empty_offsets.tolist(), [0, 0, 0])

    def test_bf16_decode_preserves_already_quantized_values(self):
        payload = torch.tensor([-448, -1.5, -0.001953125, 0, 0.5, 1.5, 224, 448])
        payload = payload.repeat(32).reshape(1, 256).to(torch.float8_e4m3fn)
        packed = torch.tensor([[105, 135, 127, 127]], dtype=torch.uint8).view(
            torch.int32
        )
        decoded = decode_quantized_activation(payload, packed)
        expected = payload.float() * torch.tensor(
            [2.0**-22, 2.0**8]
        ).repeat_interleave(128)
        self.assertEqual(decoded.dtype, torch.bfloat16)
        torch.testing.assert_close(decoded.float(), expected, rtol=0, atol=0)

    def test_unsupported_bf16_scale_conversion_is_rejected(self):
        valid = torch.tensor([[[1, 252]], [[100, 141]]], dtype=torch.uint8)
        validate_lossless_humming_scales(valid.view(torch.float8_e8m0fnu))
        for invalid in ([0, 127], [127, 253], [254, 255]):
            scale = torch.tensor([[invalid]], dtype=torch.uint8).view(
                torch.float8_e8m0fnu
            )
            with self.assertRaisesRegex(ValueError, "refusing unsupported"):
                validate_lossless_humming_scales(scale)

    def test_sm90_selects_grouped_for_ep1_and_ep8(self):
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_capability", return_value=(9, 0)
        ), patch.dict("os.environ", {"DSV4_MOE_STRATEGY": ""}):
            for ep in (1, 8):
                self.assertIs(select_strategy(_cfg(ep)), SM90HummingStrategy)
            with self.assertRaisesRegex(RuntimeError, "requested strategy"):
                select_strategy(_cfg(8), forced="mega", strict=True)


def _quantized_activation_parts(x):
    blocks = x.float().reshape(x.shape[0], -1, 128)
    scales = torch.pow(
        2.0, torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448))
    )
    values = (blocks / scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return values.reshape_as(x), scales


def _quantized_activation_reference(x):
    values, scales = _quantized_activation_parts(x)
    return (
        values.float().reshape(x.shape[0], -1, 128) * scales.unsqueeze(-1)
    ).reshape_as(x)


def _dequant_fp4_reference(weight, scales):
    # Independent nibble decode, including the FP4 signed-zero codes.
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=weight.device,
    )
    packed = weight.to(torch.int16) & 255
    nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    return lut[nibbles.long()] * scales.float().repeat_interleave(32, dim=-1)


def _eager_activation_decode_reference(payload, packed):
    """Independent frozen pre-fusion implementation; do not call the DUT helpers."""
    m, k = payload.shape
    contiguous = packed.contiguous()
    contiguous = contiguous.as_strided(contiguous.shape, (contiguous.shape[1], 1))
    scales = contiguous.view(torch.float8_e8m0fnu)[:, : k // 128].float().contiguous()
    return (
        (payload.float().reshape(m, k // 128, 128) * scales.unsqueeze(-1))
        .reshape(m, k)
        .bfloat16()
    )


def _strided_activation_decode_inputs(m, k):
    # Both views have nonzero storage offsets; column-major TMA scales also
    # retain a non-unit singleton-column stride when k <= 512.
    payload = torch.randn((m + 1, 2 * k + 4), device="cuda").to(torch.float8_e4m3fn)
    payload = payload[1:, 2 : 2 + 2 * k : 2]
    scale_cols = (k // 128 + 3) // 4
    raw = torch.randint(100, 145, (m, scale_cols * 4), dtype=torch.uint8, device="cuda")
    scale = torch.empty((2 * scale_cols + 1, m + 7), dtype=torch.int32, device="cuda").T
    scale = scale[1 : m + 1, 1 : 1 + 2 * scale_cols : 2]
    scale.copy_(raw.view(torch.int32))
    return payload, scale


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9,
    "requires SM90",
)
class SM90ActivationDecodeTest(unittest.TestCase):
    def _check_bits(self, payload, scale):
        expected = _eager_activation_decode_reference(payload, scale)
        actual = decode_quantized_activation(payload, scale)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(actual.is_contiguous())
        # Integer comparison includes signed zeros and NaN bit patterns, rather
        # than silently relaxing the nonfinite cases with equal_nan=True.
        torch.testing.assert_close(
            actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
        )

    def test_all_fp8_codes_and_all_ue8m0_scales_bitwise(self):
        # Every E4M3 byte crosses every UE8M0 byte, including negative zero,
        # NaNs, exponent-zero subnormals, and overflow to BF16 infinity.
        codes = torch.arange(256, dtype=torch.int32, device="cuda").to(torch.uint8)
        payload = codes[None, :].expand(256, -1).contiguous().view(torch.float8_e4m3fn)
        packed = codes[:, None].expand(-1, 4).contiguous().view(torch.int32)
        self._check_bits(payload, packed)

    def test_noncontiguous_payload_and_tma_scales_bitwise(self):
        for m, k in ((1, 128), (1, 384), (3, 640), (1, 5120), (17, 2304), (7, 7168)):
            with self.subTest(m=m, k=k):
                self._check_bits(*_strided_activation_decode_inputs(m, k))

    def test_real_shape_quantizer_payload_bitwise(self):
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )

        torch.manual_seed(20261014)
        for m, k in ((1, 5120), (1, 2304), (4096, 5120), (4096, 2304), (4096, 7168)):
            with self.subTest(m=m, k=k):
                x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
                payload, scale = sgl_per_token_group_quant_fp8(
                    x,
                    group_size=128,
                    eps=1e-4,
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=True,
                )
                self._check_bits(payload, scale)

    def test_empty_rows(self):
        payload = torch.empty((0, 5120), device="cuda", dtype=torch.float8_e4m3fn)
        scale = torch.empty((0, 10), device="cuda", dtype=torch.int32)
        actual = decode_quantized_activation(payload, scale)
        self.assertEqual(actual.shape, payload.shape)
        self.assertEqual(actual.dtype, torch.bfloat16)

    @unittest.skipUnless(
        os.environ.get("DSV41_SM90_DECODE_BENCH") == "1"
        or os.environ.get("DSV41_SM90_DECODE_PROFILE") == "1",
        "opt-in diagnostic microbenchmark/component trace; no performance threshold",
    )
    def test_benchmark_fused_vs_frozen_eager(self):
        import triton.testing
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )

        profile_inputs = []
        for m, k in ((1, 5120), (1, 2304), (4096, 5120), (4096, 2304), (4096, 7168)):
            x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            payload, scale = sgl_per_token_group_quant_fp8(
                x,
                group_size=128,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
            self._check_bits(payload, scale)
            baseline = lambda: _eager_activation_decode_reference(payload, scale)
            candidate = lambda: decode_quantized_activation(payload, scale)
            # Compile/warm both paths first. Alternate AB/BA to reduce order bias.
            baseline()
            candidate()
            torch.cuda.synchronize()
            if os.environ.get("DSV41_SM90_DECODE_PROFILE") == "1" and k != 7168:
                profile_inputs.append((m, k, payload, scale))
            if os.environ.get("DSV41_SM90_DECODE_BENCH") != "1":
                continue
            measurements = {"eager_ms": [], "fused_ms": []}
            for repeat in range(3):
                cases = [("eager_ms", baseline), ("fused_ms", candidate)]
                if repeat % 2:
                    cases.reverse()
                for name, fn in cases:
                    measurements[name].append(
                        triton.testing.do_bench(
                            fn, warmup=20, rep=100, return_mode="median"
                        )
                    )
            print(
                "SM90_ACTIVATION_DECODE_BENCH "
                + json.dumps(
                    {
                        "m": m,
                        "k": k,
                        "payload_stride": payload.stride(),
                        "scale_stride": scale.stride(),
                        "bitwise_equal": True,
                        **measurements,
                    }
                ),
                flush=True,
            )
        if profile_inputs:
            # This is a component-only A/B trace, not a model inference trace.
            # Keep quantization/allocation of the identical inputs outside it.
            for _, _, payload, scale in profile_inputs:
                for _ in range(10):
                    _eager_activation_decode_reference(payload, scale)
                    decode_quantized_activation(payload, scale)
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as prof:
                for repeat in range(3):
                    for m, k, payload, scale in profile_inputs:
                        cases = [
                            ("baseline_eager", _eager_activation_decode_reference),
                            ("fused", decode_quantized_activation),
                        ]
                        if repeat % 2:
                            cases.reverse()
                        with torch.profiler.record_function(
                            f"moe_dequant_component_ab/M={m}/K={k}/repeat={repeat}"
                        ):
                            for name, fn in cases:
                                with torch.profiler.record_function(name):
                                    fn(payload, scale)
                torch.cuda.synchronize()
            output_dir = os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]
            os.makedirs(output_dir, exist_ok=True)
            trace_path = os.path.join(
                output_dir, "round1_moe_dequant_component_ab.json"
            )
            prof.export_chrome_trace(trace_path)
            print(f"SM90_ACTIVATION_DECODE_COMPONENT_TRACE {trace_path}", flush=True)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9,
    "requires SM90",
)
class SM90GroupedExpertTest(unittest.TestCase):
    def setUp(self):
        previous = torch.backends.cuda.matmul.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", previous)
        torch.backends.cuda.matmul.allow_tf32 = False

    def _check_stage(self, name, actual, expected, *, rtol=0, atol=0):
        actual, expected = actual.float(), expected.float()
        different = actual != expected
        coordinates = different.nonzero()[:8].tolist()
        details = [
            (coord, actual[tuple(coord)].item(), expected[tuple(coord)].item())
            for coord in coordinates
        ]
        print(
            f"MoE stage={name} unequal={different.sum().item()}/{actual.numel()} "
            f"max_abs={(actual - expected).abs().max().item():.8g} first={details}",
            flush=True,
        )
        with self.subTest(stage=name):
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    def test_grouped_moe_matches_quantized_reference_with_pre_down_route_weights(self):
        torch.manual_seed(20261013)
        cfg = _cfg()
        local = _GroupedLocalExperts(cfg)
        weights = [
            torch.randint(-128, 128, (8, 128, 64), dtype=torch.int8, device="cuda")
            for _ in range(3)
        ]
        scales = [
            torch.randint(121, 125, (8, 128, 4), dtype=torch.uint8, device="cuda").view(
                torch.float8_e8m0fnu
            )
            for _ in range(3)
        ]
        w1, w3, w2 = weights
        s1, s3, s2 = scales
        packed_original = [weight.clone() for weight in weights]
        scales_original = [scale.view(torch.uint8).clone() for scale in scales]
        local.w13 = SM90HummingGroupedLinear(
            torch.cat((w1, w3), 1),
            torch.cat((s1.view(torch.uint8), s3.view(torch.uint8)), 1).view(s1.dtype),
        )
        local.w2 = SM90HummingGroupedLinear(w2, s2)
        for original, weight in zip(packed_original, weights):
            torch.testing.assert_close(weight, original, rtol=0, atol=0)
        for original, scale in zip(scales_original, scales):
            torch.testing.assert_close(
                scale.view(torch.uint8), original, rtol=0, atol=0
            )
        decoded = [_dequant_fp4_reference(w, s) for w, s in zip(weights, scales)]
        # A basis-vector GEMM reads every effective transformed coefficient.
        # Each output has exactly one nonzero product, so this isolates packed
        # repacking + E8M0 weight scaling from accumulation / reduction error.
        basis = torch.eye(128, dtype=torch.bfloat16, device="cuda").repeat(8, 1)
        basis_offsets = torch.arange(9, dtype=torch.int32, device="cuda") * 128
        expected_w13 = torch.cat(
            [
                torch.cat((decoded[0][expert], decoded[1][expert]), 0).T
                for expert in range(8)
            ],
            0,
        ).bfloat16()
        expected_w2 = torch.cat(
            [decoded[2][expert].T for expert in range(8)], 0
        ).bfloat16()
        self._check_stage(
            "w13_effective_weight_basis", local.w13(basis, basis_offsets), expected_w13
        )
        self._check_stage(
            "w2_effective_weight_basis", local.w2(basis, basis_offsets), expected_w2
        )
        print(
            f"MoE w13_config={local.w13.config.to_str()} compute={local.w13.compute_config}",
            flush=True,
        )
        x = (torch.randn(4, 128, device="cuda") * 12).to(torch.bfloat16)
        routes = torch.tensor(
            [[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0], [2, 0, -1, -1, -1, -1], [-1] * 6],
            device="cuda",
        )
        route_weights = torch.tensor(
            [
                [0.05, 0.1, 0.2, 0.25, 0.4, 0.5],
                [0.5, 0.4, 0.25, 0.2, 0.1, 0.05],
                [0.6, 0.9, 0, 0, 0, 0],
                [0] * 6,
            ],
            dtype=torch.float32,
            device="cuda",
        )
        captured = {}

        def capture_w13(module, args, output):
            captured["grouped_x"] = args[0].clone()
            captured["w13"] = output.clone()

        def capture_w2(module, args, output):
            captured["hidden"] = args[0].clone()
            captured["down"] = output.clone()

        handles = [
            local.w13.register_forward_hook(capture_w13),
            local.w2.register_forward_hook(capture_w2),
        ]
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )

        quant_calls = []
        gemm_calls = []
        from humming.forward import humming_forward

        def capture_gemm(config, *args, **kwargs):
            self.assertEqual(config.input_scale_group_size, 0)
            self.assertEqual(kwargs["inputs"].dtype, torch.bfloat16)
            self.assertIsNone(kwargs.get("input_scale"))
            self.assertFalse(config.use_fused_e8m0_scale)
            gemm_calls.append(kwargs["inputs"].clone())
            return humming_forward(config, *args, **kwargs)

        def capture_quant(*args, **kwargs):
            payload, scale = sgl_per_token_group_quant_fp8(*args, **kwargs)
            quant_calls.append(
                (
                    payload.clone(),
                    unpack_activation_scales(scale, args[0].shape[-1] // 128),
                )
            )
            return payload, scale

        try:
            with patch(
                "rtp_llm.models_py.kernels.cuda.fp8_kernel.sgl_per_token_group_quant_fp8",
                side_effect=capture_quant,
            ), patch(
                "humming.forward.humming_forward",
                side_effect=capture_gemm,
            ):
                result = local._forward_into_buf(x, route_weights, routes, 0, 8)
        finally:
            for handle in handles:
                handle.remove()
        expected = torch.zeros_like(result)
        reference_slots = {}
        for row in range(x.shape[0]):
            row_outputs = []
            for slot in range(routes.shape[1]):
                expert = int(routes[row, slot])
                if expert < 0:
                    continue
                qx = _quantized_activation_reference(x[row : row + 1])
                raw_gate = (qx @ decoded[0][expert].T).to(torch.bfloat16)
                raw_up = (qx @ decoded[1][expert].T).to(torch.bfloat16)
                gate, up = raw_gate.float().clamp(max=10), raw_up.float().clamp(-10, 10)
                hidden = (
                    torch.nn.functional.silu(gate) * up * route_weights[row, slot]
                ).to(torch.bfloat16)
                qhidden = _quantized_activation_reference(hidden)
                down = (qhidden @ decoded[2][expert].T).to(torch.bfloat16)
                row_outputs.append(down.float())
                reference_slots[row * routes.shape[1] + slot] = (
                    torch.cat((raw_gate, raw_up), -1),
                    hidden,
                    down,
                )
            if row_outputs:
                expected[row] = torch.stack(row_outputs).sum(0)
        self.assertTrue(torch.isfinite(result).all())
        slots, _ = group_local_routes(routes, 0, 8)
        slot_ids = slots.tolist()
        ref_w13, ref_hidden, ref_down = (
            torch.cat([reference_slots[slot][stage] for slot in slot_ids], 0)
            for stage in range(3)
        )
        self.assertEqual(len(quant_calls), 2)
        self.assertEqual(len(gemm_calls), 2)
        self._check_stage(
            "permutation",
            captured["grouped_x"],
            x.index_select(0, slots // routes.shape[1]),
        )
        for stage, input_tensor in enumerate(
            (captured["grouped_x"], captured["hidden"])
        ):
            payload, scale = _quantized_activation_parts(input_tensor)
            self._check_stage(
                f"quant{stage}_payload_same_input", quant_calls[stage][0], payload
            )
            self._check_stage(
                f"quant{stage}_scale_same_input", quant_calls[stage][1], scale
            )
            self._check_stage(
                f"gemm{stage}_decoded_g128_values_exact",
                gemm_calls[stage],
                _quantized_activation_reference(input_tensor),
            )
        self._check_stage("w13_bf16", captured["w13"], ref_w13)
        grouped_experts = routes.reshape(-1).index_select(0, slots).tolist()
        grouped_qx = _quantized_activation_reference(captured["grouped_x"])
        fp64_w13 = torch.cat(
            [
                grouped_qx[row : row + 1].double()
                @ torch.cat((decoded[0][expert], decoded[1][expert]), 0).double().T
                for row, expert in enumerate(grouped_experts)
            ],
            0,
        )
        self._check_stage("w13_oracle_fp32_vs_fp64", ref_w13, fp64_w13.bfloat16())
        self._check_stage("w13_actual_vs_fp64", captured["w13"], fp64_w13.bfloat16())
        anchors = (captured["w13"] != ref_w13).nonzero()[:8].tolist()
        for row, column in anchors:
            expert = grouped_experts[row]
            joined = torch.cat((decoded[0][expert], decoded[1][expert]), 0)
            cpu_dot = (
                (grouped_qx[row].cpu().double() * joined[column].cpu().double())
                .sum()
                .item()
            )
            fp32_dot = (grouped_qx[row : row + 1] @ joined.T)[0, column].item()
            print(
                f"MoE dot_anchor row={row} col={column} actual={captured['w13'][row, column].item()} "
                f"fp32_raw={fp32_dot:.17g} fp64_gpu={fp64_w13[row, column].item():.17g} "
                f"fp64_cpu={cpu_dot:.17g}",
                flush=True,
            )
        self._check_stage(
            "hidden_bf16_after_route_weight", captured["hidden"], ref_hidden
        )
        # Isolate activation arithmetic from any preceding w13 rounding error.
        actual_gate, actual_up = captured["w13"].float().chunk(2, -1)
        hidden_same_w13 = (
            torch.nn.functional.silu(actual_gate.clamp(max=10))
            * actual_up.clamp(-10, 10)
            * route_weights.reshape(-1).index_select(0, slots).unsqueeze(-1)
        ).bfloat16()
        self._check_stage("hidden_bf16_same_w13", captured["hidden"], hidden_same_w13)
        self._check_stage("down_bf16", captured["down"], ref_down)
        quantized_hidden = _quantized_activation_reference(captured["hidden"])
        down_same_hidden = torch.cat(
            [
                (quantized_hidden[row : row + 1] @ decoded[2][expert].T).bfloat16()
                for row, expert in enumerate(grouped_experts)
            ],
            0,
        )
        self._check_stage("down_bf16_same_hidden", captured["down"], down_same_hidden)
        torch.testing.assert_close(result, expected, rtol=0.02, atol=0.05)
        torch.testing.assert_close(
            result[-1], torch.zeros_like(result[-1]), rtol=0, atol=0
        )
        empty = local._forward_into_buf(x[:0], route_weights[:0], routes[:0], 0, 8)
        self.assertEqual(tuple(empty.shape), (0, 128))


if __name__ == "__main__":
    unittest.main()

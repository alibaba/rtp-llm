"""Exact G128 MoE quantization/metadata gates and unchanged-GEMM timing.

Run only under the parent GPU lock. These gates do not waive the separate
native WGMMA numerical acceptance tests against the model's FP32 oracle.
"""

import json
import math
import os
import statistics
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41._sm90_moe_quant_scale_fused import (
    quantize_moe_g128,
)
from rtp_llm.models_py.modules.dsv41._sm90_quant_scale_fused import use_v2


def _baseline(x, promotion):
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
    from rtp_llm.models_py.modules.dsv41.moe._sm90_humming_linear import (
        unpack_activation_scales,
    )

    q, packed = sgl_per_token_group_quant_fp8(
        x.to(torch.bfloat16).contiguous(),
        group_size=128,
        eps=1e-4,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    scale = unpack_activation_scales(packed, x.shape[1] // 128)
    if promotion == 32:
        scale = scale.repeat_interleave(4, dim=-1).contiguous()
    return q, scale


def _bits(actual, expected):
    torch.testing.assert_close(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
        rtol=0,
        atol=0,
    )


def _finite_oracle(x, promotion, v2):
    """Independent Torch G128 math and exact mathematical power-of-two table."""
    rows, width = x.shape
    values = x.float().reshape(rows, width // 128, 128)
    maximum = values.abs().amax(-1).clamp_min(1e-4)
    raw = maximum / 448.0
    exponent = torch.ceil(torch.log2(raw)).long() + 127
    powers = torch.tensor(
        [math.ldexp(1.0, code - 127) for code in range(255)],
        device=x.device,
        dtype=torch.float32,
    )
    scale = powers[exponent]
    if v2:
        inverse = powers[(254 - exponent)]
        values = torch.where(
            values.abs() < torch.finfo(torch.float32).tiny,
            torch.copysign(torch.zeros_like(values), values),
            values,
        )
        scaled = values * inverse[..., None]
        scaled = torch.where(
            scaled.abs() < torch.finfo(torch.float32).tiny,
            torch.copysign(torch.zeros_like(scaled), scaled),
            scaled,
        )
    else:
        scaled = values / scale[..., None]
    q = scaled.clamp(-448, 448).to(torch.float8_e4m3fn).reshape(rows, width)
    if promotion == 32:
        scale = scale.repeat_interleave(4, dim=-1)
    return q, scale


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SM90MoEQuantScaleFusedTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("requires SM90")
        torch.manual_seed(241017)

    def _compare(self, x, promotion, *, oracle=True):
        saved = x.clone()
        actual = quantize_moe_g128(x, promotion_group_size=promotion)
        expected = _baseline(x, promotion)
        for a, b in zip(actual, expected):
            _bits(a, b)
        if oracle:
            for a, b in zip(actual, _finite_oracle(x, promotion, use_v2(*x.shape))):
                _bits(a, b)
        _bits(x, saved)
        self.assertTrue(all(a.is_contiguous() for a in actual))
        return actual

    def test_payload_scales_shapes_tails_strides_and_auto_threshold(self):
        for policy in ("legacy", "v2", "auto"):
            with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": policy}):
                for rows, width in (
                    (0, 2304),
                    (1, 128),
                    (3, 384),
                    (16, 2304),
                    (64, 5120),
                    (64, 2304),
                    (256, 5120),
                    (256, 2304),
                    (255, 16384),
                    (256, 16384),
                ):
                    x = torch.randn(
                        (rows, width + 128), device="cuda", dtype=torch.bfloat16
                    )[:, :width]
                    for promotion in (32, 128):
                        with self.subTest(
                            policy=policy, shape=x.shape, promotion=promotion
                        ):
                            self._compare(x, promotion)

    def test_floor_ties_subnormals_extremes_nonfinite(self):
        x = torch.randn((12, 2304), device="cuda", dtype=torch.bfloat16)
        x[0].zero_()
        x[1].fill_(-0.0)
        tie = torch.tensor(
            [448, 1.0625, 1.1875, 0.015625, -1.0625, -1.1875, -0.015625, -448],
            device="cuda",
            dtype=torch.bfloat16,
        )
        x[2] = tie.repeat(288)
        x[3] = tie.repeat(288) * 2**-22
        x[4] = tie.repeat(288) * 2**-126
        x[5].fill_(torch.finfo(torch.bfloat16).tiny)
        x[6].fill_(2**-133)
        x[7].fill_(torch.finfo(torch.bfloat16).max)
        x[8].fill_(-torch.finfo(torch.bfloat16).max)
        x[9, :128].fill_(float("inf"))
        x[9, 128:256].fill_(float("-inf"))
        x[10, :128].fill_(float("nan"))
        x[10, 128] = float("nan")
        x[11, ::8] = float("inf")
        for policy in ("legacy", "v2", "auto"):
            for promotion in (32, 128):
                with self.subTest(policy=policy, promotion=promotion), patch.dict(
                    os.environ, {"DSV4_FP8_QUANT_KERNEL": policy}
                ):
                    self._compare(x, promotion, oracle=False)
                    _, scales = self._compare(x[:9], promotion)
                    _bits(scales[:2], torch.full_like(scales[:2], 2**-22))

    def test_promotion_repeats_original_group128_not_requantization(self):
        # Large values in one G32 quarter must influence all four quarters.
        x = torch.ones((64, 2304), device="cuda", dtype=torch.bfloat16)
        x.view(64, 18, 128)[..., :32] = 448
        for policy in ("legacy", "v2", "auto"):
            with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": policy}):
                q32, s32 = self._compare(x, 32)
                q128, s128 = self._compare(x, 128)
                _bits(q32, q128)
                _bits(s32, s128.repeat_interleave(4, -1))
                _bits(s32, torch.ones_like(s32))

    def test_preallocated_outputs_validation_and_graph_live_input(self):
        x = torch.randn((64, 2304), device="cuda", dtype=torch.bfloat16)
        q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scale = torch.empty((64, 72), device="cuda")
        for policy in ("legacy", "v2", "auto"):
            with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": policy}):
                for _ in range(3):
                    quantize_moe_g128(x, out_q=q, out_scale=scale)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = quantize_moe_g128(x, out_q=q, out_scale=scale)
                self.assertEqual(actual[0].data_ptr(), q.data_ptr())
                self.assertEqual(actual[1].data_ptr(), scale.data_ptr())
                for multiplier in (0.0, 1.0, 32.0):
                    x.normal_().mul_(multiplier)
                    graph.replay()
                    for a, b in zip(actual, _baseline(x, 32)):
                        _bits(a, b)
        for invalid in (
            x[:, :2240],
            x[:, ::2],
            x.float(),
            x.detach().requires_grad_(),
            torch.empty((257, 128), dtype=torch.bfloat16, device="cuda"),
        ):
            with self.assertRaises(ValueError):
                quantize_moe_g128(invalid)
        with self.assertRaises(ValueError):
            quantize_moe_g128(x, promotion_group_size=64)
        with self.assertRaises(ValueError):
            quantize_moe_g128(x, out_q=x.view(torch.float8_e4m3fn)[:, :2304])
        with self.assertRaises(ValueError):
            quantize_moe_g128(x, out_scale=scale[:, :71])

    @unittest.skipUnless(
        os.environ.get("DSV41_QUANT_SCALE_BENCH") == "1", "explicit clean benchmark"
    )
    def test_clean_quant_and_same_grouped_humming_chain_benchmark(self):
        from rtp_llm.models_py.modules.dsv41.moe._sm90_humming_a8 import (
            SM90HummingGroupedA8Linear,
        )

        def time_graph(graph):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            for _ in range(100):
                graph.replay()
            stop.record()
            stop.synchronize()
            return start.elapsed_time(stop) * 10

        # Both actual routed FC widths; identical weight/GEMM/offsets in A/B.
        # Numerical acceptance of native WGMMA remains a separate strict gate.
        for width, output in ((5120, 4608), (2304, 5120)):
            experts = 48  # Actual local expert count for EP8.
            weight = torch.randint(
                -128,
                128,
                (experts, output, width // 2),
                device="cuda",
                dtype=torch.int8,
            )
            raw_scale = torch.randint(
                117,
                128,
                (experts, output, width // 32),
                device="cuda",
                dtype=torch.uint8,
            )
            linear = SM90HummingGroupedA8Linear(
                weight, raw_scale.view(torch.float8_e8m0fnu), promotion_group_size=32
            )
            for rows in (64, 256):
                x = torch.randn((rows, width), device="cuda", dtype=torch.bfloat16)
                # Keep E48 metadata including empty experts; the final offset
                # must equal M even though neither M is divisible by 48.
                active_experts = 32
                offsets = torch.tensor(
                    [
                        min(expert, active_experts) * rows // active_experts
                        for expert in range(experts + 1)
                    ],
                    device="cuda",
                    dtype=torch.int32,
                )
                for policy in ("legacy", "v2", "auto"):
                    with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": policy}):
                        before = lambda: _baseline(x, 32)
                        after = lambda: quantize_moe_g128(x)
                        for a, b in zip(before(), after()):
                            _bits(a, b)
                        for a, b in zip(
                            after(), _finite_oracle(x, 32, use_v2(rows, width))
                        ):
                            _bits(a, b)

                        def gemm(quantizer):
                            return linear.prepared(*quantizer(), offsets)

                        _bits(gemm(before), gemm(after))
                        for chain in (False, True):
                            funcs = (
                                (lambda: gemm(before), lambda: gemm(after))
                                if chain
                                else (before, after)
                            )
                            graphs = []
                            for fn in funcs:
                                for _ in range(3):
                                    fn()
                                graph = torch.cuda.CUDAGraph()
                                with torch.cuda.graph(graph):
                                    fn()
                                graphs.append(graph)
                            samples = [[], []]
                            for iteration in range(5):
                                for arm in ((0, 1) if iteration % 2 == 0 else (1, 0)):
                                    samples[arm].append(time_graph(graphs[arm]))
                            print(
                                "SM90_MOE_QUANT_SCALE_BENCH "
                                + json.dumps(
                                    dict(
                                        rows=rows,
                                        k=width,
                                        n=output,
                                        experts=experts,
                                        policy=policy,
                                        includes_gemm=chain,
                                        old_us=samples[0],
                                        new_us=samples[1],
                                        speedup=statistics.median(samples[0])
                                        / statistics.median(samples[1]),
                                    )
                                ),
                                flush=True,
                            )


if __name__ == "__main__":
    unittest.main()

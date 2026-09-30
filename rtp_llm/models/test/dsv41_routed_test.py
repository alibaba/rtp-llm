"""Independent FP4 expert equations and explicit grouped-provider contracts."""

import contextlib
import sys
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models.test.dsv41_cache_codec_test import load_codec_source
from rtp_llm.models.test.dsv41_expert_test import expert, scalar_swiglu
from rtp_llm.models.test.dsv41_mxfp8_test import linear, quantization_reference

with patch.dict(
    sys.modules,
    {
        "rtp_llm.models_py.modules.dsv4.v41_expert": expert,
        "rtp_llm.models_py.modules.dsv4.v41_mxfp8": linear,
    },
):
    routed = load_codec_source("_dsv41_routed_test", "v41_routed.py")


def routed_weights(experts=3, dim=64, inter=96):
    rng = torch.Generator().manual_seed(4134)
    return {
        "w13_w": torch.randint(
            0, 256, (experts, 2 * inter, dim // 2), dtype=torch.uint8, generator=rng
        ),
        "w2_w": torch.randint(
            0, 256, (experts, dim, inter // 2), dtype=torch.uint8, generator=rng
        ),
        "w13_s": torch.randint(
            120, 125, (experts, 2 * inter, dim // 32), dtype=torch.uint8, generator=rng
        ),
        "w2_s": torch.randint(
            120, 125, (experts, dim, inter // 32), dtype=torch.uint8, generator=rng
        ),
    }


def dequant_fp4(weight, scales):
    raw = weight.view(torch.uint8)
    codes = torch.empty((*raw.shape[:-1], raw.shape[-1] * 2), dtype=torch.long)
    codes[..., ::2], codes[..., 1::2] = raw & 15, raw >> 4
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    return levels[codes] * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)


def quantized_reference_mm(values, scales, weight, weight_scales):
    a = values.float() * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
    b = dequant_fp4(weight, weight_scales)
    return (a @ b.T).bfloat16()


def reference_expert(x, tensors, e, valid, route):
    active = valid & (route != 0)
    raw, scale = quantization_reference(torch.where(active[:, None], x, 0))
    gu = quantized_reference_mm(
        raw.view(torch.float8_e4m3fn), scale, tensors["w13_w"][e], tensors["w13_s"][e]
    )
    gate, up = gu.chunk(2, -1)
    intermediate = scalar_swiglu(gate, up, valid, route)
    raw, scale = quantization_reference(intermediate)
    result = quantized_reference_mm(
        raw.view(torch.float8_e4m3fn), scale, tensors["w2_w"][e], tensors["w2_s"][e]
    )
    return torch.where(active[:, None], result, 0)


def unpack_tma(packed, groups):
    restored = torch.stack([(packed.long() >> (8 * b)) & 255 for b in range(4)], -1)
    return restored.flatten(-2)[..., :groups].to(torch.uint8)


class FakeProvider:
    def __init__(self):
        self.calls = []

    def __call__(self, a, b, out, layout, expected_m=None, **kwargs):
        values, a_scales = a
        weight, b_scales = b
        assert kwargs == dict(
            recipe_a=(1, 32), recipe_b=(1, 32), disable_ue8m0_cast=False
        )
        assert a_scales.dtype == b_scales.dtype == torch.int32
        assert a_scales.stride(-2) == b_scales.stride(-2) == 1
        assert a_scales.stride(-1) % 4 == b_scales.stride(-1) % 4 == 0
        assert weight.dtype == torch.int8
        assert out.dtype == torch.bfloat16
        self.calls.append(
            (values.clone(), a_scales.clone(), layout.clone(), expected_m)
        )
        raw_a = unpack_tma(a_scales, values.shape[-1] // 32)
        raw_b = unpack_tma(b_scales, values.shape[-1] // 32)
        out.fill_(float("nan"))
        for e in range(weight.shape[0]):
            if expected_m is None:
                selected = layout == e
                selected_values = values.view(torch.uint8)[selected].view(
                    torch.float8_e4m3fn
                )
                out[selected] = quantized_reference_mm(
                    selected_values, raw_a[selected], weight[e], raw_b[e]
                )
            else:
                n = int(layout[e])
                out[e, :n] = quantized_reference_mm(
                    values[e, :n], raw_a[e, :n], weight[e], raw_b[e]
                )


@contextlib.contextmanager
def provider_runtime(capturing=False):
    def quant(x, *, backend):
        return linear.quantize_group32(x, backend="eager")

    def swiglu(*args, **kwargs):
        kwargs["backend"] = "eager"
        return expert.quantize_swiglu(*args, **kwargs)

    with patch.object(
        torch.cuda, "device", return_value=contextlib.nullcontext()
    ), patch.object(
        torch.cuda, "is_current_stream_capturing", return_value=capturing
    ), patch.object(
        routed, "quantize_group32", side_effect=quant
    ), patch.object(
        routed, "quantize_swiglu", side_effect=swiglu
    ):
        yield


def mock_provider(op):
    op.backend = "deepgemm"
    op._packed_s13 = expert.pack_group32_scales(op.s13, column_major=True)
    op._packed_s2 = expert.pack_group32_scales(op.s2, column_major=True)
    op._masked = op._contiguous = FakeProvider()
    return op._masked


class V41RoutedTest(unittest.TestCase):
    def test_fp4_oracle_decodes_all_nibbles_and_group_scales(self):
        weight = torch.arange(256, dtype=torch.uint8).reshape(8, 32)
        scales = torch.tensor([[126, 128]], dtype=torch.uint8).expand(8, 2).contiguous()
        values = torch.eye(64).to(torch.float8_e4m3fn)
        actual = routed._fp8_fp4_eager(
            values, torch.full((64, 2), 127, dtype=torch.uint8), weight, scales
        )
        torch.testing.assert_close(
            actual, dequant_fp4(weight, scales).T.bfloat16(), rtol=0, atol=0
        )

    def test_masked_experts_match_checkpoint_equations_with_empty_and_dead_rows(self):
        tensors = routed_weights()
        op = routed.V41RoutedExperts(tensors, backend="eager")
        x = torch.randn(
            3, 5, 64, generator=torch.Generator().manual_seed(17)
        ).bfloat16()
        counts = torch.tensor([5, 0, 3], dtype=torch.int32)
        valid = torch.arange(5)[None] < counts[:, None]
        route = torch.linspace(0, 1.5, 15).reshape(3, 5)
        x[~valid] = float("nan")
        route[~valid] = float("nan")
        expected = torch.stack(
            [reference_expert(x[e], tensors, e, valid[e], route[e]) for e in range(3)]
        )
        out = torch.empty_like(x)
        self.assertIs(op.forward_masked(x, counts, route, out=out), out)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        self.assertEqual(op.w13.data_ptr(), tensors["w13_w"].data_ptr())
        self.assertEqual(op.w2.data_ptr(), tensors["w2_w"].data_ptr())
        self.assertEqual(op.recipe, expert.V41_EXPERT_RECIPE)

    def test_contiguous_and_masked_experts_agree_across_uneven_occupancy(self):
        op = routed.V41RoutedExperts(routed_weights(), backend="eager")
        x = torch.randn(
            3, 5, 64, generator=torch.Generator().manual_seed(63)
        ).bfloat16()
        counts = torch.tensor([5, 0, 2], dtype=torch.int32)
        route = torch.full((3, 5), 0.37)
        expected = op.forward_masked(x, counts, route)
        packed = torch.full((256, 64), float("nan"), dtype=torch.bfloat16)
        valid = torch.zeros(256, dtype=torch.bool)
        weights = torch.full((256,), float("nan"))
        for e, start, n in ((0, 0, 5), (2, 128, 2)):
            packed[start : start + n] = x[e, :n]
            weights[start : start + n] = route[e, :n]
            valid[start : start + n] = True
        actual = op.forward_contiguous(packed, (128, 0, 128), valid, weights)
        torch.testing.assert_close(actual[:5], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[128:130], expected[2, :2], rtol=0, atol=0)
        self.assertTrue(bool((actual[~valid] == 0).all()))

    def test_empty_batches_and_all_zero_routes(self):
        op = routed.V41RoutedExperts(routed_weights(), backend="eager")
        x = torch.empty(3, 0, 64, dtype=torch.bfloat16)
        actual = op.forward_masked(
            x, torch.zeros(3, dtype=torch.int32), torch.empty(3, 0)
        )
        self.assertEqual(actual.shape, x.shape)
        actual = op.forward_contiguous(
            x.reshape(0, 64),
            (0, 0, 0),
            torch.empty(0, dtype=torch.bool),
            torch.empty(0),
        )
        self.assertEqual(actual.shape, (0, 64))
        x = torch.full((3, 2, 64), float("nan"), dtype=torch.bfloat16)
        result = op.forward_masked(
            x, torch.full((3,), 2, dtype=torch.int32), torch.zeros(3, 2)
        )
        self.assertTrue(bool((result == 0).all()))

    def test_weight_admission_rejects_shape_dtype_layout_and_backend(self):
        tensors = routed_weights()
        for key, value in (
            ("w13_w", tensors["w13_w"].float()),
            ("w2_w", tensors["w2_w"][:, :-1]),
            ("w13_s", tensors["w13_s"].int()),
            ("w2_s", tensors["w2_s"][:, :, :1]),
        ):
            with self.assertRaises(ValueError):
                routed.V41RoutedExperts({**tensors, key: value}, backend="eager")
        with self.assertRaises(ValueError):
            routed.V41RoutedExperts(tensors, backend="auto")
        with self.assertRaises(ValueError):
            routed.V41RoutedExperts(tensors, swiglu_limit=-1, backend="eager")

    def test_metadata_and_output_errors_precede_writes(self):
        op = routed.V41RoutedExperts(routed_weights(), backend="eager")
        x, counts, route = (
            torch.ones(3, 2, 64, dtype=torch.bfloat16),
            torch.tensor([1, 0, 2], dtype=torch.int32),
            torch.ones(3, 2),
        )
        out = torch.full_like(x, 42)
        for bad_counts in (
            counts.long(),
            torch.tensor([-1, 0, 2], dtype=torch.int32),
            torch.tensor([1, 0, 3], dtype=torch.int32),
        ):
            with self.assertRaises(ValueError):
                op.forward_masked(x, bad_counts, route, out=out)
            self.assertTrue(bool((out == 42).all()))
        with self.assertRaisesRegex(ValueError, "share"):
            op.forward_masked(x, counts, route, out=x)
        for bad_route in (route.bfloat16(), route[:, :1], -route):
            with self.assertRaises(ValueError):
                op.forward_masked(x, counts, bad_route, out=out)
        for sizes in ((1, 0, 0), (128, 0), (128, -128, 0), [128, 0, 0]):
            with self.assertRaises(ValueError):
                op.forward_contiguous(
                    x.reshape(6, 64),
                    sizes,
                    torch.ones(6, dtype=torch.bool),
                    torch.ones(6),
                )
        self.assertTrue(bool((out == 42).all()))

    def test_masked_provider_contract_and_graph_replay_metadata(self):
        tensors = routed_weights(dim=128, inter=128)
        op = routed.V41RoutedExperts(tensors, backend="eager")
        provider = mock_provider(op)
        x = torch.randn(
            3, 128, 128, generator=torch.Generator().manual_seed(61)
        ).bfloat16()
        counts, route = torch.tensor([2, 0, 1], dtype=torch.int32), torch.full(
            (3, 128), 0.37
        )
        with provider_runtime(capturing=True):
            with self.assertRaisesRegex(RuntimeError, "warmed"):
                op.forward_masked(x, counts, route)
        self.assertEqual(len(provider.calls), 0)
        with provider_runtime():
            actual = op.forward_masked(x, counts, route)
        self.assertEqual(len(provider.calls), 2)
        for e, n in enumerate(counts.tolist()):
            expected = reference_expert(
                x[e, :n], tensors, e, torch.ones(n, dtype=torch.bool), route[e, :n]
            )
            torch.testing.assert_close(actual[e, :n], expected, rtol=0, atol=0)
            self.assertTrue(bool((actual[e, n:] == 0).all()))
        counts.copy_(torch.tensor([0, 2, 0], dtype=torch.int32))
        with provider_runtime(capturing=True):
            replay = op.forward_masked(x, counts, route)
        self.assertTrue(bool((replay[[0, 2]] == 0).all()))
        self.assertTrue(bool(torch.isfinite(replay).all()))
        self.assertTrue(bool((unpack_tma(provider.calls[-1][1], 4)[[0, 2]] == 0).all()))

    def test_contiguous_provider_layout_and_early_router_weight(self):
        tensors = routed_weights(dim=128, inter=128)
        op = routed.V41RoutedExperts(tensors, backend="eager")
        provider = mock_provider(op)
        x = torch.full((256, 128), float("nan"), dtype=torch.bfloat16)
        x[0], x[128] = (
            torch.linspace(-1, 1, 128).bfloat16(),
            torch.linspace(0, 2, 128).bfloat16(),
        )
        valid = torch.zeros(256, dtype=torch.bool)
        valid[[0, 128]] = True
        route = torch.full((256,), float("nan"))
        route[valid] = 0.37
        with provider_runtime():
            result = op.forward_contiguous(x, (128, 0, 128), valid, route)
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(provider.calls[0][2].tolist(), [0] * 128 + [2] * 128)
        for e, row in ((0, 0), (2, 128)):
            expected = reference_expert(
                x[row : row + 1], tensors, e, valid[row : row + 1], route[row : row + 1]
            )
            torch.testing.assert_close(result[row : row + 1], expected, rtol=0, atol=0)
        self.assertTrue(bool((result[~valid] == 0).all()))

    def test_provider_failure_propagates_without_output_mutation(self):
        op = routed.V41RoutedExperts(
            routed_weights(dim=128, inter=128), backend="eager"
        )
        mock_provider(op)
        x = torch.ones(3, 128, 128, dtype=torch.bfloat16)
        out = torch.full_like(x, 42)
        with provider_runtime(), patch.object(
            op, "_masked", side_effect=RuntimeError("provider unavailable")
        ):
            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                op.forward_masked(
                    x, torch.ones(3, dtype=torch.int32), torch.ones(3, 128), out=out
                )
        self.assertTrue(bool((out == 42).all()))
        self.assertFalse(op._warmed)


if __name__ == "__main__":
    unittest.main()

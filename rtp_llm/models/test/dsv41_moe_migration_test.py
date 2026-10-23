"""Portable MoE migration contracts; provider simulation is not GPU qualification."""

import os
import sys
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models.test.dsv41_cache_codec_test import load_codec_source
from rtp_llm.models.test.dsv41_mxfp8_test import linear, quantization_reference
from rtp_llm.models.test.dsv41_routed_test import (
    mock_provider,
    provider_runtime,
    routed,
    routed_weights,
    unpack_tma,
)

with patch.dict(sys.modules, {"rtp_llm.models_py.modules.dsv4.v41_mxfp8": linear}):
    wire = load_codec_source("_dsv41_return_test", "v41_return.py")


class V41ReturnTest(unittest.TestCase):
    def test_exact_return_bytes_with_strides_and_bf16_rounding(self):
        rng = torch.Generator().manual_seed(41723)
        for dim, rows in ((32, 0), (128, 1), (5120, 3), (128, 129)):
            partial = torch.randn(rows, dim * 2, generator=rng)[:, ::2]
            if rows:
                partial[0, :4] = torch.tensor([-0.0, 1e-8, 1.00390625, -1.00390625])
            for mode in ("fp32", "mxfp8"):
                layout = wire.V41ReturnLayout(dim, mode)
                actual = wire.pack_return(partial, layout, backend="eager")
                if not rows:
                    expected = torch.empty(0, layout.payload_bytes, dtype=torch.uint8)
                elif mode == "fp32":
                    expected = partial.contiguous().view(torch.uint8)
                elif rows:
                    values, scales = quantization_reference(partial.bfloat16())
                    expected = torch.cat((values, scales), dim=-1)
                self.assertTrue(actual.is_contiguous())
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_return_identity_and_invalid_contracts(self):
        fp32 = wire.V41ReturnLayout(5120)
        mxfp8 = wire.V41ReturnLayout(5120, "mxfp8")
        self.assertEqual(fp32.identity["row_bytes"], 20480)
        self.assertEqual(mxfp8.identity["row_bytes"], 5280)
        self.assertNotEqual(fp32.identity["recipe"], mxfp8.identity["recipe"])
        for dim, mode in ((0, "fp32"), (33, "fp32"), (128, "bf16")):
            with self.assertRaises(ValueError):
                wire.V41ReturnLayout(dim, mode)
        for partial in (torch.ones(2, 64), torch.ones(2, 128).bfloat16()):
            with self.assertRaises(ValueError):
                wire.pack_return(partial, wire.V41ReturnLayout(128), backend="eager")
        with self.assertRaisesRegex(ValueError, "finite"):
            wire.pack_return(
                torch.full((1, 128), float("nan")),
                wire.V41ReturnLayout(128, "mxfp8"),
                backend="eager",
            )


class V41PrefillIntegrationTest(unittest.TestCase):
    def make_op(self):
        with patch.dict(
            os.environ,
            {
                "DSV41_MOE_LOCAL_FUSE": "1",
                "DSV41_MOE_MASK_SKIP": "1",
                "DSV41_SWIGLU_FUSE": "1",
            },
        ):
            op = routed.V41RoutedExperts(
                routed_weights(dim=128, inter=128), backend="eager"
            )
        provider = mock_provider(op)
        return op, provider

    def inputs(self, poison):
        valid = torch.zeros(256, dtype=torch.bool)
        valid[[0, 2, 128]] = True
        raw = torch.full((256, 128), 127 if poison else 0, dtype=torch.uint8)
        scales = torch.full((256, 4), 255 if poison else 0, dtype=torch.uint8)
        raw[valid] = 48
        scales[valid] = 123
        weights = torch.zeros(256)
        weights[valid] = 0.37
        return raw.view(torch.float8_e4m3fn), (128, 0, 128), valid, weights, scales

    def test_external_poison_is_masked_and_gemm_indices_reused(self):
        op, provider = self.make_op()
        x, segments, valid, weights, scales = self.inputs(poison=True)
        layouts = []

        def record(*args, **kwargs):
            layouts.append(args[3])
            return provider(*args, **kwargs)

        op._contiguous = record
        with provider_runtime(), patch.object(
            torch, "repeat_interleave", wraps=torch.repeat_interleave
        ) as repeat:
            actual = op.forward_contiguous(
                x, segments, valid, weights, input_scales=scales
            )
        self.assertEqual(repeat.call_count, 1)
        self.assertEqual(repeat.call_args.kwargs["output_size"], 256)
        self.assertIs(layouts[0], layouts[1])
        torch.testing.assert_close(
            layouts[0], torch.tensor([0] * 128 + [2] * 128, dtype=torch.int32)
        )
        self.assertTrue(bool((actual[~valid] == 0).all()))
        self.assertTrue(bool(torch.isfinite(actual).all()))
        values, packed, _, _ = provider.calls[0]
        self.assertTrue(bool((values.view(torch.uint8)[~valid] == 0).all()))
        self.assertTrue(bool((unpack_tma(packed, 4)[~valid] == 0).all()))

    def test_owned_padding_matches_control_on_all_consumed_rows(self):
        op, provider = self.make_op()
        x, segments, valid, weights, scales = self.inputs(poison=False)
        with provider_runtime():
            expected = op.forward_contiguous(
                x, segments, valid, weights, input_scales=scales
            )
            actual = op.forward_contiguous(
                x, segments, valid, weights, input_scales=scales, _owned_padding=True
            )
        torch.testing.assert_close(actual[valid], expected[valid], rtol=0, atol=0)
        torch.testing.assert_close(
            provider.calls[2][0].view(torch.uint8), x.view(torch.uint8), rtol=0, atol=0
        )
        with self.assertRaisesRegex(ValueError, "owned received"):
            op.forward_contiguous(
                x,
                segments,
                valid,
                weights,
                input_scales=scales,
                out=torch.empty(x.shape, dtype=torch.bfloat16),
                _owned_padding=True,
            )
        with self.assertRaisesRegex(ValueError, "owned received"):
            op.forward_contiguous(
                x.float().bfloat16(), segments, valid, weights, _owned_padding=True
            )

    def test_selectors_are_frozen_at_owner_creation(self):
        op, _ = self.make_op()
        with patch.dict(
            os.environ,
            {
                "DSV41_MOE_LOCAL_FUSE": "0",
                "DSV41_SWIGLU_FUSE": "0",
                "DSV41_MOE_MASK_SKIP": "0",
            },
        ):
            self.assertTrue(op.prefill_local_fuse)
            self.assertTrue(op.prefill_swiglu_fuse)
            self.assertTrue(op.prefill_mask_skip)


if __name__ == "__main__":
    unittest.main()

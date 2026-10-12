"""Dispatch wire bytes, recipe identity, padding, aliases and rejection paths."""

import dataclasses
import struct
import sys
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models.test.dsv41_cache_codec_test import load_codec_source
from rtp_llm.models.test.dsv41_expert_test import expert
from rtp_llm.models.test.dsv41_mxfp8_test import linear, quantization_reference

with patch.dict(
    sys.modules,
    {
        "rtp_llm.models_py.modules.dsv4.v41_expert": expert,
        "rtp_llm.models_py.modules.dsv4.v41_mxfp8": linear,
    },
):
    dispatch = load_codec_source("_dsv41_dispatch_test", "v41_dispatch.py")


class V41DispatchTest(unittest.TestCase):
    def test_unpack_preserves_bits_and_removes_scale_padding(self):
        for hidden in (32, 96, 5120):
            layout = dispatch.V41DispatchLayout(hidden, 2, 384)
            payload = layout.allocate(3, device="cpu")
            values, packed, weights, ids = layout.views(payload)
            values.view(torch.uint8).copy_(
                torch.arange(hidden, dtype=torch.int64).remainder(256).byte()
            )
            packed.copy_(torch.arange(layout.packed_scales, dtype=torch.int32))
            weights.copy_(torch.tensor([0.37, 0.5]))
            ids.copy_(torch.tensor([8, 12], dtype=torch.int32))
            q, s, w, i = layout.unpack(payload)
            for field in (q, s, w, i):
                self.assertTrue(field.is_contiguous())
            torch.testing.assert_close(
                q.view(torch.uint8), values.view(torch.uint8), rtol=0, atol=0
            )
            torch.testing.assert_close(
                s,
                packed.contiguous().view(torch.uint8)[:, : hidden // 32],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(w, weights, rtol=0, atol=0)
            torch.testing.assert_close(i, ids, rtol=0, atol=0)
            self.assertEqual(layout.unpack(payload[:0])[1].shape, (0, hidden // 32))

    def test_received_route_values_reject_corruption(self):
        layout = dispatch.V41DispatchLayout(64, 2, 16)
        for weights, ids in (
            ([1, -1], [1, 2]),
            ([1, float("nan")], [1, 2]),
            ([1, 1], [1, 16]),
            ([1, 1], [1, 1]),
        ):
            payload = layout.allocate(1, device="cpu").zero_()
            _, _, w, i = layout.views(payload)
            w.copy_(torch.tensor(weights))
            i.copy_(torch.tensor(ids))
            with self.assertRaises(ValueError):
                layout.unpack(payload)

    def test_real_geometry_recipe_identity_and_invalid_layouts(self):
        layout = dispatch.V41DispatchLayout(5120, 6, 384)
        self.assertEqual(
            (
                layout.packed_scales,
                layout.weight_offset,
                layout.index_offset,
                layout.payload_bytes,
            ),
            (40, 5280, 5304, 5328),
        )
        self.assertEqual(hash(layout), hash(dispatch.V41DispatchLayout(5120, 6, 384)))
        self.assertNotEqual(layout, dataclasses.replace(layout, experts=128))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            layout.hidden = 7168
        for args in ((31, 1, 2), (0, 1, 2), (32, 0, 2), (32, 3, 2), (32, 1, 2**31)):
            with self.assertRaises(ValueError):
                dispatch.V41DispatchLayout(*args)
        with self.assertRaisesRegex(ValueError, "recipe"):
            dataclasses.replace(layout, recipe="group128")

    def test_wire_bytes_match_independent_quantizer_and_struct_metadata(self):
        for hidden in (32, 96, 5120):
            layout = dispatch.V41DispatchLayout(hidden, 6, 384)
            x = torch.randn(
                3, hidden, generator=torch.Generator().manual_seed(41)
            ).bfloat16()
            weights = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]).repeat(3, 1)
            ids = torch.tensor([[383, 1, 10, 270, 88, 97]], dtype=torch.int64).repeat(
                3, 1
            )
            actual = dispatch.pack_dispatch(
                x,
                weights,
                ids,
                torch.ones(3, dtype=torch.bool),
                layout,
                backend="eager",
            )
            values, scales = quantization_reference(x)
            for row in range(3):
                expected = bytes(values[row].tolist()) + bytes(scales[row].tolist())
                expected += bytes(layout.packed_scales * 4 - hidden // 32)
                expected += struct.pack("<6f", *weights[row].tolist())
                expected += struct.pack("<6i", *ids[row].tolist())
                self.assertEqual(bytes(actual[row].tolist()), expected)

    def test_fake_zero_route_and_capacity_padding_have_canonical_bytes(self):
        layout = dispatch.V41DispatchLayout(64, 2, 384)
        x = torch.full((3, 64), float("nan"), dtype=torch.bfloat16)
        x[0] = 1
        weights = torch.tensor([[0.37, 0], [float("nan"), -1], [0, 0]])
        ids = torch.tensor([[8, 2**40], [2**40, -6], [-8, 2**40]])
        valid = torch.tensor([True, False, True])
        out = torch.full((5, layout.payload_bytes), 173, dtype=torch.uint8)
        self.assertIs(
            dispatch.pack_dispatch(
                x, weights, ids, valid, layout, out=out, backend="eager"
            ),
            out,
        )
        q, scales, w, i = layout.views(out)
        self.assertTrue(bool((q[1:].view(torch.uint8) == 0).all()))
        self.assertTrue(bool((scales[1:] == 0).all()))
        self.assertTrue(bool((w[1:] == 0).all()))
        self.assertTrue(bool((i[1:] == -1).all()))
        self.assertEqual(i[0].tolist(), [8, -1])
        self.assertEqual(w[0, 1], 0)
        self.assertEqual(scales.contiguous().view(torch.uint8)[0, 2:].tolist(), [0, 0])

    def test_views_share_storage_and_preserve_wire_row_strides(self):
        layout = dispatch.V41DispatchLayout(5120, 6, 384)
        payload = layout.allocate(7, device="cpu")
        q, scale, weight, ids = layout.views(payload)
        self.assertEqual(q.stride(), (5328, 1))
        for view in (scale, weight, ids):
            self.assertEqual(view.stride(0), 1332)
            self.assertEqual(
                view.untyped_storage().data_ptr(), payload.untyped_storage().data_ptr()
            )
        ids.fill_(257)
        self.assertEqual(payload[4, -4:].tolist(), [1, 1, 0, 0])

    def test_empty_inputs_initialize_full_capacity_and_zero_sized_views(self):
        layout = dispatch.V41DispatchLayout(32, 1, 2)
        for capacity in (0, 3):
            out = layout.allocate(capacity, device="cpu")
            result = dispatch.pack_dispatch(
                torch.empty(0, 32, dtype=torch.bfloat16),
                torch.empty(0, 1),
                torch.empty(0, 1, dtype=torch.int32),
                torch.empty(0, dtype=torch.bool),
                layout,
                out=out,
                backend="eager",
            )
            q, s, w, i = layout.views(result)
            self.assertEqual(result.shape, (capacity, layout.payload_bytes))
            self.assertTrue(bool((q.view(torch.uint8) == 0).all()))
            self.assertTrue(bool((s == 0).all() and (w == 0).all() and (i == -1).all()))

    def test_invalid_payloads_values_and_aliases_leave_output_untouched(self):
        layout = dispatch.V41DispatchLayout(64, 2, 384)
        x = torch.ones(2, 64, dtype=torch.bfloat16)
        w, ids, valid = (
            torch.ones(2, 2),
            torch.tensor([[1, 2], [3, 4]]),
            torch.ones(2, dtype=torch.bool),
        )
        out = torch.full((2, layout.payload_bytes), 173, dtype=torch.uint8)
        for bad_w, bad_ids in (
            (-w, ids),
            (w * float("nan"), ids),
            (w, ids * 0),
            (w, ids + 384),
            (w, ids + 2**32),
        ):
            with self.assertRaises(ValueError):
                dispatch.pack_dispatch(
                    x, bad_w, bad_ids, valid, layout, out=out, backend="eager"
                )
            self.assertTrue(bool((out == 173).all()))
        for bad in (out[:, :-4], out.T, out.int()):
            with self.assertRaises(ValueError):
                layout.views(bad)
        shared = torch.empty(2 * 64 * 2 + 2 * layout.payload_bytes, dtype=torch.uint8)
        aliased_x = shared[:256].view(torch.bfloat16).reshape(2, 64)
        aliased_out = shared[256:].reshape(2, layout.payload_bytes)
        with self.assertRaisesRegex(ValueError, "share"):
            dispatch.pack_dispatch(
                aliased_x, w, ids, valid, layout, out=aliased_out, backend="eager"
            )
        with self.assertRaises(ValueError):
            dispatch.pack_dispatch(
                x, w, ids, valid, layout, out=out[:1], backend="eager"
            )
        with self.assertRaises(ValueError):
            dispatch.pack_dispatch(x, w, ids, valid, layout, backend="triton")

    def test_route_slot_and_cp_row_permutations_preserve_payload_meaning(self):
        layout = dispatch.V41DispatchLayout(96, 2, 384)
        x = torch.randn(4, 96).bfloat16()
        w, ids, valid = (
            torch.rand(4, 2),
            torch.tensor([[1, 2], [5, 3], [2, 9], [6, 7]]),
            torch.tensor([True, False, True, True]),
        )
        expected = dispatch.pack_dispatch(x, w, ids, valid, layout, backend="eager")
        order = torch.tensor([2, 0, 3, 1])
        actual = dispatch.pack_dispatch(
            x[order], w[order], ids[order], valid[order], layout, backend="eager"
        )
        torch.testing.assert_close(actual[order.argsort()], expected, rtol=0, atol=0)
        swapped = dispatch.pack_dispatch(
            x, w.flip(1), ids.flip(1), valid, layout, backend="eager"
        )
        original_views, swapped_views = layout.views(expected), layout.views(swapped)
        torch.testing.assert_close(
            original_views[0].view(torch.uint8),
            swapped_views[0].view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(original_views[1], swapped_views[1], rtol=0, atol=0)
        for original, changed in zip(original_views[2:], swapped_views[2:]):
            torch.testing.assert_close(original.flip(1), changed, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

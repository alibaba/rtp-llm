"""CPU-only packed layout/isolation tests; real-weight CUDA byte gate is separate.

Run through the repository Bazel target, not a direct Python GPU entry point.
The small sparse deterministic model is intentional: these tests verify the
packing contract and exact outputs for that fixture, not general GPU rounding.
"""

import unittest
from unittest import mock

import torch

from rtp_llm.multimodal.multimodal_mixins.deepseek_v41 import packed_vision
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_vision import (
    Aligner,
    ViT,
)


class PackedVisionTest(unittest.TestCase):
    def setUp(self):
        self.inference = torch.inference_mode()
        self.inference.__enter__()
        self.addCleanup(self.inference.__exit__, None, None, None)
        self.config = {
            "vision_patch_size": 2,
            "vision_dim": 16,
            "vision_n_heads": 2,
            "vision_n_layers": 2,
            "vision_inter_dim": 16,
            "vision_downsample_ratio": 2,
            "vision_rope_theta": 10000.0,
            "hidden_size": 16,
        }
        self.vision = ViT(self.config).eval().requires_grad_(False)
        self.aligner = Aligner(self.config).eval().requires_grad_(False)
        # Sparse dyadic projections avoid asserting arbitrary CPU BLAS rounding
        # equivalence; nonzero Q/K still exercise geometry-dependent RoPE.
        for parameter in list(self.vision.parameters()) + list(
            self.aligner.parameters()
        ):
            parameter.zero_()
        self.vision.norm.weight.fill_(1)
        for row in range(16):
            self.vision.patch_embed.proj.weight[row, row % 12] = 0.25
            self.vision.patch_embed.proj.bias[row] = (row % 3) / 8
        for block in self.vision.blocks:
            block.norm1.weight.fill_(1)
            block.norm2.weight.fill_(1)
            for row in range(16):
                block.attn.wqkv.weight[row, row] = 0.125
                block.attn.wqkv.weight[16 + row, row] = 0.0625
                block.attn.wqkv.weight[32 + row, row] = 0.5
                block.attn.wo.weight[row, row] = 0.25
                # Nontrivial packed MLP, with one source element per output.
                block.mlp.w1.weight[row, row] = 0.125
                block.mlp.w1.weight[16 + row, row] = 0.25
                block.mlp.w2.weight[row, row] = 0.125
        for row in range(16):
            self.aligner.w1.weight[row, row * 4] = 0.25
            self.aligner.w2.weight[row, row] = 0.5

    def patches(self, grids):
        result = []
        for index, (height, width) in enumerate(grids):
            values = torch.arange(height * width * 12, dtype=torch.float32)
            result.append(
                ((values % 29 - 14) / 8 + index / 4).reshape(height * width, 3, 2, 2)
            )
        return result

    def byte_equal(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(
            torch.equal(
                actual.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        )

    def sequential_reference(self, patches, grids):
        return [
            self.aligner(self.vision(part, height, width), height, width)
            for part, (height, width) in zip(patches, grids)
        ]

    def compare(self, grids):
        patches = self.patches(grids)
        reference = self.sequential_reference(patches, grids)
        actual = packed_vision.forward_packed(self.vision, self.aligner, patches, grids)
        self.assertEqual(len(actual), len(grids))
        for output, expected, (height, width) in zip(actual, reference, grids):
            self.assertEqual(
                output.shape, (((height + 1) // 2) * ((width + 1) // 2), 16)
            )
            self.assertTrue(torch.isfinite(output).all())
            self.byte_equal(output, expected)
        return patches, actual

    def test_empty_and_singleton(self):
        self.assertEqual(
            packed_vision.forward_packed(self.vision, self.aligner, [], []), []
        )
        self.assertEqual(self.sequential_reference([], []), [])
        self.compare([(3, 5)])

    def test_homogeneous_sdpa_descriptor_and_byte_equivalence(self):
        for batch, length, heads in ((3, 16, 2), (1, 5, 2), (3, 1, 2), (3, 7, 1)):
            qkv = (
                torch.arange(
                    batch * length * 3 * heads * 4, dtype=torch.float32
                ).reshape(batch * length, 3 * heads * 4)
                / 1024
            )
            tensors = [
                value.view(batch * length, heads, 4) for value in qkv.chunk(3, dim=-1)
            ]
            reference_inputs = [
                torch.stack(
                    [part.transpose(0, 1) for part in value.split(length)], dim=0
                )
                for value in tensors
            ]
            expected = packed_vision.F.scaled_dot_product_attention(*reference_inputs)
            expected = torch.cat(
                [row.transpose(0, 1).reshape(length, -1) for row in expected], dim=0
            )
            original = packed_vision.F.scaled_dot_product_attention
            with mock.patch.object(
                packed_vision.F, "scaled_dot_product_attention", wraps=original
            ) as sdpa:
                actual = packed_vision._packed_attention(
                    *tensors, [(1, length)] * batch, [length] * batch
                )
                for got, want in zip(sdpa.call_args.args, reference_inputs):
                    self.assertEqual(got.shape, want.shape)
                    self.assertEqual(got.stride(), want.stride())
                    self.byte_equal(got, want)
            self.byte_equal(actual, expected)

    def test_same_grid_true_attention_batch_and_shared_dense_calls(self):
        grids = [(4, 4)] * 3
        calls = {"patch": 0, "qkv": 0, "mlp": 0, "aligner_w1": 0, "aligner_w2": 0}
        handles = []
        for key, module in (
            ("patch", self.vision.patch_embed),
            ("qkv", self.vision.blocks[0].attn.wqkv),
            ("mlp", self.vision.blocks[0].mlp),
            ("aligner_w1", self.aligner.w1),
            ("aligner_w2", self.aligner.w2),
        ):

            def record(module, args, output, key=key):
                calls[key] += 1

            handles.append(module.register_forward_hook(record))
        try:
            original = packed_vision.F.scaled_dot_product_attention
            with mock.patch.object(
                packed_vision.F, "scaled_dot_product_attention", wraps=original
            ) as sdpa:
                packed_vision.forward_packed(
                    self.vision, self.aligner, self.patches(grids), grids
                )
                self.assertEqual(sdpa.call_count, 2)
                for call in sdpa.call_args_list:
                    for tensor in call.args:
                        self.assertEqual(tensor.shape, (3, 2, 16, 8))
            self.assertEqual(
                calls,
                {"patch": 1, "qkv": 1, "mlp": 1, "aligner_w1": 3, "aligner_w2": 1},
            )
        finally:
            for handle in handles:
                handle.remove()
        self.compare(grids)

    def test_mixed_grid_groups_padding_and_equal_area_different_geometry(self):
        grids = [(2, 6), (3, 4), (3, 5), (2, 6)]
        original = packed_vision.F.scaled_dot_product_attention
        with mock.patch.object(
            packed_vision.F, "scaled_dot_product_attention", wraps=original
        ) as sdpa:
            packed_vision.forward_packed(
                self.vision, self.aligner, self.patches(grids), grids
            )
            self.assertEqual(
                [tuple(call.args[0].shape) for call in sdpa.call_args_list],
                [(2, 2, 12, 8), (1, 2, 12, 8), (1, 2, 15, 8)] * 2,
            )
        self.compare(grids)

    def test_image_isolation_and_permutation(self):
        grids = [(4, 4)] * 3
        patches, actual = self.compare(grids)
        self.assertFalse(torch.equal(actual[0], actual[1]))
        changed = list(patches)
        changed[1] = patches[1] * -3 + 2
        changed_output = packed_vision.forward_packed(
            self.vision, self.aligner, changed, grids
        )
        self.byte_equal(actual[0], changed_output[0])
        self.byte_equal(actual[2], changed_output[2])
        self.assertFalse(torch.equal(actual[1], changed_output[1]))
        order = [2, 0, 1]
        permuted = packed_vision.forward_packed(
            self.vision,
            self.aligner,
            [patches[i] for i in order],
            [grids[i] for i in order],
        )
        for output, index in zip(permuted, order):
            self.byte_equal(output, actual[index])

    def test_forward_does_not_revalidate_model_or_prepared_metadata(self):
        grids = [(3, 5), (3, 5)]
        patches = self.patches(grids)
        with mock.patch.object(
            self.vision, "modules", side_effect=AssertionError("model traversal")
        ), mock.patch.object(
            self.vision, "parameters", side_effect=AssertionError("parameter traversal")
        ):
            outputs = packed_vision.forward_packed(
                self.vision, self.aligner, patches, grids
            )
        self.assertEqual(len(outputs), 2)
        self.assertEqual(
            [tuple(output.shape) for output in outputs], [(6, 16), (6, 16)]
        )


class PackedRotaryLaunchTest(unittest.TestCase):
    """CPU/meta mocks validate launch partitioning, never substitute GPU parity."""

    def test_actual_cuda_grid_boundary_and_large_tail(self):
        limit = packed_vision._MAX_ROPE_ROWS_PER_LAUNCH
        self.assertEqual(limit, 524280)
        for rows, expected in (
            (1, [1]),
            (limit - 1, [limit - 1]),
            (limit, [limit]),
            (limit + 1, [limit, 1]),
            (2 * limit, [limit, limit]),
            (2 * limit + 3, [limit, limit, 3]),
        ):
            with self.subTest(rows=rows):
                # Meta tensors test the real limit without allocating large
                # CPU buffers or invoking a CUDA driver/model/kernel.
                q, k = [torch.empty((rows, 1, 2), device="meta") for _ in range(2)]
                cos, sin = [torch.empty((rows, 1, 1), device="meta") for _ in range(2)]
                attention = mock.Mock()
                attention._apply_rotary.side_effect = lambda a, b, c, s: (a, b)
                with mock.patch.object(
                    packed_vision.torch, "cat", wraps=torch.cat
                ) as cat:
                    actual_q, actual_k = packed_vision._packed_rotary(
                        attention, q, k, cos, sin
                    )
                    self.assertEqual(cat.call_count, 0 if rows <= limit else 2)
                self.assertEqual(
                    [
                        call.args[0].shape[0]
                        for call in attention._apply_rotary.call_args_list
                    ],
                    expected,
                )
                self.assertEqual(actual_q.shape, q.shape)
                self.assertEqual(actual_k.shape, k.shape)
                if rows <= limit:
                    self.assertIs(actual_q, q)
                    self.assertIs(actual_k, k)
                    for actual, original in zip(
                        attention._apply_rotary.call_args.args, (q, k, cos, sin)
                    ):
                        self.assertIs(actual, original)

    def test_chunked_qk_and_tables_keep_absolute_rows_strides_and_v(self):
        rows, chunk = 19, 8
        qkv = (
            (torch.arange(rows * 3072, dtype=torch.float32) / 128)
            .reshape(rows, 3072)
            .to(torch.bfloat16)
        )
        original = qkv.clone()
        q, k, v = [part.view(rows, 16, 64) for part in qkv.chunk(3, dim=-1)]
        v_before = v.clone()
        cos = torch.arange(rows * 32, dtype=torch.float32).reshape(rows, 1, 32) / 16
        sin = -cos - 0.5
        spans = []

        def rotate(a, b, c, s):
            start = sum(length for _, length in spans)
            count = a.shape[0]
            spans.append((start, count))
            self.assertEqual(a.stride(), (3072, 64, 1))
            self.assertEqual(b.stride(), (3072, 64, 1))
            self.assertEqual(
                a.untyped_storage().data_ptr(), b.untyped_storage().data_ptr()
            )
            self.assertEqual(b.data_ptr() - a.data_ptr(), 1024 * a.element_size())
            self.assertEqual(a.storage_offset(), q.storage_offset() + start * 3072)
            self.assertEqual(b.storage_offset(), k.storage_offset() + start * 3072)
            self.assertTrue(c.is_contiguous())
            self.assertTrue(s.is_contiguous())
            self.assertTrue(torch.equal(c, cos[start : start + count]))
            self.assertTrue(torch.equal(s, sin[start : start + count]))
            return (
                (a.float() + c.repeat(1, 1, 2)).to(a.dtype),
                (b.float() - s.repeat(1, 1, 2)).to(b.dtype),
            )

        attention = mock.Mock()
        attention._apply_rotary.side_effect = rotate
        with mock.patch.object(packed_vision, "_MAX_ROPE_ROWS_PER_LAUNCH", chunk):
            actual_q, actual_k = packed_vision._packed_rotary(attention, q, k, cos, sin)
        expected_q = (q.float() + cos.repeat(1, 1, 2)).to(q.dtype)
        expected_k = (k.float() - sin.repeat(1, 1, 2)).to(k.dtype)
        self.assertEqual(spans, [(0, 8), (8, 8), (16, 3)])
        self.assertTrue(
            torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
        )
        self.assertTrue(
            torch.equal(actual_k.view(torch.uint8), expected_k.view(torch.uint8))
        )
        self.assertTrue(
            torch.equal(
                v.contiguous().view(torch.uint8),
                v_before.contiguous().view(torch.uint8),
            )
        )
        self.assertTrue(torch.equal(qkv.view(torch.uint8), original.view(torch.uint8)))

    def test_chunk_failure_propagates_without_retry_or_partial_concat(self):
        rows = 19
        q, k = [torch.empty((rows, 1, 2), device="meta") for _ in range(2)]
        cos, sin = [torch.empty((rows, 1, 1), device="meta") for _ in range(2)]
        error = RuntimeError("rotary launch failed")
        attention = mock.Mock()
        attention._apply_rotary.side_effect = [(q[:8], k[:8]), error]
        with mock.patch.object(
            packed_vision, "_MAX_ROPE_ROWS_PER_LAUNCH", 8
        ), mock.patch.object(
            packed_vision.torch,
            "cat",
            side_effect=AssertionError("must not publish partial output"),
        ) as cat:
            with self.assertRaises(RuntimeError) as caught:
                packed_vision._packed_rotary(attention, q, k, cos, sin)
            self.assertIs(caught.exception, error)
            self.assertEqual(attention._apply_rotary.call_count, 2)
            cat.assert_not_called()


if __name__ == "__main__":
    unittest.main()

"""Exact backend-owned DeepSeek-V4.1 canonical span assembly tests."""

import unittest

import torch
from torch import nn

from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_processor import (
    V41ImageProcessorConfig,
    num_image_tokens,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision import (
    DeepSeekV41VisionEmbedding,
)


def make_adapter(hidden=17, dtype=torch.bfloat16, device="cpu"):
    adapter = object.__new__(DeepSeekV41VisionEmbedding)
    nn.Module.__init__(adapter)
    adapter.processor_config = V41ImageProcessorConfig()
    values = torch.arange(hidden, dtype=torch.float32, device=device)
    adapter.image_start = nn.Parameter((values + 11).to(dtype), requires_grad=False)
    adapter.image_newline = nn.Parameter((values - 31).to(dtype), requires_grad=False)
    adapter.image_end = nn.Parameter((values + 71).to(dtype), requires_grad=False)
    return adapter


def reference_span(adapter, aligned, height, width):
    rows = []
    for row in aligned.reshape(height, width, -1):
        rows.extend(row.unbind(0))
        rows.append(adapter.image_newline)
    return torch.stack([adapter.image_start, *rows, adapter.image_end]).contiguous()


class CanonicalSpanAssemblyTest(unittest.TestCase):
    def assert_bits_equal(self, expected, actual):
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected.dtype, actual.dtype)
        self.assertEqual(expected.device, actual.device)
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(
            torch.equal(
                expected.contiguous().view(torch.uint8), actual.view(torch.uint8)
            )
        )

    def test_rectangular_grids_dtypes_and_hidden_widths(self):
        for height, width in ((1, 1), (1, 7), (4, 5), (39, 39), (59, 149)):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for hidden in (1, 17, 64):
                    with self.subTest(grid=(height, width), dtype=dtype, hidden=hidden):
                        adapter = make_adapter(hidden, dtype)
                        aligned = torch.randn(
                            (height * width, hidden),
                            generator=torch.Generator().manual_seed(23),
                        ).to(dtype)
                        expected = reference_span(adapter, aligned, height, width)
                        actual = adapter._assemble_canonical_span(
                            aligned, height, width
                        )
                        self.assertEqual(
                            actual.shape[0], num_image_tokens(height, width)
                        )
                        self.assert_bits_equal(expected, actual)

    def test_noncontiguous_input_and_special_float_bits(self):
        adapter = make_adapter(8, torch.float32)
        storage = torch.empty((6, 16), dtype=torch.float32)
        aligned = storage[:, ::2]
        self.assertFalse(aligned.is_contiguous())
        bits = aligned.contiguous().view(torch.int32)
        bits.flatten()[:6].copy_(
            torch.tensor(
                [
                    0x00000000,
                    -0x80000000,
                    0x7F800000,
                    -0x00800000,
                    0x7FC00001,
                    0x7FA00001,
                ],
                dtype=torch.int32,
            )
        )
        aligned.copy_(bits.view(torch.float32))

        expected = reference_span(adapter, aligned, 2, 3)
        actual = adapter._assemble_canonical_span(aligned, 2, 3)
        self.assert_bits_equal(expected, actual)

    def test_each_result_owns_storage(self):
        adapter = make_adapter(4)
        first_input = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
        first = adapter._assemble_canonical_span(first_input, 1, 2)
        snapshot = first.clone()
        second = adapter._assemble_canonical_span(torch.full_like(first_input, 9), 1, 2)
        first_input.fill_(123)
        second.fill_(456)
        self.assertTrue(torch.equal(first, snapshot))
        self.assertNotEqual(first.data_ptr(), second.data_ptr())

    def test_shape_mismatch_is_rejected(self):
        adapter = make_adapter(4)
        with self.assertRaises(RuntimeError):
            adapter._assemble_canonical_span(torch.zeros((5, 4)), 2, 3)


if __name__ == "__main__":
    unittest.main()

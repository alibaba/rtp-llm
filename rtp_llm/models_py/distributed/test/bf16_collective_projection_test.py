"""Contracts that distinguish BF16 projection fusion from FP8 producers."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.factory.linear.impl.cuda.bf16_comm_fusion import (
    Bf16CollectiveProjection,
    prepare_bf16_projection,
)


class Bf16CollectiveProjectionTest(unittest.TestCase):
    def test_physical_padding_preserves_all_logical_branches(self):
        weight = torch.arange(6284 * 8, dtype=torch.float32).reshape(6284, 8)
        projection = SimpleNamespace(weight=(weight % 17).to(torch.bfloat16), bias=None)
        prepared = prepare_bf16_projection(projection, alignment=128)
        self.assertEqual(prepared.shape, (8, 6400))
        self.assertTrue(prepared.is_contiguous())
        self.assertEqual(projection.logical_out_features, 6284)
        torch.testing.assert_close(
            prepared[:, :6284], (weight % 17).to(torch.bfloat16).T, rtol=0, atol=0
        )
        self.assertEqual(prepared[:, 6284:].count_nonzero(), 0)
        x = torch.ones((3, 8), dtype=torch.bfloat16)
        full = x @ prepared
        reference = x @ projection.weight.T
        offset = 0
        for width in (4608, 1536, 128, 12):
            torch.testing.assert_close(
                full[:, offset : offset + width],
                reference[:, offset : offset + width],
                rtol=0,
                atol=0,
            )
            offset += width

    def test_non_bf16_or_biased_weights_are_rejected_during_preparation(self):
        for dtype in (torch.float32, torch.float8_e4m3fn):
            projection = SimpleNamespace(weight=torch.ones((8, 16)).to(dtype), bias=None)
            with self.assertRaisesRegex(ValueError, "bias-free BF16"):
                prepare_bf16_projection(projection)
        projection = SimpleNamespace(
            weight=torch.ones((8, 16), dtype=torch.bfloat16), bias=torch.ones(8)
        )
        with self.assertRaisesRegex(ValueError, "bias-free BF16"):
            prepare_bf16_projection(projection)

    def test_decode_verify_and_capacity_keep_existing_execution_path(self):
        collective = object.__new__(Bf16CollectiveProjection)
        collective.world_size, collective.max_m = 8, 65536
        inputs = SimpleNamespace(is_prefill=False)
        meta = SimpleNamespace(is_target_verify=False)
        self.assertFalse(collective.eligible(65536, inputs, meta, gathered=True))
        inputs.is_prefill = True
        self.assertTrue(collective.eligible(8192, inputs, meta))
        self.assertTrue(collective.eligible(65536, inputs, meta, gathered=True))
        meta.is_target_verify = True
        self.assertFalse(collective.eligible(65536, inputs, meta, gathered=True))
        meta.is_target_verify = False
        self.assertFalse(collective.eligible(65544, inputs, meta, gathered=True))


if __name__ == "__main__":
    unittest.main()

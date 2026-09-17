"""CPU oracle tests; no collective or GPU is needed."""

import unittest
from types import ModuleType
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.topk_bt_fused import (
    compact_bf16_pages_for_topk,
)


class CompactPrefillPagesTest(unittest.TestCase):
    def setUp(self):
        self.k = torch.arange(6 * 2 * 4 * 8).reshape(6, 2, 4, 8).to(torch.bfloat16)
        self.v = -self.k
        self.table = torch.tensor([[2, 4, 1], [2, 4, 5]], dtype=torch.int32)
        self.topk = torch.tensor(
            [[[0, 2, -1], [1, 2, -1]], [[1, 0, -1], [0, 1, -1]]],
            dtype=torch.int32,
        )
        self.cu = torch.tensor([0, 1, 2], dtype=torch.int32)

    def test_shared_prefix_distinct_bf16_suffix_and_padding(self):
        # Pages 2/4 are a shared prefix. Pages 1/5 are distinct BF16 suffixes.
        # Values deliberately include BF16 values not exactly representable in FP8.
        self.k[1].fill_(1.0078125)
        self.k[5].fill_(1.015625)
        k, v, pt, ids = compact_bf16_pages_for_topk(
            self.k, self.v, self.topk, self.table, self.cu
        )
        self.assertEqual(ids.tolist(), [1, 2, 4, 5])
        self.assertEqual(k.shape[0], 5)  # unique pages plus zero sentinel
        self.assertEqual(pt[0, 0], pt[1, 0])
        self.assertEqual(pt[0, 1], pt[1, 1])
        self.assertNotEqual(pt[0, 2], pt[1, 2])
        self.assertEqual(pt.stride(0) % 4, 0)
        for row in range(2):
            for page in range(3):
                self.assertTrue(
                    torch.equal(k[pt[row, page]], self.k[self.table[row, page]])
                )
                self.assertTrue(
                    torch.equal(v[pt[row, page]], self.v[self.table[row, page]])
                )

    def test_new_mapping_same_geometry(self):
        first = compact_bf16_pages_for_topk(
            self.k, self.v, self.topk, self.table, self.cu
        )
        table = self.table.clone()
        table[1, 2] = 3
        second = compact_bf16_pages_for_topk(self.k, self.v, self.topk, table, self.cu)
        self.assertEqual(second[3].tolist(), [1, 2, 3, 4])
        self.assertTrue(torch.equal(first[0][first[2][1, 2]], self.k[5]))
        self.assertTrue(torch.equal(second[0][second[2][1, 2]], self.k[3]))

    def test_step3_adapter_receives_main_map(self):
        from rtp_llm.models_py.triton_kernels.sparse_msa.prefill import (
            topk_bt_fused as tbf,
        )

        q = torch.zeros(2, 2, 8, dtype=torch.bfloat16)
        api = ModuleType("fmha_sm100.api")
        api.sparse_fmha = Mock(return_value=(q.clone(), None))
        with patch.dict("sys.modules", {"fmha_sm100.api": api}), patch.object(
            tbf, "_sparse_attn_chunk_enabled", return_value=False
        ):
            output = tbf.sparse_prefill_from_topk(
                q, self.k, self.v, self.topk, self.table.reshape(-1), {}, 3, 4, 1.0
            )
        self.assertTrue(torch.equal(q, output))
        kwargs = api.sparse_fmha.call_args.kwargs
        self.assertTrue(torch.equal(kwargs["kv_indices"], self.table.reshape(-1)))
        self.assertNotIn("main_kv_indices", kwargs)

    def test_all_padding(self):
        topk = torch.full_like(self.topk, -1)
        k, v, pt, ids = compact_bf16_pages_for_topk(
            self.k, self.v, topk, self.table, self.cu
        )
        self.assertEqual(ids.numel(), 0)
        self.assertEqual(k.shape[0], 1)
        self.assertFalse(k.any() or v.any() or pt.any())

    def test_reject_invalid_boundaries(self):
        for boundaries in ([1, 1, 2], [0, 3, 2], [0, 1, 3], [0, 2]):
            with self.subTest(boundaries=boundaries):
                with self.assertRaisesRegex(ValueError, "query boundaries"):
                    compact_bf16_pages_for_topk(
                        self.k,
                        self.v,
                        self.topk,
                        self.table,
                        torch.tensor(boundaries, dtype=torch.int32),
                    )

    def test_reject_empty_batch_and_bad_shape(self):
        with self.assertRaisesRegex(ValueError, "nonempty batch"):
            compact_bf16_pages_for_topk(
                self.k, self.v, self.topk[:, :0], self.table[:0], self.cu[:1]
            )
        for topk in (self.topk[0], self.topk[:1], self.topk[:, :, :0]):
            with self.subTest(shape=topk.shape):
                with self.assertRaisesRegex(ValueError, "topk must have shape"):
                    compact_bf16_pages_for_topk(
                        self.k, self.v, topk, self.table, self.cu
                    )
        with self.assertRaisesRegex(ValueError, "rank 1"):
            compact_bf16_pages_for_topk(
                self.k, self.v, self.topk, self.table, self.cu[None]
            )

    def test_reject_noninteger_metadata_and_mixed_device(self):
        metadata = [self.topk, self.table, self.cu]
        for index in range(3):
            invalid = list(metadata)
            invalid[index] = invalid[index].float()
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValueError, "integer tensors"):
                    compact_bf16_pages_for_topk(self.k, self.v, *invalid)
        # Meta creates no device allocation and exercises the device guard on CPU.
        with self.assertRaisesRegex(ValueError, "same device"):
            compact_bf16_pages_for_topk(
                self.k, self.v, self.topk.to("meta"), self.table, self.cu
            )

    def test_reject_fp8_and_invalid_selected_page(self):
        with self.assertRaisesRegex(ValueError, "BF16"):
            compact_bf16_pages_for_topk(
                self.k.float(), self.v, self.topk, self.table, self.cu
            )
        table = self.table.clone()
        table[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "physical page"):
            compact_bf16_pages_for_topk(self.k, self.v, self.topk, table, self.cu)


if __name__ == "__main__":
    unittest.main()

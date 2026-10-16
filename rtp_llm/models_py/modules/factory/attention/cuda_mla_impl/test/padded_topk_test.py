"""Exact index mapping, padding and real FlashMLA dispatch coverage."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rtp_llm.models_py.modules.dsv4.forward_metadata import _FORWARD_METADATA
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    flashmla_sparse_impl as mla,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.block_index_to_global import (
    triton_convert_req_index_to_global_index as convert,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PaddedTopkTest(unittest.TestCase):
    def test_mapping_and_padding(self):
        torch.manual_seed(5313)
        for rows in (0, 1, 17, 81):
            for width in (7, 2051):
                for children in (1, 4):
                    for prefill in (False, True):
                        req = torch.arange(rows, dtype=torch.int32) % 4
                        table = torch.randint(-1, 128, (4, 23), dtype=torch.int32)
                        ids = torch.randint(-3, 3100, (rows, width), dtype=torch.int32)
                        prefill_ids = req.clone()
                        prefill_ids[::2] = -1
                        starts = torch.arange(4, dtype=torch.int32) * 10000
                        physical = children * 128 + 64
                        output_width = (width + 127) // 128 * 128
                        expected = torch.full(
                            (rows, output_width), -1, dtype=torch.int32
                        )
                        blocks = ids.div(128, rounding_mode="floor")
                        base = table[req.long()[:, None], blocks.clamp(0, 22).long()]
                        parent = base.div(children, rounding_mode="floor")
                        value = parent * physical + (base % children) * 128 + ids % 128
                        valid = (ids >= 0) & (blocks < 23)
                        memory_valid = (base > 0) & (parent > 0)
                        if prefill:
                            mask = prefill_ids[:, None] >= 0
                            value = torch.where(
                                mask,
                                starts[prefill_ids.clamp_min(0).long()][:, None] + ids,
                                value,
                            )
                            memory_valid |= mask
                        expected[:, :width] = torch.where(
                            valid & memory_valid, value, -1
                        )
                        kwargs = dict(
                            req_id=req.cuda(),
                            block_table=table.cuda(),
                            token_indices=ids.cuda(),
                            TOKENS_PER_BLOCK_FOR_BLOCK_TABLE=128,
                            ENTRIES_PER_BLOCK=128,
                            NUM_TOPK_TOKENS=width,
                            PHYSICAL_ENTRIES_PER_BLOCK=physical,
                            BLOCKS_PER_PHYSICAL_BLOCK=children,
                            HAS_PREFILL_WORKSPACE=prefill,
                            prefill_workspace_request_ids=prefill_ids.cuda(),
                            prefill_workspace_starts=starts.cuda(),
                        )
                        actual = convert(**kwargs, OUTPUT_TOPK_TOKENS=output_width)
                        self.assertTrue(
                            torch.equal(actual.cpu(), expected),
                            (rows, width, children, prefill),
                        )
                        self.assertTrue(
                            torch.equal(convert(**kwargs).cpu(), expected[:, :width])
                        )

    def test_real_attention_scope_and_async(self):
        torch.manual_seed(1313)
        op = mla.SparseMlaOp(
            64, 512, 0, 256, 128, 0.0625, 2051, indexer_top_k=512, indexer_group_size=4
        )
        rows = 17
        req = torch.arange(rows, device="cuda", dtype=torch.int32) % 2
        table = torch.arange(1, 129, device="cuda", dtype=torch.int32).reshape(2, 64)
        positions = torch.arange(rows, device="cuda", dtype=torch.int32) * 113 + 19
        op.plan(SimpleNamespace(batch_indice_d=req, positions_d=positions), table)
        ids = torch.randint(-2, 2048, (rows, 512), device="cuda", dtype=torch.int32)
        q = torch.randn(rows, 64, 512, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(129 * 128, 1, 512, device="cuda", dtype=torch.bfloat16)
        with patch.object(mla, "_PREFILL_PAD_TOPK", True):
            original = op._convert_topk_indices_to_global(ids)
            self.assertEqual(original.shape[-1], 2051)
            token = _FORWARD_METADATA.set({})
            try:
                padded = op._convert_topk_indices_to_global(ids)
                self.assertEqual(padded.shape[-1], op.kernel_top_k)
                self.assertTrue(
                    torch.equal(
                        padded, mla._pad_flashmla_topk(original, op.kernel_top_k)
                    )
                )
                self.assertTrue(
                    torch.equal(
                        op._forward_sparse(q, kv, original),
                        op._forward_sparse(q, kv, padded),
                    )
                )
                torch.cuda.synchronize()
                torch.cuda._sleep(300_000_000)
                pending = torch.cuda.Event()
                pending.record()
                op._convert_topk_indices_to_global(ids)
                self.assertFalse(pending.query(), "index conversion waited on GPU")
                torch.cuda.synchronize()
            finally:
                _FORWARD_METADATA.reset(token)

    def test_output_offsets_beyond_int32(self):
        rows, width = 1048576, 2176
        output = convert(
            torch.zeros(rows, dtype=torch.int32, device="cuda"),
            torch.ones((1, 1), dtype=torch.int32, device="cuda"),
            torch.zeros((rows, 1), dtype=torch.int32, device="cuda"),
            128,
            128,
            NUM_TOPK_TOKENS=1,
            OUTPUT_TOPK_TOKENS=width,
        )
        boundary = 2**31 // width
        selected = torch.tensor(
            [0, boundary - 1, boundary, boundary + 1, rows - 1], device="cuda"
        )
        expected = torch.full((5, width), -1, dtype=torch.int32, device="cuda")
        expected[:, 0] = 128
        self.assertTrue(torch.equal(output[selected], expected))


if __name__ == "__main__":
    unittest.main()

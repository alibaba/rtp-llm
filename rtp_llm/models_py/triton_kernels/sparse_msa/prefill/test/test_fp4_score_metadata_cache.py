"""CPU regression tests; no package imports or CUDA initialization required."""

import importlib.util
import unittest
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location(
    "fp4_score_metadata_test", Path(__file__).resolve().parents[1] / "score_chunk.py"
)
op = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(op)


class FP4ScoreMetadataCacheTest(unittest.TestCase):
    def setUp(self):
        self.host = op.PrefillScoreHostMetadata(
            (37, 0, 38), (259, 0, 181), (222, 0, 143), (0, 1, 2)
        )
        self.cu = torch.tensor([0, 37, 37, 75], dtype=torch.int32)
        self.seq = torch.tensor(self.host.seq_lens, dtype=torch.int32)
        self.prefix = torch.tensor(self.host.prefix_lens, dtype=torch.int32)
        self.table = torch.tensor([5, 1, 4, 2, 6], dtype=torch.int32)

    def prepare(self, plan, **kwargs):
        params = dict(
            chunk_rows=17,
            block_size_k=128,
            kv_indices=self.table,
            index_score_plan=plan,
            host_metadata=self.host,
        )
        params.update(kwargs)
        return op.prepare_fp4_prefill_score_chunks(
            self.cu, self.seq, self.prefix, **params
        )

    def test_hit_is_per_forward_plan(self):
        first, second = {}, {}
        op.publish_fp4_prefill_metadata_table(first, self.table)
        op.publish_fp4_prefill_metadata_table(second, self.table)
        prepared = self.prepare(first)
        self.assertIs(prepared, self.prepare(first))
        self.assertIsNot(prepared, self.prepare(second))

    def test_no_producer_marker_is_uncached(self):
        plan = {}
        self.assertIsNot(self.prepare(plan), self.prepare(plan))
        self.assertNotIn("_fp4_prepared_chunks", plan)

    def test_ragged_split_chunks_and_offsets_equal_original(self):
        plan = {}
        op.publish_fp4_prefill_metadata_table(plan, self.table)
        chunks, offsets = self.prepare(plan)
        reference = op.build_prefill_score_chunks(
            self.cu,
            self.seq,
            self.prefix,
            None,
            17,
            128,
            kv_indices=self.table,
            host_metadata=self.host,
        )
        self.assertEqual(len(chunks), len(reference))
        for actual, expected, offset in zip(chunks, reference, offsets):
            self.assertEqual(actual.host_metadata, expected.host_metadata)
            self.assertEqual(
                (actual.q_start, actual.q_end), (expected.q_start, expected.q_end)
            )
            for field in (
                "cu_seqlens",
                "seq_lens",
                "prefix_lens",
                "slot_ids",
                "kv_indices",
            ):
                torch.testing.assert_close(
                    getattr(actual, field), getattr(expected, field), rtol=0, atol=0
                )
            pages = (expected.seq_lens + 127) // 128
            original = torch.cat(
                [torch.zeros(1, dtype=torch.int32), pages.cumsum(0).int()]
            )
            torch.testing.assert_close(offset, original, rtol=0, atol=0)

    def test_table_epoch_identity_and_version_invalidation(self):
        plan = {}
        op.publish_fp4_prefill_metadata_table(plan, self.table)
        old = self.prepare(plan)
        self.table.add_(1)
        newer = self.prepare(plan)
        self.assertIsNot(old, newer)
        op.publish_fp4_prefill_metadata_table(plan, self.table)
        self.assertIsNot(newer, self.prepare(plan))
        self.assertIsNot(
            self.prepare(plan), self.prepare(plan, kv_indices=self.table.clone())
        )

    def test_geometry_chunk_and_page_size_invalidation(self):
        plan = {}
        op.publish_fp4_prefill_metadata_table(plan, self.table)
        old = self.prepare(plan)
        self.assertIsNot(old, self.prepare(plan, chunk_rows=19))
        self.assertIsNot(
            old,
            self.prepare(
                plan, host_metadata=self.host._replace(prefix_lens=(221, 0, 142))
            ),
        )
        # Different page size changes the number of table entries: use a valid new table.
        table = torch.arange(2, dtype=torch.int32)
        op.publish_fp4_prefill_metadata_table(plan, table)
        self.assertIsNot(old, self.prepare(plan, block_size_k=512, kv_indices=table))

    def test_inference_tensor_uses_explicit_producer_epoch(self):
        with torch.inference_mode():
            self.table = self.table.clone()
            plan = {}
            op.publish_fp4_prefill_metadata_table(plan, self.table)
            old = self.prepare(plan)
            self.assertIs(old, self.prepare(plan))
            op.publish_fp4_prefill_metadata_table(plan, self.table)
            self.assertIsNot(old, self.prepare(plan))


if __name__ == "__main__":
    unittest.main()

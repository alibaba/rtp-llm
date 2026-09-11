"""Keep future owner page IDs intact while protecting/restoring a checkpoint."""

import dataclasses
import unittest

import test_prefill as fixture
import torch

from rtp_llm.models_py.modules.dsv41.ced import ReplayMode
from rtp_llm.models_py.modules.dsv41.prefill import V41LocalSnapshot


class SnapshotMetadataTailTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.PrefillGpuTest.setUpClass()
        cls.helper = fixture.PrefillGpuTest()

    @classmethod
    def tearDownClass(cls):
        fixture.PrefillGpuTest.tearDownClass()

    def alias_future_id(self, cache, region, overlap):
        owner, binding = cache.owners[20], cache.swa[42]
        original = (
            owner.global_kv.page_table if region == "global" else owner.index_table
        )
        prefix = cache.layout.reuse_unit // cache.layout.token_block_size
        self.assertEqual(original.shape, (1, prefix + 1))
        self.assertEqual(int(original[0, prefix]), prefix + 1)
        if overlap == "payload":
            # Only the unused table suffix overlaps the live SWA page; all
            # checkpoint page IDs reside in the preceding unused page0 bytes.
            first = binding.pages.data.shape[1] - prefix * original.element_size()
            raw = binding.pages.data.flatten()[first : first + original.numel() * 4]
            table = raw.view(torch.int32).view_as(original)
            table.copy_(original)
            if region == "global":
                owner.global_kv = dataclasses.replace(owner.global_kv, page_table=table)
            else:
                owner.index_table = table
        else:
            table = original
            cache.swa[42] = dataclasses.replace(
                binding, valid_ends=table[0, prefix : prefix + 1]
            )
        return table

    def cache_tensors(self, cache):
        tensors = []
        for binding in cache.swa.values():
            tensors.extend(
                (
                    binding.pages.data,
                    binding.page_ids,
                    binding.valid_starts,
                    binding.valid_ends,
                )
            )
        for owner in cache.owners.values():
            tensors.extend(
                (
                    owner.global_kv.pages.data,
                    owner.global_kv.page_table,
                    owner.index_pages.data,
                    owner.index_table,
                )
            )
        return tensors

    @torch.inference_mode()
    def test_restore_rejects_future_table_alias_before_any_copy_or_publication(self):
        for mode in (ReplayMode.FULL, ReplayMode.BOUNDED):
            source, snapshot = self.helper.snapshot_payload_fixture(mode)
            for region in ("global", "index"):
                for overlap in ("payload", "bounds"):
                    with self.subTest(mode=mode.value, region=region, overlap=overlap):
                        cache = self.helper.cache(
                            source.max_tokens, mode, request="receiver"
                        )
                        table = self.alias_future_id(cache, region, overlap)
                        before_table = table.tolist()
                        values = self.cache_tensors(cache)
                        before = [value.clone() for value in values]
                        rejected = False
                        try:
                            snapshot.restore(cache)
                        except ValueError as error:
                            rejected = "snapshot copy aliases" in str(error)
                        self.helper.records.append(
                            {
                                "test": self.id(),
                                "mode": mode.value,
                                "region": region,
                                "overlap": overlap,
                                "rejected": rejected,
                                "table_before": before_table,
                                "table_after": table.tolist(),
                                "published_end": cache.owners[20].materialized_end,
                            }
                        )
                        self.assertTrue(
                            rejected, "restore accepted a future page-table alias"
                        )
                        for actual, expected in zip(values, before):
                            self.helper.equal(actual, expected)
                        self.helper.assert_unpublished_restore(cache)
                        self.assertFalse(cache.poisoned)

    @torch.inference_mode()
    def test_protect_rejects_future_table_alias_without_changing_source(self):
        for mode in (ReplayMode.FULL, ReplayMode.BOUNDED):
            for region in ("global", "index"):
                with self.subTest(mode=mode.value, region=region):
                    source, previous = self.helper.snapshot_payload_fixture(mode)
                    table = self.alias_future_id(source, region, "payload")
                    values = self.cache_tensors(source)
                    before = [value.clone() for value in values]
                    rejected = False
                    try:
                        V41LocalSnapshot.protect(
                            source,
                            end=previous.checkpoint.materialized_end,
                            replay_floor=previous.checkpoint.replay_floor,
                            history_rows=previous.history_rows,
                        )
                    except ValueError as error:
                        rejected = "snapshot copy aliases" in str(error)
                    self.helper.records.append(
                        {
                            "test": self.id(),
                            "mode": mode.value,
                            "region": region,
                            "rejected": rejected,
                            "table": table.tolist(),
                        }
                    )
                    self.assertTrue(
                        rejected, "protect accepted a future page-table alias"
                    )
                    for actual, expected in zip(values, before):
                        self.helper.equal(actual, expected)
                    self.assertFalse(source.poisoned)


if __name__ == "__main__":
    unittest.main()

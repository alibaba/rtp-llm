import unittest
from dataclasses import replace

from rtp_llm.models_py.modules.dsv41.cache_layout import (
    ENCODINGS,
    CacheIdentity,
    CacheLayout,
    CacheRegion,
    GlobalBlock,
    MemoryCheckpoint,
    RegionSlot,
    layer_sources,
    select_complete_checkpoint,
    visible_global_entries,
)


class LayerSourcesTest(unittest.TestCase):
    def test_only_four_layers_write_global_and_index_k(self):
        layers = [layer_sources(layer) for layer in range(43)]
        self.assertEqual([x.layer for x in layers if x.writes_global], [2, 8, 14, 20])
        self.assertEqual([x.layer for x in layers if x.writes_index_k], [2, 8, 14, 20])
        self.assertEqual(
            [x.layer for x in layers if x.scores_queries],
            [2, 8, 14, 20, 24, 28, 32, 36],
        )

    def test_consumers_use_their_source_and_topk_group(self):
        for first, last, owner, topk in (
            (2, 8, 2, 2),
            (8, 14, 8, 8),
            (14, 20, 14, 14),
            (20, 24, 20, 20),
            (24, 28, 20, 24),
            (28, 32, 20, 28),
            (32, 36, 20, 32),
            (36, 40, 20, 36),
        ):
            for layer in range(first, last):
                with self.subTest(layer=layer):
                    sources = layer_sources(layer)
                    self.assertEqual(sources.global_owner, owner)
                    self.assertEqual(sources.index_k_owner, owner)
                    self.assertEqual(sources.topk_owner, topk)
                    self.assertLessEqual(owner, layer)
        for layer in (0, 1, 40, 41, 42):
            self.assertEqual(layer_sources(layer).ratio, 0)
            self.assertIsNone(layer_sources(layer).global_owner)

    def test_incomplete_ratio_two_pairs_are_not_visible(self):
        self.assertEqual(
            [visible_global_entries(2, p) for p in range(6)], [0, 1, 1, 2, 2, 3]
        )
        self.assertEqual(
            [visible_global_entries(20, p) for p in range(6)], [1, 2, 3, 4, 5, 6]
        )

    def test_invalid_layers_and_swa_visibility_fail(self):
        for layer in (-1, 43):
            with self.assertRaises(ValueError):
                layer_sources(layer)
        for layer, position in ((0, 3), (2, -1)):
            with self.assertRaises(ValueError):
                visible_global_entries(layer, position)


class CacheLayoutTest(unittest.TestCase):
    def test_official_compact_encodings_are_distinct(self):
        self.assertEqual(ENCODINGS[CacheRegion.SWA].entry_bytes, 528)
        self.assertEqual(ENCODINGS[CacheRegion.GLOBAL].entry_bytes, 288)
        self.assertEqual(ENCODINGS[CacheRegion.INDEX_K].entry_bytes, 68)
        self.assertEqual(ENCODINGS[CacheRegion.GLOBAL].group_size, 16)
        self.assertEqual(ENCODINGS[CacheRegion.GLOBAL].scale_dtype, "fp8_e4m3")
        self.assertEqual(ENCODINGS[CacheRegion.INDEX_K].group_size, 32)
        self.assertEqual(ENCODINGS[CacheRegion.INDEX_K].scale_dtype, "ue8m0")

    def test_physical_owner_slots_are_not_duplicated_for_consumers(self):
        layout = CacheLayout()
        self.assertEqual(len(layout.pages), 51)
        self.assertEqual(len(layout.required_slots), 51)
        self.assertEqual(len(layout.global_slots), 8)
        self.assertEqual(
            [
                p.slot.owner_layer
                for p in layout.pages
                if p.slot.region == CacheRegion.GLOBAL
            ],
            [2, 8, 14, 20],
        )
        self.assertEqual(len(replace(layout, draft_enabled=False).pages), 48)

    def test_cp8_ring_slack_and_index_page_stride(self):
        layout = CacheLayout()
        self.assertEqual(layout.swa_entries, 136)
        self.assertEqual(layout.reuse_unit, 1024)
        pages = {page.slot: page for page in layout.pages}
        swa = pages[RegionSlot(CacheRegion.SWA, 0)]
        self.assertEqual(swa.page_stride_bytes, 72192)
        self.assertEqual(swa.prefill_shard_bytes, 9024)
        slices = [swa.swa_byte_slice(rank) for rank in range(8)]
        self.assertEqual(slices[0][0], 0)
        self.assertEqual(slices[-1][1], swa.page_stride_bytes)
        self.assertTrue(all(a[1] == b[0] for a, b in zip(slices, slices[1:])))
        for owner, stride in ((2, 4608), (20, 8704)):
            page = pages[RegionSlot(CacheRegion.INDEX_K, owner)]
            self.assertEqual(page.page_stride_bytes, stride)
            self.assertEqual(page.page_stride_bytes % 512, 0)

    def test_paged_mapping_round_trips_all_cp8_shards_and_boundaries(self):
        for block_size in (128, 256):
            layout = CacheLayout(token_block_size=block_size)
            for page in layout.pages:
                if page.slot.region == CacheRegion.SWA:
                    continue
                for compressed in range(2 * layout.reuse_unit // page.ratio + 1):
                    rank, block, row = page.paged_location(compressed)
                    token = (block * 8 + rank) * block_size + row * page.ratio
                    self.assertEqual(token, compressed * page.ratio)
                    self.assertLess(row, page.entries)
                    self.assertLess(rank, 8)

    def test_ratio_two_keeps_all_seven_verify_snapshots(self):
        states = CacheLayout().pair_states
        self.assertEqual([state.owner_layer for state in states], [2, 8, 14])
        self.assertTrue(all(state.snapshots == 7 for state in states))
        self.assertTrue(all(state.partial_elements == 7168 for state in states))

    def test_layout_changes_cannot_share_cache_identity(self):
        layout = CacheLayout()
        self.assertEqual(layout.fingerprint, CacheLayout().fingerprint)
        for other in (
            replace(layout, token_block_size=256),
            replace(layout, cp_size=1),
            replace(layout, draft_enabled=False),
            replace(layout, speculative_tokens=0),
        ):
            self.assertNotEqual(layout.fingerprint, other.fingerprint)

    def test_invalid_layout_fails_before_pool_allocation(self):
        for values in (
            {"cp_size": 4},
            {"token_block_size": 64},
            {"page_alignment": 256},
            {"speculative_tokens": -1},
            {"version": 2},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                CacheLayout(**values)


class CompleteCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.layout = CacheLayout()
        self.identity = CacheIdentity(
            "frozen-model", self.layout.fingerprint, "bounded-v1"
        )

    def checkpoint(self, end):
        return MemoryCheckpoint(
            end,
            self.identity,
            self.layout.required_slots,
            end - 128,
            end - 128,
            end - 128,
            history_ready=True,
            copy_complete=True,
        )

    def blocks(self, count):
        return [
            GlobalBlock(i, self.identity, self.layout.global_slots, copy_complete=True)
            for i in range(count)
        ]

    def test_intermediate_global_blocks_do_not_need_swa_snapshots(self):
        older, latest = self.checkpoint(1024), self.checkpoint(4096)
        matched = select_complete_checkpoint(
            self.layout, self.identity, self.blocks(5), [older, latest], 5120
        )
        self.assertEqual(matched, latest)

    def test_global_chain_hole_limits_checkpoint_selection(self):
        blocks = self.blocks(4)
        blocks[2] = replace(blocks[2], valid_slots=frozenset())
        older, latest = self.checkpoint(2048), self.checkpoint(4096)
        self.assertEqual(
            select_complete_checkpoint(
                self.layout, self.identity, blocks, [older, latest], 4096
            ),
            older,
        )

    def test_allocated_or_uncopied_regions_are_not_complete(self):
        valid = self.checkpoint(2048)
        for incomplete in (
            replace(valid, copy_complete=False),
            replace(valid, history_ready=False),
            replace(valid, draft_swa_start=None),
            replace(valid, target_swa_start=2048),
            replace(valid, target_swa_start=0),
            replace(valid, draft_swa_start=0),
            replace(valid, replay_floor=valid.target_swa_start + 1),
            replace(valid, valid_slots=self.layout.global_slots),
            replace(valid, materialized_end=2049),
            replace(valid, materialized_end=1049600),
            replace(valid, identity=replace(self.identity, replay_fingerprint="full")),
        ):
            with self.subTest(checkpoint=incomplete):
                self.assertFalse(incomplete.is_complete(self.layout, self.identity))
                self.assertIsNone(
                    select_complete_checkpoint(
                        self.layout, self.identity, self.blocks(2), [incomplete], 2048
                    )
                )

    def test_swa_valid_ranges_use_layout_capacity_including_speculative_slack(self):
        end = 2048
        valid = self.checkpoint(end)
        full_ring = replace(
            valid,
            target_swa_start=end - self.layout.swa_entries,
            draft_swa_start=end - self.layout.swa_entries,
            replay_floor=end - self.layout.swa_entries,
        )
        self.assertTrue(full_ring.is_complete(self.layout, self.identity))
        for field in ("target_swa_start", "draft_swa_start"):
            with self.subTest(field=field):
                overflow = replace(
                    full_ring, **{field: end - self.layout.swa_entries - 1}
                )
                self.assertFalse(overflow.is_complete(self.layout, self.identity))

    def test_missing_or_incomplete_global_copy_does_not_extend_prefix(self):
        for blocks in (
            self.blocks(1),
            [self.blocks(2)[1]],
            [replace(self.blocks(1)[0], copy_complete=False)],
        ):
            self.assertIsNone(
                select_complete_checkpoint(
                    self.layout, self.identity, blocks, [self.checkpoint(2048)], 2048
                )
            )

    def test_requested_prefix_and_revision_limit_match(self):
        checkpoint = self.checkpoint(2048)
        self.assertIsNone(
            select_complete_checkpoint(
                self.layout, self.identity, self.blocks(2), [checkpoint], 2047
            )
        )
        blocks = self.blocks(2)
        blocks[0] = replace(
            blocks[0], identity=replace(self.identity, model_revision="other")
        )
        self.assertIsNone(
            select_complete_checkpoint(
                self.layout, self.identity, blocks, [checkpoint], 2048
            )
        )

    def test_duplicate_ordinals_and_inconsistent_layout_fail(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_complete_checkpoint(
                self.layout, self.identity, self.blocks(1) * 2, [], 2048
            )
        with self.assertRaisesRegex(ValueError, "layout"):
            select_complete_checkpoint(
                CacheLayout(cp_size=1), self.identity, [], [], 1024
            )


if __name__ == "__main__":
    unittest.main()

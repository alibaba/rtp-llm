"""Tests for the per-rank V4.1 KV/state pools (cache.py, L2 deliverable).

Covers the L2/L3 seam contract: pool entry sizes (288/288/68/528 B + the
fp32 CSA state), disjoint allocation, write->read byte round-trips,
cross-layer source references (consumers read the source layer's pool, no
copy), fixture export/load round-trips (bytes + manifest fields) and
fail-loud handling of invalid/out-of-range inputs.
"""

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv41.ppu.cache import (
    CSA_STATE_FLOATS,
    FORMAT_VERSION,
    GLOBAL_KV_ENTRY_BYTES,
    INDEXER_KV_ENTRY_BYTES,
    SWA_KV_ENTRY_BYTES,
    CacheGeometry,
    PPUKVCache,
)
from rtp_llm.models_py.modules.dsv41.ppu.weight import synthetic_config

BLOCK_ENTRIES = 8  # synthetic_config() kv_block_entries / swa_block_entries


def _uint8(rows, width, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (rows, width), generator=generator).to(torch.uint8)


def _cache():
    return PPUKVCache(synthetic_config())


class EntrySizeTest(unittest.TestCase):
    def test_entry_sizes_match_the_seam_contract(self):
        self.assertEqual(GLOBAL_KV_ENTRY_BYTES, 288)
        self.assertEqual(INDEXER_KV_ENTRY_BYTES, 68)
        self.assertEqual(SWA_KV_ENTRY_BYTES, 528)
        self.assertEqual(CSA_STATE_FLOATS, 1024)

    def test_descriptors_carry_entry_sizes(self):
        cache = _cache()
        global_desc = cache.descriptors(2)
        self.assertEqual(global_desc["global_kv_2"].entry_bytes, 288)
        self.assertEqual(global_desc["global_kv_2"].ratio, 2)
        self.assertEqual(global_desc["global_kv_2"].kind, "fp4_e2m1_e4m3")
        self.assertEqual(global_desc["indexer_kv"].entry_bytes, 68)
        self.assertTrue(global_desc["indexer_kv"].planar)
        ratio1 = cache.descriptors(6)
        self.assertEqual(ratio1["global_kv_1"].entry_bytes, 288)
        self.assertEqual(ratio1["global_kv_1"].ratio, 1)
        swa = cache.descriptors(0)["swa_kv"]
        self.assertEqual(swa.entry_bytes, 528)
        self.assertEqual(swa.window, 8)
        self.assertEqual(swa.kind, "fp8_e4m3_ue8m0")

    def test_pool_row_widths_after_growth(self):
        cache = _cache()
        cache.alloc(0, 16)
        self.assertEqual(
            tuple(cache.pool(2, "global_kv_2").shape), (1, BLOCK_ENTRIES * 288)
        )
        self.assertEqual(
            tuple(cache.pool(2, "indexer_kv").shape), (1, BLOCK_ENTRIES * 68)
        )
        self.assertEqual(
            tuple(cache.pool(2, "csa_state").shape), (2, BLOCK_ENTRIES * 1024)
        )
        self.assertEqual(tuple(cache.pool(0, "swa_kv").shape), (2, BLOCK_ENTRIES * 528))
        self.assertEqual(cache.pool(2, "csa_state").dtype, torch.float32)
        self.assertEqual(cache.pool(0, "swa_kv").dtype, torch.uint8)
        # Consumers own no pool of their own beyond swa_kv.
        self.assertEqual(set(cache.descriptors(3)), {"swa_kv"})

    def test_geometry_signature_is_stable(self):
        config = synthetic_config()
        self.assertEqual(
            CacheGeometry.from_config(config).signature(),
            CacheGeometry.from_config(config).signature(),
        )
        other = synthetic_config(kv_sources=(2,))
        self.assertNotEqual(
            CacheGeometry.from_config(config).signature(),
            CacheGeometry.from_config(other).signature(),
        )


class AllocDisjointTest(unittest.TestCase):
    def test_two_requests_get_disjoint_slots(self):
        cache = _cache()
        first = cache.alloc(0, 10)
        second = cache.alloc(1, 10)
        for layer in range(8):
            with self.subTest(layer=layer):
                self.assertTrue(
                    not (
                        set(first.swa_slots[layer].tolist())
                        & set(second.swa_slots[layer].tolist())
                    )
                )
        self.assertTrue(
            not (set(first.swa_slots[0].tolist()) & set(second.swa_slots[0].tolist()))
        )
        self.assertTrue(
            not (
                set(first.global_slots[2].tolist())
                & set(second.global_slots[2].tolist())
            )
        )
        self.assertTrue(
            not (
                set(first.indexer_slots[2].tolist())
                & set(second.indexer_slots[2].tolist())
            )
        )
        self.assertTrue(
            not (set(first.csa_slots[2].tolist()) & set(second.csa_slots[2].tolist()))
        )

    def test_extension_keeps_old_slots_and_extends_disjointly(self):
        cache = _cache()
        first = cache.alloc(0, 10)
        other = cache.alloc(1, 10)
        extended = cache.alloc(0, 6)
        self.assertEqual(extended.total_tokens, 16)
        self.assertEqual(extended.n_tokens, 6)
        self.assertEqual(cache.request_length(0), 16)
        # Old region unchanged; the new tail is disjoint from request 1.
        self.assertTrue(torch.equal(extended.swa_slots[0][:10], first.swa_slots[0]))
        self.assertEqual(extended.global_slots[2].numel(), 8)  # 16 // 2
        self.assertTrue(
            not (set(extended.swa_slots[0].tolist()) & set(other.swa_slots[0].tolist()))
        )

    def test_slot_geometry_is_block_major(self):
        cache = _cache()
        slots = cache.alloc(0, 20)  # 20 tokens, 8 entries per block
        swa = slots.swa_slots[0].tolist()
        self.assertEqual(swa[:8], [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(swa[8:16], [8, 9, 10, 11, 12, 13, 14, 15])
        self.assertEqual(slots.global_slots[2].tolist(), list(range(10)))

    def test_negative_token_count_fails_loud(self):
        cache = _cache()
        with self.assertRaises(ValueError):
            cache.alloc(0, -1)


class WriteReadRoundTripTest(unittest.TestCase):
    def test_byte_round_trip_through_write_and_read(self):
        cache = _cache()
        slots = cache.alloc(7, 12)
        global_rows = _uint8(6, 288, seed=1)
        indexer_rows = _uint8(6, 68, seed=2)
        swa_rows = _uint8(12, 528, seed=3)
        csa_rows = torch.randn(12, 1024, generator=torch.Generator().manual_seed(4))
        cache.write(
            2,
            slots,
            global_kv=global_rows,
            indexer_k=indexer_rows,
            swa_kv=swa_rows,
            csa_state=csa_rows,
        )
        read_global, read_indexer = cache.read_source(2, 7)
        self.assertTrue(torch.equal(read_global, global_rows))
        self.assertTrue(torch.equal(read_indexer, indexer_rows))
        self.assertTrue(torch.equal(cache.read_swa(2, 7, torch.arange(12)), swa_rows))
        self.assertTrue(torch.equal(cache.read_csa(2, 7, torch.arange(12)), csa_rows))
        everything = cache.read(2, slots=slots)
        self.assertEqual(
            set(everything), {"global_kv_2", "indexer_kv", "swa_kv", "csa_state"}
        )
        self.assertTrue(torch.equal(everything["global_kv_2"], global_rows))
        self.assertTrue(torch.equal(everything["swa_kv"], swa_rows))
        self.assertTrue(torch.equal(everything["csa_state"], csa_rows))

    def test_ratio1_source_round_trip(self):
        cache = _cache()
        slots = cache.alloc(3, 9)
        global_rows = _uint8(9, 288, seed=5)
        swa_rows = _uint8(9, 528, seed=6)
        cache.write(6, slots, global_kv=global_rows, swa_kv=swa_rows)
        read_global, _ = cache.read_source(6, 3)
        self.assertTrue(torch.equal(read_global, global_rows))
        self.assertTrue(torch.equal(cache.read_swa(6, 3, torch.arange(9)), swa_rows))

    def test_read_by_request_id_gathers_valid_region(self):
        cache = _cache()
        slots = cache.alloc(5, 10)
        swa_rows = _uint8(10, 528, seed=7)
        cache.write(0, slots, swa_kv=swa_rows)
        everything = cache.read(0, request_id=5)
        self.assertTrue(torch.equal(everything["swa_kv"], swa_rows))

    def test_partial_write_at_explicit_slots(self):
        cache = _cache()
        slots = cache.alloc(0, 16)
        rows = _uint8(4, 288, seed=8)
        cache.insert_rows(2, "global_kv_2", slots.global_slots[2][2:6], rows)
        read_global, _ = cache.read_source(2, 0, global_entries=6)
        self.assertTrue(torch.equal(read_global[2:6], rows))
        self.assertTrue(bool((read_global[:2] == 0).all()))

    def test_negative_slots_are_skipped_not_crash(self):
        """Documented padding semantics: -1 slots neither write nor read."""
        cache = _cache()
        slots = cache.alloc(0, 4)
        rows = _uint8(2, 288, seed=9)
        cache.insert_rows(
            2, "global_kv_2", torch.tensor([-1, slots.global_slots[2][1]]), rows
        )
        read_global, _ = cache.read_source(2, 0, global_entries=2)
        self.assertTrue(torch.equal(read_global[1], rows[1]))
        self.assertTrue(bool((read_global[0] == 0).all()))


class CrossLayerSourceTest(unittest.TestCase):
    def test_consumer_view_references_source_pool_without_copy(self):
        cache = _cache()
        cache.alloc(0, 12)
        view = cache.cache_view(3)  # ratio-2 consumer of source layer 2
        self.assertIs(view.pools["global_kv_2"], cache.pool(2, "global_kv_2"))
        self.assertIs(view.pools["swa_kv"], cache.pool(3, "swa_kv"))
        descriptor = view.descriptors["global_kv_2"]
        self.assertEqual(descriptor.producer_layer_id, 2)
        self.assertEqual(descriptor.compression_ratio, 2)
        self.assertNotIn("global_kv_1", view.pools)

    def test_ratio1_consumer_references_its_own_source(self):
        cache = _cache()
        cache.alloc(0, 4)
        view = cache.cache_view(7)
        self.assertIs(view.pools["global_kv_1"], cache.pool(6, "global_kv_1"))

    def test_swa_only_layer_sees_only_its_window_pool(self):
        cache = _cache()
        cache.alloc(0, 4)
        view = cache.cache_view(0)
        self.assertEqual(set(view.pools), {"swa_kv"})
        self.assertEqual(view.descriptors["swa_kv"].producer_layer_id, 0)

    def test_block_tables_are_request_indexed(self):
        cache = _cache()
        cache.alloc(2, 20)
        view = cache.cache_view(3)
        table = view.block_tables["global_kv_2"]
        self.assertEqual(table.shape[0], 3)  # request id VALUE is the row
        self.assertEqual(table[2, 0].item(), 0)  # first allocated block


class FailLoudTest(unittest.TestCase):
    def test_read_source_rejects_non_source_layer(self):
        cache = _cache()
        cache.alloc(0, 4)
        with self.assertRaises(ValueError):
            cache.read_source(3, 0)

    def test_write_rejects_pools_the_layer_does_not_own(self):
        cache = _cache()
        slots = cache.alloc(0, 4)
        with self.assertRaises(ValueError):
            cache.write(3, slots, global_kv=_uint8(2, 288, seed=1))
        with self.assertRaises(ValueError):
            cache.write(3, slots, indexer_k=_uint8(2, 68, seed=2))
        with self.assertRaises(ValueError):
            cache.write(6, slots, csa_state=torch.zeros(4, 1024))  # ratio 1
        with self.assertRaises(ValueError):
            cache.write(2, slots, csa_state=_uint8(4, 1024, seed=3))  # uint8

    def test_insert_rows_rejects_unknown_pool(self):
        cache = _cache()
        slots = cache.alloc(0, 4)
        with self.assertRaises(ValueError):
            cache.insert_rows(3, "global_kv_2", slots.swa_slots[3], _uint8(4, 288, 1))
        with self.assertRaises(ValueError):
            cache.insert_rows(2, "nope", slots.swa_slots[2], _uint8(4, 288, 1))

    def test_insert_rows_rejects_wrong_row_width(self):
        cache = _cache()
        slots = cache.alloc(0, 4)
        for tag, width in (
            ("global_kv_2", 288),
            ("indexer_kv", 68),
            ("swa_kv", 528),
        ):
            with self.subTest(tag=tag):
                with self.assertRaises(ValueError):
                    cache.insert_rows(
                        2 if tag != "swa_kv" else 2,
                        tag,
                        slots.swa_slots[2],
                        _uint8(4, width + 1, seed=10),
                    )

    def test_out_of_range_positions_fail_loud(self):
        # Position bounds are block-table bounds: a request with 8 tokens
        # owns exactly one swa/csa block, so position 8 is out of range.
        cache = _cache()
        cache.alloc(0, 8)
        with self.assertRaises(ValueError):
            cache.read_swa(0, 0, torch.tensor([8]))
        with self.assertRaises(ValueError):
            cache.slots_at_positions(2, "csa_state", 0, torch.tensor([8]))
        with self.assertRaises(ValueError):
            cache.read_csa(2, 0, torch.tensor([8]))

    def test_unallocated_request_fails_loud(self):
        cache = _cache()
        # read_swa/read_csa surface the missing request as KeyError, the
        # slots_at_positions seam as ValueError: both fail loud.
        with self.assertRaises((KeyError, ValueError)):
            cache.read_swa(0, 42, torch.tensor([0]))
        with self.assertRaises(ValueError):
            cache.slots_at_positions(0, "swa_kv", 42, torch.tensor([0]))


class FixtureRoundTripTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="dsv41_c1_"))
        self.config = synthetic_config()
        self.cache = PPUKVCache(self.config)
        slots = self.cache.alloc(0, 12)
        self.cache.write(
            2,
            slots,
            global_kv=_uint8(6, 288, seed=11),
            indexer_k=_uint8(6, 68, seed=12),
            swa_kv=_uint8(12, 528, seed=13),
            csa_state=torch.randn(
                12, 1024, generator=torch.Generator().manual_seed(14)
            ),
        )
        self.cache.write(6, slots, global_kv=_uint8(12, 288, seed=15))
        self.cache.alloc(1, 5)
        self.manifest = self.cache.export_fixture(
            self.root, meta={"source_cp": 4, "snapshot": "snap-1", "note": "hi"}
        )

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_round_trip_preserves_bytes_and_bookkeeping(self):
        loaded, manifest = PPUKVCache.load_fixture(self.root)
        for owner, tag in self.cache.pool_keys():
            self.assertTrue(
                torch.equal(loaded.pool(owner, tag), self.cache.pool(owner, tag)),
                (owner, tag),
            )
        self.assertEqual(loaded.request_length(0), 12)
        self.assertEqual(loaded.request_length(1), 5)
        self.assertEqual(loaded.valid_regions(), self.cache.valid_regions())
        original, _ = self.cache.read_source(2, 0)
        restored, _ = loaded.read_source(2, 0)
        self.assertTrue(torch.equal(original, restored))
        self.assertTrue(
            torch.equal(
                loaded.read_swa(0, 0, torch.arange(12)),
                self.cache.read_swa(0, 0, torch.arange(12)),
            )
        )
        # The reloaded cache keeps allocating disjointly.
        slots = loaded.alloc(2, 4)
        self.assertEqual(slots.total_tokens, 4)

    def test_manifest_fields(self):
        manifest = self.manifest
        self.assertEqual(manifest["format_version"], FORMAT_VERSION)
        self.assertEqual(manifest["source_cp"], 4)
        self.assertEqual(manifest["producer_snapshot"]["snapshot"], "snap-1")
        self.assertEqual(manifest["producer_snapshot"]["module"], PPUKVCache.__module__)
        self.assertEqual(manifest["lengths"], {"0": 12, "1": 5})
        self.assertEqual(manifest["valid_regions"], self.cache.valid_regions())
        self.assertEqual(manifest["meta"], {"source_cp": 4, "note": "hi"})
        checksum = hashlib.sha256((self.root / "pools.pt").read_bytes()).hexdigest()
        self.assertEqual(manifest["checksum"], checksum)
        self.assertEqual(
            manifest["geometry_signature"],
            CacheGeometry.from_config(self.config).signature(),
        )
        on_disk = json.loads((self.root / "manifest.json").read_text())
        self.assertEqual(on_disk, manifest)

    def test_tampered_payload_is_refused(self):
        payload_path = self.root / "pools.pt"
        payload_path.write_bytes(payload_path.read_bytes() + b"x")
        with self.assertRaises(ValueError):
            PPUKVCache.load_fixture(self.root)

    def test_incompatible_format_version_is_refused(self):
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["format_version"] = FORMAT_VERSION + 1
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            PPUKVCache.load_fixture(self.root)

    def test_geometry_signature_mismatch_is_refused(self):
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["geometry_signature"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            PPUKVCache.load_fixture(self.root)

    def test_missing_files_are_refused(self):
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(ValueError):
            PPUKVCache.load_fixture(empty)

    def test_bytes_summary_reports_pool_tags(self):
        summary = self.cache.bytes_summary()
        self.assertEqual(
            set(summary),
            {"global_kv_2", "global_kv_1", "indexer_kv", "swa_kv", "csa_state"},
        )
        self.assertGreater(summary["swa_kv"], 8 * 528)


if __name__ == "__main__":
    unittest.main()

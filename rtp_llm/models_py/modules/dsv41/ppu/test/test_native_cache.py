"""Native page geometry, planar indexer, ring boundary and continuation tests."""

import unittest
from types import SimpleNamespace as NS

import torch

from rtp_llm.models_py.modules.dsv41.ppu.cache import PPUKVCache
from rtp_llm.models_py.modules.dsv41.ppu.native_cache import NativeCacheBridge
from rtp_llm.models_py.modules.dsv41.ppu.native_protocol import FRAMEWORK_POOL_TAGS
from rtp_llm.models_py.modules.dsv41.ppu.weight import synthetic_config


class NativePools:
    """Same duck-typed surface as the C++ KVCache, with deliberately padded pages."""

    def __init__(self, config, batch=2, blocks=12, span=8):
        logical = PPUKVCache(config)
        self.config = config
        self.span = span
        self.handles = {}
        self.calls = []
        self.entries = {}
        self.group_tags = list({FRAMEWORK_POOL_TAGS[key[1]] for key in logical.pool_keys()})
        self.inputs = {}
        for owner, tag in logical.pool_keys():
            framework_tag = FRAMEWORK_POOL_TAGS[tag]
            if tag == "global_kv_2":
                entries = span // 2
            elif tag in ("global_kv_1", "indexer_kv"):
                entries = span
            elif tag == "swa_kv":
                entries = logical.geometry.window + 2
            else:
                entries = 4
            self.entries[framework_tag] = entries
            width = logical._row_width((owner, tag))
            dtype = torch.float32 if tag == "csa_state" else torch.uint8
            pool = torch.full((blocks, entries * width + 16), 173, dtype=dtype)
            self.handles[(owner, framework_tag)] = NS(
                kv_cache_base=pool, tag=framework_tag, layer_id=owner
            )
            if framework_tag not in self.inputs:
                # Independent physical blocks for each request; logical rows
                # deliberately do not equal physical block ids.
                table = torch.arange(blocks).reshape(blocks // batch, batch).T.flip(1)
                self.inputs[framework_tag] = NS(
                    kv_cache_kernel_block_id_device=table,
                    kv_cache_block_id_device=table,
                    cache_store_inputs=framework_tag,
                    cache_store_writer=NS(write=self.write),
                )

    def write(self, params, handle):
        assert params == handle.tag
        self.calls.append((handle.layer_id, handle.tag))

    def get_layer_cache(self, owner, tag):
        return self.handles[(owner, tag)]

    def get_entries_per_block(self, tag):
        return self.entries[tag]

    def get_kernel_seq_size_per_block(self, tag):
        return self.span

    def get_seq_size_per_block(self, tag):
        return self.span


class NativeCacheTest(unittest.TestCase):
    def test_framework_names_are_required_at_the_native_boundary(self):
        config = synthetic_config()
        native = NativePools(config)
        bridge = NativeCacheBridge(config, native, native.inputs)
        self.assertIn("hca_kv", native.inputs)
        self.assertIn("csa_kv", native.inputs)
        for (owner, tag), pool in bridge.pools.items():
            self.assertEqual(pool.framework_tag, FRAMEWORK_POOL_TAGS[tag])
            self.assertIs(pool.handle, native.handles[(owner, pool.framework_tag)])
        old_inputs = dict(native.inputs)
        old_inputs["global_kv_1"] = old_inputs.pop("hca_kv")
        with self.assertRaisesRegex(ValueError, "missing native attention.*hca_kv"):
            NativeCacheBridge(config, native, old_inputs)

    def test_native_planes_padding_rings_and_disjoint_requests(self):
        config = synthetic_config()
        cache = PPUKVCache(config)
        lengths = [17, 9]
        generator = torch.Generator().manual_seed(7)
        for rid, length in enumerate(lengths):
            cache.alloc(rid, length)
            for key in cache.pool_keys():
                count = cache._valid_entries[key][rid]
                dtype = cache.pool(*key).dtype
                packed = torch.randint(
                    1, 150, (count, cache._row_width(key)), generator=generator
                ).to(dtype)
                cache.insert_rows(
                    *key, cache._slots_for_blocks(key, rid, count), packed
                )
        native = NativePools(config)
        bridge = NativeCacheBridge(config, native, native.inputs)
        bridge.export_requests(cache, lengths, publish=True)
        self.assertEqual(
            set(native.calls),
            {(owner, FRAMEWORK_POOL_TAGS[tag]) for owner, tag in cache.pool_keys()},
        )
        restored = NativeCacheBridge(config, native, native.inputs).import_requests(
            lengths
        )
        for key in cache.pool_keys():
            for rid, length in enumerate(lengths):
                positions = bridge._positions(key, length)
                if key[1] in ("global_kv_1", "global_kv_2", "indexer_kv"):
                    slots = cache._slots_for_blocks(key, rid, positions.numel())
                    restored_slots = restored._slots_for_blocks(
                        key, rid, positions.numel()
                    )
                else:
                    slots = cache.slots_at_positions(*key, rid, positions)
                    restored_slots = restored.slots_at_positions(*key, rid, positions)
                self.assertTrue(
                    torch.equal(
                        cache._gather(key, slots), restored._gather(key, restored_slots)
                    ),
                    key,
                )
            self.assertTrue(
                bool(
                    (
                        native.handles[(key[0], FRAMEWORK_POOL_TAGS[key[1]])]
                        .kv_cache_base[:, -16:]
                        == 173
                    ).all()
                )
            )
        # The indexer scale plane begins at RAW capacity * 64, even when
        # ratio=2 produces only half as many valid compressed entries.
        table = native.inputs["indexer_kv"].kv_cache_block_id_device
        block = int(table[0, 0])
        expected = cache.read_source(2, 0)[1][:4]
        actual = native.handles[(2, "indexer_kv")].kv_cache_base[block]
        self.assertTrue(torch.equal(actual[: 4 * 64], expected[:, :64].reshape(-1)))
        self.assertTrue(
            torch.equal(actual[8 * 64 : 8 * 64 + 4 * 4], expected[:, 64:].reshape(-1))
        )

    def test_rejects_missing_tags_and_incompatible_geometry(self):
        config = synthetic_config()
        native = NativePools(config)
        with self.assertRaisesRegex(ValueError, "missing native attention"):
            NativeCacheBridge(config, native, {})
        native.entries["swa_kv"] = 2
        with self.assertRaisesRegex(ValueError, "complete attention window"):
            NativeCacheBridge(config, native, native.inputs)

    def test_stale_or_truncated_block_table_is_rejected(self):
        config = synthetic_config()
        native = NativePools(config)
        native.inputs["swa_kv"].kv_cache_block_id_device = torch.full((2, 1), -1)
        bridge = NativeCacheBridge(config, native, native.inputs)
        with self.assertRaisesRegex(ValueError, "unallocated physical block"):
            bridge.import_requests([1])


if __name__ == "__main__":
    unittest.main()

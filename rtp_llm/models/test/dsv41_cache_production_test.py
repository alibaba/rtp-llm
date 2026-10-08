"""Drive the native allocator geometry and PP projection without GPU allocation."""

import json
import pickle
import unittest
from collections import Counter
from pathlib import Path

from rtp_llm.models.dsv41_kv_cache import (
    build_dsv41_kv_cache_spec_descs,
    dsv41_cache_bindings,
    dsv41_page_layouts,
)

# Load the config types and Torch libraries before the test binding.
# isort: off
from rtp_llm.ops import (
    KVCacheConfig,
    ModelConfig,
    ParallelismConfig,
    RoleType,
)
from rtp_llm.cpp.cache.test.libcache_config_creator_py_test import cache_pool_topology

# isort: on


class V41ProductionCacheTest(unittest.TestCase):
    def setUp(self):
        fixture = (
            Path(__file__).parents[2] / "config/test/testdata/deepseek_v41_flash.json"
        )
        self.text = json.loads(fixture.read_text())["text_config"]
        self.model = ModelConfig()
        self.model.model_type = "deepseek_v41"
        self.model.num_layers = 40
        self.model.data_type = "bf16"
        self.model.attn_config.tokens_per_block = 256
        self.model.hybrid_attention_config.enable_independent_kv_cache_pools = True
        self.model.kv_cache_spec_descs = build_dsv41_kv_cache_spec_descs(self.text)
        self.kv_config = KVCacheConfig()
        self.kv_config.seq_size_per_block = 256
        self.kv_config.kernel_seq_size_per_block = 256

    def parallelism(self, pp_size=1, pp_rank=0):
        config = ParallelismConfig()
        config.pp_size, config.pp_rank = pp_size, pp_rank
        config.pp_stage_layer_counts = [21, 19] if pp_size == 2 else [40]
        config.tp_size = 4 if pp_size == 2 else 1
        config.role_type = RoleType.PREFILL if pp_size == 2 else RoleType.DECODE
        config.prefill_cp_config.kv_cache_sharded = False
        return config

    def probe(self, pp_size=1, pp_rank=0, gen=1):
        return cache_pool_topology(
            self.model, self.parallelism(pp_size, pp_rank), self.kv_config, gen
        )

    def test_native_geometry_matches_planes_and_rings(self):
        for gen in (0, 1, 3, 8):
            groups = self.probe(gen=gen)["groups"]
            for tag, layout in dsv41_page_layouts(gen).items():
                self.assertEqual(groups[tag]["block_size_bytes"], layout.stride_bytes)
                self.assertEqual(groups[tag]["block_payload_bytes"], layout.used_bytes)
                self.assertEqual(groups[tag]["seq_size_per_block"], 256)
                self.assertEqual(groups[tag]["cp_mapping"], 0)
                self.assertEqual(groups[tag]["cp_slice"], 0)
                state = tag in ("swa_kv", "csa_state")
                self.assertEqual(groups[tag]["single_block_per_request"], state)
                self.assertEqual(groups[tag]["enable_prefix_reuse"], not state)
        self.assertEqual(self.probe()["groups"]["indexer_kv"]["layers"], [2, 8, 14, 20])

    def test_stage_storage_and_publication_match_global_bindings(self):
        exported = []
        for rank, count, begin in ((0, 21, 0), (1, 19, 21)):
            stage = self.probe(2, rank)
            self.assertEqual(stage["layer_num"], count)
            self.assertEqual(stage["global_layer_begin"], begin)
            actual = set()
            for tag, group in stage["groups"].items():
                source = group["read_replica_source_layer_id"]
                actual.update(
                    (
                        tag,
                        layer,
                        source if source is not None else layer + begin,
                        source is not None,
                    )
                    for layer in group["layers"]
                )
            expected = {
                (b.tag, b.storage_layer_id, b.source_layer_id, b.is_read_replica)
                for b in dsv41_cache_bindings(self.text, 2, rank)
            }
            self.assertEqual(actual, expected)
            exported.extend(
                (layer + begin, tag)
                for layer, tags in enumerate(stage["transfer_tags"])
                for tag in tags
            )
        expected_exports = [
            (b.source_layer_id, b.tag) for b in dsv41_cache_bindings(self.text)
        ]
        self.assertEqual(Counter(exported), Counter(expected_exports))
        self.assertEqual(self.probe(2, 1)["transfer_tags"], [["swa_kv"]] * 19)

    def test_replica_budget_is_charged(self):
        stage = self.probe(2, 1)
        replica_bytes = 0
        for tag in ("hca_kv", "indexer_kv"):
            group = stage["groups"][tag]
            self.assertEqual(group["read_replica_source_layer_id"], 20)
            self.assertEqual(group["block_num"], 16)
            self.assertEqual(group["group_bytes_per_block"], group["block_size_bytes"])
            replica_bytes += group["group_bytes_per_block"] * group["block_num"]
        self.assertEqual(replica_bytes, 16 * 91136)
        self.assertEqual(stage["paged_block_bytes"], 91136)

    def test_descriptor_pickle_preserves_ownership(self):
        desc = self.model.kv_cache_spec_descs[20][1]
        self.assertEqual(desc.tag, "hca_kv")
        desc.read_replica_source_layer_id = 20
        restored = pickle.loads(pickle.dumps(desc))
        self.assertEqual(restored.read_replica_source_layer_id, 20)
        self.assertEqual(restored.cp.mapping, desc.cp.mapping)
        self.assertEqual(restored.entry_elems, 288)
        legacy = type(desc).__new__(type(desc))
        legacy.__setstate__(desc.__getstate__()[:20])
        self.assertIsNone(legacy.read_replica_source_layer_id)
        self.assertEqual(legacy.tag, "hca_kv")

    def test_rejects_wrong_partition_and_sharding(self):
        config = self.parallelism(2, 0)
        config.pp_stage_layer_counts = [20, 20]
        with self.assertRaisesRegex(RuntimeError, "layer counts"):
            cache_pool_topology(self.model, config, self.kv_config, 1)
        config = self.parallelism(2, 0)
        config.prefill_cp_config.kv_cache_sharded = True
        with self.assertRaisesRegex(RuntimeError, "unsharded"):
            cache_pool_topology(self.model, config, self.kv_config, 1)

    def test_state_capacity_pickle_and_legacy_defaults(self):
        desc = self.model.kv_cache_spec_descs[2][-1]
        restored = pickle.loads(pickle.dumps(desc))
        self.assertTrue(restored.capacity.single_block_per_request)
        self.assertFalse(restored.reuse.enable_prefix_reuse)
        legacy = type(desc.capacity).__new__(type(desc.capacity))
        legacy.__setstate__(desc.capacity.__getstate__()[:3])
        self.assertIsNone(legacy.single_block_per_request)

    def test_rejects_rotating_state_or_prefix_reuse(self):
        for tag in ("swa_kv", "csa_state"):
            for invalid in ("allocation", "reuse"):
                with self.subTest(tag=tag, invalid=invalid):
                    descs = build_dsv41_kv_cache_spec_descs(self.text)
                    desc = next(d for d in descs[2] if d.tag == tag)
                    if invalid == "allocation":
                        desc.capacity = None
                    else:
                        reuse = desc.reuse
                        reuse.enable_prefix_reuse = True
                        desc.reuse = reuse
                    self.model.kv_cache_spec_descs = descs
                    with self.assertRaisesRegex(RuntimeError, "persistent state"):
                        self.probe()

    def test_rejects_page_overrides_and_wrong_index_capacity(self):
        self.kv_config.kernel_seq_size_per_block = 128
        with self.assertRaisesRegex(RuntimeError, "256/256"):
            self.probe()
        self.kv_config.kernel_seq_size_per_block = 256
        descs = self.model.kv_cache_spec_descs
        descs[2][2].compression_ratio = 2
        self.model.kv_cache_spec_descs = descs
        with self.assertRaisesRegex(RuntimeError, "geometry"):
            self.probe()

    def test_rejects_replica_or_extra_producer_in_global_input(self):
        descs = self.model.kv_cache_spec_descs
        descs[20][1].read_replica_source_layer_id = 20
        self.model.kv_cache_spec_descs = descs
        with self.assertRaisesRegex(RuntimeError, "ownership"):
            self.probe(2, 1)
        descs = build_dsv41_kv_cache_spec_descs(self.text)
        descs[21].append(descs[20][1])
        self.model.kv_cache_spec_descs = descs
        with self.assertRaisesRegex(RuntimeError, "unexpected tag"):
            self.probe(2, 1)


if __name__ == "__main__":
    unittest.main()

"""Full-feature env parsing tests for the DSv4 KV cache group.

Parses the exact full-feature production environment (the DSv4 prefill
candidates on both platform variants) through the server args parser and
verifies the effective KVCacheConfig fields, including the GPU prefix-tree
dependency wiring and the removed legacy independent-eviction alias.
"""

import os
import sys
from unittest import TestCase
from unittest.mock import patch

# Exact full-feature prefill candidate envs (GB200 baseline; B300 differs only in block reserve).
GB200_PREFILL_CANDIDATE = {
    "REUSE_CACHE": "1",
    "ENABLE_DEVICE_CACHE": "1",
    "ENABLE_TIERED_MEMORY_CACHE": "1",
    "ENABLE_PREFIX_TREE_MEMORY_CACHE": "1",
    "PREFILL_CP_KV_CACHE_SHARDED": "1",
    "ENABLE_MEMORY_CACHE": "1",
    "ENABLE_MEMORY_CACHE_DISK": "1",
    "MEMORY_CACHE_DISK_PATHS": (
        "/tmp/dsv4_main_20260911_release_disk/rank0,"
        "/tmp/dsv4_main_20260911_release_disk/rank1,"
        "/tmp/dsv4_main_20260911_release_disk/rank2,"
        "/tmp/dsv4_main_20260911_release_disk/rank3"
    ),
    "ENABLE_GPU_PREFIX_TREE": "1",
    "ENABLE_INDEPENDENT_GROUP_EVICTION": "1",
    "SEQ_SIZE_PER_BLOCK": "128",
    "KERNEL_SEQ_SIZE_PER_BLOCK": "128",
    "DEVICE_CACHE_MIN_FREE_BLOCKS": "44000",
}

B300_DIFFERENCES = {
    "DEVICE_CACHE_MIN_FREE_BLOCKS": "10000",
}

LEGACY_ALIAS = "ENABLE_DSV4_STATE_BLOCK_INDEPENDENT_EVICTION"


def candidate_for(platform: str) -> dict:
    envs = dict(GB200_PREFILL_CANDIDATE)
    if platform == "b300":
        envs.update(B300_DIFFERENCES)
    return envs


class KVCacheEnvCandidateTest(TestCase):
    def _parse(self, envs: dict):
        from rtp_llm.server.server_args import server_args

        with patch.dict(os.environ, envs, clear=True), patch.object(sys, "argv", ["rtp_llm_server"]):
            return server_args.setup_args()

    def test_gb200_prefill_candidate_parses_to_kv_cache_config(self):
        configs = self._parse(candidate_for("gb200"))
        kv = configs.kv_cache_config
        self.assertTrue(kv.reuse_cache)
        self.assertTrue(kv.enable_device_cache)
        self.assertTrue(kv.enable_tiered_memory_cache)
        self.assertTrue(kv.enable_prefix_tree_memory_cache)
        self.assertTrue(kv.enable_memory_cache)
        self.assertTrue(kv.enable_memory_cache_disk)
        self.assertEqual(
            kv.memory_cache_disk_paths,
            GB200_PREFILL_CANDIDATE["MEMORY_CACHE_DISK_PATHS"],
        )
        # GPU prefix-tree dependency wiring: the flag must reach KVCacheConfig
        # so KVCacheManager::init passes it into SharedBlockCache.
        self.assertTrue(kv.enable_gpu_prefix_tree)
        self.assertTrue(kv.enable_independent_group_eviction)
        self.assertEqual(kv.seq_size_per_block, 128)
        self.assertEqual(kv.kernel_seq_size_per_block, 128)
        self.assertEqual(kv.device_cache_min_free_blocks, 44000)

    def test_b300_prefill_candidate_parses_to_kv_cache_config(self):
        configs = self._parse(candidate_for("b300"))
        kv = configs.kv_cache_config
        self.assertTrue(kv.enable_gpu_prefix_tree)
        self.assertTrue(kv.enable_independent_group_eviction)
        self.assertTrue(kv.enable_memory_cache)
        self.assertTrue(kv.enable_prefix_tree_memory_cache)
        self.assertEqual(kv.device_cache_min_free_blocks, 10000)
        self.assertEqual(kv.seq_size_per_block, 128)
        self.assertEqual(kv.kernel_seq_size_per_block, 128)

    def test_gpu_prefix_tree_defaults_off(self):
        # Negative control: without the env, the default is off so a silent
        # config regression cannot hide behind the parser default.
        configs = self._parse({})
        kv = configs.kv_cache_config
        self.assertFalse(kv.enable_gpu_prefix_tree)
        self.assertFalse(kv.enable_independent_group_eviction)
        self.assertFalse(kv.enable_memory_cache)

    def test_gpu_prefix_tree_disabled_with_dependencies_enabled(self):
        # Negative control: ENABLE_GPU_PREFIX_TREE=0 must stay off even when
        # every canonical dependency flag is enabled; SharedBlockCache requires
        # prefix_tree_enabled_ for independent group eviction.
        envs = dict(candidate_for("gb200"))
        envs["ENABLE_GPU_PREFIX_TREE"] = "0"
        configs = self._parse(envs)
        kv = configs.kv_cache_config
        self.assertFalse(kv.enable_gpu_prefix_tree)
        self.assertTrue(kv.enable_memory_cache)
        self.assertTrue(kv.enable_prefix_tree_memory_cache)
        self.assertTrue(kv.enable_independent_group_eviction)

    def test_legacy_independent_eviction_alias_is_not_registered(self):
        # The legacy DSv4 alias must be absent from the new config layers; the
        # canonical flag is the only binding. Setting the old env name must not
        # flip the canonical field.
        envs = dict(candidate_for("gb200"))
        envs.pop("ENABLE_INDEPENDENT_GROUP_EVICTION")
        envs[LEGACY_ALIAS] = "1"
        configs = self._parse(envs)
        kv = configs.kv_cache_config
        self.assertFalse(kv.enable_independent_group_eviction)

    def test_disk_paths_parse_as_single_string_with_four_entries(self):
        configs = self._parse(candidate_for("gb200"))
        kv = configs.kv_cache_config
        paths = kv.memory_cache_disk_paths.split(",")
        self.assertEqual(len(paths), 4)
        self.assertTrue(all(p.strip() for p in paths))
        self.assertEqual(len(set(paths)), 4, "disk rank paths must be unique")
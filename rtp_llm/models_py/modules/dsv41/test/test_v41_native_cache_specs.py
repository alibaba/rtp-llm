"""V4.1 indexer descriptors resolved by the real C++ cache topology builder."""

import importlib.machinery
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.dsv41_kv_cache import configure_v41_kv_cache


def load_native_bridge():
    relative = Path("rtp_llm/cpp/cache/test")
    roots = [
        Path(os.environ.get("TEST_SRCDIR", "."))
        / os.environ.get("TEST_WORKSPACE", "rtp_llm"),
        Path(__file__).absolute().parents[5],
        Path(__file__).resolve().parents[5],
    ]
    for root in roots:
        directory = root / relative
        for filename in (
            "libcache_config_creator_py_test.so",
            "cache_config_creator_py_test.so",
            "cache_config_creator_py_test",
        ):
            path = directory / filename
            if not path.is_file():
                continue
            sys.path.insert(0, str(directory))
            name = "libcache_config_creator_py_test"
            loader = importlib.machinery.ExtensionFileLoader(name, str(path))
            spec = importlib.util.spec_from_file_location(name, path, loader=loader)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            loader.exec_module(module)
            return module
    raise ImportError("native cache bridge is missing from this test's Bazel data")


# Import ModelConfig above first so the shared pybind types are registered.
native = load_native_bridge()


LEGACY_INDEX = "indexer_kv"
COMPACT_INDEX = "indexer_kv_c2"


def make_config(ratios, *, compact=True, physical_tokens=128, bounded=False):
    config = ModelConfig()
    config.num_layers = len(ratios)
    config.is_mtp = False
    config.attn_config.tokens_per_block = physical_tokens
    config.attn_config.kernel_tokens_per_block = 128
    config.attn_config.layer_compress_ratios = list(ratios)
    config.attn_config.v41_kv_source_layer_ids = [
        index for index, ratio in enumerate(ratios) if ratio
    ]
    config.attn_config.size_per_head = 512
    config.attn_config.use_mla = True
    config.attn_config.kv_lora_rank = 512
    config.attn_config.rope_head_dim = 64
    config.attn_config.sliding_window = 128
    with patch.dict(
        os.environ,
        {
            "DSV41_COMPACT_INDEXER_KV": "1" if compact else "0",
            "DSV41_SWA_BOUNDED_REPLAY": "1" if bounded else "0",
        },
    ):
        configure_v41_kv_cache(config)
    return config


class V41NativeCacheSpecsTest(unittest.TestCase):
    def assert_index_geometry(self, group, *, physical_tokens, entries, layers):
        kernel_pages = physical_tokens // 128
        self.assertEqual(group["seq_size_per_block"], physical_tokens)
        self.assertEqual(group["kernel_seq_size_per_block"], 128)
        self.assertEqual(group["kernel_blocks_per_kv_block"], kernel_pages)
        self.assertEqual(group["entries_per_kernel_block"], entries)
        self.assertEqual(group["entries_per_block"], entries * kernel_pages)
        self.assertEqual(group["block_stride_bytes"], entries * 68 * kernel_pages)
        self.assertEqual(group["block_payload_bytes"], entries * 68 * kernel_pages)
        self.assertEqual(group["layer_ids"], layers)

    def test_mixed_ratios_resolve_to_distinct_physical_prototypes(self):
        for cp_size in (1, 2, 4):
            for physical_tokens in (128, 256):
                for prefill in (False, True):
                    with self.subTest(
                        cp_size=cp_size,
                        physical_tokens=physical_tokens,
                        prefill=prefill,
                    ):
                        config = make_config(
                            [2, 2, 1, 0], physical_tokens=physical_tokens
                        )
                        groups = native.describe_basic_config(config, cp_size, prefill)
                        self.assert_index_geometry(
                            groups[COMPACT_INDEX],
                            physical_tokens=physical_tokens,
                            entries=64,
                            layers=[0, 1],
                        )
                        self.assert_index_geometry(
                            groups[LEGACY_INDEX],
                            physical_tokens=physical_tokens,
                            entries=128,
                            layers=[2],
                        )

    def test_ratio_two_only_does_not_require_legacy_group(self):
        for cp_size in (1, 2, 4):
            groups = native.describe_basic_config(make_config([2, 0, 2]), cp_size)
            self.assertNotIn(LEGACY_INDEX, groups)
            self.assert_index_geometry(
                groups[COMPACT_INDEX],
                physical_tokens=128,
                entries=64,
                layers=[0, 2],
            )

    def test_ratio_one_only_keeps_existing_geometry(self):
        groups = native.describe_basic_config(make_config([1, 0, 1]))
        self.assertNotIn(COMPACT_INDEX, groups)
        self.assert_index_geometry(
            groups[LEGACY_INDEX],
            physical_tokens=128,
            entries=128,
            layers=[0, 2],
        )

    def test_disabled_compaction_uses_one_legacy_prototype(self):
        for cp_size in (1, 2, 4):
            config = make_config([2, 1, 0], compact=False)
            groups = native.describe_basic_config(config, cp_size)
            self.assertNotIn(COMPACT_INDEX, groups)
            self.assert_index_geometry(
                groups[LEGACY_INDEX],
                physical_tokens=128,
                entries=128,
                layers=[0, 1],
            )

    def test_real_builder_rejects_mixed_geometry_under_one_tag(self):
        config = make_config([2, 1])
        descs = config.kv_cache_spec_descs
        for layer in descs:
            for desc in layer:
                if desc.tag == COMPACT_INDEX:
                    desc.tag = LEGACY_INDEX
        config.kv_cache_spec_descs = descs
        with self.assertRaisesRegex(RuntimeError, "multiple physical prototypes"):
            native.describe_basic_config(config)

    def test_cache_key_namespace_tracks_compact_layout_and_replay_mode(self):
        seeds = {
            make_config([2, 1], compact=compact, bounded=bounded).cache_key_hash_seed
            for compact in (False, True)
            for bounded in (False, True)
        }
        self.assertEqual(len(seeds), 4)


if __name__ == "__main__":
    unittest.main()

"""V4.1 cache ownership and byte-layout contracts on the main descriptors."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class Descriptor:
    pass


def _load_specs():
    root = Path(__file__).resolve().parents[5]
    modules = {}
    for name in ("rtp_llm", "rtp_llm.models"):
        module = ModuleType(name)
        module.__path__ = [str(root.joinpath(*name.split(".")))]
        modules[name] = module
    ops = ModuleType("rtp_llm.ops")
    for name in (
        "CacheCapacityPolicyDesc",
        "CacheCpPolicyDesc",
        "CacheMemoryPolicyDesc",
        "CacheReusePolicyDesc",
        "CacheTailPolicyDesc",
        "KVCacheSpecDesc",
    ):
        setattr(ops, name, Descriptor)
    for name, members in {
        "CacheMemoryPlacement": ("HOST_PINNED",),
        "CpBlockSliceMode": ("PAYLOAD_BYTES", "EQUAL_BYTES"),
        "CpPrefillSliceLayout": ("PAYLOAD", "BLOCK_STRIDE"),
        "DataType": ("TYPE_UINT8", "TYPE_FP32"),
        "KVCacheSpecType": ("OPAQUE_KV", "OPAQUE_STATE"),
        "OpaqueBlockEntryCountMode": ("KERNEL_BLOCK_COMPRESSED", "STATE_RING"),
        "HybridAttentionType": ("NONE",),
    }.items():
        setattr(ops, name, SimpleNamespace(**{member: member for member in members}))
    modules[ops.__name__] = ops
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            "v41_specs_contract", root / "rtp_llm/models/dsv41_kv_cache.py"
        )
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result


SPECS = _load_specs()


def _mock_device_modules():
    device = ModuleType("rtp_llm.device")
    device.__path__ = []
    device_type = ModuleType("rtp_llm.device.device_type")
    device_type.is_sm90 = lambda: False
    device.device_type = device_type
    return {device.__name__: device, device_type.__name__: device_type}


class CacheSpecsTest(unittest.TestCase):
    def test_sm90_indexer_mode_defaults_to_fp8_with_fp4_rollback(self):
        from rtp_llm.models_py.modules.dsv41.attn_type import (
            compact_ratio2_indexer_enabled,
            sm90_indexer_mode,
        )

        device_modules = _mock_device_modules()
        with patch.dict(sys.modules, device_modules), patch.object(
            device_modules["rtp_llm.device.device_type"], "is_sm90", return_value=True
        ):
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(sm90_indexer_mode(), "fp8_dg")
                self.assertTrue(compact_ratio2_indexer_enabled())
            with patch.dict(os.environ, {"DSV41_SM90_INDEXER_MODE": "fp4"}):
                self.assertEqual(sm90_indexer_mode(), "fp4")
                self.assertFalse(compact_ratio2_indexer_enabled())
        with patch.dict(sys.modules, device_modules):
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(sm90_indexer_mode(), "fp4")
            with patch.dict(os.environ, {"DSV41_SM90_INDEXER_MODE": "fp8_dg"}):
                with self.assertRaisesRegex(ValueError, "requires SM90"):
                    sm90_indexer_mode()
        with patch.dict(sys.modules, device_modules), patch.dict(
            os.environ, {"DSV41_SM90_INDEXER_MODE": "fp4_dg"}
        ):
            with self.assertRaises(ValueError):
                sm90_indexer_mode()

    def test_hopper_v4_layout_preserves_ownership_and_indexer_format(self):
        descs = SPECS.build_v41_kv_cache_spec_descs(
            5,
            [0, 2, 2, 1, 1],
            [1, 3],
            512,
            bounded_replay=True,
            v4_kv_layout=True,
        )
        for layer in descs:
            for desc in layer:
                if desc.tag in (
                    "swa_kv",
                    "decoder_swa_kv",
                    "global_kv_1",
                    "global_kv_2",
                ):
                    self.assertEqual(desc.entry_elems, 584)
                    self.assertEqual(desc.block_stride_bytes_alignment, 576)
                elif desc.tag == "indexer_kv":
                    self.assertEqual(desc.entry_elems, 68)
        self.assertFalse(descs[4][0].reuse.enable_prefix_reuse)
        self.assertEqual(len(descs[2]), 1)

    def test_only_producer_layers_allocate_global_and_indexer_storage(self):
        descs = SPECS.build_v41_kv_cache_spec_descs(5, [0, 2, 2, 1, 1], [1, 3], 512)
        self.assertEqual(
            [[d.tag for d in layer] for layer in descs],
            [
                ["swa_kv"],
                ["swa_kv", "global_kv_2", "indexer_kv", "csa_state"],
                ["swa_kv"],
                ["swa_kv", "global_kv_1", "indexer_kv"],
                ["swa_kv"],
            ],
        )
        self.assertEqual(descs[1][1].entry_elems, 288)
        self.assertEqual(descs[1][1].compression_ratio, 2)
        self.assertEqual(descs[3][1].compression_ratio, 1)
        self.assertEqual(descs[1][2].entry_elems, 68)
        self.assertEqual(descs[1][2].compression_ratio, 1)
        self.assertEqual(descs[1][3].entry_elems, 1024)
        self.assertEqual(descs[1][3].compression_ratio, 2)
        self.assertEqual(descs[1][3].state_ring_overlap, 0)

    def test_ratio2_indexer_compaction_uses_distinct_tag_and_64_entries(self):
        from rtp_llm.models_py.modules.dsv41.attn_type import indexer_kv_tag

        ratios = [0, 2, 2, 2, 1, 1]
        sources = [1, 3, 4]
        compact = SPECS.build_v41_kv_cache_spec_descs(
            len(ratios), ratios, sources, 512, compact_ratio2_indexer=True
        )
        baseline = SPECS.build_v41_kv_cache_spec_descs(
            len(ratios), ratios, sources, 512
        )
        source_descs = [
            next(d for d in compact[i] if d.tag.startswith("indexer_kv"))
            for i in sources
        ]
        self.assertEqual(
            [d.tag for d in source_descs],
            ["indexer_kv_2", "indexer_kv_2", "indexer_kv"],
        )
        self.assertEqual([d.compression_ratio for d in source_descs], [2, 2, 1])
        self.assertEqual([128 // d.compression_ratio for d in source_descs], [64, 64, 128])
        self.assertEqual(
            sum(d.entry_elems * (128 // d.compression_ratio) for d in source_descs),
            68 * 256,
        )
        self.assertEqual(
            [d.tag for d in baseline[sources[0]]],
            ["swa_kv", "global_kv_2", "indexer_kv", "csa_state"],
        )
        self.assertTrue(
            all(
                next(d for d in baseline[i] if d.tag == "indexer_kv").compression_ratio
                == 1
                for i in sources
            )
        )
        self.assertEqual(
            [indexer_kv_tag(ratios[i], compact_ratio2=True) for i in sources],
            [d.tag for d in source_descs],
        )
        self.assertEqual(indexer_kv_tag(2), "indexer_kv")
        with self.assertRaises(ValueError):
            indexer_kv_tag(0, compact_ratio2=True)

    def test_bounded_decoder_does_not_publish_swa_prefix(self):
        descs = SPECS.build_v41_kv_cache_spec_descs(
            5, [0, 2, 2, 1, 1], [1, 3], 512, bounded_replay=True
        )
        self.assertEqual(descs[3][0].tag, "swa_kv")
        self.assertEqual(descs[4][0].tag, "decoder_swa_kv")
        self.assertFalse(descs[4][0].reuse.enable_prefix_reuse)
        self.assertEqual(descs[4][0].entry_elems, 528)
        self.assertEqual(descs[4][0].block_stride_bytes_alignment, 16896)
        self.assertEqual(descs[4][0].cp.slice, "EQUAL_BYTES")
        self.assertTrue(descs[4][0].cp.scale_seq_size)

    def test_cache_format_and_replay_modes_have_independent_namespaces(self):
        bounded = ModuleType("rtp_llm.models_py.modules.dsv41.bounded_replay")
        layout = ModuleType("rtp_llm.models_py.modules.dsv41.kv_cache_layout")
        layout.use_v4_kv_cache = lambda: False
        device_modules = _mock_device_modules()
        seeds = []
        for enabled in (False, True):
            bounded.enabled = lambda: enabled
            config = SimpleNamespace(
                attn_config=SimpleNamespace(
                    tokens_per_block=128,
                    kernel_tokens_per_block=128,
                    layer_compress_ratios=[0, 2],
                    v41_kv_source_layer_ids=[1],
                    size_per_head=512,
                ),
                is_mtp=False,
                num_layers=2,
                hybrid_attention_config=SimpleNamespace(),
            )
            with patch.dict(os.environ, {"DSV41_SM90_INDEXER_MODE": "fp4"}), patch.dict(
                sys.modules,
                {bounded.__name__: bounded, layout.__name__: layout, **device_modules},
            ):
                SPECS.configure_v41_kv_cache(config)
            seeds.append(config.cache_key_hash_seed)
            self.assertNotIn(config.cache_key_hash_seed, (0, 0x4453563431525031))
            self.assertEqual(config.cache_min_replay_tokens, 128 if enabled else 1)
        self.assertNotEqual(*seeds)

        with patch.dict(os.environ, {"DSV41_SM90_INDEXER_MODE": "fp8_dg"}), patch.object(
            device_modules["rtp_llm.device.device_type"], "is_sm90", return_value=True
        ), patch.dict(
            sys.modules,
            {bounded.__name__: bounded, layout.__name__: layout, **device_modules},
        ):
            SPECS.configure_v41_kv_cache(config)
        self.assertNotIn(config.cache_key_hash_seed, seeds)
        self.assertEqual(
            [d.tag for d in config.kv_cache_spec_descs[1]],
            ["swa_kv", "global_kv_2", "indexer_kv_2", "csa_state"],
        )
        self.assertEqual(config.kv_cache_spec_descs[1][2].entry_elems, 132)
        self.assertEqual(config.kv_cache_spec_descs[1][2].compression_ratio, 2)

    def test_fp8_primary_indexer_block_budget_has_no_fp4_mirror(self):
        ratios = [0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1]
        sources = [2, 8, 14, 20]
        descs = SPECS.build_v41_kv_cache_spec_descs(
            len(ratios), ratios, sources, 512,
            compact_ratio2_indexer=True, fp8_indexer=True,
        )
        indexers = [
            next(d for d in descs[i] if d.tag.startswith("indexer_kv"))
            for i in sources
        ]
        self.assertEqual([d.entry_elems for d in indexers], [132] * 4)
        self.assertEqual([d.compression_ratio for d in indexers], [2, 2, 2, 1])
        self.assertEqual(
            sum(d.entry_elems * (128 // d.compression_ratio) for d in indexers),
            42_240,
        )

    def test_all_draft_layers_share_nonreusable_swa_spec(self):
        descs = SPECS.build_v41_kv_cache_spec_descs(
            3, [0, 0, 0], [], 512, bounded_replay=True, draft=True
        )
        self.assertEqual(
            [[d.tag for d in layer] for layer in descs], [["decoder_swa_kv"]] * 3
        )


if __name__ == "__main__":
    unittest.main()

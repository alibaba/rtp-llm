"""V4.1 cache ownership and byte-layout contracts on the main descriptors."""

import importlib.util
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


class CacheSpecsTest(unittest.TestCase):
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
            with patch.dict(sys.modules, {bounded.__name__: bounded}):
                SPECS.configure_v41_kv_cache(config)
            seeds.append(config.cache_key_hash_seed)
            self.assertNotIn(config.cache_key_hash_seed, (0, 0x4453563431525031))
            self.assertEqual(config.cache_min_replay_tokens, 128 if enabled else 1)
        self.assertNotEqual(*seeds)

    def test_all_draft_layers_share_nonreusable_swa_spec(self):
        descs = SPECS.build_v41_kv_cache_spec_descs(
            3, [0, 0, 0], [], 512, bounded_replay=True, draft=True
        )
        self.assertEqual(
            [[d.tag for d in layer] for layer in descs], [["decoder_swa_kv"]] * 3
        )


if __name__ == "__main__":
    unittest.main()

import itertools
import json
import os
import tempfile
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models.deepseek_v41 import DeepSeekV41
from rtp_llm.ops import (
    CacheMemoryPlacement,
    CpBlockSliceMode,
    CpPrefillSliceLayout,
    DataType,
    KVCacheSpecType,
    OpaqueBlockEntryCountMode,
)
from rtp_llm.server.server_args.server_args import setup_args


def model_config(draft=False):
    config = ModelConfig()
    config.num_layers = 1 if draft else 5
    config.hidden_size = 512
    config.is_mtp = draft
    config.is_deepseek_v41 = True
    config.attn_config.head_num = 1
    config.attn_config.size_per_head = 512
    config.attn_config.sliding_window = 128
    config.attn_config.tokens_per_block = 128
    config.attn_config.kernel_tokens_per_block = 128
    config.attn_config.layer_compress_ratios = [0] if draft else [0, 2, 0, 1, 0]
    config.attn_config.v41_kv_source_layer_ids = [] if draft else [1, 3]
    return config


def descriptors(config):
    return {d.tag: d for layer in config.kv_cache_spec_descs for d in layer}


def capacity(desc):
    return desc.capacity.explicit_block_num if desc.capacity else None


class Dsv41KvCacheConfigTest(TestCase):
    def test_native_descriptor_contract_for_remote_config_mirror(self):
        # Paired with MirroredNativeV41DescriptorsGiveMatchingPrefillAndDecodeWireConfigs.
        # This checks the real Python generator, not execution of a Python->C++
        # bridge or kernel/transfer. Keep the C++ test-only mirror in sync.
        # The C++ mirror models the legacy indexer layout; compact is the default.
        for compact, bounded in itertools.product((False, True), repeat=2):
            for draft in (False, True):
                with self.subTest(
                    compact=compact, bounded=bounded, draft=draft
                ), patch.dict(
                    os.environ,
                    {
                        "DSV41_SWA_BOUNDED_REPLAY": str(int(bounded)),
                        "DSV41_COMPACT_INDEXER_KV": str(int(compact)),
                    },
                    clear=True,
                ):
                    config = model_config(draft)
                    config.attn_config.tokens_per_block = 256
                    config.gen_num_per_cycle = 5
                    if draft:
                        config.num_layers = 3
                        config.attn_config.layer_compress_ratios = [0, 0, 0]
                    DeepSeekV41._apply_kv_cache_config(config, KVCacheConfig())
                    self.assertEqual(config.attn_config.tokens_per_block, 256)
                    self.assertEqual(config.attn_config.kernel_tokens_per_block, 128)
                    self.assertEqual(
                        config.cache_min_replay_tokens, 134 if bounded else 1
                    )
                    self.assertEqual(
                        config.cache_key_hash_seed,
                        (
                            (0x4453563431494432 if bounded else 0x4453563431494632)
                            if compact
                            else (0x4453563431494431 if bounded else 0x4453563431494631)
                        ),
                    )
                    tags = [
                        [desc.tag for desc in layer]
                        for layer in config.kv_cache_spec_descs
                    ]
                    tail_tag = "decoder_swa_kv" if bounded else "swa_kv"
                    expected = (
                        [[tail_tag]] * 3
                        if draft
                        else [
                            ["swa_kv"],
                            [
                                "swa_kv",
                                "csa_kv",
                                "indexer_kv_c2" if compact else "indexer_kv",
                                "csa_state",
                            ],
                            ["swa_kv"],
                            [tail_tag, "hca_kv", "indexer_kv"],
                            [tail_tag],
                        ]
                    )
                    self.assertEqual(tags, expected)
                    by_tag = descriptors(config)
                    for tag, desc in by_tag.items():
                        with self.subTest(tag=tag):
                            indexer = tag in ("indexer_kv", "indexer_kv_c2")
                            if indexer or tag in ("csa_kv", "hca_kv"):
                                self.assertEqual(
                                    desc.cache_type, KVCacheSpecType.OPAQUE_KV
                                )
                                self.assertEqual(
                                    desc.entry_count_mode,
                                    OpaqueBlockEntryCountMode.KERNEL_BLOCK_COMPRESSED,
                                )
                                self.assertEqual(
                                    desc.entry_elems, 68 if indexer else 288
                                )
                                self.assertEqual(
                                    desc.compression_ratio,
                                    2 if tag in ("csa_kv", "indexer_kv_c2") else 1,
                                )
                                self.assertEqual(
                                    desc.block_stride_bytes_alignment,
                                    0 if indexer else 4608,
                                )
                                self.assertEqual(desc.entry_dtype, DataType.TYPE_UINT8)
                                continue
                            state = tag == "csa_state"
                            self.assertEqual(
                                desc.cache_type, KVCacheSpecType.OPAQUE_STATE
                            )
                            self.assertTrue(desc.is_state_cache)
                            self.assertEqual(
                                desc.entry_count_mode,
                                OpaqueBlockEntryCountMode.STATE_RING,
                            )
                            self.assertEqual(desc.entry_elems, 1024 if state else 528)
                            self.assertEqual(
                                desc.entry_dtype,
                                DataType.TYPE_FP32 if state else DataType.TYPE_UINT8,
                            )
                            ring_base = 129 if tag == "decoder_swa_kv" else 128
                            self.assertEqual(
                                desc.compression_ratio, 2 if state else ring_base
                            )
                            self.assertEqual(desc.state_ring_overlap, 0)
                            self.assertTrue(desc.state_ring_include_gen_num_per_cycle)
                            self.assertEqual(
                                desc.block_stride_bytes_alignment, 0 if state else 16896
                            )
                            self.assertEqual(
                                desc.block_stride_alignment_min_entries, 128
                            )
                            self.assertTrue(desc.cp.scale_seq_size)
                            self.assertTrue(desc.cp.align_payload)
                            self.assertEqual(
                                desc.cp.prefill_slice_layout,
                                (
                                    CpPrefillSliceLayout.PAYLOAD
                                    if state
                                    else CpPrefillSliceLayout.BLOCK_STRIDE
                                ),
                            )
                            self.assertEqual(
                                desc.cp.slice,
                                (
                                    CpBlockSliceMode.PAYLOAD_BYTES
                                    if state
                                    else CpBlockSliceMode.EQUAL_BYTES
                                ),
                            )
                            self.assertEqual(
                                desc.reuse.enable_prefix_reuse,
                                False if tag == "decoder_swa_kv" else None,
                            )

    def test_model_factory_keeps_cli_overrides_after_legacy_hook(self):
        with tempfile.TemporaryDirectory() as ckpt:
            Path(ckpt, "config.json").write_text(
                json.dumps({"torch_dtype": "bfloat16"})
            )
            with patch.dict(
                os.environ,
                {"DSV41_SWA_BOUNDED_REPLAY": "1", "DSV4_FIXED_POOL_BLOCKS": "999"},
                clear=True,
            ):
                configs = setup_args(
                    [
                        "--model_type",
                        "deepseek_v41",
                        "--checkpoint_path",
                        ckpt,
                        "--act_type",
                        "BF16",
                        "--dsv4_fixed_pool_blocks",
                        "768",
                        "--dsv41_decoder_swa_pool_blocks",
                        "128",
                    ]
                )
                with patch.object(
                    DeepSeekV41, "_create_config", return_value=model_config()
                ):
                    config = ModelFactory.create_model_config(
                        model_args=configs.model_args,
                        lora_config=configs.lora_config,
                        kv_cache_config=configs.kv_cache_config,
                        profiling_debug_logging_config=configs.profiling_debug_logging_config,
                        gen_num_per_cycle=5,
                    )
                self.assertEqual(config.cache_min_replay_tokens, 134)
                by_tag = descriptors(config)
                self.assertEqual(capacity(by_tag["swa_kv"]), 768)
                self.assertEqual(capacity(by_tag["csa_state"]), 768)
                self.assertEqual(capacity(by_tag["decoder_swa_kv"]), 128)
                self.assertFalse(by_tag["decoder_swa_kv"].reuse.enable_prefix_reuse)
                self.assertIsNone(capacity(by_tag["hca_kv"]))

    def test_explicit_config_can_replace_defaults_and_fallback_is_idempotent(self):
        with patch.dict(os.environ, {"DSV41_SWA_BOUNDED_REPLAY": "1"}, clear=True):
            for draft in (False, True):
                with self.subTest(draft=draft):
                    config = model_config(draft)
                    DeepSeekV41._post_build_model_config(config)
                    settings = KVCacheConfig()
                    settings.dsv4_fixed_pool_blocks = 64
                    settings.dsv41_decoder_swa_pool_blocks = 16
                    settings.dsv4_fixed_pool_use_memory = True
                    DeepSeekV41._apply_kv_cache_config(config, settings)
                    for _ in range(2):
                        DeepSeekV41._post_build_model_config(config)
                    by_tag = descriptors(config)
                    self.assertEqual(capacity(by_tag["decoder_swa_kv"]), 16)
                    for tag in ("csa_state", "swa_kv", "decoder_swa_kv"):
                        if tag in by_tag:
                            self.assertEqual(
                                by_tag[tag].memory.placement,
                                CacheMemoryPlacement.HOST_PINNED,
                            )
                            self.assertFalse(
                                by_tag[tag].capacity.charge_to_paged_budget
                            )
                    settings.dsv41_decoder_swa_pool_blocks = 0
                    settings.dsv4_fixed_pool_use_memory = False
                    DeepSeekV41._apply_kv_cache_config(config, settings)
                    self.assertEqual(
                        capacity(descriptors(config)["decoder_swa_kv"]), 64
                    )
                    settings.dsv4_fixed_pool_blocks = 0
                    DeepSeekV41._apply_kv_cache_config(config, settings)
                    self.assertIsNone(capacity(descriptors(config)["decoder_swa_kv"]))

    def test_single_pool_override_without_shared_fixed_count(self):
        with patch.dict(os.environ, {"DSV41_SWA_BOUNDED_REPLAY": "1"}, clear=True):
            config = model_config()
            settings = KVCacheConfig()
            settings.dsv41_decoder_swa_pool_blocks = 32
            DeepSeekV41._apply_kv_cache_config(config, settings)
            by_tag = descriptors(config)
            self.assertEqual(capacity(by_tag["decoder_swa_kv"]), 32)
            self.assertIsNone(capacity(by_tag["swa_kv"]))
            self.assertIsNone(capacity(by_tag["csa_state"]))

    def test_override_rejects_missing_decoder_pool_and_sentinel_only_capacity(self):
        settings = KVCacheConfig()
        settings.dsv41_decoder_swa_pool_blocks = 16
        with patch.dict(os.environ, {"DSV41_SWA_BOUNDED_REPLAY": "0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "requires bounded replay"):
                DeepSeekV41._apply_kv_cache_config(model_config(), settings)
        settings.dsv41_decoder_swa_pool_blocks = 1
        with patch.dict(os.environ, {"DSV41_SWA_BOUNDED_REPLAY": "1"}, clear=True):
            with self.assertRaisesRegex(ValueError, "block zero"):
                DeepSeekV41._apply_kv_cache_config(model_config(), settings)


if __name__ == "__main__":
    main()

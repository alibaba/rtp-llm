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
from rtp_llm.ops import CacheMemoryPlacement
from rtp_llm.server.server_args.server_args import setup_args


def model_config(draft=False):
    config = ModelConfig()
    config.num_layers = 1 if draft else 5
    config.hidden_size = 512
    config.is_mtp = draft
    config.is_deepseek_v41 = True
    config.attn_config.head_num = 1
    config.attn_config.size_per_head = 512
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
                    )
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

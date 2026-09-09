from types import SimpleNamespace

import os
from unittest import TestCase, main
from unittest.mock import patch

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.deepseek_v4 import DeepSeekV4
from rtp_llm.models.dsv4_kv_cache import (
    CSA_KV_TAG,
    CSA_STATE_TAG,
    DSV4_FIXED_POOL_TAGS,
    DSV4_FP8_INDEXER_ENTRY_BYTES,
    DSV4_FP8_KV_ENTRY_BYTES,
    DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES,
    DSV4_HCA_STATE_POOL_BLOCKS,
    DSV4_SWA_WINDOW_ENTRIES,
    DSV4_TOKENS_PER_BLOCK,
    HCA_KV_TAG,
    HCA_STATE_TAG,
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
    SWA_KV_TAG,
    Dsv4IndexerCacheMode,
    apply_dsv4_explicit_pool_blocks,
    build_dsv4_kv_cache_spec_descs,
)
from rtp_llm.ops import (
    CacheEvictPolicy,
    CacheMemoryPlacement,
    CpBlockSliceMode,
    CpPrefillSliceLayout,
    DataType,
    HybridAttentionType,
    KvCacheDataType,
    KVCacheSpecDesc,
    KVCacheSpecType,
    OpaqueBlockEntryCountMode,
)

# CSA / HCA / SWA / SWA / CSA -- covers every routing branch plus a repeat.
LAYER_COMPRESS_RATIOS = [4, 128, 0, 0, 4]
HEAD_DIM = 512
INDEXER_HEAD_DIM = 128
FRAMEWORK_DEFAULT_TOKENS_PER_BLOCK = 64


class Dsv4KvCacheSpecTest(TestCase):
    def _build(self, fp8_kv=True, fixed_pool_use_host_memory=False):
        return build_dsv4_kv_cache_spec_descs(
            layer_num=len(LAYER_COMPRESS_RATIOS),
            layer_compress_ratios=LAYER_COMPRESS_RATIOS,
            fp8_kv=fp8_kv,
            head_dim=HEAD_DIM,
            indexer_head_dim=INDEXER_HEAD_DIM,
            fixed_pool_use_host_memory=fixed_pool_use_host_memory,
        )

    def _by_tag(self, layer_descs):
        return {desc.tag: desc for descs in layer_descs for desc in descs}

    def test_layer_routing_tags(self):
        layer_descs = self._build()
        self.assertEqual(
            [[desc.tag for desc in descs] for descs in layer_descs],
            [
                [
                    CSA_KV_TAG,
                    INDEXER_KV_TAG,
                    INDEXER_STATE_TAG,
                    CSA_STATE_TAG,
                    SWA_KV_TAG,
                ],
                [HCA_KV_TAG, HCA_STATE_TAG, SWA_KV_TAG],
                [SWA_KV_TAG],
                [SWA_KV_TAG],
                [
                    CSA_KV_TAG,
                    INDEXER_KV_TAG,
                    INDEXER_STATE_TAG,
                    CSA_STATE_TAG,
                    SWA_KV_TAG,
                ],
            ],
        )

    def test_layers_past_compress_ratios_are_swa_only(self):
        layer_descs = build_dsv4_kv_cache_spec_descs(
            layer_num=len(LAYER_COMPRESS_RATIOS) + 2,
            layer_compress_ratios=LAYER_COMPRESS_RATIOS,
            fp8_kv=True,
            head_dim=HEAD_DIM,
            indexer_head_dim=INDEXER_HEAD_DIM,
        )
        self.assertEqual([desc.tag for desc in layer_descs[-1]], [SWA_KV_TAG])
        self.assertEqual([desc.tag for desc in layer_descs[-2]], [SWA_KV_TAG])

    def test_cache_types_and_state_flags(self):
        by_tag = self._by_tag(self._build())
        for tag in (CSA_KV_TAG, HCA_KV_TAG, INDEXER_KV_TAG):
            self.assertEqual(by_tag[tag].cache_type, KVCacheSpecType.OPAQUE_KV, tag)
            self.assertFalse(by_tag[tag].is_state_cache, tag)
            self.assertEqual(
                by_tag[tag].entry_count_mode,
                OpaqueBlockEntryCountMode.KERNEL_BLOCK_COMPRESSED,
                tag,
            )
            self.assertEqual(by_tag[tag].dtype, DataType.TYPE_UINT8, tag)
            self.assertEqual(by_tag[tag].entry_dtype, DataType.TYPE_UINT8, tag)
        for tag in (INDEXER_STATE_TAG, CSA_STATE_TAG, HCA_STATE_TAG, SWA_KV_TAG):
            self.assertEqual(by_tag[tag].cache_type, KVCacheSpecType.OPAQUE_STATE, tag)
            self.assertTrue(by_tag[tag].is_state_cache, tag)
            self.assertEqual(
                by_tag[tag].entry_count_mode, OpaqueBlockEntryCountMode.STATE_RING, tag
            )
            self.assertTrue(by_tag[tag].state_ring_include_gen_num_per_cycle, tag)
            self.assertEqual(
                by_tag[tag].reuse.evict_policy, CacheEvictPolicy.INDEPENDENT
            )
        for tag in (INDEXER_STATE_TAG, CSA_STATE_TAG, HCA_STATE_TAG):
            self.assertEqual(by_tag[tag].dtype, DataType.TYPE_FP32, tag)
            self.assertEqual(by_tag[tag].entry_dtype, DataType.TYPE_FP32, tag)
        self.assertEqual(by_tag[SWA_KV_TAG].dtype, DataType.TYPE_UINT8)

    def test_fp8_entry_elems(self):
        by_tag = self._by_tag(self._build(fp8_kv=True))
        self.assertEqual(by_tag[CSA_KV_TAG].entry_elems, DSV4_FP8_KV_ENTRY_BYTES)
        self.assertEqual(by_tag[HCA_KV_TAG].entry_elems, DSV4_FP8_KV_ENTRY_BYTES)
        self.assertEqual(by_tag[SWA_KV_TAG].entry_elems, DSV4_FP8_KV_ENTRY_BYTES)
        self.assertEqual(
            by_tag[INDEXER_KV_TAG].entry_elems, DSV4_FP8_INDEXER_ENTRY_BYTES
        )
        self.assertEqual(by_tag[INDEXER_STATE_TAG].entry_elems, 4 * INDEXER_HEAD_DIM)
        self.assertEqual(by_tag[CSA_STATE_TAG].entry_elems, 4 * HEAD_DIM)
        self.assertEqual(by_tag[HCA_STATE_TAG].entry_elems, 2 * HEAD_DIM)

    def test_non_fp8_entry_elems(self):
        by_tag = self._by_tag(self._build(fp8_kv=False))
        self.assertEqual(by_tag[CSA_KV_TAG].entry_elems, HEAD_DIM * 2)
        self.assertEqual(by_tag[HCA_KV_TAG].entry_elems, HEAD_DIM * 2)
        self.assertEqual(by_tag[SWA_KV_TAG].entry_elems, HEAD_DIM * 2)
        self.assertEqual(by_tag[INDEXER_KV_TAG].entry_elems, INDEXER_HEAD_DIM * 2)
        # Fixed-state entry sizes do not depend on the KV dtype.
        self.assertEqual(by_tag[INDEXER_STATE_TAG].entry_elems, 4 * INDEXER_HEAD_DIM)
        self.assertEqual(by_tag[CSA_STATE_TAG].entry_elems, 4 * HEAD_DIM)
        self.assertEqual(by_tag[HCA_STATE_TAG].entry_elems, 2 * HEAD_DIM)

    def test_compression_ratio_and_state_ring_overlap(self):
        by_tag = self._by_tag(self._build())
        self.assertEqual(by_tag[CSA_KV_TAG].compression_ratio, 4)
        self.assertEqual(by_tag[INDEXER_KV_TAG].compression_ratio, 4)
        self.assertEqual(by_tag[HCA_KV_TAG].compression_ratio, 128)
        self.assertEqual(by_tag[INDEXER_STATE_TAG].compression_ratio, 4)
        self.assertEqual(by_tag[CSA_STATE_TAG].compression_ratio, 4)
        self.assertEqual(by_tag[HCA_STATE_TAG].compression_ratio, 128)
        self.assertEqual(by_tag[SWA_KV_TAG].compression_ratio, DSV4_SWA_WINDOW_ENTRIES)

        self.assertEqual(by_tag[INDEXER_STATE_TAG].state_ring_overlap, 1)
        self.assertEqual(by_tag[CSA_STATE_TAG].state_ring_overlap, 1)
        self.assertEqual(by_tag[HCA_STATE_TAG].state_ring_overlap, 0)
        self.assertEqual(by_tag[SWA_KV_TAG].state_ring_overlap, 0)

    def test_576_alignment_only_on_584_byte_pools(self):
        fp8_by_tag = self._by_tag(self._build(fp8_kv=True))
        for tag in (CSA_KV_TAG, HCA_KV_TAG, SWA_KV_TAG):
            self.assertEqual(fp8_by_tag[tag].entry_elems, DSV4_FP8_KV_ENTRY_BYTES, tag)
            self.assertEqual(
                fp8_by_tag[tag].block_stride_bytes_alignment,
                DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES,
                tag,
            )
        for tag in (
            INDEXER_KV_TAG,
            INDEXER_STATE_TAG,
            CSA_STATE_TAG,
            HCA_STATE_TAG,
        ):
            self.assertEqual(fp8_by_tag[tag].block_stride_bytes_alignment, 0, tag)

        non_fp8_by_tag = self._by_tag(self._build(fp8_kv=False))
        for desc in non_fp8_by_tag.values():
            self.assertEqual(desc.block_stride_bytes_alignment, 0, desc.tag)

    def test_block_stride_alignment_min_entries_only_on_state_pools(self):
        by_tag = self._by_tag(self._build())
        for tag in (INDEXER_STATE_TAG, CSA_STATE_TAG, HCA_STATE_TAG, SWA_KV_TAG):
            self.assertEqual(
                by_tag[tag].block_stride_alignment_min_entries,
                DSV4_SWA_WINDOW_ENTRIES,
                tag,
            )
        # Compressed pools must keep 0 here: with 128 the 576-byte alignment
        # above would be skipped (they hold fewer than 128 entries per block).
        for tag in (CSA_KV_TAG, HCA_KV_TAG, INDEXER_KV_TAG):
            self.assertEqual(by_tag[tag].block_stride_alignment_min_entries, 0, tag)

    def test_cp_policy_per_pool(self):
        by_tag = self._by_tag(self._build())
        for tag in (INDEXER_STATE_TAG, CSA_STATE_TAG, HCA_STATE_TAG):
            cp = by_tag[tag].cp
            self.assertIsNotNone(cp, tag)
            self.assertTrue(cp.align_payload, tag)
            self.assertTrue(cp.scale_seq_size, tag)
            self.assertEqual(cp.prefill_slice_layout, CpPrefillSliceLayout.PAYLOAD, tag)
            self.assertEqual(cp.slice, CpBlockSliceMode.PAYLOAD_BYTES, tag)

        swa_cp = by_tag[SWA_KV_TAG].cp
        self.assertIsNotNone(swa_cp)
        self.assertTrue(swa_cp.align_payload)
        self.assertTrue(swa_cp.scale_seq_size)
        self.assertEqual(swa_cp.prefill_slice_layout, CpPrefillSliceLayout.BLOCK_STRIDE)
        self.assertEqual(swa_cp.slice, CpBlockSliceMode.EQUAL_BYTES)

        # Compressed pools carry no cp policy at all.
        for tag in (CSA_KV_TAG, HCA_KV_TAG, INDEXER_KV_TAG):
            self.assertIsNone(by_tag[tag].cp, tag)

    def test_hca_state_capacity_reuse_and_tail(self):
        by_tag = self._by_tag(self._build())
        hca_state = by_tag[HCA_STATE_TAG]
        self.assertIsNotNone(hca_state.capacity)
        self.assertEqual(
            hca_state.capacity.explicit_block_num, DSV4_HCA_STATE_POOL_BLOCKS
        )
        self.assertTrue(hca_state.capacity.charge_to_paged_budget)
        self.assertFalse(hca_state.reuse.enable_prefix_reuse)
        self.assertIsNotNone(hca_state.tail)
        self.assertEqual(hca_state.tail.active_tail_blocks, 1)
        self.assertFalse(hca_state.tail.validate_tail_blocks)

        for tag in (
            CSA_KV_TAG,
            HCA_KV_TAG,
            INDEXER_KV_TAG,
            INDEXER_STATE_TAG,
            CSA_STATE_TAG,
            SWA_KV_TAG,
        ):
            self.assertIsNone(by_tag[tag].capacity, tag)
            self.assertIsNone(by_tag[tag].tail, tag)

    def test_no_memory_placement_by_default(self):
        for desc in self._by_tag(self._build()).values():
            self.assertIsNone(desc.memory, desc.tag)

    def test_host_pinned_fixed_pools_leave_paged_budget(self):
        by_tag = self._by_tag(self._build(fixed_pool_use_host_memory=True))
        for tag in (INDEXER_STATE_TAG, CSA_STATE_TAG, HCA_STATE_TAG, SWA_KV_TAG):
            desc = by_tag[tag]
            self.assertIsNotNone(desc.memory, tag)
            self.assertEqual(
                desc.memory.placement, CacheMemoryPlacement.HOST_PINNED, tag
            )
            self.assertIsNotNone(desc.capacity, tag)
            self.assertFalse(desc.capacity.charge_to_paged_budget, tag)
        # hca_state keeps its explicit sizing while leaving the HBM budget.
        self.assertEqual(
            by_tag[HCA_STATE_TAG].capacity.explicit_block_num,
            DSV4_HCA_STATE_POOL_BLOCKS,
        )
        # Compressed pools stay on device.
        for tag in (CSA_KV_TAG, HCA_KV_TAG, INDEXER_KV_TAG):
            self.assertIsNone(by_tag[tag].memory, tag)
            self.assertIsNone(by_tag[tag].capacity, tag)

    def test_explicit_pool_blocks_helper(self):
        layer_descs = self._build()
        apply_dsv4_explicit_pool_blocks(layer_descs, SWA_KV_TAG, 512)
        swa = self._by_tag(layer_descs)[SWA_KV_TAG]
        self.assertEqual(swa.capacity.explicit_block_num, 512)
        self.assertTrue(swa.capacity.charge_to_paged_budget)

    def test_explicit_pool_blocks_helper_keeps_host_pool_off_budget(self):
        layer_descs = self._build(fixed_pool_use_host_memory=True)
        apply_dsv4_explicit_pool_blocks(layer_descs, SWA_KV_TAG, 512)
        swa = self._by_tag(layer_descs)[SWA_KV_TAG]
        self.assertEqual(swa.capacity.explicit_block_num, 512)
        self.assertFalse(swa.capacity.charge_to_paged_budget)

    def test_rejects_zero_layers(self):
        with self.assertRaises(ValueError):
            build_dsv4_kv_cache_spec_descs(
                layer_num=0,
                layer_compress_ratios=[],
                fp8_kv=True,
                head_dim=HEAD_DIM,
                indexer_head_dim=INDEXER_HEAD_DIM,
            )


class Dsv4PostBuildModelConfigTest(TestCase):
    def _model_config(self, tokens_per_block=FRAMEWORK_DEFAULT_TOKENS_PER_BLOCK):
        config = ModelConfig()
        config.num_layers = len(LAYER_COMPRESS_RATIOS)
        config.attn_config.size_per_head = HEAD_DIM
        config.attn_config.indexer_head_dim = INDEXER_HEAD_DIM
        config.attn_config.kv_cache_dtype = KvCacheDataType.FP8
        config.attn_config.layer_compress_ratios = LAYER_COMPRESS_RATIOS
        config.attn_config.tokens_per_block = tokens_per_block
        config.attn_config.kernel_tokens_per_block = tokens_per_block
        return config

    def test_explicit_state_format_does_not_probe_or_register_a_device(self):
        config = self._model_config()
        with (
            patch(
                "rtp_llm.utils.backend_registry.run_backend_registrations",
                side_effect=AssertionError("metadata must not register device code"),
            ),
            patch(
                "torch.cuda.get_device_name",
                side_effect=AssertionError("metadata must not probe a device"),
            ),
        ):
            DeepSeekV4._post_build_model_config(
                config, indexer_cache_mode=Dsv4IndexerCacheMode.FP8
            )
        entries = {
            desc.entry_elems
            for layer in config.kv_cache_spec_descs
            for desc in layer
            if desc.tag == INDEXER_KV_TAG
        }
        self.assertEqual(entries, {DSV4_FP8_INDEXER_ENTRY_BYTES})

    def test_model_factory_passes_the_declared_format_in_module_mode(self):
        from rtp_llm.config.module_dispatch_config import ModuleDispatchConfig
        from rtp_llm.model_factory import ModelFactory

        config = self._model_config()
        with (
            patch.object(ModelFactory, "get_model_cls", return_value=DeepSeekV4),
            patch.object(DeepSeekV4, "_create_config", return_value=config),
            patch("rtp_llm.model_factory.build_model_config"),
            patch(
                "rtp_llm.utils.backend_registry.run_backend_registrations",
                side_effect=AssertionError(
                    "module metadata must not use legacy registration"
                ),
            ),
            patch(
                "torch.cuda.get_device_name",
                side_effect=AssertionError("module metadata must not probe a device"),
            ),
        ):
            actual = ModelFactory.create_model_config(
                model_args=SimpleNamespace(
                    model_type="deepseek_v4", ckpt_path="unused"
                ),
                lora_config=SimpleNamespace(lora_info=""),
                kv_cache_config=None,
                profiling_debug_logging_config=None,
                module_dispatch_config=ModuleDispatchConfig(mode="auto"),
            )
        self.assertIs(actual, config)
        self.assertTrue(config.kv_cache_spec_descs)

    def test_post_build_enables_independent_pools(self):
        config = self._model_config()

        DeepSeekV4._post_build_model_config(config)

        self.assertTrue(
            config.hybrid_attention_config.enable_independent_kv_cache_pools
        )
        self.assertEqual(
            list(config.hybrid_attention_config.hybrid_attention_types),
            [HybridAttentionType.NONE] * config.num_layers,
        )
        self.assertEqual(len(config.kv_cache_spec_descs), config.num_layers)
        self.assertEqual(
            [desc.tag for desc in config.kv_cache_spec_descs[1]],
            [HCA_KV_TAG, HCA_STATE_TAG, SWA_KV_TAG],
        )

    def test_post_build_promotes_default_block_size(self):
        config = self._model_config()

        DeepSeekV4._post_build_model_config(config)

        self.assertEqual(config.attn_config.tokens_per_block, DSV4_TOKENS_PER_BLOCK)
        self.assertEqual(
            config.attn_config.kernel_tokens_per_block, DSV4_TOKENS_PER_BLOCK
        )

    def test_post_build_keeps_explicit_block_size(self):
        config = self._model_config(tokens_per_block=128)

        DeepSeekV4._post_build_model_config(config)

        self.assertEqual(config.attn_config.tokens_per_block, 128)
        self.assertEqual(config.attn_config.kernel_tokens_per_block, 128)

    def test_post_build_does_not_override_existing_descs(self):
        config = self._model_config()
        sentinel = KVCacheSpecDesc()
        sentinel.tag = "sentinel"
        sentinel.cache_type = KVCacheSpecType.MHA
        config.kv_cache_spec_descs = [[sentinel]] * config.num_layers

        DeepSeekV4._post_build_model_config(config)

        self.assertEqual(config.kv_cache_spec_descs[0][0].tag, "sentinel")
        self.assertEqual(
            config.attn_config.tokens_per_block, FRAMEWORK_DEFAULT_TOKENS_PER_BLOCK
        )

    def _by_tag(self, layer_descs):
        return {desc.tag: desc for descs in layer_descs for desc in descs}

    def _parse_kv_cache_cli(self, args):
        from rtp_llm.server.server_args.kv_cache_group_args import (
            init_kv_cache_group_args,
        )
        from rtp_llm.server.server_args.server_args import EnvArgumentParser

        kv_cache_config = KVCacheConfig()
        parser = EnvArgumentParser()
        parser.set_root_config(kv_cache_config)
        init_kv_cache_group_args(parser, kv_cache_config)
        parser.parse_args(args)
        return kv_cache_config

    def test_runtime_hook_preserves_legacy_one_argument_override(self):
        class ExternalModel(BaseModel):
            @classmethod
            def _post_build_model_config(cls, model_config):
                model_config.model_name = "legacy-hook-called"

        config = ModelConfig()
        ExternalModel._apply_kv_cache_config(config, KVCacheConfig())
        ExternalModel._post_build_model_config(config)

        self.assertEqual(config.model_name, "legacy-hook-called")

    def test_post_build_cli_fixed_pool_blocks_reach_final_descriptors(self):
        with patch.dict(
            os.environ,
            {
                "DSV4_FIXED_POOL_BLOCKS": "999",
                "DSV4_HCA_STATE_POOL_BLOCKS": "888",
            },
            clear=True,
        ):
            kv_cache_config = self._parse_kv_cache_cli(
                [
                    "--dsv4_fixed_pool_blocks",
                    "777",
                    "--dsv4_hca_state_pool_blocks",
                    "321",
                ]
            )
            config = self._model_config()
            DeepSeekV4._apply_kv_cache_config(config, kv_cache_config)

        by_tag = self._by_tag(config.kv_cache_spec_descs)
        self.assertEqual(by_tag[HCA_STATE_TAG].capacity.explicit_block_num, 321)
        for tag in DSV4_FIXED_POOL_TAGS:
            if tag != HCA_STATE_TAG:
                self.assertEqual(by_tag[tag].capacity.explicit_block_num, 777)

    def test_post_build_cli_host_memory_reaches_final_descriptors(self):
        with patch.dict(
            os.environ, {"DSV4_FIXED_POOL_USE_MEMORY": "false"}, clear=True
        ):
            kv_cache_config = self._parse_kv_cache_cli(
                ["--dsv4_fixed_pool_use_memory", "true"]
            )
            config = self._model_config()
            DeepSeekV4._apply_kv_cache_config(config, kv_cache_config)

        by_tag = self._by_tag(config.kv_cache_spec_descs)
        for tag in DSV4_FIXED_POOL_TAGS:
            self.assertEqual(
                by_tag[tag].memory.placement, CacheMemoryPlacement.HOST_PINNED
            )
            self.assertFalse(by_tag[tag].capacity.charge_to_paged_budget)

    def test_legacy_post_build_reads_host_memory_environment(self):
        config = self._model_config()
        with patch.dict(
            os.environ, {"DSV4_FIXED_POOL_USE_MEMORY": "true"}, clear=False
        ):
            DeepSeekV4._post_build_model_config(config)

        by_tag = self._by_tag(config.kv_cache_spec_descs)
        for tag in DSV4_FIXED_POOL_TAGS:
            self.assertEqual(
                by_tag[tag].memory.placement, CacheMemoryPlacement.HOST_PINNED
            )

    def test_post_build_fixed_pool_blocks_env_pins_all_fixed_pools(self):
        config = self._model_config()
        with patch.dict(os.environ, {"DSV4_FIXED_POOL_BLOCKS": "777"}, clear=False):
            DeepSeekV4._post_build_model_config(config)

        by_tag = self._by_tag(config.kv_cache_spec_descs)
        for tag in DSV4_FIXED_POOL_TAGS:
            self.assertEqual(by_tag[tag].capacity.explicit_block_num, 777)

    def test_post_build_hca_env_overrides_only_hca_state(self):
        config = self._model_config()
        with patch.dict(
            os.environ,
            {"DSV4_FIXED_POOL_BLOCKS": "777", "DSV4_HCA_STATE_POOL_BLOCKS": "321"},
            clear=False,
        ):
            DeepSeekV4._post_build_model_config(config)

        by_tag = self._by_tag(config.kv_cache_spec_descs)
        self.assertEqual(by_tag[HCA_STATE_TAG].capacity.explicit_block_num, 321)
        for tag in DSV4_FIXED_POOL_TAGS:
            if tag == HCA_STATE_TAG:
                continue
            self.assertEqual(by_tag[tag].capacity.explicit_block_num, 777)

    def test_post_build_without_env_keeps_default_hca_capacity(self):
        config = self._model_config()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DSV4_FIXED_POOL_BLOCKS", None)
            os.environ.pop("DSV4_HCA_STATE_POOL_BLOCKS", None)
            DeepSeekV4._post_build_model_config(config)

        by_tag = self._by_tag(config.kv_cache_spec_descs)
        self.assertEqual(
            by_tag[HCA_STATE_TAG].capacity.explicit_block_num,
            DSV4_HCA_STATE_POOL_BLOCKS,
        )


if __name__ == "__main__":
    main()

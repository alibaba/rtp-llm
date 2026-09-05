import json
import os
import tempfile
from unittest import TestCase, main
from unittest.mock import patch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.deepseek_v4 import DeepSeekV4, _require_n_shared_experts
from rtp_llm.models.dsv4_kv_cache import (
    CSA_KV_TAG,
    CSA_STATE_TAG,
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
    apply_dsv4_explicit_pool_blocks,
    build_dsv4_kv_cache_spec_descs,
)
from rtp_llm.models_py.model_desc.deepseek_v4_model import _args_from_model_config
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


class Dsv4MoeConfigTest(TestCase):
    @staticmethod
    def _minimal_config_json(n_shared_experts: int) -> dict:
        return {
            "num_hidden_layers": 1,
            "hidden_size": 512,
            "vocab_size": 64,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "compress_ratios": [0],
            "o_groups": 1,
            "o_lora_rank": 0,
            "index_head_dim": 128,
            "index_n_heads": 1,
            "index_topk": 4,
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 1.0,
            "num_experts_per_tok": 4,
            "n_routed_experts": 16,
            "moe_intermediate_size": 256,
            "n_shared_experts": n_shared_experts,
        }

    @classmethod
    def _load_minimal_model_config(cls, n_shared_experts: int) -> ModelConfig:
        with tempfile.TemporaryDirectory() as ckpt_path:
            with open(os.path.join(ckpt_path, "config.json"), "w") as writer:
                json.dump(cls._minimal_config_json(n_shared_experts), writer)
            return DeepSeekV4._create_config(ckpt_path)

    def test_missing_shared_expert_count_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "missing required field"):
            _require_n_shared_experts({})

    def test_explicit_zero_keeps_routed_only_checkpoint_supported(self):
        self.assertEqual(_require_n_shared_experts({"n_shared_experts": 0}), 0)

    def test_one_shared_expert_is_supported(self):
        self.assertEqual(_require_n_shared_experts({"n_shared_experts": 1}), 1)

    def test_multiple_shared_experts_are_supported(self):
        self.assertEqual(_require_n_shared_experts({"n_shared_experts": 2}), 2)

    def test_shared_expert_count_rejects_invalid_values(self):
        for value in (True, False, 0.5, 1.0, "1", None, -1):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "non-negative integer"
            ):
                _require_n_shared_experts({"n_shared_experts": value})

    def test_routed_only_config_propagates_through_model_build_inputs(self):
        model_config = self._load_minimal_model_config(0)

        self.assertEqual(model_config.n_shared_experts, 0)
        self.assertEqual(model_config.inter_size, 0)
        args = _args_from_model_config(model_config)
        self.assertEqual(args.n_shared_experts, 0)
        self.assertEqual(args.moe_inter_dim, 256)

    def test_multiple_shared_experts_keep_routed_expert_width(self):
        model_config = self._load_minimal_model_config(2)

        self.assertEqual(model_config.n_shared_experts, 2)
        self.assertEqual(model_config.inter_size, 512)
        args = _args_from_model_config(model_config)
        self.assertEqual(args.n_shared_experts, 2)
        self.assertEqual(args.moe_inter_dim, 256)

    def test_zero_routed_expert_width_fails_fast(self):
        model_config = self._load_minimal_model_config(0)
        model_config.moe_inter_size = 0

        with self.assertRaisesRegex(ValueError, "positive routed expert width"):
            _args_from_model_config(model_config)

    def test_inconsistent_shared_expert_width_fails_fast(self):
        model_config = self._load_minimal_model_config(2)
        model_config.inter_size = 256

        with self.assertRaisesRegex(ValueError, "shared-expert width is inconsistent"):
            _args_from_model_config(model_config)

    def test_shared_width_can_recover_missing_routed_width(self):
        model_config = self._load_minimal_model_config(2)
        model_config.moe_inter_size = 0

        self.assertEqual(_args_from_model_config(model_config).moe_inter_dim, 256)

    def test_production_weight_graph_tracks_shared_expert_config(self):
        import torch

        from rtp_llm.config.kv_cache_config import KVCacheConfig
        from rtp_llm.ops import HWKernelConfig, ParallelismConfig
        from rtp_llm.utils.database import CkptDatabase
        from rtp_llm.utils.model_weight import W

        for n_shared_experts in (0, 1, 2):
            with self.subTest(n_shared_experts=n_shared_experts):
                model_config = self._load_minimal_model_config(n_shared_experts)
                parallelism_config = ParallelismConfig()
                parallelism_config.tp_size = 1
                parallelism_config.tp_rank = 0
                parallelism_config.dp_size = 1
                parallelism_config.dp_rank = 0
                parallelism_config.ep_size = 1
                parallelism_config.ep_rank = 0
                parallelism_config.world_size = 1
                parallelism_config.world_rank = 0
                parallelism_config.local_world_size = 1

                with tempfile.TemporaryDirectory() as checkpoint_path:
                    # A real checkpoint database drives the same constructor,
                    # metadata processing, descriptor construction, and file
                    # filtering path used by BaseModel.create_model_loader().
                    torch.save(
                        {"embed.weight": torch.ones(1)},
                        os.path.join(checkpoint_path, "model.bin"),
                    )
                    database = CkptDatabase(checkpoint_path)
                    weight_builder = DeepSeekV4.get_weight_cls()(
                        model_config=model_config,
                        parallelism_config=parallelism_config,
                        hw_kernel_config=HWKernelConfig(),
                        kv_cache_config=KVCacheConfig(),
                    )
                    weight_info = weight_builder.create_model_weight_info(database)

                self.assertEqual(weight_builder._n_shared_experts, n_shared_experts)
                layer_by_name = {
                    weight.name: weight for weight in weight_info.layer_weights[0]
                }
                shared_names = {W.v4_shared_w13_w, W.v4_shared_w2_w}
                if n_shared_experts == 0:
                    self.assertTrue(shared_names.isdisjoint(layer_by_name))
                else:
                    self.assertTrue(shared_names.issubset(layer_by_name))
                    self.assertEqual(
                        {
                            ckpt.tensor_name(0)
                            for ckpt in layer_by_name[W.v4_shared_w13_w].weights
                        },
                        {
                            "layers.0.ffn.shared_experts.w1.weight",
                            "layers.0.ffn.shared_experts.w3.weight",
                        },
                    )
                    self.assertEqual(
                        {
                            ckpt.tensor_name(0)
                            for ckpt in layer_by_name[W.v4_shared_w2_w].weights
                        },
                        {"layers.0.ffn.shared_experts.w2.weight"},
                    )

    def test_public_strategy_unsupported_by_dsv4_falls_back_to_auto(self):
        from rtp_llm.server.server_args import server_args

        with patch.dict(os.environ, {}, clear=True):
            public_config = server_args.setup_args(
                ["--moe_strategy", "fp8_per_block_no_dp"]
            ).moe_config
        with self.assertLogs(level="WARNING") as logs:
            args = _args_from_model_config(
                self._load_minimal_model_config(1),
                moe_config=public_config,
            )

        self.assertEqual(public_config.moe_strategy, "fp8_per_block_no_dp")
        self.assertEqual(args.moe_strategy, "auto")
        self.assertIn("does not support MOE_STRATEGY", "\n".join(logs.output))

    def test_public_strategy_reaches_v4_block_and_moe_selection(self):
        import torch
        import torch.nn as nn

        from rtp_llm.models_py.modules.dsv4.block import Block
        from rtp_llm.models_py.modules.dsv4.transformer import _block_kwargs
        from rtp_llm.server.server_args import server_args
        from rtp_llm.utils.model_weight import W

        class DummyModule(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

        class DummyMoe(DummyModule):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.strategy_name = kwargs["strategy"]

        weight_keys = (
            W.v4_attn_norm,
            W.v4_ffn_norm,
            W.v4_hc_attn_fn,
            W.v4_hc_attn_base,
            W.v4_hc_attn_scale,
            W.v4_hc_ffn_fn,
            W.v4_hc_ffn_base,
            W.v4_hc_ffn_scale,
        )
        cases = (("mega_moe", 0), ("mega_moe_se", 1))
        for strategy, n_shared_experts in cases:
            with self.subTest(strategy=strategy), patch.dict(
                os.environ, {}, clear=True
            ):
                public_config = server_args.setup_args(
                    ["--moe_strategy", strategy]
                ).moe_config
                model_config = self._load_minimal_model_config(n_shared_experts)
                args = _args_from_model_config(
                    model_config,
                    moe_config=public_config,
                )
                args.ep_size = 2
                args.ep_rank = 0
                layer_weights = {key: torch.ones(1) for key in weight_keys}
                kwargs = _block_kwargs(0, args, layer_weights)

                with patch(
                    "rtp_llm.models_py.modules.dsv4.block.AttentionFP8",
                    DummyModule,
                ), patch(
                    "rtp_llm.models_py.modules.dsv4.block.RMSNorm",
                    DummyModule,
                ), patch(
                    "rtp_llm.models_py.modules.dsv4.block.build_hc_unit",
                    side_effect=lambda *args, **kwargs: DummyModule(),
                ), patch.object(
                    Block,
                    "_resolve_prefill_fast_hc_impls",
                    return_value=(lambda: None,) * 4,
                ), patch(
                    "rtp_llm.models_py.modules.dsv4.block.Dsv4MoeLayer",
                    DummyMoe,
                ):
                    block = Block(**kwargs)

                self.assertEqual(public_config.moe_strategy, strategy)
                self.assertEqual(args.moe_strategy, strategy)
                self.assertEqual(kwargs["moe_strategy"], strategy)
                self.assertEqual(block.ffn.strategy_name, strategy)


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


if __name__ == "__main__":
    main()

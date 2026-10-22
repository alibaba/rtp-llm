"""CPU contracts for the startup-only Decode DCP mode.

Attention kernels and collectives are mocked here. GPU metadata, numerical
accuracy, distributed PD and Graph replay require their separate tests.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.model_loader.model_weight_info import ModelDeployWeightInfo
from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3MtpWeight, KimiK3Weight
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    mla_page_rr_cache_gather as prefix_module,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mla_metadata import (
    PageRRMlaMetadata,
)
from rtp_llm.models_py.modules.kimi_k3 import mla as mla_module
from rtp_llm.ops import DecodeCPMLABackend, KvCacheDataType, RoleType
from rtp_llm.utils.model_weight import W


class DecodeDcpModeTest(unittest.TestCase):
    def test_target_and_draft_weight_mode_matches_attention_metadata(self):
        for role in (RoleType.PREFILL, RoleType.DECODE):
            for tp in (4, 8):
                for enabled in (False, True):
                    with self.subTest(role=role, tp=tp, enabled=enabled):
                        parallelism = SimpleNamespace(
                            role_type=role, tp_size=tp, tp_rank=tp - 1,
                            decode_kv_dcp=enabled,
                            prefill_cp_config=SimpleNamespace(
                                kv_cache_sharded=True, is_enabled=lambda: False,
                            ),
                        )
                        expected_replicated = role == RoleType.DECODE and enabled
                        for weight_cls in (KimiK3Weight, KimiK3MtpWeight):
                            with patch.object(ModelDeployWeightInfo, "__init__", return_value=None):
                                weights = weight_cls(None, parallelism)
                            self.assertEqual(weights._replicate_mla_query, expected_replicated)

                        config = SimpleNamespace(
                            head_num=96 // tp, kv_lora_rank=512,
                            rope_head_dim=64, nope_head_dim=128, v_head_dim=128,
                            kernel_tokens_per_block=128, tokens_per_block=4096,
                            softmax_extra_scale=1.0, kv_cache_dtype=KvCacheDataType.BASE,
                            mla_fp8_compute=False,
                        )
                        inputs = SimpleNamespace(
                            is_target_verify=True, is_mtp_draft_update=False, is_prefill=False,
                        )
                        with (
                            patch.object(mla_module.KimiK3MlaImpl, "_init_paged"),
                            patch.object(mla_module.KimiK3MlaImpl, "prepare"),
                            patch.object(mla_module.common, "create_write_cache_store_impl"),
                        ):
                            impl = mla_module.KimiK3MlaImpl(
                                config, inputs, [], parallelism_config=parallelism,
                            )
                        sharded = role != RoleType.DECODE or enabled
                        self.assertEqual(impl.q_replicated, expected_replicated)
                        self.assertEqual(impl.page_rr_metadata.cp_size, tp if sharded else 1)
                        self.assertEqual(impl.page_rr_metadata.cp_rank, tp - 1 if sharded else 0)

    def test_full_kv_metadata_accepts_only_rank_zero(self):
        for kernel_page in (64, 128):
            PageRRMlaMetadata(4096, kernel_page, 1, 0)
            for size, rank in ((0, 0), (1, 1), (1, -1)):
                with self.assertRaises(ValueError):
                    PageRRMlaMetadata(4096, kernel_page, size, rank)

    def test_full_kv_decode_uses_local_heads_without_query_gather_or_combine(self):
        for queries in (1, 4):
            with self.subTest(queries=queries):
                impl = object.__new__(mla_module.KimiK3MlaImpl)
                impl.num_heads, impl.kv_lora_rank, impl.qk_rope_head_dim = 3, 4, 2
                impl.tokens = 2 * queries
                impl.token_per_block = 64
                impl.page_rr_communicator = None
                impl.fia2a_backend = None
                impl.q_replicated = False
                impl.fp8_compute = False
                impl.scale = 1.0
                impl._paged_workspace = torch.empty(32, dtype=torch.uint8)
                impl.page_rr_metadata = SimpleNamespace(
                    local_causal_lens=torch.ones((2, queries), dtype=torch.int32),
                    query_block_tables=torch.ones((impl.tokens, 2), dtype=torch.int32),
                )
                vc = torch.arange(3 * 4 * 2).reshape(3, 4, 2).to(torch.bfloat16)
                impl.weights = [{W.mla_vc: vc}]
                partial = torch.arange(impl.tokens * 3 * 4).reshape(
                    2, queries, 3, 4,
                ).to(torch.bfloat16)
                impl._run_paged_attention = Mock(return_value=(partial, torch.zeros(2, queries, 3)))
                query = torch.zeros(impl.tokens, 3, 6, dtype=torch.bfloat16)
                cache = SimpleNamespace(kv_cache_base=torch.empty(2, 64, 6))
                with patch.object(mla_module, "get_mla_dcp", side_effect=AssertionError("unexpected DCP")):
                    result = impl._forward_paged(query, cache, 0)
                expected = torch.bmm(partial.reshape(impl.tokens, 3, 4).transpose(0, 1), vc)
                torch.testing.assert_close(result, expected.transpose(0, 1), rtol=0, atol=0)
                args, kwargs = impl._run_paged_attention.call_args
                self.assertEqual(args[0].shape, (2, queries, 3, 6))
                self.assertIs(kwargs["normalize_empty"], True)

    def test_fia2a_replicated_query_retains_token_major_layout(self):
        impl = object.__new__(mla_module.KimiK3MlaImpl)
        impl.num_heads, impl.kv_lora_rank, impl.qk_rope_head_dim = 3, 4, 2
        impl.tokens, impl.token_per_block, impl.q_replicated = 8, 64, True
        merged = torch.arange(3 * 8 * 4).reshape(3, 8, 4).to(torch.bfloat16)
        impl.fia2a_backend = SimpleNamespace(forward=Mock(return_value=merged))
        impl.page_rr_communicator = SimpleNamespace(heads=12, query_gather=Mock())
        impl.page_rr_metadata = SimpleNamespace(local_causal_lens=torch.ones(2, 4))
        vc = torch.arange(3 * 4 * 2).reshape(3, 4, 2).to(torch.bfloat16)
        impl.weights = [{W.mla_vc: vc}]
        query = torch.zeros(8, 12, 6, dtype=torch.bfloat16)
        cache = SimpleNamespace(kv_cache_base=torch.empty(2, 64, 6))
        result = impl._forward_paged(query, cache, 0)
        self.assertIs(impl.fia2a_backend.forward.call_args.args[0], query)
        impl.page_rr_communicator.query_gather.assert_not_called()
        torch.testing.assert_close(result, torch.bmm(merged, vc).transpose(0, 1), rtol=0, atol=0)

    def test_fia2a_with_decode_dcp_disabled_is_rejected_before_allocation(self):
        impl = object.__new__(mla_module.KimiK3MlaImpl)
        impl.parallelism_config = SimpleNamespace(tp_size=4)
        impl.q_replicated, impl.num_heads = False, 24
        impl.kv_lora_rank, impl.qk_nope_head_dim, impl.qk_rope_head_dim = 512, 128, 64
        impl.token_per_block, impl.fp8_compute = 128, False
        impl.page_rr_metadata = SimpleNamespace(cp_size=1)
        impl.fmha_config = SimpleNamespace(decode_cp_mla_backend=DecodeCPMLABackend.FIA2A)
        impl.weights = [{W.mla_vc: torch.empty(24, 512, 128),
                         W.mla_kc: torch.empty(24, 128, 512)}]
        inputs = SimpleNamespace(input_lengths=torch.ones(2), physical_token_count=8,
                                 is_target_verify=True)
        with self.assertRaisesRegex(ValueError, "DECODE_KV_DCP"):
            impl._init_paged(inputs)

    def test_full_kv_prefix_and_chunk_reads_never_all_gather(self):
        adapter = prefix_module.MlaPageRRCacheGather(4096, 1, 0, 128)
        payload = torch.arange(12).reshape(2, 2, 3)
        descriptor = SimpleNamespace(total_local_pages=2)
        object.__setattr__(adapter, "_prefix_descriptor", descriptor)
        with (
            patch.object(prefix_module.MlaPageRRCacheGather, "_pack_prefix", return_value=payload),
            patch.object(prefix_module.MlaPageRRCacheGather, "_pack_prefix_chunk", return_value=payload),
            patch.object(prefix_module, "_restore_mla_page_rr_prefix", side_effect=lambda x, _: x),
            patch.object(prefix_module.collective_torch, "all_gather_into", side_effect=AssertionError("unexpected AG")),
        ):
            prefix = adapter.read_prefix(None, None, [8])
            chunk = adapter.read_prefix_chunk(None, None, descriptor)
        torch.testing.assert_close(prefix, payload.unsqueeze(0), rtol=0, atol=0)
        torch.testing.assert_close(chunk, payload.unsqueeze(0), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

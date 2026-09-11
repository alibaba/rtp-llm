"""Real Eagle3 model: target-aux Prefill, prefix continuation and draft steps."""

import json
import os
import tempfile
import unittest

import torch

from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3Eagle3
from rtp_llm.models.rotary_embedding.deepseek_rotary_embedding import (
    DeepseekV3RotaryEmbedding,
)
from rtp_llm.models_py.model_desc.kimi_k3_eagle3 import KimiK3Eagle3Model
from rtp_llm.ops import KvCacheDataType, ParallelismConfig, RoleType
from rtp_llm.ops.compute_ops import (
    CacheGroupType,
    KVCache,
    PyModelInputs,
    init_exec_ctx,
)
from rtp_llm.utils.model_weight import W, transpose_slice_k, transpose_slice_v


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class KimiK3Eagle3CacheIntegrationTest(unittest.TestCase):
    @torch.inference_mode()
    def test_target_aux_prefix_and_draft_match_fresh_recomputation(self):
        torch.manual_seed(417)
        # Preserve the deployed Eagle attention geometry; shrink only unrelated
        # embedding/MLP dimensions. Use the real parser, not generic defaults.
        raw = {
            "model_type": "deepseek_v3_swa",
            "num_hidden_layers": 1,
            "hidden_size": 64,
            "vocab_size": 32,
            "intermediate_size": 128,
            "max_position_embeddings": 16384,
            "num_attention_heads": 96,
            "num_key_value_heads": 96,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "sliding_window": 2048,
        }
        with tempfile.TemporaryDirectory(
            dir=os.environ.get("TEST_TMPDIR")
        ) as checkpoint:
            with open(
                os.path.join(checkpoint, "config.json"), "w", encoding="utf-8"
            ) as writer:
                json.dump(raw, writer)
            config = KimiK3Eagle3._create_config(checkpoint)
        config.quant_config = None
        config.attn_config.tokens_per_block = 4096
        config.attn_config.kernel_tokens_per_block = 128
        config.attn_config.kv_cache_dtype = KvCacheDataType.BASE
        parallel = ParallelismConfig()
        parallel.role_type = RoleType.PDFUSION
        init_exec_ctx(
            device_id=0,
            trace_memory=False,
            enable_comm_overlap=False,
            mla_ops_type=int(config.mla_ops_type),
        )

        def matrix(rows, columns):
            return (
                torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16) * 0.01
            )

        weights = ModelWeights(1, "cuda", torch.bfloat16)
        weights.set_global_weight(W.embedding, matrix(32, 64))
        weights.set_global_weight(
            W.final_ln_gamma, torch.ones(64, device="cuda", dtype=torch.bfloat16)
        )
        rotary = DeepseekV3RotaryEmbedding(
            64, 16384, config.attn_config.rope_config.base, device="cuda"
        )
        weights.set_global_weight(
            W.rope_cos_sin_cache,
            torch.cat((rotary.cos_cached[:, :32], rotary.sin_cached[:, :32]), -1),
        )
        kv_checkpoint = matrix(96 * 256, 512)
        weights.weights[0] = {
            W.eagle3_fc_proj: matrix(192, 64),
            W.mla_fusedqkrope_w: matrix(128, 1536 + 512 + 64 + 96 * 128),
            W.mla_q_b_w: matrix(1536, 96 * 192),
            W.mla_kv_b_w: kv_checkpoint.T.contiguous(),
            W.mla_kc: transpose_slice_k([kv_checkpoint], 96, 128, 128, 512),
            W.mla_vc: transpose_slice_v([kv_checkpoint], 96, 128, 128, 512),
            W.attn_o_w: matrix(96 * 128, 64),
            W.ffn_w1: matrix(64, 128),
            W.ffn_w3: matrix(64, 128),
            W.ffn_w2: matrix(128, 64),
        }
        for name, width in (
            (W.eagle3_input_norm_gamma, 64),
            (W.eagle3_fc_norm_gamma, 64),
            (W.post_ln_gamma, 64),
            (W.mla_q_a_ln_gamma, 1536),
            (W.mla_kv_a_ln_gamma, 512),
        ):
            weights.weights[0][name] = torch.ones(
                width, device="cuda", dtype=torch.bfloat16
            )

        cached = KimiK3Eagle3Model(config, parallel, weights, max_generate_batch_size=1)
        reference = KimiK3Eagle3Model(
            config, parallel, weights, max_generate_batch_size=1
        )
        storage = torch.full(
            (6, 4096, 576), float("nan"), device="cuda", dtype=torch.bfloat16
        )
        cache = KVCache()
        cache.kv_cache_base_by_layer = [storage]
        cache.seq_size_per_block, cache.kernel_seq_size_per_block = 4096, 128
        cache.use_mla, cache.kv_lora_rank, cache.rope_head_dim = True, 512, 64
        # MtpExecutor's runtime FULL view does not change the SWA P-unit table.
        cache.layer_group_types = [CacheGroupType.FULL]
        cached.kv_cache = cache
        table_h = torch.tensor([[3, 1, 5, 2]], dtype=torch.int32)
        table_d = table_h.cuda()

        def inputs(ids, hiddens, prefix=0, mode="prefill"):
            result = PyModelInputs()
            result.input_ids, result.input_hiddens = ids, hiddens
            attention = result.attention_inputs
            count = ids.numel()
            attention.is_prefill = mode != "decode"
            attention.is_mtp_draft_update = mode == "draft"
            attention.total_tokens = count
            attention.input_lengths_host = torch.tensor([count], dtype=torch.int32)
            attention.input_lengths = attention.input_lengths_host.cuda()
            attention.prefix_lengths_host = torch.tensor(
                [prefix] if mode != "decode" else [], dtype=torch.int32
            )
            attention.prefix_lengths = attention.prefix_lengths_host.cuda()
            attention.sequence_lengths_host = torch.tensor(
                [prefix] if mode == "decode" else [], dtype=torch.int32
            )
            attention.sequence_lengths = attention.sequence_lengths_host.cuda()
            attention.sequence_lengths_plus_1_d = torch.tensor(
                [prefix + 1] if mode == "decode" else [],
                device="cuda",
                dtype=torch.int32,
            )
            attention.cu_seqlens = torch.tensor(
                [0, count], device="cuda", dtype=torch.int32
            )
            attention.cu_kv_seqlens = torch.tensor(
                [0, prefix + count], device="cuda", dtype=torch.int32
            )
            attention.padding_offset = torch.zeros(
                count, device="cuda", dtype=torch.int32
            )
            attention.kv_cache_kernel_block_id_host = table_h
            attention.kv_cache_kernel_block_id_device = table_d
            attention.kv_cache_kernel_block_id_host_by_group = [table_h]
            attention.kv_cache_kernel_block_id_device_by_group = [table_d]
            attention.kv_cache_layer_to_group_host = torch.tensor(
                [0], dtype=torch.int32
            )
            return result

        # One real instance changes phase and input shape, with the same cache.
        length, split = 12295, 4096
        ids = torch.arange(length + 8, device="cuda", dtype=torch.int32) % 32
        target_aux = matrix(length, 192)
        first = cached(inputs(ids[:split], target_aux[:split])).hidden_states
        # The original reuse=1/S=1 pool retains each P page. Exercise several
        # rounds and the delayed terminal token using that same physical table.
        parts = []
        for begin, end in ((split, 2 * split), (2 * split, length - 1), (length - 1, length)):
            parts.append(cached(inputs(ids[begin:end], target_aux[begin:end], begin)).hidden_states)
        continuation = torch.cat(parts)
        fresh = reference(inputs(ids[:length], target_aux)).hidden_states
        torch.testing.assert_close(first, fresh[:split], atol=1e-2, rtol=2e-2)
        torch.testing.assert_close(continuation, fresh[split:], atol=1e-2, rtol=2e-2)

        # The next input has the real H-width draft contract, rather than 3H.
        recurrent = continuation[-1:].clone()
        decoded = cached(
            inputs(ids[length : length + 1], recurrent, length, "decode")
        ).hidden_states
        full_hiddens = torch.cat((reference.aux_projection(target_aux), recurrent))
        fresh = reference(inputs(ids[: length + 1], full_hiddens)).hidden_states
        torch.testing.assert_close(decoded, fresh[-1:], atol=1e-2, rtol=2e-2)

        verify_aux = matrix(7, 192)
        draft = cached(
            inputs(ids[length + 1 :], verify_aux, length + 1, "draft")
        ).hidden_states
        full_hiddens = torch.cat((full_hiddens, reference.aux_projection(verify_aux)))
        fresh = reference(inputs(ids, full_hiddens)).hidden_states
        torch.testing.assert_close(draft, fresh[-7:], atol=1e-2, rtol=2e-2)
        self.assertTrue(torch.isfinite(draft).all().item())
        self.assertTrue(torch.isnan(storage[0]).all().item())

        # Ordinary Prefill has no Graph replay adapter; preserve the registry's
        # existing capability gate instead of returning an unsupported instance.
        with self.assertRaisesRegex(Exception, "can not find mla type"):
            cached.prepare_fmha_impl(inputs(ids[:2], target_aux[:2]), is_cuda_graph=True)


if __name__ == "__main__":
    unittest.main()

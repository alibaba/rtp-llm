"""Comprehensive tests for Qwen 4.5-Flash QSA (Indexer Sparse Attention)."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from rtp_llm.config.qwen4_5_config import (
    Qwen4_5ModelConfig,
    parse_qwen4_5_config_json,
    validate_qwen4_5_arch_config,
)
from rtp_llm.models.qwen4_5.qwen4_5 import Qwen4_5
from rtp_llm.models.qwen4_5.qwen4_5_kv_cache import (
    FULL_ATTN_KV_TAG,
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
    build_qwen4_5_kv_cache_plan,
)
from rtp_llm.models.qwen4_5.qwen4_5_weight import (
    QWEN4_5_W,
    expected_checkpoint_tensor_specs,
    summarize_checkpoint_specs,
)
from rtp_llm.models_py.model_desc.qwen4_5 import Qwen4_5ForCausalLM
from rtp_llm.models_py.model_desc.qwen4_5_runtime import (
    LocalKVCache,
    LocalPyModelInputs,
    Qwen4_5RTPModel,
    build_block_tables_for_requests,
    make_tagged_attention_inputs,
)
from rtp_llm.models_py.modules.qwen4_5.attention import Qwen4_5QSAIndexer
from rtp_llm.models_py.modules.qwen4_5.checkpoint_loader import (
    load_qwen4_5_from_checkpoint,
    save_qwen4_5_safetensors_checkpoint,
)
from rtp_llm.models_py.modules.qwen4_5.contracts import Qwen4_5ArchConfig
from rtp_llm.models_py.modules.qwen4_5.qsa_bridge import (
    _select_top_tokens,
    build_qsa_neox_cos_sin,
    qsa_paged_score,
    qsa_prefill_score,
    qsa_sparse_paged_decode_attention,
    qsa_sparse_paged_prefill_attention,
    _try_run_q4_prefill,
    qsa_sparse_prefill_attention,
    score_and_select_qsa_paged,
    view_indexer_kv_pool,
    view_indexer_state_pool,
    view_mha_kv_hnd,
    write_qsa_decode_cache_gpu_,
    write_qsa_prefill_cache,
)

try:
    from rtp_llm.models.qwen4_5.qwen4_5_kv_cache import (
        build_qwen4_5_kv_cache_spec_descs,
    )
    from rtp_llm.ops import (
        CacheGroupType,
        KVCacheSpecType,
        OpaqueBlockEntryCountMode,
        TpCacheTransferMode,
    )

    _HAS_RTP_OPS = True
except Exception:
    _HAS_RTP_OPS = False


def _make_mini_arch(
    *,
    indexer_head_dim: int = 128,
    indexer_n_heads: int = 8,
    indexer_kv_heads: int = 1,
    indexer_compress_ratio: int = 4,
    indexer_budget: int = 2048,
    max_position_embeddings: int = 512,
) -> Qwen4_5ArchConfig:
    return Qwen4_5ArchConfig(
        vocab_size=256,
        hidden_size=512,
        num_hidden_layers=8,
        intermediate_size=1024,
        first_k_dense_replace=1,
        rms_norm_eps=1e-6,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        max_position_embeddings=max_position_embeddings,
        full_attention_interval=4,
        num_attention_heads=32,
        num_key_value_heads=1,
        head_dim=256,
        attn_output_gate=True,
        partial_rotary_factor=0.25,
        rope_theta=10000.0,
        use_full_attn_kv_sconv=True,
        full_attn_kv_sconv_residual=True,
        full_attn_kv_sconv_use_activation=False,
        linear_conv_kernel_dim=4,
        linear_num_key_heads=4,
        linear_key_head_dim=64,
        linear_num_value_heads=16,
        linear_value_head_dim=64,
        gdn_conv_residual=False,
        gdn_no_q_conv=False,
        norm_before_gate=True,
        output_gate_type="sigmoid",
        use_linear_attn_output_sconv=True,
        use_full_attn_output_sconv=True,
        use_ffn_output_sconv=True,
        output_sconv_use_activation=False,
        hc_count=4,
        hc_lowrank=32,
        hc_prev_mix=True,
        hc_per_branch_norm=True,
        hc_rmsnorm_bias=True,
        num_experts=16,
        num_experts_per_tok=4,
        routed_expert_hidden_size=256,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=128,
        shared_expert_gate=True,
        router_precision="fp32_logits",
        router_score_function="sigmoid",
        moe_router_pre_softmax=True,
        router_normalize_topk_scores=True,
        moe_router_enable_expert_bias=True,
        hidden_act="situ_glu",
        hidden_act_gate_beta=4.0,
        hidden_act_up_beta=25.0,
        clip_mlp_output=1000.0,
        ple_layer_ids=(2,),
        ple_embed_dim=256,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=4,
        ngram_vocab_size_base=256,
        make_ngram_vocab_size_divisible_by=64,
        split_ngram_parts=4,
        ple_quant_scheme="bf16",
        share_map="4->8,5->6,5->7",
        enable_yoco_decode=True,
        quant_method="mxfp4",
        indexer_head_dim=indexer_head_dim,
        indexer_n_heads=indexer_n_heads,
        indexer_kv_heads=indexer_kv_heads,
        indexer_compress_ratio=indexer_compress_ratio,
        indexer_budget=indexer_budget,
    )


def _stabilize_router_margins(model: Qwen4_5ForCausalLM) -> None:
    for layer in model.model.layers:
        if layer.is_moe and layer.mlp.expert_bias is not None:
            layer.mlp.expert_bias.zero_()
            shard = layer.mlp.num_experts // 4
            for r in range(4):
                layer.mlp.expert_bias[r * shard + (layer.layer_idx % shard)] = 4.0


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            a.reshape(1, -1).float(), b.reshape(1, -1).float(), dim=-1
        ).item()
    )


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Qwen4_5QSATest(unittest.TestCase):
    def test_config_validation_weights_and_backward_compat(self) -> None:
        # Backward compat: non-QSA 40L 1109 checkpoint stays bit-identical
        arch_dense = Qwen4_5ArchConfig(
            share_map="21->22,21->23,20->24,21->25,21->26,21->27,20->28,21->29,21->30,21->31,20->32,21->33,21->34,21->35,20->36,21->37,21->38,21->39,20->40"
        )
        self.assertFalse(arch_dense.has_qsa)
        summary = summarize_checkpoint_specs(
            expected_checkpoint_tensor_specs(arch_dense), arch_dense
        )
        self.assertEqual(summary["tensor_count"], 151_038)
        self.assertEqual(summary["total_bytes"], 461_163_888_730)

        # Invalid QSA contracts must fail loudly
        bad_budget = _make_mini_arch(indexer_budget=15, indexer_compress_ratio=4)
        with self.assertRaisesRegex(ValueError, "divisible"):
            validate_qwen4_5_arch_config(bad_budget)

        bad_kv_heads = _make_mini_arch(indexer_kv_heads=2)
        with self.assertRaisesRegex(ValueError, "indexer_kv_heads"):
            validate_qwen4_5_arch_config(bad_kv_heads)

        # Weight descriptor coverage on both source (layer 3) and consumer (layer 7)
        arch_qsa = _make_mini_arch(indexer_n_heads=8, indexer_budget=2048)
        self.assertTrue(arch_qsa.has_qsa)
        specs = expected_checkpoint_tensor_specs(arch_qsa)
        for lid in (3, 7):
            self.assertEqual(
                specs[f"model.layers.{lid}.self_attn.indexer.index_qk_proj.weight"],
                ("BF16", (9 * 128, 512)),
            )
            self.assertEqual(
                specs[f"model.layers.{lid}.self_attn.indexer.q_layernorm.weight"],
                ("BF16", (128,)),
            )
            self.assertEqual(
                specs[f"model.layers.{lid}.self_attn.indexer.k_layernorm.weight"],
                ("BF16", (128,)),
            )

    @unittest.skipUnless(_HAS_RTP_OPS, "requires compiled rtp_llm.ops")
    def test_kv_cache_spec_descs_and_yoco_isolation(self) -> None:
        arch_qsa = _make_mini_arch(indexer_n_heads=8, indexer_budget=2048)
        plan = build_qwen4_5_kv_cache_plan(arch_qsa)
        self.assertTrue(plan.layers[3].has_qsa_indexer_cache)
        self.assertFalse(plan.layers[7].has_qsa_indexer_cache)
        self.assertEqual(plan.layers[7].indexer_kv_source_layer, 3)

        cfg = Qwen4_5ModelConfig(arch_config=arch_qsa)
        Qwen4_5._populate_model_config(cfg, arch_qsa)
        self.assertTrue(cfg.qwen4_5_has_qsa)
        self.assertEqual(cfg._qwen4_indexer_n_heads, 8)
        self.assertEqual(cfg._qwen4_indexer_head_dim, 128)
        self.assertEqual(cfg.attn_config.indexer_head_num, 8)
        self.assertEqual(cfg.attn_config.indexer_topk, 512)

        # Default spec uses KERNEL_BLOCK_COMPRESSED
        descs_default = build_qwen4_5_kv_cache_spec_descs(cfg, arch_qsa)
        src_tags = {d.tag: d for d in descs_default[3]}
        self.assertIn(INDEXER_KV_TAG, src_tags)
        self.assertIn(INDEXER_STATE_TAG, src_tags)
        self.assertEqual(src_tags[INDEXER_KV_TAG].cache_type, KVCacheSpecType.OPAQUE_KV)
        self.assertEqual(
            src_tags[INDEXER_KV_TAG].entry_count_mode,
            OpaqueBlockEntryCountMode.KERNEL_BLOCK_COMPRESSED,
        )
        self.assertEqual(src_tags[INDEXER_KV_TAG].compression_ratio, 4)
        self.assertEqual(src_tags[INDEXER_KV_TAG].entry_elems, 256)
        self.assertTrue(src_tags[INDEXER_KV_TAG].reuse.enable_prefix_reuse)
        self.assertEqual(
            src_tags[INDEXER_KV_TAG].tp_transfer_mode, TpCacheTransferMode.REPLICATED
        )

        self.assertEqual(
            src_tags[INDEXER_STATE_TAG].cache_type, KVCacheSpecType.OPAQUE_STATE
        )
        self.assertEqual(
            src_tags[INDEXER_STATE_TAG].entry_count_mode,
            OpaqueBlockEntryCountMode.STATE_RING,
        )
        self.assertEqual(src_tags[INDEXER_STATE_TAG].state_ring_overlap, 1)
        self.assertEqual(src_tags[INDEXER_STATE_TAG].group_type, CacheGroupType.FULL)
        self.assertEqual(
            src_tags[INDEXER_STATE_TAG].tp_transfer_mode, TpCacheTransferMode.REPLICATED
        )

        # Consumer layer 7 must emit ZERO indexer pools
        cons_tags = {d.tag for d in descs_default[7]}
        self.assertNotIn(INDEXER_KV_TAG, cons_tags)
        self.assertNotIn(INDEXER_STATE_TAG, cons_tags)

        # Small-block explicit adaptation for seq_size_per_block=64 (smoke / C++ engine)
        cfg.attn_config.tokens_per_block = 64
        cfg.attn_config.kernel_tokens_per_block = 64
        Qwen4_5._post_build_model_config(cfg)
        src_64 = {d.tag: d for d in cfg.kv_cache_spec_descs[3]}
        self.assertEqual(
            src_64[INDEXER_KV_TAG].entry_count_mode,
            OpaqueBlockEntryCountMode.EXPLICIT,
        )
        self.assertEqual(src_64[INDEXER_KV_TAG].explicit_entry_count, 16)
        self.assertEqual(src_64[INDEXER_KV_TAG].compression_ratio, 4)
        self.assertEqual(src_64[INDEXER_KV_TAG].entry_elems, 256)

    def test_short_context_equivalence_with_dense(self) -> None:
        dev = _device()
        dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
        arch_dense = _make_mini_arch(indexer_head_dim=0, indexer_n_heads=0, indexer_budget=0)
        arch_qsa = _make_mini_arch(indexer_head_dim=128, indexer_n_heads=8, indexer_budget=2048)

        with torch.no_grad():
            m_dense = Qwen4_5ForCausalLM(
                arch_dense, use_triton_mxfp4=False, compute_mode="bf16", dtype=dt, device=dev
            )
            m_dense.init_synthetic_weights(seed=77, ple_mode="gpu_full", ple_fp8=False)
            # Match test_e2e_qwen4_5_model._stabilize_router_margins so 1-ULP BF16 SDPA vs Triton attention
            # differences do not flip random MoE top-k routing boundaries.
            for layer in m_dense.model.layers:
                if layer.is_moe and layer.mlp.expert_bias is not None:
                    layer.mlp.expert_bias.zero_()
                    shard = layer.mlp.num_experts // 4
                    for r in range(4):
                        layer.mlp.expert_bias[r * shard + (layer.layer_idx % shard)] = 4.0

            m_qsa = Qwen4_5ForCausalLM(
                arch_qsa, use_triton_mxfp4=False, compute_mode="bf16", dtype=dt, device=dev
            )
            m_qsa.init_synthetic_weights(seed=77, ple_mode="gpu_full", ple_fp8=False)
            # Copy identical non-indexer weights from dense model
            dense_sd = m_dense.state_dict()
            qsa_sd = m_qsa.state_dict()
            for k, v in dense_sd.items():
                if k in qsa_sd:
                    qsa_sd[k].copy_(v)
            m_qsa.load_state_dict(qsa_sd)
            self.assertIsInstance(m_qsa.model.layers[3].self_attn.indexer, Qwen4_5QSAIndexer)
            self.assertIsInstance(m_qsa.model.layers[7].self_attn.indexer, Qwen4_5QSAIndexer)

            ids = torch.randint(4, 200, (1, 36), dtype=torch.int64, device=dev)
            pos = torch.arange(36, dtype=torch.int64, device=dev).unsqueeze(0)
            st_d = m_dense.create_runtime_state(1, 64, device=dev, dtype=dt)
            st_q = m_qsa.create_runtime_state(1, 64, device=dev, dtype=dt)
            self.assertEqual(set(st_q.qsa_raw_keys.keys()), {3})

            out_d = m_dense(ids, state=st_d, positions=pos)
            out_q = m_qsa(ids, state=st_q, positions=pos)
            cos = _cosine(out_q, out_d)
            self.assertGreaterEqual(cos, 0.9995, f"Short context QSA vs Dense cos={cos}")

            # 2 incremental decode steps (still <= budget=2048)
            for step in range(2):
                nxt = torch.randint(4, 200, (1, 1), dtype=torch.int64, device=dev)
                p = torch.full((1, 1), 36 + step, dtype=torch.int64, device=dev)
                ld = m_dense(nxt, state=st_d, positions=p)
                lq = m_qsa(nxt, state=st_q, positions=p)
                cos_dec = _cosine(lq, ld)
                self.assertGreaterEqual(
                    cos_dec, 0.9995, f"Short decode step={step} cos={cos_dec}"
                )

            if dev.type == "cuda":
                q_t = torch.randn(1, 6, 150, 256, device=dev, dtype=torch.bfloat16)
                k_t = torch.randn(1, 1, 200, 256, device=dev, dtype=torch.bfloat16)
                v_t = torch.randn(1, 1, 200, 256, device=dev, dtype=torch.bfloat16)
                for has_dups in (False, True):
                    sel_t = torch.full((1, 150, 128), -1, device=dev, dtype=torch.int32)
                    for s_i in range(150):
                        row_idx = torch.randperm(51 + s_i, device=dev)[:128].to(torch.int32)
                        if has_dups:
                            row_idx[1] = row_idx[0]
                        sel_t[0, s_i, : row_idx.numel()] = row_idx
                    out_cu = qsa_sparse_prefill_attention(q_t, k_t, v_t, sel_t, validate_selected=True)
                    out_cpu = qsa_sparse_prefill_attention(
                        q_t.cpu(), k_t.cpu(), v_t.cpu(), sel_t.cpu(), validate_selected=True
                    ).cuda()
                    diff = float((out_cu - out_cpu).abs().max().item())
                    self.assertLess(diff, 0.02, f"QSA fast sparse prefill has_dups={has_dups} diff={diff}")

    def test_pruned_qsa_scorers_both_head_counts_h8_and_h4(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("Triton GPU scorer check requires CUDA")
        dev = torch.device("cuda")
        torch.manual_seed(11)
        for n_heads in (8, 4):
            for seq_len in (64, 128):
                num_blocks = seq_len // 4
                q = torch.randn((2, seq_len, n_heads, 128), dtype=torch.bfloat16, device=dev)
                k = torch.randn((2, num_blocks, 128), dtype=torch.bfloat16, device=dev)
                got = qsa_prefill_score(q, k)
                ref = (
                    torch.relu(torch.matmul(q.float(), k.float().transpose(-1, -2).unsqueeze(1)))
                    .transpose(-1, -2)
                    .sum(dim=-1)
                    * (1.0 / math.sqrt(128))
                )
                self.assertLess((got - ref).abs().max().item(), 5e-2)

            for block_size in (4, 16, 32):  # corresponds to tokens_per_block 16, 64, 128
                q_dec = torch.randn((2, 1, n_heads, 128), dtype=torch.bfloat16, device=dev)
                pool = torch.randn((16, block_size, 128), dtype=torch.bfloat16, device=dev)
                bt = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]], dtype=torch.int32, device=dev)
                ctx = torch.tensor([[block_size * 2 + 1], [block_size + 2]], dtype=torch.int32, device=dev)
                got_p = qsa_paged_score(q_dec, pool, bt, ctx, max_ctx_len=block_size * 4)
                for row in range(2):
                    c_len = int(ctx[row, 0].item())
                    for c in range(c_len):
                        bi, ei = c // block_size, c % block_size
                        k_vec = pool[bt[row, bi], ei].float()
                        expected = (
                            torch.relu(torch.matmul(q_dec[row, 0].float(), k_vec)).sum()
                            * (1.0 / math.sqrt(128))
                        )
                        self.assertAlmostEqual(
                            float(got_p[row, c].item()), float(expected.item()), places=2
                        )
                    self.assertTrue(torch.isinf(got_p[row, c_len:]).all())

    def test_pruned_prefill_reuse_decode_and_cuda_graph_for_block_sizes(self) -> None:
        dev = _device()
        dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
        if not torch.cuda.is_available():
            self.skipTest("CUDA Graph & Triton paged QSA E2E requires CUDA")

        for idx_n_heads, tpb in ((8, 64), (8, 128), (4, 64)):
            with self.subTest(indexer_n_heads=idx_n_heads, tokens_per_block=tpb):
                arch = _make_mini_arch(
                    indexer_head_dim=128,
                    indexer_n_heads=idx_n_heads,
                    indexer_kv_heads=1,
                    indexer_compress_ratio=4,
                    indexer_budget=16,
                    max_position_embeddings=512,
                )
                with torch.no_grad():
                    causal_lm = Qwen4_5ForCausalLM(
                        arch, use_triton_mxfp4=True, compute_mode="bf16", dtype=dt, device=dev
                    )
                    causal_lm.init_synthetic_weights(
                        seed=100 + idx_n_heads + tpb, ple_mode="gpu_full", ple_fp8=False
                    )
                    _stabilize_router_margins(causal_lm)
                    causal_lm.prepare_weights()
                    # Verify non-zero k_layernorm weight makes source vs consumer k_layernorm distinct
                    causal_lm.model.layers[3].self_attn.indexer.k_layernorm.weight.fill_(0.25)
                    causal_lm.model.layers[7].self_attn.indexer.k_layernorm.weight.fill_(-0.5)

                    plan = build_qwen4_5_kv_cache_plan(arch)
                    kv_cache = LocalKVCache(
                        plan,
                        tokens_per_block=tpb,
                        num_blocks_per_tag={
                            INDEXER_KV_TAG: 32,
                            INDEXER_STATE_TAG: 32,
                            FULL_ATTN_KV_TAG: 32,
                            "linear": 32,
                            "gdn_consumer_q_conv": 32,
                            "full_attn_kv_sconv": 32,
                            "full_attn_output_sconv": 32,
                            "linear_attn_output_sconv": 32,
                            "ffn_output_sconv": 32,
                            "ple_conv_state": 32,
                            "ple_ngram_ctx": 32,
                        },
                        device=dev,
                        dtype=dt,
                    )
                    # Check YoCo indexer cache ownership in LocalKVCache
                    self.assertIn(INDEXER_KV_TAG, kv_cache.layer_caches[3])
                    self.assertIn(INDEXER_STATE_TAG, kv_cache.layer_caches[3])
                    self.assertNotIn(INDEXER_KV_TAG, kv_cache.layer_caches[7])
                    self.assertNotIn(INDEXER_STATE_TAG, kv_cache.layer_caches[7])

                    rtp_model = Qwen4_5RTPModel.from_causal_lm(
                        causal_lm, kv_cache=kv_cache
                    )

                    # Full prompt T=128 tokens (8x compressed over budget=16!)
                    tok_gen = torch.Generator(device="cpu").manual_seed(1000 + idx_n_heads + tpb)
                    prompt_ids = torch.randint(4, 240, (1, 128), generator=tok_gen, dtype=torch.int64).to(dev)
                    pos_full = torch.arange(128, dtype=torch.int64, device=dev).unsqueeze(0)
                    dec_toks = torch.randint(4, 240, (1, 4), generator=tok_gen, dtype=torch.int64).to(dev)

                    # Pure-PyTorch standalone reference (prefill T=128 + 4 decodes)
                    ref_state = causal_lm.model.create_runtime_state(
                        1, 200, device=dev, dtype=dt
                    )
                    ref_prefill_h = causal_lm.model(
                        prompt_ids, state=ref_state, positions=pos_full
                    )
                    ref_prefill_logits = causal_lm.lm_head(ref_prefill_h[:, -1:, :])
                    ref_dec_logits = []
                    for step in range(4):
                        h_s = causal_lm.model(
                            dec_toks[:, step : step + 1],
                            state=ref_state,
                            positions=torch.full((1, 1), 128 + step, dtype=torch.int64, device=dev),
                        )
                        ref_dec_logits.append(causal_lm.lm_head(h_s[:, 0, :]))

                    # 1) Cold prefill T=128 in paged LocalKVCache
                    tables = build_block_tables_for_requests(kv_cache, [136])
                    p_attn = make_tagged_attention_inputs(
                        tables,
                        is_prefill=True,
                        input_lengths=[128],
                        sequence_lengths=[128],
                        prefix_lengths=[0],
                        device=dev,
                    )
                    out_cold = rtp_model.forward(
                        LocalPyModelInputs(input_ids=prompt_ids.reshape(-1), attention_inputs=p_attn)
                    )
                    logits_cold = rtp_model.compute_logits(out_cold.hidden_states[-1:])
                    self.assertGreaterEqual(_cosine(logits_cold, ref_prefill_logits), 0.999)

                    # 2) Eager Decode +4 steps (crossing 128 -> 132 block boundary when tpb=64/128)
                    for step in range(4):
                        d_attn = make_tagged_attention_inputs(
                            tables,
                            is_prefill=False,
                            input_lengths=[1],
                            sequence_lengths=[128 + step],
                            prefix_lengths=[0],
                            device=dev,
                        )
                        out_d = rtp_model.forward(
                            LocalPyModelInputs(
                                input_ids=dec_toks[:, step].contiguous(),
                                attention_inputs=d_attn,
                            )
                        )
                        logits_d = rtp_model.compute_logits(out_d.hidden_states)
                        self.assertGreaterEqual(
                            _cosine(logits_d, ref_dec_logits[step]),
                            0.999,
                            f"Eager decode step={step} cos={_cosine(logits_d, ref_dec_logits[step])}",
                        )

                    # 3) Prefix-reuse prefill (prefix=tpb, suffix=128-tpb) matching cold T=128
                    if tpb == 64:
                        # 3a) Under default backend, L0 prefix-cache states at token 64 (before MoE) are exact.
                        tables_reuse = build_block_tables_for_requests(kv_cache, [136])
                        pfx_attn = make_tagged_attention_inputs(
                            tables_reuse,
                            is_prefill=True,
                            input_lengths=[64],
                            sequence_lengths=[64],
                            prefix_lengths=[0],
                            device=dev,
                        )
                        rtp_model.forward(
                            LocalPyModelInputs(
                                input_ids=prompt_ids[0, :64].contiguous(),
                                attention_inputs=pfx_attn,
                            )
                        )
                        for tag_l0 in ("linear", "linear_attn_output_sconv", "ffn_output_sconv"):
                            b_cold = tables[tag_l0][0][0]
                            b_pfx = tables_reuse[tag_l0][0][0]
                            self.assertTrue(
                                torch.equal(
                                    kv_cache.layer_caches[0][tag_l0].kv_cache_base[b_cold],
                                    kv_cache.layer_caches[0][tag_l0].kv_cache_base[b_pfx],
                                ),
                                f"L0 {tag_l0} token-64 snapshot mismatch between cold and prefix",
                            )

                        # 3b) Full prefix-reuse parity + all-tag cache snapshot parity under unified
                        # deepgemm_masked backend (sm100_winner ensures M=64 suffix matches M=128 prefill MoE).
                        prev_dec = os.environ.get("QWEN4_5_MXFP4_DECODE_BACKEND")
                        os.environ["QWEN4_5_MXFP4_DECODE_BACKEND"] = "sm100_winner"
                        try:
                            tables_reuse_u = build_block_tables_for_requests(kv_cache, [136])
                            pfx_u = make_tagged_attention_inputs(
                                tables_reuse_u,
                                is_prefill=True,
                                input_lengths=[64],
                                sequence_lengths=[64],
                                prefix_lengths=[0],
                                device=dev,
                            )
                            rtp_model.forward(
                                LocalPyModelInputs(
                                    input_ids=prompt_ids[0, :64].contiguous(),
                                    attention_inputs=pfx_u,
                                )
                            )
                            sfx_attn = make_tagged_attention_inputs(
                                tables_reuse_u,
                                is_prefill=True,
                                input_lengths=[64],
                                sequence_lengths=[128],
                                prefix_lengths=[64],
                                device=dev,
                            )
                            out_sfx = rtp_model.forward(
                                LocalPyModelInputs(
                                    input_ids=prompt_ids[0, 64:128].contiguous(),
                                    attention_inputs=sfx_attn,
                                )
                            )
                            logits_sfx = rtp_model.compute_logits(out_sfx.hidden_states[-1:])
                            self.assertGreaterEqual(
                                _cosine(logits_sfx, ref_prefill_logits),
                                0.999,
                                f"Prefix-reuse cos={_cosine(logits_sfx, ref_prefill_logits)}",
                            )
                            # Verify prefix-cache snapshot parity across FullAttn/QSA/GDN/ShortConv/PLE blocks
                            for lid_k, tag_map in enumerate(kv_cache.layer_caches):
                                for tag_k, lc in tag_map.items():
                                    # L7 discrete QSA block-tie at blk0 (pos 61) only affects L7 blk0; verify L7 blk1 (pos 128) >= 0.999.
                                    sl_slice = slice(1, 2) if lid_k == 7 else slice(0, 2)
                                    b_cold_blks = tables[tag_k][0][sl_slice]
                                    b_reuse_blks = tables_reuse_u[tag_k][0][sl_slice]
                                    c_s = lc.kv_cache_base[b_cold_blks]
                                    r_s = lc.kv_cache_base[b_reuse_blks]
                                    if c_s.dtype in (torch.uint8, torch.int64, torch.int32):
                                        cos_k = _cosine(c_s.to(torch.float32), r_s.to(torch.float32))
                                    else:
                                        cos_k = _cosine(c_s, r_s)
                                    thr_k = 0.9999 if lid_k <= 3 else 0.995
                                    self.assertGreaterEqual(
                                        cos_k, thr_k, f"L{lid_k} {tag_k} prefix-cache snapshot cos={cos_k}"
                                    )
                        finally:
                            if prev_dec is None:
                                os.environ.pop("QWEN4_5_MXFP4_DECODE_BACKEND", None)
                            else:
                                os.environ["QWEN4_5_MXFP4_DECODE_BACKEND"] = prev_dec

                    # 4) CUDA Graph capture & multi-length replay (only for H=8, tpb in (64, 128))
                    if idx_n_heads == 8:
                        tables_cg = build_block_tables_for_requests(kv_cache, [136])
                        pfx_cg = make_tagged_attention_inputs(
                            tables_cg,
                            is_prefill=True,
                            input_lengths=[128],
                            sequence_lengths=[128],
                            prefix_lengths=[0],
                            device=dev,
                        )
                        rtp_model.forward(
                            LocalPyModelInputs(
                                input_ids=prompt_ids.reshape(-1), attention_inputs=pfx_cg
                            )
                        )
                        # Build B=2 padded CUDA Graph decode buffers (row 0 live, row 1 padded 0)
                        max_blk = max(len(r) for r in tables_cg[FULL_ATTN_KV_TAG])
                        cg_tables = {
                            t: [list(tables_cg[t][0]), [0] * len(tables_cg[t][0])]
                            for t in kv_cache.group_tags()
                        }
                        d_cg_attn = make_tagged_attention_inputs(
                            cg_tables,
                            is_prefill=False,
                            input_lengths=[1, 1],
                            sequence_lengths=[128, 0],
                            prefix_lengths=[0, 0],
                            device=dev,
                        )
                        for g in d_cg_attn.values():
                            g.is_cuda_graph = True
                            g.is_s_padded = True
                            g.sequence_lengths_plus_1_device = torch.tensor(
                                [129, 1], dtype=torch.int32, device=dev
                            )
                        static_ids = torch.tensor(
                            [int(dec_toks[0, 0].item()), 0], dtype=torch.int64, device=dev
                        )
                        static_in = LocalPyModelInputs(
                            input_ids=static_ids, attention_inputs=d_cg_attn
                        )
                        cg_handle = rtp_model.prepare_fmha_impl(static_in, is_cuda_graph=True)
                        # Snapshot ALL layers & tags before warmup/capture to keep GDN/ShortConv/QSA clean
                        snap_all = {
                            (lid_k, tag_k): lcache.kv_cache_base.clone()
                            for lid_k, tag_map in enumerate(kv_cache.layer_caches)
                            for tag_k, lcache in tag_map.items()
                        }

                        def _restore_all() -> None:
                            for (lid_k, tag_k), buf in snap_all.items():
                                kv_cache.layer_caches[lid_k][tag_k].kv_cache_base.copy_(buf)

                        # Eager B=2 padded decode baseline across steps 0..3 (same padded batch geometry)
                        eag_b2_logits = []
                        eag_b2_handle = rtp_model.prepare_fmha_impl(static_in, is_cuda_graph=False)
                        for step in range(4):
                            static_ids[0].copy_(dec_toks[0, step])
                            for grp in d_cg_attn.values():
                                grp.is_cuda_graph = False
                                grp.sequence_lengths_plus_1_device[0] = 128 + step + 1
                            out_b2 = rtp_model.forward(static_in, fmha_impl=eag_b2_handle)
                            eag_b2_logits.append(rtp_model.compute_logits(out_b2.hidden_states[:1]).clone())
                        for grp in d_cg_attn.values():
                            grp.is_cuda_graph = True
                            grp.sequence_lengths_plus_1_device[0] = 129
                        static_ids[0].copy_(dec_toks[0, 0])
                        _restore_all()

                        # Warmup + Capture step 0 (seq_before=128)
                        _ = rtp_model.forward(static_in, fmha_impl=cg_handle)
                        _restore_all()
                        g_obj = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g_obj):
                            static_out = rtp_model.forward(static_in, fmha_impl=cg_handle)
                        # Reset all caches after capture before replaying steps 0..3
                        _restore_all()
                        for step in range(4):
                            static_ids[0].copy_(dec_toks[0, step])
                            for grp in d_cg_attn.values():
                                grp.sequence_lengths_plus_1_device[0] = 128 + step + 1
                            g_obj.replay()
                            logits_cg = rtp_model.compute_logits(static_out.hidden_states[:1])
                            self.assertGreaterEqual(
                                _cosine(logits_cg, eag_b2_logits[step]),
                                0.9999,
                                f"CUDA Graph vs B=2 eager step={step} tpb={tpb} cos={_cosine(logits_cg, eag_b2_logits[step])}",
                            )
                        # Padded block 0 slot must remain untouched (0 writes)
                        self.assertEqual(
                            float(kv_cache.layer_caches[3][INDEXER_KV_TAG].kv_cache_base[0].abs().sum().item()),
                            0.0,
                        )
                        self.assertEqual(
                            float(kv_cache.layer_caches[3][INDEXER_STATE_TAG].kv_cache_base[0].abs().sum().item()),
                            0.0,
                        )

                        # Direct YoCo isolation check: consumer layer 7 must never mutate source layer 3 raw/indexer state
                        snap_ikv_before = kv_cache.layer_caches[3][INDEXER_KV_TAG].kv_cache_base.clone()
                        snap_ist_before = kv_cache.layer_caches[3][INDEXER_STATE_TAG].kv_cache_base.clone()
                        snap_raw_before = ref_state.qsa_raw_keys[3].clone()
                        _ = causal_lm.model.layers[7].self_attn(
                            torch.randn((1, 1, arch.hidden_size), dtype=dt, device=dev),
                            positions=torch.tensor([[131]], dtype=torch.int64, device=dev),
                            kv_cache=ref_state.full_attn_kv[3],
                            seq_offset=131,
                            qsa_source_raw_keys=ref_state.qsa_raw_keys[3],
                            qsa_source_k_norm_gamma=causal_lm.model.layers[3].self_attn.indexer.k_norm_gamma,
                        )
                        self.assertTrue(torch.equal(ref_state.qsa_raw_keys[3], snap_raw_before))
                        self.assertTrue(torch.equal(kv_cache.layer_caches[3][INDEXER_KV_TAG].kv_cache_base, snap_ikv_before))
                        self.assertTrue(torch.equal(kv_cache.layer_caches[3][INDEXER_STATE_TAG].kv_cache_base, snap_ist_before))


    def test_packed_vs_segmented_prefill_and_snapshots(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        dev = _device()
        dt = torch.bfloat16
        arch = _make_mini_arch(
            indexer_head_dim=128,
            indexer_n_heads=8,
            indexer_kv_heads=1,
            indexer_compress_ratio=4,
            indexer_budget=16,
            max_position_embeddings=512,
        )
        pools = {
            t: 64
            for t in [
                INDEXER_KV_TAG,
                INDEXER_STATE_TAG,
                FULL_ATTN_KV_TAG,
                "linear",
                "gdn_consumer_q_conv",
                "full_attn_kv_sconv",
                "full_attn_output_sconv",
                "linear_attn_output_sconv",
                "ffn_output_sconv",
                "ple_conv_state",
                "ple_ngram_ctx",
            ]
        }
        prev_b = os.environ.get("QWEN4_5_MXFP4_BACKEND")
        # Keep MoE BF16 activation precision consistent across M=64, M=3 segments and packed M=191
        os.environ["QWEN4_5_MXFP4_BACKEND"] = "triton"
        try:
            with torch.no_grad():
                causal_lm = Qwen4_5ForCausalLM(
                    arch, use_triton_mxfp4=True, compute_mode="bf16", dtype=dt, device=dev
                )
                causal_lm.init_synthetic_weights(seed=2029, ple_mode="gpu_full", ple_fp8=False)
                _stabilize_router_margins(causal_lm)
                causal_lm.prepare_weights()
                plan = build_qwen4_5_kv_cache_plan(arch)
                kv_pack = LocalKVCache(plan, tokens_per_block=64, num_blocks_per_tag=pools, device=dev, dtype=dt)
                kv_seg = LocalKVCache(plan, tokens_per_block=64, num_blocks_per_tag=pools, device=dev, dtype=dt)
                rtp_pack = Qwen4_5RTPModel.from_causal_lm(causal_lm, kv_cache=kv_pack)
                rtp_seg = Qwen4_5RTPModel.from_causal_lm(causal_lm, kv_cache=kv_seg)

                # B=3 ragged requests: req0 len=64 (pfx=0), req1 total=131 (pfx=64 + sfx=67), req2 len=60 (pfx=0)
                gen = torch.Generator(device="cpu").manual_seed(3029)
                ids0 = torch.randint(4, 240, (64,), generator=gen, dtype=torch.int64).to(dev)
                ids1 = torch.randint(4, 240, (131,), generator=gen, dtype=torch.int64).to(dev)
                ids2 = torch.randint(4, 240, (60,), generator=gen, dtype=torch.int64).to(dev)

                # Write req1 64-token prefix into both caches identically
                tb_pack = build_block_tables_for_requests(kv_pack, [72, 136, 72])
                tb_seg = build_block_tables_for_requests(kv_seg, [72, 136, 72])
                for tb, kv_c, rt in [(tb_pack, kv_pack, rtp_pack), (tb_seg, kv_seg, rtp_seg)]:
                    tb_r1 = {tag: [rows[1]] for tag, rows in tb.items()}
                    a_p1 = make_tagged_attention_inputs(
                        tb_r1, is_prefill=True, input_lengths=[64], sequence_lengths=[64], prefix_lengths=[0], device=dev
                    )
                    rt.forward(LocalPyModelInputs(input_ids=ids1[:64].contiguous(), attention_inputs=a_p1))

                # Packed ragged batch forward for [ids0 (64), ids1[64:] (67), ids2 (60)]
                packed_ids = torch.cat([ids0, ids1[64:], ids2], dim=0).contiguous()
                a_pack = make_tagged_attention_inputs(
                    tb_pack,
                    is_prefill=True,
                    input_lengths=[64, 67, 60],
                    sequence_lengths=[64, 131, 60],
                    prefix_lengths=[0, 64, 0],
                    device=dev,
                )
                out_pack = rtp_pack.forward(LocalPyModelInputs(input_ids=packed_ids, attention_inputs=a_pack))
                h_pack = out_pack.hidden_states
                pack_last = torch.stack([h_pack[63], h_pack[64 + 66], h_pack[64 + 67 + 59]], dim=0)
                logits_pack = rtp_pack.compute_logits(pack_last)

                # Reference segmented 64-token forward per request on kv_seg
                seg_last = []
                # req0: [0..64)
                tb_r0 = {tag: [rows[0]] for tag, rows in tb_seg.items()}
                o0 = rtp_seg.forward(LocalPyModelInputs(
                    input_ids=ids0,
                    attention_inputs=make_tagged_attention_inputs(
                        tb_r0, is_prefill=True, input_lengths=[64], sequence_lengths=[64], prefix_lengths=[0], device=dev
                    ),
                ))
                seg_last.append(o0.hidden_states[-1])
                # req1 suffix: two segments [64..128) (M=64) then [128..131) (M=3 >= 3 -> same deepgemm_masked)
                tb_r1 = {tag: [rows[1]] for tag, rows in tb_seg.items()}
                _ = rtp_seg.forward(LocalPyModelInputs(
                    input_ids=ids1[64:128].contiguous(),
                    attention_inputs=make_tagged_attention_inputs(
                        tb_r1, is_prefill=True, input_lengths=[64], sequence_lengths=[128], prefix_lengths=[64], device=dev
                    ),
                ))
                o1_b = rtp_seg.forward(LocalPyModelInputs(
                    input_ids=ids1[128:131].contiguous(),
                    attention_inputs=make_tagged_attention_inputs(
                        tb_r1, is_prefill=True, input_lengths=[3], sequence_lengths=[131], prefix_lengths=[128], device=dev
                    ),
                ))
                seg_last.append(o1_b.hidden_states[-1])
                # req2: [0..60)
                tb_r2 = {tag: [rows[2]] for tag, rows in tb_seg.items()}
                o2 = rtp_seg.forward(LocalPyModelInputs(
                    input_ids=ids2,
                    attention_inputs=make_tagged_attention_inputs(
                        tb_r2, is_prefill=True, input_lengths=[60], sequence_lengths=[60], prefix_lengths=[0], device=dev
                    ),
                ))
                seg_last.append(o2.hidden_states[-1])
                logits_seg = rtp_seg.compute_logits(torch.stack(seg_last, dim=0))

                for bi in range(3):
                    cos_b = _cosine(logits_pack[bi : bi + 1], logits_seg[bi : bi + 1])
                    self.assertGreaterEqual(cos_b, 0.999, f"Packed vs Segmented req={bi} cos={cos_b}")

                # Verify 64-token cache snapshots across ALL tags match segmented execution
                # req0 block 0 (token 64 boundary), req1 blocks 0 & 1 (token 64 & 128 boundaries), req2 block 0
                for req_i, n_blks in ((0, 1), (1, 2), (2, 1)):
                    for lid_k, tag_map in enumerate(kv_pack.layer_caches):
                        for tag_k, lc_p in tag_map.items():
                            lc_s = kv_seg.layer_caches[lid_k][tag_k]
                            p_blks = tb_pack[tag_k][req_i][:n_blks]
                            s_blks = tb_seg[tag_k][req_i][:n_blks]
                            t_p = lc_p.kv_cache_base[p_blks]
                            t_s = lc_s.kv_cache_base[s_blks]
                            if t_p.dtype in (torch.int64, torch.int32):
                                self.assertTrue(torch.equal(t_p, t_s), f"req={req_i} L{lid_k} {tag_k} int snapshot mismatch")
                            else:
                                thr_s = 0.9999 if lid_k <= 3 else 0.999
                                cos_s = _cosine(t_p.to(torch.float32), t_s.to(torch.float32))
                                self.assertGreaterEqual(cos_s, thr_s, f"req={req_i} L{lid_k} {tag_k} snapshot cos={cos_s}")

                # Quick latency check for Packed Prefill (B=3, 189 tokens)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(5):
                    rtp_pack.forward(LocalPyModelInputs(input_ids=packed_ids, attention_inputs=a_pack))
                torch.cuda.synchronize()
                ms_pack = (time.perf_counter() - t0) * 1000.0 / 5.0
                print(f"[Packed Prefill B=3, N=189 + QSA + Snapshots] avg={ms_pack:.2f} ms")
        finally:
            if prev_b is None:
                os.environ.pop("QWEN4_5_MXFP4_BACKEND", None)
            else:
                os.environ["QWEN4_5_MXFP4_BACKEND"] = prev_b

    def test_target_verify_v8_vs_sequential_and_cuda_graph(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        dev = _device()
        dt = torch.bfloat16
        arch = _make_mini_arch(
            indexer_head_dim=128,
            indexer_n_heads=8,
            indexer_kv_heads=1,
            indexer_compress_ratio=4,
            indexer_budget=16,
            max_position_embeddings=512,
        )
        pools = {
            t: 48
            for t in [
                INDEXER_KV_TAG,
                INDEXER_STATE_TAG,
                FULL_ATTN_KV_TAG,
                "linear",
                "gdn_consumer_q_conv",
                "full_attn_kv_sconv",
                "full_attn_output_sconv",
                "linear_attn_output_sconv",
                "ffn_output_sconv",
                "ple_conv_state",
                "ple_ngram_ctx",
            ]
        }
        with torch.no_grad():
            causal_lm = Qwen4_5ForCausalLM(
                arch, use_triton_mxfp4=True, compute_mode="bf16", dtype=dt, device=dev
            )
            causal_lm.init_synthetic_weights(seed=2033, ple_mode="gpu_full", ple_fp8=False)
            _stabilize_router_margins(causal_lm)
            causal_lm.prepare_weights()
            plan = build_qwen4_5_kv_cache_plan(arch)
            kv_cache = LocalKVCache(plan, tokens_per_block=64, num_blocks_per_tag=pools, device=dev, dtype=dt)
            rt = Qwen4_5RTPModel.from_causal_lm(causal_lm, kv_cache=kv_cache)

            # Prefill B=2 near tpb=64 boundary [62, 63] so V=8 verify crosses 63->64 block boundary
            gen = torch.Generator(device="cpu").manual_seed(3033)
            p0 = torch.randint(4, 240, (62,), generator=gen, dtype=torch.int64).to(dev)
            p1 = torch.randint(4, 240, (63,), generator=gen, dtype=torch.int64).to(dev)
            v_toks = torch.randint(4, 240, (2, 8), generator=gen, dtype=torch.int64).to(dev)
            next_tok = torch.randint(4, 240, (2,), generator=gen, dtype=torch.int64).to(dev)

            tbls = build_block_tables_for_requests(kv_cache, [80, 80])
            pref_in = LocalPyModelInputs(
                input_ids=torch.cat([p0, p1], dim=0),
                attention_inputs=make_tagged_attention_inputs(
                    tbls, is_prefill=True, input_lengths=[62, 63], sequence_lengths=[62, 63], prefix_lengths=[0, 0], device=dev
                ),
            )
            rt.forward(pref_in)
            pre_snap = {
                (lid_k, tag_k): lc.kv_cache_base.clone()
                for lid_k, tag_map in enumerate(kv_cache.layer_caches)
                for tag_k, lc in tag_map.items()
            }

            def _restore() -> None:
                for (lid_k, tag_k), buf in pre_snap.items():
                    kv_cache.layer_caches[lid_k][tag_k].kv_cache_base.copy_(buf)

            v_attn = make_tagged_attention_inputs(
                tbls, is_prefill=True, input_lengths=[8, 8], sequence_lengths=[], prefix_lengths=[62, 63], device=dev
            )
            for grp in v_attn.values():
                grp.is_target_verify = True
            v_in = LocalPyModelInputs(input_ids=v_toks.reshape(-1), attention_inputs=v_attn)

            # 1) Non-committed verify + rollback preserves persistent pools
            ver_out = rt.forward(v_in)
            acc = torch.tensor([5, 3], dtype=torch.int32, device=dev)
            rt.prepare_speculative_target_commit(acc)
            rt.finish_speculative_target_commit(False)
            rt.finalize_speculative_target_commit()
            for (lid_k, tag_k), buf in pre_snap.items():
                if tag_k not in ("full", FULL_ATTN_KV_TAG, INDEXER_KV_TAG):
                    self.assertTrue(
                        torch.equal(kv_cache.layer_caches[lid_k][tag_k].kv_cache_base, buf),
                        f"Rollback failed for L{lid_k} {tag_k}",
                    )

            # 2) Verify + Commit([5, 3]) + Continuation step vs 8x sequential decode
            _restore()
            ver_out = rt.forward(v_in)
            rt.prepare_speculative_target_commit(acc)
            rt.finish_speculative_target_commit(True)
            rt.finalize_speculative_target_commit()
            cont_ver = rt.forward(
                LocalPyModelInputs(
                    input_ids=next_tok,
                    attention_inputs=make_tagged_attention_inputs(
                        tbls, is_prefill=False, input_lengths=[1, 1], sequence_lengths=[67, 66], prefix_lengths=[0, 0], device=dev
                    ),
                )
            )
            ver_cont_logits = rt.compute_logits(cont_ver.hidden_states)

            # Sequential step-by-step reference (B=2 for first 3 steps, then r0 for 2 steps, then next)
            _restore()
            seq_h0, seq_h1 = [], []
            tb0 = {tag: [rows[0]] for tag, rows in tbls.items()}
            for s in range(8):
                o_s = rt.forward(
                    LocalPyModelInputs(
                        input_ids=v_toks[0 : 1, s].contiguous(),
                        attention_inputs=make_tagged_attention_inputs(
                            tb0, is_prefill=False, input_lengths=[1], sequence_lengths=[62 + s], prefix_lengths=[0], device=dev
                        ),
                    )
                )
                seq_h0.append(o_s.hidden_states[0])
            _restore()
            tb1 = {tag: [rows[1]] for tag, rows in tbls.items()}
            for s in range(8):
                o_s = rt.forward(
                    LocalPyModelInputs(
                        input_ids=v_toks[1 : 2, s].contiguous(),
                        attention_inputs=make_tagged_attention_inputs(
                            tb1, is_prefill=False, input_lengths=[1], sequence_lengths=[63 + s], prefix_lengths=[0], device=dev
                        ),
                    )
                )
                seq_h1.append(o_s.hidden_states[0])
            seq_h = torch.stack(seq_h0 + seq_h1, dim=0)
            self.assertGreaterEqual(_cosine(ver_out.hidden_states, seq_h), 0.999)

            # Also check post-accept continuation parity
            _restore()
            for s in range(5):
                rt.forward(LocalPyModelInputs(
                    input_ids=v_toks[0 : 1, s].contiguous(),
                    attention_inputs=make_tagged_attention_inputs(
                        tb0, is_prefill=False, input_lengths=[1], sequence_lengths=[62 + s], prefix_lengths=[0], device=dev
                    ),
                ))
            for s in range(3):
                rt.forward(LocalPyModelInputs(
                    input_ids=v_toks[1 : 2, s].contiguous(),
                    attention_inputs=make_tagged_attention_inputs(
                        tb1, is_prefill=False, input_lengths=[1], sequence_lengths=[63 + s], prefix_lengths=[0], device=dev
                    ),
                ))
            cont_seq = rt.forward(
                LocalPyModelInputs(
                    input_ids=next_tok,
                    attention_inputs=make_tagged_attention_inputs(
                        tbls, is_prefill=False, input_lengths=[1, 1], sequence_lengths=[67, 66], prefix_lengths=[0, 0], device=dev
                    ),
                )
            )
            seq_cont_logits = rt.compute_logits(cont_seq.hidden_states)
            self.assertGreaterEqual(_cosine(ver_cont_logits, seq_cont_logits), 0.999)

            # 3) V=8 Target Verify CUDA Graph capture & multi-replay
            _restore()
            v_cg_attn = make_tagged_attention_inputs(
                tbls, is_prefill=True, input_lengths=[8, 8], sequence_lengths=[], prefix_lengths=[62, 63], device=dev
            )
            for grp in v_cg_attn.values():
                grp.is_target_verify = True
                grp.prefix_lengths = grp.prefix_lengths.to(device=dev, dtype=torch.int32)
            static_ids = v_toks.reshape(-1).clone()
            cg_in = LocalPyModelInputs(input_ids=static_ids, attention_inputs=v_cg_attn)
            cg_handle = rt.prepare_fmha_impl(cg_in, is_cuda_graph=True)
            self.assertTrue(cg_handle.support_cuda_graph())

            _ = rt.forward(cg_in, fmha_impl=cg_handle)
            rt.on_cuda_graph_warmup(cg_in)
            _restore()
            g_ver = torch.cuda.CUDAGraph()
            cg_handle.prepare_cuda_graph(cg_in.attention_inputs)
            with torch.cuda.graph(g_ver):
                cg_out = rt.forward(cg_in, fmha_impl=cg_handle)
            torch.cuda.synchronize()
            rt.on_cuda_graph_capture(2, cg_in)
            # Mutate tokens & replay
            new_v = torch.randint(4, 240, (16,), generator=gen, dtype=torch.int64).to(dev)
            static_ids.copy_(new_v)
            _restore()
            cg_handle.prepare_cuda_graph(cg_in.attention_inputs)
            g_ver.replay()
            torch.cuda.synchronize()
            rt.on_cuda_graph_replay(2, cg_in)
            cg_h = cg_out.hidden_states.clone()
            _restore()
            v_eag_attn = make_tagged_attention_inputs(
                tbls, is_prefill=True, input_lengths=[8, 8], sequence_lengths=[], prefix_lengths=[62, 63], device=dev
            )
            for grp in v_eag_attn.values():
                grp.is_target_verify = True
            eag_out = rt.forward(LocalPyModelInputs(input_ids=new_v, attention_inputs=v_eag_attn))
            cg_cos = _cosine(cg_h, eag_out.hidden_states)
            self.assertGreaterEqual(cg_cos, 0.99999, f"V=8 Verify CG vs Eager cos={cg_cos}")

            # Quick V=8 CG vs 8x sequential decode timing
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                g_ver.replay()
            torch.cuda.synchronize()
            ms_cg = (time.perf_counter() - t0) * 1000.0 / 10.0
            print(f"[Target Verify V=8 B=2 + QSA CG] replay={ms_cg:.2f} ms, CG_cos={cg_cos:.6f}")

    def test_checkpoint_roundtrip_qsa_h8(self) -> None:
        arch = _make_mini_arch(indexer_n_heads=8, indexer_budget=2048)
        dev = torch.device("cpu")
        with tempfile.TemporaryDirectory(prefix="qwen45_qsa_ckpt_") as tmp:
            ckpt = Path(tmp)
            with torch.no_grad():
                m_src = Qwen4_5ForCausalLM(
                    arch, use_triton_mxfp4=False, compute_mode="bf16", dtype=torch.bfloat16, device=dev
                )
                m_src.init_synthetic_weights(seed=999, ple_mode="gpu_full", ple_fp8=False)
                save_qwen4_5_safetensors_checkpoint(m_src, ckpt)
                parsed_arch = parse_qwen4_5_config_json(
                    json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
                )
                self.assertTrue(parsed_arch.has_qsa)
                self.assertEqual(parsed_arch.indexer_head_dim, 128)
                self.assertEqual(parsed_arch.indexer_n_heads, 8)
                self.assertEqual(parsed_arch.indexer_kv_heads, 1)
                self.assertEqual(parsed_arch.indexer_compress_ratio, 4)
                self.assertEqual(parsed_arch.indexer_budget, 2048)

                m_dst = Qwen4_5ForCausalLM(
                    parsed_arch, use_triton_mxfp4=False, compute_mode="bf16", dtype=torch.bfloat16, device=dev
                )
                load_qwen4_5_from_checkpoint(m_dst, ckpt, ple_mode="gpu_full", num_workers=1)

                ids = torch.randint(4, 240, (1, 16), dtype=torch.int64, device=dev)
                st1 = m_src.create_runtime_state(1, 32, device=dev, dtype=torch.bfloat16)
                st2 = m_dst.create_runtime_state(1, 32, device=dev, dtype=torch.bfloat16)
                out1 = m_src(ids, state=st1)
                out2 = m_dst(ids, state=st2)
                self.assertGreaterEqual(_cosine(out1, out2), 0.99999)

    @torch.no_grad()
    def test_r0_20k_indexer_topk_sparse_prefill_and_uniform_batch(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        dev = torch.device("cuda", 0)
        torch.manual_seed(20261125)

        # 1) 20K Indexer TensorCore parity: ctx=20480 (5120 blks), BF16 + FP8, H_idx in (4, 8), S in (8, 256)
        total_blks = 5120
        kv_tpb = 64
        ent_per_blk = kv_tpb // 4  # 16
        blks_per_req = total_blks // ent_per_blk  # 320
        for n_heads in (4, 8):
            kv_pool_bf16 = torch.randn((blks_per_req * 2 + 8, ent_per_blk, 128), device=dev, dtype=torch.bfloat16)
            kv_pool_bf16[0].zero_()
            kv_pool_fp8 = kv_pool_bf16.to(torch.float8_e4m3fn)
            for bsz, s_q in ((2, 8), (64, 8), (1, 256)):
                kv_tbl = torch.randint(1, blks_per_req * 2 + 8, (bsz, blks_per_req), device=dev, dtype=torch.int32)
                q_rot = torch.randn((bsz, s_q, n_heads, 128), device=dev, dtype=torch.bfloat16)
                ctx_lens = torch.full((bsz, s_q), total_blks, device=dev, dtype=torch.int32)
                ctx_lens[0, 0] = 317
                # Pure PyTorch FP32 reference on sample batch elements
                ref_list = []
                for bi in range(min(bsz, 2)):
                    keys_bi = kv_pool_bf16[kv_tbl[bi]].reshape(total_blks, 128).float()
                    sc_bi = torch.einsum("shd,kd->shk", q_rot[bi].float(), keys_bi).relu().sum(dim=1)
                    mask_bi = torch.arange(total_blks, device=dev).unsqueeze(0) < ctx_lens[bi].unsqueeze(1)
                    ref_list.append(sc_bi.masked_fill(~mask_bi, float("-inf")))
                ref_01 = torch.stack(ref_list, dim=0)

                out_bf16 = qsa_paged_score(q_rot, kv_pool_bf16, kv_tbl, ctx_lens, max_ctx_len=total_blks).view(bsz, s_q, total_blks)
                cos_bf16 = _cosine(out_bf16[: min(bsz, 2)][torch.isfinite(ref_01)], ref_01[torch.isfinite(ref_01)])
                self.assertGreaterEqual(cos_bf16, 0.99999, f"20K BF16 TC indexer cos={cos_bf16}")
                self.assertTrue(torch.isinf(out_bf16[0, 0, 317:]).all())

                out_fp8 = qsa_paged_score(q_rot, kv_pool_fp8, kv_tbl, ctx_lens, max_ctx_len=total_blks).view(bsz, s_q, total_blks)
                cos_fp8 = _cosine(out_fp8[: min(bsz, 2)][torch.isfinite(ref_01)], ref_01[torch.isfinite(ref_01)])
                self.assertGreaterEqual(cos_fp8, 0.999, f"20K FP8 TC indexer cos={cos_fp8}")

        # 2) _select_top_tokens binary-search fast path (5120 -> 512) + vis < 2048 + M=8, 512, 1024
        for rows in (8, 512, 1024):
            sc = torch.randn((rows, 5120), device=dev, dtype=torch.float32)
            vis = torch.full((rows,), 20480, device=dev, dtype=torch.int32)
            vis[0] = 3
            vis[1] = 21
            vis[2] = 512
            sc[0, :] = float("-inf")
            sc[1, 5:] = float("-inf")
            sc[2, 128:] = float("-inf")
            sel = _select_top_tokens(sc, vis, compress_ratio=4, token_budget=2048, scores_are_masked=True)
            self.assertEqual(sel.shape, (rows, 2051))
            # Check tail indices (last 3 slots are vis-3..vis-1, or clamped >=0) and exact top-512 block set on valid rows
            from rtp_llm.models_py.modules.qwen4_5.qsa_bridge import select_qsa_paged_tokens
            sel_ref_edge = select_qsa_paged_tokens(
                sc[:3], vis[:3], compress_ratio=4, token_budget=2048, validate_lengths=False, scores_are_masked=True
            )
            for r in (0, 1, 2, rows - 1):
                v_r = int(vis[r].item())
                self.assertTrue(((sel[r] == -1) | ((sel[r] >= 0) & (sel[r] < v_r))).all())
                if r < 3:
                    self.assertEqual(
                        set(sel[r][sel[r] >= 0].tolist()),
                        set(sel_ref_edge[r][sel_ref_edge[r] >= 0].tolist()),
                    )
                    self.assertEqual(int((sel[r] == -1).sum()), int((sel_ref_edge[r] == -1).sum()))
            for r in (3, rows - 1):
                _, top_ref = torch.topk(sc[r], k=512, dim=-1)
                blk_got = set((sel[r, :2048] // 4).unique().tolist())
                blk_exp = set(top_ref.tolist())
                self.assertEqual(blk_got, blk_exp, f"bsearch topk block set mismatch row={r}")

        # 3) 90%-Hit 18K+2K Sparse Prefill (T=20480, S=2048, K=2051, CHECK_EMPTY=True) parity
        q_sp = torch.randn((1, 16, 2048, 128), device=dev, dtype=torch.bfloat16)
        k_sp = torch.randn((1, 2, 20480, 128), device=dev, dtype=torch.bfloat16) * 0.25
        v_sp = torch.randn((1, 2, 20480, 128), device=dev, dtype=torch.bfloat16)
        # Unique 512-block selection (2048 unique tokens + 3 sentinel slots -> exercises CHECK_EMPTY=True bitmap path)
        blks_sp = (torch.arange(512, device=dev, dtype=torch.int32).unsqueeze(0) * 9 + torch.arange(2048, device=dev, dtype=torch.int32).unsqueeze(1) * 13) % 5120
        tok_sp = (blks_sp.unsqueeze(-1) * 4 + torch.arange(4, device=dev, dtype=torch.int32)).reshape(1, 2048, 2048)
        sel_sp = torch.cat([tok_sp, torch.full((1, 2048, 3), -1, device=dev, dtype=torch.int32)], dim=-1)
        out_sp = qsa_sparse_prefill_attention(q_sp, k_sp, v_sp, sel_sp)
        # Check subset of queries against exact PyTorch gather reference
        sub_q_idx = [0, 511, 1500, 2047]
        ref_sub = []
        scale = 1.0 / math.sqrt(128.0)
        for qi in sub_q_idx:
            uq_idx = sel_sp[0, qi][sel_sp[0, qi] >= 0].to(torch.long)
            k_g = k_sp[0, :, uq_idx].float().repeat_interleave(8, dim=0)
            v_g = v_sp[0, :, uq_idx].float().repeat_interleave(8, dim=0)
            logits = torch.matmul(q_sp[0, :, qi].float().unsqueeze(1), k_g.transpose(1, 2)) * scale
            w = torch.softmax(logits, dim=-1)
            ref_sub.append(torch.matmul(w, v_g).squeeze(1))
        ref_stack = torch.stack(ref_sub, dim=1)
        out_sub = out_sp[0, :, sub_q_idx].float()
        self.assertGreaterEqual(_cosine(out_sub, ref_stack), 0.9999)

        # 4) Uniform batched FullAttn / QSA prefill in run_prefill_ragged vs single-req prefill (with prefix KV cache + YoCo + PLE)
        arch = _make_mini_arch(indexer_n_heads=4, indexer_budget=16, max_position_embeddings=512)
        m_u = Qwen4_5ForCausalLM(
            arch, use_triton_mxfp4=False, compute_mode="bf16", dtype=torch.bfloat16, device=dev
        ).eval()
        m_u.init_synthetic_weights(seed=777, ple_mode="gpu_full", ple_fp8=False)
        _stabilize_router_margins(m_u)
        m_u.prepare_weights()
        pools = {
            tag: 64
            for tag in [
                INDEXER_KV_TAG,
                INDEXER_STATE_TAG,
                FULL_ATTN_KV_TAG,
                "linear",
                "gdn_consumer_q_conv",
                "full_attn_kv_sconv",
                "full_attn_output_sconv",
                "linear_attn_output_sconv",
                "ffn_output_sconv",
                "ple_conv_state",
                "ple_ngram_ctx",
            ]
        }
        plan_u = build_qwen4_5_kv_cache_plan(arch)
        cache_u = LocalKVCache(plan_u, tokens_per_block=64, num_blocks_per_tag=pools, device=dev, dtype=torch.bfloat16)
        cache_s = LocalKVCache(plan_u, tokens_per_block=64, num_blocks_per_tag=pools, device=dev, dtype=torch.bfloat16)
        rt_u = Qwen4_5RTPModel.from_causal_lm(m_u, kv_cache=cache_u).eval()
        rt_s = Qwen4_5RTPModel.from_causal_lm(m_u, kv_cache=cache_s).eval()

        gen = torch.Generator(device="cpu").manual_seed(91)
        bsz_u, t_pfx, t_suf = 3, 24, 18
        pfx_ids = torch.randint(4, 240, (bsz_u, t_pfx), generator=gen, dtype=torch.int64).to(dev)
        suf_ids = torch.randint(4, 240, (bsz_u, t_suf), generator=gen, dtype=torch.int64).to(dev)
        tbl_u = build_block_tables_for_requests(cache_u, [t_pfx + t_suf] * bsz_u)
        tbl_s = build_block_tables_for_requests(cache_s, [t_pfx + t_suf] * bsz_u)
        # Seed prefix in both caches individually, then run uniform batched suffix vs individual suffix
        for rt_item, tbl_item in ((rt_u, tbl_u), (rt_s, tbl_s)):
            for bi in range(bsz_u):
                tb1 = {tag: [rows[bi]] for tag, rows in tbl_item.items()}
                ai1 = make_tagged_attention_inputs(
                    tb1, is_prefill=True, input_lengths=[t_pfx], sequence_lengths=[t_pfx], prefix_lengths=[0], device=dev
                )
                rt_item.forward(LocalPyModelInputs(input_ids=pfx_ids[bi], attention_inputs=ai1))
        ai_u = make_tagged_attention_inputs(
            tbl_u, is_prefill=True, input_lengths=[t_suf] * bsz_u, sequence_lengths=[t_pfx + t_suf] * bsz_u, prefix_lengths=[t_pfx] * bsz_u, device=dev
        )
        out_u_batched = rt_u.forward(LocalPyModelInputs(input_ids=suf_ids.reshape(-1), attention_inputs=ai_u)).hidden_states
        outs_single = []
        for bi in range(bsz_u):
            tb1 = {tag: [rows[bi]] for tag, rows in tbl_s.items()}
            ai1 = make_tagged_attention_inputs(
                tb1, is_prefill=True, input_lengths=[t_suf], sequence_lengths=[t_pfx + t_suf], prefix_lengths=[t_pfx], device=dev
            )
            outs_single.append(rt_s.forward(LocalPyModelInputs(input_ids=suf_ids[bi], attention_inputs=ai1)).hidden_states)
        out_u_single = torch.cat(outs_single, dim=0)
        # Exact parity vs per-request ragged loop (trigger via inhomogeneous trailing length) + single-request check
        cache_nh = LocalKVCache(plan_u, tokens_per_block=64, num_blocks_per_tag=pools, device=dev, dtype=torch.bfloat16)
        rt_nh = Qwen4_5RTPModel.from_causal_lm(m_u, kv_cache=cache_nh).eval()
        tbl_nh = build_block_tables_for_requests(cache_nh, [t_pfx + t_suf + 1] * bsz_u)
        for bi in range(bsz_u):
            tb1 = {tag: [rows[bi]] for tag, rows in tbl_nh.items()}
            ai1 = make_tagged_attention_inputs(
                tb1, is_prefill=True, input_lengths=[t_pfx], sequence_lengths=[t_pfx], prefix_lengths=[0], device=dev
            )
            rt_nh.forward(LocalPyModelInputs(input_ids=pfx_ids[bi], attention_inputs=ai1))
        nh_ids = torch.cat([suf_ids[0], suf_ids[1], suf_ids[2], suf_ids[2, -1:]], dim=0)
        ai_nh = make_tagged_attention_inputs(
            tbl_nh, is_prefill=True, input_lengths=[t_suf, t_suf, t_suf + 1], sequence_lengths=[t_pfx + t_suf, t_pfx + t_suf, t_pfx + t_suf + 1], prefix_lengths=[t_pfx] * bsz_u, device=dev
        )
        out_nh = rt_nh.forward(LocalPyModelInputs(input_ids=nh_ids, attention_inputs=ai_nh)).hidden_states
        self.assertGreaterEqual(_cosine(out_u_batched[: 2 * t_suf], out_nh[: 2 * t_suf]), 0.999999)
        u_cos = _cosine(out_u_batched, out_u_single)
        self.assertGreaterEqual(u_cos, 0.998, f"Uniform batched prefill vs single cos={u_cos}")


    def test_r1_qsa_gqa_tc_verify_bf16_fp8_gate_and_cg(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        dev = torch.device("cuda")
        from rtp_llm.models_py.kernels.cuda.qsa.sparse_paged_fmha import sparse_paged_gqa_attn

        torch.manual_seed(202611)
        H_q, H_kv, D, P, K = 8, 1, 256, 64, 2051
        for B, S in [(1, 8), (8, 8), (64, 8)]:
            ctx = 20480
            n_req_pages = ctx // P
            pages = B * n_req_pages + 1
            q = torch.randn((B, H_q, S, D), device=dev, dtype=torch.bfloat16)
            kv = torch.randn((pages, 2, H_kv, P, D), device=dev, dtype=torch.bfloat16) * 0.25
            bt = torch.arange(1, B * n_req_pages + 1, dtype=torch.int32, device=dev).view(B, n_req_pages)
            kv_lens = torch.full((B, S), ctx, dtype=torch.int32, device=dev)
            kv_lens[:, -1] = ctx - 3  # vary tail length
            sel = torch.full((B, S, K), -1, dtype=torch.int32, device=dev)
            for si in range(S):
                blks = torch.randint(0, ctx // 4 - 4, (B, 512), device=dev, dtype=torch.int32)
                sel[:, si, :2048] = (
                    blks.unsqueeze(-1) * 4 + torch.arange(4, device=dev, dtype=torch.int32)
                ).reshape(B, 2048)
                sel[:, si, 2048] = 0
                sel[:, si, 2049] = ctx - 4

            gate = torch.randn((B, S, H_q, D), device=dev, dtype=torch.bfloat16)
            out_tc = qsa_sparse_paged_decode_attention(
                q, kv, bt, kv_lens, sel, page_size=P, kv_heads=H_kv, head_dim=D, output_gate=gate
            )
            out_ref = sparse_paged_gqa_attn(
                q, kv, bt, kv_lens, sel, page_size=P, kv_head_num=H_kv, output_gate=gate, validate_selected=False
            )
            self.assertGreaterEqual(_cosine(out_tc, out_ref), 0.99999)
            self.assertEqual(tuple(out_tc.stride()), (S * H_q * D, D, H_q * D, 1))

            # HybridPool 2D + FP8 KV (both per-token HND scale, [2] scalar scale, and unit None)
            if B == 8:
                pool_2d = kv.reshape(pages, -1).contiguous()
                out_2d = qsa_sparse_paged_decode_attention(
                    q, pool_2d, bt, kv_lens, sel, page_size=P, kv_heads=H_kv, head_dim=D
                )
                out_ref_ng = sparse_paged_gqa_attn(
                    q, kv, bt, kv_lens, sel, page_size=P, kv_head_num=H_kv, validate_selected=False
                )
                self.assertGreaterEqual(_cosine(out_2d, out_ref_ng), 0.99999)

                kv_fp8 = kv.to(torch.float8_e4m3fn)
                sc_hnd = torch.full((pages, 2, H_kv, P), 0.5, dtype=torch.float32, device=dev)
                out_fp8_hnd = qsa_sparse_paged_decode_attention(
                    q, kv_fp8, bt, kv_lens, sel, page_size=P, kv_heads=H_kv, head_dim=D, kv_scale=sc_hnd, output_gate=gate
                )
                out_fp8_ref = sparse_paged_gqa_attn(
                    q, kv_fp8, bt, kv_lens, sel, page_size=P, kv_head_num=H_kv, kv_scale=sc_hnd, output_gate=gate, validate_selected=False
                )
                self.assertGreaterEqual(_cosine(out_fp8_hnd, out_fp8_ref), 0.9995)

                sc_vec = torch.tensor([0.5, 0.5], dtype=torch.float32, device=dev)
                out_fp8_vec = qsa_sparse_paged_decode_attention(
                    q, kv_fp8, bt, kv_lens, sel, page_size=P, kv_heads=H_kv, head_dim=D, kv_scale=sc_vec, output_gate=gate
                )
                self.assertGreaterEqual(_cosine(out_fp8_vec, out_fp8_hnd), 0.99999)

                out_fp8_unit = qsa_sparse_paged_decode_attention(
                    q, kv_fp8, bt, kv_lens, sel, page_size=P, kv_heads=H_kv, head_dim=D, kv_scale=None
                )
                self.assertGreaterEqual(_cosine(out_fp8_unit, out_ref_ng), 0.999)

                # CUDA Graph capture & replay parity
                st_q = q.clone()
                st_bt = bt.clone()
                st_len = kv_lens.clone()
                st_sel = sel.clone()
                st_gate = gate.clone()
                # Warmup
                _ = qsa_sparse_paged_decode_attention(
                    st_q, kv, st_bt, st_len, st_sel, page_size=P, kv_heads=H_kv, head_dim=D, output_gate=st_gate, is_cuda_graph=True
                )
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    cg_out = qsa_sparse_paged_decode_attention(
                        st_q, kv, st_bt, st_len, st_sel, page_size=P, kv_heads=H_kv, head_dim=D, output_gate=st_gate, is_cuda_graph=True
                    )
                st_q.copy_(torch.randn_like(st_q))
                st_gate.copy_(torch.randn_like(st_gate))
                g.replay()
                eag_out = qsa_sparse_paged_decode_attention(
                    st_q, kv, st_bt, st_len, st_sel, page_size=P, kv_heads=H_kv, head_dim=D, output_gate=st_gate
                )
                self.assertGreaterEqual(_cosine(cg_out, eag_out), 0.999999)

    def test_r1_radix2_20k_topk_and_q4_paged_prefill_parity(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        dev = torch.device("cuda")
        from rtp_llm.models_py.modules.qwen4_5.qsa_bridge import select_qsa_paged_tokens

        torch.manual_seed(202612)
        # 1) 20K (cols=20480 -> ctx=81920) radix2 chunked TopK parity
        for rows in (8, 64, 128):
            cols = 20480
            sc = torch.randn((rows, cols), device=dev, dtype=torch.float32)
            sc[0, :128] = 5.0  # force ties
            sc[1, :] = float("-inf")
            vis = torch.full((rows,), cols * 4, dtype=torch.int32, device=dev)
            vis[0] = 260 * 4 + 3  # short + tail=3
            sc[0, 260:] = float("-inf")
            vis[1] = 2  # empty complete blocks + tail=2
            sel = _select_top_tokens(
                sc, vis, compress_ratio=4, token_budget=2048, scores_are_masked=True
            )
            sel_ref = select_qsa_paged_tokens(
                sc[:3], vis[:3], compress_ratio=4, token_budget=2048, validate_lengths=False, scores_are_masked=True
            )
            for r in range(3):
                self.assertEqual(
                    set(sel[r][sel[r] >= 0].tolist()),
                    set(sel_ref[r][sel_ref[r] >= 0].tolist()),
                )
                self.assertEqual(int((sel[r] == -1).sum()), int((sel_ref[r] == -1).sum()))
            for r in (3, rows - 1):
                _, top_ref = torch.topk(sc[r], k=512, dim=-1)
                self.assertEqual(set((sel[r, :2048] // 4).unique().tolist()), set(top_ref.tolist()))

        # 2) Sparse Prefill (Dense + Paged + HybridPool + FP8, 85% Q4 overlap & 0%/dup Direct fallback)
        B, H_q, H_kv, S, T, D, P, K = 4, 8, 1, 128, 20480, 256, 64, 2051
        pages = B * (T // P) + 1
        q = torch.randn((B, H_q, S, D), device=dev, dtype=torch.bfloat16)
        kv = torch.randn((pages, 2, H_kv, P, D), device=dev, dtype=torch.bfloat16) * 0.25
        bt = torch.arange(1, pages, dtype=torch.int32, device=dev).view(B, T // P)
        k_dense = kv[bt.long(), 0].permute(0, 2, 1, 3, 4).reshape(B, H_kv, T, D).contiguous()
        v_dense = kv[bt.long(), 1].permute(0, 2, 1, 3, 4).reshape(B, H_kv, T, D).contiguous()

        n_q = T // 4
        # 85% Q4 overlap selections (distinct within row)
        sel_ov = torch.full((B, S, K), -1, dtype=torch.int32, device=dev)
        for bi in range(B):
            base = torch.randperm(n_q - 2, device=dev, dtype=torch.int32) + 1  # reserve quads 0 and n_q-1 for tail slots
            for g in range(S // 4):
                anchor = base[:512]
                for j in range(4):
                    s = g * 4 + j
                    r = j * 24
                    blks = torch.cat([anchor[r:], base[512 : 512 + r]])
                    sel_ov[bi, s, :2048] = (
                        blks.unsqueeze(-1) * 4 + torch.arange(4, device=dev, dtype=torch.int32)
                    ).reshape(-1)
                    if j == 3:
                        base = base.roll(-512)
        sel_ov[:, :, 2048] = 3
        sel_ov[:, :, 2049] = T - 1

        # Exact FP32 ref on sampled positions
        def _ref_at(b_i: int, s_i: int, sel_t: torch.Tensor) -> torch.Tensor:
            idx = sel_t[b_i, s_i]
            idx = idx[(idx >= 0) & (idx < T)].long()
            sc_l = (q[b_i, :, s_i, :].float() @ k_dense[b_i, 0, idx].float().T) * (D ** -0.5)
            return (torch.softmax(sc_l, dim=-1) @ v_dense[b_i, 0, idx].float()).to(torch.bfloat16)

        # Explicitly verify non-fallback Q4 path on both dense [B, H_kv, T, D] and paged [pages, 2, H_kv, P, D]
        out_q4_dense_direct = _try_run_q4_prefill(
            q, k_dense, v_dense, None, None, sel_ov, T=T, page_size=4, is_paged=False
        )
        self.assertIsNotNone(out_q4_dense_direct, "Dense Q4 prefill unexpectedly fell back")
        out_q4_paged_direct = _try_run_q4_prefill(
            q, kv, kv, None, bt, sel_ov, T=T, page_size=P, is_paged=True
        )
        self.assertIsNotNone(out_q4_paged_direct, "Paged Q4 prefill unexpectedly fell back")
        self.assertGreaterEqual(_cosine(out_q4_dense_direct, out_q4_paged_direct), 0.999999)

        # Also check dense Q4 S=64, T=128 causal continuation window (non-power-of-two T>=2*K edge)
        q_c64 = q[:1, :, :64].contiguous()
        k_c64 = k_dense[:1, :, :128].contiguous()
        v_c64 = v_dense[:1, :, :128].contiguous()
        sel_c64 = torch.full((1, 64, K), -1, dtype=torch.int32, device=dev)
        for s_i in range(64):
            comp_i = (64 + s_i + 1) // 4
            sel_c64[0, s_i, : comp_i * 4] = torch.arange(comp_i * 4, device=dev, dtype=torch.int32)
            for r_i in range((64 + s_i + 1) % 4):
                sel_c64[0, s_i, 2048 + r_i] = comp_i * 4 + r_i
        out_c64 = _try_run_q4_prefill(
            q_c64, k_c64, v_c64, None, None, sel_c64, T=128, page_size=4, is_paged=False
        )
        self.assertIsNotNone(out_c64)
        from rtp_llm.models_py.kernels.cuda.qsa.sparse_fmha import sparse_prefill_attn
        ref_c64 = sparse_prefill_attn(q_c64, k_c64, v_c64, sel_c64, validate_selected=False)
        self.assertGreaterEqual(_cosine(out_c64, ref_c64), 0.99999)

        out_dense_ov = qsa_sparse_prefill_attention(q, k_dense, v_dense, sel_ov)
        out_paged_ov = qsa_sparse_paged_prefill_attention(
            q, kv, bt, sel_ov, page_size=P, kv_heads=H_kv, head_dim=D, max_ctx_len=T
        )
        out_2d_fp8 = qsa_sparse_paged_prefill_attention(
            q,
            kv.reshape(pages, -1).to(torch.float8_e4m3fn),
            bt,
            sel_ov,
            page_size=P,
            kv_heads=H_kv,
            head_dim=D,
            max_ctx_len=T,
            kv_scale=torch.tensor([1.0, 1.0], dtype=torch.float32, device=dev),
        )
        # 0% overlap + duplicate indices (exercises Direct fallback)
        sel_zero = sel_ov.clone()
        for s in range(S):
            sel_zero[:, s, :2048] = (
                torch.randperm(n_q, device=dev, dtype=torch.int32)[:512].unsqueeze(-1) * 4
                + torch.arange(4, device=dev, dtype=torch.int32)
            ).reshape(-1).unsqueeze(0).expand(B, 2048)
        sel_zero[:, 0, 2050] = sel_zero[:, 0, 0]  # intentional duplicate
        out_dense_0 = qsa_sparse_prefill_attention(q, k_dense, v_dense, sel_zero)
        out_paged_0 = qsa_sparse_paged_prefill_attention(
            q, kv, bt, sel_zero, page_size=P, kv_heads=H_kv, head_dim=D, max_ctx_len=T
        )
        for bi, si in [(0, 0), (0, 3), (B - 1, S - 1)]:
            ref = _ref_at(bi, si, sel_ov)
            self.assertGreaterEqual(_cosine(out_dense_ov[bi, :, si], ref), 0.99999)
            self.assertGreaterEqual(_cosine(out_paged_ov[bi, :, si], ref), 0.99999)
            self.assertGreaterEqual(_cosine(out_2d_fp8[bi, :, si], ref), 0.999)
            ref0 = _ref_at(bi, si, sel_zero)
            self.assertGreaterEqual(_cosine(out_dense_0[bi, :, si], ref0), 0.99999)
            self.assertGreaterEqual(_cosine(out_paged_0[bi, :, si], ref0), 0.99999)

        # 3) Zero-copy Paged QSA Prefill end-to-end inside Qwen4_5RTPModel for B=1 & B=3 (t_pfx > 0)
        arch = _make_mini_arch(indexer_n_heads=8, indexer_budget=16, max_position_embeddings=512)
        m_p = Qwen4_5ForCausalLM(
            arch, use_triton_mxfp4=False, compute_mode="bf16", dtype=torch.bfloat16, device=dev
        ).eval()
        m_p.init_synthetic_weights(seed=808, ple_mode="gpu_full", ple_fp8=False)
        _stabilize_router_margins(m_p)
        m_p.prepare_weights()
        pools = {
            tag: 64
            for tag in [
                INDEXER_KV_TAG,
                INDEXER_STATE_TAG,
                FULL_ATTN_KV_TAG,
                "linear",
                "gdn_consumer_q_conv",
                "full_attn_kv_sconv",
                "full_attn_output_sconv",
                "linear_attn_output_sconv",
                "ffn_output_sconv",
                "ple_conv_state",
                "ple_ngram_ctx",
            ]
        }
        plan_p = build_qwen4_5_kv_cache_plan(arch)
        cache_p = LocalKVCache(plan_p, tokens_per_block=64, num_blocks_per_tag=pools, device=dev, dtype=torch.bfloat16)
        rt_p = Qwen4_5RTPModel.from_causal_lm(m_p, kv_cache=cache_p).eval()
        cache_ref = LocalKVCache(plan_p, tokens_per_block=64, num_blocks_per_tag=pools, device=dev, dtype=torch.bfloat16)
        rt_ref = Qwen4_5RTPModel.from_causal_lm(m_p, kv_cache=cache_ref).eval()
        tbl_p = build_block_tables_for_requests(cache_p, [192, 192])
        tbl_ref = build_block_tables_for_requests(cache_ref, [192, 193])
        ids_pfx = torch.randint(4, 240, (64,), dtype=torch.int64, device=dev)
        ids_suf = torch.randint(4, 240, (128,), dtype=torch.int64, device=dev)
        # Seed B=2 prefix in both uniform paged and non-uniform ragged reference caches
        for bi in range(2):
            for rt_i, tbl_i in ((rt_p, tbl_p), (rt_ref, tbl_ref)):
                tb1 = {tag: [rows[bi]] for tag, rows in tbl_i.items()}
                ai1 = make_tagged_attention_inputs(
                    tb1, is_prefill=True, input_lengths=[64], sequence_lengths=[64], prefix_lengths=[0], device=dev
                )
                rt_i.forward(LocalPyModelInputs(input_ids=ids_pfx, attention_inputs=ai1))
        ai_s2 = make_tagged_attention_inputs(
            tbl_p, is_prefill=True, input_lengths=[128, 128], sequence_lengths=[192, 192], prefix_lengths=[64, 64], device=dev
        )
        out_s2 = rt_p.forward(
            LocalPyModelInputs(input_ids=torch.cat([ids_suf, ids_suf]), attention_inputs=ai_s2)
        ).hidden_states
        ai_nh = make_tagged_attention_inputs(
            tbl_ref, is_prefill=True, input_lengths=[128, 129], sequence_lengths=[192, 193], prefix_lengths=[64, 64], device=dev
        )
        out_nh = rt_ref.forward(
            LocalPyModelInputs(input_ids=torch.cat([ids_suf, ids_suf, ids_suf[-1:]]), attention_inputs=ai_nh)
        ).hidden_states[:128]
        self.assertGreaterEqual(_cosine(out_s2[:128], out_nh), 0.999999)
        self.assertGreaterEqual(_cosine(out_s2[128:], out_nh), 0.999999)

    def test_r1_batched_opaque_tx_commit_and_rollback(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        dev = torch.device("cuda")
        from rtp_llm.models_py.model_desc.qwen4_5_runtime import (
            LocalLayerKVCache,
            _Qwen4_5OpaqueTxManager,
            _write_opaque_rows_raw_gpu,
        )

        torch.manual_seed(202613)
        arch = _make_mini_arch(indexer_n_heads=8, indexer_budget=16, max_position_embeddings=512)
        mgr = _Qwen4_5OpaqueTxManager(bridge=None, arch_cfg=arch)  # type: ignore[arg-type]
        B_cap, V, tpb = 4, 8, 64
        stage = mgr.acquire_stage(B_cap, V, is_cuda_graph=False)
        stage.begin_forward(
            torch.tensor([60, 125, 10, 63], dtype=torch.int32, device=dev),
            kv_cache=None,
            attention_inputs=None,
        )
        caches = []
        ref_bases = []
        for idx_e in range(17):
            num_blocks = 16
            width_b = 256 if idx_e % 3 == 0 else (4096 if idx_e % 3 == 1 else 16384)
            base = torch.randint(-100, 100, (num_blocks, width_b // 2), dtype=torch.int16, device=dev).to(torch.bfloat16)
            ref_bases.append(base.clone())
            lc = LocalLayerKVCache(kv_cache_base=base, seq_size_per_block=tpb, layer_id=idx_e, tag=f"tag_{idx_e}")
            bt = torch.tensor(
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]],
                dtype=torch.int32,
                device=dev,
            )
            st = stage.stage_tensor(
                f"tag_{idx_e}",
                idx_e,
                lc,
                bt,
                (width_b // 2,),
                torch.bfloat16,
            )
            st.copy_(torch.randint(-50, 50, st.shape, dtype=torch.int16, device=dev).to(torch.bfloat16))
            caches.append((lc, bt, st))
        stage.ready = True

        # Verify commit -> rollback bit-exact restoration across live_bs == 4 and live_bs == 2
        for live_bs, a_vals in [(4, [6, 5, 3, 1]), (2, [7, 2])]:
            accept = torch.tensor(a_vals, dtype=torch.int32, device=dev)
            mgr.prepare_commit(accept)
            self.assertIsNotNone(mgr._batched_undo_meta)
            mgr.finish_commit(True)

            # Check committed contents against reference _write_opaque_rows_raw_gpu
            for idx_e, (lc, bt, st) in enumerate(caches):
                expected = ref_bases[idx_e].clone()
                recs = mgr._undo_records[idx_e]
                _write_opaque_rows_raw_gpu(expected, recs.phys_ids, recs.commit_u8, lc.tag, idx_e)
                self.assertTrue(
                    torch.equal(lc.kv_cache_base, expected),
                    f"Commit mismatch entry={idx_e} live_bs={live_bs}",
                )

            # Rollback and assert exact original snapshot restored
            mgr.finish_commit(False)
            for idx_e, (lc, _, _) in enumerate(caches):
                self.assertTrue(
                    torch.equal(lc.kv_cache_base, ref_bases[idx_e]),
                    f"Rollback mismatch entry={idx_e} live_bs={live_bs}",
                )
            mgr._active_pending = stage

    def test_r2_qsa_fp8_kv_cache_and_lazy_gdn_verify(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        import dataclasses

        dev = torch.device("cuda")
        dt = torch.bfloat16
        from rtp_llm.models_py.kernels.cuda.qsa.sparse_paged_fmha import (
            sparse_paged_gqa_attn,
        )
        from rtp_llm.models_py.modules.qwen4_5.qsa_bridge import (
            _try_run_q4_prefill,
            qsa_paged_score,
            qsa_prefill_score,
            qsa_sparse_paged_decode_attention,
            qsa_sparse_paged_prefill_attention,
        )
        from rtp_llm.models_py.model_desc.qwen4_5_runtime import (
            _load_batched_full_attn_kv,
            _load_single_req_full_attn_kv,
            _store_verify_full_attn_kv_gpu,
        )

        torch.manual_seed(202621)
        with torch.no_grad():
            # 1) B=64, S=8 QSA FP8 paged decode (kt=32, nw=4, stg=2) + CUDA Graph parity
            B, S, H, H_kv, D, P, budget, ctx_len = 64, 8, 8, 1, 256, 64, 2048, 2560
            max_blks = ctx_len // P
            pages = B * max_blks + 8
            q_d = torch.randn((B, H, S, D), device=dev, dtype=dt)
            kv_fp8 = (torch.randn((pages, 2, H_kv, P, D), device=dev, dtype=dt) * 0.25).to(torch.float8_e4m3fn)
            sc_hnd = torch.rand((pages, 2, H_kv, P), device=dev, dtype=torch.float32) * 0.5 + 0.5
            bt_d = torch.arange(1, B * max_blks + 1, device=dev, dtype=torch.int32).view(B, max_blks)
            kv_lens_d = torch.full((B, S), ctx_len, device=dev, dtype=torch.int32)
            sel_d = torch.stack(
                [torch.randperm(ctx_len, device=dev, dtype=torch.int32)[:budget].sort().values for _ in range(B * S)]
            ).view(B, S, budget)
            out_d = qsa_sparse_paged_decode_attention(
                q_d, kv_fp8, bt_d, kv_lens_d, sel_d, page_size=P, kv_heads=H_kv, head_dim=D, kv_scale=sc_hnd, is_cuda_graph=True
            )
            ref_d = sparse_paged_gqa_attn(
                q_d[:2], kv_fp8, bt_d[:2], kv_lens_d[:2], sel_d[:2], page_size=P, kv_head_num=H_kv, kv_scale=sc_hnd, validate_selected=False
            )
            self.assertGreaterEqual(_cosine(out_d[:2], ref_d), 0.999)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                cg_d = qsa_sparse_paged_decode_attention(
                    q_d, kv_fp8, bt_d, kv_lens_d, sel_d, page_size=P, kv_heads=H_kv, head_dim=D, kv_scale=sc_hnd, is_cuda_graph=True
                )
            q_d.copy_(torch.randn_like(q_d))
            g.replay()
            eag_d = qsa_sparse_paged_decode_attention(
                q_d, kv_fp8, bt_d, kv_lens_d, sel_d, page_size=P, kv_heads=H_kv, head_dim=D, kv_scale=sc_hnd
            )
            self.assertGreaterEqual(_cosine(cg_d[:2], eag_d[:2]), 0.999999)

            # 2) QSA FP8 prefill (Q4 & Direct) + qsa_paged_score (S=128 -> qs=32) + qsa_prefill_score (S=256)
            Bp, Tp, Hp_kv, Dp, Pp = 2, 256, 1, 256, 64
            pages_p = Bp * (Tp // Pp) + 4
            kvp_fp8 = (torch.randn((pages_p, 2, Hp_kv, Pp, Dp), device=dev, dtype=dt) * 0.25).to(torch.float8_e4m3fn)
            scp = torch.rand((pages_p, 2, Hp_kv, Pp), device=dev, dtype=torch.float32) * 0.5 + 0.5
            btp = torch.arange(1, Bp * (Tp // Pp) + 1, device=dev, dtype=torch.int32).view(Bp, Tp // Pp)
            base_sel = torch.arange(0, 64, device=dev, dtype=torch.int32).view(1, 1, 64).expand(Bp, Tp // 4, 64).repeat_interleave(4, dim=1).clone()

            def _ref_paged_prefill(q_in: torch.Tensor, sel_in: torch.Tensor, b_i: int, s_i: int) -> torch.Tensor:
                k_d = (kvp_fp8[btp[b_i].long(), 0].float() * scp[btp[b_i].long(), 0].unsqueeze(-1)).transpose(0, 1).reshape(Hp_kv, Tp, Dp)
                v_d = (kvp_fp8[btp[b_i].long(), 1].float() * scp[btp[b_i].long(), 1].unsqueeze(-1)).transpose(0, 1).reshape(Hp_kv, Tp, Dp)
                idx = sel_in[b_i, s_i]
                idx = idx[(idx >= 0) & (idx < Tp)].long()
                logits = (q_in[b_i, :, s_i, :].float() @ k_d[0, idx].T) * (Dp ** -0.5)
                return (torch.softmax(logits, dim=-1) @ v_d[0, idx]).to(dt)

            qp_q4 = torch.randn((Bp, 8, Tp, Dp), device=dev, dtype=dt)
            q4_direct = _try_run_q4_prefill(qp_q4, kvp_fp8, kvp_fp8, scp, btp, base_sel, T=Tp, page_size=Pp, is_paged=True)
            self.assertIsNotNone(q4_direct)
            out_q4 = qsa_sparse_paged_prefill_attention(qp_q4, kvp_fp8, btp, base_sel, page_size=Pp, kv_heads=Hp_kv, head_dim=Dp, max_ctx_len=Tp, kv_scale=scp)
            for s_chk in (0, 63, 129, 255):
                self.assertGreaterEqual(_cosine(out_q4[0, :, s_chk], _ref_paged_prefill(qp_q4, base_sel, 0, s_chk)), 0.999)

            qp_dir = torch.randn((Bp, 16, Tp, Dp), device=dev, dtype=dt)  # q_per_kv=16 -> Direct fast kernel
            self.assertIsNone(_try_run_q4_prefill(qp_dir, kvp_fp8, kvp_fp8, scp, btp, base_sel, T=Tp, page_size=Pp, is_paged=True))
            out_dir = qsa_sparse_paged_prefill_attention(qp_dir, kvp_fp8, btp, base_sel, page_size=Pp, kv_heads=Hp_kv, head_dim=Dp, max_ctx_len=Tp, kv_scale=scp)
            for s_chk in (1, 64, 199, 254):
                self.assertGreaterEqual(_cosine(out_dir[0, :, s_chk], _ref_paged_prefill(qp_dir, base_sel, 0, s_chk)), 0.999)

            b_sc, s_sc, ent_blk, max_ctx_sc = 8, 128, 16, 256
            blks_per_req = max_ctx_sc // ent_blk
            q_sc = torch.randn((b_sc, s_sc, 8, 128), device=dev, dtype=dt)
            k_pool_sc = (torch.randn((b_sc * blks_per_req + 4, ent_blk, 128), device=dev, dtype=dt) * 0.5).to(torch.float8_e4m3fn)
            bt_sc = torch.arange(1, b_sc * blks_per_req + 1, device=dev, dtype=torch.int32).view(b_sc, blks_per_req)
            clens = torch.full((b_sc, s_sc), max_ctx_sc, device=dev, dtype=torch.int32)
            clens[0, 0] = 199
            out_psc = qsa_paged_score(q_sc, k_pool_sc, bt_sc, clens, max_ctx_len=max_ctx_sc).view(b_sc, s_sc, max_ctx_sc)
            self.assertTrue(torch.isinf(out_psc[0, 0, 199:]).all())
            ref_k0 = k_pool_sc.index_select(0, bt_sc[0].long()).reshape(max_ctx_sc, 128).float()
            ref_s0 = torch.relu(torch.einsum("qhd,kd->qhk", q_sc[0].float(), ref_k0) / math.sqrt(128.0)).sum(dim=1)
            self.assertGreaterEqual(_cosine(out_psc[0, 0, :199], ref_s0[0, :199]), 0.999)
            self.assertGreaterEqual(_cosine(out_psc[0, 1:, :], ref_s0[1:, :]), 0.999)

            k_flat = torch.randn((2, 256, 128), device=dev, dtype=dt)
            q_pre_sc = torch.randn((2, 256, 8, 128), device=dev, dtype=dt)
            pref_sc = qsa_prefill_score(q_pre_sc, k_flat)
            ref_pref_sc = torch.relu(torch.einsum("bqhd,bkd->bqhk", q_pre_sc.float(), k_flat.float()) / math.sqrt(128.0)).sum(dim=2)
            self.assertGreaterEqual(_cosine(pref_sc, ref_pref_sc), 0.999)

            # 3) LocalKVCache(kv_dtype=float8_e4m3fn, allocate_fp8_kv_scale=True) write/dequant roundtrip + >448 clamp check
            base_arch = _make_mini_arch(indexer_n_heads=8, indexer_budget=16, max_position_embeddings=512)
            plan = build_qwen4_5_kv_cache_plan(base_arch)
            pools = {t: 64 for t in ["indexer_kv", "indexer_state", FULL_ATTN_KV_TAG, "linear", "gdn_consumer_q_conv", "full_attn_kv_sconv", "full_attn_output_sconv", "linear_attn_output_sconv", "ffn_output_sconv", "ple_conv_state", "ple_ngram_ctx"]}
            fp8_cache = LocalKVCache(plan, tokens_per_block=64, num_blocks_per_tag=pools, device=dev, dtype=dt, kv_dtype=torch.float8_e4m3fn, allocate_fp8_kv_scale=True)
            kv_l3 = fp8_cache.get_layer_cache(3, FULL_ATTN_KV_TAG)
            self.assertEqual(kv_l3.kv_cache_base.dtype, torch.float8_e4m3fn)
            self.assertIsNotNone(kv_l3.kv_scale_base)
            self.assertEqual(tuple(kv_l3.kv_scale_base.shape), (64, 2, base_arch.num_key_value_heads, 64))
            kv_l3.kv_scale_base.zero_()
            bt_t = torch.tensor([[1, 2], [3, 4]], device=dev, dtype=torch.int32)
            k_in = torch.randn((2, 6, base_arch.num_key_value_heads, base_arch.head_dim), device=dev, dtype=dt)
            v_in = torch.randn((2, 6, base_arch.num_key_value_heads, base_arch.head_dim), device=dev, dtype=dt)
            k_in[0, 0, 0, 0] = 600.0
            v_in[0, 0, 0, 0] = -600.0
            _store_verify_full_attn_kv_gpu(kv_l3, bt_t, torch.tensor([62, 63], device=dev, dtype=torch.int32), k_in, v_in)
            self.assertFalse(torch.isnan(kv_l3.kv_cache_base.float()).any().item())
            k_raw, v_raw = _load_single_req_full_attn_kv(
                kv_l3.kv_cache_base, bt_t[0], 68, 64, num_kv_heads=base_arch.num_key_value_heads, head_dim=base_arch.head_dim, kv_scale=None, out_dtype=torch.float32
            )
            self.assertAlmostEqual(float(k_raw[62, 0, 0].item()), 448.0, places=1)
            self.assertAlmostEqual(float(v_raw[62, 0, 0].item()), -448.0, places=1)
            k_out_b, v_out_b = _load_batched_full_attn_kv(
                kv_l3.kv_cache_base, bt_t, 2, 68, 64, num_kv_heads=base_arch.num_key_value_heads, head_dim=base_arch.head_dim, kv_scale=kv_l3.kv_scale_base, out_dtype=dt
            )
            self.assertEqual(k_out_b.dtype, dt)
            kv_l3.kv_scale_base[2, :, :, :4].fill_(2.5)
            k_sc, _ = _load_single_req_full_attn_kv(
                kv_l3.kv_cache_base, bt_t[0], 68, 64, num_kv_heads=base_arch.num_key_value_heads, head_dim=base_arch.head_dim, kv_scale=kv_l3.kv_scale_base, out_dtype=dt
            )
            self.assertGreaterEqual(_cosine(k_sc[64:68], k_in[0, 2:6] * 2.5), 0.995)

            # 4) End-to-end FP8 KV + Lazy GDN + TxV2 verify/rollback/commit (both YoCo=False & True)
            for yoco_flag in (False, True):
                torch.manual_seed(2140 + int(yoco_flag))
                arch_y = dataclasses.replace(base_arch, enable_yoco_decode=yoco_flag)
                plan_y = build_qwen4_5_kv_cache_plan(arch_y)
                lm = Qwen4_5ForCausalLM(arch_y, use_triton_mxfp4=True, compute_mode="bf16", dtype=dt, device=dev)
                lm.init_synthetic_weights(seed=2040 + int(yoco_flag), ple_mode="gpu_full", ple_fp8=False)
                _stabilize_router_margins(lm)
                lm.prepare_weights()
                c_e2e = LocalKVCache(plan_y, tokens_per_block=64, num_blocks_per_tag=pools, device=dev, dtype=dt, kv_dtype=torch.float8_e4m3fn, allocate_fp8_kv_scale=True)
                rt = Qwen4_5RTPModel.from_causal_lm(lm, kv_cache=c_e2e)
                tbls = build_block_tables_for_requests(c_e2e, [80, 80])
                p0 = torch.randint(4, 240, (62,), device=dev, dtype=torch.int64)
                p1 = torch.randint(4, 240, (63,), device=dev, dtype=torch.int64)
                vt = torch.randint(4, 240, (2, 8), device=dev, dtype=torch.int64)
                nt = torch.randint(4, 240, (2,), device=dev, dtype=torch.int64)
                rt.forward(LocalPyModelInputs(input_ids=torch.cat([p0, p1]), attention_inputs=make_tagged_attention_inputs(tbls, is_prefill=True, input_lengths=[62, 63], sequence_lengths=[62, 63], prefix_lengths=[0, 0], device=dev)))
                snap = {
                    (k1, k2): (
                        lc.kv_cache_base.clone(),
                        lc.kv_scale_base.clone() if getattr(lc, "kv_scale_base", None) is not None else None,
                    )
                    for k1, tm in enumerate(c_e2e.layer_caches)
                    for k2, lc in tm.items()
                }
                va = make_tagged_attention_inputs(tbls, is_prefill=True, input_lengths=[8, 8], sequence_lengths=[], prefix_lengths=[62, 63], device=dev)
                for grp in va.values(): grp.is_target_verify = True
                _ = rt.forward(LocalPyModelInputs(input_ids=vt.reshape(-1), attention_inputs=va))
                acc = torch.tensor([5, 3], dtype=torch.int32, device=dev)
                rt.prepare_speculative_target_commit(acc)
                rt.finish_speculative_target_commit(False)
                rt.finalize_speculative_target_commit()
                for (k1, k2), (buf, _) in snap.items():
                    if k2 not in ("full", FULL_ATTN_KV_TAG, INDEXER_KV_TAG):
                        self.assertTrue(torch.equal(c_e2e.layer_caches[k1][k2].kv_cache_base, buf), f"Rollback mismatch yoco={yoco_flag} L{k1} {k2}")
                for (k1, k2), (buf, sc) in snap.items():
                    c_e2e.layer_caches[k1][k2].kv_cache_base.copy_(buf)
                    if sc is not None:
                        c_e2e.layer_caches[k1][k2].kv_scale_base.copy_(sc)
                _ = rt.forward(LocalPyModelInputs(input_ids=vt.reshape(-1), attention_inputs=va))
                rt.prepare_speculative_target_commit(acc)
                rt.finish_speculative_target_commit(True)
                rt.finalize_speculative_target_commit()
                c_ver = rt.forward(LocalPyModelInputs(input_ids=nt, attention_inputs=make_tagged_attention_inputs(tbls, is_prefill=False, input_lengths=[1, 1], sequence_lengths=[67, 66], prefix_lengths=[0, 0], device=dev)))
                for (k1, k2), (buf, sc) in snap.items():
                    c_e2e.layer_caches[k1][k2].kv_cache_base.copy_(buf)
                    if sc is not None:
                        c_e2e.layer_caches[k1][k2].kv_scale_base.copy_(sc)
                tb0 = {tag: [rows[0]] for tag, rows in tbls.items()}
                tb1 = {tag: [rows[1]] for tag, rows in tbls.items()}
                for s in range(5):
                    rt.forward(LocalPyModelInputs(input_ids=vt[0:1, s].contiguous(), attention_inputs=make_tagged_attention_inputs(tb0, is_prefill=False, input_lengths=[1], sequence_lengths=[62+s], prefix_lengths=[0], device=dev)))
                for s in range(3):
                    rt.forward(LocalPyModelInputs(input_ids=vt[1:2, s].contiguous(), attention_inputs=make_tagged_attention_inputs(tb1, is_prefill=False, input_lengths=[1], sequence_lengths=[63+s], prefix_lengths=[0], device=dev)))
                cont_seq = rt.forward(LocalPyModelInputs(input_ids=nt, attention_inputs=make_tagged_attention_inputs(tbls, is_prefill=False, input_lengths=[1, 1], sequence_lengths=[67, 66], prefix_lengths=[0, 0], device=dev)))
                self.assertGreaterEqual(_cosine(c_ver.hidden_states, cont_seq.hidden_states), 0.999)


if __name__ == "__main__":
    unittest.main()

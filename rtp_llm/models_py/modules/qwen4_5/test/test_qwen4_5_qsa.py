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
    build_qsa_neox_cos_sin,
    qsa_paged_score,
    qsa_prefill_score,
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


if __name__ == "__main__":
    unittest.main()

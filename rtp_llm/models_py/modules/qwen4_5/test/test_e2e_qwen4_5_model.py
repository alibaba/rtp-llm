"""End-to-end integration, numerical accuracy, and multi-GPU EP=4 tests for Qwen 4.5-Flash."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from rtp_llm.models.qwen4_5.qwen4_5 import Qwen4_5
from rtp_llm.models_py.model_desc.qwen4_5 import (
    Qwen4_5ForCausalLM,
    Qwen4_5Model,
    Qwen4_5ProfileRecorder,
    Qwen4_5RTPModel,
)
from rtp_llm.models_py.modules.qwen4_5.contracts import Qwen4_5ArchConfig


def make_mini_qwen4_5_config() -> Qwen4_5ArchConfig:
    """8-layer mini config exercising all 6 Qwen 4.5-Flash layer roles:
      L0: GDN + Dense MLP + HC is_head
      L1: GDN + PLE (FP8 8 shards) + Latent MoE (MXFP4)
      L2: GDN + Latent MoE
      L3: FullAttn Producer (MQA-1, d=256, partial RoPE 0.25, k/v_sconv) + Latent MoE
      L4: GDN YoCo Source + Latent MoE
      L5: GDN YoCo Consumer (from L4) + Latent MoE
      L6: GDN YoCo Consumer (from L4) + Latent MoE
      L7: FullAttn YoCo Consumer (from L3) + Latent MoE + HC emits_latent=False
      + Final hyper_connection_mixer + untied lm_head (no model.norm).
    """
    return Qwen4_5ArchConfig(
        vocab_size=2048,
        hidden_size=512,
        num_hidden_layers=8,
        intermediate_size=1024,
        first_k_dense_replace=1,
        rms_norm_eps=1e-6,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        max_position_embeddings=256,
        full_attention_interval=4,
        num_attention_heads=8,
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
        num_experts=32,
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
        ple_embed_dim=512,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=4,
        ngram_vocab_size_base=512,
        make_ngram_vocab_size_divisible_by=64,
        split_ngram_parts=8,
        ple_quant_scheme="fp8_pertensor",
        share_map="4->8,5->6,5->7",
        enable_yoco_decode=True,
        quant_method="mxfp4",
    )


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1)).item()
    )


def _stabilize_router_margins(model: Qwen4_5ForCausalLM) -> None:
    """Add deterministic per-layer expert_bias margins spanning all 4 EP shards.

    Prevents 1-ULP BF16 differences on near-tied isotropic random router logits
    from flipping discrete top-k indices during multi-layer numerical comparison,
    while exercising diverse experts across every EP shard.
    """
    for layer in model.model.layers:
        if layer.is_moe and layer.mlp.expert_bias is not None:
            layer.mlp.expert_bias.zero_()
            n_exp = layer.mlp.num_experts
            shard = n_exp // 4
            for r in range(4):
                e_id = r * shard + (layer.layer_idx % shard)
                layer.mlp.expert_bias[e_id] = 4.0


class TestQwen4_5E2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required for Qwen 4.5-Flash B300 E2E tests")
        cls.device = torch.device("cuda:0")
        cls.cfg = make_mini_qwen4_5_config()

    def test_01_registration_and_runtime_state_plan(self) -> None:
        # Verify factory registration strings & _py_model_class wiring
        qwen_cls = Qwen4_5.__new__(Qwen4_5)
        self.assertIs(qwen_cls._py_model_class(), Qwen4_5RTPModel)

        model = Qwen4_5ForCausalLM(self.cfg, dtype=torch.bfloat16, device=self.device)
        # Verify architectural roles
        self.assertTrue(model.model.layers[0].attn_hyper_connection.is_head)
        self.assertFalse(model.model.layers[0].is_moe)
        self.assertIsNotNone(model.model.layers[1].ple)
        self.assertTrue(model.model.layers[1].is_moe)
        self.assertTrue(model.model.layers[3].is_source)
        self.assertTrue(model.model.layers[4].is_source)
        self.assertTrue(model.model.layers[5].is_consumer)
        self.assertTrue(model.model.layers[6].is_consumer)
        self.assertTrue(model.model.layers[7].is_consumer)
        self.assertFalse(model.model.layers[7].mlp_hyper_connection.emits_latent)

        # Verify minimal YoCo state allocation
        state_yoco = model.create_runtime_state(batch_size=2, max_seq_len=64, enable_yoco_decode=True)
        self.assertEqual(sorted(state_yoco.full_attn_kv.keys()), [3])
        self.assertEqual(sorted(state_yoco.k_sconv.keys()), [3])
        self.assertEqual(sorted(state_yoco.v_sconv.keys()), [3])
        self.assertEqual(sorted(state_yoco.ple.keys()), [1])
        self.assertEqual(len(state_yoco.gdn), 6)
        self.assertIsNotNone(state_yoco.gdn[4].ssm_state)
        self.assertIsNone(state_yoco.gdn[5].ssm_state)
        self.assertIsNone(state_yoco.gdn[6].ssm_state)

        state_noyoco = model.create_runtime_state(batch_size=2, max_seq_len=64, enable_yoco_decode=False)
        self.assertIsNotNone(state_noyoco.gdn[5].ssm_state)
        self.assertIsNotNone(state_noyoco.gdn[6].ssm_state)

        # Verify real 1109 reference_config.json loads if available
        ref_cfg_path = Path("/data0/luohaocheng.lhc/artifacts/qwen45-flash-20261123/reference_config.json")
        if ref_cfg_path.exists():
            mcfg = Qwen4_5.create_config(ref_cfg_path)
            self.assertEqual(mcfg.num_layers, 40)
            self.assertEqual(mcfg.hidden_size, 5120)

    @torch.no_grad()
    def test_02_triton_vs_reference_full_model(self) -> None:
        model = Qwen4_5ForCausalLM(self.cfg, dtype=torch.bfloat16, device=self.device)
        model.init_synthetic_weights(seed=123, ple_mode="gpu_full")
        _stabilize_router_margins(model)
        model.prepare_weights()

        g = torch.Generator(device=self.device).manual_seed(999)
        input_ids = torch.randint(10, self.cfg.vocab_size, (2, 24), device=self.device, generator=g)

        st_tri = model.create_runtime_state(batch_size=2, max_seq_len=64)
        st_ref = model.create_runtime_state(batch_size=2, max_seq_len=64)

        logits_tri = model(input_ids, state=st_tri, use_reference=False)
        logits_ref = model(input_ids, state=st_ref, use_reference=True)

        self.assertTrue(torch.isfinite(logits_tri).all().item())
        self.assertTrue(torch.isfinite(logits_ref).all().item())
        max_abs = float((logits_tri.float() - logits_ref.float()).abs().max().item())
        cos = _cos_sim(logits_tri, logits_ref)
        print(f"[E2E Triton vs Ref] max_abs={max_abs:.6f}, cos={cos:.8f}")
        self.assertGreater(cos, 0.999)
        self.assertLess(max_abs, 0.15)

        # Also check logits_to_keep=1 matches last token slice
        st_last = model.create_runtime_state(batch_size=2, max_seq_len=64)
        logits_last = model(input_ids, state=st_last, logits_to_keep=1)
        self.assertEqual(tuple(logits_last.shape), (2, 1, self.cfg.vocab_size))
        self.assertEqual(
            float((logits_last - logits_tri[:, -1:, :]).abs().max().item()),
            0.0,
        )

    @torch.no_grad()
    def test_03_prefill_vs_incremental_decode_and_yoco(self) -> None:
        model = Qwen4_5ForCausalLM(self.cfg, dtype=torch.bfloat16, device=self.device)
        model.init_synthetic_weights(seed=321, ple_mode="gpu_full")
        _stabilize_router_margins(model)
        model.prepare_weights()

        B, T_prompt, T_gen = 2, 20, 8
        T_total = T_prompt + T_gen
        g = torch.Generator(device=self.device).manual_seed(777)
        tokens = torch.randint(10, self.cfg.vocab_size, (B, T_total), device=self.device, generator=g)

        # Full single-shot Prefill(28)
        st_full = model.create_runtime_state(B, max_seq_len=64, enable_yoco_decode=True)
        logits_full = model(tokens, state=st_full, enable_yoco_decode=True)

        # Prefill(20) + 8x Decode(1) with YoCo Decode ON
        st_yoco = model.create_runtime_state(B, max_seq_len=64, enable_yoco_decode=True)
        out_p_yoco = model(tokens[:, :T_prompt], state=st_yoco, enable_yoco_decode=True)
        step_logits_yoco = [out_p_yoco]
        for t in range(T_prompt, T_total):
            kv_len_before = st_yoco.full_attn_kv[3].seq_len
            step_out = model(tokens[:, t : t + 1], state=st_yoco, enable_yoco_decode=True)
            # Layer 3 producer advanced KV by 1; Layer 7 consumer did NOT advance KV
            self.assertEqual(st_yoco.full_attn_kv[3].seq_len, kv_len_before + 1)
            step_logits_yoco.append(step_out)
        logits_inc_yoco = torch.cat(step_logits_yoco, dim=1)

        # Prefill(20) + 8x Decode(1) with YoCo Decode OFF
        st_noyoco = model.create_runtime_state(B, max_seq_len=64, enable_yoco_decode=False)
        out_p_ny = model(tokens[:, :T_prompt], state=st_noyoco, enable_yoco_decode=False)
        step_logits_ny = [out_p_ny]
        for t in range(T_prompt, T_total):
            step_out = model(tokens[:, t : t + 1], state=st_noyoco, enable_yoco_decode=False)
            step_logits_ny.append(step_out)
        logits_inc_ny = torch.cat(step_logits_ny, dim=1)

        max_abs_yoco = float((logits_full.float() - logits_inc_yoco.float()).abs().max().item())
        cos_yoco = _cos_sim(logits_full, logits_inc_yoco)
        max_abs_ny = float((logits_full.float() - logits_inc_ny.float()).abs().max().item())
        cos_ny = _cos_sim(logits_full, logits_inc_ny)
        max_abs_yoco_vs_ny = float(
            (logits_inc_yoco.float() - logits_inc_ny.float()).abs().max().item()
        )
        cos_yoco_vs_ny = _cos_sim(logits_inc_yoco, logits_inc_ny)

        print(
            f"[E2E Prefill vs Decode] YoCo: max_abs={max_abs_yoco:.6f}, cos={cos_yoco:.8f} | "
            f"NoYoCo: max_abs={max_abs_ny:.6f}, cos={cos_ny:.8f} | "
            f"YoCo vs NoYoCo: max_abs={max_abs_yoco_vs_ny:.6f}, cos={cos_yoco_vs_ny:.8f}"
        )
        self.assertGreater(cos_yoco, 0.999)
        self.assertGreater(cos_ny, 0.999)
        self.assertGreater(cos_yoco_vs_ny, 0.9995)
        self.assertLess(max_abs_yoco, 0.2)
        self.assertLess(max_abs_yoco_vs_ny, 0.1)

    @torch.no_grad()
    def test_04_profiler_recorder_smoke(self) -> None:
        model = Qwen4_5ForCausalLM(self.cfg, dtype=torch.bfloat16, device=self.device)
        model.init_synthetic_weights(seed=555, ple_mode="gpu_full")
        model.prepare_weights()
        st = model.create_runtime_state(batch_size=2, max_seq_len=64)
        tokens = torch.randint(10, self.cfg.vocab_size, (2, 16), device=self.device)
        prof = Qwen4_5ProfileRecorder(enabled=True)
        _ = model(tokens, state=st, profiler=prof)
        summary = prof.summarize_ms()
        for key in (
            "embed",
            "hc",
            "ple",
            "gdn_normal",
            "gdn_consumer",
            "full_attn_producer",
            "full_attn_consumer",
            "short_conv",
            "dense_mlp",
            "latent_moe",
            "final_hc",
            "lm_head",
        ):
            self.assertIn(key, summary, f"Missing profiler category {key}")
            self.assertGreater(summary[key], 0.0)



    @torch.no_grad()
    def test_05_tp4_ep4_checkpoint_reload_and_module_parity(self) -> None:
        """Verify TP=4 checkpoint streaming, bind_qwen4_5_model_weights, GDN/QSA/DenseMLP/LMHead parity."""
        import tempfile
        from dataclasses import replace
        from rtp_llm.models_py.modules.qwen4_5.checkpoint_loader import (
            load_qwen4_5_from_checkpoint,
            save_qwen4_5_safetensors_checkpoint,
        )

        # Exercise QSA sparse (8Q->2Q/rank, 1KV+8-head indexer replicated) + GDN + DenseMLP + LatentMoE + LMHead
        cfg_qsa = replace(
            self.cfg,
            indexer_n_heads=8,
            indexer_kv_heads=1,
            indexer_head_dim=64,
            indexer_compress_ratio=4,
            indexer_budget=16,
        )
        ref = Qwen4_5ForCausalLM(cfg_qsa, dtype=torch.bfloat16, device=self.device)
        ref.init_synthetic_weights(seed=808, ple_mode="gpu_full")
        _stabilize_router_margins(ref)
        ref.prepare_weights()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_qwen4_5_safetensors_checkpoint(ref, tmpdir)
            tp_ranks = []
            for r in range(4):
                m_r = Qwen4_5ForCausalLM(
                    cfg_qsa,
                    dtype=torch.bfloat16,
                    device=self.device,
                    ep_size=4,
                    ep_rank=r,
                    tp_size=4,
                    tp_rank=r,
                )
                load_qwen4_5_from_checkpoint(m_r, tmpdir, ple_mode="gpu_sharded", num_workers=2)
                m_r.prepare_weights()
                tp_ranks.append(m_r)

        # Verify sharded shapes on rank 0 vs TP=1
        r0 = tp_ranks[0]
        self.assertEqual(r0.lm_head.weight.shape[0], cfg_qsa.vocab_size // 4)
        self.assertEqual(r0.model.layers[0].mlp.local_intermediate_size, cfg_qsa.intermediate_size // 4)
        self.assertEqual(r0.model.layers[0].linear_attn.num_k_heads, 1)
        self.assertEqual(r0.model.layers[0].linear_attn.num_v_heads, 4)
        self.assertEqual(r0.model.layers[0].linear_attn.conv1d.weight.shape[0], 384)
        self.assertEqual(r0.model.layers[5].linear_attn.conv1d.weight.shape[0], 64)
        self.assertEqual(r0.model.layers[3].self_attn.num_heads, 2)
        self.assertEqual(r0.model.layers[3].self_attn.num_kv_heads, 1)
        self.assertTrue(cfg_qsa.has_qsa)
        self.assertIsNotNone(r0.model.layers[3].self_attn.indexer)

        # Also exercise bind_qwen4_5_model_weights from unsharded ModelWeights -> TP=4
        from rtp_llm.model_loader.model_weight_info import ModelWeights
        from rtp_llm.models.qwen4_5.qwen4_5_weight import QWEN4_5_W
        from rtp_llm.models_py.modules.qwen4_5.checkpoint_loader import bind_qwen4_5_model_weights
        mw_unsharded = ModelWeights(cfg_qsa.num_hidden_layers, device=self.device, dtype=torch.bfloat16)
        mw_unsharded.set_global_weight(QWEN4_5_W.embedding, ref.model.embed_tokens.weight.detach())
        mw_unsharded.set_global_weight(QWEN4_5_W.lm_head, ref.lm_head.weight.detach())
        mx = ref.model.hyper_connection_mixer
        mw_unsharded.set_global_weight(QWEN4_5_W.hc_mixer_norm_w, mx.hc_norm_weight.detach())
        mw_unsharded.set_global_weight(QWEN4_5_W.hc_mixer_norm_b, mx.hc_norm_bias.detach())
        mw_unsharded.set_global_weight(QWEN4_5_W.hc_mixer_mix_down, mx.input_mix_weight_down.detach())
        mw_unsharded.set_global_weight(QWEN4_5_W.hc_mixer_mix_up, mx.input_mix_weight_up.detach())
        for lid, lyr in enumerate(ref.model.layers):
            mw_unsharded.set_layer_weight(lid, QWEN4_5_W.hc_attn_norm_w, lyr.attn_hyper_connection.hc_norm_weight.detach())
            mw_unsharded.set_layer_weight(lid, QWEN4_5_W.hc_attn_norm_b, lyr.attn_hyper_connection.hc_norm_bias.detach())
            mw_unsharded.set_layer_weight(lid, QWEN4_5_W.hc_attn_inject, lyr.attn_hyper_connection.block_inject_weight.detach())
            mw_unsharded.set_layer_weight(lid, QWEN4_5_W.hc_attn_mix_down, lyr.attn_hyper_connection.input_mix_weight_down.detach())
            mw_unsharded.set_layer_weight(lid, QWEN4_5_W.hc_attn_mix_up, lyr.attn_hyper_connection.input_mix_weight_up.detach())
            mw_unsharded.set_layer_weight(lid, QWEN4_5_W.hc_mlp_norm_w, lyr.mlp_hyper_connection.hc_norm_weight.detach())
            mw_unsharded.set_layer_weight(lid, QWEN4_5_W.hc_mlp_norm_b, lyr.mlp_hyper_connection.hc_norm_bias.detach())
            mw_unsharded.set_layer_weight(lid, QWEN4_5_W.hc_mlp_inject, lyr.mlp_hyper_connection.block_inject_weight.detach())
            if lyr.mlp_hyper_connection.input_mix_weight_down is not None:
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.hc_mlp_mix_down, lyr.mlp_hyper_connection.input_mix_weight_down.detach())
            mw_unsharded.set_layer_weight(lid, QWEN4_5_W.hc_mlp_mix_up, lyr.mlp_hyper_connection.input_mix_weight_up.detach())
            if lyr.ple is not None:
                ple = lyr.ple
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.ple_key_proj_w, ple.key_proj.weight.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.ple_value_proj_w, ple.value_proj.weight.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.ple_conv_w, ple.conv1d.weight.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.ple_norm_key_w, ple.norm_key.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.ple_norm_query_w, ple.norm_query.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.ple_norm_conv_w, ple.norm_conv.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.ple_multipliers, ple.layer_multipliers.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.ple_ngram_offsets, ple.ngram_heads_offsets.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.ple_ngram_vocab_sizes, ple.ngram_heads_vocab_sizes.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.ple_ngram_shards, ple._ngram_shards[2:4])
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.ple_ngram_scale, ple.weight_scale.detach())
            if lyr.linear_attn is not None:
                la = lyr.linear_attn
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.linear_in_proj_qkvz, la.in_proj_qkvz.weight.detach())
                if not lyr.is_consumer:
                    mw_unsharded.set_layer_weight(lid, QWEN4_5_W.linear_in_proj_ba, la.in_proj_ba.weight.detach())
                    mw_unsharded.set_layer_weight(lid, QWEN4_5_W.linear_a_log, la.A_log.detach())
                    mw_unsharded.set_layer_weight(lid, QWEN4_5_W.linear_dt_bias, la.dt_bias.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.linear_conv1d, la.conv1d.weight.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.linear_norm_w, la.norm.weight.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.linear_out_proj, la.out_proj.weight.detach())
                if lyr.attn_output_sconv is not None:
                    mw_unsharded.set_layer_weight(lid, QWEN4_5_W.linear_attn_output_sconv, lyr.attn_output_sconv.weight.detach())
            else:
                sa = lyr.self_attn
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.full_q_proj, sa.q_proj.weight.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.full_q_norm_w, sa.q_norm.weight.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.full_o_proj, sa.o_proj.weight.detach())
                if not lyr.is_consumer:
                    mw_unsharded.set_layer_weight(lid, QWEN4_5_W.full_k_proj, sa.k_proj.weight.detach())
                    mw_unsharded.set_layer_weight(lid, QWEN4_5_W.full_v_proj, sa.v_proj.weight.detach())
                    mw_unsharded.set_layer_weight(lid, QWEN4_5_W.full_k_norm_w, sa.k_norm.weight.detach())
                    mw_unsharded.set_layer_weight(lid, QWEN4_5_W.full_k_sconv, sa.k_sconv.weight.detach())
                    mw_unsharded.set_layer_weight(lid, QWEN4_5_W.full_v_sconv, sa.v_sconv.weight.detach())
                if lyr.attn_output_sconv is not None:
                    mw_unsharded.set_layer_weight(lid, QWEN4_5_W.full_attn_output_sconv, lyr.attn_output_sconv.weight.detach())
                if sa.indexer is not None:
                    mw_unsharded.set_layer_weight(lid, QWEN4_5_W.full_indexer_qk_proj, sa.indexer.index_qk_proj.weight.detach())
                    mw_unsharded.set_layer_weight(lid, QWEN4_5_W.full_indexer_q_norm_w, sa.indexer.q_layernorm.weight.detach())
                    mw_unsharded.set_layer_weight(lid, QWEN4_5_W.full_indexer_k_norm_w, sa.indexer.k_layernorm.weight.detach())
            if lyr.is_moe:
                moe = lyr.mlp
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.moe_latent_down_proj, moe.latent_down_proj.weight.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.moe_latent_gate_proj, moe.latent_gate_proj.weight.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.moe_latent_up_proj, moe.latent_up_proj.weight.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.moe_latent_scale_gate, moe.latent_scale_gate.weight.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.moe_pre_latent_ln_w, moe.pre_moe_latent_layernorm.weight.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.moe_router_gate, moe.gate.weight.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.moe_expert_bias, moe.expert_bias.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.moe_shared_gate_proj, moe.shared_expert.gate_proj.weight.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.moe_shared_up_proj, moe.shared_expert.up_proj.weight.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.moe_shared_down_proj, moe.shared_expert.down_proj.weight.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.moe_shared_expert_gate, moe.shared_expert_gate.weight.detach())
                m_i = cfg_qsa.moe_intermediate_size
                for e in range(cfg_qsa.num_experts):
                    mw_unsharded.set_layer_weight(lid, QWEN4_5_W.moe_expert_gate_w.format(e=e), moe.w13_u8[e, :m_i])
                    mw_unsharded.set_layer_weight(lid, QWEN4_5_W.moe_expert_gate_scale.format(e=e), moe.w13_scale[e, :m_i])
                    mw_unsharded.set_layer_weight(lid, QWEN4_5_W.moe_expert_up_w.format(e=e), moe.w13_u8[e, m_i:])
                    mw_unsharded.set_layer_weight(lid, QWEN4_5_W.moe_expert_up_scale.format(e=e), moe.w13_scale[e, m_i:])
                    mw_unsharded.set_layer_weight(lid, QWEN4_5_W.moe_expert_down_w.format(e=e), moe.w2_u8[e])
                    mw_unsharded.set_layer_weight(lid, QWEN4_5_W.moe_expert_down_scale.format(e=e), moe.w2_scale[e])
            else:
                mlp = lyr.mlp
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.dense_gate_proj, mlp.gate_proj.weight.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.dense_up_proj, mlp.up_proj.weight.detach())
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.dense_down_proj, mlp.down_proj.weight.detach())
            if lyr.ffn_output_sconv is not None:
                mw_unsharded.set_layer_weight(lid, QWEN4_5_W.ffn_output_sconv, lyr.ffn_output_sconv.weight.detach())

        m_bind = Qwen4_5ForCausalLM(
            cfg_qsa, dtype=torch.bfloat16, device=self.device, ep_size=4, ep_rank=1, tp_size=4, tp_rank=1
        )
        bind_qwen4_5_model_weights(
            m_bind.model, mw_unsharded, cfg_qsa, lm_head=m_bind.lm_head, ep_size=4, ep_rank=1, tp_size=4, tp_rank=1
        )
        self.assertTrue(torch.equal(m_bind.lm_head.weight, tp_ranks[1].lm_head.weight))
        self.assertTrue(torch.equal(m_bind.model.layers[0].linear_attn.conv1d.weight, tp_ranks[1].model.layers[0].linear_attn.conv1d.weight))
        self.assertTrue(torch.equal(m_bind.model.layers[3].self_attn.q_proj.weight, tp_ranks[1].model.layers[3].self_attn.q_proj.weight))
        self.assertTrue(torch.equal(m_bind.model.layers[0].mlp.down_proj.weight, tp_ranks[1].model.layers[0].mlp.down_proj.weight))

        # Check DenseMLP Layer 0 partial sums across TP=4 ranks match TP=1
        B, T = 2, 12
        x_in = torch.randn((B, T, cfg_qsa.hidden_size), dtype=torch.bfloat16, device=self.device) * 0.25
        ref_dense = ref.model.layers[0].mlp(x_in)
        tp_dense_sum = torch.zeros_like(ref_dense, dtype=torch.float32)
        for m_r in tp_ranks:
            tp_dense_sum.add_(m_r.model.layers[0].mlp(x_in).float())
        self.assertGreater(_cos_sim(tp_dense_sum, ref_dense), 0.9999)

        # Check GDN Layer 0 (prefill T=12 + decode T=1) partial sums across TP=4 match TP=1
        st_ref_gdn = ref.model.layers[0].linear_attn.create_state(B, device=self.device)
        ref_gdn_p, _ = ref.model.layers[0].linear_attn(x_in[:, :10], state=st_ref_gdn)
        ref_gdn_d, _ = ref.model.layers[0].linear_attn(x_in[:, 10:11], state=st_ref_gdn)
        tp_gdn_p_sum = torch.zeros_like(ref_gdn_p, dtype=torch.float32)
        tp_gdn_d_sum = torch.zeros_like(ref_gdn_d, dtype=torch.float32)
        for m_r in tp_ranks:
            st_r = m_r.model.layers[0].linear_attn.create_state(B, device=self.device)
            self.assertEqual(st_r.ssm_state.shape, (B, 4, 64, 64))
            self.assertEqual(st_r.conv_state.shape, (B, 384, 3))
            gp, _ = m_r.model.layers[0].linear_attn(x_in[:, :10], state=st_r)
            gd, _ = m_r.model.layers[0].linear_attn(x_in[:, 10:11], state=st_r)
            tp_gdn_p_sum.add_(gp.float())
            tp_gdn_d_sum.add_(gd.float())
        self.assertGreater(_cos_sim(tp_gdn_p_sum, ref_gdn_p), 0.9998)
        self.assertGreater(_cos_sim(tp_gdn_d_sum, ref_gdn_d), 0.9998)

        # Check QSA FullAttn Layer 3 partial sums across TP=4 match TP=1
        pos = torch.arange(10, device=self.device).unsqueeze(0).expand(B, 10)
        self.assertTrue(
            torch.equal(
                r0.model.layers[3].self_attn.indexer.index_qk_proj.weight,
                ref.model.layers[3].self_attn.indexer.index_qk_proj.weight,
            )
        )
        kv_ref, _ = ref.model.layers[3].self_attn(x_in[:, :10], positions=pos)
        tp_sa_sum = torch.zeros_like(kv_ref, dtype=torch.float32)
        for m_r in tp_ranks:
            o_r, _ = m_r.model.layers[3].self_attn(x_in[:, :10], positions=pos)
            tp_sa_sum.add_(o_r.float())
        self.assertGreater(_cos_sim(tp_sa_sum, kv_ref), 0.9999)

        # Check Vocab-Parallel LMHead concatenation matches full TP=1 LMHead
        h_final = torch.randn((B, 4, cfg_qsa.hidden_size), dtype=torch.bfloat16, device=self.device) * 0.25
        ref_logits = ref.lm_head(h_final)
        tp_logits = torch.cat([m_r.lm_head(h_final) for m_r in tp_ranks], dim=-1)
        self.assertLess((tp_logits.float() - ref_logits.float()).abs().max().item(), 1e-4)

    def test_06_distributed_gloo_tp4_ep4_8layer_e2e(self) -> None:
        """Real 4-process Gloo distributed TP=4 + EP=4 8-layer QSA/GDN/MoE/LMHead vs TP=1 parity."""
        import socket
        import torch.multiprocessing as mp

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]

        ctx = mp.get_context("spawn")
        err_q = ctx.Queue()
        procs = [
            ctx.Process(target=_gloo_tp4_ep4_worker, args=(r, 4, port, err_q))
            for r in range(4)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=120)
        errors = []
        while not err_q.empty():
            errors.append(err_q.get())
        for r, p in enumerate(procs):
            if p.is_alive():
                p.terminate()
                self.fail(f"Gloo TP=4 worker {r} timed out")
            self.assertEqual(p.exitcode, 0, f"Gloo TP=4 worker {r} exited {p.exitcode}: {errors}")
        self.assertEqual(errors, [])

    def test_07_distributed_gloo_multi_parallel_modes_8layer_parity(self) -> None:
        """Real 4-rank Gloo E2E parity for dp4_ep4, dp2_tp2_ep4, and sp4_tp4_ep4 (distinct DP inputs!)."""
        import socket
        import torch.multiprocessing as mp

        for pmode in ("dp4_ep4", "dp2_tp2_ep4", "sp4_tp4_ep4"):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]

            ctx = mp.get_context("spawn")
            err_q = ctx.Queue()
            procs = [
                ctx.Process(target=_gloo_multi_parallel_worker, args=(r, 4, port, pmode, err_q))
                for r in range(4)
            ]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout=120)
            errors = []
            while not err_q.empty():
                errors.append(err_q.get())
            for r, p in enumerate(procs):
                if p.is_alive():
                    p.terminate()
                    self.fail(f"Gloo {pmode} worker {r} timed out")
                self.assertEqual(p.exitcode, 0, f"Gloo {pmode} worker {r} exited {p.exitcode}: {errors}")
            self.assertEqual(errors, [])

    @torch.no_grad()
    def test_08_dense_linear_mode_auto_and_fp8_target_parity(self) -> None:
        """Verify Target SM100FP8Linear wrapping (qkvz/out_proj/q_proj/o_proj/lm_head/MLP) + zero-copy alias + parity."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from dataclasses import replace
        from rtp_llm.models_py.modules.qwen4_5.fp8_linear import SM100FP8Linear

        device = torch.device("cuda:0")
        cfg = replace(
            make_mini_qwen4_5_config(),
            indexer_n_heads=8,
            indexer_kv_heads=1,
            indexer_head_dim=64,
            indexer_compress_ratio=4,
            indexer_budget=16,
        )
        m_ref = Qwen4_5ForCausalLM(cfg, dtype=torch.bfloat16, device=device, dense_linear_mode="bf16")
        m_ref.init_synthetic_weights(seed=777, ple_mode="gpu_full")
        _stabilize_router_margins(m_ref)
        m_ref.prepare_weights()

        # Test both init-time dense_linear_mode="auto" and post-load upgrade via prepare_weights
        m_auto = Qwen4_5ForCausalLM(cfg, dtype=torch.bfloat16, device=device, dense_linear_mode="bf16")
        from rtp_llm.models_py.model_desc.qwen4_5_runtime import Qwen4_5RTPModel
        from rtp_llm.ops import ParallelismConfig

        m_auto.init_synthetic_weights(seed=777, ple_mode="gpu_full")
        _stabilize_router_margins(m_auto)
        rtp_view = Qwen4_5RTPModel.from_causal_lm(m_auto, parallelism_config=ParallelismConfig())
        qkvz_ptr = m_auto.model.layers[1].linear_attn.in_proj_qkvz.weight.data_ptr()
        qproj_ptr = m_auto.model.layers[3].self_attn.q_proj.weight.data_ptr()
        lm_ptr = m_auto.lm_head.weight.data_ptr()
        m_auto.prepare_weights(dense_linear_mode="auto")

        self.assertIsInstance(m_auto.lm_head, SM100FP8Linear)
        self.assertIs(rtp_view.lm_head, m_auto.lm_head)
        self.assertEqual(m_auto.lm_head.weight.data_ptr(), lm_ptr)
        self.assertIsInstance(m_auto.model.layers[1].linear_attn.in_proj_qkvz, SM100FP8Linear)
        self.assertIsInstance(m_auto.model.layers[1].linear_attn.out_proj, SM100FP8Linear)
        self.assertEqual(m_auto.model.layers[1].linear_attn.in_proj_qkvz.weight.data_ptr(), qkvz_ptr)
        self.assertIsInstance(m_auto.model.layers[3].self_attn.q_proj, SM100FP8Linear)
        self.assertIsInstance(m_auto.model.layers[3].self_attn.o_proj, SM100FP8Linear)
        self.assertEqual(m_auto.model.layers[3].self_attn.q_proj.weight.data_ptr(), qproj_ptr)
        self.assertIsNotNone(m_auto.model.layers[0].mlp._fp8_gate_up)
        self.assertIsNotNone(m_auto.model.layers[1].mlp._fp8_latent_down_gate)

        B, T_prompt, T_gen = 2, 16, 4
        g = torch.Generator(device=device).manual_seed(9090)
        tokens = torch.randint(10, cfg.vocab_size, (B, T_prompt + T_gen), device=device, generator=g)

        st_ref = m_ref.create_runtime_state(B, max_seq_len=64, enable_yoco_decode=True)
        st_auto = m_auto.create_runtime_state(B, max_seq_len=64, enable_yoco_decode=True)
        ref_p = m_ref(tokens[:, :T_prompt], state=st_ref, enable_yoco_decode=True)
        auto_p = m_auto(tokens[:, :T_prompt], state=st_auto, enable_yoco_decode=True)
        self.assertGreater(_cos_sim(auto_p, ref_p), 0.999)

        for t in range(T_prompt, T_prompt + T_gen):
            r_d = m_ref(tokens[:, t : t + 1], state=st_ref, enable_yoco_decode=True)
            a_d = m_auto(tokens[:, t : t + 1], state=st_auto, enable_yoco_decode=True)
            self.assertGreater(_cos_sim(a_d, r_d), 0.999)

    @torch.no_grad()
    def test_09_dp4_ep4_shared_host_ple_pss_cleanup_and_parity(self) -> None:
        """Verify 4-rank DP4+EP4 shared host PLE occupies 1 physical copy (Pss), cleans up, captures CG, and matches bitwise."""
        import glob
        import socket
        import tempfile
        import torch.multiprocessing as mp
        from rtp_llm.models_py.modules.qwen4_5.checkpoint_loader import save_qwen4_5_safetensors_checkpoint

        cfg = make_mini_qwen4_5_config()
        ref = Qwen4_5ForCausalLM(cfg, dtype=torch.bfloat16, device=self.device)
        ref.init_synthetic_weights(seed=7707, ple_mode="gpu_full")
        _stabilize_router_margins(ref)
        ref.prepare_weights()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_qwen4_5_safetensors_checkpoint(ref, tmpdir)
            del ref
            torch.cuda.empty_cache()

            for pmode in ("cpu_uva", "gpu_sharded"):
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]

                ctx = mp.get_context("spawn")
                err_q = ctx.Queue()
                res_q = ctx.Queue()
                procs = [
                    ctx.Process(
                        target=_gloo_shared_ple_worker,
                        args=(r, 4, port, tmpdir, pmode, err_q, res_q),
                    )
                    for r in range(4)
                ]
                for proc in procs:
                    proc.start()
                for proc in procs:
                    proc.join(timeout=120)
                errors = []
                while not err_q.empty():
                    errors.append(err_q.get())
                for r, proc in enumerate(procs):
                    if proc.is_alive():
                        proc.terminate()
                        self.fail(f"Gloo shared PLE ({pmode}) worker {r} timed out")
                    self.assertEqual(proc.exitcode, 0, f"worker {r} exited {proc.exitcode}: {errors}")
                self.assertEqual(errors, [])

                stats = sorted([res_q.get() for _ in range(4)], key=lambda x: x[0])
                total_pss_mb = sum(s[2] for s in stats) / 1024.0
                max_shared_mb = max(s[3] for s in stats) / 1024.0
                cleanup_pss_mb = sum(s[4] for s in stats) / 1024.0
                print(
                    f"[Shared PLE {pmode}] 128MiB 4-rank Pss_sum_delta={total_pss_mb:.1f}MiB, "
                    f"Shared_Dirty={max_shared_mb:.1f}MiB, post_cleanup_Pss={cleanup_pss_mb:.1f}MiB, "
                    f"cos={[round(s[5], 6) for s in stats]}"
                )
                self.assertGreater(total_pss_mb, 100.0)
                self.assertLess(total_pss_mb, 200.0)
                self.assertGreater(max_shared_mb, 110.0)
                self.assertLess(cleanup_pss_mb, 32.0)
                leftover = glob.glob("/tmp/qwen45_ple_*") + glob.glob("/dev/shm/qwen45_ple_*")
                self.assertEqual(leftover, [], f"Leaked shared PLE files: {leftover}")

    @torch.no_grad()
    def test_10_tbo_fast_comm_all_gather_cg_and_shm_alias_parity(self) -> None:
        """Verify Target TBO (CG replay, stateful YoCo, 0-token DP rank), dp2_tp2_ep4 fast_comm.all_gather, and shm/mmap/cpu_uva aliases."""
        import glob
        import socket
        import tempfile
        from dataclasses import replace
        import torch.multiprocessing as mp
        from rtp_llm.models_py.modules.qwen4_5.checkpoint_loader import (
            load_qwen4_5_from_checkpoint,
            save_qwen4_5_safetensors_checkpoint,
        )

        cfg = make_mini_qwen4_5_config()
        cfg_stateless = replace(cfg, share_map="", enable_yoco_decode=False)

        ref = Qwen4_5ForCausalLM(cfg, dtype=torch.bfloat16, device=self.device)
        ref.init_synthetic_weights(seed=7708, ple_mode="gpu_full")
        _stabilize_router_margins(ref)
        ref.prepare_weights()

        ref_sl = Qwen4_5ForCausalLM(cfg_stateless, dtype=torch.bfloat16, device=self.device)
        ref_sl.init_synthetic_weights(seed=7708, ple_mode="cpu_uva")
        _stabilize_router_margins(ref_sl)
        ref_sl.prepare_weights()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_qwen4_5_safetensors_checkpoint(ref, tmpdir)

            # Single-rank alias smoke check: "mmap" and "auto" + single-rank stateless enable_tbo=True & YoCo forward_tbo parity
            for alias_mode in ("mmap", "auto"):
                m_alias = Qwen4_5ForCausalLM(cfg, dtype=torch.bfloat16, device=self.device)
                load_qwen4_5_from_checkpoint(m_alias, tmpdir, ple_mode=alias_mode, num_workers=1)
                _stabilize_router_margins(m_alias)
                m_alias.prepare_weights()
                toks = torch.randint(10, cfg.vocab_size, (4, 6), device=self.device)
                st_ref0 = ref.create_runtime_state(2, max_seq_len=32, enable_yoco_decode=True)
                st_ref1 = ref.create_runtime_state(2, max_seq_len=32, enable_yoco_decode=True)
                st_al0 = m_alias.create_runtime_state(2, max_seq_len=32, enable_yoco_decode=True)
                st_al1 = m_alias.create_runtime_state(2, max_seq_len=32, enable_yoco_decode=True)
                lg_tbo0, lg_tbo1 = m_alias.forward_tbo((toks[:2], toks[2:]), states=(st_al0, st_al1))
                self.assertGreater(_cos_sim(lg_tbo0, ref(toks[:2], state=st_ref0)), 0.9999)
                self.assertGreater(_cos_sim(lg_tbo1, ref(toks[2:], state=st_ref1)), 0.9999)
                del m_alias

            class _ProbeA2A:
                def __init__(self) -> None:
                    self.tbo_inited = False
                    self.world_size = 1
                def _ensure_tbo_resources(self) -> None:
                    self.tbo_inited = True
            probe_a2a = _ProbeA2A()
            m_sl = Qwen4_5ForCausalLM(
                cfg_stateless, dtype=torch.bfloat16, device=self.device, all_to_all_comm=probe_a2a
            )
            m_sl.init_synthetic_weights(seed=7708, ple_mode="cpu_uva")
            _stabilize_router_margins(m_sl)
            m_sl.prepare_weights()
            self.assertTrue(probe_a2a.tbo_inited)

            toks_sl = torch.randint(10, cfg.vocab_size, (4, 6), device=self.device)
            lg_sl_tbo = m_sl(toks_sl, enable_tbo=True)
            lg_sl_mb = torch.cat([ref_sl(toks_sl[:2]), ref_sl(toks_sl[2:])], dim=0)
            self.assertGreater(_cos_sim(lg_sl_tbo, lg_sl_mb), 0.9999)
            self.assertGreater(_cos_sim(lg_sl_tbo, ref_sl(toks_sl)), 0.999)
            # R3: bsz=1 & stateful enable_tbo + batched forward_tbo(logits_to_keep) + fast_ar_max_tokens arg
            lg_sl_b1 = m_sl(toks_sl[:1], enable_tbo=True)
            self.assertGreater(_cos_sim(lg_sl_b1, ref_sl(toks_sl[:1])), 0.9999)
            lg_k0, lg_k1 = m_sl.forward_tbo((toks_sl[:2], toks_sl[2:]), logits_to_keep=2)
            self.assertEqual(lg_k0.shape, (2, 2, cfg.vocab_size))
            self.assertGreater(_cos_sim(lg_k0, ref_sl(toks_sl[:2], logits_to_keep=2)), 0.9999)
            self.assertGreater(_cos_sim(lg_k1, ref_sl(toks_sl[2:], logits_to_keep=2)), 0.9999)
            m_kw = Qwen4_5ForCausalLM(cfg_stateless, dtype=torch.bfloat16, device=self.device, fast_ar_max_tokens=256)
            self.assertEqual((m_kw.sp_size, m_kw.sp_rank), (1, 0))
            del m_kw

            # Single-GPU CUDA Graph capture/replay of 8-layer forward_tbo
            cg_mb0 = torch.randint(10, cfg.vocab_size, (2, 6), device=self.device)
            cg_mb1 = torch.randint(10, cfg.vocab_size, (2, 6), device=self.device)
            for _ in range(2):
                _ = m_sl.forward_tbo((cg_mb0, cg_mb1), equal_tokens=True)
            torch.cuda.synchronize(self.device)
            g_tbo = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_tbo):
                cg_out0, cg_out1 = m_sl.forward_tbo((cg_mb0, cg_mb1), equal_tokens=True)
            new_mb0 = torch.randint(10, cfg.vocab_size, (2, 6), device=self.device)
            new_mb1 = torch.randint(10, cfg.vocab_size, (2, 6), device=self.device)
            cg_mb0.copy_(new_mb0)
            cg_mb1.copy_(new_mb1)
            ref_cg0 = ref_sl(new_mb0)
            ref_cg1 = ref_sl(new_mb1)
            g_tbo.replay()
            torch.cuda.synchronize(self.device)
            self.assertGreater(_cos_sim(cg_out0, ref_cg0), 0.9999)
            self.assertGreater(_cos_sim(cg_out1, ref_cg1), 0.9999)
            del g_tbo, cg_out0, cg_out1, m_sl, ref_sl, ref
            torch.cuda.empty_cache()

            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]

            ctx = mp.get_context("spawn")
            err_q = ctx.Queue()
            procs = [
                ctx.Process(
                    target=_gloo_tbo_and_shm_worker,
                    args=(r, 4, port, tmpdir, err_q),
                )
                for r in range(4)
            ]
            for proc in procs:
                proc.start()
            for proc in procs:
                proc.join(timeout=150)
            errors = []
            while not err_q.empty():
                errors.append(err_q.get())
            for r, proc in enumerate(procs):
                if proc.is_alive():
                    proc.terminate()
                    self.fail(f"Gloo TBO/shm worker {r} timed out")
                self.assertEqual(proc.exitcode, 0, f"worker {r} exited {proc.exitcode}: {errors}")
            self.assertEqual(errors, [])
            leftover = glob.glob("/tmp/qwen45_ple_*") + glob.glob("/dev/shm/qwen45_ple_*")
            self.assertEqual(leftover, [], f"Leaked shared PLE files: {leftover}")


@torch.no_grad()
def _gloo_tbo_and_shm_worker(rank: int, world_size: int, port: int, ckpt_dir: str, err_q) -> None:
    import gc
    from dataclasses import replace
    from rtp_llm.models_py.modules.qwen4_5.checkpoint_loader import load_qwen4_5_from_checkpoint

    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        try:
            cfg = make_mini_qwen4_5_config()
            cfg_stateless = replace(cfg, share_map="", enable_yoco_decode=False)

            # 1) init_synthetic_weights(ple_mode="cpu_uva") under 4-rank DP4+EP4
            m_syn = Qwen4_5ForCausalLM(cfg, dtype=torch.bfloat16, device=device, parallel_mode="dp4_ep4")
            m_syn.init_synthetic_weights(seed=8808, ple_mode="cpu_uva")
            assert m_syn.model.layers[1].ple._ple_ep_group is None
            del m_syn
            gc.collect()

            # 2) Stateless dp4_ep4 + cpu_uva + 4-rank Gloo 3-stage TBO parity
            m_cg = Qwen4_5ForCausalLM(
                cfg_stateless, dtype=torch.bfloat16, device=device, parallel_mode="dp4_ep4", dp_equal_tokens=True
            )
            m_cg.init_synthetic_weights(seed=7708, ple_mode="cpu_uva")
            _stabilize_router_margins(m_cg)
            m_cg.prepare_weights()
            for layer in m_cg.model.layers:
                if hasattr(layer.mlp, "backend"):
                    layer.mlp.backend = "triton_splitk"
            assert m_cg.model.layers[1].ple._ple_ep_group is None
            assert m_cg.model.all_to_all_comm is not None
            assert hasattr(m_cg.model.all_to_all_comm, "dispatch_async")
            assert hasattr(m_cg.model.all_to_all_comm, "combine_async")

            m_cg_ref = Qwen4_5ForCausalLM(cfg_stateless, dtype=torch.bfloat16, device=device)
            m_cg_ref.init_synthetic_weights(seed=7708, ple_mode="cpu_uva")
            _stabilize_router_margins(m_cg_ref)
            m_cg_ref.prepare_weights()
            for layer in m_cg_ref.model.layers:
                if hasattr(layer.mlp, "backend"):
                    layer.mlp.backend = "triton_splitk"

            gen = torch.Generator(device=device).manual_seed(6600 + rank * 37)
            mb0 = torch.randint(10, cfg.vocab_size, (2, 6), device=device, generator=gen)
            mb1 = torch.randint(10, cfg.vocab_size, (2, 6), device=device, generator=gen)
            with torch.inference_mode():
                gloo_l0, gloo_l1 = m_cg.forward_tbo((mb0, mb1), equal_tokens=True)
                ref0 = m_cg_ref(mb0)
                ref1 = m_cg_ref(mb1)
                cos0 = _cos_sim(gloo_l0, ref0)
                cos1 = _cos_sim(gloo_l1, ref1)
            assert cos0 > 0.999 and cos1 > 0.999, f"rank={rank} Gloo TBO cos={cos0}, {cos1}"
            del m_cg, m_cg_ref
            gc.collect()
            torch.cuda.empty_cache()

            # 3) Full YoCo dp4_ep4 + ple_mode="shm" stateful TBO + asymmetric 0-token DP rank parity
            m_tbo = Qwen4_5ForCausalLM(
                cfg, dtype=torch.bfloat16, device=device, parallel_mode="dp4_ep4", dp_equal_tokens=True
            )
            load_qwen4_5_from_checkpoint(m_tbo, ckpt_dir, ple_mode="shm", num_workers=2)
            _stabilize_router_margins(m_tbo)
            m_tbo.prepare_weights()
            for layer in m_tbo.model.layers:
                if hasattr(layer.mlp, "backend"):
                    layer.mlp.backend = "triton_splitk"

            m_ref = Qwen4_5ForCausalLM(cfg, dtype=torch.bfloat16, device=device)
            load_qwen4_5_from_checkpoint(m_ref, ckpt_dir, ple_mode="gpu_full", num_workers=2)
            _stabilize_router_margins(m_ref)
            m_ref.prepare_weights()
            for layer in m_ref.model.layers:
                if hasattr(layer.mlp, "backend"):
                    layer.mlp.backend = "triton_splitk"

            st_t0 = m_tbo.create_runtime_state(2, max_seq_len=32, enable_yoco_decode=True)
            st_t1 = m_tbo.create_runtime_state(2, max_seq_len=32, enable_yoco_decode=True)
            st_r0 = m_ref.create_runtime_state(2, max_seq_len=32, enable_yoco_decode=True)
            st_r1 = m_ref.create_runtime_state(2, max_seq_len=32, enable_yoco_decode=True)
            toks0 = torch.randint(10, cfg.vocab_size, (2, 10), device=device, generator=gen)
            toks1 = torch.randint(10, cfg.vocab_size, (2, 10), device=device, generator=gen)
            with torch.inference_mode():
                p_t0, p_t1 = m_tbo.forward_tbo((toks0[:, :8], toks1[:, :8]), states=(st_t0, st_t1), enable_yoco_decode=True, equal_tokens=True)
                p_r0 = m_ref(toks0[:, :8], state=st_r0, enable_yoco_decode=True)
                p_r1 = m_ref(toks1[:, :8], state=st_r1, enable_yoco_decode=True)
                assert _cos_sim(p_t0, p_r0) > 0.999 and _cos_sim(p_t1, p_r1) > 0.999
                d_t0, d_t1 = m_tbo.forward_tbo((toks0[:, 8:9], toks1[:, 8:9]), states=(st_t0, st_t1), enable_yoco_decode=True, equal_tokens=True)
                d_r0 = m_ref(toks0[:, 8:9], state=st_r0, enable_yoco_decode=True)
                d_r1 = m_ref(toks1[:, 8:9], state=st_r1, enable_yoco_decode=True)
                assert _cos_sim(d_t0, d_r0) > 0.999 and _cos_sim(d_t1, d_r1) > 0.999
                # R3: stateful enable_tbo token-split MoE TBO + direct layer.forward_mlp_tbo parity in dp4_ep4
                st_single_tbo = m_tbo.create_runtime_state(2, max_seq_len=32, enable_yoco_decode=True)
                st_single_ref = m_ref.create_runtime_state(2, max_seq_len=32, enable_yoco_decode=True)
                p_moe_tbo = m_tbo(toks0[:, :8], state=st_single_tbo, enable_yoco_decode=True, enable_tbo=True, tbo_equal_tokens=True)
                p_moe_ref = m_ref(toks0[:, :8], state=st_single_ref, enable_yoco_decode=True)
                assert _cos_sim(p_moe_tbo, p_moe_ref) > 0.999
                x_mlp_tbo = torch.randn((6, cfg.hidden_size), dtype=torch.bfloat16, device=device, generator=gen) * 0.2
                assert _cos_sim(m_tbo.model.layers[1].forward_mlp_tbo(x_mlp_tbo, equal_tokens=True), m_ref.model.layers[1].mlp(x_mlp_tbo)) > 0.999

            # Asymmetric 0-token DP rank TBO (rank 0,2 empty on mb0; rank 1,3 empty on mb1)
            for layer in m_tbo.model.layers:
                if hasattr(layer.mlp, "dp_equal_tokens"):
                    layer.mlp.dp_equal_tokens = False
            b_a0 = 0 if (rank % 2 == 0) else 2
            b_a1 = 2 if (rank % 2 == 0) else 0
            a0 = torch.randint(10, cfg.vocab_size, (b_a0, 6), device=device, generator=gen)
            a1 = torch.randint(10, cfg.vocab_size, (b_a1, 6), device=device, generator=gen)
            st_a0 = m_tbo.create_runtime_state(b_a0, max_seq_len=32, enable_yoco_decode=True)
            st_a1 = m_tbo.create_runtime_state(b_a1, max_seq_len=32, enable_yoco_decode=True)
            with torch.inference_mode():
                oa0, oa1 = m_tbo.forward_tbo((a0, a1), states=(st_a0, st_a1), equal_tokens=False)
                assert oa0.shape == (b_a0, 6, cfg.vocab_size) and oa1.shape == (b_a1, 6, cfg.vocab_size)
                if b_a0 > 0:
                    sr0 = m_ref.create_runtime_state(b_a0, max_seq_len=32, enable_yoco_decode=True)
                    assert _cos_sim(oa0, m_ref(a0, state=sr0)) > 0.999
                if b_a1 > 0:
                    sr1 = m_ref.create_runtime_state(b_a1, max_seq_len=32, enable_yoco_decode=True)
                    assert _cos_sim(oa1, m_ref(a1, state=sr1)) > 0.999

            del m_tbo, m_ref
            gc.collect()
            torch.cuda.empty_cache()

            # 4) dp2_tp2_ep4 fast_comm.all_gather + forward_tbo parity
            m_dt = Qwen4_5ForCausalLM(
                cfg, dtype=torch.bfloat16, device=device, parallel_mode="dp2_tp2_ep4", dp_equal_tokens=True
            )
            m_dt.init_synthetic_weights(seed=7709, ple_mode="gpu_full")
            _stabilize_router_margins(m_dt)
            m_dt.prepare_weights()
            for layer in m_dt.model.layers:
                if hasattr(layer.mlp, "backend"):
                    layer.mlp.backend = "triton_splitk"
            assert m_dt.fast_all_reduce is not None
            assert m_dt.fast_all_reduce.world_size == 2 and m_dt.fast_all_reduce.group is m_dt.tp_group

            m_dt_ref = Qwen4_5ForCausalLM(cfg, dtype=torch.bfloat16, device=device)
            m_dt_ref.init_synthetic_weights(seed=7709, ple_mode="gpu_full")
            _stabilize_router_margins(m_dt_ref)
            m_dt_ref.prepare_weights()
            for layer in m_dt_ref.model.layers:
                if hasattr(layer.mlp, "backend"):
                    layer.mlp.backend = "triton_splitk"

            dp_gen = torch.Generator(device=device).manual_seed(9500 + m_dt.dp_rank * 41)
            dt_ids = torch.randint(10, cfg.vocab_size, (4, 6), device=device, generator=dp_gen)
            with torch.inference_mode():
                st_dt = m_dt.create_runtime_state(4, max_seq_len=32, enable_yoco_decode=True)
                dt_lg = m_dt(dt_ids, state=st_dt)
                st_dt0 = m_dt.create_runtime_state(2, max_seq_len=32, enable_yoco_decode=True)
                st_dt1 = m_dt.create_runtime_state(2, max_seq_len=32, enable_yoco_decode=True)
                dt_tbo0, dt_tbo1 = m_dt.forward_tbo((dt_ids[:2], dt_ids[2:]), states=(st_dt0, st_dt1), equal_tokens=True)
                st_r0 = m_dt_ref.create_runtime_state(2, max_seq_len=32, enable_yoco_decode=True)
                st_r1 = m_dt_ref.create_runtime_state(2, max_seq_len=32, enable_yoco_decode=True)
                dt_ref0 = m_dt_ref(dt_ids[:2], state=st_r0)
                dt_ref1 = m_dt_ref(dt_ids[2:], state=st_r1)
                assert _cos_sim(dt_lg[:2], dt_ref0) > 0.999 and _cos_sim(dt_lg[2:], dt_ref1) > 0.999
                assert _cos_sim(dt_tbo0, dt_ref0) > 0.999 and _cos_sim(dt_tbo1, dt_ref1) > 0.999

            del m_dt, m_dt_ref
            gc.collect()
            dist.barrier()
        finally:
            dist.destroy_process_group()
    except BaseException as exc:
        err_q.put(f"rank {rank}: {exc!r}")
        raise


@torch.no_grad()
def _gloo_multi_parallel_worker(rank: int, world_size: int, port: int, pmode: str, err_q) -> None:
    try:
        from dataclasses import replace
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        try:
            cfg = replace(
                make_mini_qwen4_5_config(),
                indexer_n_heads=8,
                indexer_kv_heads=1,
                indexer_head_dim=64,
                indexer_compress_ratio=4,
                indexer_budget=16,
            )
            par_model = Qwen4_5ForCausalLM(
                cfg,
                dtype=torch.bfloat16,
                device=device,
                parallel_mode=pmode,
                dp_equal_tokens=True,
                kv_dtype="bf16",
            )
            assert par_model.parallel_mode == pmode
            par_model.init_synthetic_weights(seed=2027, ple_mode="gpu_sharded")
            _stabilize_router_margins(par_model)
            par_model.prepare_weights()
            for layer in par_model.model.layers:
                if hasattr(layer.mlp, "backend"):
                    layer.mlp.backend = "triton_splitk"

            B, T_prompt, T_gen = 2, 16, 4
            g = torch.Generator(device=device).manual_seed(5151 + par_model.dp_rank * 99)
            tokens = torch.randint(10, cfg.vocab_size, (B, T_prompt + T_gen), device=device, generator=g)

            st_par = par_model.create_runtime_state(B, max_seq_len=64, enable_yoco_decode=True)
            par_outs = [par_model(tokens[:, :T_prompt], state=st_par, enable_yoco_decode=True)]
            for t in range(T_prompt, T_prompt + T_gen):
                par_outs.append(par_model(tokens[:, t : t + 1], state=st_par, enable_yoco_decode=True))
            logits_par = torch.cat(par_outs, dim=1)

            ref_model = Qwen4_5ForCausalLM(cfg, dtype=torch.bfloat16, device=device)
            ref_model.init_synthetic_weights(seed=2027, ple_mode="gpu_full")
            _stabilize_router_margins(ref_model)
            ref_model.prepare_weights()
            for layer in ref_model.model.layers:
                if hasattr(layer.mlp, "backend"):
                    layer.mlp.backend = "triton_splitk"

            st_ref = ref_model.create_runtime_state(B, max_seq_len=64, enable_yoco_decode=True)
            ref_outs = [ref_model(tokens[:, :T_prompt], state=st_ref, enable_yoco_decode=True)]
            for t in range(T_prompt, T_prompt + T_gen):
                ref_outs.append(ref_model(tokens[:, t : t + 1], state=st_ref, enable_yoco_decode=True))
            logits_ref = torch.cat(ref_outs, dim=1)

            cos_p = _cos_sim(logits_par[:, :T_prompt], logits_ref[:, :T_prompt])
            cos_d = _cos_sim(logits_par[:, T_prompt:], logits_ref[:, T_prompt:])
            assert cos_p > 0.999 and cos_d > 0.999, (
                f"{pmode} rank={rank} dp_rank={par_model.dp_rank} E2E mismatch: prefill={cos_p}, decode={cos_d}"
            )
            if pmode == "dp4_ep4":
                for layer in par_model.model.layers:
                    if hasattr(layer.mlp, "backend"):
                        layer.mlp.backend = "auto"
                for layer in ref_model.model.layers:
                    if hasattr(layer.mlp, "backend"):
                        layer.mlp.backend = "deepgemm_masked"
                st_p2 = par_model.create_runtime_state(B, max_seq_len=64, enable_yoco_decode=True)
                st_r2 = ref_model.create_runtime_state(B, max_seq_len=64, enable_yoco_decode=True)
                lp2 = par_model(tokens[:, :T_prompt], state=st_p2, enable_yoco_decode=True)
                lr2 = ref_model(tokens[:, :T_prompt], state=st_r2, enable_yoco_decode=True)
                cos_w4a8 = _cos_sim(lp2, lr2)
                assert cos_w4a8 > 0.995, f"dp4_ep4 W4A8 auto vs deepgemm_masked ref mismatch: {cos_w4a8}"
            dist.barrier()
        finally:
            dist.destroy_process_group()
    except BaseException as exc:
        err_q.put(f"rank {rank} ({pmode}): {exc!r}")
        raise


@torch.no_grad()
def _gloo_tp4_ep4_worker(rank: int, world_size: int, port: int, err_q) -> None:
    try:
        from dataclasses import replace
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        try:
            cfg = replace(
                make_mini_qwen4_5_config(),
                    indexer_n_heads=8,
                indexer_kv_heads=1,
                indexer_head_dim=64,
                indexer_compress_ratio=4,
                indexer_budget=16,
            )
            tp_model = Qwen4_5ForCausalLM(
                cfg,
                dtype=torch.bfloat16,
                device=device,
                ep_size=world_size,
                ep_rank=rank,
                ep_group=dist.group.WORLD,
                tp_size=world_size,
                tp_rank=rank,
                tp_group=dist.group.WORLD,
            )
            tp_model.init_synthetic_weights(seed=2027, ple_mode="gpu_sharded")
            _stabilize_router_margins(tp_model)
            tp_model.prepare_weights()

            B, T_prompt, T_gen = 2, 16, 4
            g = torch.Generator(device=device).manual_seed(5151)
            tokens = torch.randint(10, cfg.vocab_size, (B, T_prompt + T_gen), device=device, generator=g)

            st_tp = tp_model.create_runtime_state(B, max_seq_len=64, enable_yoco_decode=True)
            tp_outs = [tp_model(tokens[:, :T_prompt], state=st_tp, enable_yoco_decode=True)]
            for t in range(T_prompt, T_prompt + T_gen):
                tp_outs.append(tp_model(tokens[:, t : t + 1], state=st_tp, enable_yoco_decode=True))
            logits_tp = torch.cat(tp_outs, dim=1)

            r0_logits = logits_tp.clone()
            dist.broadcast(r0_logits, src=0)
            sync_diff = float((logits_tp.float() - r0_logits.float()).abs().max().item())
            assert sync_diff < 1e-4, f"Rank {rank} diverged from rank 0: {sync_diff}"

            if rank == 0:
                ref_model = Qwen4_5ForCausalLM(
                    cfg, dtype=torch.bfloat16, device=device
                )
                ref_model.init_synthetic_weights(seed=2027, ple_mode="gpu_full")
                _stabilize_router_margins(ref_model)
                ref_model.prepare_weights()

                st_ref = ref_model.create_runtime_state(B, max_seq_len=64, enable_yoco_decode=True)
                ref_outs = [ref_model(tokens[:, :T_prompt], state=st_ref, enable_yoco_decode=True)]
                for t in range(T_prompt, T_prompt + T_gen):
                    ref_outs.append(ref_model(tokens[:, t : t + 1], state=st_ref, enable_yoco_decode=True))
                logits_ref = torch.cat(ref_outs, dim=1)

                cos_p = _cos_sim(logits_tp[:, :T_prompt], logits_ref[:, :T_prompt])
                cos_d = _cos_sim(logits_tp[:, T_prompt:], logits_ref[:, T_prompt:])
                max_p = float((logits_tp[:, :T_prompt].float() - logits_ref[:, :T_prompt].float()).abs().max().item())
                max_d = float((logits_tp[:, T_prompt:].float() - logits_ref[:, T_prompt:].float()).abs().max().item())
                assert cos_p > 0.999 and cos_d > 0.999, (
                    f"TP4+EP4 E2E cos mismatch: prefill={cos_p} (max={max_p}), decode={cos_d} (max={max_d})"
                )
            dist.barrier()
        finally:
            dist.destroy_process_group()
    except BaseException as exc:
        err_q.put(f"rank {rank}: {exc!r}")
        raise


def _read_smaps_kb() -> dict:
    res = {}
    with open(f"/proc/{os.getpid()}/smaps_rollup", encoding="utf-8") as f:
        for line in f:
            k, v = line.split(":", 1)
            if k in ("Rss", "Pss", "Shared_Dirty", "Private_Dirty"):
                res[k] = int(v.strip().split()[0])
    return res


@torch.no_grad()
def _gloo_shared_ple_worker(
    rank: int,
    world_size: int,
    port: int,
    ckpt_dir: str,
    ple_mode: str,
    err_q,
    res_q,
) -> None:
    import gc
    from rtp_llm.models_py.modules.qwen4_5.checkpoint_loader import (
        alloc_shared_host_ple_shards,
        finalize_shared_host_ple_shards,
        load_qwen4_5_from_checkpoint,
    )

    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        try:
            specs = [(torch.float8_e4m3fn, (65536, 256)) for _ in range(8)]
            base_m = _read_smaps_kb()
            sbuf = alloc_shared_host_ple_shards(specs, group=dist.group.WORLD, device=device)
            for idx, sh in enumerate(sbuf.shard_bufs):
                if idx % world_size == rank:
                    sh.view(torch.uint8).fill_(0x3C)
            del sh
            pinned_ok = finalize_shared_host_ple_shards(sbuf)
            assert pinned_ok, "Expected shared PLE host buffer to pin for CUDA UVA"
            dist.barrier()
            after_m = _read_smaps_kb()
            dist.barrier()
            del sbuf
            gc.collect()
            dist.barrier()
            clean_m = _read_smaps_kb()

            cfg = make_mini_qwen4_5_config()
            m_par = Qwen4_5ForCausalLM(
                cfg,
                dtype=torch.bfloat16,
                device=device,
                parallel_mode="dp4_ep4",
                dp_equal_tokens=True,
            )
            load_qwen4_5_from_checkpoint(m_par, ckpt_dir, ple_mode=ple_mode, num_workers=2)
            _stabilize_router_margins(m_par)
            m_par.prepare_weights()
            for layer in m_par.model.layers:
                if hasattr(layer.mlp, "backend"):
                    layer.mlp.backend = "triton_splitk"

            ple_par = m_par.model.layers[1].ple
            assert len(ple_par._ngram_shards) == cfg.split_ngram_parts
            assert ple_par._ple_ep_group is None

            h_in = torch.ones((2, 4, cfg.hc_count * cfg.hidden_size), dtype=torch.bfloat16, device=device) * 0.2
            ids_in = torch.randint(
                10, cfg.vocab_size, (2, 4), device=device, generator=torch.Generator(device).manual_seed(900 + rank)
            )
            with torch.inference_mode():
                ple_eager = ple_par(h_in, ids_in).clone()
                static_h = h_in.clone()
                static_ids = ids_in.clone()
                _ = ple_par(static_h, static_ids)
                torch.cuda.synchronize(device)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    static_out = ple_par(static_h, static_ids)
                g.replay()
                torch.cuda.synchronize(device)
                assert torch.equal(static_out, ple_eager)

            m_ref = Qwen4_5ForCausalLM(cfg, dtype=torch.bfloat16, device=device)
            load_qwen4_5_from_checkpoint(m_ref, ckpt_dir, ple_mode="gpu_full", num_workers=2)
            _stabilize_router_margins(m_ref)
            m_ref.prepare_weights()
            for layer in m_ref.model.layers:
                if hasattr(layer.mlp, "backend"):
                    layer.mlp.backend = "triton_splitk"

            with torch.inference_mode():
                ple_ref = m_ref.model.layers[1].ple(h_in, ids_in)
            assert torch.equal(ple_eager, ple_ref), "DP4+EP4 shared PLE bitwise mismatch vs gpu_full"

            B, T_prompt, T_gen = 2, 16, 4
            tok_seq = torch.randint(
                10, cfg.vocab_size, (B, T_prompt + T_gen), device=device, generator=torch.Generator(device).manual_seed(5151 + rank * 73)
            )
            st_par = m_par.create_runtime_state(B, max_seq_len=64, enable_yoco_decode=True)
            st_ref = m_ref.create_runtime_state(B, max_seq_len=64, enable_yoco_decode=True)
            with torch.inference_mode():
                lp = m_par(tok_seq[:, :T_prompt], state=st_par, enable_yoco_decode=True)
                lr = m_ref(tok_seq[:, :T_prompt], state=st_ref, enable_yoco_decode=True)
                cos_e2e = _cos_sim(lp, lr)
            assert cos_e2e > 0.999, f"dp4_ep4 shared PLE E2E cos mismatch: {cos_e2e}"

            del m_par, m_ref, g, static_out
            gc.collect()
            res_q.put(
                (
                    rank,
                    pinned_ok,
                    after_m["Pss"] - base_m["Pss"],
                    after_m["Shared_Dirty"] - base_m["Shared_Dirty"],
                    clean_m["Pss"] - base_m["Pss"],
                    cos_e2e,
                )
            )
            dist.barrier()
        finally:
            dist.destroy_process_group()
    except BaseException as exc:
        err_q.put(f"rank {rank} ({ple_mode}): {exc!r}")
        raise


@torch.no_grad()
def run_distributed_ep_test() -> None:
    """Multi-GPU NCCL EP=4 vs single-GPU EP=1 equivalence test.

    Verifies:
      1. Unmodified GPU-sharded FP8 PLE EP=4 vs EP=1 (bit-exact 0.0).
      2. Unmodified random-router LatentMoE EP=4 vs EP=1 across all 32 experts (<=1 ULP).
      3. Full 8-layer Prefill + 4-step YoCo Decode EP=4 vs EP=1 with cross-shard margins.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    cfg = make_mini_qwen4_5_config()
    ep_model = Qwen4_5ForCausalLM(
        cfg,
        dtype=torch.bfloat16,
        device=device,
        ep_size=world_size,
        ep_rank=rank,
        ep_group=dist.group.WORLD,
    )
    ep_model.init_synthetic_weights(seed=2026, ple_mode="gpu_sharded")
    ep_model.prepare_weights()

    B, T_prompt, T_gen = 2, 20, 4
    g = torch.Generator(device=device).manual_seed(4242)
    tokens = torch.randint(10, cfg.vocab_size, (B, T_prompt + T_gen), device=device, generator=g)
    x_ple = torch.randn(
        B, T_prompt, cfg.hidden_size * cfg.hc_count, device=device, dtype=torch.bfloat16, generator=g
    ) * 0.2
    x_moe = torch.randn(
        B, T_prompt, cfg.hidden_size, device=device, dtype=torch.bfloat16, generator=g
    ) * 0.2

    # 1 & 2: Unmodified PLE and LatentMoE single-layer EP=4
    ple_ep = ep_model.model.layers[1].ple(x_ple, tokens[:, :T_prompt])
    moe_ep = ep_model.model.layers[1].mlp(x_moe)

    # 3: Full 8-layer Prefill + 4-step YoCo Decode EP=4
    _stabilize_router_margins(ep_model)
    st_ep = ep_model.create_runtime_state(B, max_seq_len=64, enable_yoco_decode=True)
    ep_outs = [ep_model(tokens[:, :T_prompt], state=st_ep, enable_yoco_decode=True)]
    for t in range(T_prompt, T_prompt + T_gen):
        ep_outs.append(ep_model(tokens[:, t : t + 1], state=st_ep, enable_yoco_decode=True))
    logits_ep = torch.cat(ep_outs, dim=1)

    # Check all EP ranks produced identical logits
    rank0_logits = logits_ep.clone()
    dist.broadcast(rank0_logits, src=0)
    rank_diff = float((logits_ep.float() - rank0_logits.float()).abs().max().item())
    assert rank_diff < 1e-4, f"Rank {rank} diverged from rank 0: max_abs={rank_diff}"

    if rank == 0:
        ref_model = Qwen4_5ForCausalLM(
            cfg, dtype=torch.bfloat16, device=device, ep_size=1, ep_rank=0, ep_group=None
        )
        ref_model.init_synthetic_weights(seed=2026, ple_mode="gpu_full")
        ref_model.prepare_weights()

        ple_ref = ref_model.model.layers[1].ple(x_ple, tokens[:, :T_prompt])
        moe_ref = ref_model.model.layers[1].mlp(x_moe)
        ple_max = float((ple_ep.float() - ple_ref.float()).abs().max().item())
        moe_max = float((moe_ep.float() - moe_ref.float()).abs().max().item())
        moe_cos = _cos_sim(moe_ep, moe_ref)
        assert ple_max == 0.0, f"Sharded PLE diverged: max_abs={ple_max}"
        assert moe_max < 0.01 and moe_cos > 0.99995, (
            f"Single LatentMoE EP diverged: max_abs={moe_max}, cos={moe_cos}"
        )

        _stabilize_router_margins(ref_model)
        st_ref = ref_model.create_runtime_state(B, max_seq_len=64, enable_yoco_decode=True)
        ref_outs = [ref_model(tokens[:, :T_prompt], state=st_ref, enable_yoco_decode=True)]
        for t in range(T_prompt, T_prompt + T_gen):
            ref_outs.append(
                ref_model(tokens[:, t : t + 1], state=st_ref, enable_yoco_decode=True)
            )
        logits_ref = torch.cat(ref_outs, dim=1)

        max_abs_prefill = float(
            (logits_ep[:, :T_prompt].float() - logits_ref[:, :T_prompt].float()).abs().max().item()
        )
        cos_prefill = _cos_sim(logits_ep[:, :T_prompt], logits_ref[:, :T_prompt])
        max_abs_decode = float(
            (logits_ep[:, T_prompt:].float() - logits_ref[:, T_prompt:].float()).abs().max().item()
        )
        cos_decode = _cos_sim(logits_ep[:, T_prompt:], logits_ref[:, T_prompt:])
        print(
            f"[EP={world_size} vs EP=1] rank_sync={rank_diff:.6f} | "
            f"PLE max_abs={ple_max:.6f} | SingleMoE(unmodified) max_abs={moe_max:.6f} cos={moe_cos:.8f} | "
            f"8L Prefill: max_abs={max_abs_prefill:.6f} cos={cos_prefill:.8f} | "
            f"8L Decode(4): max_abs={max_abs_decode:.6f} cos={cos_decode:.8f}"
        )
        assert cos_prefill > 0.9995 and max_abs_prefill < 0.05
        assert cos_decode > 0.9995 and max_abs_decode < 0.05
        print(f"PASSED: NCCL EP={world_size} vs EP=1 full-model equivalence!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    if "--ep-test" in sys.argv:
        run_distributed_ep_test()
    else:
        unittest.main(verbosity=2)

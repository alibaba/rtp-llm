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

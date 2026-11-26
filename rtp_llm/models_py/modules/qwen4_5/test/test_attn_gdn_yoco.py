"""Unit tests and B300 micro-benchmarks for Qwen 4.5-Flash FullAttn, GDN, and YoCo sharing."""

from __future__ import annotations

import time
import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.qwen4_5.attention import (
    GemmaRMSNorm,
    Qwen4_5Attention,
    apply_partial_neox_rope,
    gemma_rmsnorm_and_partial_rope,
)
from rtp_llm.models_py.modules.qwen4_5.contracts import (
    Qwen4_5ArchConfig,
    SharedGDNInputs,
)
from rtp_llm.models_py.modules.qwen4_5.gdn import (
    Qwen4_5GatedDeltaNet,
    RMSNormGated,
    compute_gdn_gating,
    gdn_chunk_prefill_ragged_with_snapshots_triton,
    gdn_chunk_prefill_triton,
    gdn_recurrent_ref,
    gdn_recurrent_triton,
)
from rtp_llm.models_py.modules.qwen4_5.yoco import (
    YoCoSharingPlan,
    yoco_gdn_decode_readout,
    yoco_gdn_decode_readout_ref,
)

QWEN45_SHARE_MAP = (
    "21->22,21->23,20->24,21->25,21->26,21->27,20->28,"
    "21->29,21->30,21->31,20->32,21->33,21->34,21->35,"
    "20->36,21->37,21->38,21->39,20->40"
)


def make_config(hidden_size: int = 5120) -> Qwen4_5ArchConfig:
    return Qwen4_5ArchConfig(
        hidden_size=hidden_size,
        share_map=QWEN45_SHARE_MAP,
        enable_yoco_decode=True,
    )


def init_realistic_weights(module: torch.nn.Module) -> None:
    """Initialize linear/conv/norm weights with realistic unit-variance transformer scaling."""
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "A_log" in name:
                param.uniform_(-2.0, 1.0)
            elif "dt_bias" in name:
                param.uniform_(-1.0, 1.0)
            elif name.endswith("q_norm.weight") or name.endswith("k_norm.weight"):
                param.normal_(0.0, 0.05)
            elif name.endswith("norm.weight"):
                param.normal_(1.0, 0.05)
            elif "conv" in name or "sconv" in name:
                param.normal_(0.0, 0.2)
            elif param.ndim == 2:
                param.normal_(0.0, (param.shape[1] ** -0.5))
            else:
                param.normal_(0.0, 0.02)


def report_diff(label: str, a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    af = a.float()
    bf = b.float()
    diff = (af - bf).abs()
    max_abs = diff.max().item()
    denom = bf.abs().max().item() + 1e-8
    rel = max_abs / denom
    print(f"  [DIFF] {label}: max_abs={max_abs:.6e}, max_rel_to_peak={rel:.6e}")
    return max_abs, rel


def test_yoco_sharing_plan_and_param_shapes() -> None:
    print("\n=== Test 0: YoCoSharingPlan & Checkpoint Parameter Shapes ===")
    cfg = make_config(hidden_size=5120)
    plan = YoCoSharingPlan.from_config(cfg)

    assert plan.full_attn_sources == (19,), f"Unexpected full_attn_sources: {plan.full_attn_sources}"
    assert plan.full_attn_consumers == (23, 27, 31, 35, 39)
    assert plan.gdn_sources == (20,)
    assert plan.gdn_consumers == (21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38)
    assert len(plan.full_attn_consumers) == 5
    assert len(plan.gdn_consumers) == 14
    assert plan.kv_cache_owner(23) == 19
    assert plan.kv_cache_owner(19) == 19
    assert plan.kv_cache_owner(3) == 3

    # Check exact parameter names & shapes on meta device (zero memory cost)
    with torch.device("meta"):
        attn_prod = Qwen4_5Attention.from_config(cfg, layer_idx=19, sharing_plan=plan)
        attn_cons = Qwen4_5Attention.from_config(cfg, layer_idx=23, sharing_plan=plan)
        gdn_src = Qwen4_5GatedDeltaNet.from_config(cfg, layer_idx=20, sharing_plan=plan)
        gdn_cons = Qwen4_5GatedDeltaNet.from_config(cfg, layer_idx=21, sharing_plan=plan)

    prod_sd = {k: tuple(v.shape) for k, v in attn_prod.named_parameters()}
    cons_sd = {k: tuple(v.shape) for k, v in attn_cons.named_parameters()}
    assert prod_sd == {
        "q_proj.weight": (16384, 5120),
        "k_proj.weight": (256, 5120),
        "v_proj.weight": (256, 5120),
        "o_proj.weight": (5120, 8192),
        "q_norm.weight": (256,),
        "k_norm.weight": (256,),
        "k_sconv.weight": (256, 1, 4),
        "v_sconv.weight": (256, 1, 4),
    }, f"Unexpected attn_prod params: {prod_sd}"
    assert cons_sd == {
        "q_proj.weight": (16384, 5120),
        "o_proj.weight": (5120, 8192),
        "q_norm.weight": (256,),
    }, f"Unexpected attn_cons params: {cons_sd}"

    gdn_src_sd = {k: tuple(v.shape) for k, v in gdn_src.named_parameters()}
    gdn_cons_sd = {k: tuple(v.shape) for k, v in gdn_cons.named_parameters()}
    assert gdn_src_sd == {
        "in_proj_qkvz.weight": (20480, 5120),
        "in_proj_ba.weight": (256, 5120),
        "conv1d.weight": (12288, 1, 4),
        "A_log": (128,),
        "dt_bias": (128,),
        "norm.weight": (64,),
        "out_proj.weight": (5120, 8192),
    }, f"Unexpected gdn_src params: {gdn_src_sd}"
    assert gdn_cons_sd == {
        "in_proj_qkvz.weight": (10240, 5120),
        "conv1d.weight": (2048, 1, 4),
        "norm.weight": (64,),
        "out_proj.weight": (5120, 8192),
    }, f"Unexpected gdn_cons params: {gdn_cons_sd}"

    print("  PASS: YoCoSharingPlan topology & exact 1109 parameter shapes verified.")


@torch.no_grad()
def test_full_attention_producer_and_consumer() -> None:
    print("\n=== Test 1: Qwen4_5Attention Producer (L19) + Consumer (L23) ===")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(42)

    cfg = make_config(hidden_size=5120)
    plan = YoCoSharingPlan.from_config(cfg)

    prod = Qwen4_5Attention.from_config(cfg, layer_idx=19, sharing_plan=plan).to(device=device, dtype=dtype)
    cons = Qwen4_5Attention.from_config(cfg, layer_idx=23, sharing_plan=plan).to(device=device, dtype=dtype)
    init_realistic_weights(prod)
    init_realistic_weights(cons)

    # 1a. Verify fused GemmaRMSNorm + Partial NeoX RoPE against eager PyTorch
    x_rope = torch.randn((2, 16, 32, 256), device=device, dtype=dtype)
    pos_rope = torch.arange(16, device=device).unsqueeze(0).expand(2, 16).contiguous()
    out_fused = gemma_rmsnorm_and_partial_rope(
        x_rope, prod.q_norm.weight, pos_rope, prod.inv_freq, prod.rotary_dim, eps=cfg.rms_norm_eps
    )
    # Compare against exact FP32 math golden (norm+rope in FP32 -> BF16) AND eager BF16-intermediate
    x_f = x_rope.float()
    norm_f = (x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + cfg.rms_norm_eps)) * (1.0 + prod.q_norm.weight.float())
    out_fp32_ref = apply_partial_neox_rope(norm_f, pos_rope, prod.inv_freq, prod.rotary_dim).to(dtype)
    max_abs_fp32, rel_fp32 = report_diff("Fused GemmaRMSNorm+PartialRoPE vs FP32 golden", out_fused, out_fp32_ref)
    assert max_abs_fp32 <= 1e-5 or rel_fp32 < 1e-2, f"Fused norm+rope FP32 mismatch: {max_abs_fp32} (rel={rel_fp32})"
    out_ref = apply_partial_neox_rope(prod.q_norm(x_rope), pos_rope, prod.inv_freq, prod.rotary_dim)
    _, rel_eager = report_diff("Fused GemmaRMSNorm+PartialRoPE vs eager BF16", out_fused, out_ref)
    assert rel_eager < 1.5e-2, f"Fused norm+rope eager mismatch: rel={rel_eager}"

    # 1b. Consumer NEVER allocates KV cache or sconv states
    assert cons.create_kv_cache(2, 32, device, dtype) is None
    assert cons.create_sconv_states(2, device, dtype) == (None, None)

    B, T_total, T_pre = 2, 16, 10
    h_prod = torch.randn((B, T_total, cfg.hidden_size), device=device, dtype=dtype)
    h_cons = torch.randn((B, T_total, cfg.hidden_size), device=device, dtype=dtype)

    # Full Prefill(T=16)
    kv_full = prod.create_kv_cache(B, 32, device, dtype)
    ks_full, vs_full = prod.create_sconv_states(B, device, dtype)
    out_prod_full = prod(h_prod, kv_cache=kv_full, k_sconv_state=ks_full, v_sconv_state=vs_full)
    assert kv_full.seq_len == T_total

    # Snapshot KV cache before consumer runs to prove zero mutation
    k_snap = kv_full.k_cache.clone()
    v_snap = kv_full.v_cache.clone()
    seq_snap = kv_full.seq_len
    out_cons_full = cons(h_cons, kv_cache=kv_full)
    assert kv_full.seq_len == seq_snap, "Consumer mutated kv_cache.seq_len!"
    assert torch.equal(kv_full.k_cache, k_snap), "Consumer mutated kv_cache.k_cache!"
    assert torch.equal(kv_full.v_cache, v_snap), "Consumer mutated kv_cache.v_cache!"

    # Prefill(T=10) + 6 x Decode(1)
    kv_step = prod.create_kv_cache(B, 32, device, dtype)
    ks_step, vs_step = prod.create_sconv_states(B, device, dtype)
    out_prod_chunks = [
        prod(h_prod[:, :T_pre], kv_cache=kv_step, k_sconv_state=ks_step, v_sconv_state=vs_step)
    ]
    out_cons_chunks = [cons(h_cons[:, :T_pre], kv_cache=kv_step)]

    for t in range(T_pre, T_total):
        o_p = prod(h_prod[:, t : t + 1], kv_cache=kv_step, k_sconv_state=ks_step, v_sconv_state=vs_step)
        k_before = kv_step.k_cache.clone()
        v_before = kv_step.v_cache.clone()
        o_c = cons(h_cons[:, t : t + 1], kv_cache=kv_step)
        assert kv_step.seq_len == t + 1
        assert torch.equal(kv_step.k_cache, k_before)
        assert torch.equal(kv_step.v_cache, v_before)
        out_prod_chunks.append(o_p)
        out_cons_chunks.append(o_c)

    out_prod_step = torch.cat(out_prod_chunks, dim=1)
    out_cons_step = torch.cat(out_cons_chunks, dim=1)

    d_prod, _ = report_diff("FullAttn Producer Prefill(16) vs Prefill(10)+6xDecode(1)", out_prod_step, out_prod_full)
    d_cons, _ = report_diff("FullAttn Consumer Prefill(16) vs Prefill(10)+6xDecode(1)", out_cons_step, out_cons_full)
    assert d_prod < 1.5e-2, f"Producer prefill/decode mismatch: {d_prod}"
    assert d_cons < 1.5e-2, f"Consumer prefill/decode mismatch: {d_cons}"

    # 1c. Chunked Prefill(T=10) + Chunked Prefill(T=6)
    kv_chk = prod.create_kv_cache(B, 32, device, dtype)
    ks_chk, vs_chk = prod.create_sconv_states(B, device, dtype)
    p0 = prod(h_prod[:, :T_pre], kv_cache=kv_chk, k_sconv_state=ks_chk, v_sconv_state=vs_chk)
    c0 = cons(h_cons[:, :T_pre], kv_cache=kv_chk)
    p1 = prod(h_prod[:, T_pre:], kv_cache=kv_chk, k_sconv_state=ks_chk, v_sconv_state=vs_chk)
    c1 = cons(h_cons[:, T_pre:], kv_cache=kv_chk)
    d_prod_chk, _ = report_diff("FullAttn Producer ChunkedPrefill(10+6) vs Prefill(16)", torch.cat([p0, p1], dim=1), out_prod_full)
    d_cons_chk, _ = report_diff("FullAttn Consumer ChunkedPrefill(10+6) vs Prefill(16)", torch.cat([c0, c1], dim=1), out_cons_full)
    assert d_prod_chk < 1.5e-2
    assert d_cons_chk < 1.5e-2

    print("  PASS: FullAttn Producer + Consumer prefill/decode/chunked-prefill & zero-KV-mutation verified.")


@torch.no_grad()
def test_gdn_kernels_and_module() -> None:
    print("\n=== Test 2: GDN Triton Chunk Prefill vs Triton Recurrent vs PyTorch FP32 Ref ===")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(123)

    # 2a. Standalone GDN kernels on T=128 (multiple of 64) and T=100 (non-multiple) + non-zero init state
    for seq_len, has_init in [(128, False), (100, False), (100, True)]:
        B, H_k, H_v, D_k, D_v = 2, 32, 128, 64, 64
        q = torch.randn((B, seq_len, H_k, D_k), device=device, dtype=dtype)
        k = torch.randn((B, seq_len, H_k, D_k), device=device, dtype=dtype)
        v = torch.randn((B, seq_len, H_v, D_v), device=device, dtype=dtype)
        g = -F.softplus(torch.randn((B, seq_len, H_v), device=device, dtype=torch.float32) - 1.5)
        beta = torch.sigmoid(torch.randn((B, seq_len, H_v), device=device, dtype=torch.float32))
        init_state = (
            torch.randn((B, H_v, D_k, D_v), device=device, dtype=torch.float32) * 0.1
            if has_init
            else None
        )

        out_ref, st_ref = gdn_recurrent_ref(q, k, v, g, beta, initial_state=init_state, output_final_state=True)
        out_rec, st_rec = gdn_recurrent_triton(q, k, v, g, beta, initial_state=init_state, output_final_state=True)
        out_chk, st_chk = gdn_chunk_prefill_triton(q, k, v, g, beta, initial_state=init_state, output_final_state=True)

        tag = f"T={seq_len},init={has_init}"
        d_rec_o, _ = report_diff(f"GDN Recurrent Triton vs Ref ({tag}) out", out_rec, out_ref)
        d_rec_s, _ = report_diff(f"GDN Recurrent Triton vs Ref ({tag}) state", st_rec, st_ref)
        d_chk_o, _ = report_diff(f"GDN Chunk Triton vs Ref ({tag}) out", out_chk, out_ref)
        d_chk_s, _ = report_diff(f"GDN Chunk Triton vs Ref ({tag}) state", st_chk, st_ref)
        d_chk_rec, _ = report_diff(f"GDN Chunk vs Recurrent Triton ({tag}) out", out_chk, out_rec)

        assert d_rec_o < 1e-2, f"Recurrent out mismatch ({tag}): {d_rec_o}"
        assert d_rec_s < 1e-4, f"Recurrent state mismatch ({tag}): {d_rec_s}"
        assert d_chk_o < 1e-2, f"Chunk out mismatch ({tag}): {d_chk_o}"
        assert d_chk_s < 1e-2, f"Chunk state mismatch ({tag}): {d_chk_s}"
        assert d_chk_rec < 1e-2, f"Chunk vs recurrent mismatch ({tag}): {d_chk_rec}"

    # 2b. Verify fused compute_gdn_gating & RMSNormGated kernels against PyTorch
    ba = torch.randn((2, 16, 256), device=device, dtype=dtype)
    a_log = torch.randn((128,), device=device, dtype=dtype)
    dt_bias = torch.randn((128,), device=device, dtype=dtype)
    g_tri, beta_tri = compute_gdn_gating(ba, a_log, dt_bias, num_k_heads=32, num_v_heads=128)
    b_g, a_g = ba.unflatten(-1, (32, 8)).split([4, 4], dim=-1)
    g_ref = -torch.exp(a_log.float()) * F.softplus(a_g.reshape(2, 16, 128).float() + dt_bias.float())
    beta_ref = torch.sigmoid(b_g.reshape(2, 16, 128).float())
    dg, _ = report_diff("compute_gdn_gating g vs ref", g_tri, g_ref)
    db, _ = report_diff("compute_gdn_gating beta vs ref", beta_tri, beta_ref)
    assert dg < 1e-4 and db < 1e-5

    norm_mod = RMSNormGated(64, eps=1e-6, activation="sigmoid", norm_before_gate=True).to(device=device, dtype=dtype)
    norm_mod.weight.data.normal_(1.0, 0.1)
    nx = torch.randn((2, 16, 128, 64), device=device, dtype=dtype)
    nz = torch.randn((2, 16, 128, 64), device=device, dtype=dtype)
    ny_tri = norm_mod(nx, nz)
    nx_f = nx.float()
    ny_ref = ((nx_f * torch.rsqrt(nx_f.pow(2).mean(dim=-1, keepdim=True) + 1e-6)) * norm_mod.weight.float() * torch.sigmoid(nz.float())).to(dtype)
    dn, _ = report_diff("RMSNormGated Triton vs ref", ny_tri, ny_ref)
    assert dn < 1.5e-2

    # 2c. Full Qwen4_5GatedDeltaNet module: Prefill(T=128) == Prefill(T=100) + 28 x Decode(1)
    cfg = make_config(hidden_size=5120)
    plan = YoCoSharingPlan.from_config(cfg)
    gdn = Qwen4_5GatedDeltaNet.from_config(cfg, layer_idx=0, sharing_plan=plan).to(device=device, dtype=dtype)
    init_realistic_weights(gdn)

    B, T_total, T_pre = 2, 128, 100
    h = torch.randn((B, T_total, cfg.hidden_size), device=device, dtype=dtype)

    st_full = gdn.create_state(B, device, dtype)
    out_full, pub_full = gdn(h, state=st_full)
    assert pub_full is None

    st_step = gdn.create_state(B, device, dtype)
    chunks = [gdn(h[:, :T_pre], state=st_step)[0]]
    for t in range(T_pre, T_total):
        chunks.append(gdn(h[:, t : t + 1], state=st_step)[0])
    out_step = torch.cat(chunks, dim=1)

    d_mod_o, r_mod_o = report_diff("GDN Module Prefill(128) vs Prefill(100)+28xDecode(1) out", out_step, out_full)
    d_mod_s, r_mod_s = report_diff("GDN Module Prefill(128) vs Prefill(100)+28xDecode(1) ssm_state", st_step.ssm_state, st_full.ssm_state)
    d_mod_c, r_mod_c = report_diff("GDN Module Prefill(128) vs Prefill(100)+28xDecode(1) conv_state", st_step.conv_state, st_full.conv_state)
    assert d_mod_o <= 2e-2 and r_mod_o < 1e-2, f"GDN module out mismatch: abs={d_mod_o}, rel={r_mod_o}"
    assert d_mod_s < 1.5e-2 and r_mod_s < 1.5e-2, f"GDN module state mismatch: abs={d_mod_s}, rel={r_mod_s}"
    assert d_mod_c <= 2e-2 and r_mod_c < 1e-2, f"GDN conv state mismatch: abs={d_mod_c}, rel={r_mod_c}"

    print("  PASS: GDN Triton kernels & full module Prefill(128) == Prefill(100)+28xDecode(1) verified.")


@torch.no_grad()
def test_yoco_shared_gdn_source_and_consumer() -> None:
    print("\n=== Test 3: YoCo Shared GDN Source (L20) + Consumer (L21) ===")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(999)

    # 3a. Standalone yoco_gdn_decode_readout Triton vs exact FP32 einsum ref
    q_test = torch.randn((4, 1, 32, 64), device=device, dtype=dtype)
    s_test = torch.randn((4, 128, 64, 64), device=device, dtype=torch.float32) * 0.2
    ro_tri = yoco_gdn_decode_readout(q_test, s_test, head_k_dim=64)
    ro_ref = yoco_gdn_decode_readout_ref(q_test, s_test, head_k_dim=64)
    d_ro, _ = report_diff("yoco_gdn_decode_readout Triton vs einsum ref", ro_tri, ro_ref)
    assert d_ro < 1e-3, f"YoCo readout mismatch: {d_ro}"

    # 3b. End-to-end Source (Layer 20) + Consumer (Layer 21)
    cfg = make_config(hidden_size=5120)
    plan = YoCoSharingPlan.from_config(cfg)
    src = Qwen4_5GatedDeltaNet.from_config(cfg, layer_idx=20, sharing_plan=plan).to(device=device, dtype=dtype)
    cons = Qwen4_5GatedDeltaNet.from_config(cfg, layer_idx=21, sharing_plan=plan).to(device=device, dtype=dtype)
    assert src.publishes_shared and not src.is_consumer
    assert cons.is_consumer and not cons.publishes_shared
    init_realistic_weights(src)
    init_realistic_weights(cons)

    B, T_total, T_pre = 2, 128, 100
    h_src = torch.randn((B, T_total, cfg.hidden_size), device=device, dtype=dtype)
    h_cons = torch.randn((B, T_total, cfg.hidden_size), device=device, dtype=dtype)

    # Full Prefill(T=128)
    st_src_full = src.create_state(B, device, dtype)
    st_cons_full = cons.create_state(B, device, dtype)
    assert st_cons_full.ssm_state is None, "YoCo-decode consumer should NOT allocate ssm_state!"

    out_src_full, shared_full = src(h_src, state=st_src_full)
    assert isinstance(shared_full, SharedGDNInputs)
    assert shared_full.key.shape == (B, T_total, 32, 64)
    assert shared_full.value.shape == (B, T_total, 128, 64)
    assert shared_full.g.shape == (B, T_total, 128) and shared_full.g.dtype == torch.float32
    assert shared_full.beta.shape == (B, T_total, 128) and shared_full.beta.dtype == torch.float32
    assert shared_full.ssm_state is not None and shared_full.ssm_state.shape == (B, 128, 64, 64)
    assert shared_full.ssm_state.dtype == torch.float32

    out_cons_full, pub_cons = cons(h_cons, state=st_cons_full, shared=shared_full)
    assert pub_cons is None
    assert st_cons_full.ssm_state is None

    # Prefill(T=100) + 28 x Decode(1) with YoCo Decode Readout on Consumer
    st_src_step = src.create_state(B, device, dtype)
    st_cons_step = cons.create_state(B, device, dtype)
    assert st_cons_step.ssm_state is None

    o_s0, sh_0 = src(h_src[:, :T_pre], state=st_src_step)
    o_c0, _ = cons(h_cons[:, :T_pre], state=st_cons_step, shared=sh_0)
    src_chunks = [o_s0]
    cons_chunks = [o_c0]

    max_einsum_diff = 0.0
    for t in range(T_pre, T_total):
        o_st, sh_t = src(h_src[:, t : t + 1], state=st_src_step)

        # Independently verify exact einsum readout formula against Layer 20's updated FP32 SSM state
        conv_snap = st_cons_step.conv_state.clone()
        o_ct, _ = cons(h_cons[:, t : t + 1], state=st_cons_step, shared=sh_t)
        assert st_cons_step.ssm_state is None, "Consumer allocated ssm_state during YoCo decode!"

        # Manual exact formula check on this decode step
        qz = cons.in_proj_qkvz(h_cons[:, t : t + 1])
        q_flat, z_step = cons._unpack_qz_consumer(qz)
        x_pad = torch.cat([conv_snap, q_flat.transpose(1, 2)], dim=-1)
        q_conv = F.silu(F.conv1d(x_pad, cons.conv1d.weight, groups=cons.conv_dim)).transpose(1, 2)
        q_pre_l2 = q_conv.reshape(B, 1, 32, 64)

        q_l2 = q_pre_l2.float() * torch.rsqrt(q_pre_l2.float().square().sum(dim=-1, keepdim=True) + 1e-6)
        q_rep = q_l2.repeat_interleave(4, dim=-2)
        core_einsum = (
            torch.einsum("bhkv,bthk->bthv", st_src_step.ssm_state.float(), q_rep) * (64 ** -0.5)
        ).to(dtype)
        # Compare core readout directly
        core_yoco = yoco_gdn_decode_readout(q_pre_l2, st_src_step.ssm_state, head_k_dim=64)
        step_ro_diff = (core_yoco.float() - core_einsum.float()).abs().max().item()
        max_einsum_diff = max(max_einsum_diff, step_ro_diff)

        # Also compare full consumer layer output against manual einsum -> norm -> out_proj
        manual_out = cons.out_proj(cons.norm(core_einsum, z_step).reshape(B, 1, cons.value_dim))
        manual_diff = (o_ct.float() - manual_out.float()).abs().max().item()
        assert manual_diff < 1e-2, f"Consumer decode layer mismatch vs manual einsum: {manual_diff}"

        src_chunks.append(o_st)
        cons_chunks.append(o_ct)

    out_src_step = torch.cat(src_chunks, dim=1)
    out_cons_step = torch.cat(cons_chunks, dim=1)

    print(f"  [DIFF] YoCo Decode core readout vs exact einsum (max over 28 steps): {max_einsum_diff:.6e}")
    assert max_einsum_diff < 1e-3, f"YoCo einsum readout diff too high: {max_einsum_diff}"

    d_src, r_src = report_diff("YoCo GDN Source(L20) Prefill(128) vs Prefill(100)+28xDecode(1)", out_src_step, out_src_full)
    d_cons, r_cons = report_diff("YoCo GDN Consumer(L21) Prefill(128) vs Prefill(100)+28xYoCoDecode(1)", out_cons_step, out_cons_full)
    assert d_src <= 2e-2 and r_src < 1e-2, f"Source L20 mismatch: abs={d_src}, rel={r_src}"
    assert d_cons <= 2e-2 and r_cons < 1e-2, f"Consumer L21 mismatch: abs={d_cons}, rel={r_cons}"

    print("  PASS: YoCo Shared GDN Source (L20) + Consumer (L21) Prefill & Decode readout verified.")




@torch.no_grad()
def _build_ragged_meta_for_test(
    lengths: list[int],
    *,
    block_t: int = 32,
    chunk_size: int = 64,
    tpb: int = 64,
    device: torch.device,
):
    cu_list = [0]
    for l in lengths:
        cu_list.append(cu_list[-1] + int(l))
    cu = torch.tensor(cu_list, device=device, dtype=torch.int32)
    t_req, t_t0 = [], []
    c_req, c_t0 = [], []
    req_c_start = [0]
    snap_tok, snap_req, snap_phys = [], [], []
    req_snap_start = [0]
    snap_local_end = []
    snap_meta = []  # (bi, bnd, phys)
    phys = 1
    for bi, l in enumerate(lengths):
        for t0 in range(0, l, block_t):
            t_req.append(bi)
            t_t0.append(t0)
        for c0 in range(0, l, chunk_size):
            c_req.append(bi)
            c_t0.append(c0)
        req_c_start.append(len(c_req))
        for bnd in range(tpb, l + 1, tpb):
            snap_tok.append(cu_list[bi] + bnd - 1)
            snap_req.append(bi)
            snap_local_end.append(bnd)
            snap_phys.append(phys)
            snap_meta.append((bi, bnd, phys))
            phys += 1
        req_snap_start.append(len(snap_local_end))
    return (
        cu_list,
        cu,
        torch.tensor(t_req, device=device, dtype=torch.int32),
        torch.tensor(t_t0, device=device, dtype=torch.int32),
        torch.tensor(c_req, device=device, dtype=torch.int32),
        torch.tensor(c_t0, device=device, dtype=torch.int32),
        torch.tensor(req_c_start, device=device, dtype=torch.int32),
        torch.tensor(snap_tok, device=device, dtype=torch.int32),
        torch.tensor(snap_req, device=device, dtype=torch.int32),
        torch.tensor(req_snap_start, device=device, dtype=torch.int32),
        torch.tensor(snap_local_end, device=device, dtype=torch.int32),
        torch.tensor(snap_phys, device=device, dtype=torch.int32),
        snap_meta,
        phys + 1,
    )


@torch.no_grad()
def test_gdn_fused_unpack_verify_ragged_and_tp() -> None:
    print("\n=== Test 4: GDN Packed-Ragged Snapshot Prefill, V=8 Verify (+CG/Rollback) & TP=4 Hooks ===")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(2028)

    cfg = make_config(hidden_size=5120)
    plan = YoCoSharingPlan.from_config(cfg)
    gdn_norm = Qwen4_5GatedDeltaNet.from_config(cfg, layer_idx=0, sharing_plan=plan).to(device=device, dtype=dtype).eval()
    gdn_src = Qwen4_5GatedDeltaNet.from_config(cfg, layer_idx=20, sharing_plan=plan).to(device=device, dtype=dtype).eval()
    gdn_cons = Qwen4_5GatedDeltaNet.from_config(cfg, layer_idx=21, sharing_plan=plan).to(device=device, dtype=dtype).eval()
    for m in (gdn_norm, gdn_src, gdn_cons):
        init_realistic_weights(m)

    # 4a. Ragged forward_packed_prefill (B=3, lens=[64, 130, 75]) + 64-token SSM & conv snapshots
    lens = [64, 130, 75]
    (
        cu_list, cu, t_req, t_t0, c_req, c_t0, r_c_start,
        s_tok, s_req, r_s_start, s_end, s_phys, snap_meta, num_pool_blks,
    ) = _build_ragged_meta_for_test(lens, block_t=32, chunk_size=64, tpb=64, device=device)
    B = len(lens)
    N = cu_list[-1]
    h_src = torch.randn((N, cfg.hidden_size), device=device, dtype=dtype)
    h_cons = torch.randn((N, cfg.hidden_size), device=device, dtype=dtype)
    init_conv_s = torch.randn((B, gdn_src.conv_dim, 3), device=device, dtype=dtype)
    init_ssm_s = torch.randn((B, gdn_src.num_v_heads, 64, 64), device=device, dtype=torch.float32) * 0.05
    init_conv_c = torch.randn((B, gdn_cons.conv_dim, 3), device=device, dtype=dtype)

    conv_pool_s = torch.zeros((num_pool_blks, gdn_src.conv_dim * 3), device=device, dtype=dtype)
    ssm_pool_s = torch.zeros((num_pool_blks, gdn_src.num_v_heads, 64, 64), device=device, dtype=torch.float32)
    conv_pool_c = torch.zeros((num_pool_blks, gdn_cons.conv_dim * 3), device=device, dtype=dtype)

    out_src_rag, sh_rag = gdn_src.forward_packed_prefill(
        h_src, cu, t_req, t_t0, c_req, c_t0, r_c_start,
        init_conv_state=init_conv_s, init_ssm_state=init_ssm_s,
        snap_tok_idx=s_tok, snap_req_idx=s_req,
        conv_dst_pool=conv_pool_s, conv_snap_phys=s_phys,
        req_snap_start=r_s_start, snap_local_end=s_end,
        ssm_dst_pool=ssm_pool_s, ssm_snap_phys=s_phys,
    )
    assert sh_rag is not None
    out_cons_rag, _ = gdn_cons.forward_packed_prefill(
        h_cons, cu, t_req, t_t0, c_req, c_t0, r_c_start,
        init_conv_state=init_conv_c, init_ssm_state=init_ssm_s, shared=sh_rag,
        snap_tok_idx=s_tok, snap_req_idx=s_req,
        conv_dst_pool=conv_pool_c, conv_snap_phys=s_phys,
        req_snap_start=r_s_start, snap_local_end=s_end,
    )
    for bi, l in enumerate(lens):
        st_s = gdn_src.create_state(1, device, dtype)
        st_s.conv_state.copy_(init_conv_s[bi : bi + 1])
        st_s.ssm_state.copy_(init_ssm_s[bi : bi + 1])
        ys_ref, sh_ref = gdn_src(h_src[cu_list[bi] : cu_list[bi + 1]].unsqueeze(0), state=st_s)
        ds, _ = report_diff(f"GDN Source ragged req={bi}", out_src_rag[cu_list[bi] : cu_list[bi + 1]], ys_ref.squeeze(0))
        assert ds < 1e-2

        st_c = gdn_cons.create_state(1, device, dtype)
        st_c.conv_state.copy_(init_conv_c[bi : bi + 1])
        st_c.ssm_state = init_ssm_s[bi : bi + 1].clone()
        yc_ref, _ = gdn_cons(h_cons[cu_list[bi] : cu_list[bi + 1]].unsqueeze(0), state=st_c, shared=sh_ref)
        dc, _ = report_diff(f"GDN Consumer ragged req={bi}", out_cons_rag[cu_list[bi] : cu_list[bi + 1]], yc_ref.squeeze(0))
        assert dc < 1e-2

    for bi, bnd, phys in snap_meta:
        st_bnd = gdn_src.create_state(1, device, dtype)
        st_bnd.conv_state.copy_(init_conv_s[bi : bi + 1])
        st_bnd.ssm_state.copy_(init_ssm_s[bi : bi + 1])
        _ = gdn_src(h_src[cu_list[bi] : cu_list[bi] + bnd].unsqueeze(0), state=st_bnd)
        d_conv_s, _ = report_diff(f"GDN Src conv snap req={bi} bnd={bnd}", conv_pool_s[phys].view(gdn_src.conv_dim, 3), st_bnd.conv_state[0])
        d_ssm_s, _ = report_diff(f"GDN Src ssm snap req={bi} bnd={bnd}", ssm_pool_s[phys], st_bnd.ssm_state[0])
        assert d_conv_s == 0.0 and d_ssm_s < 1e-3

        st_cbnd = gdn_cons.create_state(1, device, dtype)
        st_cbnd.conv_state.copy_(init_conv_c[bi : bi + 1])
        # Only verify consumer conv snapshot
        st_cbnd_tmp = gdn_src.create_state(1, device, dtype)
        st_cbnd_tmp.conv_state.copy_(init_conv_s[bi : bi + 1])
        st_cbnd_tmp.ssm_state.copy_(init_ssm_s[bi : bi + 1])
        _, sh_b = gdn_src(h_src[cu_list[bi] : cu_list[bi] + bnd].unsqueeze(0), state=st_cbnd_tmp)
        st_cbnd.ssm_state = init_ssm_s[bi : bi + 1].clone()
        _ = gdn_cons(h_cons[cu_list[bi] : cu_list[bi] + bnd].unsqueeze(0), state=st_cbnd, shared=sh_b)
        d_conv_c, _ = report_diff(f"GDN Cons conv snap req={bi} bnd={bnd}", conv_pool_c[phys].view(gdn_cons.conv_dim, 3), st_cbnd.conv_state[0])
        assert d_conv_c == 0.0

    # 4b. GDN forward_verify (B=3, V=8) + rollback states + CUDA Graph capture/replay (Source L20 + Consumer L21 + Normal L0)
    Bv, V = 3, 8
    hv_s = torch.randn((Bv, V, cfg.hidden_size), device=device, dtype=dtype)
    hv_c = torch.randn((Bv, V, cfg.hidden_size), device=device, dtype=dtype)
    ic_s = torch.randn((Bv, gdn_src.conv_dim, 3), device=device, dtype=dtype)
    is_s = torch.randn((Bv, gdn_src.num_v_heads, 64, 64), device=device, dtype=torch.float32) * 0.05
    ic_c = torch.randn((Bv, gdn_cons.conv_dim, 3), device=device, dtype=dtype)
    sc_s = torch.empty((Bv, V, gdn_src.conv_dim, 3), device=device, dtype=dtype)
    ss_s = torch.empty((Bv, V, gdn_src.num_v_heads, 64, 64), device=device, dtype=torch.float32)
    sc_c = torch.empty((Bv, V, gdn_cons.conv_dim, 3), device=device, dtype=dtype)

    ov_s, shv = gdn_src.forward_verify(hv_s, init_conv_state=ic_s, init_ssm_state=is_s, stage_conv_state=sc_s, stage_ssm_state=ss_s)
    assert shv is not None and shv.ssm_state is not None
    ov_c, _ = gdn_cons.forward_verify(hv_c, init_conv_state=ic_c, stage_conv_state=sc_c, shared=shv, enable_yoco_decode=True)

    st_step_s = gdn_src.create_state(Bv, device, dtype)
    st_step_s.conv_state.copy_(ic_s)
    st_step_s.ssm_state.copy_(is_s)
    st_step_c = gdn_cons.create_state(Bv, device, dtype)
    st_step_c.conv_state.copy_(ic_c)
    step_s_outs, step_c_outs = [], []
    for s in range(V):
        ys, shs = gdn_src(hv_s[:, s : s + 1], state=st_step_s)
        yc, _ = gdn_cons(hv_c[:, s : s + 1], state=st_step_c, shared=shs, enable_yoco_decode=True)
        step_s_outs.append(ys)
        step_c_outs.append(yc)
        assert (sc_s[:, s] - st_step_s.conv_state).abs().max().item() <= 2e-2
        assert (ss_s[:, s] - st_step_s.ssm_state).abs().max().item() < 1.5e-2
        assert (sc_c[:, s] - st_step_c.conv_state).abs().max().item() <= 2e-2

    d_vs, _ = report_diff("GDN Source forward_verify vs 8x decode", ov_s, torch.cat(step_s_outs, dim=1))
    d_vc, _ = report_diff("GDN Consumer forward_verify vs 8x YoCo decode", ov_c, torch.cat(step_c_outs, dim=1))
    assert d_vs < 1e-2 and d_vc < 1e-2

    # CUDA Graph replay for forward_verify (Source + Consumer)
    shs_g, shc_g = hv_s.clone(), hv_c.clone()
    ics_g, iss_g, icc_g = ic_s.clone(), is_s.clone(), ic_c.clone()
    scs_g, sss_g, scc_g = torch.empty_like(sc_s), torch.empty_like(ss_s), torch.empty_like(sc_c)
    _, _sh_w = gdn_src.forward_verify(shs_g, init_conv_state=ics_g, init_ssm_state=iss_g, stage_conv_state=scs_g, stage_ssm_state=sss_g)
    _ = gdn_cons.forward_verify(shc_g, init_conv_state=icc_g, stage_conv_state=scc_g, shared=_sh_w, enable_yoco_decode=True)
    torch.cuda.synchronize()
    cg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cg):
        cg_os, cg_sh = gdn_src.forward_verify(shs_g, init_conv_state=ics_g, init_ssm_state=iss_g, stage_conv_state=scs_g, stage_ssm_state=sss_g)
        cg_oc, _ = gdn_cons.forward_verify(shc_g, init_conv_state=icc_g, stage_conv_state=scc_g, shared=cg_sh, enable_yoco_decode=True)
    shs_g.normal_(0.0, 0.5)
    shc_g.normal_(0.0, 0.5)
    cg.replay()
    torch.cuda.synchronize()
    cg_os_chk, cg_oc_chk = cg_os.clone(), cg_oc.clone()
    eg_scs, eg_sss, eg_scc = torch.empty_like(sc_s), torch.empty_like(ss_s), torch.empty_like(sc_c)
    eg_os, eg_sh = gdn_src.forward_verify(shs_g, init_conv_state=ics_g, init_ssm_state=iss_g, stage_conv_state=eg_scs, stage_ssm_state=eg_sss)
    eg_oc, _ = gdn_cons.forward_verify(shc_g, init_conv_state=icc_g, stage_conv_state=eg_scc, shared=eg_sh, enable_yoco_decode=True)
    d_cgs, _ = report_diff("GDN Source Verify CG vs Eager", cg_os_chk, eg_os)
    d_cgc, _ = report_diff("GDN Consumer Verify CG vs Eager", cg_oc_chk, eg_oc)
    assert d_cgs < 1e-4 and d_cgc < 1e-4

    # 4c. TP=4 + fast_all_reduce Hook Parity for Qwen4_5Attention (MQA-1 L19) & Qwen4_5GatedDeltaNet (L0)
    cfg_tp = Qwen4_5ArchConfig(
        hidden_size=512, num_attention_heads=8, num_key_value_heads=1, head_dim=64,
        linear_num_key_heads=4, linear_num_value_heads=8, linear_key_head_dim=64, linear_value_head_dim=64,
        share_map=QWEN45_SHARE_MAP,
    )
    attn_tp1 = Qwen4_5Attention.from_config(cfg_tp, layer_idx=19, tp_size=1).to(device=device, dtype=dtype).eval()
    init_realistic_weights(attn_tp1)
    xt = torch.randn((2, 16, 512), device=device, dtype=dtype)
    post = torch.arange(16, device=device).unsqueeze(0).expand(2, -1)
    kv1 = attn_tp1.create_kv_cache(2, 32, device, dtype)
    ks1, vs1 = attn_tp1.create_sconv_states(2, device, dtype)
    oa1 = attn_tp1(xt, positions=post, kv_cache=kv1, k_sconv_state=ks1, v_sconv_state=vs1, seq_offset=0)
    oa_sum = torch.zeros_like(oa1)
    hq = (cfg_tp.num_attention_heads // 4) * cfg_tp.head_dim
    hq_out = hq * (2 if cfg_tp.attn_output_gate else 1)
    for r in range(4):
        ar_hits = [0]
        ar_r = Qwen4_5Attention.from_config(
            cfg_tp, layer_idx=19, tp_size=4, tp_rank=r,
            fast_all_reduce=lambda t, _h=ar_hits: (_h.__setitem__(0, _h[0] + 1) or t),
        ).to(device=device, dtype=dtype).eval()
        ar_r.q_proj.weight.copy_(attn_tp1.q_proj.weight[r * hq_out : (r + 1) * hq_out])
        ar_r.o_proj.weight.copy_(attn_tp1.o_proj.weight[:, r * hq : (r + 1) * hq])
        ar_r.q_norm.weight.copy_(attn_tp1.q_norm.weight)
        ar_r.k_proj.weight.copy_(attn_tp1.k_proj.weight)
        ar_r.v_proj.weight.copy_(attn_tp1.v_proj.weight)
        ar_r.k_norm.weight.copy_(attn_tp1.k_norm.weight)
        if ar_r.k_sconv is not None: ar_r.k_sconv.weight.copy_(attn_tp1.k_sconv.weight)
        if ar_r.v_sconv is not None: ar_r.v_sconv.weight.copy_(attn_tp1.v_sconv.weight)
        kv_r = ar_r.create_kv_cache(2, 32, device, dtype)
        ks_r, vs_r = ar_r.create_sconv_states(2, device, dtype)
        oa_sum += ar_r(xt, positions=post, kv_cache=kv_r, k_sconv_state=ks_r, v_sconv_state=vs_r, seq_offset=0)
        assert ar_hits[0] == 1
    d_att_tp, _ = report_diff("Qwen4_5Attention L19 TP=4 vs TP=1", oa_sum, oa1)
    assert d_att_tp < 1e-2

    gdn_tp1 = Qwen4_5GatedDeltaNet.from_config(cfg_tp, layer_idx=0).to(device=device, dtype=dtype).eval()
    init_realistic_weights(gdn_tp1)
    og1, _ = gdn_tp1(xt, state=gdn_tp1.create_state(2, device, dtype))
    og_sum = torch.zeros_like(og1)
    gs1 = 2 * gdn_tp1.head_k_dim + 2 * gdn_tp1.gqa_ratio * gdn_tp1.head_v_dim
    lk = gdn_tp1.num_k_heads // 4
    lv = gdn_tp1.num_v_heads // 4
    q_s, k_s, v_s = gdn_tp1.conv1d.weight.split([gdn_tp1.key_dim, gdn_tp1.key_dim, gdn_tp1.value_dim], dim=0)
    lk_d, lv_d = gdn_tp1.key_dim // 4, gdn_tp1.value_dim // 4
    for r in range(4):
        ar_hits = [0]
        gr = Qwen4_5GatedDeltaNet.from_config(
            cfg_tp, layer_idx=0, tp_size=4, tp_rank=r,
            fast_all_reduce=lambda t, _h=ar_hits: (_h.__setitem__(0, _h[0] + 1) or t),
        ).to(device=device, dtype=dtype).eval()
        gr.in_proj_qkvz.weight.copy_(gdn_tp1.in_proj_qkvz.weight[r * lk * gs1 : (r + 1) * lk * gs1])
        gr.in_proj_ba.weight.copy_(gdn_tp1.in_proj_ba.weight[r * lk * 2 * gdn_tp1.gqa_ratio : (r + 1) * lk * 2 * gdn_tp1.gqa_ratio])
        gr.conv1d.weight.copy_(torch.cat([q_s[r * lk_d : (r + 1) * lk_d], k_s[r * lk_d : (r + 1) * lk_d], v_s[r * lv_d : (r + 1) * lv_d]], dim=0))
        gr.A_log.copy_(gdn_tp1.A_log[r * lv : (r + 1) * lv])
        gr.dt_bias.copy_(gdn_tp1.dt_bias[r * lv : (r + 1) * lv])
        gr.norm.weight.copy_(gdn_tp1.norm.weight)
        gr.out_proj.weight.copy_(gdn_tp1.out_proj.weight[:, r * lv_d : (r + 1) * lv_d])
        or_g, _ = gr(xt, state=gr.create_state(2, device, dtype))
        assert ar_hits[0] == 1
        og_sum += or_g
    d_gdn_tp, _ = report_diff("Qwen4_5GatedDeltaNet L0 TP=4 vs TP=1", og_sum, og1)
    assert d_gdn_tp < 1e-2

    # R0 SP/DP/FP8 constructor hooks check
    attn_sp = Qwen4_5Attention.from_config(
        cfg_tp, layer_idx=19, tp_size=4, tp_rank=1, sp_size=4, sp_rank=2, dp_attn=True,
        fp8_indexer_kv=True, kv_cache_dtype="fp8_e4m3",
    )
    assert attn_sp.sp_size == 4 and attn_sp.sp_rank == 2 and attn_sp.dp_attn is True
    assert attn_sp.fp8_indexer_kv is True and attn_sp.kv_cache_dtype == "fp8_e4m3"
    gdn_sp = Qwen4_5GatedDeltaNet.from_config(
        cfg_tp, layer_idx=0, tp_size=4, tp_rank=1, sp_size=4, sp_rank=2, dp_attn=True,
    )
    assert gdn_sp.sp_size == 4 and gdn_sp.sp_rank == 2 and gdn_sp.dp_attn is True

    print("  PASS: GDN Ragged Snapshots, V=8 Verify (+Rollback+CG), Attention/GDN TP=4 & SP/DP/FP8 hooks verified.")


@torch.no_grad()
def test_r3_gdn_stage2_sub_dv_and_verify_nw1() -> None:
    print("[R3 Test] GDN Stage 2 BLOCK_DV=16 (num_chunks>=128) Snapshot Parity + VerifyScan nw=1...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    if device.type != "cuda":
        print("  SKIP (CUDA required)")
        return
    torch.manual_seed(202633)

    N, Hk, Hv = 8192, 16, 32
    cu_list, cu, _, _, c_req, c_t0, rcs, _, _, rss, sle, sp_phys, _, pool_sz = _build_ragged_meta_for_test([N], device=device)
    q = torch.randn((N, Hk, 64), device=device, dtype=dtype) * 0.2
    k = torch.randn((N, Hk, 64), device=device, dtype=dtype) * 0.2
    v = torch.randn((N, Hv, 64), device=device, dtype=dtype) * 0.2
    g = -torch.rand((N, Hv), device=device, dtype=torch.float32) * 0.2
    beta = torch.sigmoid(torch.randn((N, Hv), device=device, dtype=torch.float32))
    init = torch.randn((1, Hv, 64, 64), device=device, dtype=torch.float32) * 0.1
    pool_gpu = torch.zeros((pool_sz, Hv, 64, 64), device=device, dtype=torch.float32)
    out_gpu, fs_gpu = gdn_chunk_prefill_ragged_with_snapshots_triton(
        q, k, v, g, beta, cu, c_req, c_t0, rcs,
        initial_state=init, req_snap_start=rss, snap_local_end=sle, snap_phys=sp_phys,
        ssm_dst_pool=pool_gpu, output_final_states=True,
    )
    out_ref, fs_ref = gdn_chunk_prefill_triton(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), g.unsqueeze(0), beta.unsqueeze(0),
        initial_state=init, output_final_state=True,
    )
    assert F.cosine_similarity(out_gpu.float().flatten(), out_ref.squeeze(0).float().flatten(), dim=0).item() > 0.9999
    assert F.cosine_similarity(fs_gpu.float().flatten(), fs_ref.float().flatten(), dim=0).item() > 0.9999
    assert F.cosine_similarity(pool_gpu[int(sp_phys[-1])].flatten(), fs_gpu[0].flatten(), dim=0).item() > 0.99999

    cfg = make_config(2048)
    gdn = Qwen4_5GatedDeltaNet.from_config(cfg, layer_idx=0).to(device=device, dtype=dtype).eval()
    for bsz in (1, 8, 64):
        vw = 8
        vq = torch.randn((bsz, vw, gdn.num_k_heads, 64), device=device, dtype=dtype) * 0.2
        vk = torch.randn((bsz, vw, gdn.num_k_heads, 64), device=device, dtype=dtype) * 0.2
        vv = torch.randn((bsz, vw, gdn.num_v_heads, 64), device=device, dtype=dtype) * 0.2
        vg = -torch.rand((bsz, vw, gdn.num_v_heads), device=device, dtype=torch.float32) * 0.3
        vb = torch.sigmoid(torch.randn((bsz, vw, gdn.num_v_heads), device=device, dtype=torch.float32))
        s0 = torch.randn((bsz, gdn.num_v_heads, 64, 64), device=device, dtype=torch.float32) * 0.1
        st = torch.empty((bsz, vw, gdn.num_v_heads, 64, 64), device=device, dtype=torch.float32)
        vo = gdn._launch_gdn_verify_scan(vq, vk, vv, vg, vb, s0, st, verify_w=vw)
        cur = s0.clone()
        refs = []
        for s_i in range(vw):
            o_s, cur = gdn_recurrent_ref(vq[:, s_i : s_i + 1], vk[:, s_i : s_i + 1], vv[:, s_i : s_i + 1], vg[:, s_i : s_i + 1], vb[:, s_i : s_i + 1], initial_state=cur, output_final_state=True)
            refs.append(o_s.squeeze(1))
            assert F.cosine_similarity(st[:, s_i].flatten(), cur.flatten(), dim=0).item() > 0.99998
        assert F.cosine_similarity(vo.float().flatten(), torch.stack(refs, dim=1).float().flatten(), dim=0).item() > 0.99995
    print("  PASS: GDN Stage 2 BLOCK_DV=16 + VerifyScan nw=1 verified.")


@torch.no_grad()
def bench_cuda(fn, warmup: int = 10, iters: int = 50) -> tuple[float, float]:
    """Return (median_ms, mean_ms) using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    median_ms = times[len(times) // 2]
    mean_ms = sum(times) / len(times)
    return median_ms, mean_ms


@torch.no_grad()
def run_b300_benchmarks() -> None:
    print("\n=== Test 4: Real-Shape B300 Single-Layer GPU Micro-Benchmarks (hidden=5120, BF16) ===")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    print(f"  GPU: {torch.cuda.get_device_name(0)}, cap={torch.cuda.get_device_capability(0)}")

    cfg = make_config(hidden_size=5120)
    plan = YoCoSharingPlan.from_config(cfg)

    attn_prod = Qwen4_5Attention.from_config(cfg, layer_idx=19, sharing_plan=plan).to(device=device, dtype=dtype)
    attn_cons = Qwen4_5Attention.from_config(cfg, layer_idx=23, sharing_plan=plan).to(device=device, dtype=dtype)
    gdn_norm = Qwen4_5GatedDeltaNet.from_config(cfg, layer_idx=0, sharing_plan=plan).to(device=device, dtype=dtype)
    gdn_src = Qwen4_5GatedDeltaNet.from_config(cfg, layer_idx=20, sharing_plan=plan).to(device=device, dtype=dtype)
    gdn_cons = Qwen4_5GatedDeltaNet.from_config(cfg, layer_idx=21, sharing_plan=plan).to(device=device, dtype=dtype)

    for m in (attn_prod, attn_cons, gdn_norm, gdn_src, gdn_cons):
        init_realistic_weights(m)

    # ---- Decode benchmarks (B=1 and B=32, FullAttn KV cached len=1024) ----
    print("\n  -- Decode (T=1, FullAttn KV len=1024) --")
    for B in (1, 32):
        h = torch.randn((B, 1, 5120), device=device, dtype=dtype)
        kv_len = 1024
        kv_cache = attn_prod.create_kv_cache(B, kv_len + 16, device, dtype)
        kv_cache.k_cache[:, :kv_len].normal_(0.0, 0.1)
        kv_cache.v_cache[:, :kv_len].normal_(0.0, 0.1)
        kv_cache.seq_len = kv_len
        ks, vs = attn_prod.create_sconv_states(B, device, dtype)
        pos_dec = torch.full((B, 1), kv_len, device=device, dtype=torch.int64)

        def run_attn_prod():
            kv_cache.seq_len = kv_len
            return attn_prod(h, positions=pos_dec, kv_cache=kv_cache, k_sconv_state=ks, v_sconv_state=vs, seq_offset=kv_len)

        def run_attn_cons():
            kv_cache.seq_len = kv_len + 1
            return attn_cons(h, positions=pos_dec, kv_cache=kv_cache, seq_offset=kv_len)

        st_norm = gdn_norm.create_state(B, device, dtype)
        st_src = gdn_src.create_state(B, device, dtype)
        st_cons = gdn_cons.create_state(B, device, dtype)
        _, shared_dec = gdn_src(h, state=st_src)

        med_ap, mean_ap = bench_cuda(run_attn_prod)
        med_ac, mean_ac = bench_cuda(run_attn_cons)
        med_gn, mean_gn = bench_cuda(lambda: gdn_norm(h, state=st_norm))
        med_gs, mean_gs = bench_cuda(lambda: gdn_src(h, state=st_src))
        med_gc, mean_gc = bench_cuda(lambda: gdn_cons(h, state=st_cons, shared=shared_dec))

        # Standalone core kernels
        q_k = torch.randn((B, 1, 32, 64), device=device, dtype=dtype)
        k_k = torch.randn((B, 1, 32, 64), device=device, dtype=dtype)
        v_k = torch.randn((B, 1, 128, 64), device=device, dtype=dtype)
        g_k = -F.softplus(torch.randn((B, 1, 128), device=device, dtype=torch.float32))
        b_k = torch.sigmoid(torch.randn((B, 1, 128), device=device, dtype=torch.float32))
        s_k = torch.randn((B, 128, 64, 64), device=device, dtype=torch.float32) * 0.1
        med_rec, _ = bench_cuda(lambda: gdn_recurrent_triton(q_k, k_k, v_k, g_k, b_k, initial_state=s_k, inplace_state=True))
        med_yoco, _ = bench_cuda(lambda: yoco_gdn_decode_readout(q_k, s_k, head_k_dim=64))

        print(f"  [Decode B={B:2d}] FullAttn Producer(L19): median={med_ap:.4f} ms (mean={mean_ap:.4f} ms)")
        print(f"  [Decode B={B:2d}] FullAttn Consumer(L23): median={med_ac:.4f} ms (mean={mean_ac:.4f} ms)")
        print(f"  [Decode B={B:2d}] GDN Normal(L0):         median={med_gn:.4f} ms (mean={mean_gn:.4f} ms)")
        print(f"  [Decode B={B:2d}] GDN Source(L20):        median={med_gs:.4f} ms (mean={mean_gs:.4f} ms)")
        print(f"  [Decode B={B:2d}] GDN Consumer(L21 YoCo): median={med_gc:.4f} ms (mean={mean_gc:.4f} ms)")
        print(f"  [Decode B={B:2d}]   kernel gdn_recurrent_triton:    median={med_rec:.4f} ms")
        print(f"  [Decode B={B:2d}]   kernel yoco_gdn_decode_readout: median={med_yoco:.4f} ms")

    # ---- Prefill benchmarks (B=1, T=1024 and T=4096) ----
    print("\n  -- Prefill (B=1, T=1024 and T=4096) --")
    for T in (1024, 4096):
        h = torch.randn((1, T, 5120), device=device, dtype=dtype)
        pos_pre = torch.arange(T, device=device, dtype=torch.int64).unsqueeze(0)
        kv_cache = attn_prod.create_kv_cache(1, T, device, dtype)
        ks, vs = attn_prod.create_sconv_states(1, device, dtype)

        def run_attn_prod_pre():
            kv_cache.seq_len = 0
            ks.conv_state.zero_()
            vs.conv_state.zero_()
            return attn_prod(h, positions=pos_pre, kv_cache=kv_cache, k_sconv_state=ks, v_sconv_state=vs, seq_offset=0)

        # Populate kv_cache once for consumer benchmark
        run_attn_prod_pre()

        def run_attn_cons_pre():
            return attn_cons(h, positions=pos_pre, kv_cache=kv_cache, seq_offset=0)

        st_norm = gdn_norm.create_state(1, device, dtype)
        st_src = gdn_src.create_state(1, device, dtype)
        st_cons = gdn_cons.create_state(1, device, dtype)

        def run_gdn_norm_pre():
            st_norm.conv_state.zero_()
            st_norm.ssm_state.zero_()
            return gdn_norm(h, state=st_norm)

        def run_gdn_src_pre():
            st_src.conv_state.zero_()
            st_src.ssm_state.zero_()
            return gdn_src(h, state=st_src)

        _, shared_pre = run_gdn_src_pre()

        def run_gdn_cons_pre():
            st_cons.conv_state.zero_()
            return gdn_cons(h, state=st_cons, shared=shared_pre)

        med_ap, mean_ap = bench_cuda(run_attn_prod_pre, warmup=5, iters=20)
        med_ac, mean_ac = bench_cuda(run_attn_cons_pre, warmup=5, iters=20)
        med_gn, mean_gn = bench_cuda(run_gdn_norm_pre, warmup=5, iters=20)
        med_gs, mean_gs = bench_cuda(run_gdn_src_pre, warmup=5, iters=20)
        med_gc, mean_gc = bench_cuda(run_gdn_cons_pre, warmup=5, iters=20)

        # Standalone chunk prefill kernel
        q_k = torch.randn((1, T, 32, 64), device=device, dtype=dtype)
        k_k = torch.randn((1, T, 32, 64), device=device, dtype=dtype)
        v_k = torch.randn((1, T, 128, 64), device=device, dtype=dtype)
        g_k = -F.softplus(torch.randn((1, T, 128), device=device, dtype=torch.float32))
        b_k = torch.sigmoid(torch.randn((1, T, 128), device=device, dtype=torch.float32))
        med_chk, _ = bench_cuda(lambda: gdn_chunk_prefill_triton(q_k, k_k, v_k, g_k, b_k), warmup=5, iters=20)

        print(f"  [Prefill T={T:4d}] FullAttn Producer(L19): median={med_ap:.4f} ms (mean={mean_ap:.4f} ms)")
        print(f"  [Prefill T={T:4d}] FullAttn Consumer(L23): median={med_ac:.4f} ms (mean={mean_ac:.4f} ms)")
        print(f"  [Prefill T={T:4d}] GDN Normal(L0):         median={med_gn:.4f} ms (mean={mean_gn:.4f} ms)")
        print(f"  [Prefill T={T:4d}] GDN Source(L20):        median={med_gs:.4f} ms (mean={mean_gs:.4f} ms)")
        print(f"  [Prefill T={T:4d}] GDN Consumer(L21 YoCo): median={med_gc:.4f} ms (mean={mean_gc:.4f} ms)")
        print(f"  [Prefill T={T:4d}]   kernel gdn_chunk_prefill_triton: median={med_chk:.4f} ms")

    # ---- Verify (B=4, V=8) vs 8x Decode & Packed Ragged Prefill (T=1024) ----
    print("\n  -- GDN Verify (B=4, V=8) vs 8x Decode & Packed Prefill (T=1024) --")
    Bv, V = 4, 8
    hv = torch.randn((Bv, V, 5120), device=device, dtype=dtype)
    st_vn = gdn_norm.create_state(Bv, device, dtype)
    st_vs = gdn_src.create_state(Bv, device, dtype)
    st_vc = gdn_cons.create_state(Bv, device, dtype)
    sc_vb = torch.empty((Bv, V, gdn_src.conv_dim, 3), device=device, dtype=dtype)
    ss_vb = torch.empty((Bv, V, gdn_src.num_v_heads, 64, 64), device=device, dtype=torch.float32)
    sc_vcb = torch.empty((Bv, V, gdn_cons.conv_dim, 3), device=device, dtype=dtype)
    med_dec8_n, _ = bench_cuda(lambda: [gdn_norm(hv[:, i : i + 1], state=st_vn) for i in range(V)])
    med_ver8_n, _ = bench_cuda(lambda: gdn_norm.forward_verify(hv, init_conv_state=st_vn.conv_state, init_ssm_state=st_vn.ssm_state, stage_conv_state=sc_vb, stage_ssm_state=ss_vb))
    med_ver8_s, _ = bench_cuda(lambda: gdn_src.forward_verify(hv, init_conv_state=st_vs.conv_state, init_ssm_state=st_vs.ssm_state, stage_conv_state=sc_vb, stage_ssm_state=ss_vb))
    _, sh_vb = gdn_src.forward_verify(hv, init_conv_state=st_vs.conv_state, init_ssm_state=st_vs.ssm_state, stage_conv_state=sc_vb, stage_ssm_state=ss_vb)
    med_ver8_c, _ = bench_cuda(lambda: gdn_cons.forward_verify(hv, init_conv_state=st_vc.conv_state, stage_conv_state=sc_vcb, shared=sh_vb, enable_yoco_decode=True))
    print(f"  [Verify B=4,V=8] GDN Normal(L0):  8xDecode={med_dec8_n:.4f} ms vs Verify={med_ver8_n:.4f} ms ({med_dec8_n/max(med_ver8_n,1e-6):.2f}x)")
    print(f"  [Verify B=4,V=8] GDN Source(L20): Verify={med_ver8_s:.4f} ms | Consumer(L21): Verify={med_ver8_c:.4f} ms")

    _, cu_p, tr_p, tt0_p, cr_p, ct0_p, rcs_p, _, _, _, _, _, _, _ = _build_ragged_meta_for_test([1024], device=device)
    hp = torch.randn((1024, 5120), device=device, dtype=dtype)
    med_packed, _ = bench_cuda(lambda: gdn_norm.forward_packed_prefill(hp, cu_p, tr_p, tt0_p, cr_p, ct0_p, rcs_p), warmup=5, iters=20)
    print(f"  [Packed Prefill T=1024] GDN Normal(L0) forward_packed_prefill: median={med_packed:.4f} ms")


def main() -> None:
    t0 = time.time()
    test_yoco_sharing_plan_and_param_shapes()
    test_full_attention_producer_and_consumer()
    test_gdn_kernels_and_module()
    test_yoco_shared_gdn_source_and_consumer()
    test_gdn_fused_unpack_verify_ragged_and_tp()
    test_r3_gdn_stage2_sub_dv_and_verify_nw1()
    run_b300_benchmarks()
    print(f"\nALL TESTS & BENCHMARKS PASSED in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()

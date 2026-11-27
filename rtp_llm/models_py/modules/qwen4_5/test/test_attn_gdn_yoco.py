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
def test_r4_rmsnorm_gated_packed_yoco_subdv_and_prefill_tuning() -> None:
    print("[R4 Test] RMSNormGated BLOCK_H=4 + YoCo Readout BLOCK_DV=32 + Gating Prealloc + Prefill R4 Tuning...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    if device.type != "cuda":
        print("  SKIP (CUDA required)")
        return
    torch.manual_seed(202641)

    # 1. RMSNormGated packed BLOCK_H=4 (4D, 5D GQA gate, sigmoid/silu) and fallback H=6 / D=256
    for hv, gqa, dim, act, b, t in [
        (128, 1, 64, "sigmoid", 2, 128),
        (128, 8, 64, "silu", 8, 8),
        (6, 1, 64, "sigmoid", 2, 16),
        (8, 1, 256, "sigmoid", 2, 8),
    ]:
        rmsn = RMSNormGated(dim, eps=1e-6, activation=act, norm_before_gate=True).to(device=device, dtype=dtype).eval()
        torch.nn.init.normal_(rmsn.weight, mean=1.0, std=0.2)
        x = torch.randn((b, t, hv, dim), device=device, dtype=dtype) * 0.5
        if gqa > 1:
            z = torch.randn((b, t, hv // gqa, gqa, dim), device=device, dtype=dtype) * 0.5
        else:
            z = torch.randn((b, t, hv, dim), device=device, dtype=dtype) * 0.5
        y_tri = rmsn(x, z)
        g_f = z.reshape(x.shape).float()
        g_val = torch.sigmoid(g_f) if act == "sigmoid" else F.silu(g_f)
        y_ref = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6) * rmsn.weight.float() * g_val).to(dtype)
        cos = F.cosine_similarity(y_tri.float().flatten(), y_ref.float().flatten(), dim=0).item()
        max_abs = float((y_tri.float() - y_ref.float()).abs().max().item())
        assert cos > 0.99999 and max_abs < 2e-2, f"RMSNormGated hv={hv} gqa={gqa} dim={dim}: cos={cos}, max_abs={max_abs}"

    # 2. YoCo GDN Verify Readout BLOCK_DV=32 (B*V*H_v >= 512) + CUDA Graph capture/replay
    cfg = make_config(5120)
    gdn_cons = Qwen4_5GatedDeltaNet.from_config(cfg, layer_idx=22).to(device=device, dtype=dtype).eval()
    B_ro, V_ro = 8, 8  # B*V*Hv = 8*8*128 = 8192 >= 512 -> BLOCK_DV=32
    hv_in = torch.randn((B_ro, V_ro, 5120), device=device, dtype=dtype) * 0.2
    st_c = gdn_cons.create_state(B_ro, device, dtype)
    sc_out = torch.empty((B_ro, V_ro, gdn_cons.conv_dim, 3), device=device, dtype=dtype)
    sh_in = SharedGDNInputs(
        key=torch.randn((B_ro, V_ro, gdn_cons.num_k_heads, 64), device=device, dtype=dtype) * 0.2,
        value=torch.randn((B_ro, V_ro, gdn_cons.num_v_heads, 64), device=device, dtype=dtype) * 0.2,
        g=-torch.rand((B_ro, V_ro, gdn_cons.num_v_heads), device=device, dtype=torch.float32) * 0.2,
        beta=torch.sigmoid(torch.randn((B_ro, V_ro, gdn_cons.num_v_heads), device=device, dtype=torch.float32)),
        ssm_state=torch.randn((B_ro, V_ro, gdn_cons.num_v_heads, 64, 64), device=device, dtype=torch.float32) * 0.1,
    )
    out_ro, _ = gdn_cons.forward_verify(
        hv_in, init_conv_state=st_c.conv_state, stage_conv_state=sc_out, shared=sh_in, enable_yoco_decode=True
    )
    # Compare against step-by-step reference readout
    qz_ref = gdn_cons.in_proj_qkvz(hv_in)
    q_raw, z_ref = gdn_cons._unpack_qz_consumer(qz_ref)
    conv_st_ref = st_c.conv_state.clone()
    from rtp_llm.models_py.modules.qwen4_5.short_conv import causal_dwconv1d_ref
    q_conv = causal_dwconv1d_ref(q_raw, gdn_cons.conv1d.weight, conv_st_ref, residual=gdn_cons.gdn_conv_residual, use_activation=True)
    q_4d = q_conv.view(B_ro, V_ro, gdn_cons.num_k_heads, 64).repeat_interleave(gdn_cons.gqa_ratio, dim=2).float()
    q_4d = q_4d * torch.rsqrt(q_4d.square().sum(dim=-1, keepdim=True) + 1e-6) * (64 ** -0.5)
    core_ref = torch.einsum("bshk,bshkv->bshv", q_4d, sh_in.ssm_state).to(dtype)
    out_ref = gdn_cons.out_proj(gdn_cons.norm(core_ref, z_ref).reshape(B_ro, V_ro, gdn_cons.value_dim))
    assert F.cosine_similarity(out_ro.float().flatten(), out_ref.float().flatten(), dim=0).item() > 0.9999

    cg_ro = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cg_ro):
        out_cg, _ = gdn_cons.forward_verify(
            hv_in, init_conv_state=st_c.conv_state, stage_conv_state=sc_out, shared=sh_in, enable_yoco_decode=True
        )
    hv_in.copy_(torch.randn_like(hv_in) * 0.2)
    cg_ro.replay()
    out_eag, _ = gdn_cons.forward_verify(
        hv_in, init_conv_state=st_c.conv_state, stage_conv_state=torch.empty_like(sc_out), shared=sh_in, enable_yoco_decode=True
    )
    assert F.cosine_similarity(out_cg.float().flatten(), out_eag.float().flatten(), dim=0).item() > 0.99999

    # 3. compute_gdn_gating direct preallocated output parity
    ba = torch.randn((4, 8, 2 * 128), device=device, dtype=dtype)
    alog = torch.randn((128,), device=device, dtype=torch.float32)
    dt_b = torch.randn((128,), device=device, dtype=torch.float32)
    g1, b1 = compute_gdn_gating(ba, alog, dt_b, 16, 128)
    og = torch.empty_like(g1)
    ob = torch.empty_like(b1)
    g2, b2 = compute_gdn_gating(ba, alog, dt_b, 16, 128, out_g=og, out_beta=ob)
    assert g2 is og and b2 is ob
    assert float((g1 - og).abs().max().item()) == 0.0 and float((b1 - ob).abs().max().item()) == 0.0

    # 4. Ragged prefill Stage 1/2/3 R4 tuning on small & multi-request batches
    cu_list, cu, _, _, c_req, c_t0, rcs, _, _, rss, sle, sp_phys, _, pool_sz = _build_ragged_meta_for_test([192, 128], device=device)
    Nt, Hk, Hv = int(cu[-1].item()), 16, 128
    qr = torch.randn((Nt, Hk, 64), device=device, dtype=dtype) * 0.2
    kr = torch.randn((Nt, Hk, 64), device=device, dtype=dtype) * 0.2
    vr = torch.randn((Nt, Hv, 64), device=device, dtype=dtype) * 0.2
    gr = -torch.rand((Nt, Hv), device=device, dtype=torch.float32) * 0.2
    br = torch.sigmoid(torch.randn((Nt, Hv), device=device, dtype=torch.float32))
    out_r, fs_r = gdn_chunk_prefill_ragged_with_snapshots_triton(
        qr, kr, vr, gr, br, cu, c_req, c_t0, rcs,
        req_snap_start=rss, snap_local_end=sle, snap_phys=sp_phys,
        ssm_dst_pool=torch.zeros((pool_sz, Hv, 64, 64), device=device, dtype=torch.float32),
        output_final_states=True,
    )
    for req_i, ln in enumerate([192, 128]):
        s_i, e_i = cu_list[req_i], cu_list[req_i + 1]
        o_ref_i, f_ref_i = gdn_recurrent_ref(
            qr[s_i:e_i].unsqueeze(0), kr[s_i:e_i].unsqueeze(0), vr[s_i:e_i].unsqueeze(0),
            gr[s_i:e_i].unsqueeze(0), br[s_i:e_i].unsqueeze(0), output_final_state=True,
        )
        assert F.cosine_similarity(out_r[s_i:e_i].float().flatten(), o_ref_i.squeeze(0).float().flatten(), dim=0).item() > 0.9999
        assert F.cosine_similarity(fs_r[req_i].flatten(), f_ref_i[0].flatten(), dim=0).item() > 0.9999
    print("  PASS: R4 GDN RMSNormGated + YoCo BLOCK_DV=32 + Gating prealloc + Prefill tuning verified.")



def test_r5_gdn_verify_readout_stage1_and_fused_projections() -> None:
    print("[R5 Test] GDN Recurrent/Verify/Readout/Stage1 + Attention prepare_fused_projections...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    if device.type != "cuda":
        print("  SKIP (CUDA required)")
        return
    torch.manual_seed(202651)
    from rtp_llm.models_py.modules.qwen4_5.fp8_linear import SM100FP8Linear

    # 1. gdn_recurrent_triton B>=2 IS_CONTIG_INIT/FS & non-contiguous initial_state fallback
    for is_nc in (False, True):
        B_r, T_r, Hk_r, Hv_r = 8, 1, 16, 128
        qr = torch.randn((B_r, T_r, Hk_r, 64), device=device, dtype=dtype) * 0.2
        kr = torch.randn((B_r, T_r, Hk_r, 64), device=device, dtype=dtype) * 0.2
        vr = torch.randn((B_r, T_r, Hv_r, 64), device=device, dtype=dtype) * 0.2
        gr = -torch.rand((B_r, T_r, Hv_r), device=device, dtype=torch.float32) * 0.2
        br = torch.sigmoid(torch.randn((B_r, T_r, Hv_r), device=device, dtype=torch.float32))
        st0 = torch.randn((B_r, Hv_r, 64, 64), device=device, dtype=torch.float32) * 0.1
        if is_nc:
            st0 = torch.randn((B_r, Hv_r, 64, 128), device=device, dtype=torch.float32)[..., ::2] * 0.1
        o_tri, fs_tri = gdn_recurrent_triton(qr, kr, vr, gr, br, initial_state=st0, output_final_state=True)
        o_ref, fs_ref = gdn_recurrent_ref(qr, kr, vr, gr, br, initial_state=st0, output_final_state=True)
        assert F.cosine_similarity(o_tri.float().flatten(), o_ref.float().flatten(), dim=0).item() > 0.9999
        assert float((fs_tri - fs_ref).abs().max().item()) < 1e-3

    # 2. GDN Verify Scan IS_CONTIG+EVICT_ST + YoCo Readout BK=8 + Stage1 Phase-split d_ssm < 1e-3
    cfg = make_config(5120)
    plan = YoCoSharingPlan.from_config(cfg)
    gdn_src = Qwen4_5GatedDeltaNet.from_config(cfg, layer_idx=20, sharing_plan=plan).to(device=device, dtype=dtype).eval()
    gdn_cons = Qwen4_5GatedDeltaNet.from_config(cfg, layer_idx=21, sharing_plan=plan).to(device=device, dtype=dtype).eval()
    init_realistic_weights(gdn_src)
    init_realistic_weights(gdn_cons)

    B_v, V_v = 8, 8
    hv_s = torch.randn((B_v, V_v, 5120), device=device, dtype=dtype) * 0.15
    st_s = gdn_src.create_state(B_v, device, dtype)
    st_c = gdn_cons.create_state(B_v, device, dtype)
    sc_s = torch.empty((B_v, V_v, gdn_src.conv_dim, 3), device=device, dtype=dtype)
    ss_s = torch.empty((B_v, V_v, gdn_src.num_v_heads, 64, 64), device=device, dtype=torch.float32)
    sc_c = torch.empty((B_v, V_v, gdn_cons.conv_dim, 3), device=device, dtype=dtype)
    y_vs_ref, sh_vs_ref = gdn_src.forward_verify(
        hv_s, init_conv_state=st_s.conv_state, init_ssm_state=st_s.ssm_state, stage_conv_state=sc_s, stage_ssm_state=ss_s
    )
    y_vc_ref, _ = gdn_cons.forward_verify(
        hv_s, init_conv_state=st_c.conv_state, stage_conv_state=sc_c, shared=sh_vs_ref, enable_yoco_decode=True
    )

    # Check Stage 1 Phase-split tf32 SSM state precision < 1e-3
    _, cu, _, _, c_req, c_t0, rcs, _, _, rss, sle, sp_phys, _, pool_sz = _build_ragged_meta_for_test([256, 192], device=device)
    cu_list = cu.tolist()
    Nt = int(cu[-1].item())
    q_s1 = torch.randn((Nt, 16, 64), device=device, dtype=dtype) * 0.2
    k_s1 = torch.randn((Nt, 16, 64), device=device, dtype=dtype) * 0.2
    v_s1 = torch.randn((Nt, 128, 64), device=device, dtype=dtype) * 0.2
    g_s1 = -torch.rand((Nt, 128), device=device, dtype=torch.float32) * 0.2
    b_s1 = torch.sigmoid(torch.randn((Nt, 128), device=device, dtype=torch.float32))
    out_s1, fs_s1 = gdn_chunk_prefill_ragged_with_snapshots_triton(
        q_s1, k_s1, v_s1, g_s1, b_s1, cu, c_req, c_t0, rcs,
        req_snap_start=rss, snap_local_end=sle, snap_phys=sp_phys,
        ssm_dst_pool=torch.zeros((pool_sz, 128, 64, 64), device=device, dtype=torch.float32),
        output_final_states=True,
    )
    for ri in range(2):
        si, ei = int(cu_list[ri]), int(cu_list[ri + 1])
        _, f_ref = gdn_recurrent_ref(
            q_s1[si:ei].unsqueeze(0), k_s1[si:ei].unsqueeze(0), v_s1[si:ei].unsqueeze(0),
            g_s1[si:ei].unsqueeze(0), b_s1[si:ei].unsqueeze(0), output_final_state=True,
        )
        assert float((fs_s1[ri] - f_ref[0]).abs().max().item()) < 1e-3

    # 3. GDN Verify CG replay + separated in_proj_qkvz (128-aligned) & BF16 in_proj_ba guard
    assert not hasattr(gdn_src, "in_proj_qkvz_ba") and not hasattr(gdn_src, "prepare_fused_projections")
    sc_sf = torch.empty_like(sc_s)
    ss_sf = torch.empty_like(ss_s)
    cg_v = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cg_v):
        y_vs_cg, sh_vs_cg = gdn_src.forward_verify(
            hv_s, init_conv_state=st_s.conv_state, init_ssm_state=st_s.ssm_state, stage_conv_state=sc_sf, stage_ssm_state=ss_sf
        )
        y_vc_cg, _ = gdn_cons.forward_verify(
            hv_s, init_conv_state=st_c.conv_state, stage_conv_state=sc_c, shared=sh_vs_cg, enable_yoco_decode=True
        )
    hv_s.copy_(torch.randn_like(hv_s) * 0.15)
    cg_v.replay()
    y_vs_e, sh_vs_e = gdn_src.forward_verify(
        hv_s, init_conv_state=st_s.conv_state, init_ssm_state=st_s.ssm_state, stage_conv_state=torch.empty_like(sc_s), stage_ssm_state=torch.empty_like(ss_s)
    )
    y_vc_e, _ = gdn_cons.forward_verify(
        hv_s, init_conv_state=st_c.conv_state, stage_conv_state=torch.empty_like(sc_c), shared=sh_vs_e, enable_yoco_decode=True
    )
    assert F.cosine_similarity(y_vs_cg.float().flatten(), y_vs_e.float().flatten(), dim=0).item() > 0.99999
    assert F.cosine_similarity(y_vc_cg.float().flatten(), y_vc_e.float().flatten(), dim=0).item() > 0.99999

    def _wrap_sm100(mod: torch.nn.Module) -> torch.nn.Module:
        if not isinstance(mod, torch.nn.Linear):
            return mod
        inst = SM100FP8Linear(
            in_features=mod.in_features,
            out_features=mod.out_features,
            bias=(mod.bias is not None),
            mode="auto",
            keep_bf16=False,
            prepare_trtllm_ll=True,
            prepare_mxfp8=False,
            dtype=mod.weight.dtype,
            device=mod.weight.device,
        )
        inst.weight = mod.weight
        inst.prepare_weights()
        return inst

    gdn_tp4 = Qwen4_5GatedDeltaNet.from_config(cfg, layer_idx=0, sharing_plan=plan, tp_size=4, tp_rank=0).to(device=device, dtype=dtype).eval()
    init_realistic_weights(gdn_tp4)
    assert gdn_tp4.in_proj_qkvz.weight.shape[0] % 128 == 0 and gdn_tp4.in_proj_ba.weight.shape[0] == 64
    h_tp4 = torch.randn((4, 8, 5120), device=device, dtype=dtype) * 0.15
    y_tp4_ref, _ = gdn_tp4.forward(h_tp4, state=gdn_tp4.create_state(4, device, dtype))
    gdn_tp4.in_proj_qkvz = _wrap_sm100(gdn_tp4.in_proj_qkvz)
    assert isinstance(gdn_tp4.in_proj_qkvz, SM100FP8Linear) and isinstance(gdn_tp4.in_proj_ba, torch.nn.Linear)
    y_tp4_f, _ = gdn_tp4.forward(h_tp4, state=gdn_tp4.create_state(4, device, dtype))
    assert F.cosine_similarity(y_tp4_f.float().flatten(), y_tp4_ref.float().flatten(), dim=0).item() > 0.9999

    # 4. Qwen4_5Attention prepare_fused_projections (Producer Q+K+V+Idx & Consumer Q+Idx, sconv=True/False)
    for use_sconv in (True, False):
        cfg_qsa_c = Qwen4_5ArchConfig(
            hidden_size=512,
            num_hidden_layers=24,
            full_attention_interval=4,
            num_attention_heads=16,
            num_key_value_heads=1,
            head_dim=256,
            attn_output_gate=True,
            use_full_attn_kv_sconv=use_sconv,
            indexer_head_dim=128,
            indexer_n_heads=8,
            indexer_kv_heads=1,
            indexer_compress_ratio=4,
            indexer_budget=16,
            share_map="19->23",
            enable_yoco_decode=True,
        )
        assert cfg_qsa_c.has_qsa
        p_a = YoCoSharingPlan.from_config(cfg_qsa_c)
        sa_p = Qwen4_5Attention.from_config(cfg_qsa_c, layer_idx=19, sharing_plan=p_a).to(device=device, dtype=dtype).eval()
        sa_c = Qwen4_5Attention.from_config(cfg_qsa_c, layer_idx=23, sharing_plan=p_a).to(device=device, dtype=dtype).eval()
        init_realistic_weights(sa_p)
        init_realistic_weights(sa_c)
        ha = torch.randn((2, 16, 512), device=device, dtype=dtype) * 0.2
        posa = torch.arange(16, device=device, dtype=torch.int64).unsqueeze(0).expand(2, 16)
        raw_src = []
        kv_ref = sa_p.create_kv_cache(2, 32, device, dtype)
        ks_r, vs_r = sa_p.create_sconv_states(2, device, dtype)
        out_p_ref = sa_p(ha, positions=posa, kv_cache=kv_ref, k_sconv_state=ks_r, v_sconv_state=vs_r, seq_offset=0, qsa_on_raw_keys=raw_src.append)
        out_c_ref = sa_c(ha, positions=posa, kv_cache=kv_ref, seq_offset=0, qsa_source_raw_keys=raw_src[0])

        assert sa_p.prepare_fused_projections() is not None
        assert sa_c.prepare_fused_projections() is not None
        assert sa_p.q_proj.weight.untyped_storage().data_ptr() == sa_p.qkv_idx_proj.weight.untyped_storage().data_ptr()
        assert sa_p.indexer.qk_proj.data_ptr() == sa_p.indexer.index_qk_proj.weight.data_ptr()
        assert sa_c.indexer.qk_proj.data_ptr() == sa_c.indexer.index_qk_proj.weight.data_ptr()

        raw_f = []
        kv_f = sa_p.create_kv_cache(2, 32, device, dtype)
        ks_f, vs_f = sa_p.create_sconv_states(2, device, dtype)
        out_p_f = sa_p(ha, positions=posa, kv_cache=kv_f, k_sconv_state=ks_f, v_sconv_state=vs_f, seq_offset=0, qsa_on_raw_keys=raw_f.append)
        out_c_f = sa_c(ha, positions=posa, kv_cache=kv_f, seq_offset=0, qsa_source_raw_keys=raw_f[0])
        assert F.cosine_similarity(out_p_f.float().flatten(), out_p_ref.float().flatten(), dim=0).item() > 0.9999
        assert F.cosine_similarity(out_c_f.float().flatten(), out_c_ref.float().flatten(), dim=0).item() > 0.9999

    # Non-QSA Consumer returns None; Q_proj reload triggers data_ptr() mismatch fallback
    sa_dense_c = Qwen4_5Attention.from_config(make_config(512), layer_idx=23).to(device=device, dtype=dtype).eval()
    assert sa_dense_c.prepare_fused_projections() is None
    sa_p.q_proj.weight = torch.nn.Parameter(torch.randn_like(sa_p.q_proj.weight) * 0.05)
    assert not sa_p._fused_qkv_idx_ptrs_valid()
    print("  PASS: R5 GDN kernels, Stage1 Phase-split, and Attention prepare_fused_projections verified.")


@torch.no_grad()
def test_r6_gdn_decode_opaque_and_direct_chunk64_snaps() -> None:
    print("\n=== [3f] R6 GDN Direct-Opaque Decode & Track 2 Direct-Chunk64 Ragged Prefill ===")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(202661)

    # 1. forward_decode_opaque parity across Normal(L0), YoCo Source(L20), YoCo Consumer(L21), packed vs split pools, and CG
    cfg = make_config(hidden_size=512)
    plan = YoCoSharingPlan.from_config(cfg)
    gdn_norm = Qwen4_5GatedDeltaNet.from_config(cfg, layer_idx=0, sharing_plan=plan).to(device=device, dtype=dtype).eval()
    gdn_src = Qwen4_5GatedDeltaNet.from_config(cfg, layer_idx=20, sharing_plan=plan).to(device=device, dtype=dtype).eval()
    gdn_cons = Qwen4_5GatedDeltaNet.from_config(cfg, layer_idx=21, sharing_plan=plan).to(device=device, dtype=dtype).eval()
    for mod in (gdn_norm, gdn_src, gdn_cons):
        init_realistic_weights(mod)

    B, ps, num_blocks = 4, 64, 24
    bt = torch.tensor([[1 + b * 2, 2 + b * 2, 0] for b in range(B)], device=device, dtype=torch.int32)
    seq_before = torch.tensor([0, 64, 65, 128], device=device, dtype=torch.int32)
    h_dec = torch.randn((B, 1, 512), device=device, dtype=dtype) * 0.25

    import math
    _st_s0 = gdn_src.create_state(1, device, dtype)
    _st_c0 = gdn_cons.create_state(1, device, dtype)
    ssm_shape = tuple(_st_s0.ssm_state.shape[1:])
    conv_shape = tuple(_st_s0.conv_state.shape[1:])
    conv_c_shape = tuple(_st_c0.conv_state.shape[1:])
    ssm_n = math.prod(ssm_shape)
    conv_n = math.prod(conv_shape)
    conv_c_n = math.prod(conv_c_shape)

    # 1a) Packed LINEAR_ATTN_TAG layout (ssm_n fp32 + 2*conv_n bf16)
    row_fp32 = ssm_n + ((conv_n + 1) // 2)
    lin_pool = torch.zeros((num_blocks, row_fp32), device=device, dtype=torch.float32)
    lin_pool[:, :ssm_n].normal_(0.0, 0.1)
    lin_pool[:, ssm_n:].view(torch.bfloat16)[:, :conv_n].normal_(0.0, 0.2)
    lin_pool_ref = lin_pool.clone()
    ssm_pool_2d = lin_pool.view(num_blocks, row_fp32)
    conv_pool_2d = lin_pool.view(torch.bfloat16).view(num_blocks, row_fp32 * 2)

    st_ref_n = gdn_norm.create_state(B, device, dtype)
    for b in range(B):
        sb = int(seq_before[b].item())
        if sb > 0:
            phys = int(bt[b, (sb - 1) // ps].item())
            if 0 < phys < num_blocks:
                st_ref_n.ssm_state[b].copy_(lin_pool_ref[phys, :ssm_n].view(ssm_shape))
                st_ref_n.conv_state[b].copy_(
                    lin_pool_ref[phys, ssm_n:].view(torch.bfloat16)[:conv_n].view(conv_shape)
                )
    out_ref_n, _ = gdn_norm.forward(h_dec, state=st_ref_n)
    out_opq_n, _ = gdn_norm.forward_decode_opaque(
        h_dec,
        conv_pool_2d=conv_pool_2d, bt_conv=bt, tpb_conv=ps, conv_col_offset=2 * ssm_n,
        ssm_pool_2d=ssm_pool_2d, bt_ssm=bt, tpb_ssm=ps, ssm_col_offset=0,
        seq_before=seq_before,
    )
    assert F.cosine_similarity(out_opq_n.float().flatten(), out_ref_n.float().flatten(), dim=0).item() > 0.9999
    for b in range(B):
        sb = int(seq_before[b].item())
        phys_w = int(bt[b, sb // ps].item())
        if 0 < phys_w < num_blocks:
            ssm_got = ssm_pool_2d[phys_w, :ssm_n].view(ssm_shape)
            conv_got = conv_pool_2d[phys_w, 2 * ssm_n : 2 * ssm_n + conv_n].view(conv_shape)
            assert (ssm_got - st_ref_n.ssm_state[b]).abs().max().item() < 1e-4
            assert (conv_got - st_ref_n.conv_state[b]).abs().max().item() == 0.0

    # 1b) Split pools + YoCo Source/Consumer + CG capture/replay
    split_ssm = torch.randn((num_blocks, ssm_n), device=device, dtype=torch.float32) * 0.1
    split_conv = torch.randn((num_blocks, conv_n), device=device, dtype=dtype) * 0.2
    split_cons_conv = torch.randn((num_blocks, conv_c_n), device=device, dtype=dtype) * 0.2
    st_ref_s = gdn_src.create_state(B, device, dtype)
    st_ref_c = gdn_cons.create_state(B, device, dtype)
    for b in range(B):
        sb = int(seq_before[b].item())
        if sb > 0:
            phys = int(bt[b, (sb - 1) // ps].item())
            if 0 < phys < num_blocks:
                st_ref_s.ssm_state[b].copy_(split_ssm[phys].view(ssm_shape))
                st_ref_s.conv_state[b].copy_(split_conv[phys].view(conv_shape))
                st_ref_c.conv_state[b].copy_(split_cons_conv[phys].view(conv_c_shape))
    out_ref_s, sh_ref = gdn_src.forward(h_dec, state=st_ref_s)
    out_ref_c, _ = gdn_cons.forward(h_dec, state=st_ref_c, shared=sh_ref)

    ssm_cg, conv_cg, cconv_cg = split_ssm.clone(), split_conv.clone(), split_cons_conv.clone()
    h_cg = h_dec.clone()
    cg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cg):
        out_cg_s, sh_cg = gdn_src.forward_decode_opaque(
            h_cg, conv_pool_2d=conv_cg, bt_conv=bt, tpb_conv=ps, conv_col_offset=0,
            ssm_pool_2d=ssm_cg, bt_ssm=bt, tpb_ssm=ps, ssm_col_offset=0,
            seq_before=seq_before, enable_yoco_decode=True,
        )
        out_cg_c, _ = gdn_cons.forward_decode_opaque(
            h_cg, conv_pool_2d=cconv_cg, bt_conv=bt, tpb_conv=ps, conv_col_offset=0,
            ssm_pool_2d=None, bt_ssm=None, tpb_ssm=ps, ssm_col_offset=0,
            seq_before=seq_before, shared=sh_cg, enable_yoco_decode=True,
        )
    ssm_cg.copy_(split_ssm); conv_cg.copy_(split_conv); cconv_cg.copy_(split_cons_conv)
    h_new = torch.randn_like(h_dec) * 0.25
    h_cg.copy_(h_new)
    cg.replay()
    st_r2_s = gdn_src.create_state(B, device, dtype)
    st_r2_c = gdn_cons.create_state(B, device, dtype)
    for b in range(B):
        sb = int(seq_before[b].item())
        if sb > 0:
            phys = int(bt[b, (sb - 1) // ps].item())
            if 0 < phys < num_blocks:
                st_r2_s.ssm_state[b].copy_(split_ssm[phys].view(ssm_shape))
                st_r2_s.conv_state[b].copy_(split_conv[phys].view(conv_shape))
                st_r2_c.conv_state[b].copy_(split_cons_conv[phys].view(conv_c_shape))
    out_r2_s, sh_r2 = gdn_src.forward(h_new, state=st_r2_s)
    out_r2_c, _ = gdn_cons.forward(h_new, state=st_r2_c, shared=sh_r2)
    assert F.cosine_similarity(out_cg_s.float().flatten(), out_r2_s.float().flatten(), dim=0).item() > 0.9999
    assert F.cosine_similarity(out_cg_c.float().flatten(), out_r2_c.float().flatten(), dim=0).item() > 0.9999

    # 2. Track 2 Direct-Chunk64 Stage 1/2 tuning across Hv=32 (bdv=16, ns=3, nw=1 at ctas>=1024) & Hv=128
    for hv, lens in [(32, [64, 192, 130]), (32, [2048]), (128, [64, 512])]:
        hk = hv // 2
        N = sum(lens)
        _, cu, _, _, chunk_req, chunk_t0, req_chunk_start, _, _, s_end, r_ss, _, _, _ = _build_ragged_meta_for_test(
            lens, block_t=64, device=device
        )
        # Filter s_end to exact per-64-token chunk boundaries matching chunk count
        snaps = []
        r_sstart = [0]
        for ln in lens:
            for e in range(64, ln + 64, 64):
                snaps.append(min(ln, e))
            r_sstart.append(len(snaps))
        snap_le = torch.tensor(snaps, device=device, dtype=torch.int32)
        req_ss = torch.tensor(r_sstart, device=device, dtype=torch.int32)
        snap_phys = torch.arange(1, len(snaps) + 1, device=device, dtype=torch.int32)
        pool_d = torch.zeros((len(snaps) + 2, hv * 64 * 64), device=device, dtype=torch.float32)
        pool_w = torch.zeros_like(pool_d)

        qt = torch.randn((N, hk, 64), device=device, dtype=dtype) * 0.1
        kt = torch.randn((N, hk, 64), device=device, dtype=dtype) * 0.1
        vt = torch.randn((N, hv, 64), device=device, dtype=dtype) * 0.1
        gt = -torch.rand((N, hv), device=device, dtype=torch.float32) * 0.2
        bt_g = torch.sigmoid(torch.randn((N, hv), device=device, dtype=dtype))
        init_st = torch.randn((len(lens), hv, 64, 64), device=device, dtype=torch.float32) * 0.05

        out_d, fs_d = gdn_chunk_prefill_ragged_with_snapshots_triton(
            qt, kt, vt, gt, bt_g, cu, chunk_req, chunk_t0, req_chunk_start,
            initial_state=init_st, req_snap_start=req_ss, snap_local_end=snap_le,
            snap_phys=snap_phys, ssm_dst_pool=pool_d, snaps_direct_chunk64=True, output_final_states=True,
        )
        out_w, fs_w = gdn_chunk_prefill_ragged_with_snapshots_triton(
            qt, kt, vt, gt, bt_g, cu, chunk_req, chunk_t0, req_chunk_start,
            initial_state=init_st, req_snap_start=req_ss, snap_local_end=snap_le,
            snap_phys=snap_phys, ssm_dst_pool=pool_w, snaps_direct_chunk64=False, output_final_states=True,
        )
        assert float((out_d - out_w).abs().max().item()) == 0.0
        assert float((fs_d - fs_w).abs().max().item()) == 0.0
        assert float((pool_d - pool_w).abs().max().item()) == 0.0
        if N <= 512:
            for bi, ln in enumerate(lens):
                s0, s1 = sum(lens[:bi]), sum(lens[:bi + 1])
                o_ref, fs_r = gdn_recurrent_ref(
                    qt[s0:s1].unsqueeze(0), kt[s0:s1].unsqueeze(0), vt[s0:s1].unsqueeze(0),
                    gt[s0:s1].unsqueeze(0), bt_g[s0:s1].unsqueeze(0), initial_state=init_st[bi:bi+1],
                    output_final_state=True,
                )
                assert F.cosine_similarity(out_d[s0:s1].float().flatten(), o_ref.float().flatten(), dim=0).item() > 0.9998
                assert (fs_d[bi:bi+1] - fs_r).abs().max().item() < 5e-3
    # Cross-dtype conv1d.weight (fp32 -> bf16) ckey parity on forward_packed_prefill & live _GDN_TRITON_CACHE hit
    import copy
    from rtp_llm.models_py.modules.qwen4_5.gdn import _GDN_TRITON_CACHE

    gdn_f32 = copy.deepcopy(gdn_norm)
    gdn_f32.conv1d.weight = torch.nn.Parameter(gdn_f32.conv1d.weight.detach().float())
    _, cu_ck, tr_ck, tt0_ck, cr_ck, ct0_ck, rcs_ck, _, _, _, _, _, _, _ = _build_ragged_meta_for_test([64, 65], device=device)
    hp_ck = torch.randn((129, 512), device=device, dtype=dtype) * 0.25
    o_f32, _ = gdn_f32.forward_packed_prefill(hp_ck, cu_ck, tr_ck, tt0_ck, cr_ck, ct0_ck, rcs_ck)
    o_bf16, _ = gdn_norm.forward_packed_prefill(hp_ck, cu_ck, tr_ck, tt0_ck, cr_ck, ct0_ck, rcs_ck)
    assert F.cosine_similarity(o_f32.float().flatten(), o_bf16.float().flatten(), dim=0).item() > 0.9999
    assert len(_GDN_TRITON_CACHE) > 0
    print("  PASS: GDN forward_decode_opaque (packed & split, YoCo, CG), Direct-Chunk64 & conv_w dtype ckey verified.")


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
def _r7_ple_sp_worker(rank: int, world_size: int, init_file: str, q: object) -> None:
    import torch.distributed as dist
    from rtp_llm.models_py.modules.qwen4_5.contracts import PLELayerState
    from rtp_llm.models_py.modules.qwen4_5.fast_comm import sp_slice_bounds
    from rtp_llm.models_py.modules.qwen4_5.ple import DEFAULT_PLE_WEIGHT_SCALE, Qwen4_5PLELayer

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device("cuda", 0)
        dtype = torch.bfloat16
        H = 128
        hc = 4
        ple_dim = 160
        total_shards = 8
        shard_rows = 256
        vocab_sizes = [64] * 20

        def _make_ple(sharded: bool) -> Qwen4_5PLELayer:
            torch.manual_seed(7001)
            ple = Qwen4_5PLELayer(
                hidden_size=H,
                hc_count=hc,
                ple_embed_dim=ple_dim,
                ngram_size=3,
                heads_per_ngram=10,
                ngram_heads_vocab_sizes=vocab_sizes,
                split_ngram_parts=total_shards,
                dtype=dtype,
                device=device,
                sp_size=world_size,
                sp_rank=rank,
                sp_group=dist.group.WORLD,
            )
            shards = [
                torch.empty((shard_rows, ple.head_dim), dtype=dtype, device=device)
                for _ in range(total_shards)
            ]
            for idx_s, sh in enumerate(shards):
                torch.manual_seed(9100 + idx_s)
                sh.normal_(0.0, 0.05)
            if sharded:
                per_rank = total_shards // world_size
                ple.set_ngram_table(
                    shards[rank * per_rank : (rank + 1) * per_rank],
                    weight_scale=DEFAULT_PLE_WEIGHT_SCALE,
                    pin_cpu=False,
                    enable_cpu_uva=False,
                    total_shards=total_shards,
                    shard_offset=rank * per_rank,
                    ep_group=dist.group.WORLD,
                )
            else:
                ple.set_ngram_table(
                    shards,
                    weight_scale=DEFAULT_PLE_WEIGHT_SCALE,
                    pin_cpu=False,
                    enable_cpu_uva=False,
                )
            return ple

        def _build_local_ragged(in_lens, pfx_lens, s0, s1, h_full, ids_full):
            cu_global = [0]
            for ln in in_lens:
                cu_global.append(cu_global[-1] + ln)
            loc_cu = [0]
            loc_treq = []
            loc_tr = []
            loc_tt0 = []
            sub_req_indices = []
            loc_start = 0
            req0_start = 0
            cur_loc = 0
            for ri, ln in enumerate(in_lens):
                rs, re = cu_global[ri], cu_global[ri + 1]
                ov_s = max(s0, rs)
                ov_e = min(s1, re)
                if ov_e <= ov_s:
                    continue
                sub_bi = len(sub_req_indices)
                if sub_bi == 0:
                    loc_start = ov_s - rs
                    req0_start = rs
                sub_req_indices.append(ri)
                sub_ln = ov_e - ov_s
                loc_treq.extend([sub_bi] * sub_ln)
                for t0 in range(0, sub_ln, 16):
                    loc_tr.append(sub_bi)
                    loc_tt0.append(t0)
                cur_loc += sub_ln
                loc_cu.append(cur_loc)
            return (
                h_full[s0:s1].contiguous(),
                ids_full[s0:s1].contiguous(),
                torch.tensor(loc_cu, dtype=torch.int32, device=device),
                torch.tensor(loc_treq, dtype=torch.int32, device=device),
                torch.tensor(loc_tr, dtype=torch.int32, device=device),
                torch.tensor(loc_tt0, dtype=torch.int32, device=device),
                sub_req_indices,
                loc_start,
                req0_start,
            )

        # Case 1: Packed ragged [37, 29] with prefix context + straddling SP boundaries + boundary snapshots
        for sharded in (False, True):
            ple = _make_ple(sharded=sharded)
            ple_ref = _make_ple(sharded=False)
            ple_ref.sp_size = 1
            ple_ref.sp_rank = 0
            ple_ref.sp_group = None

            in_lens = [37, 29]
            pfx_lens = [11, 5]
            T_tot = sum(in_lens)
            torch.manual_seed(4201)
            h_full = torch.randn((T_tot, hc * H), dtype=dtype, device=device) * 0.25
            ids_full = torch.randint(10, 5000, (T_tot,), dtype=torch.int64, device=device)
            init_ctx = torch.randint(10, 5000, (2, 2), dtype=torch.int64, device=device)
            init_conv = torch.randn((2, hc * H, 3), dtype=dtype, device=device) * 0.1

            cu_g = torch.tensor([0, 37, 66], dtype=torch.int32, device=device)
            treq_g = torch.tensor([0] * 37 + [1] * 29, dtype=torch.int32, device=device)
            tr_g = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32, device=device)
            tt0_g = torch.tensor([0, 16, 32, 0, 16], dtype=torch.int32, device=device)
            snap_g = [15, 31, 36, 40, 55, 65]
            snap_req_g = [0, 0, 0, 1, 1, 1]
            snap_end_g = [16, 32, 37, 4, 19, 29]
            snap_phys_g = [1, 2, 3, 4, 5, 6]
            rss_g = [0, 3, 6]
            conv_pool_ref = torch.zeros((8, hc * H * 3), dtype=dtype, device=device)
            ctx_pool_ref = torch.full((8, 2), ple.eos_token_id, dtype=torch.int64, device=device)
            out_ref, _ = ple_ref.forward_packed_prefill_with_hc(
                h_full,
                ids_full,
                cu_g,
                treq_g,
                tr_g,
                tt0_g,
                prev_ngram_ctx=init_ctx,
                init_conv_state=init_conv,
                snap_tok_idx=torch.tensor(snap_g, dtype=torch.int32, device=device),
                snap_req_idx=torch.tensor(snap_req_g, dtype=torch.int32, device=device),
                conv_dst_pool=conv_pool_ref,
                conv_snap_phys=torch.tensor(snap_phys_g, dtype=torch.int32, device=device),
                req_snap_start=torch.tensor(rss_g, dtype=torch.int32, device=device),
                snap_local_ends=torch.tensor(snap_end_g, dtype=torch.int32, device=device),
                snap_aligned=False,
                ctx_dst_pool=ctx_pool_ref,
                ctx_snap_phys=torch.tensor(snap_phys_g, dtype=torch.int32, device=device),
                skip_invalid_phys=True,
                block_t=16,
            )

            s0, s1 = sp_slice_bounds(T_tot, world_size, rank)
            h_loc, ids_loc, cu_l, treq_l, tr_l, tt0_l, sub_reqs, loc_start, req0_start = _build_local_ragged(
                in_lens, pfx_lens, s0, s1, h_full, ids_full
            )
            sub_init_ctx = torch.stack([init_ctx[ri].clone() for ri in sub_reqs], dim=0)
            sub_init_conv = torch.stack([init_conv[ri].clone() for ri in sub_reqs], dim=0)
            if loc_start == 1:
                sub_init_ctx[0, 0] = sub_init_ctx[0, 1]
                sub_init_ctx[0, 1] = ids_full[s0 - 1]
            elif loc_start >= 2:
                sub_init_ctx[0, 0] = ids_full[s0 - 2]
                sub_init_ctx[0, 1] = ids_full[s0 - 1]
            if loc_start >= 3:
                sub_init_conv[0].zero_()
            elif loc_start > 0:
                keep = 3 - loc_start
                sub_init_conv[0, :, :keep] = init_conv[sub_reqs[0], :, loc_start:]
                sub_init_conv[0, :, keep:].zero_()

            cu_global = [0, 37, 66]
            loc_snaps = []
            loc_sreqs = []
            loc_sphys = []
            loc_sends = []
            loc_rss = [0]
            for sub_bi, ri in enumerate(sub_reqs):
                rs = cu_global[ri]
                for gi, g_tok in enumerate(snap_g):
                    if s0 <= g_tok < s1 and snap_req_g[gi] == ri:
                        loc_snaps.append(g_tok - s0)
                        loc_sreqs.append(sub_bi)
                        loc_sphys.append(snap_phys_g[gi])
                        loc_sends.append((g_tok - rs + 1) - (max(s0, rs) - rs))
                loc_rss.append(len(loc_snaps))
            conv_pool_sp = torch.zeros_like(conv_pool_ref)
            ctx_pool_sp = torch.full_like(ctx_pool_ref, ple.eos_token_id)

            out_loc, _ = ple.forward_packed_prefill_with_hc(
                h_loc,
                ids_loc,
                cu_l,
                treq_l,
                tr_l,
                tt0_l,
                prev_ngram_ctx=sub_init_ctx,
                init_conv_state=sub_init_conv,
                snap_tok_idx=torch.tensor(loc_snaps, dtype=torch.int32, device=device),
                snap_req_idx=torch.tensor(loc_sreqs, dtype=torch.int32, device=device),
                conv_dst_pool=conv_pool_sp,
                conv_snap_phys=torch.tensor(loc_sphys, dtype=torch.int32, device=device),
                req_snap_start=torch.tensor(loc_rss, dtype=torch.int32, device=device),
                snap_local_ends=torch.tensor(loc_sends, dtype=torch.int32, device=device),
                snap_aligned=False,
                ctx_dst_pool=ctx_pool_sp,
                ctx_snap_phys=torch.tensor(loc_sphys, dtype=torch.int32, device=device),
                skip_invalid_phys=True,
                block_t=16,
                sp_tokens_sharded=True,
                sp_total_tokens=T_tot,
                sp_halo_exchange=True,
                sp_halo_loc_start=loc_start,
                sp_halo_req0_start=req0_start,
            )
            diff_max = float((out_loc.float() - out_ref[s0:s1].float()).abs().max().item())
            assert diff_max < 2e-2, f"rank={rank} sharded={sharded} ragged diff={diff_max}"
            for p_idx in loc_sphys:
                assert torch.equal(ctx_pool_sp[p_idx], ctx_pool_ref[p_idx]), "ctx snap mismatch"
                c_diff = float((conv_pool_sp[p_idx].float() - conv_pool_ref[p_idx].float()).abs().max().item())
                assert c_diff < 2e-2, f"rank={rank} sharded={sharded} conv snap diff={c_diff}"

        # Case 2: T=3 < sp_size=4 (rank 3 gets N_loc==0, tail spans multi-rank + non-divisible gpu_sharded)
        ple_sh = _make_ple(sharded=True)
        ple_ref = _make_ple(sharded=False)
        ple_ref.sp_size = 1
        ple_ref.sp_group = None
        torch.manual_seed(4202)
        h3 = torch.randn((1, 3, hc * H), dtype=dtype, device=device) * 0.25
        ids3 = torch.randint(10, 5000, (1, 3), dtype=torch.int64, device=device)
        st_ref3 = PLELayerState(
            ngram_context=torch.full((1, 2), ple_ref.eos_token_id, dtype=torch.int64, device=device),
            conv_state=torch.zeros((1, hc * H, 3), dtype=dtype, device=device),
        )
        out_ref3 = ple_ref.forward(h3, ids3, state=st_ref3)
        s0, s1 = sp_slice_bounds(3, world_size, rank)
        st_sp3 = PLELayerState(
            ngram_context=torch.full((1, 2), ple_sh.eos_token_id, dtype=torch.int64, device=device),
            conv_state=torch.zeros((1, hc * H, 3), dtype=dtype, device=device),
        )
        out_loc3 = ple_sh.forward(
            h3[:, s0:s1].contiguous(),
            ids3[:, s0:s1].contiguous(),
            state=st_sp3,
            sp_tokens_sharded=True,
            sp_total_tokens=3,
            sp_halo_exchange=True,
        )
        if s1 > s0:
            d3 = float((out_loc3.float() - out_ref3[:, s0:s1].float()).abs().max().item())
            assert d3 < 2e-2, f"T=3 SP diff={d3}"
            assert torch.equal(st_sp3.ngram_context, st_ref3.ngram_context)
            sd3 = float((st_sp3.conv_state.float() - st_ref3.conv_state.float()).abs().max().item())
            assert sd3 < 2e-2, f"T=3 conv_state diff={sd3}"

        # Case 3: Batch-SP decode with B=3 < sp_size=4 (rank 3 gets B_loc==0) under gpu_sharded
        torch.manual_seed(4203)
        hd = torch.randn((3, 1, hc * H), dtype=dtype, device=device) * 0.25
        ids_d = torch.randint(10, 5000, (3, 1), dtype=torch.int64, device=device)
        st_ref_d = PLELayerState(
            ngram_context=torch.randint(10, 5000, (3, 2), dtype=torch.int64, device=device),
            conv_state=torch.randn((3, hc * H, 3), dtype=dtype, device=device) * 0.1,
        )
        out_ref_d = ple_ref.forward(
            hd, ids_d, state=PLELayerState(st_ref_d.ngram_context.clone(), st_ref_d.conv_state.clone())
        )
        b0, b1 = sp_slice_bounds(3, world_size, rank)
        st_sp_d = PLELayerState(
            ngram_context=st_ref_d.ngram_context[b0:b1].clone(),
            conv_state=st_ref_d.conv_state[b0:b1].clone(),
        )
        out_loc_d = ple_sh.forward(
            hd[b0:b1].contiguous(),
            ids_d[b0:b1].contiguous(),
            state=st_sp_d,
            sp_tokens_sharded=True,
            sp_total_tokens=3,
            sp_halo_exchange=False,
        )
        if b1 > b0:
            dd = float((out_loc_d.float() - out_ref_d[b0:b1].float()).abs().max().item())
            assert dd < 2e-2, f"Batch-SP decode diff={dd}"

        # Case 4: Verify Step-SP forward_verify (B=1, V=8, sp_halo_exchange=True) under gpu_sharded
        torch.manual_seed(4204)
        V_st = 8
        hv = torch.randn((1, V_st, hc * H), dtype=dtype, device=device) * 0.25
        ids_v = torch.randint(10, 5000, (1, V_st), dtype=torch.int64, device=device)
        in_ctx_v = torch.randint(10, 5000, (1, 2), dtype=torch.int64, device=device)
        in_conv_v = torch.randn((1, hc * H, 3), dtype=dtype, device=device) * 0.1
        st_ctx_ref = torch.empty((1, V_st, 2), dtype=torch.int64, device=device)
        st_conv_ref = torch.empty((1, V_st, hc * H, 3), dtype=dtype, device=device)
        out_v_ref = ple_ref.forward_verify(
            hv, ids_v, in_ctx_v.clone(), in_conv_v.clone(), stage_ctx=st_ctx_ref, stage_conv=st_conv_ref
        )
        v0, v1 = sp_slice_bounds(V_st, world_size, rank)
        v_loc = v1 - v0
        st_ctx_sp = torch.empty((1, v_loc, 2), dtype=torch.int64, device=device)
        st_conv_sp = torch.empty((1, v_loc, hc * H, 3), dtype=dtype, device=device)
        out_v_sp = ple_sh.forward_verify(
            hv[:, v0:v1].contiguous(),
            ids_v[:, v0:v1].contiguous(),
            in_ctx_v.clone(),
            in_conv_v.clone(),
            stage_ctx=st_ctx_sp,
            stage_conv=st_conv_sp,
            sp_tokens_sharded=True,
            sp_total_tokens=V_st,
            sp_halo_exchange=True,
            sp_halo_step_start=v0,
            sp_full_input_ids_2d=ids_v,
        )
        if v_loc > 0:
            dv = float((out_v_sp.float() - out_v_ref[:, v0:v1].float()).abs().max().item())
            assert dv < 2e-2, f"Verify Step-SP forward_verify diff={dv}"
            assert torch.equal(st_ctx_sp, st_ctx_ref[:, v0:v1]), "Verify Step-SP stage_ctx mismatch"
            dcv = float((st_conv_sp.float() - st_conv_ref[:, v0:v1].float()).abs().max().item())
            assert dcv < 2e-2, f"Verify Step-SP stage_conv diff={dcv}"

        if rank == 0:
            q.put(("ok", True))
    except Exception:
        import traceback
        q.put((f"rank={rank} err", traceback.format_exc()))
        raise
    finally:
        dist.destroy_process_group()


def test_r7_ple_sp_and_sharded_table() -> None:
    import multiprocessing as mp
    import os
    import tempfile

    print("=== [R7 PLE] SP Sharded Table + Boundary Halo + Empty-Rank 4-Rank Gloo ===")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    with tempfile.TemporaryDirectory(prefix="q45_ple_sp_") as td:
        init_file = os.path.join(td, "dist_init")
        procs = []
        for r in range(4):
            p = ctx.Process(target=_r7_ple_sp_worker, args=(r, 4, init_file, q))
            p.start()
            procs.append(p)
        for p in procs:
            p.join(timeout=120)
            assert p.exitcode == 0, f"PLE SP worker exit={p.exitcode}"
    msg, val = q.get(timeout=5)
    assert msg == "ok" and val is True, f"PLE SP test failed: {msg}: {val}"
    print("  [R7 PLE SP PASS] ragged [37,29] + T=3<SP + Batch-SP Decode B=3<SP (gpu_full & gpu_sharded)")


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




@torch.no_grad()
def _r8_ple_exact_halo_worker(rank: int, world_size: int, init_file: str, q: object) -> None:
    import os
    import torch.distributed as dist
    from rtp_llm.models_py.modules.qwen4_5.fast_comm import sp_slice_bounds
    from rtp_llm.models_py.modules.qwen4_5.ple import DEFAULT_PLE_WEIGHT_SCALE, Qwen4_5PLELayer

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        dtype = torch.bfloat16
        H, hc, ple_dim, total_shards, shard_rows = 128, 4, 160, 8, 256
        vocab_sizes = [64] * 20

        def _make_ple(use_sp: bool) -> Qwen4_5PLELayer:
            torch.manual_seed(4801)
            ple = Qwen4_5PLELayer(
                hidden_size=H,
                hc_count=hc,
                ple_embed_dim=ple_dim,
                ngram_size=3,
                heads_per_ngram=10,
                ngram_heads_vocab_sizes=vocab_sizes,
                split_ngram_parts=total_shards,
                dtype=dtype,
                device=device,
                sp_size=(world_size if use_sp else 1),
                sp_rank=(rank if use_sp else 0),
                sp_group=(dist.group.WORLD if use_sp else None),
            )
            shards = [torch.empty((shard_rows, ple.head_dim), dtype=dtype, device=device) for _ in range(total_shards)]
            for idx_s, sh in enumerate(shards):
                torch.manual_seed(9800 + idx_s)
                sh.normal_(0.0, 0.05)
            ple.set_ngram_table(shards, weight_scale=DEFAULT_PLE_WEIGHT_SCALE, pin_cpu=False, enable_cpu_uva=False)
            return ple

        ple_sp = _make_ple(True)
        ple_ref = _make_ple(False)
        for case_idx, T_tot in enumerate((12, 14)):
            torch.manual_seed(4850 + case_idx)
            h_full = torch.randn((T_tot, hc * H), dtype=dtype, device=device) * 0.25
            ids_full = torch.randint(10, 200, (T_tot,), dtype=torch.int64, device=device)
            init_ctx = torch.randint(10, 200, (1, 2), dtype=torch.int64, device=device)
            init_conv = torch.randn((1, hc * H, 3), dtype=dtype, device=device) * 0.1
            cu_g = torch.tensor([0, T_tot], dtype=torch.int32, device=device)
            treq_g = torch.zeros((T_tot,), dtype=torch.int32, device=device)
            tr_g = torch.zeros((1,), dtype=torch.int32, device=device)
            tt0_g = torch.zeros((1,), dtype=torch.int32, device=device)
            out_ref, _ = ple_ref.forward_packed_prefill_with_hc(
                h_full, ids_full, cu_g, treq_g, tr_g, tt0_g,
                prev_ngram_ctx=init_ctx, init_conv_state=init_conv, block_t=16,
            )

            s0, s1 = sp_slice_bounds(T_tot, world_size, rank)
            loc_len = s1 - s0
            sub_ctx = init_ctx.clone()
            sub_conv = init_conv.clone()
            if s0 == 1:
                sub_ctx[0, 0] = sub_ctx[0, 1]
                sub_ctx[0, 1] = ids_full[0]
            elif s0 >= 2:
                sub_ctx[0, 0] = ids_full[s0 - 2]
                sub_ctx[0, 1] = ids_full[s0 - 1]
            if s0 >= 3:
                sub_conv.zero_()
            elif s0 > 0:
                keep = 3 - s0
                sub_conv[0, :, :keep] = init_conv[0, :, s0:]
                sub_conv[0, :, keep:].zero_()

            cu_l = torch.tensor([0, loc_len], dtype=torch.int32, device=device)
            treq_l = torch.zeros((loc_len,), dtype=torch.int32, device=device)
            tr_l = torch.zeros((1,), dtype=torch.int32, device=device)
            tt0_l = torch.zeros((1,), dtype=torch.int32, device=device)
            out_loc, _ = ple_sp.forward_packed_prefill_with_hc(
                h_full[s0:s1].contiguous(),
                ids_full[s0:s1].contiguous(),
                cu_l, treq_l, tr_l, tt0_l,
                prev_ngram_ctx=sub_ctx,
                init_conv_state=sub_conv,
                block_t=16,
                sp_tokens_sharded=True,
                sp_total_tokens=T_tot,
                sp_halo_exchange=True,
                sp_halo_loc_start=s0,
                sp_halo_req0_start=0,
            )
            d = float((out_loc.float() - out_ref[s0:s1].float()).abs().max().item())
            assert d < 2e-2, f"R8 PLE exact halo T={T_tot} r{rank} diff={d}"
        if rank == 0:
            q.put(("ok", True))
    except Exception:
        import traceback
        q.put((f"rank={rank} err", traceback.format_exc()))
        raise
    finally:
        dist.destroy_process_group()


@torch.no_grad()
def test_r8_gdn_prefill_fast_launch_and_ple_halo_parity() -> None:
    import multiprocessing as mp
    import os
    import tempfile
    from rtp_llm.models_py.modules.qwen4_5.gdn import _GDN_TRITON_CACHE

    print("\n=== [R8 GDN & PLE] triton_fast_launch Cache Hit + Boundary Zero-Copy SP Halo ===")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(4800)

    cfg = make_config(hidden_size=5120)
    plan = YoCoSharingPlan.from_config(cfg)
    gdn = Qwen4_5GatedDeltaNet.from_config(cfg, layer_idx=0, sharing_plan=plan).to(device=device, dtype=dtype).eval()
    init_realistic_weights(gdn)

    lens = [49, 65, 64]
    (
        cu_list, cu, t_req, t_t0, c_req, c_t0, r_c_start,
        s_tok, s_req, r_s_start, s_end, s_phys, snap_meta, num_pool_blks,
    ) = _build_ragged_meta_for_test(lens, block_t=32, chunk_size=64, tpb=64, device=device)
    B = len(lens)
    N = cu_list[-1]
    h = torch.randn((N, cfg.hidden_size), device=device, dtype=dtype) * 0.25
    init_conv = torch.randn((B, gdn.conv_dim, 3), device=device, dtype=dtype) * 0.1
    init_ssm = torch.randn((B, gdn.num_v_heads, 64, 64), device=device, dtype=torch.float32) * 0.05

    _GDN_TRITON_CACHE.clear()
    conv_p1 = torch.zeros((num_pool_blks, gdn.conv_dim * 3), device=device, dtype=dtype)
    ssm_p1 = torch.zeros((num_pool_blks, gdn.num_v_heads, 64, 64), device=device, dtype=torch.float32)
    out1, _ = gdn.forward_packed_prefill(
        h, cu, t_req, t_t0, c_req, c_t0, r_c_start,
        init_conv_state=init_conv, init_ssm_state=init_ssm,
        snap_tok_idx=s_tok, snap_req_idx=s_req,
        conv_dst_pool=conv_p1, conv_snap_phys=s_phys,
        req_snap_start=r_s_start, snap_local_end=s_end,
        ssm_dst_pool=ssm_p1, ssm_snap_phys=s_phys,
    )
    cache_sz_1 = len(_GDN_TRITON_CACHE)
    assert cache_sz_1 >= 4, f"Expected >=4 triton_fast_launch cache entries, got {cache_sz_1}"

    conv_p2 = torch.zeros_like(conv_p1)
    ssm_p2 = torch.zeros_like(ssm_p1)
    out2, _ = gdn.forward_packed_prefill(
        h, cu, t_req, t_t0, c_req, c_t0, r_c_start,
        init_conv_state=init_conv, init_ssm_state=init_ssm,
        snap_tok_idx=s_tok, snap_req_idx=s_req,
        conv_dst_pool=conv_p2, conv_snap_phys=s_phys,
        req_snap_start=r_s_start, snap_local_end=s_end,
        ssm_dst_pool=ssm_p2, ssm_snap_phys=s_phys,
    )
    assert len(_GDN_TRITON_CACHE) == cache_sz_1
    assert float((out1 - out2).abs().max().item()) == 0.0
    assert float((conv_p1 - conv_p2).abs().max().item()) == 0.0
    assert float((ssm_p1 - ssm_p2).abs().max().item()) == 0.0

    for bi, _ln in enumerate(lens):
        st = gdn.create_state(1, device, dtype)
        st.conv_state.copy_(init_conv[bi : bi + 1])
        st.ssm_state.copy_(init_ssm[bi : bi + 1])
        y_ref, _ = gdn(h[cu_list[bi] : cu_list[bi + 1]].unsqueeze(0), state=st)
        d, _ = report_diff(f"R8 GDN ragged req={bi}", out1[cu_list[bi] : cu_list[bi + 1]], y_ref.squeeze(0))
        assert d < 1e-2
    for bi, bnd, phys in snap_meta:
        st_b = gdn.create_state(1, device, dtype)
        st_b.conv_state.copy_(init_conv[bi : bi + 1])
        st_b.ssm_state.copy_(init_ssm[bi : bi + 1])
        _ = gdn(h[cu_list[bi] : cu_list[bi] + bnd].unsqueeze(0), state=st_b)
        assert float((conv_p1[phys].view_as(st_b.conv_state[0]).float() - st_b.conv_state[0].float()).abs().max().item()) < 1e-2
        assert float((ssm_p1[phys].float() - st_b.ssm_state[0].float()).abs().max().item()) < 1e-2

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    with tempfile.TemporaryDirectory() as tmpd:
        init_file = os.path.join(tmpd, "r8_ple_sp_init")
        procs = [ctx.Process(target=_r8_ple_exact_halo_worker, args=(rk, 4, init_file, q)) for rk in range(4)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)
            assert p.exitcode == 0, f"R8 PLE SP worker exit={p.exitcode}"
    msg = q.get(timeout=5)
    assert msg == ("ok", True), f"R8 PLE worker err: {msg}"
    print("  [R8 GDN & PLE PASS] triton_fast_launch ragged+snap & N==halo_w SP zero-copy halo verified.")


def main() -> None:
    t0 = time.time()
    test_yoco_sharing_plan_and_param_shapes()
    test_full_attention_producer_and_consumer()
    test_gdn_kernels_and_module()
    test_yoco_shared_gdn_source_and_consumer()
    test_gdn_fused_unpack_verify_ragged_and_tp()
    test_r3_gdn_stage2_sub_dv_and_verify_nw1()
    test_r4_rmsnorm_gated_packed_yoco_subdv_and_prefill_tuning()
    test_r5_gdn_verify_readout_stage1_and_fused_projections()
    test_r6_gdn_decode_opaque_and_direct_chunk64_snaps()
    test_r7_ple_sp_and_sharded_table()
    test_r8_gdn_prefill_fast_launch_and_ple_halo_parity()
    run_b300_benchmarks()
    print(f"\nALL TESTS & BENCHMARKS PASSED in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()

"""Unit tests & B300 GPU micro-benchmarks for Qwen 4.5-Flash HC, ShortConv, and PLE."""

from __future__ import annotations

import sys
import time
from typing import Callable, List, Tuple

import torch

from rtp_llm.models_py.modules.qwen4_5.contracts import PLELayerState, ShortConvState
from rtp_llm.models_py.modules.qwen4_5.hyper_connection import (
    Qwen4_5HyperConnection,
    grouped_gemma_rmsnorm_bias_ref,
    grouped_gemma_rmsnorm_bias_triton,
    hc_combine_ref,
    hc_combine_triton,
)
from rtp_llm.models_py.modules.qwen4_5.ple import (
    DEFAULT_PLE_WEIGHT_SCALE,
    Qwen4_5PLELayer,
    compute_ngram_vocab_sizes,
)
from rtp_llm.models_py.modules.qwen4_5.short_conv import Qwen4_5ShortConv


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    af = a.float()
    bf = b.float()
    abs_diff = (af - bf).abs()
    max_abs = float(abs_diff.max().item()) if abs_diff.numel() else 0.0
    rms_rel = float(
        (abs_diff.square().sum().sqrt() / bf.square().sum().sqrt().clamp_min(1e-12)).item()
    ) if abs_diff.numel() else 0.0
    return max_abs, rms_rel


def _bench_us(fn: Callable[[], object], warmup: int = 20, iters: int = 100) -> float:
    """Pure GPU execution time per call via CUDA Graph replay (us)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


# ---------------------------------------------------------------------------
# 1. Hyper-Connection Tests
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_hyper_connection() -> None:
    print("\n=== [1] Testing Qwen4_5HyperConnection (B300 CUDA BF16 vs FP32 ref) ===")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(42)

    H = 5120
    HC = 4
    LR = 128
    B, T = 2, 16

    # 1a. Standalone grouped_gemma_rmsnorm_bias kernel check
    x_norm_in = torch.randn(B, T, HC * H, device=device, dtype=dtype) * 1.5
    w_norm = torch.randn(HC * H, device=device, dtype=dtype) * 0.2
    b_norm = torch.randn(HC * H, device=device, dtype=dtype) * 0.1
    y_triton = grouped_gemma_rmsnorm_bias_triton(x_norm_in, w_norm, b_norm, HC, eps=1e-6)
    y_ref = grouped_gemma_rmsnorm_bias_ref(x_norm_in, w_norm, b_norm, HC, eps=1e-6)
    max_abs, max_rel = _diff_stats(y_triton, y_ref)
    print(f"  grouped_gemma_rmsnorm_bias_triton: max_abs={max_abs:.6f}, max_rel={max_rel:.6f}")
    assert max_abs < 1.5e-2, f"RMSNorm+bias diff too large: {max_abs}"

    # 1b. Standalone hc_combine kernel check
    block_out = torch.randn(B, T, H, device=device, dtype=dtype) * 0.5
    inj_w = torch.rand(B, T, HC, device=device, dtype=dtype) * 2.0
    c_triton = hc_combine_triton(block_out, x_norm_in, inj_w)
    c_ref = hc_combine_ref(block_out, x_norm_in, inj_w)
    max_abs, max_rel = _diff_stats(c_triton, c_ref)
    print(f"  hc_combine_triton:                 max_abs={max_abs:.6f}, max_rel={max_rel:.6f}")
    assert max_abs < 1.5e-2, f"hc_combine diff too large: {max_abs}"

    # 1c. All 4 architectural HC modes
    configs = [
        ("Layer0_head (is_head=True, prev_mix=True)", dict(is_head=True, prev_mix=True, emits_latent=True, use_combine=True)),
        ("Normal_layer (prev_mix=True)", dict(is_head=False, prev_mix=True, emits_latent=True, use_combine=True)),
        ("Layer39_mlp (emits_latent=False)", dict(is_head=False, prev_mix=True, emits_latent=False, use_combine=True)),
        ("Final_mixer (use_combine=False, prev_mix=False)", dict(is_head=False, prev_mix=False, emits_latent=True, use_combine=False)),
    ]

    for name, kwargs in configs:
        hc = Qwen4_5HyperConnection(
            hidden_size=H,
            hc_count=HC,
            hc_lowrank=LR,
            dtype=dtype,
            device=device,
            **kwargs,
        )
        # Non-zero norm weight & bias to test GemmaRMSNorm (1 + w) + b thoroughly
        hc.hc_norm_weight.normal_(0.0, 0.1)
        hc.hc_norm_bias.normal_(0.0, 0.05)

        hyper_in = torch.randn(B, T, HC * H, device=device, dtype=dtype) * 0.8
        prev_lat = torch.randn(B, T, LR, device=device, dtype=dtype) * 0.5

        mixed, res, inj, nxt = hc.mix_with_latent(hyper_in, prev_latent=prev_lat)
        mixed_r, res_r, inj_r, nxt_r = hc.mix_with_latent_ref(hyper_in, prev_latent=prev_lat)

        assert res.data_ptr() == hyper_in.data_ptr()
        m_abs, m_rel = _diff_stats(mixed, mixed_r)
        assert m_abs < 1.5e-2, f"{name} mixed diff {m_abs}"

        if kwargs["use_combine"]:
            assert inj is not None and inj_r is not None
            i_abs, i_rel = _diff_stats(inj, inj_r)
            assert i_abs < 1.5e-2, f"{name} inject_w diff {i_abs}"
        else:
            assert inj is None and inj_r is None
            i_abs, i_rel = 0.0, 0.0

        if kwargs["emits_latent"]:
            assert nxt is not None and nxt_r is not None
            l_abs, l_rel = _diff_stats(nxt, nxt_r)
            assert l_abs < 1.5e-2, f"{name} next_latent diff {l_abs}"
        else:
            assert nxt is None and nxt_r is None
            l_abs, l_rel = 0.0, 0.0

        # Also verify combine_and_mix_with_latent (fused combine + norm + mix)
        prev_inj = torch.rand(B, T, HC, device=device, dtype=dtype) * 2.0
        sub_out = torch.randn(B, T, H, device=device, dtype=dtype) * 0.5
        cm_m, cm_res, cm_inj, cm_nxt = hc.combine_and_mix_with_latent(
            sub_out, hyper_in, prev_inj, prev_latent=prev_lat
        )
        cm_mr, cm_resr, cm_injr, cm_nxtr = hc.combine_and_mix_with_latent_ref(
            sub_out, hyper_in, prev_inj, prev_latent=prev_lat
        )
        c_m_abs, _ = _diff_stats(cm_m, cm_mr)
        c_r_abs, _ = _diff_stats(cm_res, cm_resr)
        assert c_m_abs < 1.5e-2 and c_r_abs < 1.5e-2, (
            f"{name} combine_and_mix diff: mixed={c_m_abs}, res={c_r_abs}"
        )
        if cm_inj is not None and cm_injr is not None:
            assert _diff_stats(cm_inj, cm_injr)[0] < 1.5e-2
        if cm_nxt is not None and cm_nxtr is not None:
            assert _diff_stats(cm_nxt, cm_nxtr)[0] < 1.5e-2

        print(
            f"  [PASS] {name:48s} | mixed={m_abs:.5f} (rel={m_rel:.5f}) "
            f"| inj={i_abs:.5f} | lat={l_abs:.5f} | fused_cm={c_m_abs:.5f}"
        )


# ---------------------------------------------------------------------------
# 2. Short-Conv Tests
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_short_conv() -> None:
    print("\n=== [2] Testing Qwen4_5ShortConv (Prefill/Decode Triton vs Ref & Step Equivalence) ===")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(123)

    B = 2
    T_full = 16
    T_prefill = 10

    for channels in (256, 5120, 20480):
        for residual in (False, True):
            for use_act in (False, True):
                sc = Qwen4_5ShortConv(
                    channels=channels,
                    kernel_size=4,
                    residual=residual,
                    use_activation=use_act,
                    dtype=dtype,
                    device=device,
                )
                sc.weight.normal_(0.0, 0.25)

                x = torch.randn(B, T_full, channels, device=device, dtype=dtype)

                # 2a. Full prefill: Triton vs FP32 PyTorch ref
                st_tri_full = sc.alloc_state(B, device=device, dtype=dtype)
                st_ref_full = sc.alloc_state(B, device=device, dtype=dtype)
                y_tri_full = sc(x, state=st_tri_full)
                y_ref_full = sc.forward_ref(x, state=st_ref_full)
                pf_abs, pf_rel = _diff_stats(y_tri_full, y_ref_full)
                st_abs, _ = _diff_stats(st_tri_full.conv_state, st_ref_full.conv_state)
                assert pf_abs < 1e-2, f"C={channels} res={residual} act={use_act} prefill diff {pf_abs}"
                assert st_abs == 0.0, f"conv_state mismatch {st_abs}"

                # 2b. Prefill(T=10) + 6x Decode(1) == Prefill(T=16)
                st_inc = sc.alloc_state(B, device=device, dtype=dtype)
                y_pre = sc(x[:, :T_prefill, :], state=st_inc)
                dec_outs = [y_pre]
                for step in range(T_prefill, T_full):
                    y_step = sc(x[:, step : step + 1, :], state=st_inc)
                    dec_outs.append(y_step)
                y_inc = torch.cat(dec_outs, dim=1)

                eq_tri_abs, eq_tri_rel = _diff_stats(y_inc, y_tri_full)
                eq_ref_abs, _ = _diff_stats(y_inc, y_ref_full)
                eq_st_abs, _ = _diff_stats(st_inc.conv_state, st_tri_full.conv_state)
                assert eq_tri_abs < 1e-2, (
                    f"C={channels} res={residual} act={use_act} Prefill==Pre+Dec diff {eq_tri_abs}"
                )
                assert eq_ref_abs < 1e-2, (
                    f"C={channels} res={residual} act={use_act} Inc vs Ref diff {eq_ref_abs}"
                )
                assert eq_st_abs == 0.0, f"Final conv_state mismatch {eq_st_abs}"

                # 2c. Multi-step decode chunks (T=9 + T=2 + T=1 + T=4 == T=16)
                st_chunk = sc.alloc_state(B, device=device, dtype=dtype)
                y_c = torch.cat(
                    [
                        sc(x[:, :9, :], state=st_chunk),
                        sc(x[:, 9:11, :], state=st_chunk),
                        sc(x[:, 11:12, :], state=st_chunk),
                        sc(x[:, 12:16, :], state=st_chunk),
                    ],
                    dim=1,
                )
                chunk_abs, _ = _diff_stats(y_c, y_tri_full)
                assert chunk_abs < 1e-2, f"Multi-step chunk diff {chunk_abs}"

                print(
                    f"  [PASS] C={channels:5d} res={str(residual):5s} silu={str(use_act):5s} | "
                    f"tri_vs_ref={pf_abs:.5f} (rel={pf_rel:.5f}) | "
                    f"pre16==pre10+6dec={eq_tri_abs:.5f} | chunk={chunk_abs:.5f}"
                )


# ---------------------------------------------------------------------------
# 3. PLE N-gram Embedding Tests
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_ple_layer() -> None:
    print("\n=== [3] Testing Qwen4_5PLELayer (EOS Hashing, FP8 Gather CUDA/CPU-UVA/CPU-H2D, Pre==Pre+Dec) ===")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(777)

    H = 5120
    HC = 4
    PLE_DIM = 5120
    HEADS_PER_NGRAM = 10
    NUM_HEADS = 20
    HEAD_DIM = PLE_DIM // NUM_HEADS  # 256
    EOS = 248044
    B = 2
    T_full = 16
    T_prefill = 10

    # Small primes for unit test (total_vocab ~ 25k -> 128 shards x 256 rows = 32,768 rows)
    small_primes = compute_ngram_vocab_sizes(1024, NUM_HEADS)
    total_vocab = sum(small_primes)
    num_shards = 128
    shard_rows = (total_vocab + num_shards - 1) // num_shards
    total_rows = shard_rows * num_shards

    ple = Qwen4_5PLELayer(
        hidden_size=H,
        hc_count=HC,
        ple_embed_dim=PLE_DIM,
        ngram_size=3,
        heads_per_ngram=HEADS_PER_NGRAM,
        conv_kernel_size=4,
        split_ngram_parts=num_shards,
        eos_token_id=EOS,
        ngram_heads_vocab_sizes=small_primes,
        weight_scale=DEFAULT_PLE_WEIGHT_SCALE,
        dtype=dtype,
        device=device,
    )
    ple.norm_key.normal_(0.0, 0.1)
    ple.norm_query.normal_(0.0, 0.1)
    ple.norm_conv.normal_(0.0, 0.1)
    ple.conv1d.weight.normal_(0.0, 0.15)

    # Create deterministic FP8 table
    raw_fp8_cpu = (torch.randn(total_rows, HEAD_DIM, dtype=torch.float32) * 120.0).clamp(-448, 448).to(
        torch.float8_e4m3fn
    )
    raw_fp8_cuda = raw_fp8_cpu.to(device)
    shards_cuda = [raw_fp8_cuda[i * shard_rows : (i + 1) * shard_rows].contiguous() for i in range(num_shards)]
    shards_cpu_unpinned = [raw_fp8_cpu[i * shard_rows : (i + 1) * shard_rows].contiguous() for i in range(num_shards)]
    shards_cpu_pinned = [s.clone().pin_memory() for s in shards_cpu_unpinned]

    # Build challenging token sequence with EOS at start, mid-prefill, split boundary (t=8, 9, 10), and decode!
    input_ids = torch.randint(10, 240000, (B, T_full), device=device, dtype=torch.int64)
    input_ids[0, 0] = EOS        # EOS at position 0
    input_ids[0, 4] = EOS        # EOS mid-prefill
    input_ids[0, 9] = EOS        # EOS at last token of prefill (t=9, right before decode!)
    input_ids[0, 12] = EOS       # EOS mid-decode
    input_ids[1, 8] = EOS        # EOS at t=8 (second-to-last token of prefill!)
    input_ids[1, 10] = EOS       # EOS at first decode token!
    input_ids[1, 11] = EOS       # Consecutive EOS in decode!

    hidden_states = torch.randn(B, T_full, HC * H, device=device, dtype=dtype) * 0.7

    # 3a. Bit-exact int64 hash verification: Triton vs _shift_right_ignore_eos ref (full + incremental)
    ids_tri_full = ple.compute_ngram_ids(input_ids, state=None)
    ids_ref_full = ple.compute_ngram_ids_ref(input_ids, state=None)
    assert torch.equal(ids_tri_full, ids_ref_full), "Triton hash != PyTorch _shift_right_ignore_eos!"

    st_hash_tri = ple.alloc_state(B, device=device, dtype=dtype)
    st_hash_ref = ple.alloc_state(B, device=device, dtype=dtype)
    h_pre_tri = ple.compute_ngram_ids(input_ids[:, :T_prefill], state=st_hash_tri)
    h_pre_ref = ple.compute_ngram_ids_ref(input_ids[:, :T_prefill], state=st_hash_ref)
    assert torch.equal(h_pre_tri, h_pre_ref)
    assert torch.equal(st_hash_tri.ngram_context, st_hash_ref.ngram_context)
    h_steps = [h_pre_tri]
    for step in range(T_prefill, T_full):
        h_s_tri = ple.compute_ngram_ids(input_ids[:, step : step + 1], state=st_hash_tri)
        h_s_ref = ple.compute_ngram_ids_ref(input_ids[:, step : step + 1], state=st_hash_ref)
        assert torch.equal(h_s_tri, h_s_ref), f"Decode step {step} hash mismatch!"
        assert torch.equal(st_hash_tri.ngram_context, st_hash_ref.ngram_context)
        h_steps.append(h_s_tri)
    assert torch.equal(torch.cat(h_steps, dim=1), ids_ref_full), "Incremental hash != Full prefill hash!"
    print("  [PASS] N-gram hash bit-exact match (EOS at t=0,4,8,9,10,11,12; Prefill16 == Prefill10+6xDec1)")

    # 3b. Verify FP8 gather bit-exact equivalence across all 4 table backends
    ple.set_ngram_table(raw_fp8_cuda)
    emb_cuda_flat = ple.gather_embeddings(ids_tri_full, out_dtype=dtype)

    ple.set_ngram_table(shards_cuda)
    emb_cuda_sharded = ple.gather_embeddings(ids_tri_full, out_dtype=dtype)

    ple.set_ngram_table(shards_cpu_unpinned, pin_cpu=False, enable_cpu_uva=False)
    emb_cpu_h2d = ple.gather_embeddings(ids_tri_full, out_dtype=dtype)

    ple.set_ngram_table(shards_cpu_pinned, pin_cpu=False, enable_cpu_uva=True)
    assert ple._ensure_pointer_table(device) is not None and ple._cpu_uva_active, (
        "Expected CPU pinned UVA to be active!"
    )
    emb_cpu_uva = ple.gather_embeddings(ids_tri_full, out_dtype=dtype)

    assert torch.equal(emb_cuda_flat, emb_cuda_sharded), "CUDA flat vs sharded gather mismatch!"
    assert torch.equal(emb_cuda_flat, emb_cpu_h2d), "CUDA vs CPU unpinned H2D gather mismatch!"
    assert torch.equal(emb_cuda_flat, emb_cpu_uva), "CUDA vs CPU pinned UVA gather mismatch!"
    print("  [PASS] FP8 gather bit-exact across CUDA-flat, CUDA-128shards, CPU-H2D, and CPU-pinned-UVA")

    # 3c. Full PLE forward vs FP32 PyTorch ref & Prefill(16) == Prefill(10) + 6x Decode(1)
    for mode_name, table_obj, pin_cpu, uva in [
        ("CUDA_128shards_FP8", shards_cuda, False, True),
        ("CPU_unpinned_H2D_FP8", shards_cpu_unpinned, False, False),
        ("CPU_pinned_UVA_FP8", shards_cpu_pinned, False, True),
    ]:
        ple.set_ngram_table(table_obj, pin_cpu=pin_cpu, enable_cpu_uva=uva)

        st_full_tri = ple.alloc_state(B, device=device, dtype=dtype)
        st_full_ref = ple.alloc_state(B, device=device, dtype=dtype)
        out_tri_full = ple(hidden_states, input_ids, state=st_full_tri)
        out_ref_full = ple.forward_ref(hidden_states, input_ids, state=st_full_ref)

        ref_abs, ref_rel = _diff_stats(out_tri_full, out_ref_full)
        ref_mean = float((out_tri_full.float() - out_ref_full.float()).abs().mean().item())
        # At |y| in [2, 4), 1 BF16 ULP = 0.015625; verify <= 1 ULP max & < 1e-4 mean
        assert ref_abs <= 0.015625 and ref_mean < 1e-4, (
            f"{mode_name} Triton vs FP32 ref diff max={ref_abs}, mean={ref_mean}"
        )

        # Prefill(10) + 6x Decode(1)
        st_inc = ple.alloc_state(B, device=device, dtype=dtype)
        out_pre = ple(hidden_states[:, :T_prefill, :], input_ids[:, :T_prefill], state=st_inc)
        outs = [out_pre]
        for step in range(T_prefill, T_full):
            out_s = ple(
                hidden_states[:, step : step + 1, :],
                input_ids[:, step : step + 1],
                state=st_inc,
            )
            outs.append(out_s)
        out_inc = torch.cat(outs, dim=1)

        eq_abs, eq_rel = _diff_stats(out_inc, out_tri_full)
        ctx_eq = torch.equal(st_inc.ngram_context, st_full_tri.ngram_context)
        conv_st_abs, _ = _diff_stats(st_inc.conv_state, st_full_tri.conv_state)
        assert eq_abs < 1e-2, f"{mode_name} Prefill16 == Prefill10+6xDec1 diff {eq_abs}"
        assert ctx_eq, f"{mode_name} final ngram_context mismatch!"
        assert conv_st_abs < 1e-2, f"{mode_name} final conv_state diff {conv_st_abs}"

        print(
            f"  [PASS] {mode_name:22s} | tri_vs_ref={ref_abs:.6f} (mean={ref_mean:.2e}, rel={ref_rel:.5f}) "
            f"| pre16==pre10+6dec={eq_abs:.6f} | conv_st_diff={conv_st_abs:.6f}"
        )

    # 3d. FP32 mathematical exactness check (proves zero kernel formula discrepancy)
    ple_f32 = Qwen4_5PLELayer(
        hidden_size=H,
        hc_count=HC,
        ple_embed_dim=PLE_DIM,
        ngram_size=3,
        heads_per_ngram=HEADS_PER_NGRAM,
        conv_kernel_size=4,
        split_ngram_parts=num_shards,
        eos_token_id=EOS,
        ngram_heads_vocab_sizes=small_primes,
        weight_scale=DEFAULT_PLE_WEIGHT_SCALE,
        dtype=torch.float32,
        device=device,
    )
    ple_f32.norm_key.normal_(0.0, 0.1)
    ple_f32.norm_query.normal_(0.0, 0.1)
    ple_f32.norm_conv.normal_(0.0, 0.1)
    ple_f32.conv1d.weight.normal_(0.0, 0.15)
    ple_f32.set_ngram_table(shards_cuda)
    h_f32 = hidden_states.float()
    out_f32_tri = ple_f32(h_f32, input_ids)
    out_f32_ref = ple_f32.forward_ref(h_f32, input_ids)
    f32_abs, f32_rel = _diff_stats(out_f32_tri, out_f32_ref)
    assert f32_abs < 1e-4, f"PLE FP32 exactness diff too large: {f32_abs}"
    print(f"  [PASS] FP32_exact_math        | tri_vs_ref={f32_abs:.2e} (rel={f32_rel:.2e})")


# ---------------------------------------------------------------------------
# 4. B300 GPU Micro-Benchmarks at REAL Qwen 4.5-Flash Shapes
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_benchmarks() -> None:
    print("\n=== [4] B300 (SM103) CUDA-Graph Pure-GPU Micro-Benchmarks at REAL Qwen 4.5-Flash Shapes ===")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    H = 5120
    HC = 4
    LR = 128
    shapes = [(1, 1), (32, 1), (1, 1024)]

    # 4a. Hyper-Connection benchmarks
    hc = Qwen4_5HyperConnection(
        hidden_size=H, hc_count=HC, hc_lowrank=LR, dtype=dtype, device=device
    ).eval()
    print("\n  [HyperConnection] hidden=5120, hc=4, lowrank=128:")
    for B, T in shapes:
        x = torch.randn(B, T, HC * H, device=device, dtype=dtype)
        sub = torch.randn(B, T, H, device=device, dtype=dtype)
        inj = torch.rand(B, T, HC, device=device, dtype=dtype) * 2.0
        plat = torch.randn(B, T, LR, device=device, dtype=dtype)

        t_norm = _bench_us(
            lambda: grouped_gemma_rmsnorm_bias_triton(
                x, hc.hc_norm_weight, hc.hc_norm_bias, HC, 1e-6
            )
        )
        t_mix = _bench_us(lambda: hc.mix_with_latent(x, prev_latent=plat))
        t_comb = _bench_us(lambda: hc.combine(sub, x, inj))
        t_fused_cm = _bench_us(
            lambda: hc.combine_and_mix_with_latent(sub, x, inj, prev_latent=plat)
        )
        print(
            f"    B={B:2d}, T={T:4d} | norm={t_norm:7.2f} us | mix={t_mix:7.2f} us | "
            f"combine={t_comb:7.2f} us | combine+mix={t_fused_cm:7.2f} us"
        )

    # 4b. ShortConv benchmarks
    print("\n  [ShortConv] kernel_size=4:")
    for C, res, label in [
        (256, True, "KV_sconv(C=256, res=True)"),
        (5120, True, "Out_sconv(C=5120, res=True)"),
        (20480, False, "PLE_conv(C=20480, res=False)"),
    ]:
        sc = Qwen4_5ShortConv(
            channels=C, kernel_size=4, residual=res, use_activation=False, dtype=dtype, device=device
        ).eval()
        for B, T in shapes:
            x = torch.randn(B, T, C, device=device, dtype=dtype)
            st = sc.alloc_state(B, device=device, dtype=dtype)
            t_sc = _bench_us(lambda: sc(x, state=st))
            print(f"    {label:28s} B={B:2d}, T={T:4d} | latency={t_sc:7.2f} us")

    # 4c. PLE Layer benchmarks (both CUDA table and CPU Pinned UVA 128-shard table)
    print("\n  [PLELayer] hidden=5120, hc=4, ple_dim=5120, 20 heads x 256, FP8 per-tensor:")
    bench_primes = compute_ngram_vocab_sizes(65536, 20)
    total_v = sum(bench_primes)
    n_shards = 128
    s_rows = (total_v + n_shards - 1) // n_shards
    ple = Qwen4_5PLELayer(
        hidden_size=H,
        hc_count=HC,
        ple_embed_dim=5120,
        ngram_size=3,
        heads_per_ngram=10,
        conv_kernel_size=4,
        split_ngram_parts=n_shards,
        ngram_heads_vocab_sizes=bench_primes,
        weight_scale=DEFAULT_PLE_WEIGHT_SCALE,
        dtype=dtype,
        device=device,
    ).eval()

    # CUDA 128-shard FP8 table (~340 MB)
    cuda_shards = [
        torch.randint(-100, 100, (s_rows, 256), dtype=torch.int8, device=device).view(torch.float8_e4m3fn)
        for _ in range(n_shards)
    ]
    ple.set_ngram_table(cuda_shards)
    for B, T in shapes:
        h_in = torch.randn(B, T, HC * H, device=device, dtype=dtype)
        toks = torch.randint(0, 248000, (B, T), device=device, dtype=torch.int64)
        st = ple.alloc_state(B, device=device, dtype=dtype)
        t_ple = _bench_us(lambda: ple(h_in, toks, state=st))
        print(f"    PLE (CUDA 128-shard FP8)    B={B:2d}, T={T:4d} | latency={t_ple:7.2f} us")

    # Free CUDA shards before pinning CPU shards
    del cuda_shards
    torch.cuda.empty_cache()

    cpu_pinned_shards = [
        torch.randint(-100, 100, (s_rows, 256), dtype=torch.int8).view(torch.float8_e4m3fn).pin_memory()
        for _ in range(n_shards)
    ]
    ple.set_ngram_table(cpu_pinned_shards, pin_cpu=False, enable_cpu_uva=True)
    for B, T in shapes:
        h_in = torch.randn(B, T, HC * H, device=device, dtype=dtype)
        toks = torch.randint(0, 248000, (B, T), device=device, dtype=torch.int64)
        st = ple.alloc_state(B, device=device, dtype=dtype)
        t_ple_uva = _bench_us(lambda: ple(h_in, toks, state=st))
        print(f"    PLE (CPU-UVA 128-shard FP8) B={B:2d}, T={T:4d} | latency={t_ple_uva:7.2f} us")


def main() -> int:
    print(
        f"Device: {torch.cuda.get_device_name(0)} "
        f"(SM {torch.cuda.get_device_capability(0)}), "
        f"PyTorch {torch.__version__}"
    )
    t0 = time.perf_counter()
    test_hyper_connection()
    test_short_conv()
    test_ple_layer()
    run_benchmarks()
    print(f"\nALL TESTS & BENCHMARKS PASSED in {time.perf_counter() - t0:.2f}s!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

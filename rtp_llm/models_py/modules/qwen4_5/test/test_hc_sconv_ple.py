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
    ple_add3_and_hc_norm_triton,
    ple_add3_triton,
    ple_gate_and_norm_triton,
)
from rtp_llm.models_py.modules.qwen4_5.short_conv import (
    Qwen4_5ShortConv,
    causal_dwconv1d_prefill_with_snapshots_triton,
    causal_dwconv1d_ragged_prefill_with_snapshots_triton,
    causal_dwconv1d_verify_triton,
)


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
# 3e. Phase 1 & 5 Tests: ShortConv/PLE Packed Prefill, Verify + CG, TP Hooks
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_short_conv_verify_and_packed_prefill() -> None:
    print("\n=== [3b] ShortConv Verify (CUDA Graph + Rollback) & Packed-Snapshot Prefill ===")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(2026)

    C = 5120
    B, T_total = 2, 192
    for residual, use_act in [(True, False), (True, True)]:
        sc = Qwen4_5ShortConv(
            channels=C,
            kernel_size=4,
            residual=residual,
            use_activation=use_act,
            dtype=dtype,
            device=device,
        ).eval()
        sc.weight.normal_(0.0, 0.15)
        x = torch.randn((B, T_total, C), device=device, dtype=dtype)
        init_st = torch.randn((B, C, 3), device=device, dtype=dtype)

        # 1. Whole-sequence causal_dwconv1d_prefill_with_snapshots_triton vs 64-token segmented prefill
        out_snap, snaps = causal_dwconv1d_prefill_with_snapshots_triton(
            x,
            sc.weight,
            init_state=init_st,
            seq_size_per_block=64,
            prefix_lengths=[64, 0],
            residual=residual,
            use_activation=use_act,
            clamp_val=1000.0,
        )
        assert snaps.shape == (6, C, 3), f"Expected 6 boundary snapshots, got {snaps.shape}"

        st_seg = ShortConvState(init_st.clone())
        seg_outs = []
        seg_snaps = []
        for s0 in range(0, T_total, 64):
            seg_outs.append(
                sc(x[:, s0 : s0 + 64], state=st_seg).clamp(-1000.0, 1000.0)
            )
            seg_snaps.append(st_seg.conv_state.clone())
        out_seg = torch.cat(seg_outs, dim=1)
        out_abs, _ = _diff_stats(out_snap, out_seg)
        assert out_abs < 1e-2, f"ShortConv prefill snapshot out diff: {out_abs}"
        for b_idx in range(B):
            for s_idx in range(3):
                snap_abs, _ = _diff_stats(snaps[b_idx * 3 + s_idx], seg_snaps[s_idx][b_idx])
                assert snap_abs == 0.0, f"ShortConv snapshot b={b_idx} s={s_idx} diff={snap_abs}"

        # Also check seq_size_per_block=16 snapshot count
        _, snaps_16 = causal_dwconv1d_prefill_with_snapshots_triton(
            x[:, :32],
            sc.weight,
            init_state=init_st,
            seq_size_per_block=16,
            residual=residual,
            use_activation=use_act,
        )
        assert snaps_16.shape[0] == 4, f"Expected 4 snaps at tpb=16, got {snaps_16.shape}"

        # 2. Ragged prefill with non-uniform lengths [64, 70, 128] & opaque pool boundary snapshots
        lens = [64, 70, 128]
        cu_list = [0]
        for l in lens:
            cu_list.append(cu_list[-1] + l)
        cu = torch.tensor(cu_list, device=device, dtype=torch.int32)
        t_req, t_t0, s_tok, s_req, s_dst = [], [], [], [], []
        blk_t = 32
        snap_meta = []
        blk_id = 1
        for bi, l in enumerate(lens):
            for t0 in range(0, l, blk_t):
                t_req.append(bi)
                t_t0.append(t0)
            for bnd in range(64, l + 1, 64):
                s_tok.append(cu_list[bi] + bnd - 1)
                s_req.append(bi)
                s_dst.append(blk_id)
                snap_meta.append((bi, bnd, blk_id))
                blk_id += 1
        x_rag = torch.cat([torch.randn((l, C), device=device, dtype=dtype) for l in lens], dim=0)
        init_rag = torch.randn((len(lens), C, 3), device=device, dtype=dtype)
        dst_pool = torch.zeros((blk_id + 1, C * 3), device=device, dtype=dtype)
        out_rag = causal_dwconv1d_ragged_prefill_with_snapshots_triton(
            x_rag,
            sc.weight,
            cu,
            torch.tensor(t_req, device=device, dtype=torch.int32),
            torch.tensor(t_t0, device=device, dtype=torch.int32),
            init_state=init_rag,
            snap_tok_idx=torch.tensor(s_tok, device=device, dtype=torch.int32),
            snap_req_idx=torch.tensor(s_req, device=device, dtype=torch.int32),
            dst_pool=dst_pool,
            snap_dst_idx=torch.tensor(s_dst, device=device, dtype=torch.int32),
            residual=residual,
            use_activation=use_act,
            clamp_val=1000.0,
            block_t=blk_t,
        )
        for bi, l in enumerate(lens):
            st_ref = ShortConvState(init_rag[bi : bi + 1].clone())
            y_ref = sc(x_rag[cu_list[bi] : cu_list[bi + 1]].unsqueeze(0), state=st_ref).squeeze(0).clamp(-1000.0, 1000.0)
            rd, _ = _diff_stats(out_rag[cu_list[bi] : cu_list[bi + 1]], y_ref)
            assert rd < 1e-2, f"Ragged ShortConv req={bi} diff={rd}"
        for bi, bnd, phys in snap_meta:
            st_bnd = ShortConvState(init_rag[bi : bi + 1].clone())
            _ = sc(x_rag[cu_list[bi] : cu_list[bi] + bnd].unsqueeze(0), state=st_bnd)
            sd, _ = _diff_stats(dst_pool[phys].view(C, 3), st_bnd.conv_state[0])
            assert sd == 0.0, f"Ragged ShortConv snapshot diff req={bi} bnd={bnd}: {sd}"

        # 3. Fused Verify (B=4, V=8) + rollback states + CUDA Graph capture/replay
        Bv, V = 4, 8
        xv = torch.randn((Bv, V, C), device=device, dtype=dtype)
        st_v_init = torch.randn((Bv, C, 3), device=device, dtype=dtype)
        stage_v = torch.empty((Bv, V, C, 3), device=device, dtype=dtype)
        out_v = causal_dwconv1d_verify_triton(
            xv,
            sc.weight,
            st_v_init,
            stage_v,
            residual=residual,
            use_activation=use_act,
            clamp_val=1000.0,
        )
        st_dec = ShortConvState(st_v_init.clone())
        dec_steps = []
        for s in range(V):
            y_s = sc(xv[:, s : s + 1], state=st_dec).clamp(-1000.0, 1000.0)
            dec_steps.append(y_s)
            rb_d, _ = _diff_stats(stage_v[:, s], st_dec.conv_state)
            assert rb_d == 0.0, f"Verify rollback step={s} diff={rb_d}"
        out_dec = torch.cat(dec_steps, dim=1)
        vd, _ = _diff_stats(out_v, out_dec)
        assert vd < 1e-2, f"Verify out vs 8x decode diff={vd}"

        # CUDA Graph capture/replay for verify kernel
        static_x = xv.clone()
        static_init = st_v_init.clone()
        static_stage = torch.empty_like(stage_v)
        # warmup
        causal_dwconv1d_verify_triton(
            static_x, sc.weight, static_init, static_stage, residual=residual, use_activation=use_act, clamp_val=1000.0
        )
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = causal_dwconv1d_verify_triton(
                static_x, sc.weight, static_init, static_stage, residual=residual, use_activation=use_act, clamp_val=1000.0
            )
        static_x.normal_(0.0, 0.5)
        static_init.normal_(0.0, 0.5)
        g.replay()
        torch.cuda.synchronize()
        eager_chk = causal_dwconv1d_verify_triton(
            static_x, sc.weight, static_init, torch.empty_like(stage_v), residual=residual, use_activation=use_act, clamp_val=1000.0
        )
        cg_abs, _ = _diff_stats(static_out, eager_chk)
        assert cg_abs == 0.0, f"ShortConv verify CUDA Graph mismatch: {cg_abs}"
        print(f"  [PASS] ShortConv res={residual} act={use_act}: packed_snap=0.0, ragged=0.0, verify_v8+rb=0.0, CG={cg_abs:.1e}")


@torch.no_grad()
def test_ple_packed_prefill_verify_and_tp_hooks() -> None:
    print("\n=== [3c] PLE Packed Prefill, Verify (Rollback + CG), and TP=4 Hook Parity ===")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(2027)

    H, HC, PLE_DIM, HEADS_PER_NGRAM = 128, 4, 256, 4
    num_shards = 8
    primes = compute_ngram_vocab_sizes(256, 2 * HEADS_PER_NGRAM)
    total_vocab = sum(primes)
    shard_rows = (total_vocab + num_shards - 1) // num_shards
    ple = Qwen4_5PLELayer(
        hidden_size=H,
        hc_count=HC,
        ple_embed_dim=PLE_DIM,
        ngram_size=3,
        heads_per_ngram=HEADS_PER_NGRAM,
        conv_kernel_size=4,
        split_ngram_parts=num_shards,
        eos_token_id=248044,
        ngram_heads_vocab_sizes=primes,
        weight_scale=DEFAULT_PLE_WEIGHT_SCALE,
        dtype=dtype,
        device=device,
    ).eval()
    ple.norm_key.normal_(0.0, 0.1)
    ple.norm_query.normal_(0.0, 0.1)
    ple.norm_conv.normal_(0.0, 0.1)
    ple.conv1d.weight.normal_(0.0, 0.15)
    shards = [
        (torch.randn((shard_rows, 32), device=device, dtype=torch.float32) * 0.5).to(torch.float8_e4m3fn)
        for _ in range(num_shards)
    ]
    ple.set_ngram_table(shards)

    # 1. forward_packed_prefill (B=3 ragged, with boundary snapshots at token 64/128)
    lens = [70, 64, 130]
    cu_list = [0]
    for l in lens:
        cu_list.append(cu_list[-1] + l)
    cu = torch.tensor(cu_list, device=device, dtype=torch.int32)
    tok_req, t_req, t_t0, s_tok, s_req, s_dst = [], [], [], [], [], []
    snap_meta = []
    blk_id = 1
    for bi, l in enumerate(lens):
        tok_req.extend([bi] * l)
        for t0 in range(0, l, 16):
            t_req.append(bi)
            t_t0.append(t0)
        for bnd in range(64, l + 1, 64):
            s_tok.append(cu_list[bi] + bnd - 1)
            s_req.append(bi)
            s_dst.append(blk_id)
            snap_meta.append((bi, bnd, blk_id))
            blk_id += 1
    N = cu_list[-1]
    h_rag = torch.randn((N, HC * H), device=device, dtype=dtype)
    ids_rag = torch.randint(0, 1000, (N,), device=device, dtype=torch.int64)
    init_conv = torch.randn((len(lens), HC * H, 3), device=device, dtype=dtype)
    prev_ctx = torch.randint(0, 1000, (len(lens), 2), device=device, dtype=torch.int64)
    conv_pool = torch.zeros((blk_id + 1, HC * H * 3), device=device, dtype=dtype)
    ctx_pool = torch.zeros((blk_id + 1, 2), device=device, dtype=torch.int64)
    snap_phys_t = torch.tensor(s_dst, device=device, dtype=torch.int32)

    out_rag = ple.forward_packed_prefill(
        h_rag,
        ids_rag,
        cu,
        torch.tensor(tok_req, device=device, dtype=torch.int32),
        torch.tensor(t_req, device=device, dtype=torch.int32),
        torch.tensor(t_t0, device=device, dtype=torch.int32),
        prev_ngram_ctx=prev_ctx,
        init_conv_state=init_conv,
        snap_tok_idx=torch.tensor(s_tok, device=device, dtype=torch.int32),
        snap_req_idx=torch.tensor(s_req, device=device, dtype=torch.int32),
        conv_dst_pool=conv_pool,
        conv_snap_phys=snap_phys_t,
        ctx_dst_pool=ctx_pool,
        ctx_snap_phys=snap_phys_t,
        block_t=16,
    )
    for bi, l in enumerate(lens):
        st_ref = PLELayerState(conv_state=init_conv[bi : bi + 1].clone(), ngram_context=prev_ctx[bi : bi + 1].clone())
        y_ref = ple(
            h_rag[cu_list[bi] : cu_list[bi + 1]].unsqueeze(0),
            ids_rag[cu_list[bi] : cu_list[bi + 1]].unsqueeze(0),
            state=st_ref,
        )
        pd, _ = _diff_stats(out_rag[cu_list[bi] : cu_list[bi + 1]], y_ref.squeeze(0))
        assert pd < 1e-2, f"PLE packed_prefill req={bi} diff={pd}"
    for bi, bnd, phys in snap_meta:
        st_bnd = PLELayerState(conv_state=init_conv[bi : bi + 1].clone(), ngram_context=prev_ctx[bi : bi + 1].clone())
        _ = ple(
            h_rag[cu_list[bi] : cu_list[bi] + bnd].unsqueeze(0),
            ids_rag[cu_list[bi] : cu_list[bi] + bnd].unsqueeze(0),
            state=st_bnd,
        )
        assert torch.equal(ctx_pool[phys], st_bnd.ngram_context[0]), "PLE ctx snapshot mismatch"
        cd, _ = _diff_stats(conv_pool[phys].view(HC * H, 3), st_bnd.conv_state[0])
        assert cd < 1e-3, f"PLE conv snapshot req={bi} bnd={bnd} diff={cd}"

    # 2. forward_verify (B=3, V=8) + rollback states + CUDA Graph capture/replay
    Bv, V = 3, 8
    hv = torch.randn((Bv, V, HC * H), device=device, dtype=dtype)
    idsv = torch.randint(0, 1000, (Bv, V), device=device, dtype=torch.int64)
    init_ctx_v = torch.randint(0, 1000, (Bv, 2), device=device, dtype=torch.int64)
    init_conv_v = torch.randn((Bv, HC * H, 3), device=device, dtype=dtype)
    stage_ctx = torch.empty((Bv, V, 2), device=device, dtype=torch.int64)
    stage_conv = torch.empty((Bv, V, HC * H, 3), device=device, dtype=dtype)
    out_v = ple.forward_verify(hv, idsv, init_ctx_v, init_conv_v, stage_ctx=stage_ctx, stage_conv=stage_conv)

    st_step = PLELayerState(conv_state=init_conv_v.clone(), ngram_context=init_ctx_v.clone())
    step_outs = []
    for s in range(V):
        ys = ple(hv[:, s : s + 1], idsv[:, s : s + 1], state=st_step)
        step_outs.append(ys)
        assert torch.equal(stage_ctx[:, s], st_step.ngram_context), f"PLE verify ctx step={s}"
        sc_d, _ = _diff_stats(stage_conv[:, s], st_step.conv_state)
        assert sc_d < 1e-3, f"PLE verify conv step={s} diff={sc_d}"
    vd, _ = _diff_stats(out_v, torch.cat(step_outs, dim=1))
    assert vd < 1e-2, f"PLE forward_verify diff={vd}"

    # CUDA Graph replay parity for forward_verify
    sh, sids = hv.clone(), idsv.clone()
    sictx, siconv = init_ctx_v.clone(), init_conv_v.clone()
    sst_ctx, sst_conv = torch.empty_like(stage_ctx), torch.empty_like(stage_conv)
    ple.forward_verify(sh, sids, sictx, siconv, stage_ctx=sst_ctx, stage_conv=sst_conv)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        s_out = ple.forward_verify(sh, sids, sictx, siconv, stage_ctx=sst_ctx, stage_conv=sst_conv)
    sh.normal_(0.0, 0.5)
    sids.random_(0, 1000)
    g.replay()
    torch.cuda.synchronize()
    eager_v = ple.forward_verify(sh, sids, sictx, siconv)
    cg_d, _ = _diff_stats(s_out, eager_v)
    assert cg_d < 1e-4, f"PLE verify CG mismatch={cg_d}"

    # 3. TP=4 sharded fast_all_reduce hook parity
    embs_tp = []
    for r in range(4):
        cap = []
        ple_r = Qwen4_5PLELayer(
            hidden_size=H, hc_count=HC, ple_embed_dim=PLE_DIM, ngram_size=3,
            heads_per_ngram=HEADS_PER_NGRAM, conv_kernel_size=4, split_ngram_parts=num_shards,
            ngram_heads_vocab_sizes=primes, weight_scale=DEFAULT_PLE_WEIGHT_SCALE,
            dtype=dtype, device=device, tp_size=4, tp_rank=r,
        ).eval()
        ple_r.set_ngram_table(
            shards[2 * r : 2 * r + 2],
            shard_offset=2 * r,
            total_shards=num_shards,
            fast_all_reduce=lambda t, _cap=cap: (_cap.append(t.clone()) or t),
        )
        ple_r.load_state_dict(ple.state_dict())
        _ = ple_r(hv[:, :1], idsv[:, :1], state=ple_r.alloc_state(Bv, device, dtype))
        embs_tp.append(cap[0])
    emb_ref = ple.gather_embeddings(ple.compute_ngram_ids(idsv[:, :1]))
    tp_d, _ = _diff_stats(sum(embs_tp), emb_ref)
    assert tp_d < 1e-2, f"PLE TP=4 sharded embedding sum mismatch={tp_d}"

    # 4. R0 Fusions: single-stream emb_flat HC, ple_add3_and_hc_norm_triton, forward_packed_prefill_with_hc, and SP/DP hooks
    hc_head = Qwen4_5HyperConnection(
        hidden_size=H, hc_count=HC, hc_lowrank=32, is_head=True, dtype=dtype, device=device,
        sp_size=4, sp_rank=1, dp_attn=True,
    ).eval()
    hc_next = Qwen4_5HyperConnection(
        hidden_size=H, hc_count=HC, hc_lowrank=32, is_head=False, dtype=dtype, device=device,
    ).eval()
    assert hc_head.sp_size == 4 and hc_head.sp_rank == 1 and hc_head.dp_attn is True
    emb_single = torch.randn((N, H), device=device, dtype=dtype)
    m_s, r_s, inj_s, lat_s = hc_head.mix_with_latent(emb_single)
    m_f, r_f, inj_f, lat_f = hc_head.mix_with_latent(emb_single.repeat(1, HC))
    assert r_s.shape == (N, H)
    assert _diff_stats(m_s, m_f)[0] == 0.0 and _diff_stats(inj_s, inj_f)[0] == 0.0 and _diff_stats(lat_s, lat_f)[0] == 0.0
    sub_b0 = torch.randn((N, H), device=device, dtype=dtype)
    comb_s = hc_next.combine(sub_b0, r_s, inj_s)
    comb_f = hc_next.combine(sub_b0, r_f, inj_f)
    assert comb_s.shape == (N, HC * H) and _diff_stats(comb_s, comb_f)[0] == 0.0
    cm_s, c_s, cinj_s, clat_s = hc_next.combine_and_mix_with_latent(sub_b0, r_s, inj_s, lat_s)
    cm_f, c_f, cinj_f, clat_f = hc_next.combine_and_mix_with_latent(sub_b0, r_f, inj_f, lat_f)
    assert _diff_stats(c_s, c_f)[0] == 0.0 and _diff_stats(cm_s, cm_f)[0] == 0.0
    assert _diff_stats(cinj_s, cinj_f)[0] == 0.0 and _diff_stats(clat_s, clat_f)[0] == 0.0

    # Standalone ple_add3_and_hc_norm_triton + pre-combine ple_gate_and_norm_triton + forward_packed_prefill_with_hc
    u_t = torch.randn((1, N, HC * H), device=device, dtype=dtype)
    c_t = torch.randn((1, N, HC * H), device=device, dtype=dtype)
    h_t = torch.randn((1, N, HC * H), device=device, dtype=dtype)
    sub_t = torch.randn((1, N, H), device=device, dtype=dtype)
    inj_t = torch.rand((1, N, HC), device=device, dtype=dtype)
    ah_ref = ple_add3_triton(h_t, u_t, c_t)
    an_ref = grouped_gemma_rmsnorm_bias_triton(ah_ref, hc_next.hc_norm_weight, hc_next.hc_norm_bias, HC, hc_next.eps)
    ah_fus, an_fus = ple_add3_and_hc_norm_triton(h_t, u_t, c_t, hc_next.hc_norm_weight, hc_next.hc_norm_bias, HC, hc_next.eps)
    assert _diff_stats(ah_fus, ah_ref)[0] == 0.0 and _diff_stats(an_fus, an_ref)[0] == 0.0
    ah_pc_ref = hc_combine_triton(sub_t.squeeze(0), ah_ref.squeeze(0), inj_t.squeeze(0)).unsqueeze(0)
    an_pc_ref = grouped_gemma_rmsnorm_bias_triton(ah_pc_ref, hc_next.hc_norm_weight, hc_next.hc_norm_bias, HC, hc_next.eps)
    ah_pc, an_pc = ple_add3_and_hc_norm_triton(
        h_t, u_t, c_t, hc_next.hc_norm_weight, hc_next.hc_norm_bias, HC, hc_next.eps,
        combine_sub=sub_t, combine_inj=inj_t,
    )
    assert _diff_stats(ah_pc, ah_pc_ref)[0] == 0.0 and _diff_stats(an_pc, an_pc_ref)[0] == 0.0

    tok_req_t = torch.tensor(tok_req, device=device, dtype=torch.int32)
    t_req_t = torch.tensor(t_req, device=device, dtype=torch.int32)
    t_t0_t = torch.tensor(t_t0, device=device, dtype=torch.int32)
    h_pre_comb_ref = hc_next.combine(sub_b0, h_rag, inj_s)
    ple_unfus = ple.forward_packed_prefill(
        h_pre_comb_ref, ids_rag, cu, tok_req_t, t_req_t, t_t0_t, prev_ngram_ctx=prev_ctx, init_conv_state=init_conv, block_t=16
    )
    mx_unfus, res_unfus, mi_unfus, ml_unfus = hc_next.mix_with_latent(ple_unfus, lat_s)
    ple_fus, normed_fus = ple.forward_packed_prefill_with_hc(
        h_rag, ids_rag, cu, tok_req_t, t_req_t, t_t0_t, prev_ngram_ctx=prev_ctx, init_conv_state=init_conv, block_t=16,
        pre_combine_sub=sub_b0, pre_combine_inj=inj_s,
        hc_norm_weight=hc_next.hc_norm_weight, hc_norm_bias=hc_next.hc_norm_bias, hc_eps=hc_next.eps,
    )
    mx_fus, res_fus, mi_fus, ml_fus = hc_next.mix_from_normed(normed_fus, ple_fus, lat_s)
    assert _diff_stats(ple_fus, ple_unfus)[0] == 0.0
    assert _diff_stats(mx_fus, mx_unfus)[0] == 0.0
    assert _diff_stats(mi_fus, mi_unfus)[0] == 0.0 and _diff_stats(ml_fus, ml_unfus)[0] == 0.0

    # ShortConv & PLE SP/DP constructor + exchange_sp_left_halo sanity
    ple_sp = Qwen4_5PLELayer(
        hidden_size=H, hc_count=HC, ple_embed_dim=PLE_DIM, ngram_size=3,
        heads_per_ngram=HEADS_PER_NGRAM, conv_kernel_size=4, split_ngram_parts=num_shards,
        ngram_heads_vocab_sizes=primes, weight_scale=DEFAULT_PLE_WEIGHT_SCALE,
        dtype=dtype, device=device, sp_size=4, sp_rank=2, dp_attn=True,
    )
    assert ple_sp.sp_size == 4 and ple_sp.sp_rank == 2 and ple_sp.dp_attn is True
    assert ple_sp.conv1d.sp_size == 4 and ple_sp.conv1d.sp_rank == 2 and ple_sp.conv1d.dp_attn is True
    halo = ple_sp.conv1d.exchange_sp_left_halo(h_rag[:, :H])
    assert halo.shape == (1, H, 3)

    print(f"  [PASS] PLE packed_prefill(3 reqs), verify(V=8)+rollback+CG(diff={cg_d:.1e}), TP=4 & R0 PLE+HC fusions(diff=0.0)")


# ---------------------------------------------------------------------------
# 4. B300 GPU Micro-Benchmarks at REAL Qwen 4.5-Flash Shapes
# ---------------------------------------------------------------------------


@torch.no_grad()

def test_r2_hc_aligned_projections_and_ple_p2p() -> None:
    from unittest.mock import patch
    from rtp_llm.models_py.modules.qwen4_5.hyper_connection import _aligned_hc_out_dim

    print("\n=== [3d] R2 HC Zero-cutlass_80 Aligned Projections & PLE P2P Check ===")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    assert _aligned_hc_out_dim(36, hidden_size=5120) == 64
    assert _aligned_hc_out_dim(4, hidden_size=5120) == 16
    assert _aligned_hc_out_dim(36, hidden_size=512) == 36

    torch.manual_seed(202620)
    for use_prev in (True, False):
        hc = Qwen4_5HyperConnection(
            hidden_size=5120,
            hc_count=4,
            hc_lowrank=32,
            prev_mix=use_prev,
            use_combine=True,
            emits_latent=True,
            dtype=dtype,
            device=device,
        ).eval()
        for m in (1, 8, 64, 512):
            x = torch.randn((1, m, 4 * 5120), device=device, dtype=dtype) * 0.5
            prev = torch.randn((1, m, 32), device=device, dtype=dtype) * 0.5 if use_prev else None
            mixed_tri, _, inj_tri, lat_tri = hc.mix_with_latent(x, prev_latent=prev)
            mixed_ref, _, inj_ref, lat_ref = hc.mix_with_latent_ref(x, prev_latent=prev)
            m_abs, r_m = _diff_stats(mixed_tri, mixed_ref)
            i_abs, r_i = _diff_stats(inj_tri, inj_ref)
            l_abs, r_l = _diff_stats(lat_tri, lat_ref)
            assert m_abs < 1.5e-2 and i_abs < 1.5e-2 and l_abs < 1.5e-2, (
                f"use_prev={use_prev} m={m} abs=({m_abs}, {i_abs}, {l_abs}) rel=({r_m}, {r_i}, {r_l})"
            )

    # Verify zero cutlass_80 kernel launch on H=5120, lowrank=32 decode & prefill
    hc_prof = Qwen4_5HyperConnection(
        hidden_size=5120, hc_count=4, hc_lowrank=32, prev_mix=False, use_combine=True, emits_latent=True, dtype=dtype, device=device
    ).eval()
    for m in (64, 512):
        xp = torch.randn((1, m, 4 * 5120), device=device, dtype=dtype) * 0.5
        _ = hc_prof.mix_with_latent(xp)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            _ = hc_prof.mix_with_latent(xp)
        torch.cuda.synchronize()
        bad = [e.key for e in prof.key_averages() if "cutlass_80" in e.key]
        assert not bad, f"Unexpected cutlass_80 kernel at H=5120 LR=32 m={m}: {bad}"

    # Verify CUDA Graph capture & replay on H=5120, LR=32
    x_cg = torch.randn((1, 64, 4 * 5120), device=device, dtype=dtype) * 0.5
    p_cg = torch.randn((1, 64, 32), device=device, dtype=dtype) * 0.5
    hc_cg = Qwen4_5HyperConnection(
        hidden_size=5120, hc_count=4, hc_lowrank=32, prev_mix=True, use_combine=True, emits_latent=True, dtype=dtype, device=device
    ).eval()
    _ = hc_cg.mix_with_latent(x_cg, prev_latent=p_cg)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        om, _, oi, ol = hc_cg.mix_with_latent(x_cg, prev_latent=p_cg)
    x_cg.copy_(torch.randn_like(x_cg) * 0.5)
    p_cg.copy_(torch.randn_like(p_cg) * 0.5)
    g.replay()
    rm, _, ri, rl = hc_cg.mix_with_latent(x_cg, prev_latent=p_cg)
    assert float((om.float() - rm.float()).abs().max().item()) == 0.0
    assert float((oi.float() - ri.float()).abs().max().item()) == 0.0
    assert float((ol.float() - rl.float()).abs().max().item()) == 0.0

    # Verify PLE peer-access check accepts P2P CUDA shards and builds pointer table
    primes = compute_ngram_vocab_sizes(256, 20)
    ple = Qwen4_5PLELayer(
        hidden_size=64, hc_count=4, ple_embed_dim=160, ngram_size=3, heads_per_ngram=10,
        conv_kernel_size=4, split_ngram_parts=4, eos_token_id=151645, ngram_heads_vocab_sizes=primes,
        weight_scale=DEFAULT_PLE_WEIGHT_SCALE, dtype=dtype, device=device,
    ).eval()
    s_rows = (sum(primes) + 3) // 4
    shards = [torch.zeros((s_rows, 8), dtype=torch.int8, device=device).view(torch.float8_e4m3fn) for _ in range(4)]
    with patch("torch.cuda.can_device_access_peer", return_value=True):
        ple.set_ngram_table(shards)
        assert ple._ensure_pointer_table(device).numel() == 4
    print("  [PASS] R2 HC H=5120 LR=32 zero cutlass_80, CG parity=0.0, & PLE P2P shard check")

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
    sc_v = Qwen4_5ShortConv(channels=5120, kernel_size=4, residual=True, use_activation=False, dtype=dtype, device=device).eval()
    xv_b = torch.randn(4, 8, 5120, device=device, dtype=dtype)
    stv_b = sc_v.alloc_state(4, device=device, dtype=dtype)
    stg_b = torch.empty(4, 8, 5120, 3, device=device, dtype=dtype)
    t_dec8 = _bench_us(lambda: [sc_v(xv_b[:, i : i + 1], state=stv_b) for i in range(8)])
    t_ver8 = _bench_us(lambda: causal_dwconv1d_verify_triton(xv_b, sc_v.weight, stv_b.conv_state, stg_b, residual=True))
    print(f"    Out_sconv B=4,V=8: 8xDecode={t_dec8:7.2f} us vs Verify={t_ver8:7.2f} us (speedup={t_dec8/max(t_ver8,1e-6):.2f}x)")

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
    test_short_conv_verify_and_packed_prefill()
    test_ple_packed_prefill_verify_and_tp_hooks()
    test_r2_hc_aligned_projections_and_ple_p2p()
    run_benchmarks()
    print(f"\nALL TESTS & BENCHMARKS PASSED in {time.perf_counter() - t0:.2f}s!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

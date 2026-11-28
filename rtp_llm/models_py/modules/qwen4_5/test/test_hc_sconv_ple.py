"""Unit tests & B300 GPU micro-benchmarks for Qwen 4.5-Flash HC, ShortConv, and PLE."""

from __future__ import annotations

import sys
import time
from typing import Callable, List, Tuple

import torch

from rtp_llm.models_py.modules.qwen4_5.contracts import PLELayerState, ShortConvState
from rtp_llm.models_py.modules.qwen4_5.hyper_connection import (
    HCStaticWorkspace,
    Qwen4_5HyperConnection,
    grouped_gemma_rmsnorm_bias_ref,
    grouped_gemma_rmsnorm_bias_triton,
    hc_combine_ref,
    hc_combine_and_norm_triton,
    hc_combine_triton,
    hc_dflash_gconv_combine_norm_triton,
    hc_down_inject_epilogue_triton,
    hc_down_silu_epilogue_triton,
    hc_inject_epilogue_triton,
    hc_mix_reduce_triton,
    hc_mix_reduce_with_down_inject_triton,
    hc_sconv_decode_opaque_combine_norm_triton,
)
from rtp_llm.models_py.modules.qwen4_5.ple import (
    DEFAULT_PLE_WEIGHT_SCALE,
    Qwen4_5PLELayer,
    compute_ngram_vocab_sizes,
    ple_add3_and_hc_norm_triton,
    ple_add3_triton,
    ple_gather_fp8_triton,
    ple_gate_and_norm_triton,
)
from rtp_llm.models_py.modules.qwen4_5.short_conv import (
    Qwen4_5ShortConv,
    ShortConvStaticWorkspace,
    causal_dwconv1d_prefill_with_snapshots_triton,
    causal_dwconv1d_ragged_prefill_with_snapshots_triton,
    causal_dwconv1d_verify_opaque_triton,
    _SC_TRITON_CACHE,
    _has_active_hooks,
    causal_dwconv1d_decode_opaque_triton,
    causal_dwconv1d_decode_triton,
    causal_dwconv1d_ref,
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

def test_r4_shortconv_opaque_verify_and_clip_value() -> None:
    print("\n=== [3e] R4 ShortConv Direct-Opaque Verify & Fused clip_value Check ===")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(202640)

    B, V, ps, num_blocks = 8, 8, 64, 48
    for C, pool_slots, offset_idx in [(5120, 1, 0), (256, 2, 1)]:
        pool_3d = torch.randn((num_blocks, pool_slots, C, 3), device=device, dtype=dtype) * 0.5
        pool_2d = pool_3d.reshape(num_blocks, pool_slots * C * 3)
        bt = torch.tensor(
            [[1 + b * 2, 2 + b * 2, 0] for b in range(B)],
            device=device,
            dtype=torch.int32,
        )
        # Exercise seq=0 (no prefix), seq=64 (page 0), seq=128 (page 1), and phys=0 unallocated sentinel (page 2)
        seq_before = torch.tensor([0, 64, 65, 128, 130, 1, 64, 150], device=device, dtype=torch.int32)
        w = torch.randn((C, 1, 4), device=device, dtype=dtype) * 0.2
        x = torch.randn((B, V, C), device=device, dtype=dtype) * 2.0

        init_ref = torch.zeros((B, C, 3), device=device, dtype=dtype)
        for b in range(B):
            sb = int(seq_before[b].item())
            if sb > 0:
                phys = int(bt[b, (sb - 1) // ps].item())
                if 0 < phys < num_blocks:
                    init_ref[b].copy_(pool_3d[phys, offset_idx])

        st_ref = torch.empty((B, V, C, 3), device=device, dtype=dtype)
        st_opq = torch.empty((B, V, C, 3), device=device, dtype=dtype)
        y_ref = causal_dwconv1d_verify_triton(
            x, w, init_ref, st_ref, residual=True, use_activation=(C == 256), round_pre_act=(C == 256), clamp_val=1.5
        )
        y_opq = causal_dwconv1d_verify_opaque_triton(
            x,
            w,
            pool_2d,
            bt,
            seq_before,
            st_opq,
            page_size=ps,
            pool_col_offset=offset_idx * C * 3,
            residual=True,
            use_activation=(C == 256),
            round_pre_act=(C == 256),
            clamp_val=1.5,
        )
        assert float((y_opq - y_ref).abs().max().item()) == 0.0
        assert float((st_opq - st_ref).abs().max().item()) == 0.0

    # Verify CUDA Graph capture/replay on C=5120 direct-opaque verify
    C_cg = 5120
    pool_cg = torch.randn((num_blocks, C_cg * 3), device=device, dtype=dtype) * 0.5
    bt_cg = torch.arange(1, 1 + B * 2, device=device, dtype=torch.int32).view(B, 2)
    sb_cg = torch.full((B,), 64, device=device, dtype=torch.int32)
    x_cg = torch.randn((B, V, C_cg), device=device, dtype=dtype)
    w_cg = torch.randn((C_cg, 1, 4), device=device, dtype=dtype) * 0.2
    st_cg = torch.empty((B, V, C_cg, 3), device=device, dtype=dtype)
    _ = causal_dwconv1d_verify_opaque_triton(x_cg, w_cg, pool_cg, bt_cg, sb_cg, st_cg, page_size=ps, residual=True, clamp_val=2.0)
    cg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cg):
        out_cg = causal_dwconv1d_verify_opaque_triton(x_cg, w_cg, pool_cg, bt_cg, sb_cg, st_cg, page_size=ps, residual=True, clamp_val=2.0)
    x_cg.copy_(torch.randn_like(x_cg) * 4.0)
    cg.replay()
    st_eag = torch.empty_like(st_cg)
    out_eag = causal_dwconv1d_verify_opaque_triton(x_cg, w_cg, pool_cg, bt_cg, sb_cg, st_eag, page_size=ps, residual=True, clamp_val=2.0)
    assert float((out_cg - out_eag).abs().max().item()) == 0.0
    assert float((st_cg - st_eag).abs().max().item()) == 0.0

    # Verify clip_value on Qwen4_5ShortConv.forward (Prefill & Decode)
    sconv = Qwen4_5ShortConv(512, kernel_size=4, residual=True, use_activation=False, dtype=dtype, device=device).eval()
    xp = torch.randn((2, 17, 512), device=device, dtype=dtype) * 3.0
    stp_a = sconv.alloc_state(2, device=device, dtype=dtype)
    stp_b = sconv.alloc_state(2, device=device, dtype=dtype)
    yp_clip = sconv(xp, state=stp_a, clip_value=2.5)
    yp_ref = sconv.forward_ref(xp, state=stp_b, clip_value=2.5)
    assert float((yp_clip.float() - yp_ref.float()).abs().max().item()) < 2e-2
    xd = torch.randn((2, 1, 512), device=device, dtype=dtype) * 3.0
    yd_clip = sconv(xd, state=stp_a, clip_value=2.5)
    yd_ref = sconv.forward_ref(xd, state=stp_b, clip_value=2.5)
    assert float((yd_clip.float() - yd_ref.float()).abs().max().item()) < 2e-2
    assert float(yd_clip.abs().max().item()) <= 2.5
    print("  [PASS] R4 ShortConv direct-opaque verify + clip_value parity & CG!")


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



def test_r5_ple_u64_hc_split5120_and_strided_shortconv() -> None:
    print("\n=== [3f] R5 PLE u64 Head-Grouped + HC Split5120 + ShortConv Strided Decode ===")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(202650)

    # 1. PLE u64 aligned gather (1D shard u64 + 2D/3D head-grouped u64, GPU + CPU-UVA) & CUDA Graph
    B_p, T_p, H_ng, D_ple = 16, 16, 4, 256
    n_shards, shard_rows = 8, 256
    gpu_shards = [
        (torch.randn((shard_rows, D_ple), device=device) * 0.5).to(torch.float8_e4m3fn)
        for _ in range(n_shards)
    ]
    ptr_gpu = torch.tensor([int(s.data_ptr()) for s in gpu_shards], device=device, dtype=torch.int64)
    w_sc = torch.tensor(float(DEFAULT_PLE_WEIGHT_SCALE), device=device, dtype=torch.float32)
    ids_hg = torch.randint(0, n_shards * shard_rows, (B_p, T_p, H_ng), device=device, dtype=torch.int64)
    out_u64 = ple_gather_fp8_triton(ids_hg, ptr_gpu, shard_rows, D_ple, w_sc, allow_u64_vec=True)
    out_ref = ple_gather_fp8_triton(ids_hg, ptr_gpu, shard_rows, D_ple, w_sc, allow_u64_vec=False)
    assert float((out_u64 - out_ref).abs().max().item()) == 0.0
    ids_1d = ids_hg.reshape(-1)
    assert float((
        ple_gather_fp8_triton(ids_1d, ptr_gpu, shard_rows, D_ple, w_sc, allow_u64_vec=True)
        - ple_gather_fp8_triton(ids_1d, ptr_gpu, shard_rows, D_ple, w_sc, allow_u64_vec=False)
    ).abs().max().item()) == 0.0

    ple = Qwen4_5PLELayer(
        hidden_size=256,
        hc_count=4,
        ple_embed_dim=H_ng * D_ple,
        ngram_size=3,
        heads_per_ngram=2,
        split_ngram_parts=n_shards,
        ngram_heads_vocab_sizes=[512] * H_ng,
        dtype=dtype,
        device=device,
    ).eval()
    cpu_shards = [
        (torch.randn((shard_rows, D_ple)) * 0.5).to(torch.float8_e4m3fn).pin_memory()
        for _ in range(n_shards)
    ]
    ple.set_ngram_table(cpu_shards, pin_cpu=False, enable_cpu_uva=True)
    assert ple._pointer_u64_aligned
    h_ple = torch.randn((B_p, T_p, 4 * 256), device=device, dtype=dtype)
    toks_ple = torch.randint(0, 2048, (B_p, T_p), device=device, dtype=torch.int64)
    st_uva = ple.alloc_state(B_p, device=device, dtype=dtype)
    y_uva = ple(h_ple, toks_ple, state=st_uva)
    ple._pointer_u64_aligned = False
    y_uva_ref = ple(h_ple, toks_ple, state=ple.alloc_state(B_p, device=device, dtype=dtype))
    ple._pointer_u64_aligned = True
    assert float((y_uva - y_uva_ref).abs().max().item()) == 0.0
    cg_ple = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cg_ple):
        y_uva_cg = ple(h_ple, toks_ple)
    h_ple.copy_(torch.randn_like(h_ple))
    toks_ple.copy_(torch.randint(0, 2048, (B_p, T_p), device=device, dtype=torch.int64))
    cg_ple.replay()
    assert float((y_uva_cg - ple(h_ple, toks_ple)).abs().max().item()) == 0.0

    # 2. HC _hc_combine_and_norm_split5120_shared_sub_kernel (rows >= 1024, H=5120, HC=4) + CG
    H_hc, HC_n = 5120, 4
    for rows, single_stream, use_bias in [(1024, False, True), (1280, True, False), (1280, False, True)]:
        res = (
            torch.randn((rows, H_hc), device=device, dtype=dtype) * 0.4
            if single_stream
            else torch.randn((rows, HC_n * H_hc), device=device, dtype=dtype) * 0.4
        )
        sub = torch.randn((rows, H_hc), device=device, dtype=dtype) * 0.4
        inj = torch.randn((rows, HC_n), device=device, dtype=dtype) * 0.25
        nw = torch.randn((HC_n * H_hc,), device=device, dtype=dtype) * 0.2
        nb = (torch.randn((HC_n * H_hc,), device=device, dtype=dtype) * 0.1) if use_bias else None
        c_tri, n_tri = hc_combine_and_norm_triton(sub, res, inj, nw, nb, eps=1e-6)
        c_ref = hc_combine_ref(sub, res, inj)
        n_ref = grouped_gemma_rmsnorm_bias_ref(c_ref, nw, nb, HC_n, eps=1e-6)
        c_max, _ = _diff_stats(c_tri, c_ref)
        n_max, _ = _diff_stats(n_tri, n_ref)
        cos_n = torch.nn.functional.cosine_similarity(n_tri.float().flatten(), n_ref.float().flatten(), dim=0).item()
        assert c_max < 2e-2 and n_max < 0.08 and cos_n > 0.99999, (
            f"HC split5120 rows={rows} ss={single_stream} bias={use_bias}: c_max={c_max} n_max={n_max} cos={cos_n}"
        )
    cg_hc = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cg_hc):
        c_cg, n_cg = hc_combine_and_norm_triton(sub, res, inj, nw, nb, eps=1e-6)
    sub.copy_(torch.randn_like(sub) * 0.4)
    cg_hc.replay()
    c_eag, n_eag = hc_combine_and_norm_triton(sub, res, inj, nw, nb, eps=1e-6)
    assert float((c_cg - c_eag).abs().max().item()) == 0.0 and float((n_cg - n_eag).abs().max().item()) == 0.0

    # 3. ShortConv strided decode (slice [B, 1, 256] from wide [B, 1, 19072], zero .contiguous())
    B_sc, C_sc = 16, 256
    wide_buf = torch.randn((B_sc, 1, 19072), device=device, dtype=dtype)
    x_strided = wide_buf[:, :, 16384 : 16384 + C_sc]
    assert x_strided.stride() == (19072, 19072, 1) and not x_strided.is_contiguous()
    w_scv = torch.randn((C_sc, 1, 4), device=device, dtype=dtype) * 0.2
    st_s = torch.randn((B_sc, C_sc, 3), device=device, dtype=dtype) * 0.2
    st_c = st_s.clone()
    y_s = causal_dwconv1d_decode_triton(x_strided, w_scv, st_s, residual=True, use_activation=False)
    y_c = causal_dwconv1d_decode_triton(x_strided.contiguous(), w_scv, st_c, residual=True, use_activation=False)
    assert float((y_s - y_c).abs().max().item()) == 0.0 and float((st_s - st_c).abs().max().item()) == 0.0
    cg_sc = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cg_sc):
        y_scg = causal_dwconv1d_decode_triton(x_strided, w_scv, st_s, residual=True, use_activation=False)
    wide_buf[:, :, 16384 : 16384 + C_sc].copy_(torch.randn_like(x_strided))
    st_ref_cg = st_s.clone()
    cg_sc.replay()
    y_se = causal_dwconv1d_decode_triton(x_strided.contiguous(), w_scv, st_ref_cg, residual=True, use_activation=False)
    assert float((y_scg - y_se).abs().max().item()) == 0.0
    print("  PASS: PLE u64, HC Split5120, ShortConv strided decode & CUDA Graph parity verified.")


@torch.no_grad()
def test_r6_fast_launch_opaque_decode_and_ple_snap_aligned() -> None:
    print("\n=== [3f] R6 HookChain fast_launch, Direct-Opaque Decode ShortConv+HC & PLE snap_aligned Check ===")
    import triton

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(202660)

    # 1. HookChain empty-calls detection & triton_fast_launch cache-hit parity
    knobs = triton.knobs.runtime
    assert not _has_active_hooks(knobs.launch_enter_hook), (
        f"Unexpected active launch_enter_hook: {getattr(knobs.launch_enter_hook, 'calls', None)}"
    )
    assert not _has_active_hooks(knobs.launch_exit_hook)

    B, ps, num_blocks = 8, 64, 48
    for C, pool_slots, off_idx in [(256, 2, 1), (512, 1, 0)]:
        pool_3d = torch.randn((num_blocks, pool_slots, C, 3), device=device, dtype=dtype) * 0.4
        pool_ref = pool_3d.clone()
        pool_2d = pool_3d.reshape(num_blocks, pool_slots * C * 3)
        bt = torch.tensor([[1 + b * 2, 2 + b * 2, 0] for b in range(B)], device=device, dtype=torch.int32)
        seq_before = torch.tensor([0, 1, 63, 64, 65, 128, 130, 192], device=device, dtype=torch.int32)
        w_sc = torch.randn((C, 1, 4), device=device, dtype=dtype) * 0.25
        x = torch.randn((B, 1, C), device=device, dtype=dtype) * 1.8

        init_ref = torch.zeros((B, C, 3), device=device, dtype=dtype)
        for b in range(B):
            sb = int(seq_before[b].item())
            if sb > 0 and ((sb - 1) // ps) < bt.shape[1]:
                phys = int(bt[b, (sb - 1) // ps].item())
                if 0 < phys < num_blocks:
                    init_ref[b].copy_(pool_ref[phys, off_idx])
        state_ref = init_ref.clone()
        y_ref = causal_dwconv1d_decode_triton(
            x, w_sc, state_ref, residual=True, use_activation=True, round_pre_act=True, clamp_val=1.5
        )
        # Warmup + steady-state cache hit
        pool_2d_w = pool_2d.clone()
        _ = causal_dwconv1d_decode_opaque_triton(
            x, w_sc, pool_2d_w, bt, seq_before,
            page_size=ps, pool_col_offset=off_idx * C * 3,
            residual=True, use_activation=True, round_pre_act=True, clamp_val=1.5,
        )
        y_opq = causal_dwconv1d_decode_opaque_triton(
            x, w_sc, pool_2d, bt, seq_before,
            page_size=ps, pool_col_offset=off_idx * C * 3,
            residual=True, use_activation=True, round_pre_act=True, clamp_val=1.5,
        )
        assert float((y_opq - y_ref).abs().max().item()) == 0.0
        for b in range(B):
            sb = int(seq_before[b].item())
            l_nxt = sb // ps
            if l_nxt < bt.shape[1]:
                phys_w = int(bt[b, l_nxt].item())
                if 0 < phys_w < num_blocks:
                    assert float((pool_3d[phys_w, off_idx] - state_ref[b]).abs().max().item()) == 0.0

        # Active hook still invoked cleanly when registered
        hook_calls = [0]
        def _count_hook(*_args, **_kwargs):
            hook_calls[0] += 1
        enter_hook = knobs.launch_enter_hook
        if hasattr(enter_hook, "calls") and isinstance(enter_hook.calls, list):
            enter_hook.calls.append(_count_hook)
            try:
                assert _has_active_hooks(enter_hook)
                pool_hk = pool_ref.clone().reshape(num_blocks, pool_slots * C * 3)
                y_hk = causal_dwconv1d_decode_opaque_triton(
                    x, w_sc, pool_hk, bt, seq_before,
                    page_size=ps, pool_col_offset=off_idx * C * 3,
                    residual=True, use_activation=True, round_pre_act=True, clamp_val=1.5,
                )
                assert hook_calls[0] >= 1
                assert float((y_hk - y_ref).abs().max().item()) == 0.0
            finally:
                enter_hook.calls.remove(_count_hook)

        # CUDA Graph capture & replay
        pool_cg_3d = pool_ref.clone()
        pool_cg = pool_cg_3d.reshape(num_blocks, pool_slots * C * 3)
        x_cg = x.clone()
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg):
            y_cg = causal_dwconv1d_decode_opaque_triton(
                x_cg, w_sc, pool_cg, bt, seq_before,
                page_size=ps, pool_col_offset=off_idx * C * 3,
                residual=True, use_activation=True, round_pre_act=True, clamp_val=1.5,
            )
        pool_cg_3d.copy_(pool_ref)
        x_new = torch.randn_like(x_cg) * 1.4
        x_cg.copy_(x_new)
        cg.replay()
        y_eag = causal_dwconv1d_decode_opaque_triton(
            x_new, w_sc, pool_ref.clone().reshape(num_blocks, pool_slots * C * 3), bt, seq_before,
            page_size=ps, pool_col_offset=off_idx * C * 3,
            residual=True, use_activation=True, round_pre_act=True, clamp_val=1.5,
        )
        assert float((y_cg - y_eag).abs().max().item()) == 0.0

    # 2. Fused hc_sconv_decode_opaque_combine_norm_triton (5120 split + 256 generic, single_stream, bias, CG)
    for H_hc, single_stream, use_bias in [(5120, False, True), (5120, True, False), (256, False, True)]:
        HC_n = 4
        res = torch.randn((B, 1, H_hc if single_stream else HC_n * H_hc), device=device, dtype=dtype) * 0.4
        sub = torch.randn((B, 1, H_hc), device=device, dtype=dtype) * 1.6
        inj = torch.randn((B, 1, HC_n), device=device, dtype=dtype) * 0.2
        nw = torch.randn((HC_n * H_hc,), device=device, dtype=dtype) * 0.2
        nb = (torch.randn((HC_n * H_hc,), device=device, dtype=dtype) * 0.1) if use_bias else None
        cw = torch.randn((H_hc, 1, 4), device=device, dtype=dtype) * 0.2
        pool_h3 = torch.randn((num_blocks, H_hc, 3), device=device, dtype=dtype) * 0.3
        pool_href = pool_h3.clone()
        bt_h = torch.tensor([[1 + b * 2, 2 + b * 2, 0] for b in range(B)], device=device, dtype=torch.int32)
        sb_h = torch.tensor([0, 1, 63, 64, 65, 128, 130, 192], device=device, dtype=torch.int32)

        y_sc_ref = causal_dwconv1d_decode_opaque_triton(
            sub, cw, pool_href.reshape(num_blocks, H_hc * 3), bt_h, sb_h,
            page_size=ps, residual=True, use_activation=True, round_pre_act=False, clamp_val=1.5,
        )
        c_ref, n_ref = hc_combine_and_norm_triton(y_sc_ref, res, inj, nw, nb, eps=1e-6)
        c_fus, n_fus = hc_sconv_decode_opaque_combine_norm_triton(
            sub, res, inj, nw, nb, cw, pool_h3.reshape(num_blocks, H_hc * 3), bt_h, sb_h,
            page_size=ps, eps=1e-6, conv_residual=True, conv_use_silu=True, clamp_val=1.5,
        )
        c_diff, _ = _diff_stats(c_fus, c_ref)
        n_diff, _ = _diff_stats(n_fus, n_ref)
        cos_n = torch.nn.functional.cosine_similarity(n_fus.float().flatten(), n_ref.float().flatten(), dim=0).item()
        assert c_diff <= 2e-2 and n_diff <= 0.08 and cos_n > 0.99995, (
            f"hc_sconv H={H_hc} ss={single_stream} bias={use_bias}: c_diff={c_diff} n_diff={n_diff} cos={cos_n}"
        )
        for b in range(B):
            sb = int(sb_h[b].item())
            l_nxt = sb // ps
            if l_nxt < bt_h.shape[1]:
                phys_w = int(bt_h[b, l_nxt].item())
                if 0 < phys_w < num_blocks:
                    assert float((pool_h3[phys_w] - pool_href[phys_w]).abs().max().item()) == 0.0

    # 3. PLE forward_packed_prefill_with_hc snap_aligned=True vs False parity
    ple = Qwen4_5PLELayer(
        hidden_size=512,
        hc_count=4,
        ple_embed_dim=256,
        ngram_size=3,
        heads_per_ngram=4,
        conv_kernel_size=4,
        split_ngram_parts=4,
        eos_token_id=0,
        ngram_heads_vocab_sizes=[251] * 8,
        dtype=dtype,
        device=device,
    ).eval()
    ple.norm_key.normal_(0.0, 0.1)
    ple.norm_query.normal_(0.0, 0.1)
    ple.norm_conv.normal_(0.0, 0.1)
    ple.conv1d.weight.normal_(0.0, 0.15)
    ple.set_ngram_table(
        (torch.randn((ple.total_vocab_size, ple.head_dim), device=device, dtype=torch.float32) * 100.0)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    lens = [64, 130]
    cu = torch.tensor([0, 64, 194], device=device, dtype=torch.int32)
    tok_req = torch.cat([torch.zeros(64, device=device, dtype=torch.int32), torch.ones(130, device=device, dtype=torch.int32)])
    tr16 = torch.tensor([0]*4 + [1]*9, device=device, dtype=torch.int32)
    tt16 = torch.tensor([0, 16, 32, 48] + [k*16 for k in range(9)], device=device, dtype=torch.int32)
    s_ends = torch.tensor([64, 64, 128, 130], device=device, dtype=torch.int32)
    s_toks = torch.tensor([63, 64+63, 64+127, 193], device=device, dtype=torch.int32)
    s_reqs = torch.tensor([0, 1, 1, 1], device=device, dtype=torch.int32)
    r_ss = torch.tensor([0, 1, 4], device=device, dtype=torch.int32)
    s_phys = torch.tensor([1, 2, 3, 4], device=device, dtype=torch.int32)
    conv_a = torch.zeros((8, 4 * 512 * 3), device=device, dtype=dtype)
    conv_b = torch.zeros_like(conv_a)
    ctx_a = torch.zeros((8, 2), device=device, dtype=torch.int64)
    ctx_b = torch.zeros_like(ctx_a)
    hh = torch.randn((194, 4 * 512), device=device, dtype=dtype) * 0.2
    f_ids = torch.randint(1, 250, (194,), device=device, dtype=torch.int64)
    nw_p = torch.randn((4 * 512,), device=device, dtype=dtype) * 0.2
    o_a, n_a = ple.forward_packed_prefill_with_hc(
        hh, f_ids, cu, tok_req, tr16, tt16,
        snap_tok_idx=s_toks, snap_req_idx=s_reqs, conv_dst_pool=conv_a, conv_snap_phys=s_phys,
        req_snap_start=r_ss, snap_local_ends=s_ends, snap_aligned=True,
        ctx_dst_pool=ctx_a, ctx_snap_phys=s_phys, hc_norm_weight=nw_p,
    )
    o_b, n_b = ple.forward_packed_prefill_with_hc(
        hh, f_ids, cu, tok_req, tr16, tt16,
        snap_tok_idx=s_toks, snap_req_idx=s_reqs, conv_dst_pool=conv_b, conv_snap_phys=s_phys,
        req_snap_start=r_ss, snap_local_ends=s_ends, snap_aligned=False,
        ctx_dst_pool=ctx_b, ctx_snap_phys=s_phys, hc_norm_weight=nw_p,
    )
    assert float((o_a - o_b).abs().max().item()) == 0.0
    assert float((n_a - n_b).abs().max().item()) == 0.0
    assert float((conv_a - conv_b).abs().max().item()) == 0.0
    assert torch.equal(ctx_a, ctx_b)
    # Cross-dtype weight_scale cache-key regression on BOTH u64 PLE gather paths + HC b_c.dtype + live cache assertions
    from rtp_llm.models_py.modules.qwen4_5.ple import _PLE_TRITON_CACHE, ple_gather_fp8_triton
    from rtp_llm.models_py.modules.qwen4_5.short_conv import _SC_TRITON_CACHE
    from rtp_llm.models_py.modules.qwen4_5.hyper_connection import _HC_TRITON_CACHE, grouped_gemma_rmsnorm_bias_ref

    one_shard = torch.ones((16, 256), device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
    ptr_1 = torch.tensor([one_shard.data_ptr()], device=device, dtype=torch.int64)
    sc_f32 = torch.full((1,), 2.0, device=device, dtype=torch.float32)
    sc_bf16 = torch.full((1,), 2.0, device=device, dtype=torch.bfloat16)
    sc_bf16_3 = torch.full((1,), 3.0, device=device, dtype=torch.bfloat16)
    for ids_probe in (
        torch.zeros((4, 2), device=device, dtype=torch.int64),
        torch.zeros((8,), device=device, dtype=torch.int64),
    ):
        g_f32 = ple_gather_fp8_triton(ids_probe, ptr_1, 16, 256, sc_f32, allow_u64_vec=True)
        g_bf16 = ple_gather_fp8_triton(ids_probe, ptr_1, 16, 256, sc_bf16, allow_u64_vec=True)
        g_bf16_3 = ple_gather_fp8_triton(ids_probe, ptr_1, 16, 256, sc_bf16_3, allow_u64_vec=True)
        assert float((g_f32 - 2.0).abs().max().item()) == 0.0
        assert float((g_bf16 - 2.0).abs().max().item()) == 0.0
        assert float((g_bf16_3 - 3.0).abs().max().item()) == 0.0

    # Mixed bias dtype (fp32 -> bf16) in HC grouped norm & combine+norm
    x_hc = torch.randn((16, 4 * 512), device=device, dtype=dtype) * 0.3
    sub_hc = torch.randn((16, 512), device=device, dtype=dtype) * 0.2
    inj_hc = torch.randn((16, 4), device=device, dtype=dtype) * 0.3
    w_hc = torch.randn((4 * 512,), device=device, dtype=dtype) * 0.2
    for b_hc in (
        torch.randn((4 * 512,), device=device, dtype=torch.float32) * 0.2,
        torch.randn((4 * 512,), device=device, dtype=torch.bfloat16) * 0.2,
    ):
        yn = grouped_gemma_rmsnorm_bias_triton(x_hc, w_hc, b_hc, 4, eps=1e-6)
        yn_r = grouped_gemma_rmsnorm_bias_ref(x_hc, w_hc, b_hc, 4, eps=1e-6)
        assert float((yn - yn_r).abs().max().item()) < 2e-2
        _, cn = hc_combine_and_norm_triton(sub_hc, x_hc, inj_hc, w_hc, b_hc, eps=1e-6)
        cn_r = grouped_gemma_rmsnorm_bias_ref(hc_combine_triton(sub_hc, x_hc, inj_hc), w_hc, b_hc, 4, eps=1e-6)
        assert float((cn - cn_r).abs().max().item()) < 2e-2

    assert len(_PLE_TRITON_CACHE) > 0 and len(_SC_TRITON_CACHE) > 0 and len(_HC_TRITON_CACHE) > 0
    print("  PASS: HookChain fast_launch, opaque decode ShortConv+HC, PLE snap_aligned & cross-dtype ckey parity verified.")


def _r7_sp_shortconv_gloo_worker(rank: int, world_size: int, port: int, cases: dict, err_q) -> None:
    import os
    import torch.distributed as dist

    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        try:
            from rtp_llm.models_py.modules.qwen4_5.fast_comm import (
                Qwen4_5FastAllReduce,
                sp_slice_bounds,
            )

            dev = torch.device("cuda:0")
            torch.cuda.set_device(dev)
            sp_group = dist.new_group(list(range(world_size)))
            C = int(cases["C"])
            sp_fc = Qwen4_5FastAllReduce(sp_group, dev, max_tokens=64, hidden_dims=(C,))
            sc = Qwen4_5ShortConv(
                channels=C,
                kernel_size=4,
                residual=True,
                use_activation=True,
                dtype=torch.bfloat16,
                device=dev,
                sp_size=world_size,
                sp_rank=rank,
                sp_group=sp_group,
            ).eval()
            with torch.no_grad():
                sc.weight.copy_(cases["weight"].to(dev))

                # 1. T=17 non-divisible B=1 packed + state sync
                x17 = cases["x17"].to(dev)
                b0, b1 = sp_slice_bounds(17, world_size, rank)
                st17 = ShortConvState(torch.zeros((1, C, 3), dtype=torch.bfloat16, device=dev))
                y17_sp = sc.forward_sp_packed(
                    x17.view(17, C)[b0:b1], 1, 17, state=st17, sp_fast_comm=sp_fc
                )
                y17_ref = cases["y17_ref"].to(dev).view(17, C)[b0:b1]
                assert float((y17_sp.float() - y17_ref.float()).abs().max().item()) < 2e-2
                assert float((st17.conv_state.float() - cases["st17_ref"].to(dev).float()).abs().max().item()) < 2e-2

                # 2. B=3, T=7 multi-request boundary isolation (mid-request SP splits)
                xb3 = cases["xb3"].to(dev)
                s0, s1 = sp_slice_bounds(3 * 7, world_size, rank)
                stb3 = ShortConvState(torch.zeros((3, C, 3), dtype=torch.bfloat16, device=dev))
                yb3_sp = sc.forward_sp_packed(
                    xb3.view(21, C)[s0:s1], 3, 7, state=stb3, sp_fast_comm=sp_fc
                )
                yb3_ref = cases["yb3_ref"].to(dev).view(21, C)[s0:s1]
                assert float((yb3_sp.float() - yb3_ref.float()).abs().max().item()) < 2e-2
                # Perturb request 0; request 1 & 2 outputs must remain invariant
                xb3_pert = xb3.clone()
                xb3_pert[0, -2:].add_(1.5)
                stb3_p = ShortConvState(torch.zeros((3, C, 3), dtype=torch.bfloat16, device=dev))
                yb3_pert = sc.forward_sp_packed(
                    xb3_pert.view(21, C)[s0:s1], 3, 7, state=stb3_p, sp_fast_comm=sp_fc
                )
                full_orig = sp_fc.all_gather_sp(yb3_sp, total_tokens=21).view(3, 7, C)
                full_pert = sp_fc.all_gather_sp(yb3_pert, total_tokens=21).view(3, 7, C)
                assert float((full_orig[1:].float() - full_pert[1:].float()).abs().max().item()) == 0.0

                # 3. T=3 < sp_size=4 (t_loc=1 < 3 multi-rank left halo + empty rank M_loc=0)
                x3 = cases["x3"].to(dev)
                e0, e1 = sp_slice_bounds(3, world_size, rank)
                st3 = ShortConvState(torch.zeros((1, C, 3), dtype=torch.bfloat16, device=dev))
                y3_sp = sc.forward_sp_packed(
                    x3.view(3, C)[e0:e1], 1, 3, state=st3, sp_fast_comm=sp_fc
                )
                assert y3_sp.shape == (e1 - e0, C)
                if e1 > e0:
                    assert float((y3_sp.float() - cases["y3_ref"].to(dev).view(3, C)[e0:e1].float()).abs().max().item()) < 2e-2
                assert float((st3.conv_state.float() - cases["st3_ref"].to(dev).float()).abs().max().item()) < 2e-2
                halo3 = sc.exchange_sp_left_halo(x3[:, e0:e1, :])
                assert halo3.shape == (1, C, 3)
                if e1 > e0 and e0 >= 3:
                    assert float((halo3[0].float() - x3[0, e0 - 3 : e0, :].transpose(0, 1).float()).abs().max().item()) == 0.0

                # 4. B=5, T=1 decode across 4 ranks + replicated state sync
                xd5 = cases["xd5"].to(dev)
                d0, d1 = sp_slice_bounds(5, world_size, rank)
                std5 = ShortConvState(cases["std5_init"].to(dev).clone())
                yd5_sp = sc.forward_sp_packed(
                    xd5.view(5, C)[d0:d1], 5, 1, state=std5, sp_fast_comm=sp_fc
                )
                assert yd5_sp.shape == (d1 - d0, C)
                if d1 > d0:
                    assert float((yd5_sp.float() - cases["yd5_ref"].to(dev).view(5, C)[d0:d1].float()).abs().max().item()) < 2e-2
                assert float((std5.conv_state.float() - cases["std5_ref"].to(dev).float()).abs().max().item()) < 2e-2
            dist.barrier()
        finally:
            dist.destroy_process_group()
    except BaseException as exc:
        err_q.put(f"rank {rank}: {exc!r}")
        raise


def test_r7_fused_hc_epilogues_branch5120_workspaces_and_sp_shortconv() -> None:
    print("\n=== [3g] R7 Fused HC Epilogues, Branch5120 Decode, Static Workspaces & SP ShortConv ===")
    import socket
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torch.nn.functional as F

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(202670)

    with torch.no_grad():
        # 1. hc_mix_reduce_with_down_inject_triton (has_latent=True/False) & hc_down_silu_epilogue_triton
        for H in (512, 5120):
            HC, lowrank, pad_dim = 4, 32 if H == 512 else 128, 64 if H == 512 else 144
            for M in (0, 1, 7, 32):
                logits = torch.randn((1, M, HC * H), device=device, dtype=dtype) * 0.3
                normed = torch.randn((1, M, HC * H), device=device, dtype=dtype) * 0.3
                raw = torch.randn((1, M, pad_dim), device=device, dtype=dtype) * 0.3
                for has_latent in (True, False):
                    out_f, lat_f, inj_f = hc_mix_reduce_with_down_inject_triton(
                        logits, normed, raw, lowrank, HC, has_latent=has_latent, has_inj=True
                    )
                    out_r = hc_mix_reduce_triton(logits, normed, HC)
                    if has_latent:
                        lat_r, inj_r = hc_down_inject_epilogue_triton(raw, lowrank, HC)
                        assert _diff_stats(lat_f, lat_r)[0] < 1e-2
                    else:
                        inj_r = hc_inject_epilogue_triton(raw[..., :HC], HC)
                    assert _diff_stats(out_f, out_r)[0] < 1e-2
                    assert _diff_stats(inj_f, inj_r)[0] < 1e-2
                silu_f = hc_down_silu_epilogue_triton(raw, lowrank, HC)
                silu_r = F.silu(raw[..., :lowrank].float() / HC).to(dtype)
                assert _diff_stats(silu_f, silu_r)[0] < 1e-2

        # 2. H=5120 _hc_combine_and_norm_branch5120 & opaque decode fast path (B=1, 8, 16, 65) + CG
        H, HC, ps = 5120, 4, 64
        w_sc = torch.randn((H, 1, 4), device=device, dtype=dtype) * 0.2
        w_hc = torch.randn((HC * H,), device=device, dtype=dtype) * 0.2
        b_hc = torch.randn((HC * H,), device=device, dtype=dtype) * 0.2
        for B in (1, 8, 16, 65):
            raw_sub = torch.randn((B, 1, H), device=device, dtype=dtype) * 0.3
            res = torch.randn((B, 1, HC * H), device=device, dtype=dtype) * 0.3
            inj = torch.rand((B, 1, HC), device=device, dtype=dtype) * 1.5
            pool_3d = torch.randn((B * 2 + 2, 1, H, 3), device=device, dtype=dtype) * 0.2
            pool_fast = pool_3d.clone().reshape(-1, H * 3)
            pool_ref = pool_3d.clone().reshape(-1, H * 3)
            bt = torch.arange(1, B * 2 + 1, dtype=torch.int32, device=device).view(B, 2)
            sb = torch.full((B,), 10, dtype=torch.int32, device=device)

            comb_f, norm_f = hc_sconv_decode_opaque_combine_norm_triton(
                raw_sub, res, inj, w_hc, b_hc, w_sc, pool_fast, bt, sb,
                page_size=ps, conv_residual=True, conv_use_silu=True,
                clamp_val=1.5,
            )
            sub_ref = causal_dwconv1d_decode_opaque_triton(
                raw_sub, w_sc, pool_ref, bt, sb,
                page_size=ps, pool_col_offset=0, residual=True, use_activation=True,
                round_pre_act=False, clamp_val=1.5,
            )
            comb_r, norm_r = hc_combine_and_norm_triton(sub_ref, res, inj, w_hc, b_hc, eps=1e-6)
            assert _diff_stats(comb_f, comb_r)[0] < 2e-2
            assert _diff_stats(norm_f, norm_r)[0] < 2e-2
            assert float((pool_fast - pool_ref).abs().max().item()) == 0.0

        # Measure B=1/8/16 CG latency for _hc_sconv_decode_opaque_combine_norm (branch5120 path)
        for B in (1, 8, 16):
            raw_b = torch.randn((B, 1, H), device=device, dtype=dtype) * 0.3
            res_b = torch.randn((B, 1, HC * H), device=device, dtype=dtype) * 0.3
            inj_b = torch.rand((B, 1, HC), device=device, dtype=dtype) * 1.5
            pool_b = torch.randn((B * 2 + 2, H * 3), device=device, dtype=dtype) * 0.2
            bt_b = torch.arange(1, B * 2 + 1, dtype=torch.int32, device=device).view(B, 2)
            sb_b = torch.full((B,), 10, dtype=torch.int32, device=device)
            us_b = _bench_us(
                lambda: hc_sconv_decode_opaque_combine_norm_triton(
                    raw_b, res_b, inj_b, w_hc, b_hc, w_sc, pool_b, bt_b, sb_b,
                    page_size=ps, conv_residual=True, conv_use_silu=True,
                    clamp_val=1.5,
                )
            )
            print(f"  [Bench R7] hc_sconv_decode_opaque_combine_norm B={B} (branch5120): {us_b:.2f} us")

        # 3. hc_dflash_gconv_combine_norm_triton parity (H=512 & H=5120 rows>=64)
        for H_gc, rows_gc in ((512, 16), (5120, 64)):
            taps, bs, gs = 4, 8, 512
            n_grp = H_gc // gs
            raw_gc = torch.randn((rows_gc, H_gc), device=device, dtype=dtype) * 0.3
            delta_gc = torch.randn((rows_gc, taps, n_grp), device=device, dtype=dtype) * 0.2
            base_gc = torch.randn((taps, H_gc), device=device, dtype=dtype) * 0.2
            res_gc = torch.randn((rows_gc, HC * H_gc), device=device, dtype=dtype) * 0.3
            inj_gc = torch.rand((rows_gc, HC), device=device, dtype=dtype) * 1.5
            w_gc = torch.randn((HC * H_gc,), device=device, dtype=dtype) * 0.2
            b_gc = torch.randn((HC * H_gc,), device=device, dtype=dtype) * 0.2
            comb_d, norm_d = hc_dflash_gconv_combine_norm_triton(
                raw_gc, delta_gc, base_gc, bs, gs, taps, res_gc, inj_gc, w_gc, b_gc, clip_value=10.0
            )
            row_in_blk = torch.arange(rows_gc, device=device) % bs
            grp_ids = torch.arange(H_gc, device=device) // gs
            acc = torch.zeros_like(raw_gc, dtype=torch.float32)
            for k in range(taps):
                src_row = (torch.arange(rows_gc, device=device) - k).clamp_min(0)
                mask = (row_in_blk >= k).float().unsqueeze(-1)
                eff_w = base_gc[k].float().unsqueeze(0) + delta_gc[:, k, :][:, grp_ids].float()
                acc += eff_w * raw_gc[src_row].float() * mask
            sub_gc_ref = acc.clamp(-10.0, 10.0).to(dtype)
            comb_ref = hc_combine_ref(sub_gc_ref, res_gc, inj_gc)
            norm_ref = grouped_gemma_rmsnorm_bias_ref(comb_ref, w_gc, b_gc, HC, eps=1e-6)
            assert _diff_stats(comb_d, comb_ref)[0] < 2e-2
            assert _diff_stats(norm_d, norm_ref)[0] < 2e-2

        # 4. HCStaticWorkspace & ShortConvStaticWorkspace ping-pong non-aliasing + eager reuse + grad safety
        hc_mod = Qwen4_5HyperConnection(hidden_size=512, hc_count=4, hc_lowrank=32, dtype=dtype, device=device).eval()
        hc_ws = HCStaticWorkspace()
        hi = torch.randn((2, 8, 4 * 512), device=device, dtype=dtype) * 0.2
        prev_l = torch.randn((2, 8, 32), device=device, dtype=dtype) * 0.2
        m0, r0, ij0, lt0 = hc_mod.mix_with_latent(hi, prev_l, ws=hc_ws)
        m0_snap, r0_snap = m0.clone(), r0.clone()
        m1, r1, ij1, lt1 = hc_mod.combine_and_mix_with_latent(m0, r0, ij0, lt0, ws=hc_ws)
        assert float((m0 - m0_snap).abs().max().item()) == 0.0
        assert float((r0 - r0_snap).abs().max().item()) == 0.0
        assert len(hc_ws._pool) > 0
        # Grad mode bypasses static pool cleanly
        with torch.enable_grad():
            hi_g = hi.detach().requires_grad_(True)
            mg, rg, ijg, ltg = hc_mod.mix_with_latent(hi_g, prev_l, ws=hc_ws)
            assert rg.requires_grad and mg.data_ptr() != m0.data_ptr() and mg.data_ptr() != m1.data_ptr()
            assert float((m0 - m0_snap).abs().max().item()) == 0.0

        sc_ws = ShortConvStaticWorkspace()
        w_rag = torch.randn((512, 1, 4), device=device, dtype=dtype) * 0.2
        x_rag = torch.randn((18, 512), device=device, dtype=dtype) * 0.3
        cu_rag = torch.tensor([0, 7, 18], dtype=torch.int32, device=device)
        t_req_rag = torch.tensor([0, 1], dtype=torch.int32, device=device)
        t_t0_rag = torch.tensor([0, 0], dtype=torch.int32, device=device)
        y_rag0 = causal_dwconv1d_ragged_prefill_with_snapshots_triton(
            x_rag, w_rag, cu_rag, t_req_rag, t_t0_rag, ws=sc_ws
        )
        y_rag0_snap = y_rag0.clone()
        y_rag1 = causal_dwconv1d_ragged_prefill_with_snapshots_triton(
            x_rag + 0.5, w_rag, cu_rag, t_req_rag, t_t0_rag, ws=sc_ws
        )
        assert float((y_rag0 - y_rag0_snap).abs().max().item()) == 0.0
        assert len(sc_ws._pool) > 0

    # 5. Real 4-rank Gloo+CUDA Qwen4_5ShortConv.forward_sp_packed & exchange_sp_left_halo
    with torch.no_grad():
        C_sp = 256
        sc_ref = Qwen4_5ShortConv(
            channels=C_sp, kernel_size=4, residual=True, use_activation=True, dtype=dtype, device=device
        ).eval()
        sc_ref.weight.normal_(0.0, 0.15)
        x17 = torch.randn((1, 17, C_sp), device=device, dtype=dtype) * 0.4
        xb3 = torch.randn((3, 7, C_sp), device=device, dtype=dtype) * 0.4
        x3 = torch.randn((1, 3, C_sp), device=device, dtype=dtype) * 0.4
        xd5 = torch.randn((5, 1, C_sp), device=device, dtype=dtype) * 0.4
        std5_init = torch.randn((5, C_sp, 3), device=device, dtype=dtype) * 0.2

        st17_ref = ShortConvState(torch.zeros((1, C_sp, 3), dtype=dtype, device=device))
        y17_ref = sc_ref(x17, state=st17_ref)
        stb3_ref = ShortConvState(torch.zeros((3, C_sp, 3), dtype=dtype, device=device))
        yb3_ref = sc_ref(xb3, state=stb3_ref)
        st3_ref = ShortConvState(torch.zeros((1, C_sp, 3), dtype=dtype, device=device))
        y3_ref = sc_ref(x3, state=st3_ref)
        std5_ref = ShortConvState(std5_init.clone())
        yd5_ref = sc_ref(xd5, state=std5_ref)
        cases = {
            "C": C_sp,
            "weight": sc_ref.weight.cpu(),
            "x17": x17.cpu(), "y17_ref": y17_ref.cpu(), "st17_ref": st17_ref.conv_state.cpu(),
            "xb3": xb3.cpu(), "yb3_ref": yb3_ref.cpu(),
            "x3": x3.cpu(), "y3_ref": y3_ref.cpu(), "st3_ref": st3_ref.conv_state.cpu(),
            "xd5": xd5.cpu(), "std5_init": std5_init.cpu(), "yd5_ref": yd5_ref.cpu(), "std5_ref": std5_ref.conv_state.cpu(),
        }

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ctx = mp.get_context("spawn")
    err_q = ctx.Queue()
    procs = [ctx.Process(target=_r7_sp_shortconv_gloo_worker, args=(r, 4, port, cases, err_q)) for r in range(4)]
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
            raise TimeoutError(f"R7 ShortConv SP worker {r} timed out")
        assert p.exitcode == 0, f"R7 ShortConv SP worker {r} exited {p.exitcode}: {errors}"
    assert errors == [], f"R7 ShortConv SP worker errors: {errors}"
    print("  PASS: R7 fused HC epilogues, branch5120, static workspaces & 4-rank SP ShortConv verified.")


@torch.no_grad()
def test_r8_hc_splitk_and_dflash_branch5120() -> None:
    import torch.nn.functional as F

    print("[Test R8] HC Triton Split-K down_inject + DFlash2 branch5120...")
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    hc = Qwen4_5HyperConnection(
        hidden_size=5120, hc_count=4, hc_lowrank=128, dtype=dtype, device=device
    ).eval()
    ws = HCStaticWorkspace()
    for M in (1, 8, 16, 32, 64, 128, 256, 512):
        hi = torch.randn((M, 4 * 5120), dtype=dtype, device=device) * 0.25
        prev_l = torch.randn((M, 128), dtype=dtype, device=device) * 0.25
        m_ws, r_ws, ij_ws, lt_ws = hc.mix_with_latent(hi, prev_latent=prev_l, ws=ws)
        m_no, r_no, ij_no, lt_no = hc.mix_with_latent(hi, prev_latent=prev_l, ws=None)
        m_ref, _, ij_ref, lt_ref = hc.mix_with_latent_ref(hi, prev_latent=prev_l)
        for a, b in ((m_ws, m_no), (r_ws, r_no), (ij_ws, ij_no), (lt_ws, lt_no)):
            assert float((a - b).abs().max().item()) == 0.0, f"M={M} ws vs no_ws mismatch"
        for name, a, b in (("mixed", m_ws, m_ref), ("inj", ij_ws, ij_ref), ("latent", lt_ws, lt_ref)):
            cos = F.cosine_similarity(a.float().reshape(-1), b.float().reshape(-1), dim=0).item()
            assert cos >= 0.9999, f"R8 HC Split-K M={M} {name} cos={cos}"

    # Verify CUDA Graph capture + replay and 0 splitKreduce / 0 cutlass_80 at M=16
    hi16 = torch.randn((16, 4 * 5120), dtype=dtype, device=device) * 0.25
    pl16 = torch.randn((16, 128), dtype=dtype, device=device) * 0.25
    sub16 = torch.randn((16, 5120), dtype=dtype, device=device) * 0.25
    ij16 = torch.rand((16, 4), dtype=dtype, device=device)
    _ = hc.combine_and_mix_with_latent(sub16, hi16, ij16, prev_latent=pl16, ws=ws)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        cm_cg, cr_cg, cij_cg, clt_cg = hc.combine_and_mix_with_latent(sub16, hi16, ij16, prev_latent=pl16, ws=ws)
    g.replay()
    torch.cuda.synchronize()
    cm_ref, cr_ref, cij_ref, clt_ref = hc.combine_and_mix_with_latent_ref(sub16, hi16, ij16, prev_latent=pl16)
    for a, b in ((cm_cg, cm_ref), (cr_cg, cr_ref), (cij_cg, cij_ref), (clt_cg, clt_ref)):
        assert F.cosine_similarity(a.float().reshape(-1), b.float().reshape(-1), dim=0).item() >= 0.9999

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        _ = hc.mix_with_latent(hi16, prev_latent=pl16, ws=ws)
        torch.cuda.synchronize()
    cnames = [
        str(getattr(e, "name", "")).lower()
        for e in prof.events()
        if e.device_type == torch.profiler.DeviceType.CUDA and getattr(e, "device_time_total", 0) > 0
    ]
    assert not any("splitkreduce" in nm for nm in cnames), f"Unexpected splitKreduce: {cnames}"
    assert not any("cutlass_80" in nm for nm in cnames), f"Unexpected cutlass_80: {cnames}"
    print("  PASS: R8 HC Split-K & DFlash2 branch5120 verified (0 splitKreduce, 0 cutlass_80).")


def test_r9_l2_swapped_hc_and_norm5120() -> None:
    import torch.nn.functional as F

    print("[Test R9] Swapped (HC, row) HC combine+norm, split-5120 grouped norm & DFlash2/mix_reduce parity...")
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.manual_seed(202690)
    H, HC, LR = 5120, 4, 128

    with torch.no_grad():
        # 1. grouped_gemma_rmsnorm_bias_triton & mix_with_latent split-5120 across crossover
        hc = Qwen4_5HyperConnection(H, HC, LR, dtype=dtype, device=device).eval()
        for rows, single_stream, use_bias in (
            (16, False, True),
            (64, True, False),
            (128, False, True),
            (256, True, True),
            (1024, False, False),
            (1536, False, True),
        ):
            xr = torch.randn((rows, H if single_stream else HC * H), device=device, dtype=dtype) * 0.3
            nw = torch.randn((HC * H,), device=device, dtype=dtype) * 0.2
            nb = (torch.randn((HC * H,), device=device, dtype=dtype) * 0.1) if use_bias else None
            n_tri = grouped_gemma_rmsnorm_bias_triton(xr, nw, nb, HC, eps=1e-6)
            n_ref = grouped_gemma_rmsnorm_bias_ref(xr, nw, nb, HC, eps=1e-6)
            cos_g = F.cosine_similarity(n_tri.float().flatten(), n_ref.float().flatten(), dim=0).item()
            assert cos_g >= 0.99999 and _diff_stats(n_tri, n_ref)[0] < 0.08, f"grouped norm rows={rows}: cos={cos_g}"

            if not single_stream:
                pl = torch.randn((rows, LR), device=device, dtype=dtype) * 0.2
                m_tri, _, ij_tri, lt_tri = hc.mix_with_latent(xr, prev_latent=pl)
                m_ref, _, ij_ref, lt_ref = hc.mix_with_latent_ref(xr, prev_latent=pl)
                for a, b in ((m_tri, m_ref), (ij_tri, ij_ref), (lt_tri, lt_ref)):
                    assert F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item() >= 0.9999

        # 2. hc_combine_and_norm_triton across branch/swapped crossover (16..2048) + CG
        for rows, single_stream, use_bias in (
            (16, False, True),
            (64, True, False),
            (128, False, True),
            (512, False, True),
            (2048, False, True),
            (2048, True, False),
        ):
            res = torch.randn((rows, H if single_stream else HC * H), device=device, dtype=dtype) * 0.35
            sub = torch.randn((rows, H), device=device, dtype=dtype) * 0.35
            inj = torch.randn((rows, HC), device=device, dtype=dtype) * 0.25
            c_tri, n_tri = hc_combine_and_norm_triton(sub, res, inj, hc.hc_norm_weight, hc.hc_norm_bias if use_bias else None, eps=1e-6)
            c_ref = hc_combine_ref(sub, res, inj)
            n_ref = grouped_gemma_rmsnorm_bias_ref(c_ref, hc.hc_norm_weight, hc.hc_norm_bias if use_bias else None, HC, eps=1e-6)
            assert _diff_stats(c_tri, c_ref)[0] < 2e-2
            assert F.cosine_similarity(n_tri.float().flatten(), n_ref.float().flatten(), dim=0).item() >= 0.99999

        # 3. hc_dflash_gconv_combine_norm_triton across rows <128 and >=128 + CG
        taps, bs, gs = 4, 8, 512
        n_grp = H // gs
        base_gc = torch.randn((taps, H), device=device, dtype=dtype) * 0.2
        w_gc = torch.randn((HC * H,), device=device, dtype=dtype) * 0.2
        b_gc = torch.randn((HC * H,), device=device, dtype=dtype) * 0.2
        for rows_gc in (32, 128, 256, 1024):
            raw_gc = torch.randn((rows_gc, H), device=device, dtype=dtype) * 0.3
            delta_gc = torch.randn((rows_gc, taps, n_grp), device=device, dtype=dtype) * 0.2
            res_gc = torch.randn((rows_gc, HC * H), device=device, dtype=dtype) * 0.3
            inj_gc = torch.rand((rows_gc, HC), device=device, dtype=dtype) * 1.5
            comb_d, norm_d = hc_dflash_gconv_combine_norm_triton(
                raw_gc, delta_gc, base_gc, bs, gs, taps, res_gc, inj_gc, w_gc, b_gc, clip_value=10.0
            )
            row_in_blk = torch.arange(rows_gc, device=device) % bs
            grp_ids = torch.arange(H, device=device) // gs
            acc = torch.zeros_like(raw_gc, dtype=torch.float32)
            for k in range(taps):
                src_row = (torch.arange(rows_gc, device=device) - k).clamp_min(0)
                mask = (row_in_blk >= k).float().unsqueeze(-1)
                eff_w = base_gc[k].float().unsqueeze(0) + delta_gc[:, k, :][:, grp_ids].float()
                acc += eff_w * raw_gc[src_row].float() * mask
            sub_gc_ref = acc.clamp(-10.0, 10.0).to(dtype)
            comb_ref = hc_combine_ref(sub_gc_ref, res_gc, inj_gc)
            norm_ref = grouped_gemma_rmsnorm_bias_ref(comb_ref, w_gc, b_gc, HC, eps=1e-6)
            assert _diff_stats(comb_d, comb_ref)[0] < 2e-2
            assert F.cosine_similarity(norm_d.float().flatten(), norm_ref.float().flatten(), dim=0).item() >= 0.99999
        g_gc = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_gc):
            c_gc_cg, n_gc_cg = hc_dflash_gconv_combine_norm_triton(
                raw_gc, delta_gc, base_gc, bs, gs, taps, res_gc, inj_gc, w_gc, b_gc, clip_value=10.0
            )
        raw_gc.copy_(torch.randn_like(raw_gc) * 0.3)
        g_gc.replay()
        c_gc_eag, n_gc_eag = hc_dflash_gconv_combine_norm_triton(
            raw_gc, delta_gc, base_gc, bs, gs, taps, res_gc, inj_gc, w_gc, b_gc, clip_value=10.0
        )
        assert float((c_gc_cg - c_gc_eag).abs().max().item()) == 0.0
        assert float((n_gc_cg - n_gc_eag).abs().max().item()) == 0.0

        # 4. hc_mix_reduce_with_down_inject_triton across <2048 and >=2048 + CG
        for rows_m in (32, 2048, 2560):
            lg = torch.randn((rows_m, HC * H), device=device, dtype=dtype) * 0.3
            nm = torch.randn((rows_m, HC * H), device=device, dtype=dtype) * 0.3
            rw = torch.randn((rows_m, 144), device=device, dtype=dtype) * 0.3
            out_f, lat_f, inj_f = hc_mix_reduce_with_down_inject_triton(lg, nm, rw, LR, HC, has_latent=True, has_inj=True)
            out_r = hc_mix_reduce_triton(lg, nm, HC)
            lat_r, inj_r = hc_down_inject_epilogue_triton(rw, LR, HC)
            assert _diff_stats(out_f, out_r)[0] < 1e-2
            assert _diff_stats(lat_f, lat_r)[0] < 1e-2
            assert _diff_stats(inj_f, inj_r)[0] < 1e-2
        g_m = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_m):
            om_cg, lm_cg, im_cg = hc_mix_reduce_with_down_inject_triton(lg, nm, rw, LR, HC, has_latent=True, has_inj=True)
        lg.copy_(torch.randn_like(lg) * 0.3)
        g_m.replay()
        om_eag, lm_eag, im_eag = hc_mix_reduce_with_down_inject_triton(lg, nm, rw, LR, HC, has_latent=True, has_inj=True)
        for a, b in ((om_cg, om_eag), (lm_cg, lm_eag), (im_cg, im_eag)):
            assert float((a - b).abs().max().item()) == 0.0

    with torch.enable_grad():
        xr_g = (torch.randn((256, HC * H), device=device, dtype=dtype) * 0.2).requires_grad_(True)
        pl_g = torch.randn((256, LR), device=device, dtype=dtype) * 0.2
        mg, rg, _, _ = hc.mix_with_latent(xr_g, prev_latent=pl_g, ws=HCStaticWorkspace())
        assert rg.requires_grad and not mg.requires_grad
    print("  PASS: R9 swapped (HC, row) HC combine+norm, split-5120 norm & DFlash2/mix_reduce verified.")


def test_r11_hc_up_mix_sk_1d_and_sconv_decode() -> None:
    import torch.nn.functional as F

    print("[Test R11] Fused _hc_up_mix_sk_1d_kernel (0 nvjet) + ShortConv BLOCK_C/nw parity...")
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.manual_seed(202611)
    H, HC, LR = 5120, 4, 128
    ws = HCStaticWorkspace()

    with torch.no_grad():
        hc = Qwen4_5HyperConnection(H, HC, LR, dtype=dtype, device=device).eval()
        for M in (1, 4, 8, 16, 17, 24, 32, 48, 64):
            hi_pad = torch.randn((M, HC * H + 8), dtype=dtype, device=device) * 0.25
            hi = hi_pad[:, : HC * H]
            pl_pad = torch.randn((M, LR + 4), dtype=dtype, device=device) * 0.25
            pl = pl_pad[:, :LR]
            m_ws, r_ws, ij_ws, lt_ws = hc.mix_with_latent(hi, prev_latent=pl, ws=ws)
            m_no, r_no, ij_no, lt_no = hc.mix_with_latent(hi, prev_latent=pl, ws=None)
            m_ref, _, ij_ref, lt_ref = hc.mix_with_latent_ref(hi, prev_latent=pl)
            for a, b in ((m_ws, m_no), (r_ws, r_no), (ij_ws, ij_no), (lt_ws, lt_no)):
                assert float((a - b).abs().max().item()) == 0.0, f"M={M} ws vs no_ws mismatch"
            for name, a, b in (("mixed", m_ws, m_ref), ("inj", ij_ws, ij_ref), ("latent", lt_ws, lt_ref)):
                cos = F.cosine_similarity(a.float().reshape(-1), b.float().reshape(-1), dim=0).item()
                assert cos >= 0.9999, f"R11 HC M={M} {name} cos={cos}"

        # Verify CUDA Graph + 0 splitKreduce / 0 cutlass_80 / 0 nvjet across B=1,16,48
        for cg_b in (1, 16, 48):
            hi_cg = torch.randn((cg_b, HC * H), dtype=dtype, device=device) * 0.25
            pl_cg = torch.randn((cg_b, LR), dtype=dtype, device=device) * 0.25
            _ = hc.mix_with_latent(hi_cg, prev_latent=pl_cg, ws=ws)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                mo_cg, _, io_cg, lo_cg = hc.mix_with_latent(hi_cg, prev_latent=pl_cg, ws=ws)
            g.replay()
            torch.cuda.synchronize()
            mo_ref, _, io_ref, lo_ref = hc.mix_with_latent_ref(hi_cg, prev_latent=pl_cg)
            for a, b in ((mo_cg, mo_ref), (io_cg, io_ref), (lo_cg, lo_ref)):
                assert F.cosine_similarity(a.float().reshape(-1), b.float().reshape(-1), dim=0).item() >= 0.9999

            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                _ = hc.mix_with_latent(hi_cg, prev_latent=pl_cg, ws=ws)
                torch.cuda.synchronize()
            cnames = [
                str(getattr(e, "name", "")).lower()
                for e in prof.events()
                if e.device_type == torch.profiler.DeviceType.CUDA and getattr(e, "device_time_total", 0) > 0
            ]
            assert any("_hc_up_mix_sk_1d_kernel" in nm for nm in cnames), f"M={cg_b} missing _hc_up_mix_sk_1d_kernel: {cnames}"
            assert not any("nvjet" in nm or "cutlass_80" in nm or "splitkreduce" in nm for nm in cnames), (
                f"Unexpected GEMM/splitkreduce kernel at M={cg_b}: {cnames}"
            )

        # ShortConv C=256 and C=5120 opaque/dense decode BLOCK_C/num_warps parity across B=1,16,32,64
        for C in (256, 5120):
            conv = Qwen4_5ShortConv(C, kernel_size=4, dtype=dtype, device=device).eval()
            tpb, max_blk, n_blk = 16, 4, 256
            for b_sc in (1, 16, 32, 64):
                x_sc = torch.randn((b_sc, 1, C), dtype=dtype, device=device) * 0.25
                pool = torch.randn((n_blk, C * 3), dtype=dtype, device=device) * 0.25
                pool_ref = pool.clone()
                bt = torch.arange(1, b_sc * max_blk + 1, dtype=torch.int32, device=device).reshape(b_sc, max_blk)
                seq_b = torch.full((b_sc,), 15, dtype=torch.int32, device=device)
                out_op = causal_dwconv1d_decode_opaque_triton(
                    x_sc, conv.weight, pool, bt, seq_b, page_size=tpb,
                    residual=conv.residual, use_activation=conv.use_activation, clamp_val=10.0,
                )
                st_dense = ShortConvState(pool_ref[bt[:, 0]].reshape(b_sc, C, 3).clone())
                out_de = causal_dwconv1d_decode_triton(
                    x_sc, conv.weight, st_dense.conv_state,
                    residual=conv.residual, use_activation=conv.use_activation, clamp_val=10.0,
                )
                assert F.cosine_similarity(out_op.float().reshape(-1), out_de.float().reshape(-1), dim=0).item() >= 0.99999
                assert float((out_op - out_de).abs().max().item()) == 0.0

    print("  PASS: R11 HC 1D fused up-mix (M=1..48, 0 nvjet/cutlass_80) & ShortConv parity verified.")


def test_r14_ple_add3_gather_l2_and_sconv_t8_unroll() -> None:
    print("[Test R14-D0] PLE add3/gather & ShortConv 8-step unrolled ragged prefill parity...")
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # 1. PLE add3 (num_warps=8) + gpu_sharded gather parity across odd M and 16K
    for M in (7, 256, 1024, 4097):
        h = torch.randn((1, M, 20480), dtype=dtype, device=device)
        u = torch.randn((1, M, 20480), dtype=dtype, device=device)
        c = torch.randn((1, M, 20480), dtype=dtype, device=device)
        out = ple_add3_triton(h, u, c)
        ref = h + (u + c)
        assert float((out - ref).abs().max().item()) == 0.0, f"ple_add3 M={M} mismatch"

    shard_rows = 256
    shards = [torch.randint(0, 127, (shard_rows, 256), dtype=torch.uint8, device=device).view(torch.float8_e4m3fn) for _ in range(16)]
    ptrs = torch.tensor([s.data_ptr() for s in shards], dtype=torch.int64, device=device)
    w_scale = torch.tensor(0.02, dtype=dtype, device=device)
    ng_ids = torch.randint(0, 16 * shard_rows - 1, (1, 1024, 20), dtype=torch.int64, device=device)
    out_g = ple_gather_fp8_triton(ng_ids, ptrs, shard_rows, 256, w_scale, is_fp8=True, is_cpu_uva=False)
    flat_table = torch.cat(shards, dim=0).to(dtype) * w_scale
    ref_g = flat_table[ng_ids]
    assert float((out_g - ref_g).abs().max().item()) == 0.0

    # 2. ShortConv 8-step unroll ragged prefill with fused snapshots & tail lengths (<8, ==8, %8 != 0)
    for C, res, silu in ((256, True, False), (5120, True, True), (12288, False, True), (20480, False, False)):
        w = torch.randn((C, 1, 4), dtype=dtype, device=device) * 0.25
        lens = [5, 8, 13, 135]
        N = sum(lens)
        x = torch.randn((N, C), dtype=dtype, device=device) * 0.25
        cu = torch.tensor([0, 5, 13, 26, N], dtype=torch.int32, device=device)
        bt = 32
        t_req, t_t0 = [], []
        for b, ln in enumerate(lens):
            for t0 in range(0, ln, bt):
                t_req.append(b)
                t_t0.append(t0)
        tile_req = torch.tensor(t_req, dtype=torch.int32, device=device)
        tile_t0 = torch.tensor(t_t0, dtype=torch.int32, device=device)
        num_blocks = 32
        pool = torch.zeros((num_blocks, C * 3), dtype=dtype, device=device)
        pool_ref = torch.zeros_like(pool)
        # Snapshots at request 3 page_size=16 mid-tile & tile-end positions plus non-aligned tail
        snap_ends_list = [16, 32, 48, 64, 80, 96, 112, 128, 132]
        snap_dst_list = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        snap_ends = torch.tensor(snap_ends_list, dtype=torch.int32, device=device)
        req_ss = torch.tensor([0, 0, 0, 0, len(snap_ends_list)], dtype=torch.int32, device=device)
        snap_dst = torch.tensor(snap_dst_list, dtype=torch.int32, device=device)
        out = causal_dwconv1d_ragged_prefill_with_snapshots_triton(
            x, w, cu, tile_req, tile_t0,
            dst_pool=pool, snap_dst_idx=snap_dst, req_snap_start=req_ss, snap_local_ends=snap_ends,
            snap_aligned=True, residual=res, use_activation=silu, block_t=bt, clamp_val=2.0,
        )
        # Reference per request using causal_dwconv1d_ref
        pos = 0
        for b, ln in enumerate(lens):
            xb = x[pos : pos + ln].unsqueeze(0)
            ref_b = causal_dwconv1d_ref(xb, w, None, residual=res, use_activation=silu, clamp_val=2.0)
            assert float((out[pos : pos + ln] - ref_b.squeeze(0)).abs().max().item()) <= 1e-2, f"C={C} req={b}"
            if b == 3:
                for idx_s, end_l in zip(snap_dst_list, snap_ends_list):
                    st_expected = x[pos + end_l - 3 : pos + end_l].t().reshape(C * 3)
                    pool_ref[idx_s].copy_(st_expected)
            pos += ln
        for idx_s, end_l in zip(snap_dst_list, snap_ends_list):
            assert float((pool[idx_s] - pool_ref[idx_s]).abs().max().item()) == 0.0, f"C={C} snap end={end_l}"

        # CUDA Graph capture + replay for ragged prefill
        out_cg = torch.empty_like(out)
        _ = causal_dwconv1d_ragged_prefill_with_snapshots_triton(
            x, w, cu, tile_req, tile_t0,
            dst_pool=pool, snap_dst_idx=snap_dst, req_snap_start=req_ss, snap_local_ends=snap_ends,
            snap_aligned=True, residual=res, use_activation=silu, block_t=bt, out=out_cg, clamp_val=2.0,
        )
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            causal_dwconv1d_ragged_prefill_with_snapshots_triton(
                x, w, cu, tile_req, tile_t0,
                dst_pool=pool, snap_dst_idx=snap_dst, req_snap_start=req_ss, snap_local_ends=snap_ends,
                snap_aligned=True, residual=res, use_activation=silu, block_t=bt, out=out_cg, clamp_val=2.0,
            )
        g.replay()
        torch.cuda.synchronize()
        assert float((out_cg - out).abs().max().item()) == 0.0

    print("  PASS: R14-D0 PLE add3/gather & ShortConv 8-step unroll parity + CG verified.")


def test_r15_hc_nn_down_inject_and_zero_cutlass80() -> None:
    """Verify single-copy NN fused HC down+inject weight, parity (M=1..128), CG, grad fallback, and zero cutlass_80."""
    print("\n=== Test R15-M0 HC Single-Copy NN Down+Inject Layout ===")
    device = torch.device("cuda:0")
    torch.manual_seed(15)
    H, HC, LR = 5120, 4, 128
    hc = Qwen4_5HyperConnection(
        hidden_size=H,
        hc_count=HC,
        hc_lowrank=LR,
        prev_mix=True,
        is_head=False,
        dtype=torch.bfloat16,
        device=device,
    ).eval()

    with torch.inference_mode():
        w_t = hc._get_fused_down_inject_weight_t()
        w_v = hc._get_fused_down_inject_weight()
        assert tuple(w_t.shape) == (H * HC, 144) and w_t.is_contiguous()
        assert tuple(w_v.shape) == (144, H * HC)
        assert w_v.untyped_storage().data_ptr() == w_t.untyped_storage().data_ptr()

        ws = HCStaticWorkspace()
        for rows in (1, 16, 32, 48, 64, 96, 128):
            res = torch.randn((rows, HC * H), dtype=torch.bfloat16, device=device) * 0.3
            pl = torch.randn((rows, LR), dtype=torch.bfloat16, device=device) * 0.3
            mixed_fast, _, inj_fast, lat_fast = hc.mix_with_latent(
                res, prev_latent=pl, ws=ws
            )

            normed = grouped_gemma_rmsnorm_bias_triton(
                res, hc.hc_norm_weight, hc.hc_norm_bias, HC, hc.eps
            )
            raw_ref = torch.nn.functional.linear(normed.float(), w_v.float())
            gl_ref = torch.nn.functional.linear(
                pl.float(), hc.input_mix_weight_up.float()
            )
            mix_ref, lat_ref, inj_ref = hc_mix_reduce_with_down_inject_triton(
                gl_ref.bfloat16(),
                normed,
                raw_ref.bfloat16(),
                LR,
                HC,
                has_latent=True,
                has_inj=True,
            )
            for a, b, tag in (
                (mixed_fast, mix_ref, "mixed"),
                (inj_fast, inj_ref, "inj"),
                (lat_fast, lat_ref, "latent"),
            ):
                assert a is not None and b is not None
                cos = torch.nn.functional.cosine_similarity(
                    a.float().reshape(-1), b.float().reshape(-1), dim=0
                ).item()
                assert cos >= 0.9999, f"rows={rows} {tag} cos={cos}"

        # CUDA Graph capture + replay across SK (rows=48) and MM NN (rows=64, 96)
        for rows in (48, 64, 96):
            res_cg = torch.randn((rows, HC * H), dtype=torch.bfloat16, device=device) * 0.3
            pl_cg = torch.randn((rows, LR), dtype=torch.bfloat16, device=device) * 0.3
            ws_cg = HCStaticWorkspace()
            m_ref, _, i_ref, l_ref = hc.mix_with_latent(res_cg, prev_latent=pl_cg, ws=ws_cg)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                m_out, _, i_out, l_out = hc.mix_with_latent(
                    res_cg, prev_latent=pl_cg, ws=ws_cg
                )
            g.replay()
            torch.cuda.synchronize(device)
            assert torch.equal(m_out, m_ref)
            assert torch.equal(i_out, i_ref)
            assert torch.equal(l_out, l_ref)

        # Verify zero cutlass_80 kernels across rows=48, 64, 96, 128
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            for rows in (48, 64, 96, 128):
                res_p = torch.randn((rows, HC * H), dtype=torch.bfloat16, device=device) * 0.3
                pl_p = torch.randn((rows, LR), dtype=torch.bfloat16, device=device) * 0.3
                _ = hc.mix_with_latent(res_p, prev_latent=pl_p, ws=ws)
            torch.cuda.synchronize(device)
        c80 = [
            ev.name
            for ev in prof.events()
            if getattr(ev, "device_type", None) == torch.profiler.DeviceType.CUDA
            and "cutlass_80" in (getattr(ev, "name", "") or "").lower()
        ]
        assert not c80, f"Unexpected cutlass_80 kernels in HC NN path: {c80[:5]}"

    # Grad-enabled training fallback check
    with torch.enable_grad():
        hc_tr = Qwen4_5HyperConnection(
            hidden_size=H,
            hc_count=HC,
            hc_lowrank=LR,
            prev_mix=True,
            is_head=False,
            dtype=torch.bfloat16,
            device=device,
        ).train()
        normed_tr = (
            torch.randn((64, HC * H), dtype=torch.bfloat16, device=device) * 0.3
        ).requires_grad_(True)
        raw_tr = hc_tr._linear_from_t(
            normed_tr, hc_tr._get_fused_down_inject_weight_t()
        )
        raw_tr.sum().backward()
        assert (
            normed_tr.grad is not None
            and hc_tr.input_mix_weight_down.grad is not None
            and hc_tr.block_inject_weight.grad is not None
        )
        pl_tr = torch.randn((48, LR), dtype=torch.bfloat16, device=device) * 0.3
        res_tr = (
            torch.randn((48, HC * H), dtype=torch.bfloat16, device=device) * 0.3
        ).requires_grad_(True)
        _ = hc_tr.mix_with_latent(res_tr, prev_latent=pl_tr)
    print("  PASS: R15-M0 HC single-copy NN down+inject, CG, grad fallback, and zero cutlass_80.")


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
    test_r4_shortconv_opaque_verify_and_clip_value()
    test_r5_ple_u64_hc_split5120_and_strided_shortconv()
    test_r6_fast_launch_opaque_decode_and_ple_snap_aligned()
    test_r7_fused_hc_epilogues_branch5120_workspaces_and_sp_shortconv()
    test_r8_hc_splitk_and_dflash_branch5120()
    test_r9_l2_swapped_hc_and_norm5120()
    test_r11_hc_up_mix_sk_1d_and_sconv_decode()
    test_r14_ple_add3_gather_l2_and_sconv_t8_unroll()
    test_r15_hc_nn_down_inject_and_zero_cutlass80()
    run_benchmarks()
    print(f"\nALL TESTS & BENCHMARKS PASSED in {time.perf_counter() - t0:.2f}s!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

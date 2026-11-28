"""SM103 baseline/after microbenchmark & parity checks for P1 prefill ops."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any, Dict

import flashinfer
import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    fp8_gemm_nt,
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    requant_weight_ue8m0,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_fp8_quant_triton import (
    rmsnorm_fp8_quant_ue8m0,
)
from rtp_llm.models_py.modules.dsv4.fp8._compressor_vllm_triton import (
    run_fused_compress_kv_write,
    run_save_partial_states,
)
from rtp_llm.models_py.modules.dsv4.fp8._fused_qnorm_quant_triton import (
    fused_rmsnorm_fp8_ue8m0_quant,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_ops_triton import (
    combine_topk_swa_indices,
    combine_topk_swa_indices_cp,
)
from rtp_llm.models_py.modules.dsv4.test.dsv4_kernel_perf_utils import (
    ensure_triton_cc,
    iters_for_m,
    measure_kernel,
)

ensure_triton_cc()


def _alloc_hca_kv_cache(num_slots: int, device: torch.device) -> torch.Tensor:
    ts, sd = 576, 8
    block_bytes = num_slots * (ts + sd)
    flat = torch.zeros((2, block_bytes), dtype=torch.uint8, device=device)
    cache = flat.as_strided(
        size=(2, num_slots, ts),
        stride=(block_bytes, ts, 1),
    )
    cache._flat_backing = flat  # type: ignore[attr-defined]
    return cache


def _inspect_triton_regs(fn_jit) -> list[dict[str, Any]]:
    out = []

    def _walk(node, depth: int = 0) -> None:
        if depth > 4 or len(out) >= 8 or node is None:
            return
        if hasattr(node, "n_regs") or hasattr(node, "n_spills"):
            out.append(
                {
                    "n_regs": getattr(node, "n_regs", None),
                    "n_spills": getattr(node, "n_spills", None),
                    "shared": getattr(getattr(node, "metadata", None), "shared", None),
                }
            )
            return
        if isinstance(node, dict):
            for v in node.values():
                _walk(v, depth + 1)
        elif isinstance(node, (tuple, list)):
            for v in node:
                _walk(v, depth + 1)

    for attr in ("device_caches", "cache"):
        root = getattr(fn_jit, attr, None)
        if root is not None:
            _walk(root)
            if out:
                break
    return out


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class P1PrefillOpsPerfTest(unittest.TestCase):
    @torch.inference_mode()
    def test_benchmark_all_p1_ops(self) -> None:
        device = torch.device("cuda")
        trace_dir = os.path.join(
            os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR") or tempfile.gettempdir(),
            "p1_perf_traces",
        )
        report: Dict[str, Any] = {
            "device": torch.cuda.get_device_name(0),
            "hca_compressor": [],
            "swa_combine": [],
            "qnorm_quant": [],
            "triton_regs": {},
        }

        print(f"\n[P1 Prefill Ops Microbenchmark] device={report['device']}")

        # ------------------------------------------------------------------
        # 1. HCA ratio=128 compressor (M=1024..16384, raw & may_use_cache)
        # ------------------------------------------------------------------
        print("\n--- 1. HCA ratio=128 compressor (head_dim=512, overlap=False) ---")
        print(
            "  {:>6} | {:>10} {:>10} | {:>10} {:>10}".format(
                "M", "raw_kern", "raw_e2e", "cac_kern", "cac_e2e"
            )
        )
        head_dim = 512
        rope_dim = 64
        ratio = 128
        state_page = 256
        rms_w = torch.rand(head_dim, dtype=torch.bfloat16, device=device) + 0.5
        ape = torch.randn(ratio, head_dim, dtype=torch.float32, device=device) * 0.1
        cos_sin = torch.zeros(32768, rope_dim, dtype=torch.float32, device=device)
        cos_sin[:, : rope_dim // 2] = 1.0

        for m in [1024, 2048, 4096, 8192, 16384]:
            kv_raw = torch.randn(m, head_dim, dtype=torch.float32, device=device) * 0.1
            score_raw = (
                torch.randn(m, head_dim, dtype=torch.float32, device=device) * 0.1
            )
            num_blocks = (m + state_page - 1) // state_page + 2
            state_cache = (
                torch.randn(
                    num_blocks,
                    state_page,
                    2 * head_dim,
                    dtype=torch.float32,
                    device=device,
                )
                * 0.1
            )
            state_bt = torch.arange(
                num_blocks, dtype=torch.int32, device=device
            ).unsqueeze(0)
            token_to_req = torch.zeros(m, dtype=torch.int32, device=device)

            pos_raw = torch.arange(m, dtype=torch.int64, device=device)
            state_slots_raw = (pos_raw // state_page) * state_page + (
                pos_raw % state_page
            )
            is_b_raw = ((pos_raw + 1) % ratio) == 0
            active_raw = (
                torch.nonzero(is_b_raw).squeeze(-1).to(torch.int32).contiguous()
            )
            kv_block_size = max(int(active_raw.numel()), 1)
            kv_cache = _alloc_hca_kv_cache(kv_block_size, device)
            b_idx_raw = torch.cumsum(is_b_raw.to(torch.int64), dim=0) - 1
            kv_slots_raw = torch.where(
                is_b_raw,
                kv_block_size + b_idx_raw,
                torch.full_like(b_idx_raw, -1),
            )

            sp_cont = 64
            pos_cont = torch.arange(
                sp_cont, sp_cont + m, dtype=torch.int64, device=device
            )
            state_slots_cont = (pos_cont // state_page) * state_page + (
                pos_cont % state_page
            )
            is_b_cont = ((pos_cont + 1) % ratio) == 0
            active_cont = (
                torch.nonzero(is_b_cont).squeeze(-1).to(torch.int32).contiguous()
            )
            kv_block_size_cont = max(int(active_cont.numel()), 1)
            kv_cache_cont = _alloc_hca_kv_cache(kv_block_size_cont, device)
            b_idx_cont = torch.cumsum(is_b_cont.to(torch.int64), dim=0) - 1
            kv_slots_cont = torch.where(
                is_b_cont,
                kv_block_size_cont + b_idx_cont,
                torch.full_like(b_idx_cont, -1),
            )

            iters = max(iters_for_m(m), 20)
            raw_m = measure_kernel(
                lambda: run_fused_compress_kv_write(
                    None,
                    token_to_req,
                    pos_raw,
                    state_slots_raw,
                    None,
                    rms_w,
                    1e-6,
                    cos_sin,
                    kv_cache,
                    kv_slots_raw,
                    kv_raw,
                    score_raw,
                    ape,
                    0,
                    disable_raw_path=False,
                    head_dim=head_dim,
                    rope_head_dim=rope_dim,
                    compress_ratio=ratio,
                    overlap=False,
                    state_tokens_per_block=state_page,
                    may_use_cache=False,
                    active_token_indices=active_raw,
                ),
                label=f"hca_raw_M{m}",
                trace_dir=trace_dir,
                warmup=8,
                iters=iters,
                profile_iters=10,
            )
            cache_m = measure_kernel(
                lambda: run_fused_compress_kv_write(
                    state_cache,
                    token_to_req,
                    pos_cont,
                    state_slots_cont,
                    state_bt,
                    rms_w,
                    1e-6,
                    cos_sin,
                    kv_cache_cont,
                    kv_slots_cont,
                    kv_raw,
                    score_raw,
                    ape,
                    sp_cont,
                    disable_raw_path=False,
                    head_dim=head_dim,
                    rope_head_dim=rope_dim,
                    compress_ratio=ratio,
                    overlap=False,
                    state_tokens_per_block=state_page,
                    may_use_cache=True,
                    active_token_indices=active_cont,
                ),
                label=f"hca_cache_M{m}",
                trace_dir=trace_dir,
                warmup=8,
                iters=iters,
                profile_iters=10,
            )
            print(
                f"  {m:6d} | {raw_m.kernel_sum_us:10.2f} {raw_m.event_us:10.2f} "
                f"(span {raw_m.kernel_span_us:7.2f}) | "
                f"{cache_m.kernel_sum_us:10.2f} {cache_m.event_us:10.2f} "
                f"(span {cache_m.kernel_span_us:7.2f})"
            )
            report["hca_compressor"].append(
                {
                    "M": m,
                    "raw_kern_sum_us": raw_m.kernel_sum_us,
                    "raw_kern_us": raw_m.kernel_span_us,
                    "raw_e2e_us": raw_m.event_us,
                    "cache_kern_sum_us": cache_m.kernel_sum_us,
                    "cache_kern_us": cache_m.kernel_span_us,
                    "cache_e2e_us": cache_m.event_us,
                }
            )
            del kv_raw, score_raw, state_cache, kv_cache, kv_cache_cont

        from rtp_llm.models_py.modules.dsv4.fp8 import _compressor_vllm_triton as _cvt

        print(
            "\n--- 1b. HCA single-pass baseline vs split (M=1024..16384, raw & cache) ---"
        )
        print(
            "  {:>6} | {:>10} {:>10} | {:>10} {:>10}".format(
                "M", "orig_raw", "orig_e2e", "orig_cac", "orig_ce2e"
            )
        )
        for m in [1024, 2048, 4096, 8192, 16384]:
            kv_raw = torch.randn(m, head_dim, dtype=torch.float32, device=device) * 0.1
            score_raw = (
                torch.randn(m, head_dim, dtype=torch.float32, device=device) * 0.1
            )
            num_blocks = (m + state_page - 1) // state_page + 2
            state_cache = (
                torch.randn(
                    num_blocks,
                    state_page,
                    2 * head_dim,
                    dtype=torch.float32,
                    device=device,
                )
                * 0.1
            )
            state_bt = torch.arange(
                num_blocks, dtype=torch.int32, device=device
            ).unsqueeze(0)
            token_to_req = torch.zeros(m, dtype=torch.int32, device=device)
            sp_cont = 64
            pos_raw = torch.arange(m, dtype=torch.int64, device=device)
            state_slots_raw = (pos_raw // state_page) * state_page + (
                pos_raw % state_page
            )
            is_b_raw = ((pos_raw + 1) % ratio) == 0
            active_raw = (
                torch.nonzero(is_b_raw).squeeze(-1).to(torch.int32).contiguous()
            )
            kv_bs = max(int(active_raw.numel()), 1)
            kv_cache = _alloc_hca_kv_cache(kv_bs, device)
            b_idx_raw = torch.cumsum(is_b_raw.to(torch.int64), dim=0) - 1
            kv_slots_raw = torch.where(
                is_b_raw, kv_bs + b_idx_raw, torch.full_like(b_idx_raw, -1)
            )
            pos_cont = torch.arange(
                sp_cont, sp_cont + m, dtype=torch.int64, device=device
            )
            state_slots_cont = (pos_cont // state_page) * state_page + (
                pos_cont % state_page
            )
            is_b_cont = ((pos_cont + 1) % ratio) == 0
            active_cont = (
                torch.nonzero(is_b_cont).squeeze(-1).to(torch.int32).contiguous()
            )
            kv_bs_c = max(int(active_cont.numel()), 1)
            kv_cache_cont = _alloc_hca_kv_cache(kv_bs_c, device)
            b_idx_cont = torch.cumsum(is_b_cont.to(torch.int64), dim=0) - 1
            kv_slots_cont = torch.where(
                is_b_cont, kv_bs_c + b_idx_cont, torch.full_like(b_idx_cont, -1)
            )
            iters = max(iters_for_m(m), 20)
            orig_raw_m = measure_kernel(
                lambda: _cvt._fused_kv_compress_norm_rope_insert_sparse_attn[
                    (int(active_raw.shape[0]),)
                ](
                    kv_raw,
                    0,
                    0,
                    token_to_req,
                    pos_raw,
                    state_slots_raw,
                    pos_raw,
                    1,
                    state_page,
                    kv_raw,
                    int(kv_raw.stride(0)),
                    score_raw,
                    int(score_raw.stride(0)),
                    ape,
                    int(ape.stride(0)),
                    0,
                    m,
                    pos_raw,
                    pos_raw,
                    rms_w,
                    1e-6,
                    cos_sin,
                    int(cos_sin.stride(0)),
                    kv_cache,
                    kv_slots_raw,
                    kv_bs,
                    pos_raw,
                    active_raw,
                    int(active_raw.shape[0]),
                    HEAD_SIZE=head_dim,
                    TRITON_BLOCK_SIZE=512,
                    STATE_WIDTH=head_dim,
                    COMPRESS_RATIO=ratio,
                    OVERLAP=False,
                    ROPE_HEAD_DIM=rope_dim,
                    FP8_MAX=448.0,
                    QUANT_BLOCK=64,
                    TOKEN_STRIDE=576,
                    SCALE_DIM=8,
                    KV_BLOCK_STRIDE=int(kv_cache.stride(0)),
                    NUM_STATE_BLOCKS=0,
                    NUM_KV_BLOCKS=2,
                    BATCHED=False,
                    TRAP_INVALID_KV_ACCESS=True,
                    STATE_RING_ENTRIES=state_page,
                    MAY_USE_CACHE=False,
                    USE_UNPAD_RESTORE=False,
                    USE_ACTIVE_INDICES=True,
                    TILE_T=4,
                    ALIGN4=True,
                    num_warps=4,
                    num_stages=1,
                ),
                label=f"hca_orig_raw_M{m}",
                trace_dir=trace_dir,
                warmup=6,
                iters=iters,
                profile_iters=10,
            )
            orig_cac_m = measure_kernel(
                lambda: _cvt._fused_kv_compress_norm_rope_insert_sparse_attn[
                    (int(active_cont.shape[0]),)
                ](
                    state_cache,
                    int(state_cache.stride(0)),
                    int(state_cache.stride(1)),
                    token_to_req,
                    pos_cont,
                    state_slots_cont,
                    state_bt,
                    int(state_bt.stride(0)),
                    state_page,
                    kv_raw,
                    int(kv_raw.stride(0)),
                    score_raw,
                    int(score_raw.stride(0)),
                    ape,
                    int(ape.stride(0)),
                    sp_cont,
                    m,
                    pos_cont,
                    pos_cont,
                    rms_w,
                    1e-6,
                    cos_sin,
                    int(cos_sin.stride(0)),
                    kv_cache_cont,
                    kv_slots_cont,
                    kv_bs_c,
                    pos_cont,
                    active_cont,
                    int(active_cont.shape[0]),
                    HEAD_SIZE=head_dim,
                    TRITON_BLOCK_SIZE=512,
                    STATE_WIDTH=head_dim,
                    COMPRESS_RATIO=ratio,
                    OVERLAP=False,
                    ROPE_HEAD_DIM=rope_dim,
                    FP8_MAX=448.0,
                    QUANT_BLOCK=64,
                    TOKEN_STRIDE=576,
                    SCALE_DIM=8,
                    KV_BLOCK_STRIDE=int(kv_cache_cont.stride(0)),
                    NUM_STATE_BLOCKS=num_blocks,
                    NUM_KV_BLOCKS=2,
                    BATCHED=False,
                    TRAP_INVALID_KV_ACCESS=True,
                    STATE_RING_ENTRIES=state_page,
                    MAY_USE_CACHE=True,
                    USE_UNPAD_RESTORE=False,
                    USE_ACTIVE_INDICES=True,
                    TILE_T=4,
                    ALIGN4=True,
                    num_warps=4,
                    num_stages=1,
                ),
                label=f"hca_orig_cac_M{m}",
                trace_dir=trace_dir,
                warmup=6,
                iters=iters,
                profile_iters=10,
            )
            print(
                f"  {m:6d} | {orig_raw_m.kernel_sum_us:10.2f} {orig_raw_m.event_us:10.2f} | "
                f"{orig_cac_m.kernel_sum_us:10.2f} {orig_cac_m.event_us:10.2f}"
            )
            report.setdefault("hca_orig", []).append(
                {
                    "M": m,
                    "orig_raw_kern_sum_us": orig_raw_m.kernel_sum_us,
                    "orig_raw_e2e_us": orig_raw_m.event_us,
                    "orig_cac_kern_sum_us": orig_cac_m.kernel_sum_us,
                    "orig_cac_e2e_us": orig_cac_m.event_us,
                }
            )
            del kv_raw, score_raw, state_cache, kv_cache, kv_cache_cont

        for kname in [
            "_fused_kv_compress_norm_rope_insert_sparse_attn",
            "_hca_compress_split_stage1_kernel",
            "_hca_compress_split_finalize_kernel",
            "_hca_compress_stage1_kernel",
            "_hca_compress_finalize_kernel",
        ]:
            kobj = getattr(_cvt, kname, None)
            if kobj is not None:
                report["triton_regs"][kname] = _inspect_triton_regs(kobj)

        # ------------------------------------------------------------------
        # 2. _combine_topk_swa_indices_cp & combine_topk_swa_indices
        # ------------------------------------------------------------------
        print(
            "\n--- 2. combine_topk_swa_indices_cp & non-CP (kernel_sum_us / kernel_span_us) ---"
        )
        print(
            "  {:>6} | {:>10} {:>12} {:>10} {:>10}".format(
                "M", "cp_csa_b1", "cp_csa_var", "cp_hca", "noncp_csa"
            )
        )
        win = 512
        for m in [1024, 4096, 16384]:
            iters = max(iters_for_m(m), 20)
            topk_csa = 2048
            topk_idx_csa = torch.randint(
                0, max(1, m // 4), (m, topk_csa), dtype=torch.int32, device=device
            )
            gpos_b1 = torch.arange(m, dtype=torch.int32, device=device)
            cp_csa_b1_m = measure_kernel(
                lambda: combine_topk_swa_indices_cp(
                    topk_indices=topk_idx_csa,
                    global_positions=gpos_b1,
                    sp_int=0,
                    window_size=win,
                    compress_ratio=4,
                    topk=topk_csa,
                    M=m // 4 + m,
                    N=m // 4,
                    flash_mla_indices=True,
                ),
                label=f"cp_csa_b1_M{m}",
                trace_dir=trace_dir,
                warmup=8,
                iters=iters,
                profile_iters=10,
            )
            b = 4
            per_b = m // b
            req_ids = torch.arange(
                b, dtype=torch.int32, device=device
            ).repeat_interleave(per_b)
            prefixes = torch.full((b,), 64, dtype=torch.int32, device=device)
            cu_cmp = torch.arange(b, dtype=torch.int32, device=device) * (
                per_b // 4 + 64 + per_b
            )
            cu_swa = cu_cmp + (per_b // 4 + 16)
            cp_csa_var_m = measure_kernel(
                lambda: combine_topk_swa_indices_cp(
                    topk_indices=topk_idx_csa,
                    global_positions=gpos_b1 + 64,
                    sp_int=64,
                    window_size=win,
                    compress_ratio=4,
                    topk=topk_csa,
                    req_id_per_token=req_ids,
                    prefix_lengths=prefixes,
                    cu_cmp_offsets=cu_cmp,
                    cu_swa_offsets=cu_swa,
                    flash_mla_indices=True,
                ),
                label=f"cp_csa_var_M{m}",
                trace_dir=trace_dir,
                warmup=8,
                iters=iters,
                profile_iters=10,
            )
            n_hca = max(1, m // 128)
            dense_hca = (
                torch.arange(n_hca, dtype=torch.int32, device=device)
                .view(1, n_hca)
                .expand(m, n_hca)
            )
            cp_hca_m = measure_kernel(
                lambda: combine_topk_swa_indices_cp(
                    topk_indices=dense_hca,
                    global_positions=gpos_b1,
                    sp_int=0,
                    window_size=win,
                    compress_ratio=128,
                    topk=n_hca,
                    M=n_hca + m,
                    N=n_hca,
                    flash_mla_indices=True,
                    dense_arange_topk=True,
                ),
                label=f"cp_hca_M{m}",
                trace_dir=trace_dir,
                warmup=8,
                iters=iters,
                profile_iters=10,
            )
            qsl = torch.tensor([0, m], dtype=torch.int32, device=device)
            seq_lens = torch.tensor([m], dtype=torch.int32, device=device)
            gather_lens = torch.tensor([m], dtype=torch.int32, device=device)
            noncp_csa_m = measure_kernel(
                lambda: combine_topk_swa_indices(
                    topk_indices=topk_idx_csa,
                    query_start_loc=qsl,
                    seq_lens=seq_lens,
                    gather_lens=gather_lens,
                    window_size=win,
                    compress_ratio=4,
                    topk=topk_csa,
                    M=m // 4 + m,
                    N=m // 4,
                    flash_mla_indices=True,
                ),
                label=f"noncp_csa_M{m}",
                trace_dir=trace_dir,
                warmup=8,
                iters=iters,
                profile_iters=10,
            )
            print(
                f"  {m:6d} | {cp_csa_b1_m.kernel_sum_us:10.2f} "
                f"{cp_csa_var_m.kernel_sum_us:12.2f} "
                f"{cp_hca_m.kernel_sum_us:10.2f} {noncp_csa_m.kernel_sum_us:10.2f} "
                f"(span: {cp_csa_b1_m.kernel_span_us:.2f}/{cp_csa_var_m.kernel_span_us:.2f}/"
                f"{cp_hca_m.kernel_span_us:.2f}/{noncp_csa_m.kernel_span_us:.2f})"
            )
            report["swa_combine"].append(
                {
                    "M": m,
                    "cp_csa_b1_kern_sum_us": cp_csa_b1_m.kernel_sum_us,
                    "cp_csa_var_kern_sum_us": cp_csa_var_m.kernel_sum_us,
                    "cp_hca_kern_sum_us": cp_hca_m.kernel_sum_us,
                    "noncp_csa_kern_sum_us": noncp_csa_m.kernel_sum_us,
                    "cp_csa_b1_kern_us": cp_csa_b1_m.kernel_span_us,
                    "cp_csa_var_kern_us": cp_csa_var_m.kernel_span_us,
                    "cp_hca_kern_us": cp_hca_m.kernel_span_us,
                    "noncp_csa_kern_us": noncp_csa_m.kernel_span_us,
                    "cp_csa_b1_e2e_us": cp_csa_b1_m.event_us,
                    "noncp_csa_e2e_us": noncp_csa_m.event_us,
                }
            )

        from rtp_llm.models_py.modules.dsv4.fp8 import _swa_ops_triton as _swa_t

        for kname in [
            "_combine_topk_swa_indices_kernel",
            "_combine_topk_swa_indices_cp_kernel",
        ]:
            kobj = getattr(_swa_t, kname, None)
            if kobj is not None:
                report["triton_regs"][kname] = _inspect_triton_regs(kobj)

        # ------------------------------------------------------------------
        # 3. q_norm -> wq_b + indexer.wq_b fusion & attn_norm_quant (kernel_span_us)
        # ------------------------------------------------------------------
        if has_deep_gemm() and is_deep_gemm_e8m0_used():
            print(
                "\n--- 3. q_norm + DeepGEMM wq_b / attn_norm_quant (kernel_span_us) ---"
            )
            print(
                "  {:>6} | {:>10} {:>10} | {:>10} {:>10} | {:>10} {:>10}".format(
                    "M",
                    "1lin_old",
                    "1lin_new",
                    "2lin_old",
                    "2lin_new",
                    "an_old",
                    "an_new",
                )
            )
            d_qr = 1536
            n_attn = 192 * 16
            n_idx = 64 * 128
            w_qn = torch.rand(d_qr, dtype=torch.bfloat16, device=device) + 0.5

            def _make_w(n_out: int, k_in: int):
                wf = (
                    torch.randn(n_out, k_in, dtype=torch.bfloat16, device=device) * 0.03
                ).to(torch.float8_e4m3fn)
                ws = torch.ones(
                    n_out // 128, k_in // 128, dtype=torch.float32, device=device
                )
                return requant_weight_ue8m0(wf, ws)

            wa_q, wa_s = _make_w(n_attn, d_qr)
            wi_q, wi_s = _make_w(n_idx, d_qr)

            d_an = 4096
            w_an = torch.rand(d_an, dtype=torch.bfloat16, device=device) + 0.5

            for m in [2, 6, 1024, 4096, 16384]:
                iters = max(iters_for_m(m), 20)
                q_proj = (
                    torch.randn(m, d_qr, dtype=torch.bfloat16, device=device) * 0.4
                ).contiguous()
                out_a = torch.empty(m, n_attn, dtype=torch.bfloat16, device=device)
                out_i = torch.empty(m, n_idx, dtype=torch.bfloat16, device=device)

                def _one_lin_old():
                    qr = flashinfer.norm.rmsnorm(q_proj, w_qn, eps=1e-6)
                    qq, qs = sgl_per_token_group_quant_fp8(
                        qr,
                        group_size=128,
                        eps=1e-4,
                        column_major_scales=True,
                        scale_tma_aligned=True,
                        scale_ue8m0=True,
                    )
                    fp8_gemm_nt((qq, qs), (wa_q, wa_s), out_a, disable_ue8m0_cast=False)
                    return out_a

                def _one_lin_new():
                    _, (qq, qs) = fused_rmsnorm_fp8_ue8m0_quant(
                        q_proj, w_qn, eps=1e-6, emit_norm=False
                    )
                    fp8_gemm_nt((qq, qs), (wa_q, wa_s), out_a, disable_ue8m0_cast=False)
                    return out_a

                def _two_lin_old():
                    qr = flashinfer.norm.rmsnorm(q_proj, w_qn, eps=1e-6)
                    qq1, qs1 = sgl_per_token_group_quant_fp8(
                        qr,
                        group_size=128,
                        eps=1e-4,
                        column_major_scales=True,
                        scale_tma_aligned=True,
                        scale_ue8m0=True,
                    )
                    fp8_gemm_nt(
                        (qq1, qs1), (wa_q, wa_s), out_a, disable_ue8m0_cast=False
                    )
                    qq2, qs2 = sgl_per_token_group_quant_fp8(
                        qr,
                        group_size=128,
                        eps=1e-4,
                        column_major_scales=True,
                        scale_tma_aligned=True,
                        scale_ue8m0=True,
                    )
                    fp8_gemm_nt(
                        (qq2, qs2), (wi_q, wi_s), out_i, disable_ue8m0_cast=False
                    )
                    return out_a, out_i

                def _two_lin_new():
                    _, (qq, qs) = fused_rmsnorm_fp8_ue8m0_quant(
                        q_proj, w_qn, eps=1e-6, emit_norm=False
                    )
                    fp8_gemm_nt((qq, qs), (wa_q, wa_s), out_a, disable_ue8m0_cast=False)
                    fp8_gemm_nt((qq, qs), (wi_q, wi_s), out_i, disable_ue8m0_cast=False)
                    return out_a, out_i

                x_an = (
                    torch.randn(m, d_an, dtype=torch.bfloat16, device=device) * 0.4
                ).contiguous()

                def _an_old():
                    return rmsnorm_fp8_quant_ue8m0(
                        x_an,
                        w_an,
                        eps=1e-6,
                        group_size=128,
                        clamp_eps=1e-4,
                        out_norm=x_an,
                    )

                def _an_new():
                    return fused_rmsnorm_fp8_ue8m0_quant(
                        x_an,
                        w_an,
                        eps=1e-6,
                        group_size=128,
                        clamp_eps=1e-4,
                        emit_norm=True,
                        out_norm=x_an,
                    )

                u1_old = measure_kernel(
                    _one_lin_old,
                    label=f"1lin_old_M{m}",
                    trace_dir=trace_dir,
                    warmup=6,
                    iters=iters,
                    profile_iters=10,
                )
                u1_new = measure_kernel(
                    _one_lin_new,
                    label=f"1lin_new_M{m}",
                    trace_dir=trace_dir,
                    warmup=6,
                    iters=iters,
                    profile_iters=10,
                )
                u2_old = measure_kernel(
                    _two_lin_old,
                    label=f"2lin_old_M{m}",
                    trace_dir=trace_dir,
                    warmup=6,
                    iters=iters,
                    profile_iters=10,
                )
                u2_new = measure_kernel(
                    _two_lin_new,
                    label=f"2lin_new_M{m}",
                    trace_dir=trace_dir,
                    warmup=6,
                    iters=iters,
                    profile_iters=10,
                )
                an_old = measure_kernel(
                    _an_old,
                    label=f"an_old_M{m}",
                    trace_dir=trace_dir,
                    warmup=6,
                    iters=iters,
                    profile_iters=10,
                )
                an_new = measure_kernel(
                    _an_new,
                    label=f"an_new_M{m}",
                    trace_dir=trace_dir,
                    warmup=6,
                    iters=iters,
                    profile_iters=10,
                )
                print(
                    f"  {m:6d} | {u1_old.kernel_sum_us:10.2f} {u1_new.kernel_sum_us:10.2f} | "
                    f"{u2_old.kernel_sum_us:10.2f} {u2_new.kernel_sum_us:10.2f} | "
                    f"{an_old.kernel_sum_us:10.2f} {an_new.kernel_sum_us:10.2f}"
                )
                report["qnorm_quant"].append(
                    {
                        "M": m,
                        "one_lin_old_kern_us": u1_old.kernel_sum_us,
                        "one_lin_new_kern_us": u1_new.kernel_sum_us,
                        "two_lin_old_kern_us": u2_old.kernel_sum_us,
                        "two_lin_new_kern_us": u2_new.kernel_sum_us,
                        "attn_norm_old_kern_us": an_old.kernel_sum_us,
                        "attn_norm_new_kern_us": an_new.kernel_sum_us,
                    }
                )

        # ------------------------------------------------------------------
        # 4. _save_partial_states_kernel (HCA tail32 / CP8 & CSA sparse / full)
        # ------------------------------------------------------------------
        print("\n--- 4. save_partial_states (N=2227, active_indices=None) ---")
        print(
            "  {:>22} | {:>10} {:>10} {:>10}".format(
                "case", "kern_sum", "kern_span", "e2e_us"
            )
        )
        n_save = 2227
        state_page_save = 256
        report["save_partial_states"] = []
        save_cases = [
            ("hca_tail32", 512, 128, "tail32"),
            ("hca_cp8_32of256", 512, 128, "cp8"),
            ("hca_full", 512, 128, "full"),
            ("csa_sparse4", 1024, 4, "sparse4"),
            ("csa_tail32", 1024, 4, "tail32"),
            ("csa_cp8_32of256", 1024, 4, "cp8"),
            ("csa_full", 1024, 4, "full"),
        ]
        for cname, h_size, ratio_s, pattern in save_cases:
            kv_s = torch.randn(n_save, h_size, dtype=torch.float32, device=device) * 0.1
            sc_s = torch.randn(n_save, h_size, dtype=torch.float32, device=device) * 0.1
            ape_s = (
                torch.randn(ratio_s, h_size, dtype=torch.float32, device=device) * 0.1
            )
            pos_s = torch.arange(n_save, dtype=torch.int64, device=device)
            n_blks = (n_save + state_page_save - 1) // state_page_save + 2
            slots_s = torch.full((n_save,), -1, dtype=torch.int64, device=device)
            if pattern == "tail32":
                t_idx = torch.arange(
                    n_save - 32, n_save, dtype=torch.int64, device=device
                )
                slots_s[t_idx] = (t_idx // state_page_save) * state_page_save + (
                    t_idx % state_page_save
                )
            elif pattern == "cp8":
                mask_p = (pos_s % state_page_save) < 32
                slots_s[mask_p] = (
                    pos_s[mask_p] // state_page_save
                ) * state_page_save + (pos_s[mask_p] % state_page_save)
            elif pattern == "sparse4":
                mask_p = (pos_s % 32) < 4
                slots_s[mask_p] = (
                    pos_s[mask_p] // state_page_save
                ) * state_page_save + (pos_s[mask_p] % state_page_save)
            else:
                slots_s = (pos_s // state_page_save) * state_page_save + (
                    pos_s % state_page_save
                )

            st_cache = torch.zeros(
                (n_blks, state_page_save, 2 * h_size),
                dtype=torch.float32,
                device=device,
            )
            st_ref = torch.zeros_like(st_cache)
            active_idx = (
                torch.nonzero(slots_s >= 0).squeeze(-1).to(torch.int32).contiguous()
            )
            run_save_partial_states(
                kv_s,
                sc_s,
                ape_s,
                pos_s,
                st_ref,
                slots_s,
                ratio_s,
                active_indices=active_idx,
            )
            run_save_partial_states(
                kv_s,
                sc_s,
                ape_s,
                pos_s,
                st_cache,
                slots_s,
                ratio_s,
                active_indices=None,
            )
            torch.cuda.synchronize()
            self.assertTrue(
                torch.equal(st_cache, st_ref),
                f"save_partial_states mismatch for {cname}",
            )
            m_save = measure_kernel(
                lambda: run_save_partial_states(
                    kv_s,
                    sc_s,
                    ape_s,
                    pos_s,
                    st_cache,
                    slots_s,
                    ratio_s,
                    active_indices=None,
                ),
                label=f"save_{cname}",
                trace_dir=trace_dir,
                warmup=10,
                iters=100,
                profile_iters=20,
            )
            print(
                f"  {cname:>22} | {m_save.kernel_sum_us:10.2f} "
                f"{m_save.kernel_span_us:10.2f} {m_save.event_us:10.2f}"
            )
            report["save_partial_states"].append(
                {
                    "case": cname,
                    "N": n_save,
                    "head_size": h_size,
                    "active": int(active_idx.numel()),
                    "kern_sum_us": m_save.kernel_sum_us,
                    "kern_span_us": m_save.kernel_span_us,
                    "e2e_us": m_save.event_us,
                }
            )
            del kv_s, sc_s, st_cache, st_ref

        kobj_save = getattr(_cvt, "_save_partial_states_kernel", None)
        if kobj_save is not None:
            report["triton_regs"]["_save_partial_states_kernel"] = _inspect_triton_regs(
                kobj_save
            )

        print("\nTriton registers:", json.dumps(report["triton_regs"]))
        undeclared_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if undeclared_dir:
            os.makedirs(undeclared_dir, exist_ok=True)
            with open(
                os.path.join(undeclared_dir, "p1_prefill_ops_perf.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(report, f, indent=2)
        self.assertTrue(report["hca_compressor"])


if __name__ == "__main__":
    unittest.main()

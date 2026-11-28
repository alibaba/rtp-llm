"""SM103 baseline/after microbenchmark for W4-3 DSV4 prefill GPU kernels."""

from __future__ import annotations

import tempfile
import unittest

import torch

from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
from rtp_llm.models_py.modules.dsv4._fused_inv_rope_fp8_quant_triton import (
    fused_inv_rope_fp8_quant,
)
from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import (
    fused_rmsnorm_rope,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
    try_restore_dequantize_scatter_packed_k_cache_flat,
)
from rtp_llm.models_py.modules.dsv4.hc import (
    can_use_fused_mhc_pre_norm,
    fused_mhc_pre_norm,
    fused_mhc_pre_norm_quant,
)
from rtp_llm.models_py.modules.dsv4.hc.tilelang_impl import TileLangHCUnit
from rtp_llm.models_py.modules.dsv4.test.dsv4_kernel_perf_utils import (
    ensure_triton_cc,
    iters_for_m,
    measure_kernel,
)
from rtp_llm.models_py.triton_kernels.legacy_fp8_indexer_score import (
    legacy_fp8_mqa_logits,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops

ensure_triton_cc()


def _bench_pair(
    fn, *, label: str, trace_dir: str, m_for_iters: int, kernel_regex: str | None = None
):
    warmup = 10 if m_for_iters <= 12288 else 5
    iters = max(iters_for_m(m_for_iters), 20)
    return measure_kernel(
        fn,
        label=label,
        trace_dir=trace_dir,
        kernel_regex=kernel_regex,
        warmup=warmup,
        iters=iters,
        profile_enabled=True,
        profile_iters=5,
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class W43PrefillKernelPerfTest(unittest.TestCase):
    @torch.inference_mode()
    def test_w4_3_prefill_suite(self) -> None:
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(dev)
        print(
            f"\n[W4-3 Prefill Microbenchmark] device={props.name} "
            f"SM={props.major}.{props.minor} SMs={props.multi_processor_count} "
            f"torch={torch.__version__}",
            flush=True,
        )

        with tempfile.TemporaryDirectory() as trace_dir:
            self._bench_indexer_score(dev, trace_dir)
            self._bench_topk_prefill(dev, trace_dir)
            self._bench_mhc_pre_post(dev, trace_dir)
            self._bench_inv_rope_fp8_quant(dev, trace_dir)
            self._bench_rmsnorm_rope(dev, trace_dir)
            self._bench_swa_restore_dequant(dev, trace_dir)

    def _bench_indexer_score(self, dev: torch.device, trace_dir: str) -> None:
        print(
            "\n=== 1. _score_prefill_bq2 (legacy_fp8_mqa_logits, H=64, D=128) ===",
            flush=True,
        )
        print(
            f"  {'M':>6} {'N':>6} {'clean':>5} {'window':>7} | "
            f"{'e2e_us':>10} {'kern_us':>10} {'sum_us':>10} {'TFLOP/s':>8}",
            flush=True,
        )
        torch.manual_seed(101)
        configs = [
            (9190, 10240, False, "causal"),
            (9190, 10240, True, "causal"),
            (9190, 12288, False, "causal"),
            (9190, 10240, False, "full"),
            (7584, 7585, False, "causal"),
            (7584, 7585, True, "causal"),
            (5242, 5242, False, "causal"),
            (5242, 5242, True, "causal"),
        ]
        for m, n, clean, window in configs:
            q = torch.randn(m, 64, 128, device=dev).to(torch.float8_e4m3fn)
            k = torch.randn(n, 128, device=dev).to(torch.float8_e4m3fn)
            scales = torch.rand(n, device=dev, dtype=torch.float32) * 0.01 + 0.001
            weights = torch.randn(m, 64, device=dev, dtype=torch.float32)
            starts = torch.zeros(m, device=dev, dtype=torch.int32)
            if window == "causal":
                prefix = max(0, n - m)
                ends = torch.arange(
                    prefix + 1, prefix + m + 1, device=dev, dtype=torch.int32
                ).clamp_max(n)
            else:
                ends = torch.full((m,), n, device=dev, dtype=torch.int32)

            res = _bench_pair(
                lambda: legacy_fp8_mqa_logits(
                    q, (k, scales), weights, starts, ends, clean_logits=clean
                ),
                label=f"mqa_M{m}_N{n}_c{int(clean)}_{window}",
                trace_dir=trace_dir,
                m_for_iters=m,
                kernel_regex=r".*_score_prefill.*",
            )
            active_tokens = float((ends - starts).clamp_min(0).sum().item())
            tflops = (
                (2.0 * active_tokens * 64.0 * 128.0) / (res.kernel_sum_us * 1e-6) / 1e12
            )
            print(
                f"  {m:6d} {n:6d} {str(clean):>5} {window:>7} | "
                f"{res.event_us:10.2f} {res.kernel_span_us:10.2f} {res.kernel_sum_us:10.2f} {tflops:8.2f}",
                flush=True,
            )
            del q, k, scales, weights, starts, ends
            torch.cuda.empty_cache()

    def _bench_topk_prefill(self, dev: torch.device, trace_dir: str) -> None:
        print(
            "\n=== 2. topKPerRowPrefill<512> vs fast_topk_v2_variable<512> ===",
            flush=True,
        )
        print(
            f"  {'M':>6} {'N':>6} {'dist':>11} {'backend':>22} | "
            f"{'e2e_us':>10} {'kern_us':>10} {'BW_TB/s':>8}",
            flush=True,
        )
        torch.manual_seed(202)
        k_top = 512
        cases = [
            (9190, 10240, "real_mqa"),
            (9190, 10240, "relu_sparse"),
            (9190, 10240, "dense_randn"),
            (9190, 12288, "relu_sparse"),
            (7584, 7585, "relu_sparse"),
            (5242, 5242, "relu_sparse"),
        ]
        for m, n, dist in cases:
            starts = torch.zeros(m, device=dev, dtype=torch.int32)
            prefix = max(0, n - m)
            ends = torch.arange(
                prefix + 1, prefix + m + 1, device=dev, dtype=torch.int32
            ).clamp_max(n)
            # Keep every row len >= k_top so both fast_topk_v2 and prefill topk run heavy path,
            # EXCEPT real_mqa keeps true causal starts/ends (first few rows <512 like production).
            if dist != "real_mqa":
                ends = ends.clamp_min(min(n, 1024))
            lengths = (ends - starts).contiguous()

            if dist == "real_mqa":
                q = torch.randn(m, 64, 128, device=dev).to(torch.float8_e4m3fn)
                k = torch.randn(n, 128, device=dev).to(torch.float8_e4m3fn)
                scales = torch.rand(n, device=dev, dtype=torch.float32) * 0.01 + 0.001
                weights = torch.randn(m, 64, device=dev, dtype=torch.float32)
                logits = legacy_fp8_mqa_logits(
                    q, (k, scales), weights, starts, ends, clean_logits=True
                )
                del q, k, scales, weights
            elif dist == "relu_sparse":
                logits = torch.relu(
                    torch.randn(m, n, device=dev, dtype=torch.float32) - 0.35
                )
                if prefix == 0:
                    # Causal triangle -inf tail matching clean logits
                    col = torch.arange(n, device=dev).view(1, n)
                    logits = torch.where(col < ends.view(m, 1), logits, float("-inf"))
            else:
                logits = torch.randn(m, n, device=dev, dtype=torch.float32)

            out = torch.empty((m, k_top), dtype=torch.int32, device=dev)
            row_idx = torch.randperm(m, dtype=torch.int32, device=dev)
            active_bytes = float(lengths.sum().item()) * 4.0 + float(m * k_top * 4)

            runners = [
                (
                    "dsv4_prefill_radix=T",
                    lambda: rtp_llm_ops.dsv4_top_k_per_row_prefill(
                        logits,
                        starts,
                        ends,
                        out,
                        m,
                        logits.stride(0),
                        logits.stride(1),
                        k_top,
                        True,
                    ),
                    r".*topKPerRow.*",
                ),
                (
                    "dsv4_prefill_rowidx",
                    lambda: rtp_llm_ops.dsv4_top_k_per_row_prefill(
                        logits,
                        starts,
                        ends,
                        out,
                        m,
                        logits.stride(0),
                        logits.stride(1),
                        k_top,
                        True,
                        row_idx,
                    ),
                    r".*topKPerRow.*",
                ),
                (
                    "dsv4_prefill_radix=F",
                    lambda: rtp_llm_ops.dsv4_top_k_per_row_prefill(
                        logits,
                        starts,
                        ends,
                        out,
                        m,
                        logits.stride(0),
                        logits.stride(1),
                        k_top,
                        False,
                    ),
                    r".*topKPerRow.*",
                ),
            ]
            if hasattr(rtp_llm_ops, "fast_topk_v2_variable"):
                runners.append(
                    (
                        "fast_topk_v2_var<512>",
                        lambda: rtp_llm_ops.fast_topk_v2_variable(
                            logits, out, lengths, starts, k_top
                        ),
                        r".*topk_variable_kernel.*",
                    )
                )

            for bname, fn, regex in runners:
                res = _bench_pair(
                    fn,
                    label=f"topk_{dist}_M{m}_N{n}_{bname}",
                    trace_dir=trace_dir,
                    m_for_iters=m,
                    kernel_regex=regex,
                )
                bw = active_bytes / (res.kernel_sum_us * 1e-6) / 1e12
                print(
                    f"  {m:6d} {n:6d} {dist:>11} {bname:>22} | "
                    f"{res.event_us:10.2f} {res.kernel_sum_us:10.2f} {bw:8.2f}",
                    flush=True,
                )
            del logits, out, starts, ends, lengths, row_idx
            torch.cuda.empty_cache()

    def _bench_mhc_pre_post(self, dev: torch.device, trace_dir: str) -> None:
        print(
            "\n=== 3. mHC pre_norm (_mhc_pre_big_fuse_rmsnorm) & dsv4_mhc_post_fwd (D=4096, H=4) ===",
            flush=True,
        )
        print(
            f"  {'M':>6} {'op':>28} | {'e2e_us':>10} {'kern_us':>10} {'BW_TB/s':>8}",
            flush=True,
        )
        dim = 4096
        hc = 4
        has_post = hasattr(rtp_llm_ops, "dsv4_mhc_post_fwd")
        for m in (558, 4096, 5242, 7584, 9190):
            torch.manual_seed(303 + m)
            fn = torch.randn(24, hc * dim, device=dev, dtype=torch.float32) * 0.02
            base = torch.zeros(24, device=dev, dtype=torch.float32)
            scale = torch.ones(3, device=dev, dtype=torch.float32)
            unit = TileLangHCUnit(
                fn,
                base,
                scale,
                dim=dim,
                hc_mult=hc,
                hc_sinkhorn_iters=10,
                norm_eps=1e-6,
                hc_eps=1e-6,
            )
            norm = RMSNorm(
                (torch.rand(dim, device=dev, dtype=torch.bfloat16) + 0.5).contiguous(),
                eps=1e-6,
            )
            res_in = (
                torch.randn(m, hc, dim, device=dev, dtype=torch.bfloat16) * 0.25
            ).contiguous()

            if can_use_fused_mhc_pre_norm(
                res_in, unit, norm, quant_ue8m0=True, min_tokens=0
            ):
                for quant in (False, True):
                    res = _bench_pair(
                        (
                            (lambda: fused_mhc_pre_norm_quant(res_in, unit, norm))
                            if quant
                            else (lambda: fused_mhc_pre_norm(res_in, unit, norm))
                        ),
                        label=f"mhc_pre_M{m}_q{int(quant)}",
                        trace_dir=trace_dir,
                        m_for_iters=m,
                        kernel_regex=r".*_mhc_pre_big_fuse_rmsnorm_kernel.*",
                    )
                    read_b = m * hc * dim * 2 + dim * 2
                    write_b = (
                        m * dim * 2
                        + m * 20 * 4
                        + (m * dim + m * ((dim // 128 + 3) // 4) * 4 if quant else 0)
                    )
                    bw = (read_b + write_b) / (res.kernel_sum_us * 1e-6) / 1e12
                    tag = f"mhc_pre_norm(q={quant})"
                    print(
                        f"  {m:6d} {tag:>28} | {res.event_us:10.2f} {res.kernel_sum_us:10.2f} {bw:8.2f}",
                        flush=True,
                    )

            if has_post:
                x = torch.randn(m, dim, device=dev, dtype=torch.bfloat16) * 0.2
                post = torch.rand(m, hc, 1, device=dev, dtype=torch.float32) * 0.25
                comb = torch.softmax(
                    torch.randn(m, hc, hc, device=dev, dtype=torch.float32), dim=-1
                )
                res = _bench_pair(
                    lambda: rtp_llm_ops.dsv4_mhc_post_fwd(
                        x, res_in, post, comb, res_in
                    ),
                    label=f"mhc_post_M{m}",
                    trace_dir=trace_dir,
                    m_for_iters=m,
                    kernel_regex=r".*dsv4_mhc_post.*",
                )
                dram_b = (2 * hc + 1) * m * dim * 2 + m * 20 * 4
                bw = dram_b / (res.kernel_sum_us * 1e-6) / 1e12
                print(
                    f"  {m:6d} {'dsv4_mhc_post_fwd':>28} | {res.event_us:10.2f} {res.kernel_sum_us:10.2f} {bw:8.2f}",
                    flush=True,
                )
                del x, post, comb
            del fn, base, scale, unit, norm, res_in
            torch.cuda.empty_cache()

    def _bench_inv_rope_fp8_quant(self, dev: torch.device, trace_dir: str) -> None:
        print(
            "\n=== 4a. dsv4_fused_inv_rope_fp8_quant (impl=cuda, H=64, D=512, G=8) ===",
            flush=True,
        )
        print(
            f"  {'M':>6} {'pos_mode':>12} | {'e2e_us':>10} {'kern_us':>10} {'BW_TB/s':>8}",
            flush=True,
        )
        n_heads, head_dim, rope_dim = 64, 512, 64
        nope_dim = head_dim - rope_dim
        n_groups, heads_per_group = 8, 8
        for m in (558, 4096, 5242, 7584, 9190):
            o = (
                torch.randn(m, n_heads, head_dim, dtype=torch.bfloat16, device=dev)
                * 0.3
            )
            tma_m = ((m + 3) // 4) * 4
            fp8_buf = torch.empty(
                (n_groups, m, heads_per_group * head_dim),
                dtype=torch.float8_e4m3fn,
                device=dev,
            )
            scale_buf = torch.empty(
                (n_groups * heads_per_group * tma_m,), dtype=torch.int32, device=dev
            ).as_strided(
                (n_groups, m, heads_per_group), (heads_per_group * tma_m, 1, tma_m)
            )
            for use_pos in (False, True):
                if use_pos:
                    ang = torch.rand(16384, rope_dim // 2, device=dev) * 6.28
                    freqs = (
                        torch.polar(torch.ones_like(ang), ang)
                        .to(torch.complex64)
                        .contiguous()
                    )
                    pos = torch.randint(0, 16384, (m,), dtype=torch.int32, device=dev)
                else:
                    ang = torch.rand(m, rope_dim // 2, device=dev) * 6.28
                    freqs = (
                        torch.polar(torch.ones_like(ang), ang)
                        .to(torch.complex64)
                        .contiguous()
                    )
                    pos = None
                res = _bench_pair(
                    lambda: fused_inv_rope_fp8_quant(
                        o,
                        freqs,
                        n_groups=n_groups,
                        heads_per_group=heads_per_group,
                        nope_dim=nope_dim,
                        rope_head_dim=rope_dim,
                        fp8_buf=fp8_buf,
                        scale_buf=scale_buf,
                        position_ids=pos,
                        impl="cuda",
                    ),
                    label=f"inv_rope_M{m}_pos{int(use_pos)}",
                    trace_dir=trace_dir,
                    m_for_iters=m,
                    kernel_regex=r".*dsv4_fused_inv_rope_fp8_quant.*",
                )
                total_b = (
                    m * n_heads * head_dim * 3
                    + m * (rope_dim // 2) * 8
                    + tma_m * n_heads * 4
                    + (m * 4 if use_pos else 0)
                )
                bw = total_b / (res.kernel_sum_us * 1e-6) / 1e12
                pm = "pos_ids" if use_pos else "direct"
                print(
                    f"  {m:6d} {pm:>12} | {res.event_us:10.2f} {res.kernel_sum_us:10.2f} {bw:8.2f}",
                    flush=True,
                )
                del freqs, pos
            del o, fp8_buf, scale_buf
            torch.cuda.empty_cache()

    def _bench_rmsnorm_rope(self, dev: torch.device, trace_dir: str) -> None:
        print(
            "\n=== 4b. dsv4_fused_rmsnorm_rope (CUDA, D=512 grouped / D=128) ===",
            flush=True,
        )
        print(
            f"  {'M':>6} {'shape':>16} {'w':>2} {'pos':>4} | {'e2e_us':>10} {'kern_us':>10} {'BW_TB/s':>8}",
            flush=True,
        )
        for m in (558, 4096, 5242, 7584, 9190):
            for shape_tag, h, d, has_w in (
                ("Q(1,M,64,512)", 64, 512, False),
                ("Q(1,M,64,512)w", 64, 512, True),
                ("Q(1,M,64,128)", 64, 128, False),
                ("KV(1,M,1,512)w", 1, 512, True),
            ):
                x = torch.randn(1, m, h, d, dtype=torch.bfloat16, device=dev) * 0.3
                w = (
                    (torch.rand(d, dtype=torch.bfloat16, device=dev) + 0.5)
                    if has_w
                    else None
                )
                for use_pos in (False, True):
                    if use_pos:
                        ang = torch.rand(16384, 32, device=dev) * 6.28
                        freqs = (
                            torch.polar(torch.ones_like(ang), ang)
                            .to(torch.complex64)
                            .contiguous()
                        )
                        pos = torch.randint(
                            0, 16384, (m,), dtype=torch.int32, device=dev
                        )
                    else:
                        ang = torch.rand(m, 32, device=dev) * 6.28
                        freqs = (
                            torch.polar(torch.ones_like(ang), ang)
                            .to(torch.complex64)
                            .contiguous()
                        )
                        pos = None
                    res = _bench_pair(
                        lambda: fused_rmsnorm_rope(
                            x, w, freqs, 64, inplace=True, position_ids=pos
                        ),
                        label=f"rmsnorm_rope_{shape_tag}_M{m}_pos{int(use_pos)}",
                        trace_dir=trace_dir,
                        m_for_iters=m,
                        kernel_regex=r".*dsv4_fused_rmsnorm_rope.*",
                    )
                    dram_b = (
                        2 * m * h * d * 2
                        + (d * 2 if has_w else 0)
                        + m * 64 * 4
                        + (m * 4 if use_pos else 0)
                    )
                    bw = dram_b / (res.kernel_sum_us * 1e-6) / 1e12
                    print(
                        f"  {m:6d} {shape_tag:>16} {int(has_w):>2d} {int(use_pos):>4d} | "
                        f"{res.event_us:10.2f} {res.kernel_sum_us:10.2f} {bw:8.2f}",
                        flush=True,
                    )
                    del freqs, pos
                del x, w
            torch.cuda.empty_cache()

    def _bench_swa_restore_dequant(self, dev: torch.device, trace_dir: str) -> None:
        print(
            "\n=== 5. _restore_dequantize_scatter_packed_k_cache_flat_kernel ===",
            flush=True,
        )
        print(
            f"  {'T':>6} {'B':>3} | {'e2e_us':>10} {'kern_us':>10} {'BW_TB/s':>8}",
            flush=True,
        )
        head_dim = 512
        packed_stride = 584
        for t in (5242, 7584, 9190, 16384):
            b = 4
            gathered = torch.randint(
                0, 255, (t, packed_stride), dtype=torch.uint8, device=dev
            )
            restore_idx = torch.randperm(t, dtype=torch.int64, device=dev)
            per_req = t // b
            seq_vals = [per_req] * (b - 1) + [t - per_req * (b - 1)]
            seq_l = torch.tensor(seq_vals, dtype=torch.int32, device=dev)
            cu_dst = torch.tensor(
                [i * per_req for i in range(b)], dtype=torch.int32, device=dev
            )
            ws_out = torch.empty((t, head_dim), dtype=torch.bfloat16, device=dev)

            res = _bench_pair(
                lambda: try_restore_dequantize_scatter_packed_k_cache_flat(
                    ws_out,
                    gathered,
                    restore_idx,
                    seq_l,
                    offset=0,
                    seq_lens_total=t,
                    cu_dst_offsets=cu_dst,
                ),
                label=f"swa_restore_T{t}_B{b}",
                trace_dir=trace_dir,
                m_for_iters=t,
                kernel_regex=r".*_restore_dequantize_scatter_packed_k_cache_flat_kernel.*",
            )
            dram_b = t * (packed_stride + 8 + head_dim * 2)
            bw = dram_b / (res.kernel_sum_us * 1e-6) / 1e12
            print(
                f"  {t:6d} {b:3d} | {res.event_us:10.2f} {res.kernel_sum_us:10.2f} {bw:8.2f}",
                flush=True,
            )
            del gathered, restore_idx, seq_l, cu_dst, ws_out
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()

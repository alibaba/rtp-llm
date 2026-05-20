"""Performance benchmark for the vLLM-parity DSV4 Q/KV fused kernels.

Sweeps ``N_tok`` over both the decode regime (per-request batched,
small) and the prefill regime (long sequence) and measures the two
Triton kernels for **both V4 variants**:

  * V4-Flash: ``n_heads=64, head_dim=512, q_lora_rank=1024, rope=64``
  * V4-Pro:   ``n_heads=128, head_dim=512, q_lora_rank=1536, rope=64``

Two test methods:

  * ``test_perf_report`` — production-config sweep (the shape-keyed
    dispatch table in ``_fused_qkv_rmsnorm_rope_triton._LAUNCH_CONFIGS``
    picks GROUP_HEADS / num_warps / num_stages per shape). Default
    gate. Optionally enforces decode latency + prefill bandwidth via
    ``DSV4_QKV_RMSNORM_ROPE_PERF_ASSERT=1`` (prefill threshold is
    70 % of device HBM peak).

  * ``test_perf_tune`` — gated on ``DSV4_QKV_RMSNORM_ROPE_TUNE=1``.
    Sweeps ``(GROUP_HEADS, num_warps, num_stages)`` candidates per
    ``(variant, bucket)`` and prints a markdown table identifying the
    best config. Used offline to populate ``_LAUNCH_CONFIGS``.

Acceptance gate env vars (all optional):
  * ``DSV4_QKV_RMSNORM_ROPE_PERF_ASSERT=1`` — turn on asserts.
  * ``DSV4_QKV_RMSNORM_ROPE_DECODE_BUDGET_US`` (default 5.0) —
    combined Q/KV-stage kernel_span for N_tok <= 8.  Each launch
    floor is ~2 µs (CUDA dispatch latency on cuda13 / B300), so two
    back-to-back launches sit at ~4 µs minimum on the smallest
    shapes.  The default leaves ~1 µs noise headroom.
  * ``DSV4_QKV_RMSNORM_ROPE_PREFILL_MIN_FRAC`` (default 0.70) —
    fraction of HBM peak required at N_tok >= 4096 (best value across
    the prefill sweep).  Achievable on B300: V4-Flash hits ~84%,
    V4-Pro ~85% with the tuned dispatch table — both well above 70%.
"""

from __future__ import annotations

import os
import time
import unittest
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from dsv4_kernel_perf_utils import (
    device_payload,
    ensure_triton_cc,
    env_flag,
    git_commit,
    iters_for_m,
    measure_kernel,
    measurement_payload,
    parse_int_list,
    report_path_from_env,
    trace_dir_from_report,
    write_json_report,
)

from rtp_llm.models_py.modules.dsv4._fused_qkv_rmsnorm_rope_triton import (
    fused_q_kv_rmsnorm,
    fused_q_perhead_norm_qkv_rope,
)


ensure_triton_cc()


EPS = 1e-6


class V4Variant:
    __slots__ = ("name", "n_heads", "head_dim", "q_lora_rank", "rope_dim")

    def __init__(self, name, n_heads, head_dim, q_lora_rank, rope_dim):
        self.name = name
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.rope_dim = rope_dim

    @property
    def total_dim(self):
        return self.q_lora_rank + self.head_dim


VARIANTS: Tuple[V4Variant, ...] = (
    V4Variant("v4_flash", n_heads=64, head_dim=512, q_lora_rank=1024, rope_dim=64),
    V4Variant("v4_pro", n_heads=128, head_dim=512, q_lora_rank=1536, rope_dim=64),
)


# Prime + power-of-2 sweep — covers decode (low end) and prefill (high end).
DEFAULT_N_SWEEP = [
    1, 2, 4, 8, 16, 32, 64,
    128, 256, 512, 1024, 2048, 4096,
    8192, 16384, 32768, 65536,
]

# Configs explored in tuning mode. G covers {1,2,4,8} (KV always uses 1
# internally); both 64 and 128 head counts divide cleanly by all of these.
TUNE_CONFIGS: Tuple[Tuple[int, int, int], ...] = tuple(
    (g, w, s)
    for g in (1, 2, 4, 8)
    for w in (1, 2, 4, 8)
    for s in (2, 3, 4)
)

# Representative N per bucket for the tuning sweep — a single sample per
# bucket keeps total wall-clock to ~80s while still measuring the regime
# the bucket targets.
TUNE_BUCKETS: Tuple[Tuple[str, int], ...] = (
    ("small", 1),
    ("small", 8),
    ("small", 32),
    ("mid", 128),
    ("mid", 512),
    ("mid", 1024),
    ("large", 4096),
    ("large", 16384),
    ("large", 65536),
)


def _make_inputs(variant: V4Variant, n_tok: int, device: torch.device):
    torch.manual_seed(2026 + n_tok + variant.n_heads)
    qkv_a = (
        torch.randn(n_tok, variant.total_dim, dtype=torch.float32, device=device) * 0.3
    ).to(torch.bfloat16)
    q = (
        torch.randn(
            n_tok, variant.n_heads, variant.head_dim, dtype=torch.float32, device=device
        )
        * 0.3
    ).to(torch.bfloat16)
    q_norm = torch.rand(variant.q_lora_rank, dtype=torch.bfloat16, device=device) + 0.5
    kv_norm = torch.rand(variant.head_dim, dtype=torch.bfloat16, device=device) + 0.5
    angles = (
        torch.rand(n_tok, variant.rope_dim // 2, device=device) - 0.5
    ) * 2 * 3.14159
    freqs = torch.complex(torch.cos(angles), torch.sin(angles)).contiguous()
    return qkv_a, q, q_norm, kv_norm, freqs


def _rmsnorm_traffic_bytes(variant: V4Variant, n_tok: int) -> int:
    # Read qkv_a (q_lora + head_dim bf16), write qr + kv bf16, broadcast norms.
    bf16 = 2
    return (
        n_tok * variant.total_dim * 2 * bf16  # read + write of total
        + (variant.q_lora_rank + variant.head_dim) * bf16  # norm vectors
    )


def _rope_traffic_bytes(variant: V4Variant, n_tok: int) -> int:
    bf16 = 2
    q_bytes = n_tok * variant.n_heads * variant.head_dim * 2 * bf16  # in-place R+W
    kv_bytes = n_tok * variant.head_dim * 2 * bf16
    freq_bytes = n_tok * (variant.rope_dim // 2) * 8  # complex64 read once
    return q_bytes + kv_bytes + freq_bytes


def _measure_run_rmsnorm(
    variant: V4Variant,
    n_tok: int,
    qkv_a: torch.Tensor,
    q_norm: torch.Tensor,
    kv_norm: torch.Tensor,
    qr_scratch: torch.Tensor,
    kv_out_norm: torch.Tensor,
    *,
    trace_dir: str,
    profile_enabled: bool,
    warmup: int,
    iters: int,
):
    def run():
        qr, kv = fused_q_kv_rmsnorm(
            qkv_a,
            q_norm,
            kv_norm,
            q_size=variant.q_lora_rank,
            kv_offset=variant.q_lora_rank,
            eps=EPS,
        )
        qr_scratch.copy_(qr)
        kv_out_norm.copy_(kv)

    return measure_kernel(
        run,
        label=f"fused_q_kv_rmsnorm_{variant.name}_N{n_tok}",
        trace_dir=trace_dir,
        kernel_regex=r".*fused_q_kv_rmsnorm.*",
        warmup=warmup,
        iters=iters,
        profile_enabled=profile_enabled,
    )


def _measure_run_rope(
    variant: V4Variant,
    n_tok: int,
    q: torch.Tensor,
    kv_out_norm: torch.Tensor,
    freqs: torch.Tensor,
    kv_out_rope: torch.Tensor,
    *,
    launch_override: Optional[Tuple[int, int, int]],
    trace_dir: str,
    profile_enabled: bool,
    warmup: int,
    iters: int,
    label_suffix: str = "",
):
    def run():
        fused_q_perhead_norm_qkv_rope(
            q,
            kv_out_norm,
            freqs,
            variant.rope_dim,
            eps=EPS,
            kv_out=kv_out_rope,
            _launch_override=launch_override,
        )

    label = f"fused_q_perhead_norm_qkv_rope_{variant.name}_N{n_tok}{label_suffix}"
    return measure_kernel(
        run,
        label=label,
        trace_dir=trace_dir,
        kernel_regex=r".*fused_q_perhead_norm_qkv_rope.*",
        warmup=warmup,
        iters=iters,
        profile_enabled=profile_enabled,
    )


def _bench_variant_n(
    variant: V4Variant,
    n_tok: int,
    *,
    trace_dir: str,
    profile_enabled: bool,
) -> List[dict]:
    device = torch.device("cuda")
    qkv_a, q, q_norm, kv_norm, freqs = _make_inputs(variant, n_tok, device)

    kv_out_norm = torch.empty(n_tok, variant.head_dim, dtype=torch.bfloat16, device=device)
    kv_out_rope = torch.empty(n_tok, variant.head_dim, dtype=torch.bfloat16, device=device)
    qr_scratch = torch.empty(
        n_tok, variant.q_lora_rank, dtype=torch.bfloat16, device=device
    )

    warmup = 30 if n_tok <= 4096 else 10
    iters = iters_for_m(n_tok)

    m_rmsnorm = _measure_run_rmsnorm(
        variant,
        n_tok,
        qkv_a,
        q_norm,
        kv_norm,
        qr_scratch,
        kv_out_norm,
        trace_dir=trace_dir,
        profile_enabled=profile_enabled,
        warmup=warmup,
        iters=iters,
    )
    # Refresh kv_out_norm so the rope bench sees a valid post-rmsnorm buffer.
    qr, kv = fused_q_kv_rmsnorm(
        qkv_a,
        q_norm,
        kv_norm,
        q_size=variant.q_lora_rank,
        kv_offset=variant.q_lora_rank,
        eps=EPS,
    )
    kv_out_norm.copy_(kv)
    torch.cuda.synchronize()

    m_rope = _measure_run_rope(
        variant,
        n_tok,
        q,
        kv_out_norm,
        freqs,
        kv_out_rope,
        launch_override=None,
        trace_dir=trace_dir,
        profile_enabled=profile_enabled,
        warmup=warmup,
        iters=iters,
    )

    rmsnorm_bytes = _rmsnorm_traffic_bytes(variant, n_tok)
    rope_bytes = _rope_traffic_bytes(variant, n_tok)
    rows: List[dict] = []
    rows.append(
        {
            "variant": variant.name,
            "N_tok": n_tok,
            "kernel": "fused_q_kv_rmsnorm",
            "traffic_bytes": rmsnorm_bytes,
            "effective_gbps": rmsnorm_bytes / m_rmsnorm.kernel_span_us / 1000.0,
            **measurement_payload(m_rmsnorm),
        }
    )
    rows.append(
        {
            "variant": variant.name,
            "N_tok": n_tok,
            "kernel": "fused_q_perhead_norm_qkv_rope",
            "traffic_bytes": rope_bytes,
            "effective_gbps": rope_bytes / m_rope.kernel_span_us / 1000.0,
            **measurement_payload(m_rope),
        }
    )
    combined_us = m_rmsnorm.kernel_span_us + m_rope.kernel_span_us
    rows.append(
        {
            "variant": variant.name,
            "N_tok": n_tok,
            "kernel": "combined",
            "traffic_bytes": rmsnorm_bytes + rope_bytes,
            "effective_gbps": (rmsnorm_bytes + rope_bytes) / combined_us / 1000.0,
            "event_us": combined_us,
            "kernel_span_us": combined_us,
            "kernel_sum_us": m_rmsnorm.kernel_sum_us + m_rope.kernel_sum_us,
            "kernel_count": m_rmsnorm.kernel_count + m_rope.kernel_count,
            "idle_gap_us": 0.0,
            "trace_path": "",
            "measure_method": "summed_kernel_span_us",
        }
    )
    return rows


def _print_summary(rows: List[dict]) -> None:
    print("\n[fused_qkv_rmsnorm_rope]")
    print(
        "  {:>10} {:>8} {:>34} {:>14} {:>14} {:>10}".format(
            "variant", "N_tok", "kernel", "kernel_span_us", "kernel_sum_us", "GB/s"
        )
    )
    for row in rows:
        print(
            "  {:>10s} {:8d} {:>34s} {:14.3f} {:14.3f} {:10.1f}".format(
                row["variant"],
                row["N_tok"],
                row["kernel"],
                row["kernel_span_us"],
                row["kernel_sum_us"],
                row["effective_gbps"],
            )
        )


def _bench_rope_with_override(
    variant: V4Variant,
    n_tok: int,
    override: Tuple[int, int, int],
    *,
    trace_dir: str,
    warmup: int,
    iters: int,
) -> float:
    """Return kernel_span_us for one (G, warps, stages) candidate."""
    device = torch.device("cuda")
    _, q, _, _, freqs = _make_inputs(variant, n_tok, device)
    kv_out_norm = (
        torch.randn(n_tok, variant.head_dim, dtype=torch.float32, device=device) * 0.3
    ).to(torch.bfloat16)
    kv_out_rope = torch.empty(n_tok, variant.head_dim, dtype=torch.bfloat16, device=device)
    m = _measure_run_rope(
        variant,
        n_tok,
        q,
        kv_out_norm,
        freqs,
        kv_out_rope,
        launch_override=override,
        trace_dir=trace_dir,
        profile_enabled=False,  # cuda-event timing is enough for tuning
        warmup=warmup,
        iters=iters,
        label_suffix=f"_G{override[0]}w{override[1]}s{override[2]}",
    )
    return m.event_us


def _tune_variant(
    variant: V4Variant,
    *,
    trace_dir: str,
    bench_n: Sequence[Tuple[str, int]] = TUNE_BUCKETS,
) -> List[dict]:
    """Sweep TUNE_CONFIGS at each representative bucket sample, return rows."""
    rows: List[dict] = []
    best_per_bucket: Dict[Tuple[str, int], Tuple[Tuple[int, int, int], float]] = {}
    for bucket, n_tok in bench_n:
        warmup = 30 if n_tok <= 4096 else 10
        iters = iters_for_m(n_tok)
        best_cfg = None
        best_us = float("inf")
        for cfg in TUNE_CONFIGS:
            try:
                us = _bench_rope_with_override(
                    variant, n_tok, cfg, trace_dir=trace_dir, warmup=warmup, iters=iters,
                )
            except Exception as exc:  # noqa: BLE001 — sweep keeps going on a bad config
                rows.append(
                    {
                        "variant": variant.name,
                        "bucket": bucket,
                        "N_tok": n_tok,
                        "group_heads": cfg[0],
                        "num_warps": cfg[1],
                        "num_stages": cfg[2],
                        "event_us": float("nan"),
                        "error": str(exc),
                    }
                )
                continue
            rows.append(
                {
                    "variant": variant.name,
                    "bucket": bucket,
                    "N_tok": n_tok,
                    "group_heads": cfg[0],
                    "num_warps": cfg[1],
                    "num_stages": cfg[2],
                    "event_us": us,
                }
            )
            if us < best_us:
                best_us = us
                best_cfg = cfg
        if best_cfg is not None:
            key = (bucket, n_tok)
            best_per_bucket[key] = (best_cfg, best_us)
    _print_tune_table(variant, rows, best_per_bucket)
    return rows


def _print_tune_table(
    variant: V4Variant,
    rows: List[dict],
    best: Dict[Tuple[str, int], Tuple[Tuple[int, int, int], float]],
) -> None:
    print(f"\n## Tuning sweep — {variant.name} (n_heads={variant.n_heads}, D={variant.head_dim})")
    print(
        "\n| bucket | N_tok | best (G, warps, stages) | best µs | next-best µs | gain vs G=1,w=4,s=3 |"
    )
    print("|---|---|---|---|---|---|")
    for (bucket, n_tok), (cfg, us) in best.items():
        # Pull baseline + next-best from rows
        bucket_rows = [
            r
            for r in rows
            if r["bucket"] == bucket
            and r["N_tok"] == n_tok
            and r.get("event_us") == r.get("event_us")  # NaN filter
        ]
        bucket_rows.sort(key=lambda r: r["event_us"])
        baseline = next(
            (
                r
                for r in bucket_rows
                if r["group_heads"] == 1 and r["num_warps"] == 4 and r["num_stages"] == 3
            ),
            None,
        )
        next_best = bucket_rows[1]["event_us"] if len(bucket_rows) > 1 else float("nan")
        gain = baseline["event_us"] / us if baseline else float("nan")
        print(
            f"| {bucket} | {n_tok} | {cfg} | {us:.3f} | {next_best:.3f} | {gain:.2f}x |"
        )


def _bench_all_variants(
    n_list: List[int], trace_dir: str, profile_enabled: bool
) -> List[dict]:
    rows: List[dict] = []
    for variant in VARIANTS:
        for n_tok in n_list:
            rows.extend(
                _bench_variant_n(
                    variant,
                    n_tok,
                    trace_dir=trace_dir,
                    profile_enabled=profile_enabled,
                )
            )
            torch.cuda.empty_cache()
    return rows


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FusedQKVRmsnormRopePerfTest(unittest.TestCase):
    def test_perf_report(self):
        started = time.time()
        n_list = parse_int_list("DSV4_QKV_RMSNORM_ROPE_N_LIST", DEFAULT_N_SWEEP)
        report_path = report_path_from_env(
            "DSV4_QKV_RMSNORM_ROPE_JSON",
            "dsv4_fused_qkv_rmsnorm_rope_perf.json",
        )
        trace_dir = trace_dir_from_report(
            report_path, "DSV4_QKV_RMSNORM_ROPE_TRACE_DIR"
        )
        profile_enabled = env_flag("DSV4_QKV_RMSNORM_ROPE_PROFILE", True)

        rows = _bench_all_variants(n_list, trace_dir, profile_enabled)

        _print_summary(rows)
        dev = device_payload()
        payload = {
            "elapsed_sec": time.time() - started,
            "git_commit": git_commit(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
            ),
            "device": dev,
            "n_list": n_list,
            "profile_enabled": profile_enabled,
            "metric_for_acceptance": "kernel_span_us",
            "variants": [
                {
                    "name": v.name,
                    "n_heads": v.n_heads,
                    "head_dim": v.head_dim,
                    "q_lora_rank": v.q_lora_rank,
                    "rope_head_dim": v.rope_dim,
                }
                for v in VARIANTS
            ],
            "results": rows,
        }
        write_json_report(report_path, payload)
        print(f"\nWrote fused-QKV RMSNorm-RoPE perf JSON: {report_path}")
        self.assertTrue(rows)

        if env_flag("DSV4_QKV_RMSNORM_ROPE_PERF_ASSERT", False):
            self._check_perf_assert(rows, dev)

    def _check_perf_assert(self, rows: List[dict], dev: dict) -> None:
        decode_budget_us = float(
            os.environ.get("DSV4_QKV_RMSNORM_ROPE_DECODE_BUDGET_US", "5.0")
        )
        prefill_min_frac = float(
            os.environ.get("DSV4_QKV_RMSNORM_ROPE_PREFILL_MIN_FRAC", "0.70")
        )
        hbm_peak_gbps = float(dev.get("hbm_peak_gbps", 0.0))
        self.assertGreater(
            hbm_peak_gbps, 0.0,
            "device_payload() did not report an HBM peak — cannot compute prefill gate",
        )
        prefill_min_gbps = hbm_peak_gbps * prefill_min_frac

        for variant in VARIANTS:
            variant_rows = [r for r in rows if r["variant"] == variant.name]
            combined_rows = [r for r in variant_rows if r["kernel"] == "combined"]
            decode_rows = [r for r in combined_rows if r["N_tok"] <= 8]
            self.assertTrue(
                decode_rows, f"no decode-shape rows for {variant.name}"
            )
            worst_decode = max(r["kernel_span_us"] for r in decode_rows)
            self.assertLessEqual(
                worst_decode,
                decode_budget_us,
                f"[{variant.name}] decode combined latency {worst_decode:.3f}us > "
                f"budget {decode_budget_us:.3f}us",
            )
            prefill_rows = [r for r in combined_rows if r["N_tok"] >= 4096]
            self.assertTrue(
                prefill_rows, f"no prefill-shape rows for {variant.name}"
            )
            best_prefill = max(r["effective_gbps"] for r in prefill_rows)
            self.assertGreaterEqual(
                best_prefill,
                prefill_min_gbps,
                f"[{variant.name}] prefill best effective bandwidth "
                f"{best_prefill:.1f} GB/s < min {prefill_min_gbps:.1f} GB/s "
                f"({prefill_min_frac*100:.0f}% of HBM peak {hbm_peak_gbps:.0f} GB/s)",
            )

    @unittest.skipUnless(
        env_flag("DSV4_QKV_RMSNORM_ROPE_TUNE", False),
        "set DSV4_QKV_RMSNORM_ROPE_TUNE=1 to run the (G, warps, stages) sweep",
    )
    def test_perf_tune(self):
        started = time.time()
        report_path = report_path_from_env(
            "DSV4_QKV_RMSNORM_ROPE_TUNE_JSON",
            "dsv4_fused_qkv_rmsnorm_rope_tune.json",
        )
        trace_dir = trace_dir_from_report(
            report_path, "DSV4_QKV_RMSNORM_ROPE_TRACE_DIR"
        )

        rows: List[dict] = []
        for variant in VARIANTS:
            rows.extend(_tune_variant(variant, trace_dir=trace_dir))
            torch.cuda.empty_cache()

        payload = {
            "elapsed_sec": time.time() - started,
            "git_commit": git_commit(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
            ),
            "device": device_payload(),
            "tune_buckets": [
                {"bucket": b, "N_tok": n} for b, n in TUNE_BUCKETS
            ],
            "tune_configs": [
                {"group_heads": g, "num_warps": w, "num_stages": s}
                for g, w, s in TUNE_CONFIGS
            ],
            "variants": [
                {
                    "name": v.name,
                    "n_heads": v.n_heads,
                    "head_dim": v.head_dim,
                    "q_lora_rank": v.q_lora_rank,
                    "rope_head_dim": v.rope_dim,
                }
                for v in VARIANTS
            ],
            "results": rows,
        }
        write_json_report(report_path, payload)
        print(f"\nWrote fused-QKV RMSNorm-RoPE tuning JSON: {report_path}")
        self.assertTrue(rows)


if __name__ == "__main__":
    unittest.main()

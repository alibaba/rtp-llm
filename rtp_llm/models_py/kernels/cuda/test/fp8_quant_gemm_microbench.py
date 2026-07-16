import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.ops.compute_ops import (
    per_token_group_quant_fp8,
    per_token_group_quant_fp8_v2,
)


TOKEN_SIZES = [16, 64, 256, 1024, 4096, 16384, 65536]
HIDDEN_SIZES = [512, 1024, 4096, 8192, 32768]
TRACE_GEMM_SHAPES = [
    # (M, N, K) templates observed around per-token quant in the 64K prefill trace.
    (512, 4096, 224),
    (1024, 4096, 224),
    (4096, 8192, 240),
    (8192, 1024, 240),
    (32768, 1024, 240),
]


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    idx = min(len(values) - 1, int(round((len(values) - 1) * pct)))
    return sorted(values)[idx]


def _summary_ms(samples: List[float]) -> Dict[str, float]:
    return {
        "min_ms": min(samples),
        "mean_ms": statistics.mean(samples),
        "p50_ms": statistics.median(samples),
        "p90_ms": _percentile(samples, 0.90),
        "p99_ms": _percentile(samples, 0.99),
    }


def _time_cuda(fn: Callable[[], None], warmup: int, iters: int) -> List[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def _scale_tensor(m: int, k: int, device: torch.device) -> torch.Tensor:
    return create_per_token_group_quant_fp8_output_scale(
        (m, k),
        device,
        group_size=128,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )


def _quant_runner(
    kernel: str,
    x: torch.Tensor,
    out_q: torch.Tensor,
    out_s: torch.Tensor,
    eps: float,
) -> Callable[[], None]:
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_min, fp8_max = fp8_info.min, fp8_info.max

    if kernel == "legacy":
        return lambda: per_token_group_quant_fp8(
            x, out_q, out_s, 128, eps, fp8_min, fp8_max, True
        )
    if kernel == "v2":
        return lambda: per_token_group_quant_fp8_v2(
            x, out_q, out_s, 128, eps, fp8_min, fp8_max, True, False, None
        )
    if kernel == "auto":
        return lambda: sgl_per_token_group_quant_fp8(
            x,
            group_size=128,
            eps=eps,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
    raise ValueError(f"unknown kernel {kernel}")


def _can_run_kernel(kernel: str, eps: float) -> Tuple[bool, str]:
    return True, ""


def _parse_int_list(value: str) -> List[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def bench_quant(args: argparse.Namespace) -> List[Dict]:
    rows: List[Dict] = []
    device = torch.device("cuda")
    kernels = args.kernels.split(",")
    fp8_dtype = torch.float8_e4m3fn
    token_sizes = _parse_int_list(args.token_sizes)
    hidden_sizes = _parse_int_list(args.hidden_sizes)
    total = len(token_sizes) * len(hidden_sizes) * len(kernels)
    done = 0

    for m in token_sizes:
        for k in hidden_sizes:
            if k % 512 != 0:
                continue
            torch.manual_seed(m * 100000 + k)
            x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
            ref_q = torch.empty((m, k), device=device, dtype=fp8_dtype)
            ref_s = _scale_tensor(m, k, device)
            _quant_runner("legacy", x, ref_q, ref_s, args.eps)()
            torch.cuda.synchronize()

            for kernel in kernels:
                done += 1
                can_run, skip_reason = _can_run_kernel(kernel, args.eps)
                row = {
                    "kind": "quant",
                    "kernel": kernel,
                    "M": m,
                    "K": k,
                    "N": "",
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "skip_reason": skip_reason,
                }
                if not can_run:
                    rows.append(row)
                    continue

                out_q = torch.empty((m, k), device=device, dtype=fp8_dtype)
                out_s = _scale_tensor(m, k, device)
                fn = _quant_runner(kernel, x, out_q, out_s, args.eps)
                samples = _time_cuda(fn, args.warmup, args.iters)

                torch.cuda.synchronize()
                if kernel == "auto":
                    auto_q, auto_s = sgl_per_token_group_quant_fp8(
                        x,
                        group_size=128,
                        eps=args.eps,
                        column_major_scales=True,
                        scale_tma_aligned=True,
                        scale_ue8m0=True,
                    )
                    q_equal = torch.equal(auto_q, ref_q)
                    s_equal = torch.equal(auto_s, ref_s)
                else:
                    q_equal = torch.equal(out_q, ref_q)
                    s_equal = torch.equal(out_s, ref_s)
                bytes_read = m * k * 2
                bytes_written = m * k
                bytes_scale = m * (k // 512) * 4
                total_bytes = bytes_read + bytes_written + bytes_scale

                row.update(
                    _summary_ms(samples)
                    | {
                        "q_equal_legacy": q_equal,
                        "scale_equal_legacy": s_equal,
                        "effective_gbs_p50": total_bytes / (_summary_ms(samples)["p50_ms"] / 1000.0) / 1.0e9,
                        "total_bytes": total_bytes,
                    }
                )
                rows.append(row)
                print(
                    f"[{done}/{total}] {kernel} M={m} K={k} "
                    f"p50={row['p50_ms']:.4f}ms q_equal={q_equal} s_equal={s_equal}",
                    flush=True,
                )
    return rows


def bench_quant_gemm(args: argparse.Namespace) -> List[Dict]:
    try:
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_gemm_nt, has_deep_gemm
        from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import per_block_cast_to_fp8
    except Exception as exc:
        return [{"kind": "quant_gemm", "skip_reason": f"deepgemm import failed: {exc}"}]

    if not has_deep_gemm():
        return [{"kind": "quant_gemm", "skip_reason": "deep_gemm is not available"}]

    rows: List[Dict] = []
    device = torch.device("cuda")
    fp8_dtype = torch.float8_e4m3fn
    fp8_info = torch.finfo(fp8_dtype)

    for m, n, k in TRACE_GEMM_SHAPES:
        torch.manual_seed(m * 1000000 + n * 1000 + k)
        x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
        w_bf16 = torch.randn((n, k), device=device, dtype=torch.bfloat16) * 0.01
        w_fp8, w_scale = per_block_cast_to_fp8(w_bf16, use_ue8m0=True)
        out_q = torch.empty((m, k), device=device, dtype=fp8_dtype)
        out_s = _scale_tensor(m, k, device)
        out = torch.empty((m, n), device=device, dtype=torch.bfloat16)

        for kernel in args.kernels.split(","):
            can_run, skip_reason = _can_run_kernel(kernel, args.eps)
            row = {
                "kind": "quant_gemm",
                "kernel": kernel,
                "M": m,
                "N": n,
                "K": k,
                "warmup": args.warmup,
                "iters": args.iters,
                "skip_reason": skip_reason,
            }
            if not can_run:
                rows.append(row)
                continue

            quant = _quant_runner(kernel, x, out_q, out_s, args.eps)

            def fn() -> None:
                quant()
                fp8_gemm_nt((out_q, out_s), (w_fp8, w_scale), out, disable_ue8m0_cast=False)

            samples = _time_cuda(fn, args.warmup, args.iters)
            row.update(_summary_ms(samples))
            row["policy_candidate"] = False
            rows.append(row)
            print(
                f"[quant_gemm] {kernel} M={m} N={n} K={k} p50={row['p50_ms']:.4f}ms",
                flush=True,
            )
    return rows


def write_outputs(rows: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "microbench.json"
    csv_path = out_dir / "microbench.csv"
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "env": {
            "DSV4_FP8_QUANT_KERNEL": os.environ.get("DSV4_FP8_QUANT_KERNEL", ""),
        },
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="docs/dsv4/profiling/quant_gemm_kernel")
    parser.add_argument("--kernels", default="legacy,v2,auto")
    parser.add_argument("--token-sizes", default=",".join(str(x) for x in TOKEN_SIZES))
    parser.add_argument("--hidden-sizes", default=",".join(str(x) for x in HIDDEN_SIZES))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1.0e-4)
    parser.add_argument("--skip-gemm", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fp8 quant microbench")

    rows = bench_quant(args)
    if not args.skip_gemm:
        rows.extend(bench_quant_gemm(args))
    write_outputs(rows, Path(args.out_dir))


if __name__ == "__main__":
    main()

"""Reproducible Blackwell masked SiLU/FP8/packed-UE8M0 CUDA-graph benchmark.

Run from the source root after checking develop/pro5000-opt and rebuilding the
installed wheel to that baseline (see workspace AGENTS.md)::

    python -m rtp_llm.models_py.triton_kernels.common.test.silu_mul_masked_packed_benchmark \
        --output /tmp/sm120-packed.json
    # Include the real masked Down GEMM, with a Qwen-sized output dimension:
    python -m rtp_llm.models_py.triton_kernels.common.test.silu_mul_masked_packed_benchmark \
        --shape 128,128,768 --down-hidden 2048 --output /tmp/sm120-down.json

E,M,H describe expert count, per-expert token capacity and SiLU output width.
Inputs, activation outputs, FP32 scales and fused packed scales are preallocated.
Every baseline call zeroes the whole FP32 scale, invokes the existing activation,
then calls the public pack launcher, including its packed-storage zeroing. Its
allocations become fixed graph-owned allocations on replay. Thus timings include
all device initialization and conversion work, but exclude Python dispatch and
host allocator overhead. Fused output scales are completely overwritten by the
kernel; they do not need an extra initialization pass. Activation output padding
is initialized once outside timing and does not contribute to valid Down GEMM
outputs; hardware may still load padding as part of a complete tile.

Use --round-intermediate to compare the original rounded packed arithmetic
instead of the ordinary FP32 arithmetic used by SM120.

Default counts are fixed during each measurement. This is a steady-state device
microbenchmark, not request latency or changing-routing performance. Compilation,
input generation, weight packing, validation and graph capture are outside timing.
"""

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_shape(value):
    try:
        expert_num, capacity, hidden = map(int, value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be E,M,H") from exc
    if min(expert_num, capacity, hidden) <= 0 or hidden % 128:
        raise argparse.ArgumentTypeError(
            "E,M,H must be positive and H divisible by 128"
        )
    return expert_num, capacity, hidden


def make_counts(expert_num, capacity, profile):
    if profile == "full":
        return [capacity] * expert_num
    if profile == "sparse":
        return [
            min(1 + (i // 8) % 8, capacity) if i % 8 == 0 else 0
            for i in range(expert_num)
        ]
    if expert_num == 1:
        return [max(1, capacity - 1)]
    return [
        capacity if i == 0 else capacity // 2 if i == 1 else int(i % 8 == 0)
        for i in range(expert_num)
    ]


def capture(torch, function, warmup, inner_repeats):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            function()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(inner_repeats):
            function()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def time_pair(torch, functions, args):
    graphs = {
        name: capture(torch, fn, args.warmup, args.inner_repeats)
        for name, fn in functions.items()
    }
    samples = {name: [] for name in functions}
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for round_index in range(args.rounds):
        # Alternate order to reduce systematic thermal/order bias.
        order = list(graphs)
        if round_index % 2:
            order.reverse()
        for name in order:
            events[0].record()
            for _ in range(args.replays):
                graphs[name].replay()
            events[1].record()
            events[1].synchronize()
            samples[name].append(
                events[0].elapsed_time(events[1])
                * 1000
                / (args.inner_repeats * args.replays)
            )
    result = {
        name: {
            "median_us": statistics.median(values),
            "min_us": min(values),
            "max_us": max(values),
            "round_us": values,
        }
        for name, values in samples.items()
    }
    result["speedup"] = result["baseline"]["median_us"] / result["fused"]["median_us"]
    return result


def benchmark_case(torch, wrapper, activation, shape, profile, args):
    expert_num, capacity, hidden = shape
    counts_host = make_counts(expert_num, capacity, profile)
    expected_m = max(1, (sum(counts_host) + expert_num - 1) // expert_num)
    counts = torch.tensor(counts_host, dtype=torch.int32, device="cuda")
    x = torch.randn(
        (expert_num, capacity, 2 * hidden), device="cuda", dtype=torch.bfloat16
    )
    baseline_out = torch.zeros(
        (expert_num, capacity, hidden), device="cuda", dtype=torch.float8_e4m3fn
    )
    fused_out = torch.zeros_like(baseline_out)
    fp32_scale = torch.zeros(
        (expert_num, capacity, hidden // 128), device="cuda", dtype=torch.float32
    )
    fused_scale = activation.create_packed_scale_tensor(
        expert_num, capacity, 2 * hidden, 128, x.device
    )
    aligned_capacity = fused_scale.stride(2)
    fused_scale_storage = fused_scale.as_strided(
        (expert_num, aligned_capacity, fused_scale.shape[2]), fused_scale.stride()
    )
    # Poison physical TMA padding as well as logical rows before validation.
    fused_scale_storage.fill_(-1)
    baseline_state = {}

    def baseline():
        fp32_scale.zero_()
        if args.round_intermediate:
            activation.silu_and_mul_masked_post_quant_fwd(
                x, baseline_out, fp32_scale, 128, counts, scale_ue8m0=True
            )
        else:
            activation.silu_mul_masked_fp8_post_quant_fwd(
                x,
                baseline_out,
                fp32_scale,
                128,
                counts,
                expected_m,
                input_layout=activation.MaskedSiluInputLayout.PER_EXPERT_CAPACITY,
                scale_ue8m0=True,
            )
        baseline_state["scale"] = wrapper.pack_ue8m0_kernel_launcher(fp32_scale, 1)

    def fused_call():
        activation.silu_and_mul_masked_post_quant_packed_fwd(
            x,
            fused_out,
            fused_scale,
            128,
            counts,
            round_intermediate=args.round_intermediate,
        )

    baseline()
    fused_call()
    valid = torch.arange(capacity, device="cuda")[None, :] < counts[:, None]
    old_values = baseline_out.float()[valid]
    new_values = fused_out.float()[valid]
    old_bits = baseline_out.view(torch.uint8)[valid]
    new_bits = fused_out.view(torch.uint8)[valid]
    baseline_scale = baseline_state["scale"]
    baseline_scale_storage = baseline_scale.as_strided(
        (expert_num, aligned_capacity, baseline_scale.shape[2]),
        baseline_scale.stride(),
    )
    # Include invalid rows, unused tail bytes and physical TMA padding. The
    # public pack launcher zero-initializes the complete baseline allocation.
    torch.testing.assert_close(
        fused_scale_storage, baseline_scale_storage, rtol=0, atol=0
    )
    torch.testing.assert_close(new_bits, old_bits, rtol=0, atol=0)
    validation = {
        "packed_scales_exact": True,
        "packed_physical_storage_exact": True,
        "physical_padding_rows_per_expert": aligned_capacity - capacity,
        "active_fp8_exact": True,
        "active_fp8_mismatch_count": int((old_bits != new_bits).sum().item()),
        "active_fp8_max_abs_difference": float(
            (old_values - new_values).abs().max().item()
        ),
        "fp8_comparison": "bitwise_uint8",
    }
    result = {
        "experts": expert_num,
        "capacity": capacity,
        "hidden": hidden,
        "profile": profile,
        "counts": counts_host,
        "active_tokens": sum(counts_host),
        "expected_m": expected_m,
        "packed_scale_shape": list(fused_scale.shape),
        "packed_scale_stride": list(fused_scale.stride()),
        "validation": validation,
        "activation_pack": time_pair(
            torch, {"baseline": baseline, "fused": fused_call}, args
        ),
    }

    if args.down_hidden:
        # Finite FP8 values with legal, exact power-of-two unit scales. Pack
        # weights once, outside timing, so only the activation producer differs.
        weight = torch.randn(
            (expert_num, args.down_hidden, hidden), device="cuda", dtype=torch.bfloat16
        )
        weight = weight.clamp_(-2, 2).to(torch.float8_e4m3fn)
        weight_scale = wrapper.pack_ue8m0_kernel_launcher(
            torch.ones(
                (expert_num, args.down_hidden // 128, hidden // 128), device="cuda"
            ),
            128,
        )
        old_down = torch.zeros(
            (expert_num, capacity, args.down_hidden),
            device="cuda",
            dtype=torch.bfloat16,
        )
        new_down = torch.zeros_like(old_down)

        def baseline_down():
            baseline()
            wrapper.m_grouped_fp8_gemm_nt_masked(
                (baseline_out, baseline_state["scale"]),
                (weight, weight_scale),
                old_down,
                counts,
                expected_m,
                disable_ue8m0_cast=False,
            )

        def fused_down():
            fused_call()
            wrapper.m_grouped_fp8_gemm_nt_masked(
                (fused_out, fused_scale),
                (weight, weight_scale),
                new_down,
                counts,
                expected_m,
                disable_ue8m0_cast=False,
            )

        baseline_down()
        fused_down()
        old_valid = old_down.float()[valid]
        new_valid = new_down.float()[valid]
        if (
            not torch.isfinite(old_valid).all().item()
            or not torch.isfinite(new_valid).all().item()
        ):
            raise AssertionError("Down GEMM produced non-finite active output")
        normalized_rmse = (
            (old_valid - new_valid).square().mean().sqrt()
            / old_valid.square().mean().sqrt().clamp_min(1e-12)
        ).item()
        torch.testing.assert_close(
            new_down.view(torch.int16)[valid],
            old_down.view(torch.int16)[valid],
            rtol=0,
            atol=0,
        )
        result["down_hidden"] = args.down_hidden
        result["validation"]["down_active_bitwise_exact"] = True
        result["validation"]["down_normalized_rmse"] = normalized_rmse
        result["validation"]["down_max_abs_difference"] = float(
            (old_valid - new_valid).abs().max().item()
        )
        result["activation_pack_down"] = time_pair(
            torch, {"baseline": baseline_down, "fused": fused_down}, args
        )
    torch.cuda.synchronize()
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--shape",
        type=parse_shape,
        action="append",
        help="repeatable E,M,H; H is output width",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=("sparse", "skew", "full"),
        default=["sparse", "skew", "full"],
    )
    parser.add_argument(
        "--down-hidden",
        type=int,
        default=0,
        help="also benchmark Down GEMM with this output width (multiple of 128)",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--inner-repeats",
        type=int,
        default=32,
        help="pipeline calls captured per graph",
    )
    parser.add_argument(
        "--replays", type=int, default=4, help="graph replays per timed round"
    )
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument(
        "--round-intermediate",
        action="store_true",
        help="match original packed input-dtype arithmetic (default: ordinary FP32 arithmetic)",
    )
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, help="also write JSON here (stdout always receives JSON)"
    )
    args = parser.parse_args()
    if min(args.warmup, args.inner_repeats, args.replays, args.rounds) < 1:
        parser.error("warmup, inner-repeats, replays and rounds must be positive")
    if args.down_hidden < 0 or args.down_hidden % 128:
        parser.error("down-hidden must be zero or a positive multiple of 128")

    # Lazy imports keep --help usable without an installed CUDA runtime.
    import torch
    import triton

    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability()[0] not in (10, 12):
        raise RuntimeError("This benchmark requires SM100/SM120 hardware")
    from rtp_llm.models_py.kernels.cuda import deepgemm_wrapper as wrapper
    from rtp_llm.models_py.triton_kernels.common import activation

    torch.manual_seed(args.seed)
    shapes = args.shape or [
        (128, 128, 768),
        (128, 256, 768),
        (1, 128, 384),
        (1, 128, 512),
        (1, 128, 1024),
        (1, 128, 2048),
    ]
    source_root = Path(__file__).resolve().parents[5]

    def git_output(*command):
        run = subprocess.run(
            ["git", "-C", str(source_root), *command], capture_output=True, text=True
        )
        return run.stdout.strip() if run.returncode == 0 else None

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "source_head": git_output("rev-parse", "HEAD"),
        "source_status": git_output("status", "--porcelain"),
        "activation_module": activation.__file__,
        "fused_module": activation.__file__,
        "seed": args.seed,
        "round_intermediate": args.round_intermediate,
        "warmup_calls": args.warmup,
        "calls_per_graph": args.inner_repeats,
        "replays_per_round": args.replays,
        "rounds": args.rounds,
        "timing": "CUDA events around CUDA graph replay; microseconds per pipeline call",
        "baseline_initialization": "FP32 scale zero each call; public pack launcher includes packed output zero",
        "allocation": "preallocated buffers plus fixed graph-owned baseline pack outputs; host allocator excluded",
        "routing": "fixed counts per case; steady-state device timing",
        "down_weights": "synthetic finite clamped FP8, unit UE8M0 scales packed before timing",
    }
    report = {"metadata": metadata, "results": []}
    for shape in shapes:
        for profile in args.profiles:
            print(
                f"Benchmark E,M,H={shape} profile={profile}",
                file=sys.stderr,
                flush=True,
            )
            report["results"].append(
                benchmark_case(torch, wrapper, activation, shape, profile, args)
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

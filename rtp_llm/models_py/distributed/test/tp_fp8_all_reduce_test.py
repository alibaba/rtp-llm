#!/usr/bin/env python3
"""Real two-rank CUDA correctness and microbenchmark for TP FP8 all-reduce.

Run with ``torchrun --standalone --nproc_per_node=2``.  This is deliberately a
standalone script: it exercises the communicator API with a real NCCL process
group and does not represent a Qwen serving or TTFT benchmark.
"""

import argparse
import json
import os
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.tp_fp8_all_reduce import TpFp8AllReduceCommunicator

WARMUP = 30
ITERS = 100
HIDDEN_SIZE = 2048
NATIVE_KERNEL_NAMES = (
    "preprocess_tp2_bf16_fp8",
    "two_shot_tp2",
)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of no samples")
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _group_max_ms(value: float) -> float:
    local = torch.tensor([value], device="cuda", dtype=torch.float64)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return max(item.item() for item in gathered)


def _make_payload(num_bytes: int, rank: int, iteration: int) -> torch.Tensor:
    """Mixed signs, zeroes, outliers, and cancellation without host payload I/O."""
    if num_bytes % 2:
        raise ValueError("BF16 byte size must be even")
    numel = num_bytes // 2
    base = torch.arange(numel, device="cuda", dtype=torch.int32)
    values = (base.remainder(31) - 15).to(torch.float32) / 16.0
    values = torch.where((base % 23) == 0, 0.0, values)
    # The rank-dependent term makes the reduced value exercise cancellation;
    # the rare outlier protects scale selection from only seeing unit values.
    values = values + (1.0 if rank == 0 else -0.875) + iteration * 0.03125
    values = torch.where(
        (base % 4093) == 0,
        torch.full_like(values, 64.0 if rank == 0 else -63.5),
        values,
    )
    values = values.to(torch.bfloat16).contiguous()
    if numel % HIDDEN_SIZE == 0:
        return values.reshape(-1, HIDDEN_SIZE)
    return values


def _reference(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bf16 = input_tensor.clone(memory_format=torch.contiguous_format)
    fp32 = bf16.float()
    dist.all_reduce(bf16)
    dist.all_reduce(fp32)
    return bf16, fp32


def _codec_quantize_dequantize(values: torch.Tensor) -> torch.Tensor:
    """TRT TP2's FP8 E4M3 warp codec, including a 496-value partial warp."""
    flat = values.reshape(-1).float()
    warp_values = 31 * 16
    padded = torch.nn.functional.pad(flat, (0, (-flat.numel()) % warp_values))
    groups = padded.reshape(-1, warp_values)
    max_abs = groups.abs().amax(dim=1, keepdim=True)
    scale = torch.where(max_abs == 0, torch.zeros_like(max_abs), 448.0 / max_abs)
    encoded = (groups * scale).to(torch.float8_e4m3fn)
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    decoded = torch.where(
        scale == 0, torch.zeros_like(groups), encoded.float() / safe_scale
    )
    return decoded.reshape(-1)[: flat.numel()]


def _codec_reference(input_tensor: torch.Tensor) -> torch.Tensor:
    """Reference the native two-shot TP2 packet protocol, not BF16 NCCL."""
    if input_tensor.numel() % 2:
        raise ValueError("TP2 codec requires an even element count")
    inputs = [
        torch.empty_like(input_tensor, memory_format=torch.contiguous_format)
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(inputs, input_tensor.contiguous())
    partition = input_tensor.numel() // 2
    reduced = []
    for start in range(0, input_tensor.numel(), partition):
        end = start + partition
        phase_one = sum(
            _codec_quantize_dequantize(source.reshape(-1)[start:end])
            for source in inputs
        )
        reduced.append(_codec_quantize_dequantize(phase_one))
    return torch.cat(reduced).to(torch.bfloat16).reshape_as(input_tensor)


def _error_stats(actual: torch.Tensor, bf16: torch.Tensor, fp32: torch.Tensor) -> dict:
    actual32 = actual.float()

    def compare(reference: torch.Tensor) -> dict:
        diff = actual32 - reference.float()
        numerator = torch.linalg.vector_norm(diff)
        denominator = torch.linalg.vector_norm(reference.float()).clamp_min(1e-30)
        return {
            "max_abs": float(diff.abs().max().item()),
            "relative_l2": float((numerator / denominator).item()),
        }

    return {"vs_bf16_nccl": compare(bf16), "vs_fp32_nccl": compare(fp32)}


def _assert_ranks_bitwise_equal(output: torch.Tensor) -> None:
    copies = [torch.empty_like(output) for _ in range(dist.get_world_size())]
    dist.all_gather(copies, output)
    if not all(torch.equal(copies[0], item) for item in copies[1:]):
        raise AssertionError("TP FP8 all-reduce produced different BF16 bits on ranks")


def _run_sync(
    communicator: TpFp8AllReduceCommunicator, input_tensor: torch.Tensor
) -> torch.Tensor:
    return communicator.all_reduce(input_tensor)


def _run_async(
    communicator: TpFp8AllReduceCommunicator, input_tensor: torch.Tensor
) -> torch.Tensor:
    return communicator.all_reduce_async(input_tensor).wait()


def _correctness_case(
    communicator: TpFp8AllReduceCommunicator,
    num_bytes: int,
    rank: int,
    iteration: int,
    mode: str,
) -> dict:
    input_tensor = _make_payload(num_bytes, rank, iteration)
    before = input_tensor.clone()
    bf16_ref, fp32_ref = _reference(input_tensor)
    codec_ref = _codec_reference(input_tensor)
    runner = _run_sync if mode == "sync" else _run_async
    output = runner(communicator, input_tensor)
    torch.cuda.synchronize()
    if output.data_ptr() == input_tensor.data_ptr():
        raise AssertionError("default all_reduce output must not alias the input")
    if not torch.equal(input_tensor, before):
        raise AssertionError("default all_reduce unexpectedly modified its input")
    if not torch.isfinite(output).all():
        raise AssertionError("TP FP8 all-reduce emitted NaN or Inf")
    _assert_ranks_bitwise_equal(output)
    if not torch.equal(output, codec_ref):
        raise AssertionError("native output differs from the TRT TP2 codec oracle")
    return {
        "bytes": num_bytes,
        "shape": list(input_tensor.shape),
        "iteration": iteration,
        "mode": mode,
        "error": _error_stats(output, bf16_ref, fp32_ref),
        "codec_oracle_bitwise_equal": True,
    }


def _verify_explicit_output_and_nondefault_stream(
    communicator: TpFp8AllReduceCommunicator, rank: int
) -> None:
    input_tensor = _make_payload(32 * 1024 * 1024, rank, 7)
    output = torch.empty_like(input_tensor)
    returned = communicator.all_reduce(input_tensor, out=output)
    if returned is not output:
        raise AssertionError("explicit output was not returned")
    _assert_ranks_bitwise_equal(output)

    inplace = _make_payload(32 * 1024 * 1024, rank, 8)
    returned = communicator.all_reduce(inplace, out=inplace)
    if returned is not inplace:
        raise AssertionError("in-place output was not returned")
    _assert_ranks_bitwise_equal(inplace)

    consumer = torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        pending = communicator.all_reduce_async(input_tensor)
        result = pending.wait()
        # A consuming kernel on this stream makes a missing wait/event visible.
        consumed = result.float().sum()
    torch.cuda.current_stream().wait_stream(consumer)
    torch.cuda.synchronize()
    if not torch.isfinite(consumed):
        raise AssertionError("non-default-stream consumer observed invalid output")


def _verify_async_workspace_reuse(
    communicator: TpFp8AllReduceCommunicator, rank: int
) -> list[dict]:
    """Queue changing payloads without an intervening wait or NCCL barrier."""
    inputs = [
        _make_payload(32 * 1024 * 1024, rank, iteration) for iteration in range(3)
    ]
    references = [
        (*_reference(input_tensor), _codec_reference(input_tensor))
        for input_tensor in inputs
    ]
    # This is intentionally consecutive: it catches an IPC workspace/event
    # ownership race which a per-call synchronize would hide.
    pending = [communicator.all_reduce_async(input_tensor) for input_tensor in inputs]
    outputs = [handle.wait() for handle in pending]
    torch.cuda.synchronize()
    results = []
    for iteration, (output, (bf16_ref, fp32_ref, codec_ref)) in enumerate(
        zip(outputs, references)
    ):
        _assert_ranks_bitwise_equal(output)
        if not torch.isfinite(output).all():
            raise AssertionError("queued TP FP8 all-reduce emitted NaN or Inf")
        if not torch.equal(output, codec_ref):
            raise AssertionError("queued native output differs from the codec oracle")
        results.append(
            {
                "iteration": iteration,
                "mode": "async_consecutive_no_wait",
                "error": _error_stats(output, bf16_ref, fp32_ref),
                "codec_oracle_bitwise_equal": True,
            }
        )
    return results


def _verify_exact_codec_cases(
    communicator: TpFp8AllReduceCommunicator, rank: int
) -> list[dict]:
    results = []
    for label, local_value in (("zero", 0.0), ("constant", float(rank + 1))):
        input_tensor = torch.full(
            (32,), local_value, device="cuda", dtype=torch.bfloat16
        )
        bf16_ref, fp32_ref = _reference(input_tensor)
        codec_ref = _codec_reference(input_tensor)
        output = communicator.all_reduce(input_tensor)
        torch.cuda.synchronize()
        _assert_ranks_bitwise_equal(output)
        if not torch.equal(output, codec_ref):
            raise AssertionError(f"{label} output differs from the codec oracle")
        results.append(
            {
                "case": label,
                "shape": list(input_tensor.shape),
                "codec_oracle_bitwise_equal": True,
                "error": _error_stats(output, bf16_ref, fp32_ref),
            }
        )
    return results


def _make_rank_asymmetric_layout(payload: torch.Tensor, rank: int) -> torch.Tensor:
    if rank == 0:
        return payload.contiguous()
    storage = torch.empty(payload.numel() + 1, device="cuda", dtype=torch.bfloat16)
    # offset one BF16 element and transpose the logical [tokens, hidden] view:
    # rank 1 is both non-contiguous and not 16-byte aligned.
    view = storage[1:].reshape(HIDDEN_SIZE, -1).transpose(0, 1)
    view.copy_(payload)
    if view.is_contiguous() or view.data_ptr() % 16 == 0:
        raise AssertionError("failed to construct the rank-1 asymmetric layout")
    return view


def _verify_asymmetric_layout(
    communicator: TpFp8AllReduceCommunicator, rank: int
) -> list[dict]:
    results = []
    for mode in ("sync", "async"):
        input_tensor = _make_rank_asymmetric_layout(
            _make_payload(32 * 1024 * 1024, rank, 29), rank
        )
        before = input_tensor.clone(memory_format=torch.contiguous_format)
        bf16_ref, fp32_ref = _reference(input_tensor)
        codec_ref = _codec_reference(input_tensor)
        if mode == "sync":
            output = communicator.all_reduce(input_tensor, out=input_tensor)
        else:
            output = communicator.all_reduce_async(
                input_tensor, out=input_tensor
            ).wait()
        torch.cuda.synchronize()
        if output is not input_tensor:
            raise AssertionError("in-place asymmetric-layout output was not returned")
        if torch.equal(input_tensor, before):
            raise AssertionError("asymmetric in-place output was not copied back")
        _assert_ranks_bitwise_equal(output.contiguous())
        if not torch.equal(output.contiguous(), codec_ref):
            raise AssertionError("asymmetric layout differs from the codec oracle")
        results.append(
            {
                "mode": f"{mode}_inplace_asymmetric_layout",
                "rank0_contiguous_rank1_noncontiguous_offset1": True,
                "codec_oracle_bitwise_equal": True,
                "error": _error_stats(output, bf16_ref, fp32_ref),
            }
        )
    return results


def _timed_samples(
    fn: Callable[[], torch.Tensor], warmup: int, iters: int
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(_group_max_ms(start.elapsed_time(end)))
    return samples


def _profile(
    fn: Callable[[], torch.Tensor],
    output_dir: Path,
    label: str,
    rank: int,
    *,
    require_native: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = output_dir / f"tp_fp8_all_reduce_{label}_rank{rank}.json"
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    profiler.export_chrome_trace(str(trace))
    names = {event.key for event in profiler.key_averages()}
    matched = [key for key in names if any(name in key for name in NATIVE_KERNEL_NAMES)]
    missing = [
        name for name in NATIVE_KERNEL_NAMES if not any(name in key for key in names)
    ]
    if require_native and missing:
        raise AssertionError(
            f"native FP8 all-reduce kernels missing from profile: {missing}"
        )
    return {"trace": str(trace), "native_kernel_events": sorted(matched)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes-mib", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("requires two visible CUDA GPUs")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError(f"requires torchrun world_size=2, got {world_size}")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    communicator = None
    try:
        sizes = [size * 1024 * 1024 for size in args.sizes_mib]
        max_bytes = max(sizes)
        communicator = TpFp8AllReduceCommunicator(
            dist.group.WORLD,
            torch.device("cuda", rank),
            max_bytes=max_bytes,
            min_bytes=64,
        )
        edge_sizes = (64, 1024, 7936 * 2)
        correctness = [
            _correctness_case(communicator, size, rank, iteration, mode)
            for size in (*edge_sizes, *sizes)
            for iteration in range(3)
            for mode in ("sync", "async")
        ]
        correctness.extend(_verify_async_workspace_reuse(communicator, rank))
        correctness.extend(_verify_exact_codec_cases(communicator, rank))
        correctness.extend(_verify_asymmetric_layout(communicator, rank))
        _verify_explicit_output_and_nondefault_stream(communicator, rank)

        timing = {}
        profile = {}
        for size in sizes:
            # Fixed zero inputs keep in-place NCCL from growing values across
            # 100 iterations. Allocation stays outside the timed region.
            input_tensor = torch.zeros(
                (size // 2 // HIDDEN_SIZE, HIDDEN_SIZE),
                device="cuda",
                dtype=torch.bfloat16,
            )
            native_output = torch.empty_like(input_tensor)
            variants = (
                ("nccl_bf16", lambda: dist.all_reduce(input_tensor), False),
                (
                    "tp_fp8_sync",
                    lambda: communicator.all_reduce(input_tensor, out=native_output),
                    True,
                ),
                (
                    "tp_fp8_async",
                    lambda: communicator.all_reduce_async(
                        input_tensor, out=native_output
                    ).wait(),
                    True,
                ),
            )
            for label, runner, is_native in variants:
                samples = _timed_samples(
                    runner,
                    args.warmup,
                    args.iters,
                )
                timing[f"{size // (1024 * 1024)}MiB_{label}"] = {
                    "input_shape": list(input_tensor.shape),
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "timing": "CUDA events; per iteration all-rank maximum",
                    "p50_ms": _percentile(samples, 0.50),
                    "p90_ms": _percentile(samples, 0.90),
                    "min_ms": min(samples),
                    "max_ms": max(samples),
                }
                if args.profile_dir is not None:
                    profile[f"{size // (1024 * 1024)}MiB_{label}"] = _profile(
                        runner,
                        args.profile_dir,
                        f"{size // (1024 * 1024)}MiB_{label}",
                        rank,
                        require_native=is_native,
                    )

        result = {
            "rank": rank,
            "world_size": world_size,
            "blocks": communicator.blocks,
            "native_calls": communicator.calls,
            "precision": "native TRT low-precision all-reduce; errors are reported, not hidden by tolerance",
            "workload": "synthetic BF16 payloads only; not a Qwen or serving benchmark",
            "correctness": correctness,
            "timing": timing,
            "profile": profile,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        rank_json = args.json.with_name(
            f"{args.json.stem}.rank{rank}{args.json.suffix}"
        )
        rank_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        dist.barrier()
    except Exception:
        # A failing rank must not enter teardown collectives while its peer is
        # still in a different test collective. Let torchrun terminate peers.
        traceback.print_exc()
        os._exit(1)
    finally:
        if communicator is not None:
            communicator.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

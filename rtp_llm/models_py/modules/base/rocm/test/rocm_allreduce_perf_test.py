import argparse
import socket
import sys
from dataclasses import asdict
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rocm_allreduce_perf_lib import (
    DEFAULT_BACKENDS,
    MB,
    ResultRow,
    format_result_header,
    format_result_row,
    generate_shapes,
    parse_backends,
    parse_int_csv,
    parse_shapes,
    summarize_us,
    write_results_json,
)

DEFAULT_BYTE_TARGETS = (
    "1_token,2_token,4_token,8_token,16_token,"
    "32KB,64KB,128KB,256KB,512KB,1MB,2MB,4MB,8MB,16MB,32MB,64MB"
)


def dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def dtype_label(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    raise ValueError(f"unsupported dtype: {dtype}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROCm all-reduce microbenchmark")
    parser.add_argument("--backends", default=",".join(DEFAULT_BACKENDS))
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--hidden-sizes", default="4096,5120")
    parser.add_argument("--one-token-hidden-size", type=int, default=5120)
    parser.add_argument("--max-bytes-mb", type=int, default=64)
    parser.add_argument("--byte-targets", default=DEFAULT_BYTE_TARGETS)
    parser.add_argument("--shapes", default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--quick-reduce-min-size-mb", type=int, default=None)
    parser.add_argument("--quick-reduce-max-size-mb", type=int, default=None)
    parser.add_argument(
        "--quick-reduce-quantization-min-size-kb", type=int, default=None
    )
    parser.add_argument("--disable-correctness-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_args(argv):
    parser = build_parser()
    args = parser.parse_args(argv)
    dtype = dtype_from_name(args.dtype)
    dtype_size = torch.tensor([], dtype=dtype).element_size()
    args.backend_specs = parse_backends(args.backends)
    args.hidden_size_values = parse_int_csv(args.hidden_sizes)
    args.byte_target_values = [
        item.strip() for item in args.byte_targets.split(",") if item.strip()
    ]
    args.shape_values = parse_shapes(args.shapes) if args.shapes else None
    args.dtype_value = dtype
    args.dtype_size = dtype_size
    args.dtype_label = dtype_label(dtype)
    args.max_bytes = args.max_bytes_mb * MB
    args.bench_shapes = generate_shapes(
        byte_targets=args.byte_target_values,
        hidden_sizes=args.hidden_size_values,
        dtype_size=args.dtype_size,
        one_token_hidden_size=args.one_token_hidden_size,
        max_bytes=args.max_bytes,
        explicit_shapes=args.shape_values,
    )
    return args


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def setup_distributed(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{rank}"),
    )


def cleanup_distributed(metadata_group=None) -> None:
    try:
        if dist.is_initialized():
            dist.barrier()
            torch.cuda.synchronize()
    except Exception:
        pass
    try:
        if metadata_group is not None:
            dist.destroy_process_group(metadata_group)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def make_input(rank: int, shape, dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(20260622 + rank)
    return torch.randn(shape.shape, device=f"cuda:{rank}", dtype=dtype)


def rccl_all_reduce(inp: torch.Tensor) -> torch.Tensor:
    out = inp.clone()
    dist.all_reduce(out, group=dist.group.WORLD)
    return out


def time_backend(fn, inp: torch.Tensor, warmup: int, iters: int) -> List[float]:
    for _ in range(warmup):
        fn(inp)
    torch.cuda.synchronize()
    dist.barrier()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for idx in range(iters):
        start_events[idx].record()
        fn(inp)
        end_events[idx].record()
    torch.cuda.synchronize()
    dist.barrier()
    return [
        start_events[idx].elapsed_time(end_events[idx]) * 1000.0 for idx in range(iters)
    ]


def ok_row(backend: str, args, shape, times: List[float]) -> ResultRow:
    summary = summarize_us(times)
    avg_seconds = summary["avg_us"] / 1_000_000.0
    algbw = shape.actual_bytes / avg_seconds / 1_000_000_000.0
    return ResultRow(
        backend=backend,
        dtype=args.dtype_label,
        target=shape.target,
        shape=shape.shape_label,
        bytes=shape.actual_bytes,
        status="OK",
        avg_us=summary["avg_us"],
        p50_us=summary["p50_us"],
        p90_us=summary["p90_us"],
        min_us=summary["min_us"],
        max_us=summary["max_us"],
        algbw_GBps=algbw,
        note="",
    )


def skip_row(backend: str, args, shape, note: str) -> ResultRow:
    return ResultRow(
        backend=backend,
        dtype=args.dtype_label,
        target=shape.target,
        shape=shape.shape_label,
        bytes=shape.actual_bytes,
        status="SKIP",
        avg_us=None,
        p50_us=None,
        p90_us=None,
        min_us=None,
        max_us=None,
        algbw_GBps=None,
        note=note,
    )


def fail_row(backend: str, args, shape, note: str) -> ResultRow:
    return ResultRow(
        backend=backend,
        dtype=args.dtype_label,
        target=shape.target,
        shape=shape.shape_label,
        bytes=shape.actual_bytes,
        status="FAIL",
        avg_us=None,
        p50_us=None,
        p90_us=None,
        min_us=None,
        max_us=None,
        algbw_GBps=None,
        note=note,
    )


def assert_close_to_ref(
    backend_name: str, out: Optional[torch.Tensor], ref: torch.Tensor
) -> Optional[str]:
    if out is None:
        return f"{backend_name} returned None"
    try:
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
        return None
    except AssertionError:
        max_abs = (out - ref).abs().max().item()
        return f"{backend_name} correctness failed max_abs={max_abs:.6g}"


def quick_reduce_tolerance(backend_name: str):
    return {
        "quick_reduce_fp": (1e-2, 1e-2),
        "quick_reduce_int8": (8e-2, 8e-2),
        "quick_reduce_int6": (2e-1, 2e-1),
        "quick_reduce_int4": (1.0, 1.0),
    }[backend_name]


def assert_quick_reduce_close(
    backend_name: str, out: torch.Tensor, ref: torch.Tensor
) -> Optional[str]:
    rtol, atol = quick_reduce_tolerance(backend_name)
    try:
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
        return None
    except AssertionError:
        max_abs = (out - ref).abs().max().item()
        return f"{backend_name} correctness failed max_abs={max_abs:.6g}"


def worker(rank: int, world_size: int, port: int, args, return_dict) -> None:
    setup_distributed(rank, world_size, port)
    metadata_group = None
    rows: List[ResultRow] = []
    trt_env = None
    trt_supported_hidden_sizes = frozenset()
    trt_init_note = None
    vllm_custom = None
    vllm_init_note = None
    quick_reduce_managers = {}
    quick_reduce_init_notes = {}
    try:
        needs_metadata_group = any(
            spec.name == "vllm_custom" or spec.quantization is not None
            for spec in args.backend_specs
        )
        if needs_metadata_group:
            metadata_group = dist.new_group(
                ranks=list(range(world_size)),
                backend="gloo",
            )
        if any(spec.name == "trt" for spec in args.backend_specs):
            try:
                from rtp_llm.models_py.modules.base.rocm.trt_allreduce import (
                    ALLREDUCE_SUPPORTED_HIDDEN_SIZES,
                    TrtllmDistEnv,
                )

                trt_supported_hidden_sizes = ALLREDUCE_SUPPORTED_HIDDEN_SIZES
                trt_env = TrtllmDistEnv(group=dist.group.WORLD, device_id=rank)
            except Exception as exc:
                trt_init_note = f"TRT init failed: {exc}"
        if any(spec.name == "vllm_custom" for spec in args.backend_specs):
            try:
                from rtp_llm.models_py.modules.base.rocm.vllm_custom_allreduce import (
                    RocmVllmCustomAllReduce,
                )

                vllm_custom = RocmVllmCustomAllReduce(metadata_group, device=rank)
            except Exception as exc:
                vllm_init_note = f"vLLM custom init failed: {exc}"
        quick_reduce_specs = [
            spec for spec in args.backend_specs if spec.quantization is not None
        ]
        if quick_reduce_specs:
            try:
                from rtp_llm.models_py.modules.base.rocm.quick_reduce import (
                    RocmQuickReduce,
                )

                for spec in quick_reduce_specs:
                    try:
                        quick_reduce_managers[spec.name] = RocmQuickReduce(
                            metadata_group,
                            device=rank,
                            quantization=spec.quantization,
                            min_size_mb=args.quick_reduce_min_size_mb,
                            max_size_mb=args.quick_reduce_max_size_mb,
                            quantization_min_size_kb=args.quick_reduce_quantization_min_size_kb,
                        )
                    except Exception as exc:
                        quick_reduce_init_notes[spec.name] = (
                            f"QuickReduce init failed: {exc}"
                        )
            except Exception as exc:
                for spec in quick_reduce_specs:
                    quick_reduce_init_notes[spec.name] = (
                        f"QuickReduce import failed: {exc}"
                    )
        for shape in args.bench_shapes:
            inp = make_input(rank, shape, args.dtype_value)
            for backend in args.backend_specs:
                if backend.name == "rccl":
                    times = time_backend(rccl_all_reduce, inp, args.warmup, args.iters)
                    rows.append(ok_row(backend.name, args, shape, times))
                    continue

                if backend.name == "trt":
                    if trt_init_note is not None:
                        rows.append(skip_row(backend.name, args, shape, trt_init_note))
                        continue
                    if shape.hidden_size not in trt_supported_hidden_sizes:
                        rows.append(
                            skip_row(
                                backend.name, args, shape, "hidden size unsupported"
                            )
                        )
                        continue
                    if trt_env is None or trt_env.handle is None:
                        rows.append(
                            skip_row(backend.name, args, shape, "TRT init disabled")
                        )
                        continue

                    ref = rccl_all_reduce(inp)

                    def trt_fn(tensor):
                        out = torch.empty_like(tensor)
                        trt_env.allreduce_op(tensor, out)
                        return out

                    if not args.disable_correctness_check:
                        error = assert_close_to_ref(backend.name, trt_fn(inp), ref)
                        if error is not None:
                            rows.append(fail_row(backend.name, args, shape, error))
                            continue
                    times = time_backend(trt_fn, inp, args.warmup, args.iters)
                    rows.append(ok_row(backend.name, args, shape, times))
                    continue

                if backend.name == "vllm_custom":
                    if vllm_init_note is not None:
                        rows.append(skip_row(backend.name, args, shape, vllm_init_note))
                        continue
                    if vllm_custom is None or getattr(vllm_custom, "disabled", True):
                        rows.append(
                            skip_row(
                                backend.name,
                                args,
                                shape,
                                "vLLM custom init disabled",
                            )
                        )
                        continue
                    if not vllm_custom.should_custom_ar(inp):
                        rows.append(skip_row(backend.name, args, shape, "ineligible"))
                        continue

                    ref = rccl_all_reduce(inp)

                    def vllm_fn(tensor):
                        return vllm_custom.custom_all_reduce(tensor)

                    if not args.disable_correctness_check:
                        error = assert_close_to_ref(backend.name, vllm_fn(inp), ref)
                        if error is not None:
                            rows.append(fail_row(backend.name, args, shape, error))
                            continue
                    times = time_backend(vllm_fn, inp, args.warmup, args.iters)
                    rows.append(ok_row(backend.name, args, shape, times))
                    continue

                if backend.quantization is not None:
                    if backend.name in quick_reduce_init_notes:
                        rows.append(
                            skip_row(
                                backend.name,
                                args,
                                shape,
                                quick_reduce_init_notes[backend.name],
                            )
                        )
                        continue
                    manager = quick_reduce_managers.get(backend.name)
                    if manager is None or getattr(manager, "disabled", True):
                        rows.append(
                            skip_row(
                                backend.name,
                                args,
                                shape,
                                "QuickReduce init disabled",
                            )
                        )
                        continue
                    if not manager.should_quick_allreduce(inp):
                        rows.append(skip_row(backend.name, args, shape, "ineligible"))
                        continue

                    ref = rccl_all_reduce(inp)

                    def quick_reduce_fn(tensor):
                        return manager.quick_all_reduce(tensor)

                    if not args.disable_correctness_check:
                        error = assert_quick_reduce_close(
                            backend.name, quick_reduce_fn(inp), ref
                        )
                        if error is not None:
                            rows.append(fail_row(backend.name, args, shape, error))
                            continue
                    times = time_backend(quick_reduce_fn, inp, args.warmup, args.iters)
                    rows.append(ok_row(backend.name, args, shape, times))
                    continue
        return_dict[rank] = [asdict(row) for row in rows]
    finally:
        for manager in quick_reduce_managers.values():
            manager.close()
        if vllm_custom is not None:
            vllm_custom.close()
        cleanup_distributed(metadata_group)


def run_distributed(args) -> List[ResultRow]:
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm/CUDA is not available")
    if torch.cuda.device_count() < args.world_size:
        raise RuntimeError(
            f"requires {args.world_size} GPUs, found {torch.cuda.device_count()}"
        )
    port = find_free_port()
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    return_dict = manager.dict()
    procs = [
        ctx.Process(
            target=worker, args=(rank, args.world_size, port, args, return_dict)
        )
        for rank in range(args.world_size)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=300)
    for proc in procs:
        if proc.exitcode != 0:
            raise RuntimeError(f"rank process exited with code {proc.exitcode}")
    rank0_rows = return_dict.get(0, [])
    return [ResultRow(**row) for row in rank0_rows]


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.world_size < 2:
        raise ValueError("--world-size must be at least 2")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.dry_run:
        print(format_result_header())
        for shape in args.bench_shapes:
            print(f"DRY-RUN {shape.target} {shape.shape_label} {shape.actual_bytes}")
        return 0
    rows = run_distributed(args)
    print(format_result_header())
    for row in rows:
        print(format_result_row(row))
    write_results_json(
        args={
            "backends": args.backends,
            "world_size": args.world_size,
            "dtype": args.dtype,
            "hidden_sizes": args.hidden_sizes,
            "one_token_hidden_size": args.one_token_hidden_size,
            "max_bytes_mb": args.max_bytes_mb,
            "byte_targets": args.byte_targets,
            "shapes": args.shapes,
            "warmup": args.warmup,
            "iters": args.iters,
            "quick_reduce_min_size_mb": args.quick_reduce_min_size_mb,
            "quick_reduce_max_size_mb": args.quick_reduce_max_size_mb,
            "quick_reduce_quantization_min_size_kb": args.quick_reduce_quantization_min_size_kb,
        },
        rows=rows,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

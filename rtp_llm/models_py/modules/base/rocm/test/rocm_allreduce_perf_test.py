import argparse
import sys

import torch
from rocm_allreduce_perf_lib import (
    DEFAULT_BACKENDS,
    MB,
    format_result_header,
    generate_shapes,
    parse_backends,
    parse_int_csv,
    parse_shapes,
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
    raise RuntimeError("distributed benchmark worker is absent")


if __name__ == "__main__":
    sys.exit(main())

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

KB = 1024
MB = 1024 * KB

DEFAULT_BACKENDS = [
    "rccl",
    "trt",
    "vllm_custom",
    "quick_reduce_fp",
    "quick_reduce_int8",
    "quick_reduce_int6",
    "quick_reduce_int4",
]

QUICK_REDUCE_QUANTIZATION = {
    "quick_reduce_fp": "FP",
    "quick_reduce_int8": "INT8",
    "quick_reduce_int6": "INT6",
    "quick_reduce_int4": "INT4",
}


@dataclass(frozen=True)
class BenchShape:
    target: str
    rows: int
    hidden_size: int
    dtype_size: int

    @property
    def actual_bytes(self) -> int:
        return self.rows * self.hidden_size * self.dtype_size

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.rows, self.hidden_size)

    @property
    def shape_label(self) -> str:
        return f"{self.rows}x{self.hidden_size}"


@dataclass(frozen=True)
class BackendSpec:
    name: str
    quantization: Optional[str]


@dataclass
class ResultRow:
    backend: str
    dtype: str
    target: str
    shape: str
    bytes: int
    status: str
    avg_us: Optional[float]
    p50_us: Optional[float]
    p90_us: Optional[float]
    min_us: Optional[float]
    max_us: Optional[float]
    algbw_GBps: Optional[float]
    note: str


def parse_byte_target(value: str) -> Tuple[str, int]:
    text = value.strip()
    upper = text.upper()
    if upper.endswith("_TOKEN"):
        count_text = upper[: -len("_TOKEN")]
        count = int(count_text)
        if count <= 0:
            raise ValueError(f"token target must be positive: {value}")
        return ("token", count)
    if upper.endswith("KB"):
        count = int(upper[: -len("KB")])
        if count <= 0:
            raise ValueError(f"KB target must be positive: {value}")
        return ("bytes", count * KB)
    if upper.endswith("MB"):
        count = int(upper[: -len("MB")])
        if count <= 0:
            raise ValueError(f"MB target must be positive: {value}")
        return ("bytes", count * MB)
    raise ValueError(f"unsupported byte target: {value}")


def parse_int_csv(value: str) -> List[int]:
    result = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        parsed = int(text)
        if parsed <= 0:
            raise ValueError(f"integer values must be positive: {value}")
        result.append(parsed)
    if not result:
        raise ValueError("at least one integer value is required")
    return result


def parse_shapes(value: str) -> List[Tuple[int, int, str]]:
    result: List[Tuple[int, int, str]] = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        if "x" not in text.lower():
            raise ValueError(f"shape must use ROWSxHIDDEN format: {text}")
        lhs, rhs = text.lower().split("x", 1)
        rows = int(lhs)
        hidden_size = int(rhs)
        if rows <= 0 or hidden_size <= 0:
            raise ValueError(f"shape dimensions must be positive: {text}")
        result.append((rows, hidden_size, f"{rows}x{hidden_size}"))
    if not result:
        raise ValueError("at least one explicit shape is required")
    return result


def unique_shapes(shapes: Iterable[BenchShape]) -> List[BenchShape]:
    seen = set()
    result: List[BenchShape] = []
    for shape in shapes:
        key = (shape.target, shape.rows, shape.hidden_size, shape.dtype_size)
        if key in seen:
            continue
        seen.add(key)
        result.append(shape)
    return result


def generate_shapes(
    byte_targets: Sequence[str],
    hidden_sizes: Sequence[int],
    dtype_size: int,
    one_token_hidden_size: int,
    max_bytes: int,
    explicit_shapes: Optional[Sequence[Tuple[int, int, str]]] = None,
) -> List[BenchShape]:
    if dtype_size <= 0:
        raise ValueError("dtype_size must be positive")
    if one_token_hidden_size <= 0:
        raise ValueError("one_token_hidden_size must be positive")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if explicit_shapes is not None:
        return [
            BenchShape(
                target=label,
                rows=rows,
                hidden_size=hidden_size,
                dtype_size=dtype_size,
            )
            for rows, hidden_size, label in explicit_shapes
            if rows * hidden_size * dtype_size <= max_bytes
        ]

    shapes: List[BenchShape] = []
    for target in byte_targets:
        target_kind, target_value = parse_byte_target(target)
        if target_kind == "token":
            shape = BenchShape(
                target=target,
                rows=target_value,
                hidden_size=one_token_hidden_size,
                dtype_size=dtype_size,
            )
            if shape.actual_bytes <= max_bytes:
                shapes.append(shape)
            continue

        for hidden_size in hidden_sizes:
            if hidden_size <= 0:
                raise ValueError("hidden sizes must be positive")
            rows = max(1, math.ceil(target_value / (hidden_size * dtype_size)))
            shape = BenchShape(
                target=target,
                rows=rows,
                hidden_size=hidden_size,
                dtype_size=dtype_size,
            )
            if shape.actual_bytes <= max_bytes:
                shapes.append(shape)
    return unique_shapes(shapes)


def parse_backends(value: str) -> List[BackendSpec]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("at least one backend is required")
    valid = set(DEFAULT_BACKENDS)
    result: List[BackendSpec] = []
    for name in names:
        if name not in valid:
            raise ValueError(f"unsupported backend: {name}")
        result.append(
            BackendSpec(
                name=name,
                quantization=QUICK_REDUCE_QUANTIZATION.get(name),
            )
        )
    return result


def percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute percentile of empty values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize_us(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        raise ValueError("at least one timing value is required")
    sorted_values = sorted(float(value) for value in values)
    return {
        "avg_us": sum(sorted_values) / len(sorted_values),
        "p50_us": percentile(sorted_values, 0.5),
        "p90_us": percentile(sorted_values, 0.9),
        "min_us": sorted_values[0],
        "max_us": sorted_values[-1],
    }


def _format_float(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def format_result_header() -> str:
    return (
        f"{'backend':<20} {'dtype':<7} {'target':<10} {'shape':<12} "
        f"{'bytes':>10} {'status':<6} {'avg_us':>10} {'p50_us':>10} "
        f"{'p90_us':>10} {'min_us':>10} {'max_us':>10} {'algbw_GBps':>12} note"
    )


def format_result_row(row: ResultRow) -> str:
    return (
        f"{row.backend:<20} {row.dtype:<7} {row.target:<10} {row.shape:<12} "
        f"{row.bytes:>10} {row.status:<6} {_format_float(row.avg_us):>10} "
        f"{_format_float(row.p50_us):>10} {_format_float(row.p90_us):>10} "
        f"{_format_float(row.min_us):>10} {_format_float(row.max_us):>10} "
        f"{_format_float(row.algbw_GBps):>12} {row.note}"
    )


def write_results_json(
    args: Dict[str, Any], rows: Sequence[ResultRow]
) -> Optional[str]:
    output_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if not output_dir:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "rocm_allreduce_perf_results.json")
    payload = {
        "args": args,
        "rows": [asdict(row) for row in rows],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path

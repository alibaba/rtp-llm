import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

KB = 1024
MB = 1024 * KB


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

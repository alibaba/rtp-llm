"""DeepGEMM warmup planning and execution for the internal PPU backend."""

import fcntl
import logging
import math
import os
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Literal, Optional, cast

import torch

from .deepgemm_wrapper import (
    Int8ScaledTensor,
    configure_deep_gemm_num_sms,
    deep_gemm_default_num_sms,
    has_deep_gemm,
    int8_gemm_nt,
    m_grouped_bf16_gemm_nt_contiguous,
    m_grouped_bf16_gemm_nt_masked,
    m_grouped_bf16_gemm_nt_nopad,
    m_grouped_int8_gemm_nt_contiguous,
    m_grouped_int8_gemm_nt_masked,
    m_grouped_int8_gemm_nt_nopad,
)

logger = logging.getLogger(__name__)

WarmupMode = Literal["skip", "relax", "full"]
GroupedLayout = Literal["masked", "contiguous", "nopad"]
WarmupLayout = Literal["dense", "masked", "contiguous", "nopad"]
WarmupDtype = Literal["int8", "bf16"]

WARMUP_MODE_ENV = "RTP_LLM_PPU_DEEPGEMM_WARMUP"
WARMUP_MAX_TOKENS_ENV = "RTP_LLM_PPU_DEEPGEMM_WARMUP_MAX_TOKENS"
DEFAULT_WARMUP_MAX_TOKENS = 8192

_WARMUP_LOCK_FILE = "rtp_llm_deepgemm_warmup.lock"
_warmup_cache: set[tuple[Any, ...]] = set()
_graph_warmup_cache: set[tuple[Any, ...]] = set()


@dataclass(frozen=True)
class DeepGemmWarmupSpec:
    """Kernel shape family whose JIT configurations should be prepared."""

    layout: WarmupLayout
    dtype: WarmupDtype
    n: int
    k: int
    num_groups: int
    max_m: int
    num_sms: int
    exact_m_values: tuple[int, ...] = ()


def get_deep_gemm_warmup_mode() -> WarmupMode:
    mode = os.environ.get(WARMUP_MODE_ENV, "relax")
    if mode not in ("skip", "relax", "full"):
        raise ValueError(
            f"{WARMUP_MODE_ENV} must be skip, relax, or full; got {mode!r}"
        )
    return cast(WarmupMode, mode)


def resolve_deep_gemm_warmup_max_tokens(
    model_max_tokens: Optional[int] = None,
    *,
    default: int = DEFAULT_WARMUP_MAX_TOKENS,
) -> int:
    """Resolve the token bound, allowing an explicit environment override."""
    configured = os.environ.get(WARMUP_MAX_TOKENS_ENV)
    if configured is not None:
        try:
            max_tokens = int(configured)
        except ValueError as error:
            raise ValueError(
                f"{WARMUP_MAX_TOKENS_ENV} must be a positive integer; "
                f"got {configured!r}"
            ) from error
    elif model_max_tokens is not None:
        max_tokens = min(int(model_max_tokens), default)
    else:
        max_tokens = default
    if max_tokens <= 0:
        raise ValueError(
            f"{WARMUP_MAX_TOKENS_ENV} must resolve to a positive integer; "
            f"got {max_tokens}"
        )
    return max_tokens


def plan_deep_gemm_warmup_m_values(
    spec: DeepGemmWarmupSpec,
    mode: WarmupMode,
) -> tuple[int, ...]:
    """Select M values covering DeepGEMM block and SM-wave transitions.

    The relaxed policy follows DeepGEMM's heuristic dimensions, as used by
    vLLM's PPU warmup: small decode shapes, every candidate block-M boundary,
    and the first ten SM-wave transitions for each block-M/block-N pair.
    """
    if mode == "skip" or spec.max_m <= 0:
        return ()
    if min(spec.n, spec.k, spec.num_groups, spec.num_sms) <= 0:
        raise ValueError(f"Invalid DeepGEMM warmup spec: {spec}")
    if mode == "full":
        return tuple(range(1, spec.max_m + 1))
    if mode != "relax":
        raise ValueError(f"Unsupported DeepGEMM warmup mode: {mode}")

    values = {1, 2, 4, spec.max_m}
    values.update(range(8, min(spec.max_m, 64) + 1, 8))
    block_ms = (64, 128, 256)
    block_ns = range(16, min(256, spec.n) + 1, 16)
    for block_m in block_ms:
        values.update(range(block_m, spec.max_m + 1, block_m))
        for block_n in block_ns:
            n_blocks = math.ceil(spec.n / block_n)
            for wave in range(1, 11):
                m = wave * spec.num_sms * block_m // n_blocks
                if 1 <= m <= spec.max_m:
                    values.add(m)

    values.update(m for m in spec.exact_m_values if 1 <= m <= spec.max_m)
    return tuple(sorted(values))


def _resolve_num_sms(rhs: torch.Tensor, num_sms: Optional[int]) -> int:
    """Resolve the grid width to compile for.

    Callers should pass the same value they execute with -- ``num_sms`` is not
    part of DeepGEMM's JIT key but the tile config derived from it is, so a
    warmup/execution mismatch compiles kernels the forward pass never looks up.
    ``None`` falls back to :func:`deep_gemm_default_num_sms`, the shared
    default, rather than to ``multi_processor_count``.
    """
    if num_sms is not None:
        if num_sms <= 0:
            raise ValueError(f"num_sms must be positive, got {num_sms}")
        return num_sms
    resolved = deep_gemm_default_num_sms()
    if resolved is None:
        return torch.cuda.get_device_properties(rhs.device).multi_processor_count
    return resolved


@contextmanager
def deep_gemm_compile_only() -> Generator[None, None, None]:
    """Compile DeepGEMM kernels without launching them."""
    if not has_deep_gemm():
        raise RuntimeError("deep_gemm is unavailable on PPU")
    import deep_gemm

    get_compile_mode = getattr(deep_gemm, "get_compile_mode", None)
    set_compile_mode = getattr(deep_gemm, "set_compile_mode", None)
    if get_compile_mode is None or set_compile_mode is None:
        raise RuntimeError("PPU deep_gemm wheel does not support compile-only mode")
    original_mode = get_compile_mode()
    set_compile_mode(1)
    try:
        yield
    finally:
        set_compile_mode(original_mode)


def _deep_gemm_cache_root() -> Path:
    root = os.environ.get("DG_CACHE_DIR") or os.environ.get("DG_JIT_CACHE_DIR")
    return Path(root).expanduser() if root else Path.home() / ".deep_gemm"


@contextmanager
def _deep_gemm_compile_lock() -> Generator[None, None, None]:
    """Serialize JIT writers sharing one DeepGEMM cache directory."""
    cache_root = _deep_gemm_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    with (cache_root / _WARMUP_LOCK_FILE).open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def generate_graph_capture_m_values(
    capture_batch_sizes: Iterable[int],
    *,
    top_k: int,
    gen_num_per_cycle: int = 0,
    max_generate_batch_size: Optional[int] = None,
) -> tuple[int, ...]:
    """Return exact compact MoE row counts used by decode graph capture."""
    batches = tuple(sorted({int(value) for value in capture_batch_sizes if value > 0}))
    if not batches:
        max_batch = max(1, int(max_generate_batch_size or 128))
        default_batches = [value for value in (1, 8, 16, 24, 32) if value <= max_batch]
        default_batches.extend(range(48, max_batch + 1, 16))
        if default_batches[-1] != max_batch:
            default_batches.append(max_batch)
        batches = tuple(default_batches)
    multipliers = (1,) if gen_num_per_cycle <= 0 else (1, gen_num_per_cycle + 1)
    return tuple(
        sorted(
            {
                batch * multiplier * top_k
                for batch in batches
                for multiplier in multipliers
            }
        )
    )


def make_contiguous_group_metadata(
    m: int,
    num_groups: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build balanced contiguous expert blocks for eager graph warmup."""
    rows_per_group, remainder = divmod(m, num_groups)
    m_rows = torch.full((num_groups,), rows_per_group, device=device, dtype=torch.int32)
    if remainder:
        m_rows[:remainder] += 1
    m_indices = torch.repeat_interleave(
        torch.arange(num_groups, device=device, dtype=torch.int32),
        m_rows.to(torch.int64),
    )
    return m_indices, m_rows


def is_graph_warmed(key: tuple[Any, ...]) -> bool:
    return key in _graph_warmup_cache


def mark_graph_warmed(key: tuple[Any, ...]) -> None:
    _graph_warmup_cache.add(key)


def _warmup_cases(
    spec: DeepGemmWarmupSpec,
    mode: WarmupMode,
) -> tuple[int, ...]:
    """Return shapes to JIT with DeepGEMM's own config selection."""
    return plan_deep_gemm_warmup_m_values(spec, mode)


def _storage_view(
    source: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a compile-only tensor by aliasing contiguous weight storage."""
    if not source.is_contiguous():
        raise ValueError("DeepGEMM warmup requires contiguous weight storage")
    typed_source = source if source.dtype == dtype else source.view(dtype)
    required = math.prod(shape)
    if required > typed_source.numel():
        raise RuntimeError(
            "DeepGEMM warmup shape does not fit reusable weight storage: "
            f"shape={shape}, dtype={dtype}, available={typed_source.numel()}"
        )
    return typed_source.reshape(-1)[:required].view(shape)


def _cap_spec_to_storage(
    spec: DeepGemmWarmupSpec,
    max_storage_m: int,
) -> DeepGemmWarmupSpec:
    capped_max_m = min(spec.max_m, max_storage_m)
    if capped_max_m < spec.max_m:
        logger.warning(
            "DeepGEMM %s/%s warmup capped M at %d to reuse weight storage; "
            "longer shapes will use lazy JIT",
            spec.dtype,
            spec.layout,
            max_storage_m,
        )
    return DeepGemmWarmupSpec(
        layout=spec.layout,
        dtype=spec.dtype,
        n=spec.n,
        k=spec.k,
        num_groups=spec.num_groups,
        max_m=capped_max_m,
        num_sms=spec.num_sms,
        exact_m_values=tuple(m for m in spec.exact_m_values if m <= capped_max_m),
    )


def _warmup_grouped_gemm(
    rhs: torch.Tensor,
    rhs_scale: Optional[torch.Tensor],
    *,
    max_m: int,
    layout: GroupedLayout,
    dtype: WarmupDtype,
    mode: WarmupMode,
    num_sms: Optional[int],
    exact_m_values: tuple[int, ...] = (),
) -> int:
    if rhs.ndim != 3:
        raise ValueError(f"Grouped DeepGEMM weight must be 3D, got {rhs.shape}")
    num_groups, n, k = rhs.shape
    resolved_num_sms = _resolve_num_sms(rhs, num_sms)
    spec = DeepGemmWarmupSpec(
        layout=layout,
        dtype=dtype,
        n=n,
        k=k,
        num_groups=num_groups,
        max_m=max_m,
        num_sms=resolved_num_sms,
        exact_m_values=exact_m_values,
    )
    max_storage_m = min(n, k if dtype == "bf16" else k // 2)
    if layout != "masked":
        max_storage_m *= num_groups
    spec = _cap_spec_to_storage(spec, max_storage_m)
    cases = _warmup_cases(spec, mode)
    cache_key = (
        spec.dtype,
        spec.layout,
        tuple(rhs.shape),
        cases,
        spec.num_sms,
    )
    if not cases or cache_key in _warmup_cache:
        return 0

    start = time.monotonic()
    with _deep_gemm_compile_lock(), deep_gemm_compile_only():
        with configure_deep_gemm_num_sms(spec.num_sms):
            for m in cases:
                if layout == "masked":
                    lhs_shape = (num_groups, m, k)
                    out_shape = (num_groups, m, n)
                    metadata_shape = (num_groups,)
                else:
                    lhs_shape = (m, k)
                    out_shape = (m, n)
                    metadata_shape = (m,)

                output = _storage_view(rhs, out_shape, torch.bfloat16)
                metadata = _storage_view(rhs, metadata_shape, torch.int32)
                if dtype == "int8":
                    if rhs_scale is None:
                        raise ValueError("INT8 DeepGEMM warmup requires weight scales")
                    lhs = (
                        _storage_view(rhs, lhs_shape, torch.int8),
                        _storage_view(rhs_scale, (*lhs_shape[:-1], 1), torch.float32),
                    )
                    scaled_rhs = (rhs, rhs_scale)
                    if layout == "masked":
                        m_grouped_int8_gemm_nt_masked(
                            lhs, scaled_rhs, output, metadata, m
                        )
                    elif layout == "nopad":
                        m_rows = _storage_view(rhs, (num_groups,), torch.int32)
                        m_grouped_int8_gemm_nt_nopad(
                            lhs, scaled_rhs, output, metadata, m_rows
                        )
                    else:
                        m_grouped_int8_gemm_nt_contiguous(
                            lhs, scaled_rhs, output, metadata
                        )
                else:
                    lhs = _storage_view(rhs, lhs_shape, torch.bfloat16)
                    if layout == "masked":
                        m_grouped_bf16_gemm_nt_masked(lhs, rhs, output, metadata, m)
                    elif layout == "nopad":
                        m_rows = _storage_view(rhs, (num_groups,), torch.int32)
                        m_grouped_bf16_gemm_nt_nopad(lhs, rhs, output, metadata, m_rows)
                    else:
                        m_grouped_bf16_gemm_nt_contiguous(lhs, rhs, output, metadata)

    _warmup_cache.add(cache_key)
    logger.info(
        "DeepGEMM %s/%s warmup covered %d M shape(s) for G/N/K=%d/%d/%d in %.2fs",
        dtype,
        layout,
        len(cases),
        num_groups,
        n,
        k,
        time.monotonic() - start,
    )
    return len(cases)


def warmup_grouped_int8_gemm(
    rhs: Int8ScaledTensor,
    *,
    max_m: int,
    layout: GroupedLayout,
    mode: WarmupMode,
    num_sms: Optional[int] = None,
    exact_m_values: tuple[int, ...] = (),
) -> int:
    return _warmup_grouped_gemm(
        rhs[0],
        rhs[1],
        max_m=max_m,
        layout=layout,
        dtype="int8",
        mode=mode,
        num_sms=num_sms,
        exact_m_values=exact_m_values,
    )


def warmup_grouped_bf16_gemm(
    rhs: torch.Tensor,
    *,
    max_m: int,
    layout: GroupedLayout,
    mode: WarmupMode,
    num_sms: Optional[int] = None,
    exact_m_values: tuple[int, ...] = (),
) -> int:
    return _warmup_grouped_gemm(
        rhs,
        None,
        max_m=max_m,
        layout=layout,
        dtype="bf16",
        mode=mode,
        num_sms=num_sms,
        exact_m_values=exact_m_values,
    )


def warmup_dense_int8_gemm(
    rhs: Int8ScaledTensor,
    *,
    max_m: int,
    mode: WarmupMode,
    num_sms: Optional[int] = None,
    exact_m_values: tuple[int, ...] = (),
) -> int:
    weight, weight_scale = rhs
    if weight.ndim != 2:
        raise ValueError(f"Dense DeepGEMM weight must be 2D, got {weight.shape}")
    n, k = weight.shape
    resolved_num_sms = _resolve_num_sms(weight, num_sms)
    spec = _cap_spec_to_storage(
        DeepGemmWarmupSpec(
            layout="dense",
            dtype="int8",
            n=n,
            k=k,
            num_groups=1,
            max_m=max_m,
            num_sms=resolved_num_sms,
            exact_m_values=exact_m_values,
        ),
        min(n, k // 2),
    )
    cases = _warmup_cases(spec, mode)
    cache_key = (
        spec.dtype,
        spec.layout,
        tuple(weight.shape),
        cases,
        spec.num_sms,
    )
    if not cases or cache_key in _warmup_cache:
        return 0

    start = time.monotonic()
    with _deep_gemm_compile_lock(), deep_gemm_compile_only():
        with configure_deep_gemm_num_sms(spec.num_sms):
            for m in cases:
                lhs = (
                    _storage_view(weight, (m, k), torch.int8),
                    _storage_view(weight_scale, (m, 1), torch.float32),
                )
                output = _storage_view(weight, (m, n), torch.bfloat16)
                int8_gemm_nt(lhs, rhs, output)

    _warmup_cache.add(cache_key)
    logger.info(
        "DeepGEMM int8/dense warmup covered %d M shape(s) for N/K=%d/%d in %.2fs",
        len(cases),
        n,
        k,
        time.monotonic() - start,
    )
    return len(cases)

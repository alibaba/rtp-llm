"""PPU DeepGEMM BF16 and W8A8 INT8 GEMM bindings.

The PPU wheel exposes the legacy ``m_grouped_gemm_bf16_bf16_bf16_nt_*``
symbols and does not accept CUDA's ``compiled_dims`` keyword. Keeping this
adapter in the PPU kernel layer avoids rewriting the shared CUDA wrapper at
runtime or coupling it to a specific linear/MoE executor.

RTP-LLM deliberately does not maintain a second, deployment-specific config
LUT. Passing ``None`` delegates config selection to the DeepGEMM wheel's own
LUT/heuristic before its JIT cache is populated by warmup or normal execution.
"""

import importlib.util
import logging
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

Int8ScaledTensor = Tuple[torch.Tensor, torch.Tensor]

_contiguous_impl: Optional[Callable[..., Any]] = None
_masked_impl: Optional[Callable[..., Any]] = None
_nopad_impl: Optional[Callable[..., Any]] = None
_int8_dense_impl: Optional[Callable[..., Any]] = None
_int8_contiguous_impl: Optional[Callable[..., Any]] = None
_int8_masked_impl: Optional[Callable[..., Any]] = None
_int8_nopad_impl: Optional[Callable[..., Any]] = None
_initialized = False
_available: Optional[bool] = None


def has_deep_gemm() -> bool:
    """Return whether ``deep_gemm`` is importable, retrying until it appears."""
    global _available
    if _available is True:
        return True
    available = importlib.util.find_spec("deep_gemm") is not None
    if available:
        _available = True
    return available


def _resolve(module: Any, *names: str) -> Optional[Callable[..., Any]]:
    for name in names:
        symbol = getattr(module, name, None)
        if symbol is not None:
            return symbol
    return None


def _ensure_initialized() -> None:
    global _contiguous_impl, _masked_impl, _nopad_impl
    global _int8_dense_impl, _int8_contiguous_impl, _int8_masked_impl
    global _int8_nopad_impl, _initialized
    if _initialized or not has_deep_gemm():
        return

    import deep_gemm

    _contiguous_impl = _resolve(
        deep_gemm,
        "m_grouped_bf16_gemm_nt_contiguous",
        "m_grouped_gemm_bf16_bf16_bf16_nt_contiguous",
    )
    _masked_impl = _resolve(
        deep_gemm,
        "m_grouped_bf16_gemm_nt_masked",
        "m_grouped_gemm_bf16_bf16_bf16_nt_masked",
    )
    _nopad_impl = _resolve(
        deep_gemm,
        "m_grouped_bf16_gemm_nt_nopad",
        "m_grouped_gemm_bf16_bf16_bf16_nt_nopad",
    )
    _int8_dense_impl = _resolve(
        deep_gemm,
        "gemm_int8_int8_bf16_nt",
        "gemm_nt_i8i8bf16",
    )
    _int8_contiguous_impl = _resolve(
        deep_gemm,
        "m_grouped_gemm_int8_int8_bf16_nt_contiguous",
        "m_grouped_int8_gemm_nt_contiguous",
    )
    _int8_masked_impl = _resolve(
        deep_gemm,
        "m_grouped_gemm_int8_int8_bf16_nt_masked",
        "m_grouped_int8_gemm_nt_masked",
    )
    _int8_nopad_impl = _resolve(
        deep_gemm,
        "m_grouped_gemm_int8_int8_bf16_nt_nopad",
        "m_grouped_int8_gemm_nt_nopad",
    )
    _initialized = True


def has_deep_gemm_bf16_grouped() -> bool:
    if not has_deep_gemm():
        return False
    try:
        _ensure_initialized()
    except Exception:
        return False
    return _contiguous_impl is not None and _masked_impl is not None


def has_deep_gemm_bf16_grouped_nopad() -> bool:
    if not has_deep_gemm():
        return False
    try:
        _ensure_initialized()
    except Exception:
        return False
    return _nopad_impl is not None


def has_deep_gemm_int8_dense() -> bool:
    if not has_deep_gemm():
        return False
    try:
        _ensure_initialized()
    except Exception:
        return False
    return _int8_dense_impl is not None


def has_deep_gemm_int8_grouped() -> bool:
    if not has_deep_gemm():
        return False
    try:
        _ensure_initialized()
    except Exception:
        return False
    return _int8_masked_impl is not None and (
        _int8_nopad_impl is not None or _int8_contiguous_impl is not None
    )


def has_deep_gemm_int8_grouped_masked() -> bool:
    if not has_deep_gemm():
        return False
    try:
        _ensure_initialized()
    except Exception:
        return False
    return _int8_masked_impl is not None


def has_deep_gemm_int8_grouped_nopad() -> bool:
    if not has_deep_gemm():
        return False
    try:
        _ensure_initialized()
    except Exception:
        return False
    return _int8_nopad_impl is not None


def _require(symbol: Optional[Callable[..., Any]], name: str) -> Callable[..., Any]:
    if symbol is None:
        raise RuntimeError(f"PPU DeepGEMM symbol {name} is unavailable")
    return symbol


def _grouped_shape(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    masked: bool,
) -> tuple[int, int, int, int]:
    if masked:
        num_groups, _, k = lhs.shape
        rhs_groups, n, rhs_k = rhs.shape
    else:
        m, k = lhs.shape
        rhs_groups, n, rhs_k = rhs.shape
        num_groups = rhs_groups
    if rhs_groups != num_groups or rhs_k != k:
        raise ValueError(
            f"DeepGEMM grouped shape mismatch: lhs={lhs.shape}, rhs={rhs.shape}"
        )
    return (0 if masked else m), n, k, num_groups


def deep_gemm_default_num_sms() -> Optional[int]:
    """The wheel's own grid width for this device, or None if unavailable.

    This is the single source of ``num_sms`` for both warmup and execution, and
    it takes no arguments on purpose: there is no shape table, no M table and no
    tuning knob, which is exactly what the reference upstream integrations do on
    this accelerator (they never override the wheel's value either).

    DeepGEMM feeds ``num_sms`` to two places: the tile heuristic
    (``get_best_configs``) and the persistent-CTA grid of the launch itself. The
    wheel reports a value well below ``multi_processor_count`` here, and that is
    what its own tuner runs with -- ``perf_results/deepgemm_num_sms_sweep_20260802``
    measured that forcing the full device grid instead costs 1.2-1.5x on every
    large-M shape. ``None`` (no ``deep_gemm``, or no CUDA context yet) makes
    :func:`configure_deep_gemm_num_sms` a no-op, which is also correct.
    """
    if not has_deep_gemm():
        return None
    import deep_gemm

    getter = getattr(deep_gemm, "get_num_sms", None)
    if getter is None:
        return None
    try:
        return int(getter())
    except Exception:
        # Querying the device can fail before a CUDA context exists (CPU-only
        # unit tests). Fall back to letting the wheel choose per launch.
        logger.debug("deep_gemm.get_num_sms() failed; leaving num_sms unset")
        return None


@contextmanager
def configure_deep_gemm_num_sms(
    num_sms: Optional[int],
) -> Generator[None, None, None]:
    if num_sms is None:
        yield
        return
    if not has_deep_gemm():
        raise RuntimeError("deep_gemm is unavailable on PPU")
    import deep_gemm

    original_num_sms = deep_gemm.get_num_sms()
    deep_gemm.set_num_sms(num_sms)
    try:
        yield
    finally:
        deep_gemm.set_num_sms(original_num_sms)


def m_grouped_bf16_gemm_nt_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    m_indices: torch.Tensor,
) -> None:
    _ensure_initialized()
    impl = _require(_contiguous_impl, "m_grouped_bf16_gemm_nt_contiguous")
    _grouped_shape(a, b, masked=False)
    impl(a, b, output, m_indices, None)


def m_grouped_bf16_gemm_nt_masked(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
) -> None:
    _ensure_initialized()
    impl = _require(_masked_impl, "m_grouped_bf16_gemm_nt_masked")
    _grouped_shape(a, b, masked=True)
    impl(a, b, output, masked_m, expected_m, None)


def m_grouped_bf16_gemm_nt_nopad(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor,
) -> None:
    _ensure_initialized()
    impl = _require(_nopad_impl, "m_grouped_bf16_gemm_nt_nopad")
    _grouped_shape(a, b, masked=False)
    impl(a, b, output, m_indices, m_rows, None)


def int8_gemm_nt(
    lhs: Int8ScaledTensor,
    rhs: Int8ScaledTensor,
    output: torch.Tensor,
) -> None:
    _ensure_initialized()
    impl = _require(_int8_dense_impl, "gemm_int8_int8_bf16_nt")
    k = lhs[0].shape[1]
    rhs_k = rhs[0].shape[1]
    if rhs_k != k:
        raise ValueError(
            f"DeepGEMM dense shape mismatch: lhs={lhs[0].shape}, " f"rhs={rhs[0].shape}"
        )
    impl(lhs, rhs, output, None)


def m_grouped_int8_gemm_nt_contiguous(
    lhs: Int8ScaledTensor,
    rhs: Int8ScaledTensor,
    output: torch.Tensor,
    m_indices: torch.Tensor,
) -> None:
    _ensure_initialized()
    impl = _require(
        _int8_contiguous_impl,
        "m_grouped_gemm_int8_int8_bf16_nt_contiguous",
    )
    _grouped_shape(lhs[0], rhs[0], masked=False)
    impl(lhs, rhs, output, m_indices, None)


def m_grouped_int8_gemm_nt_masked(
    lhs: Int8ScaledTensor,
    rhs: Int8ScaledTensor,
    output: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
) -> None:
    _ensure_initialized()
    impl = _require(
        _int8_masked_impl,
        "m_grouped_gemm_int8_int8_bf16_nt_masked",
    )
    _grouped_shape(lhs[0], rhs[0], masked=True)
    impl(lhs, rhs, output, masked_m, expected_m, None)


def m_grouped_int8_gemm_nt_nopad(
    lhs: Int8ScaledTensor,
    rhs: Int8ScaledTensor,
    output: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor,
) -> None:
    _ensure_initialized()
    impl = _require(
        _int8_nopad_impl,
        "m_grouped_gemm_int8_int8_bf16_nt_nopad",
    )
    _grouped_shape(lhs[0], rhs[0], masked=False)
    impl(lhs, rhs, output, m_indices, m_rows, None)

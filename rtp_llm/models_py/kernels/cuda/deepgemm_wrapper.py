import functools
from contextlib import contextmanager
from typing import Any, Callable, Generator, List, NoReturn, Optional, Tuple

import torch

from rtp_llm.utils.module_util import has_module, resolve_symbol

__all__ = [
    "fp8_gemm_nt",
    "m_grouped_fp8_gemm_nt_contiguous",
    "m_grouped_fp8_gemm_nt_masked",
    "bf16_gemm_nt",
    "m_grouped_bf16_gemm_nt_contiguous",
    "m_grouped_bf16_gemm_nt_masked",
    "has_deep_gemm",
    "is_deep_gemm_e8m0_used",
    "configure_deep_gemm_num_sms",
]

_deep_gemm_impl_new_map = {
    "fp8_gemm_nt": "fp8_gemm_nt",
    "m_grouped_fp8_gemm_nt_contiguous": "m_grouped_fp8_gemm_nt_contiguous",
    "m_grouped_fp8_gemm_nt_masked": "m_grouped_fp8_gemm_nt_masked",
    "bf16_gemm_nt": "bf16_gemm_nt",
    "m_grouped_bf16_gemm_nt_contiguous": "m_grouped_bf16_gemm_nt_contiguous",
    "m_grouped_bf16_gemm_nt_masked": "m_grouped_bf16_gemm_nt_masked",
}

_deep_gemm_impl_old_map = {
    "fp8_gemm_nt": "fp8_gemm_nt",
    "m_grouped_fp8_gemm_nt_contiguous": "m_grouped_fp8_gemm_nt_contiguous",
    "m_grouped_fp8_gemm_nt_masked": "fp8_m_grouped_gemm_nt_masked",
    "bf16_gemm_nt": "bf16_gemm_nt",
    "m_grouped_bf16_gemm_nt_contiguous": "m_grouped_bf16_gemm_nt_contiguous",
    "m_grouped_bf16_gemm_nt_masked": "m_grouped_bf16_gemm_nt_masked",
}


_fp8_gemm_nt_impl: Callable[..., Any] | None = None
_m_grouped_fp8_gemm_nt_contiguous_impl: Callable[..., Any] | None = None
_m_grouped_fp8_gemm_nt_masked_impl: Callable[..., Any] | None = None
_bf16_gemm_nt_impl: Callable[..., Any] | None = None
_m_grouped_bf16_gemm_nt_contiguous_impl: Callable[..., Any] | None = None
_m_grouped_bf16_gemm_nt_masked_impl: Callable[..., Any] | None = None


@functools.cache
def has_deep_gemm() -> bool:
    """Whether the optional `deep_gemm` package is available."""
    return has_module("deep_gemm")


@functools.cache
def is_deep_gemm_e8m0_used() -> bool:
    return torch.cuda.get_device_capability()[0] in [10, 12]


@contextmanager
def configure_deep_gemm_num_sms(num_sms: int) -> Generator[None, None, None]:
    """Configure the number of sms for deep gemm."""
    if not has_deep_gemm():
        raise RuntimeError(
            "DeepGEMM is not available. Please install the `deep_gemm` package to enable DeepGEMM kernels."
        )
    import deep_gemm

    # get original num sms
    original_num_sms = deep_gemm.get_num_sms()
    # set num sms
    deep_gemm.set_num_sms(num_sms)
    try:
        yield
    finally:
        # restore original num sms
        deep_gemm.set_num_sms(original_num_sms)


def _missing_deep_gemm() -> NoReturn:
    """Placeholder for unavailable DeepGEMM package."""
    raise RuntimeError(
        "DeepGEMM is not available. Please install the `deep_gemm` package to enable DeepGEMM kernels."
    )


def _lazy_init_deep_gemm(symbols: List[str]) -> None:
    """Import deep_gemm and resolve symbols on first use."""
    global _fp8_gemm_nt_impl, _m_grouped_fp8_gemm_nt_contiguous_impl, _m_grouped_fp8_gemm_nt_masked_impl
    global _bf16_gemm_nt_impl, _m_grouped_bf16_gemm_nt_contiguous_impl, _m_grouped_bf16_gemm_nt_masked_impl

    symbol_impls = [f"_{symbol}_impl" for symbol in symbols]
    # check if the symbols are valid
    if any(symbol not in _deep_gemm_impl_new_map for symbol in symbols):
        raise ValueError(f"Invalid symbols: {symbols}")
    if all(
        getattr(globals(), symbol_impl, None) is not None
        for symbol_impl in symbol_impls
    ):
        # already initialized
        return
    if not has_deep_gemm():
        # deep_gemm is not available
        return

    import deep_gemm

    # resolve symbols
    for i, symbol in enumerate(symbols):
        symbol_impl = symbol_impls[i]
        try:
            globals()[symbol_impl] = resolve_symbol(
                deep_gemm,
                _deep_gemm_impl_new_map[symbol],
                _deep_gemm_impl_old_map[symbol],
            )
        except AttributeError:
            raise RuntimeError(
                f"DeepGEMM symbol {_deep_gemm_impl_new_map[symbol]} and {_deep_gemm_impl_old_map[symbol]} not found in deep_gemm module"
            )


def _lazy_init_deep_gemm_once():
    _lazy_init_deep_gemm(
        [
            "fp8_gemm_nt",
            "m_grouped_fp8_gemm_nt_contiguous",
            "m_grouped_fp8_gemm_nt_masked",
            "bf16_gemm_nt",
            "m_grouped_bf16_gemm_nt_contiguous",
            "m_grouped_bf16_gemm_nt_masked",
        ]
    )


_lazy_init_deep_gemm_once()


def fp8_gemm_nt(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: Optional[bool] = None,
) -> None:
    """Execute FP8 GEMM (A * B^T).

    Args:
        a (Tuple[torch.Tensor, torch.Tensor]): FP8 data and scales for the first matrix.
        b (Tuple[torch.Tensor, torch.Tensor]): FP8 data and scales for the second matrix.
        output (torch.Tensor): Output tensor.
        c (Optional[torch.Tensor], optional): Optional bias tensor. Defaults to None.
        compiled_dims (str, optional): Compiled dimensions. Defaults to "nk".
        disable_ue8m0_cast (bool, optional): Whether to disable E8M0 type cast for E8M0 scale.
            Defaults to None, which will be set to False if E8M0 scale is used, otherwise True.

    Returns:
        None
    """
    global _fp8_gemm_nt_impl
    if _fp8_gemm_nt_impl is None:
        return _missing_deep_gemm()
    _fp8_gemm_nt_impl(
        a,
        b,
        output,
        c,
        compiled_dims=compiled_dims,
        # normal gemm tmp not use ue8m0 cast default
        disable_ue8m0_cast=(
            disable_ue8m0_cast if disable_ue8m0_cast is not None else True
        ),
    )


def m_grouped_fp8_gemm_nt_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    m_indices: torch.Tensor,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: Optional[bool] = None,
) -> None:
    """Execute grouped FP8 GEMM (A * B^T) with contiguous layout.

    Args:
        a (Tuple[torch.Tensor, torch.Tensor]): FP8 data and scales for the first matrix with contiguous layout.
        b (Tuple[torch.Tensor, torch.Tensor]): FP8 data and scales for the second matrix.
        output (torch.Tensor): Output tensor.
        m_indices (torch.Tensor): Grouped indices for valid tokens in each group.
            The length of m_indices is the a[0].shape[0], and the corresponding value of valid tokens is group_idx.
        compiled_dims (str, optional): Compiled dimensions. Defaults to "nk".
        disable_ue8m0_cast (bool, optional): Whether to disable E8M0 type cast for E8M0 scale.
            Defaults to None, which will be set to False if E8M0 scale is used, otherwise True.
    """
    global _m_grouped_fp8_gemm_nt_contiguous_impl
    if _m_grouped_fp8_gemm_nt_contiguous_impl is None:
        return _missing_deep_gemm()
    _m_grouped_fp8_gemm_nt_contiguous_impl(
        a,
        b,
        output,
        m_indices,
        compiled_dims=compiled_dims,
        disable_ue8m0_cast=(
            disable_ue8m0_cast
            if disable_ue8m0_cast is not None
            else not is_deep_gemm_e8m0_used()
        ),
    )


def m_grouped_fp8_gemm_nt_masked(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: Optional[bool] = None,
) -> None:
    """Execute grouped FP8 GEMM (A * B^T) with masked layout.

    Args:
        a (Tuple[torch.Tensor, torch.Tensor]): FP8 data and scales for the first matrix with masked layout.
        b (Tuple[torch.Tensor, torch.Tensor]): FP8 data and scales for the second matrix.
        output (torch.Tensor): Output tensor.
        masked_m (torch.Tensor): the number of valid tokens in each group.
        expected_m (int): Expected number of valid tokens in each group.
        compiled_dims (str, optional): Compiled dimensions. Defaults to "nk".
        disable_ue8m0_cast (bool, optional): Whether to disable E8M0 type cast for E8M0 scale.
            Defaults to None, which will be set to False if E8M0 scale is used, otherwise True.
    """
    global _m_grouped_fp8_gemm_nt_masked_impl
    if _m_grouped_fp8_gemm_nt_masked_impl is None:
        return _missing_deep_gemm()
    _m_grouped_fp8_gemm_nt_masked_impl(
        a,
        b,
        output,
        masked_m,
        expected_m,
        compiled_dims=compiled_dims,
        disable_ue8m0_cast=(
            disable_ue8m0_cast
            if disable_ue8m0_cast is not None
            else not is_deep_gemm_e8m0_used()
        ),
    )


def bf16_gemm_nt(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "nk",
) -> None:
    """Execute BF16 GEMM (A * B^T).

    Args:
        a (torch.Tensor): BF16 data for the first matrix.
        b (torch.Tensor): BF16 data for the second matrix.
        output (torch.Tensor): Output tensor.
        c (Optional[torch.Tensor], optional): Optional bias tensor. Defaults to None.
        compiled_dims (str, optional): Compiled dimensions. Defaults to "nk".
    """
    global _bf16_gemm_nt_impl
    if _bf16_gemm_nt_impl is None:
        return _missing_deep_gemm()
    _bf16_gemm_nt_impl(a, b, output, c, compiled_dims)


def m_grouped_bf16_gemm_nt_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    m_indices: torch.Tensor,
    compiled_dims: str = "nk",
) -> None:
    """Execute grouped BF16 GEMM (A * B^T) with contiguous layout.

    Args:
        a (torch.Tensor): BF16 data for the first matrix with contiguous layout.
        b (torch.Tensor): BF16 data for the second matrix.
        output (torch.Tensor): Output tensor.
        m_indices (torch.Tensor): Grouped indices for valid tokens in each group.
            The length of m_indices is the a.shape[0], and the corresponding value of valid tokens is group_idx.
        compiled_dims (str, optional): Compiled dimensions. Defaults to "nk".
    """
    global _m_grouped_bf16_gemm_nt_contiguous_impl
    if _m_grouped_bf16_gemm_nt_contiguous_impl is None:
        return _missing_deep_gemm()
    _m_grouped_bf16_gemm_nt_contiguous_impl(
        a,
        b,
        output,
        m_indices,
        compiled_dims,
    )


def m_grouped_bf16_gemm_nt_masked(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    compiled_dims: str = "nk",
) -> None:
    """Execute grouped BF16 GEMM (A * B^T) with masked layout.

    Args:
        a (torch.Tensor): BF16 data for the first matrix with masked layout.
        b (torch.Tensor): BF16 data for the second matrix.
        output (torch.Tensor): Output tensor.
        masked_m (torch.Tensor): the number of valid tokens in each group.
        expected_m (int): Expected number of valid tokens in each group.
        compiled_dims (str, optional): Compiled dimensions. Defaults to "nk".
    """
    global _m_grouped_bf16_gemm_nt_masked_impl
    if _m_grouped_bf16_gemm_nt_masked_impl is None:
        return _missing_deep_gemm()
    _m_grouped_bf16_gemm_nt_masked_impl(
        a,
        b,
        output,
        masked_m,
        expected_m,
        compiled_dims,
    )

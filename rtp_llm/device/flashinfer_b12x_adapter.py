"""Compatibility boundary for RTP-LLM's pinned FlashInfer B12x APIs."""

import functools
import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Callable, NamedTuple, Optional

import torch

from rtp_llm.config.quant_config import NVFP4_BLOCK_SIZE
from rtp_llm.utils.util import str_to_bool

logger = logging.getLogger(__name__)

SUPPORTED_FLASHINFER_VERSION = "0.6.12rc1+rtp.260523"
DISABLE_CUDA12_9_COMPAT_ENV = "RTP_LLM_DISABLE_B12X_CUDA12_9_COMPAT"

_version_gate_warned = False
_version_gate_lock = threading.Lock()


class _B12xSymbols(NamedTuple):
    wrapper: type
    convert_sf_to_mma_layout: Callable[..., torch.Tensor]
    kernel_tile_n: int


def get_disable_cuda12_9_compat() -> bool:
    raw_value = os.getenv(DISABLE_CUDA12_9_COMPAT_ENV)
    if not raw_value:
        return False
    try:
        return str_to_bool(raw_value)
    except ValueError as error:
        raise ValueError(
            f"{DISABLE_CUDA12_9_COMPAT_ENV} must be one of "
            f"yes/true/1/no/false/0, got {raw_value!r}"
        ) from error


@functools.cache
def _load_b12x_symbols() -> _B12xSymbols:
    """Validate and cache the private APIs provided by the pinned wheel."""
    try:
        import flashinfer
    except ImportError as error:
        raise RuntimeError(
            "b12x FP4 requires flashinfer-python; reinstall the CUDA 12.9 lock"
        ) from error

    version = getattr(flashinfer, "__version__", "unknown")
    if version != SUPPORTED_FLASHINFER_VERSION:
        raise RuntimeError(
            "b12x FP4 was validated with flashinfer-python "
            f"{SUPPORTED_FLASHINFER_VERSION}, got {version}; update the pinned "
            "dependency and this adapter together"
        )

    try:
        from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
        from flashinfer.fused_moe.cute_dsl import B12xMoEWrapper
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            _level_tile_n,
        )
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "b12x FP4 requires the pinned FlashInfer B12x private APIs; "
            "reinstall the CUDA 12.9 lock or update the RTP-LLM adapter"
        ) from error

    tile_n = _level_tile_n()
    if not isinstance(tile_n, int) or tile_n <= 0:
        raise RuntimeError(
            f"FlashInfer returned an invalid b12x tile width: {tile_n!r}"
        )
    return _B12xSymbols(B12xMoEWrapper, convert_sf_to_mma_layout, tile_n)


def get_b12x_kernel_tile_n() -> int:
    return _load_b12x_symbols().kernel_tile_n


def convert_b12x_blockscale_to_mma_layout(
    blockscale: torch.Tensor,
    *,
    m: int,
    k: int,
    num_groups: int,
) -> torch.Tensor:
    return _load_b12x_symbols().convert_sf_to_mma_layout(
        blockscale,
        m=m,
        k=k,
        num_groups=num_groups,
        sf_vec_size=NVFP4_BLOCK_SIZE,
    )


@contextmanager
def relaxed_b12x_cuda_version_gate():
    """Relax the CUDA>=13 probe only for synchronous wrapper construction."""
    from flashinfer.jit import cpp_ext

    # Serialize the context so overlapping constructors cannot restore another
    # constructor's temporary probe.
    with _version_gate_lock:
        try:
            original = cpp_ext.get_cuda_version
        except AttributeError as error:
            raise RuntimeError(
                "b12x FP4 requires flashinfer.jit.cpp_ext.get_cuda_version; "
                "check the pinned flashinfer-python version"
            ) from error
        real_version = original()
        if (real_version.major, real_version.minor) != (
            12,
            9,
        ) or get_disable_cuda12_9_compat():
            # The native gate passes, the toolchain is not the validated 12.9
            # combination, or the operator disabled compatibility. Preserve
            # FlashInfer's behavior.
            yield
            return

        global _version_gate_warned
        if not _version_gate_warned:
            logger.warning(
                "b12x NVFP4 MoE: temporarily reporting CUDA 13.0 to flashinfer "
                "while constructing the wrapper (real CUDA %s).",
                real_version,
            )
            _version_gate_warned = True
        fake_version = type(real_version)("13.0")
        constructing_thread = threading.get_ident()

        def get_compatible_cuda_version():
            if threading.get_ident() == constructing_thread:
                return fake_version
            return original()

        cpp_ext.get_cuda_version = get_compatible_cuda_version
        try:
            yield
        finally:
            cpp_ext.get_cuda_version = original


def create_b12x_wrappers(
    wrapper_args: dict[str, Any], enable_cuda_graph: bool
) -> tuple[Any, Optional[Any]]:
    """Create the graph wrapper and its optional oversized-prefill fallback."""
    wrapper_class = _load_b12x_symbols().wrapper
    # FusedMoeFactory constructs executors under torch.inference_mode().
    # FlashInfer mutates its routing workspace on every run, so wrapper-owned
    # buffers must be normal tensors while model weights stay inference tensors.
    with torch.inference_mode(False), relaxed_b12x_cuda_version_gate():
        graph_wrapper = wrapper_class(
            **wrapper_args,
            use_cuda_graph=enable_cuda_graph,
        )
        eager_wrapper = (
            wrapper_class(**wrapper_args, use_cuda_graph=False)
            if enable_cuda_graph
            else None
        )
    return graph_wrapper, eager_wrapper

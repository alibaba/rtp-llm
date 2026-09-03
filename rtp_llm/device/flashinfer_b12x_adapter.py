"""Compatibility boundary for the FlashInfer B12x APIs used by RTP-LLM."""

import functools
import sys
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional

import torch

from rtp_llm.config.quant_config import NVFP4_BLOCK_SIZE


class _B12xSymbols(NamedTuple):
    wrapper: type
    convert_sf_to_mma_layout: Callable[..., torch.Tensor]
    kernel_tile_n: int


def _ensure_cutlass_python_path() -> None:
    """Expose the nested CUTLASS package when wheel .pth files are not loaded."""
    try:
        import nvidia_cutlass_dsl
    except ImportError:
        return

    for package_root in nvidia_cutlass_dsl.__path__:
        python_packages = Path(package_root) / "python_packages"
        if python_packages.is_dir():
            path = str(python_packages)
            if path not in sys.path:
                sys.path.insert(0, path)
            return


@functools.cache
def _load_b12x_symbols() -> _B12xSymbols:
    """Validate and cache the FlashInfer APIs required by the B12x executor."""
    _ensure_cutlass_python_path()
    try:
        from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
        from flashinfer.fused_moe.cute_dsl import B12xMoEWrapper
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "b12x FP4 requires FlashInfer's B12x APIs; reinstall the matching "
            "CUDA lock or update the RTP-LLM adapter"
        ) from error

    level_tile_n = getattr(moe_dispatch, "_level_tile_n", None)
    tile_n = (
        level_tile_n()
        if callable(level_tile_n)
        else getattr(moe_dispatch, "_LEVEL_TILE_N", None)
    )
    if not isinstance(tile_n, int) or tile_n <= 0:
        raise RuntimeError(
            "b12x FP4 requires FlashInfer's _level_tile_n() or _LEVEL_TILE_N "
            f"API to return a positive integer, got {tile_n!r}"
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


def create_b12x_wrappers(
    wrapper_args: dict[str, Any],
    enable_cuda_graph: bool,
) -> tuple[Any, Optional[Any]]:
    """Create the graph wrapper and its optional oversized-prefill fallback."""
    wrapper_class = _load_b12x_symbols().wrapper
    # FusedMoeFactory constructs executors under torch.inference_mode().
    # FlashInfer mutates its routing workspace on every run, so wrapper-owned
    # buffers must be normal tensors while model weights stay inference tensors.
    with torch.inference_mode(False):
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

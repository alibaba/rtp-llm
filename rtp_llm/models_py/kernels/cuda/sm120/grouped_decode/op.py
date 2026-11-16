"""Bind the private SM120 grouped GEMM to the RTP decode contracts."""

import importlib.util
import threading
from functools import cache, partial
from pathlib import Path

import torch

from rtp_llm.utils.jit_cache_env import (
    SM120_GROUPED_CACHE_ENV,
    configure_writable_cache_env,
)

_INIT_LOCK = threading.Lock()


@cache
def _binding():
    from . import rtp_dg120_grouped

    # Resolve the locked CUTLASS headers without importing the default
    # DeepGEMM runtime or changing its settings.
    spec = importlib.util.find_spec("deep_gemm")
    if spec is None or spec.origin is None:
        raise RuntimeError("The locked DeepGEMM wheel must supply CUTLASS headers")
    cutlass = Path(spec.origin).absolute().parent / "include"
    if not (cutlass / "cutlass/version.h").is_file():
        raise RuntimeError(f"Missing DeepGEMM CUTLASS headers in {cutlass}")

    configure_writable_cache_env(
        SM120_GROUPED_CACHE_ENV,
        Path.home() / ".cache/rtp_llm/sm120_grouped",
        "sm120_grouped",
    )
    rtp_dg120_grouped.init(
        str(Path(__file__).absolute().parent / "native"), str(cutlass)
    )
    return rtp_dg120_grouped


def _check_operands(device, a, b) -> None:
    if a[0].device != device:
        raise ValueError(f"SM120 grouped FP8×FP4 decode is bound to {device}")
    if a[1].dtype != torch.int32 or b[1].dtype != torch.int32:
        raise RuntimeError(
            "SM120 grouped FP8×FP4 decode requires prepacked int32 scales; "
            f"got a_scale={a[1].dtype}, b_scale={b[1].dtype}"
        )


def _run_grouped(
    implementation,
    device: torch.device,
    /,
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    grouped_layout: torch.Tensor,
    recipe: tuple[int, int, int] | None = None,
    recipe_a: tuple[int, int] | None = None,
    recipe_b: tuple[int, int] | None = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool | None = None,
    use_psum_layout: bool = False,
    expected_m_for_psum_layout: int | None = None,
) -> None:
    if use_psum_layout or expected_m_for_psum_layout is not None:
        raise ValueError("SM120 grouped FP8×FP4 decode does not support psum layout")
    _check_operands(device, a, b)
    implementation(
        a,
        b,
        output,
        grouped_layout,
        recipe=recipe,
        recipe_a=recipe_a,
        recipe_b=recipe_b,
        compiled_dims=compiled_dims,
        # Exact SM120 uses packed UE8M0 scales, so RTP leaves the cast enabled
        # when the caller does not override the policy.
        disable_ue8m0_cast=False if disable_ue8m0_cast is None else disable_ue8m0_cast,
        use_psum_layout=False,
        expected_m_for_psum_layout=None,
    )


def _run_masked_grouped(
    implementation,
    device: torch.device,
    /,
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    *,
    recipe_a: tuple[int, int],
    recipe_b: tuple[int, int],
) -> None:
    _check_operands(device, a, b)
    implementation(
        a,
        b,
        output,
        masked_m,
        expected_m,
        recipe_a=recipe_a,
        recipe_b=recipe_b,
    )


@cache
def _grouped_op_for_device(index: int, masked: bool):
    if torch.cuda.get_device_capability(index) != (12, 0):
        raise RuntimeError("The private grouped FP8×FP4 GEMM requires SM120")
    binding = _binding()
    binding.warmup_device(index)
    device = torch.device("cuda", index)
    if masked:
        return partial(
            _run_masked_grouped, binding.m_grouped_fp8_fp4_gemm_nt_masked, device
        )
    return partial(_run_grouped, binding.m_grouped_fp8_fp4_gemm_nt_contiguous, device)


def grouped_op(device: torch.device | str, *, masked: bool = False):
    """Return the prepared RTP callable for a concrete SM120 device."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The private grouped FP8×FP4 GEMM requires SM120")
    index = torch.cuda.current_device() if device.index is None else device.index
    with _INIT_LOCK:
        return _grouped_op_for_device(index, masked)


def is_available(device: torch.device) -> bool:
    try:
        grouped_op(device)
    except (ImportError, OSError, RuntimeError):
        return False
    return True

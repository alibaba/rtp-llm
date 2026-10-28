"""Optional small-M plan for FlashInfer's explicitly selected b12x backend.

Private FlashInfer APIs are confined to initialization here. The caller retains
the public mm_fp4 path when no plan is available or M/output dtype is outside
the plan's domain. Inputs retain the existing NVFP4, unshuffled 128x4 SF layout;
this helper neither quantizes tensors nor changes their scale or bias semantics.
"""

import inspect
import logging
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

_LOGGER = logging.getLogger(__name__)
_PLAN_LOCK = threading.RLock()
_PLANS: dict[tuple, "B12xMinMPlan | None"] = {}


@dataclass(frozen=True)
class B12xMinMPlan:
    """A BF16-only plan; no weight, input or alpha tensor is retained.

    The caller must use this only for 0 < M < min_m, with the same N/K/device
    and NVFP4 SF16/128x4 contract passed to the factory. The tensors are exactly
    the public mm_fp4 operands, including transposed weight and weight scales.
    Ordinary graph warmup must compile this runner before capture.
    """

    min_m: int
    tactic: tuple
    default_tactic: tuple
    valid_tactics: tuple
    device: torch.device
    n: int
    k: int
    enable_pdl: bool
    _runner: Any
    _workspace: torch.Tensor

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        a_descale: torch.Tensor,
        b_descale: torch.Tensor,
        alpha: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        if (
            out_dtype != torch.bfloat16
            or a.ndim != 2
            or not 0 < a.shape[0] < self.min_m
        ):
            raise ValueError("b12x minimum-M plan requires BF16 and 0 < M < min_m")
        # The private runner reconstructs B using A's K and ignores strides.
        # Preserve the public entry point's checks before any kernel launch.
        if (
            a.shape[1] != self.k // 2
            or tuple(b.shape) != (self.k // 2, self.n)
            or a.dtype != torch.uint8
            or b.dtype != torch.uint8
            or not a.is_contiguous()
            or not b.T.is_contiguous()
            or a.device != self.device
            or b.device != self.device
        ):
            raise ValueError(
                "b12x operands must match the plan's packed shape, layout and device"
            )
        if (
            tuple(a_descale.shape) != (128, self.k // 16)
            or tuple(b_descale.shape) != (self.k // 16, ((self.n + 127) // 128) * 128)
            or a_descale.dtype not in (torch.uint8, torch.float8_e4m3fn)
            or b_descale.dtype not in (torch.uint8, torch.float8_e4m3fn)
            or not a_descale.is_contiguous()
            or not b_descale.T.is_contiguous()
            or a_descale.device != self.device
            or b_descale.device != self.device
            or alpha.device != self.device
            or alpha.dtype != torch.float32
            or alpha.numel() != 1
            or not alpha.is_contiguous()
        ):
            raise ValueError(
                "b12x scales and alpha must match the plan's NVFP4 layout and device"
            )
        # Match mm_fp4's allocation and let the runner use the current stream.
        # Do not catch launch/compile errors and retry after a possible CUDA fault.
        out = torch.empty((a.shape[0], self.n), device=a.device, dtype=out_dtype)
        self._runner(
            inputs=[
                a,
                b,
                a_descale,
                b_descale,
                alpha,
                out_dtype,
                out,
                16,
                True,
                self._workspace,
            ],
            tactic=self.tactic,
        )
        return out


def _freeze(value):
    return (
        tuple(_freeze(x) for x in value) if isinstance(value, (tuple, list)) else value
    )


def _select_tactic(default: tuple, valid: tuple) -> tuple[int, tuple | None]:
    if not valid or any(
        len(t) != 6 or len(t[0]) != 2 or any(type(d) is not int or d <= 0 for d in t[0])
        for t in valid
    ):
        raise ValueError("incompatible b12x tactic format")
    if default not in valid:
        raise ValueError("b12x default tuple is absent from valid tactics")
    min_m = min(t[0][0] for t in valid)
    if default[0][0] <= min_m:
        return min_m, None
    candidate = ((min_m, default[0][1]), *default[1:])
    if candidate not in valid or candidate[3] is not False:
        raise ValueError("b12x minimum-M tuple cannot preserve the default tail")
    return min_m, candidate


@lru_cache(maxsize=1)
def _runtime():
    from flashinfer.gemm import gemm_base as gemm

    if tuple(inspect.signature(gemm._b12x_gemm_fp4_runner).parameters) != (
        "sm_major",
        "sm_minor",
        "enable_pdl",
        "out_dtype",
        "use_nvfp4",
    ):
        raise ValueError("incompatible b12x runner factory")
    if tuple(inspect.signature(gemm._select_default_sm120_mma_tiler).parameters) != (
        "m",
        "n",
        "sm_count",
    ):
        raise ValueError("incompatible b12x default tile selector")
    if not callable(gemm._b12x_gemm_fp4_requirement) or not callable(
        gemm._get_cache_buf
    ):
        raise ValueError("missing b12x initialization API")
    return gemm


@lru_cache(maxsize=None)
def _hardware(device: torch.device) -> tuple[int, int, int]:
    properties = torch.cuda.get_device_properties(device)
    return properties.major, properties.minor, properties.multi_processor_count


def _build_plan(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    alpha: torch.Tensor,
    enable_pdl: bool,
) -> B12xMinMPlan | None:
    gemm = _runtime()
    major, minor, sm_count = _hardware(weight.device)
    requirement = gemm._b12x_gemm_fp4_requirement
    # supported_compute_capability attaches metadata; calling the requirement
    # alone does not enforce its architecture list.
    if not requirement.is_compute_capability_supported(major * 10 + minor):
        return None
    # Reuse the backend's architecture, CUDA, CuTe and SF-layout eligibility.
    if not requirement(
        weight,
        weight.T,
        weight_scales,
        weight_scales.T,
        alpha,
        out_dtype=torch.bfloat16,
        block_size=16,
        use_8x4_sf_layout=False,
        backend="b12x",
        use_nvfp4=True,
        enable_pdl=enable_pdl,
    ):
        return None
    n, packed_k = weight.shape
    k = packed_k * 2
    runner = gemm._b12x_gemm_fp4_runner(major, minor, enable_pdl, torch.bfloat16, True)
    if (
        tuple(inspect.signature(runner.get_valid_tactics).parameters)
        != (
            "inputs",
            "profile",
        )
        or "tactic" not in inspect.signature(runner.forward).parameters
    ):
        raise ValueError("incompatible b12x runner methods")
    # get_valid_tactics only inspects shapes/dtypes. Meta tensors avoid dummy
    # device allocations and ensure no checkpoint operand enters the cache.
    inputs = [
        torch.empty((1, packed_k), dtype=torch.uint8, device="meta"),
        torch.empty((n, packed_k), dtype=torch.uint8, device="meta").T,
        torch.empty((128, k // 16), dtype=torch.uint8, device="meta"),
        torch.empty_like(weight_scales, device="meta").T,
        torch.empty((), dtype=torch.float32, device="meta"),
        torch.bfloat16,
        torch.empty((1, n), dtype=torch.bfloat16, device="meta"),
        16,
        True,
        None,
    ]
    valid = tuple(_freeze(t) for t in runner.get_valid_tactics(inputs, None))
    default_tile = tuple(gemm._select_default_sm120_mma_tiler(1, n, sm_count))
    # This is the supported six-field private runner ABI. Require the complete
    # default and replacement tuples, rather than selecting by tile alone.
    default = (default_tile, (1, 1), False, False, "sm120", None)
    min_m, candidate = _select_tactic(default, valid)
    if candidate is None:
        return None
    if any(
        tuple(gemm._select_default_sm120_mma_tiler(m, n, sm_count)) != default_tile
        for m in range(1, min_m)
    ):
        raise ValueError("b12x default tile varies within the minimum-M interval")
    workspace = gemm._get_cache_buf(
        "mm_fp4_workspace", gemm.DEFAULT_WORKSPACE_SIZE, weight.device
    )
    return B12xMinMPlan(
        min_m,
        candidate,
        default,
        valid,
        weight.device,
        n,
        k,
        enable_pdl,
        runner,
        workspace,
    )


def create_b12x_min_m_plan(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    alpha: torch.Tensor,
    *,
    enable_pdl: bool = True,
) -> B12xMinMPlan | None:
    """Initialize an optional plan after explicit backend='b12x' selection.

    Weight is row-major packed FP4; scales must already be unshuffled SF16
    E4M3 in FlashInfer's 128x4 layout. Shape/stride checks cannot establish the
    scale contents/layout provenance: the existing weight loader owns that
    contract. There is no conversion, tuning-cache mutation or kernel launch.

    Unsupported tensors, hardware/dependencies or private API changes return
    None. This preserves the caller's public mm_fp4 behavior, including its
    errors if the explicitly requested backend itself is unavailable. Plans
    are shared by device/N/K/SF dtype/PDL, never by tensor identity or contents.
    """
    if (
        not isinstance(weight, torch.Tensor)
        or not isinstance(weight_scales, torch.Tensor)
        or not isinstance(alpha, torch.Tensor)
        or not weight.is_cuda
        or weight.dtype != torch.uint8
        or weight.ndim != 2
        or not weight.is_contiguous()
        or weight_scales.device != weight.device
        or weight_scales.dtype not in (torch.uint8, torch.float8_e4m3fn)
        or not weight_scales.is_contiguous()
        or alpha.device != weight.device
        or alpha.dtype != torch.float32
        or alpha.numel() != 1
        or not alpha.is_contiguous()
        or type(enable_pdl) is not bool
    ):
        return None
    n, packed_k = weight.shape
    k = packed_k * 2
    if (
        n <= 0
        or k <= 0
        or k % 128 != 0
        or tuple(weight_scales.shape) != (((n + 127) // 128) * 128, k // 16)
    ):
        return None
    key = (weight.device, n, k, weight_scales.dtype, enable_pdl)
    with _PLAN_LOCK:
        if key not in _PLANS:
            try:
                _PLANS[key] = _build_plan(weight, weight_scales, alpha, enable_pdl)
            # Optional initialization crosses private APIs and launches no
            # kernel. Their failure must leave the public path authoritative.
            except Exception as error:
                _LOGGER.debug(
                    "b12x minimum-M plan unavailable: %s", type(error).__name__
                )
                _PLANS[key] = None
        return _PLANS[key]

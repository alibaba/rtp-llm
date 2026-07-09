"""ROCm FP8 PTPC (Per-Token Per-Channel) quantized Linear implementation."""

import importlib
import importlib.metadata as importlib_metadata
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from rtp_llm.models_py.kernels.rocm.fp8_kernel import rocm_per_token_quant_fp8
from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.ops import HWKernelConfig
from rtp_llm.utils.aiter_jit_patch import load_aiter

load_aiter()
_gemm_ops = importlib.import_module("aiter.ops.gemm_op_a8w8")
_gradlib_ops = importlib.import_module("aiter.ops.gradlib")
gemm_a8w8_bpreshuffle_cktile = _gemm_ops.gemm_a8w8_bpreshuffle_cktile

logger = logging.getLogger(__name__)


class RocmFp8PTPCLinearBase(LinearBase):
    """Common ROCm FP8 PTPC helpers shared by concrete layout strategies."""

    @classmethod
    def _can_handle_fp8_ptpc(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
    ) -> bool:
        if weight_scales is None or quant_config is None:
            return False
        if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            return False
        return quant_config.get_method() in (
            "FP8_PER_CHANNEL_COMPRESSED",
            "FP8_PER_CHANNEL_QUARK",
        )

    @classmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        raise NotImplementedError("Subclasses must implement can_handle().")

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_config: object = None,
        weight_scale_2: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            weight, weight_scales, input_scales, bias, quant_config, weight_scale_2
        )
        self.hidden_size = weight.shape[0]  # K
        self.output_size = weight.shape[1]  # N
        self.bias = bias

    @staticmethod
    @lru_cache(maxsize=1)
    def init_hipblas() -> None:
        _gradlib_ops.hipb_create_extension()

    @staticmethod
    def _as_hipb_weight_view(weight: torch.Tensor) -> torch.Tensor:
        """Expose swizzled FP8 weight as column-major metadata for hipBLASLt.

        Weight loading has already applied the swizzleA physical transform. The
        hipBLASLt bpreshuffle path identifies the FP8 COL16_4R16 layout through
        a column-major stride view, so this is intentionally metadata-only and
        must not copy or transpose the underlying storage.
        """
        return weight.as_strided(weight.shape, (1, weight.shape[0]))

    @staticmethod
    def _as_hipb_scale_b(weight_scales: torch.Tensor, output_size: int) -> torch.Tensor:
        """Normalize per-channel scaleB to hipBLASLt rowwise [1, N] layout."""
        if weight_scales.dim() == 1:
            return weight_scales.reshape(1, output_size).contiguous()
        if tuple(weight_scales.shape) == (output_size, 1):
            return weight_scales.T.contiguous()
        if tuple(weight_scales.shape) == (1, output_size):
            return weight_scales.contiguous()
        return weight_scales.reshape(1, output_size).contiguous()

    def _quantize_input(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.dtype]:
        original_dtype = input.dtype
        input_bf16 = (
            input if input.dtype == torch.bfloat16 else input.to(torch.bfloat16)
        )
        input_fp8, input_scales = rocm_per_token_quant_fp8(input_bf16, eps=1e-10)
        return input_fp8, input_scales.to(torch.float32), input_bf16, original_dtype

    @staticmethod
    def _restore_dtype(output: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return output if output.dtype == dtype else output.to(dtype)


class RocmFp8PTPCLinearNoSwizzle(RocmFp8PTPCLinearBase):
    """Original CK-preswizzled FP8 PTPC path used when swizzleA is disabled."""

    @classmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        if not cls._can_handle_fp8_ptpc(quant_config, weight, weight_scales):
            return False
        return hw_kernel_config is None or not hw_kernel_config.use_swizzleA

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_config: object = None,
        weight_scale_2: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            weight, weight_scales, input_scales, bias, quant_config, weight_scale_2
        )
        self.weight = weight.reshape([weight.shape[1], weight.shape[0]])
        self.weight_scales = weight_scales.reshape(
            [weight_scales.shape[1], weight_scales.shape[0]]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_fp8, input_scales, input_bf16, original_dtype = self._quantize_input(
            input
        )

        M = input_bf16.shape[0]
        N = self.output_size
        K = input_fp8.shape[-1]

        # Dispatch rules from origin/main, validated on MI308X with sweep
        # M=[1..16384], N={1024,2816}, K=1024. Keep this NoSwizzle policy
        # unchanged so shapes without the swizzleA layout continue using the
        # tuned CK/default split instead of being forced onto hipBLASLt.
        use_cktile = K < 192 or M >= 1536 or (M >= 512 and N > 1536)
        if use_cktile:
            output = torch.empty(
                (M, N), dtype=input_bf16.dtype, device=input_bf16.device
            )
            gemm_a8w8_bpreshuffle_cktile(
                input_fp8, self.weight, input_scales, self.weight_scales, output
            )
        else:
            output = _gemm_ops.gemm_a8w8_bpreshuffle(
                input_fp8,
                self.weight,
                input_scales,
                self.weight_scales,
                None,
                input_bf16.dtype,
            )

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return self._restore_dtype(output, original_dtype)


class RocmFp8PTPCLinearWithSwizzle(RocmFp8PTPCLinearBase):
    """hipBLASLt bpreshuffle path for FP8 PTPC weights loaded with swizzleA."""

    @classmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        if not cls._can_handle_fp8_ptpc(quant_config, weight, weight_scales):
            return False
        return hw_kernel_config is not None and hw_kernel_config.use_swizzleA

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_config: object = None,
        weight_scale_2: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            weight, weight_scales, input_scales, bias, quant_config, weight_scale_2
        )
        self.weight = self._as_hipb_weight_view(weight)
        self.weight_scales = self._as_hipb_scale_b(weight_scales, self.output_size)

    @staticmethod
    def _solution_cache_path() -> Path:
        return Path(__file__).resolve().parent / "data" / "fp8_ptpc_hipb_solutions.json"

    @staticmethod
    def _target_matches(payload: dict) -> bool:
        target = payload.get("target", {})
        arch_prefix = str(target.get("arch_prefix", ""))
        hip_prefix = str(target.get("torch_hip_prefix", ""))
        aiter_version_prefix = str(target.get("aiter_version_prefix", ""))

        try:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            gpu_arch = str(getattr(props, "gcnArchName", ""))
        except Exception:
            gpu_arch = ""
        torch_hip = str(getattr(torch.version, "hip", "") or "")
        try:
            installed_aiter = importlib_metadata.version("aiter")
        except Exception:
            installed_aiter = ""

        return (
            (not arch_prefix or gpu_arch.startswith(arch_prefix))
            and (not hip_prefix or torch_hip.startswith(hip_prefix))
            and (
                not aiter_version_prefix
                or installed_aiter.startswith(aiter_version_prefix)
            )
        )

    @staticmethod
    def _parse_solution_cache(payload: dict) -> dict[Tuple[int, int, int, str], int]:
        cache: dict[Tuple[int, int, int, str], int] = {}
        for row in payload.get("solutions", []):
            try:
                key = (
                    int(row["m"]),
                    int(row["k"]),
                    int(row["n"]),
                    str(row["epilogue"]),
                )
                cache[key] = int(row["solution_index"])
            except (KeyError, TypeError, ValueError):
                logger.debug("Skipping malformed FP8 PTPC HIPB solution row: %s", row)
        return cache

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_solution_cache() -> dict[Tuple[int, int, int, str], int]:
        cache_path = RocmFp8PTPCLinearWithSwizzle._solution_cache_path()
        try:
            payload = json.loads(cache_path.read_text())
        except FileNotFoundError:
            logger.debug("FP8 PTPC HIPB solution cache not found: %s", cache_path)
            return {}
        except Exception as exc:
            logger.debug(
                "Failed to read FP8 PTPC HIPB solution cache %s: %s", cache_path, exc
            )
            return {}

        if payload.get("format_version") != 1:
            logger.debug(
                "Unsupported FP8 PTPC HIPB solution cache format: %s", cache_path
            )
            return {}
        if not RocmFp8PTPCLinearWithSwizzle._target_matches(payload):
            logger.debug(
                "FP8 PTPC HIPB solution cache metadata mismatch: %s", cache_path
            )
            return {}
        return RocmFp8PTPCLinearWithSwizzle._parse_solution_cache(payload)

    @staticmethod
    def _epilogue_name(add_bias: bool, has_bias: bool, use_gelu: bool) -> str:
        if use_gelu and add_bias and has_bias:
            return "bias_gelu"
        if add_bias and has_bias:
            return "bias"
        return "none"

    def _get_solution_index(self, m: int, k: int, n: int, epilogue: str) -> int:
        return self._load_solution_cache().get((m, k, n, epilogue), -1)

    def _forward_hipb(
        self, input: torch.Tensor, add_bias: bool, use_gelu: bool = False
    ) -> torch.Tensor:
        self.init_hipblas()
        input_fp8, input_scales, input_bf16, original_dtype = self._quantize_input(
            input
        )
        bias = self.bias if add_bias else None
        epilogue = self._epilogue_name(add_bias, self.bias is not None, use_gelu)
        solution_index = self._get_solution_index(
            input_fp8.shape[0], self.hidden_size, self.output_size, epilogue
        )
        kwargs = dict(
            bias=bias,
            out_dtype=input_bf16.dtype,
            scaleA=input_scales,
            scaleB=self.weight_scales,
            bpreshuffle=True,
        )
        if use_gelu:
            kwargs["use_gelu"] = True
        output = _gradlib_ops.hipb_mm(input_fp8, self.weight, solution_index, **kwargs)
        return self._restore_dtype(output, original_dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._forward_hipb(input, add_bias=True)

    def forward_with_bias_gelu(self, input: torch.Tensor) -> torch.Tensor:
        if self.bias is None:
            return F.gelu(self._forward_hipb(input, add_bias=False))
        return self._forward_hipb(input, add_bias=True, use_gelu=True)

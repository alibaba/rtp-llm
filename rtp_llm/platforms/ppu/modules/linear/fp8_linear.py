"""M890P block-FP8 dense linear and quantization contracts.

Weights use E4M3 payloads with FP32 block scales. UE8M0 checkpoint scales
convert once at construction. The operator has no dequantized GEMM fallback
and is usable independently of the DeepSeek V4 model.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Callable, Optional, Sequence

import torch
from rtp_llm.platforms.ppu.runtime import install_deep_gemm_build_lock, require_symbol
from torch import nn

FP8_BLOCK_SIZE = 128
FP8_QUANT_EPS = 1.0e-4
M890P_DEVICE_NAME = "ZW-M890P"
_DENSE_SYMBOL = "fp8_gemm_nt"
_QUANT_LEGACY_SYMBOL = "per_token_group_quant_fp8"
_QUANT_V2_SYMBOL = "per_token_group_quant_fp8_v2"
_QUANT_V2_MIN_ELEMENTS = 4 * 1024 * 1024


def _validate_fp8_quantization(quantization: str) -> None:
    if quantization not in ("auto", "v2_row", "v2_column"):
        raise ValueError("PPU FP8 quantization must be auto, v2_row, or v2_column")


def _validate_activation_scale_layout(scale: torch.Tensor, quantization: str) -> None:
    if quantization == "v2_column":
        expected = (1, max(1, scale.shape[0]))
        if scale.stride() != expected:
            raise ValueError(f"PPU FP8 column scales require strides {expected}")
    elif not scale.is_contiguous():
        raise ValueError("PPU FP8 row scales must be contiguous")


def _require_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise RuntimeError(f"M890P DSV4 FP8 requires torch.{name}")
    return dtype


def _resolve_deep_gemm_symbol(name: str) -> Callable:
    symbol = require_symbol("deep_gemm", name)
    install_deep_gemm_build_lock()
    return symbol


@lru_cache(maxsize=1)
def _resolve_ppu_quant_symbols() -> tuple[Callable, Callable]:
    """Resolve both M890P quant kernels from the real compute-ops module."""

    try:
        compute_ops = importlib.import_module("rtp_llm.ops.compute_ops")
    except ImportError as exc:
        raise RuntimeError("M890P DSV4 FP8 requires rtp_llm.ops.compute_ops") from exc

    resolved = []
    for name in (_QUANT_LEGACY_SYMBOL, _QUANT_V2_SYMBOL):
        symbol = getattr(compute_ops, name, None)
        if not callable(symbol):
            raise RuntimeError(
                f"M890P DSV4 FP8 requires callable rtp_llm.ops.compute_ops.{name}"
            )
        resolved.append(symbol)
    return resolved[0], resolved[1]


def _require_cuda_contiguous(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA/PPU tensor, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _require_m890p(tensor: torch.Tensor, name: str) -> None:
    _require_cuda_contiguous(tensor, name)
    device_index = tensor.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    if device_name != M890P_DEVICE_NAME:
        raise RuntimeError(
            f"{name} must be on {M890P_DEVICE_NAME}, got {device_name!r} "
            f"at {tensor.device}"
        )


def checkpoint_ue8m0_scale_to_fp32(
    scale: torch.Tensor,
    weight_shape: Sequence[int],
) -> torch.Tensor:
    """Convert a checkpoint F8_E8M0 block grid to PPU DeepGEMM FP32.

    ``weight_shape`` is ``(N, K)`` and both dimensions must be block-128
    aligned.  The scale shape is exactly ``(N / 128, K / 128)``.  PyTorch's
    F8_E8M0-to-FP32 cast implements the checkpoint value semantics
    ``2 ** (encoded_byte - 127)``; no row replication or TMA/int32 packing is
    valid on the M890P path.
    """

    if len(weight_shape) != 2:
        raise ValueError(f"weight_shape must be (N, K), got {tuple(weight_shape)}")
    n, k = (int(weight_shape[0]), int(weight_shape[1]))
    if n <= 0 or k <= 0 or n % FP8_BLOCK_SIZE or k % FP8_BLOCK_SIZE:
        raise ValueError(
            "M890P DSV4 FP8 weight dimensions must be positive multiples of "
            f"{FP8_BLOCK_SIZE}, got {(n, k)}"
        )
    expected = (n // FP8_BLOCK_SIZE, k // FP8_BLOCK_SIZE)
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"checkpoint scale shape must be {expected} for weight {(n, k)}, "
            f"got {tuple(scale.shape)}"
        )
    if scale.dtype != _require_dtype("float8_e8m0fnu"):
        raise TypeError(
            "checkpoint scale must be torch.float8_e8m0fnu, " f"got {scale.dtype}"
        )
    _require_m890p(scale, "checkpoint scale")
    return scale.to(torch.float32).contiguous()


def quantize_ppu_fp8_activation(
    x: torch.Tensor,
    *,
    quantization: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 through the M890P compute-ops ABI.

    The contract is group 128, epsilon ``1e-4``, FP32 scales and no UE8M0
    packing. The default uses row-major scales and switches from the legacy
    kernel to v2 at four million elements, matching the established RTP
    dispatch threshold. Explicit v2_row/v2_column select the v2 arithmetic
    and scale layout for all shapes. Column scales feed PPU DeepGEMM directly
    without a transpose copy; the installed M890P ABI uses unpadded M strides.
    """

    _validate_fp8_quantization(quantization)
    if x.ndim != 2:
        raise ValueError(f"activation must be 2D, got {x.ndim}D")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"activation must be torch.bfloat16, got {x.dtype}")
    _require_m890p(x, "activation")
    if x.shape[1] % FP8_BLOCK_SIZE:
        raise ValueError(
            f"activation K must be divisible by {FP8_BLOCK_SIZE}, got {x.shape[1]}"
        )

    expected_scale_shape = (x.shape[0], x.shape[1] // FP8_BLOCK_SIZE)
    payload = torch.empty_like(x, dtype=_require_dtype("float8_e4m3fn"))
    if quantization == "v2_column":
        scale = torch.empty(
            expected_scale_shape[::-1], dtype=torch.float32, device=x.device
        ).t()
    else:
        scale = torch.empty(expected_scale_shape, dtype=torch.float32, device=x.device)
    if x.shape[0] > 0:
        legacy_quant, v2_quant = _resolve_ppu_quant_symbols()
        legacy_symbol_name = _QUANT_LEGACY_SYMBOL
        v2_symbol_name = _QUANT_V2_SYMBOL
        fp8_max = torch.finfo(payload.dtype).max
        if quantization != "auto" or x.numel() >= _QUANT_V2_MIN_ELEMENTS:
            try:
                v2_quant(
                    x,
                    payload,
                    scale,
                    FP8_BLOCK_SIZE,
                    FP8_QUANT_EPS,
                    -fp8_max,
                    fp8_max,
                    False,
                    False,
                    None,
                )
            except TypeError as exc:
                raise RuntimeError(
                    f"rtp_llm.ops.compute_ops.{v2_symbol_name} ABI "
                    "mismatch for M890P DSV4 FP8; expected (input, output_q, "
                    "output_s, group_size, eps, fp8_min, fp8_max, "
                    "scale_ue8m0, fuse_silu_and_mul, masked_m" + ")"
                ) from exc
        else:
            try:
                legacy_quant(
                    x,
                    payload,
                    scale,
                    FP8_BLOCK_SIZE,
                    FP8_QUANT_EPS,
                    -fp8_max,
                    fp8_max,
                    False,
                )
            except TypeError as exc:
                raise RuntimeError(
                    f"rtp_llm.ops.compute_ops.{legacy_symbol_name} ABI "
                    "mismatch for M890P DSV4 FP8; expected (input, output_q, "
                    "output_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0" + ")"
                ) from exc

    if payload.dtype != _require_dtype("float8_e4m3fn"):
        raise RuntimeError(
            f"PPU activation helper returned unexpected payload dtype {payload.dtype}"
        )
    if payload.shape != x.shape or not payload.is_contiguous():
        raise RuntimeError(
            "PPU activation helper returned an invalid payload layout: "
            f"shape={tuple(payload.shape)}, contiguous={payload.is_contiguous()}"
        )
    if payload.device != x.device:
        raise RuntimeError(
            "PPU activation helper returned payload on the wrong device: "
            f"got {payload.device}, expected {x.device}"
        )
    if scale.dtype != torch.float32 or tuple(scale.shape) != expected_scale_shape:
        raise RuntimeError(
            "PPU activation helper returned an invalid scale contract: "
            f"dtype={scale.dtype}, shape={tuple(scale.shape)}, "
            f"expected=torch.float32/{expected_scale_shape}"
        )
    _validate_activation_scale_layout(scale, quantization)
    if scale.device != x.device:
        raise RuntimeError(
            "PPU activation helper returned scale on the wrong device: "
            f"got {scale.device}, expected {x.device}"
        )
    return payload, scale


def _shares_untyped_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Conservatively detect aliases, including views with storage offsets."""

    a_storage = a.untyped_storage()
    b_storage = b.untyped_storage()
    a_identity = getattr(a_storage, "_cdata", None)
    b_identity = getattr(b_storage, "_cdata", None)
    if a_identity is not None and b_identity is not None:
        return a_identity == b_identity
    a_ptr = a_storage.data_ptr()
    return a_ptr != 0 and a_ptr == b_storage.data_ptr()


def _validate_out(
    out: Optional[torch.Tensor],
    expected_shape: tuple[int, ...],
    device: torch.device,
    aliases: Sequence[torch.Tensor],
) -> torch.Tensor:
    if out is None:
        return torch.empty(expected_shape, dtype=torch.bfloat16, device=device)
    if tuple(out.shape) != expected_shape:
        raise ValueError(f"out shape must be {expected_shape}, got {tuple(out.shape)}")
    if out.dtype != torch.bfloat16:
        raise TypeError(f"out must be torch.bfloat16, got {out.dtype}")
    if out.device != device:
        raise ValueError(f"out device must be {device}, got {out.device}")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    if any(_shares_untyped_storage(out, tensor) for tensor in aliases):
        raise ValueError("out must not alias activation, weight, or scale storage")
    return out


class PpuFp8Linear(nn.Module):
    """Reusable M890P dense/output FP8 linear primitive."""

    _dsv4_ppu_block_fp8 = True

    def __init__(
        self,
        weight: torch.Tensor,
        checkpoint_scale: torch.Tensor,
        *,
        share_input_quantization: bool = False,
        quantization: str = "auto",
        scale_is_prepared: bool = False,
    ):
        super().__init__()
        _validate_fp8_quantization(quantization)
        self.quantization = quantization
        if not isinstance(share_input_quantization, bool):
            raise TypeError("share_input_quantization must be bool")
        if weight.ndim != 2:
            raise ValueError(f"weight must be 2D, got {weight.ndim}D")
        if weight.dtype != _require_dtype("float8_e4m3fn"):
            raise TypeError(f"weight must be torch.float8_e4m3fn, got {weight.dtype}")
        _require_m890p(weight, "weight")
        self.n, self.k = (int(weight.shape[0]), int(weight.shape[1]))
        if scale_is_prepared:
            if self.n <= 0 or self.k <= 0 or self.n % 128 or self.k % 128:
                raise ValueError(
                    "Prepared FP8 weights must preserve block-128 geometry"
                )
            if checkpoint_scale.dtype != torch.float32:
                raise TypeError("Prepared FP8 scale must be FP32")
            if tuple(checkpoint_scale.shape) != (self.n // 128, self.k // 128):
                raise ValueError(
                    "Prepared FP8 scale must match the block-128 weight grid"
                )
            _require_m890p(checkpoint_scale, "prepared weight scale")
            weight_scale = checkpoint_scale
        else:
            weight_scale = checkpoint_ue8m0_scale_to_fp32(
                checkpoint_scale, weight.shape
            )
        if weight_scale.device != weight.device:
            raise ValueError(
                f"weight and checkpoint scale must share a device, got "
                f"{weight.device} and {weight_scale.device}"
            )

        self.register_buffer("weight", weight, persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=False)
        self._gemm = _resolve_deep_gemm_symbol(_DENSE_SYMBOL)

        self._share_input_quantization = share_input_quantization

    def can_share_input_quantization(self, other: nn.Module) -> bool:
        """Provider-local ABI gate; never mix PPU FP32 scales with CUDA UE8M0.

        Subclasses may change the
        quantizer contract and must opt in with their own implementation.
        """
        return (
            type(self) is PpuFp8Linear
            and type(other) is PpuFp8Linear
            and self._share_input_quantization
            and other._share_input_quantization
            and self.k == other.k
            and self.weight.device == other.weight.device
            and self.quantization == other.quantization
        )

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim < 2 or int(x.shape[-1]) != self.k:
            raise ValueError(
                f"activation shape must end in K={self.k}, got {tuple(x.shape)}"
            )
        if x.dtype != torch.bfloat16:
            raise TypeError(f"activation must be torch.bfloat16, got {x.dtype}")
        _require_m890p(x, "activation")
        if x.device != self.weight.device:
            raise ValueError(
                f"activation device must be {self.weight.device}, got {x.device}"
            )

    def quantize_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize one 2D input with the existing PPU format."""
        self._validate_input(x)
        if x.ndim != 2:
            raise ValueError(f"shared quantization input must be 2D, got {x.ndim}D")
        return quantize_ppu_fp8_activation(x, quantization=self.quantization)

    def forward_quantized(
        self,
        x_fp8: torch.Tensor,
        x_scale: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Consume E4M3/FP32 tensors in this instance's selected scale layout.

        The caller retains the quantized tensors through all consumers on
        their producer stream (or supplies an explicit cross-stream fence).
        """
        if x_fp8.ndim != 2 or x_fp8.shape[1] != self.k:
            raise ValueError(f"quantized activation must have shape (M, {self.k})")
        if x_fp8.dtype != _require_dtype("float8_e4m3fn"):
            raise TypeError("quantized activation must be torch.float8_e4m3fn")
        _require_m890p(x_fp8, "quantized activation")
        expected_scale = (x_fp8.shape[0], self.k // FP8_BLOCK_SIZE)
        if x_scale.dtype != torch.float32:
            raise TypeError("quantized activation scales must be torch.float32")
        if tuple(x_scale.shape) != expected_scale:
            raise ValueError(
                f"quantized activation scale shape must be {expected_scale}"
            )
        if not x_scale.is_cuda:
            raise ValueError("quantized activation scale must be a CUDA/PPU tensor")
        _validate_activation_scale_layout(x_scale, self.quantization)
        if x_fp8.device != self.weight.device or x_scale.device != x_fp8.device:
            raise ValueError(
                "quantized activation, scale and weight must share a device"
            )
        aliases = (x_fp8, x_scale, self.weight, self.weight_scale)
        output = _validate_out(out, (x_fp8.shape[0], self.n), x_fp8.device, aliases)
        if x_fp8.shape[0] > 0:
            self._run_gemm(x_fp8, x_scale, output)
        return output

    def forward(
        self,
        x: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_input(x)

        output_shape = tuple(x.shape[:-1]) + (self.n,)
        output = _validate_out(
            out,
            output_shape,
            x.device,
            (x, self.weight, self.weight_scale),
        )
        m = x.numel() // self.k
        if m == 0:
            return output

        x_2d = x.view(m, self.k)
        output_2d = output.view(m, self.n)
        x_fp8, x_scale = quantize_ppu_fp8_activation(
            x_2d, quantization=self.quantization
        )
        self._run_gemm(x_fp8, x_scale, output_2d)
        return output

    def _run_gemm(
        self, x_fp8: torch.Tensor, x_scale: torch.Tensor, output: torch.Tensor
    ) -> None:
        try:
            self._gemm(
                (x_fp8, x_scale),
                (self.weight, self.weight_scale),
                output,
            )
        except TypeError as exc:
            raise RuntimeError(
                "deep_gemm.fp8_gemm_nt ABI mismatch for the M890P DSV4 "
                "contract; expected (lhs_pair, rhs_pair, out)"
            ) from exc


def concatenate_ppu_fp8_linears(linears: Sequence[PpuFp8Linear]) -> PpuFp8Linear:
    """Build an owned output-axis concatenation before Graph capture.

    Sources remain valid and are not mutated. The result owns a weight copy;
    callers retain or release the source projections according to their model
    lifecycle. All consumers must have one quantization contract.
    """
    if len(linears) < 2 or any(type(part) is not PpuFp8Linear for part in linears):
        raise TypeError(
            "Concatenation requires at least two exact PpuFp8Linear instances"
        )
    first = linears[0]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("FP8 projection concatenation must precede Graph capture")
    for part in linears:
        if (
            part.k != first.k
            or part.weight.device != first.weight.device
            or part.quantization != first.quantization
        ):
            raise ValueError("FP8 projections must share K, device, quantization")
    weight = torch.cat([part.weight.view(torch.uint8) for part in linears], dim=0).view(
        first.weight.dtype
    )
    scales = torch.cat([part.weight_scale for part in linears], dim=0)
    checkpoint_scales = scales.to(_require_dtype("float8_e8m0fnu"))
    if not torch.equal(scales, checkpoint_scales.float()):
        raise ValueError(
            "FP8 projection scales no longer satisfy the checkpoint E8M0 contract"
        )
    return PpuFp8Linear(
        weight,
        checkpoint_scales,
        share_input_quantization=all(
            part._share_input_quantization for part in linears
        ),
        quantization=first.quantization,
    )

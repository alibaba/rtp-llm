"""CUDA FP8 quant helpers without grouped-GEMM package initialization."""

import importlib
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import torch

fp8_dtype = torch.float8_e4m3fn
fp8_max = torch.finfo(fp8_dtype).max
fp8_min = -fp8_max

_NATIVE_INPUT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _validate_native_quant_input(
    input: torch.Tensor, label: str, *, allow_empty: bool = False
) -> None:
    if not input.is_cuda:
        raise ValueError(f"{label} input must be on a CUDA device")
    if input.dtype not in _NATIVE_INPUT_DTYPES:
        raise TypeError(
            f"{label} input must be float32, float16, or bfloat16, got "
            f"{input.dtype}"
        )
    if not input.is_contiguous():
        raise ValueError(f"{label} input must be contiguous")
    if input.numel() == 0 and not allow_empty:
        raise ValueError(f"{label} input must not be empty")


def _validate_native_output(
    output: torch.Tensor, input: torch.Tensor, dtype: torch.dtype, label: str
) -> None:
    if output.shape != input.shape or output.dtype != dtype:
        raise ValueError(
            f"{label} output must have shape {tuple(input.shape)} and dtype "
            f"{dtype}, got {tuple(output.shape)} and {output.dtype}"
        )
    if output.device != input.device:
        raise ValueError(
            f"{label} output must be on {input.device}, got {output.device}"
        )
    if not output.is_contiguous():
        raise ValueError(f"{label} output must be contiguous")


@lru_cache(maxsize=None)
def _resolve_compute_op(name: str):
    """Resolve only the CUDA quant op requested by the selected backend."""
    try:
        compute_ops = importlib.import_module("librtp_compute_ops.rtp_llm_ops")
    except ImportError as error:
        raise ImportError("RTP FP8 compute ops are unavailable") from error
    op = getattr(compute_ops, name, None)
    if not callable(op):
        raise ImportError(f"librtp_compute_ops.rtp_llm_ops.{name} is unavailable")
    return op


def require_cuda_fp8_quant_helpers(quant_kind: str) -> None:
    """Fail before model materialization if a selected CUDA helper is absent."""
    required_ops = {
        "per_tensor": ("per_tensor_quant_fp8",),
        "per_token": ("per_token_quant_fp8",),
        "group": ("per_token_group_quant_fp8",),
    }
    try:
        op_names = required_ops[quant_kind]
    except KeyError as error:
        raise ValueError(f"Unknown CUDA FP8 helper kind {quant_kind!r}") from error
    for op_name in op_names:
        _resolve_compute_op(op_name)


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def ceil_align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def _transform_scale_ue8m0(sf, mn):
    import deep_gemm.utils.layout

    if not sf.is_cuda:
        raise ValueError("UE8M0 scale packing requires a CUDA tensor")
    sf = sf.index_select(-2, torch.arange(mn, device=sf.device) // 128)
    return deep_gemm.utils.layout.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)


def create_per_token_group_quant_fp8_output_scale(
    x_shape,
    device,
    group_size,
    column_major_scales: bool,
    scale_tma_aligned: bool,
    scale_ue8m0: bool,
):
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size <= 0
    ):
        raise ValueError(f"group_size must be a positive integer, got {group_size!r}")
    if len(x_shape) < 2:
        raise ValueError(
            f"FP8 group quantization requires at least 2D shape, got {x_shape}"
        )
    if x_shape[-1] % group_size != 0:
        raise ValueError(
            f"FP8 quantized width {x_shape[-1]} must be divisible by group_size "
            f"{group_size}"
        )
    if scale_ue8m0:
        if not column_major_scales or not scale_tma_aligned:
            raise ValueError(
                "UE8M0 scales require column_major_scales and scale_tma_aligned"
            )
        if group_size != 128:
            raise ValueError(
                f"UE8M0 scale packing requires group_size=128, got {group_size}"
            )
        *x_batch, x_q_mn, x_q_k = x_shape
        x_s_mn, x_s_k = x_q_mn, x_q_k // group_size
        aligned_mn = ceil_align(x_s_mn, 4)
        aligned_k = ceil_align(x_s_k, 4)
        return torch.empty(
            (*x_batch, aligned_k // 4, aligned_mn),
            device=device,
            dtype=torch.int,
        ).transpose(-1, -2)[..., :x_s_mn, :]
    if column_major_scales:
        if scale_tma_aligned:
            aligned_size = (x_shape[-2] + 3) // 4 * 4
            return torch.empty(
                x_shape[:-2] + (x_shape[-1] // group_size, aligned_size),
                device=device,
                dtype=torch.float32,
            ).permute(-1, -2)[: x_shape[-2], :]
        return torch.empty(
            (x_shape[-1] // group_size,) + x_shape[:-1],
            device=device,
            dtype=torch.float32,
        ).permute(-1, -2)
    return torch.empty(
        x_shape[:-1] + (x_shape[-1] // group_size,),
        device=device,
        dtype=torch.float32,
    )


def sgl_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
):
    if x.ndim < 2:
        raise ValueError(
            f"FP8 group quantization requires at least 2D input, got {x.shape}"
        )
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size <= 0
    ):
        raise ValueError(f"group_size must be a positive integer, got {group_size!r}")
    if not x.is_contiguous():
        raise ValueError("FP8 group quantization input must be contiguous")
    if fuse_silu_and_mul and x.shape[-1] % 2 != 0:
        raise ValueError(f"fused SiLU input width must be even, got {x.shape[-1]}")

    out_shape = (*x.shape[:-1], x.shape[-1] // (2 if fuse_silu_and_mul else 1))
    if out_shape[-1] % group_size != 0:
        raise ValueError(
            f"FP8 quantized width {out_shape[-1]} must be divisible by group_size "
            f"{group_size}"
        )
    _validate_native_quant_input(x, "FP8 group quantization", allow_empty=True)
    x_q = torch.empty(out_shape, device=x.device, dtype=fp8_dtype)
    x_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=out_shape,
        device=x.device,
        group_size=group_size,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
    )
    if x.numel() > 0:
        if masked_m is not None:
            with torch.cuda.device(x.device):
                _resolve_compute_op("per_token_group_quant_fp8_v2")(
                    x,
                    x_q,
                    x_s,
                    group_size,
                    eps,
                    fp8_min,
                    fp8_max,
                    scale_ue8m0,
                    fuse_silu_and_mul,
                    masked_m,
                )
        else:
            with torch.cuda.device(x.device):
                _resolve_compute_op("per_token_group_quant_fp8")(
                    x, x_q, x_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0
                )
    return x_q, x_s


def scaled_fp8_per_tensor_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if input.ndim != 2:
        raise ValueError(
            f"FP8 per-tensor quantization requires 2D input, got {input.shape}"
        )
    _validate_native_quant_input(input, "FP8 per-tensor quantization")
    vector_width = 16 // input.element_size()
    if input.numel() % vector_width != 0:
        raise ValueError(
            f"FP8 per-tensor input element count {input.numel()} must be divisible "
            f"by native vector width {vector_width}"
        )
    shape: Union[Tuple[int, int], torch.Size] = input.shape
    out_dtype = torch.float8_e4m3fn
    if output is None:
        output = torch.empty(shape, device=input.device, dtype=out_dtype)
    else:
        _validate_native_output(output, input, out_dtype, "FP8 per-tensor")

    if scale is None:
        scale = torch.zeros(1, device=input.device, dtype=torch.float32)
        with torch.cuda.device(input.device):
            _resolve_compute_op("per_tensor_quant_fp8")(input, output, scale, False)
    else:
        if scale.numel() != 1:
            raise ValueError(
                f"FP8 per-tensor scale must contain one value, got {scale.shape}"
            )
        if scale.device != input.device:
            raise ValueError(f"FP8 scale must be on {input.device}, got {scale.device}")
        if scale.dtype != torch.float32 or not scale.is_contiguous():
            raise TypeError("FP8 per-tensor scale must be contiguous float32")
        with torch.cuda.device(input.device):
            _resolve_compute_op("per_tensor_quant_fp8")(input, output, scale, True)
    return output, scale


def scaled_fp8_per_token_quant(
    input: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if input.ndim != 2:
        raise ValueError(
            f"FP8 per-token quantization requires 2D input, got {input.shape}"
        )
    _validate_native_quant_input(input, "FP8 per-token quantization")
    if input.shape[1] % 8 != 0:
        raise ValueError(
            f"FP8 per-token input width must be divisible by 8, got {input.shape[1]}"
        )
    scale = torch.zeros(input.size(0), device=input.device, dtype=torch.float32)
    if output is not None:
        _validate_native_output(output, input, torch.float8_e4m3fn, "FP8 per-token")
    else:
        output = torch.empty(
            input.shape, device=input.device, dtype=torch.float8_e4m3fn
        )
    with torch.cuda.device(input.device):
        _resolve_compute_op("per_token_quant_fp8")(input, output, scale)
    return output, scale.reshape(-1, 1)


def block_quant_dequant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
    block_size: List[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    if (
        not isinstance(block_size, (list, tuple))
        or len(block_size) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in block_size
        )
    ):
        raise ValueError(
            f"block_size must contain two positive integers, got {block_size!r}"
        )
    if x_q_block.ndim < 2 or x_s.ndim < 2:
        raise ValueError("FP8 block weight and scale tensors must be at least 2D")
    block_n, block_k = block_size[0], block_size[1]
    *_, n, k = x_q_block.shape
    expected_scale_shape = (
        *x_q_block.shape[:-2],
        ceil_div(n, block_n),
        ceil_div(k, block_k),
    )
    if tuple(x_s.shape) != expected_scale_shape:
        raise ValueError(
            f"FP8 block scale shape must be {expected_scale_shape}, got {tuple(x_s.shape)}"
        )
    x_scale_repeat = x_s.repeat_interleave(block_n, dim=-2).repeat_interleave(
        block_k, dim=-1
    )
    x_scale_repeat = x_scale_repeat[..., :n, :k]
    return (x_q_block.to(torch.float32) * x_scale_repeat).to(dtype)


def per_block_cast_to_fp8(
    x: torch.Tensor, use_ue8m0: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 2:
        raise ValueError(f"FP8 block quantization requires 2D input, got {x.shape}")
    m, n = x.shape
    x_padded = torch.zeros(
        (align(m, 128), align(n, 128)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2)
    )


def quant_weight_ue8m0(
    weight_dequant: torch.Tensor,
    weight_block_size: List[int],
):
    if not isinstance(weight_block_size, (list, tuple)) or list(weight_block_size) != [
        128,
        128,
    ]:
        raise ValueError(
            f"UE8M0 weight conversion requires block size [128, 128], got "
            f"{weight_block_size!r}"
        )
    if weight_dequant.dtype != torch.bfloat16:
        raise TypeError(
            "UE8M0 weight conversion requires bfloat16 input, got "
            f"{weight_dequant.dtype} with shape {tuple(weight_dequant.shape)}"
        )
    *batch_dims, n, k = weight_dequant.shape
    weight_dequant_flat = weight_dequant.view((-1, k))
    out_w_flat, out_s_flat = per_block_cast_to_fp8(weight_dequant_flat, True)
    out_w = out_w_flat.view((*batch_dims, n, k))
    out_s = out_s_flat.view(
        (
            *batch_dims,
            ceil_div(n, weight_block_size[0]),
            ceil_div(k, weight_block_size[1]),
        )
    )
    return out_w, out_s


def requant_weight_ue8m0(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    weight_block_size = [128, 128]
    weight_dequant = block_quant_dequant(
        weight, weight_scale_inv, weight_block_size, torch.bfloat16
    )
    out_w, out_s = quant_weight_ue8m0(weight_dequant, weight_block_size)
    return out_w, _transform_scale_ue8m0(out_s, mn=out_w.shape[-2])

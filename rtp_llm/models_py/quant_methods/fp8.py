import logging
from typing import Optional, Tuple

import torch
from torch import nn

from rtp_llm.models_py.quant_methods.base import (
    QuantizeMethodBase,
    register_quant_method,
)

logger = logging.getLogger(__name__)


def _convert_e4m3fn_to_fnuz(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the legacy ROCm loader's lossless FP8 format conversion.

    For an identical bit pattern, e4m3fnuz represents half the e4m3fn value,
    so the scale must be doubled.  The e4m3fn negative-zero pattern (-128)
    represents NaN in e4m3fnuz and is normalized to zero before the bitcast.
    """
    bits = weight.contiguous().view(torch.int8)
    bits[bits == -128] = 0
    return bits.view(torch.float8_e4m3fnuz), scale.float() * 2.0


def _runtime_fp8_dtype() -> torch.dtype:
    if getattr(torch.version, "hip", None) is None:
        return torch.float8_e4m3fn
    from rtp_llm.device.device_impl import is_gfx950

    return torch.float8_e4m3fn if is_gfx950() else torch.float8_e4m3fnuz


def _requant_per_tensor_to_runtime_fp8(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale.numel() != 1:
        raise ValueError(f"FP8 per-tensor scale must be scalar, got {scale.shape}")
    scale_value = float(scale.float().reshape(-1)[0].item())
    if not torch.isfinite(scale.float()).all() or scale_value <= 0:
        raise ValueError(
            f"FP8 per-tensor scale must be finite and positive: {scale_value}"
        )
    runtime_dtype = _runtime_fp8_dtype()
    if weight.dtype == runtime_dtype:
        return weight.contiguous(), scale.view(1).contiguous()
    if weight.dtype == torch.float8_e4m3fn and runtime_dtype == torch.float8_e4m3fnuz:
        converted, converted_scale = _convert_e4m3fn_to_fnuz(weight, scale)
        return converted.contiguous(), converted_scale.view(1).contiguous()

    raise TypeError(
        f"Unsupported FP8 weight conversion from {weight.dtype} to {runtime_dtype}"
    )


def _requant_per_channel_to_runtime_fp8(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    runtime_dtype = _runtime_fp8_dtype()
    scale_rows = scale.float().view(-1, 1)
    if scale_rows.shape[0] != weight.shape[0]:
        raise ValueError(
            f"FP8 per-channel scale/weight mismatch: "
            f"weight={tuple(weight.shape)} scale={tuple(scale.shape)}"
        )
    if not torch.isfinite(scale_rows).all() or bool((scale_rows <= 0).any()):
        raise ValueError("FP8 per-channel scales must be finite and positive")
    if weight.dtype == runtime_dtype:
        return weight.contiguous(), scale_rows.contiguous()

    if weight.dtype == torch.float8_e4m3fn and runtime_dtype == torch.float8_e4m3fnuz:
        converted, converted_scale = _convert_e4m3fn_to_fnuz(weight, scale_rows)
        return converted.contiguous(), converted_scale.contiguous()

    raise TypeError(
        f"Unsupported FP8 weight conversion from {weight.dtype} to {runtime_dtype}"
    )


def _is_hip_runtime() -> bool:
    return getattr(torch.version, "hip", None) is not None


_CUDA_SCALED_MM = "cuda_scaled_mm"
_ROCM_SCALED_MM = "rocm_scaled_mm"
_ROCM_AITER_PTPC = "rocm_aiter_ptpc"
_ROCM_HIPBLASLT_PTPC = "rocm_hipblaslt_ptpc"
_ROCM_AITER_BLOCK = "rocm_aiter_block"
_CUDA_DEEP_GEMM = "cuda_deep_gemm"


def _device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else device.index


def _device_arch(device: torch.device) -> tuple[tuple[int, int], str]:
    try:
        index = _device_index(device)
        capability = torch.cuda.get_device_capability(index)
        properties = torch.cuda.get_device_properties(index)
    except (AssertionError, RuntimeError, ValueError) as error:
        raise RuntimeError(f"Unable to query FP8 capability for {device}: {error}")
    return capability, getattr(properties, "gcnArchName", "")


def _aiter_has_symbol(symbol: str) -> bool:
    try:
        import aiter
    except ImportError:
        return False
    return callable(getattr(aiter, symbol, None))


def _use_rocm_swizzle_a(hw_kernel_config: object) -> bool:
    value = getattr(hw_kernel_config, "use_swizzleA", False)
    if not isinstance(value, bool):
        raise TypeError("hw_kernel_config.use_swizzleA must be a bool")
    return value


def _rocm_cktile_ptpc_available() -> bool:
    try:
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle_cktile
    except ImportError:
        return False
    return callable(gemm_a8w8_bpreshuffle_cktile)


def _select_fp8_runtime_backend(
    device: torch.device,
    quant_kind: str,
    hw_kernel_config: object = None,
) -> str:
    """Resolve an executable FP8 backend before retaining runtime weights."""
    device = torch.device(device)
    if quant_kind not in ("per_tensor", "per_channel", "block"):
        raise ValueError(f"Unknown FP8 quantization kind {quant_kind!r}")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            f"FP8 {quant_kind} requires a supported accelerator, got {device}"
        )

    capability, gcn_arch = _device_arch(device)
    if _is_hip_runtime():
        supported_arch = any(name in gcn_arch for name in ("gfx942", "gfx950"))
        if not supported_arch:
            raise RuntimeError(
                f"FP8 {quant_kind} is not supported on ROCm architecture "
                f"{gcn_arch or capability}"
            )
        if quant_kind == "per_tensor" and hasattr(torch, "_scaled_mm"):
            return _ROCM_SCALED_MM
        if quant_kind == "per_channel":
            if _use_rocm_swizzle_a(hw_kernel_config):
                if _aiter_has_symbol("hipb_create_extension") and _aiter_has_symbol(
                    "hipb_mm"
                ):
                    return _ROCM_HIPBLASLT_PTPC
                raise RuntimeError(
                    "ROCm FP8 per-channel use_swizzleA requires AITer hipBLASLt"
                )
            if (
                _aiter_has_symbol("gemm_a8w8_bpreshuffle")
                and _rocm_cktile_ptpc_available()
            ):
                return _ROCM_AITER_PTPC
            raise RuntimeError(
                "ROCm FP8 per-channel without use_swizzleA requires both "
                "AITer and CKTile PTPC kernels"
            )
        if quant_kind == "block" and _aiter_has_symbol(
            "gemm_a8w8_blockscale_bpreshuffle"
        ):
            return _ROCM_AITER_BLOCK
        raise RuntimeError(
            f"No executable ROCm FP8 {quant_kind} backend is available on {gcn_arch}"
        )

    if quant_kind == "block":
        if is_deep_gemm_runtime_available(device):
            return _CUDA_DEEP_GEMM
        raise RuntimeError(
            f"FP8 block requires DeepGEMM on CUDA device {device}; "
            f"current capability is {capability}"
        )
    if hasattr(torch, "_scaled_mm") and capability >= (8, 9):
        return _CUDA_SCALED_MM
    raise RuntimeError(
        f"FP8 {quant_kind} requires torch._scaled_mm on CUDA SM89 or newer; "
        f"device {device} has capability {capability}"
    )


def _shuffle_rocm_fp8_weight(weight: torch.Tensor) -> torch.Tensor:
    from aiter.ops.shuffle import shuffle_weight

    return shuffle_weight(weight, layout=(16, 16))


def _prepare_rocm_fp8_ptpc_executor(
    layer,
    weight: torch.Tensor,
    scale: torch.Tensor,
    runtime_backend: str,
):
    """Materialize the exact layout consumed by the legacy ROCm strategies."""
    if runtime_backend == _ROCM_HIPBLASLT_PTPC:
        from rtp_llm.models_py.modules.factory.linear.impl.rocm.fp8_ptpc_linear import (
            RocmFp8PTPCLinearWithSwizzle,
        )
        from rtp_llm.utils.swizzle_utils import swizzle_tensor

        runtime_weight = swizzle_tensor(weight.contiguous(), False).T
        runtime_scale = scale.float().reshape(-1, 1).T.contiguous()
        executor_type = RocmFp8PTPCLinearWithSwizzle
    elif runtime_backend == _ROCM_AITER_PTPC:
        from rtp_llm.models_py.modules.factory.linear.impl.rocm.fp8_ptpc_linear import (
            run_rocm_fp8_ptpc_no_swizzle,
        )

        runtime_weight = _shuffle_rocm_fp8_weight(weight).contiguous()
        runtime_scale = scale.float().reshape(-1, 1).contiguous()
        executor_type = run_rocm_fp8_ptpc_no_swizzle
    else:
        raise RuntimeError(f"Unsupported ROCm FP8 PTPC backend {runtime_backend!r}")

    del layer.weight
    layer.register_parameter(
        "weight", nn.Parameter(runtime_weight, requires_grad=False)
    )
    del layer.weight_scale
    layer.register_parameter(
        "weight_scale", nn.Parameter(runtime_scale, requires_grad=False)
    )
    if runtime_backend == _ROCM_AITER_PTPC:
        return executor_type
    return executor_type(
        weight=layer.weight, weight_scales=layer.weight_scale, bias=None
    )


def _rocm_fp8_ptpc_executor_name(runtime_backend: str, executor) -> str:
    if runtime_backend not in (_ROCM_AITER_PTPC, _ROCM_HIPBLASLT_PTPC):
        return "torch._scaled_mm"
    if executor is None:
        return runtime_backend
    return getattr(executor, "__name__", type(executor).__name__)


def _run_rocm_fp8_ptpc_executor(
    executor,
    runtime_backend: str,
    input_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    if executor is None:
        raise RuntimeError("ROCm FP8 PTPC executor was not initialized")
    if runtime_backend == _ROCM_AITER_PTPC:
        return executor(input_2d, weight, weight_scale, output_size)
    if runtime_backend == _ROCM_HIPBLASLT_PTPC:
        return executor(input_2d)
    raise RuntimeError(f"Unsupported ROCm FP8 PTPC backend {runtime_backend!r}")


def _apply_rocm_fp8_block(
    input_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
) -> torch.Tensor:
    import aiter

    from rtp_llm.models_py.kernels.rocm.fp8_kernel import rocm_per_token_group_quant_fp8

    original_dtype = input_2d.dtype
    input_bf16 = (
        input_2d if original_dtype == torch.bfloat16 else input_2d.to(torch.bfloat16)
    )
    qinput, input_scale = rocm_per_token_group_quant_fp8(
        input_bf16,
        group_size=block_size,
        eps=1e-4,
        column_major_scales=False,
        scale_tma_aligned=False,
    )
    fp8_max = float(torch.finfo(qinput.dtype).max)
    input_scale = torch.clamp(input_scale, min=1e-4 / fp8_max).to(torch.float32)
    shuffled_input_scale = input_scale.transpose(0, 1).contiguous().view_as(input_scale)
    output = aiter.gemm_a8w8_blockscale_bpreshuffle(
        qinput,
        weight,
        shuffled_input_scale,
        weight_scale,
    )
    return output if original_dtype == torch.bfloat16 else output.to(out_dtype)


def _validate_fp8_block_scales(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_n: int,
    block_k: int,
) -> None:
    if block_n <= 0 or block_k <= 0:
        raise ValueError(f"Invalid FP8 block size {(block_n, block_k)}")
    expected = (
        (weight.shape[0] + block_n - 1) // block_n,
        (weight.shape[1] + block_k - 1) // block_k,
    )
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"FP8 block scale shape must be {expected}, got {tuple(scale.shape)}"
        )
    if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise ValueError("FP8 block scales must be finite and positive")


def _prepare_rocm_fp8_block_weight(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    runtime_dtype = _runtime_fp8_dtype()
    if weight.dtype == runtime_dtype:
        runtime_weight = weight.contiguous()
        runtime_scale = scale.to(torch.float32).contiguous()
    elif weight.dtype == torch.float8_e4m3fn and runtime_dtype == torch.float8_e4m3fnuz:
        runtime_weight, runtime_scale = _convert_e4m3fn_to_fnuz(weight, scale)
    else:
        raise TypeError(
            f"Unsupported ROCm FP8 block conversion from {weight.dtype} "
            f"to {runtime_dtype}"
        )
    return (
        _shuffle_rocm_fp8_weight(runtime_weight).contiguous(),
        runtime_scale.to(torch.float32).contiguous(),
    )


# Hoist kernel imports to module scope. apply() is on the per-token decode hot
# path: doing the import inside apply() (even though sys.modules caches it)
# still costs a sys.modules lookup + LOAD_ATTR on every call. Importing once at
# module load amortizes that cost across the lifetime of the process.
#
# Each method that uses a kernel still falls back to a lazy `_resolve_*()` helper
# below when the module-level symbol is None. The FP8 provider itself is imported
# only after runtime dispatch selects an FP8 key.
try:
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
        requant_weight_ue8m0,
        scaled_fp8_per_tensor_quant,
        scaled_fp8_per_token_quant,
        sgl_per_token_group_quant_fp8,
    )
except (
    ImportError
) as e:  # pragma: no cover - kernel package may be absent on CPU-only setups
    logger.warning(
        "fp8 kernel imports unavailable: %s (will fall back to lazy import)", e
    )
    requant_weight_ue8m0 = None
    scaled_fp8_per_tensor_quant = None
    scaled_fp8_per_token_quant = None
    sgl_per_token_group_quant_fp8 = None

try:
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
        fp8_gemm_nt,
        is_deep_gemm_e8m0_used,
        is_deep_gemm_runtime_available,
    )
except ImportError as e:  # pragma: no cover
    logger.warning(
        "deepgemm_wrapper imports unavailable: %s (will fall back to lazy import)", e
    )
    fp8_gemm_nt = None

    def is_deep_gemm_runtime_available(  # type: ignore[no-redef]
        device: Optional[torch.device] = None,
    ) -> bool:
        return False

    def is_deep_gemm_e8m0_used() -> bool:  # type: ignore[no-redef]
        return False


def _hip_scaled_fp8_per_tensor_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if input.ndim != 2:
        raise ValueError(
            f"FP8 per-tensor quantization requires 2D input, got {input.shape}"
        )
    if not input.is_cuda:
        raise ValueError("ROCm FP8 per-tensor quantization requires a GPU tensor")
    if not input.is_contiguous():
        input = input.contiguous()
    runtime_dtype = _runtime_fp8_dtype()
    from rtp_llm.models_py.kernels.rocm.fp8_kernel import (
        dynamic_per_tensor_quant,
        static_per_tensor_quant,
    )

    if output is None:
        output = torch.empty_like(input, dtype=runtime_dtype)
    elif output.shape != input.shape or output.dtype != runtime_dtype:
        raise ValueError(
            f"FP8 output must have shape {tuple(input.shape)} and dtype "
            f"{runtime_dtype}, got {tuple(output.shape)} and {output.dtype}"
        )
    elif output.device != input.device:
        raise ValueError(f"FP8 output must be on {input.device}, got {output.device}")

    if scale is None:
        scale = torch.empty(1, dtype=torch.float32, device=input.device)
        dynamic_per_tensor_quant(output, input, scale)
    else:
        static_per_tensor_quant(output, input, scale.reshape(1))
    return output, scale


def _resolve_per_tensor_quant():
    """Return scaled_fp8_per_tensor_quant, lazy-importing on a hoist miss."""
    if _is_hip_runtime():
        return _hip_scaled_fp8_per_tensor_quant
    global scaled_fp8_per_tensor_quant
    if scaled_fp8_per_tensor_quant is None:
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            scaled_fp8_per_tensor_quant as _fn,
        )

        scaled_fp8_per_tensor_quant = _fn
    return scaled_fp8_per_tensor_quant


def per_block_quant_like_legacy(
    input: torch.Tensor, block_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block FP8 quantization without importing the legacy loader graph."""
    if input.dim() != 2:
        raise ValueError(f"Block FP8 quantization requires 2D input, got {input.shape}")
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 0
    ):
        raise ValueError(f"block_size must be a positive integer, got {block_size!r}")
    rows, columns = input.shape
    padded_rows = (rows + block_size - 1) // block_size * block_size
    padded_columns = (columns + block_size - 1) // block_size * block_size
    padded = torch.zeros(
        padded_rows, padded_columns, dtype=torch.float32, device=input.device
    )
    padded[:rows, :columns].copy_(input)
    blocks = padded.view(
        padded_rows // block_size,
        block_size,
        padded_columns // block_size,
        block_size,
    )
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-4)
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    quantized = (blocks * (fp8_max / amax)).to(torch.float8_e4m3fn)
    quantized = quantized.view(padded_rows, padded_columns)[:rows, :columns]
    scales = (amax / fp8_max).squeeze(1).squeeze(-1).to(torch.float32)
    return quantized.contiguous(), scales.contiguous()


def _resolve_per_token_quant():
    """Return scaled_fp8_per_token_quant, lazy-importing on a hoist miss."""
    if _is_hip_runtime():
        from rtp_llm.models_py.kernels.rocm.fp8_kernel import rocm_per_token_quant_fp8

        return rocm_per_token_quant_fp8
    global scaled_fp8_per_token_quant
    if scaled_fp8_per_token_quant is None:
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            scaled_fp8_per_token_quant as _fn,
        )

        scaled_fp8_per_token_quant = _fn
    return scaled_fp8_per_token_quant


def _resolve_requant_weight_ue8m0():
    """Return requant_weight_ue8m0, lazy-importing on a hoist miss."""
    global requant_weight_ue8m0
    if requant_weight_ue8m0 is None:
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            requant_weight_ue8m0 as _fn,
        )

        requant_weight_ue8m0 = _fn
    return requant_weight_ue8m0


def _resolve_sgl_per_token_group_quant():
    """Return sgl_per_token_group_quant_fp8, lazy-importing on a hoist miss."""
    global sgl_per_token_group_quant_fp8
    if sgl_per_token_group_quant_fp8 is None:
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8 as _fn,
        )

        sgl_per_token_group_quant_fp8 = _fn
    return sgl_per_token_group_quant_fp8


def _resolve_fp8_gemm_nt():
    """Return fp8_gemm_nt, lazy-importing on a hoist miss."""
    global fp8_gemm_nt
    if fp8_gemm_nt is None:
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_gemm_nt as _fn

        fp8_gemm_nt = _fn
    return fp8_gemm_nt


@register_quant_method("fp8", "FP8_PER_TENSOR_COMPRESSED")
class Fp8LinearMethod(QuantizeMethodBase):
    """Already-quantized FP8 per-tensor (e.g. compressed-tensors) ckpt loader.

    ckpt provides:
      - weight: float8_e4m3fn [N, K]
      - weight_scale: fp32 scalar (per-tensor)
      - input_scale: fp32 scalar (optional; static activation scale)

    Forward: torch._scaled_mm with either the checkpoint's static activation
    scale or a dynamically computed per-tensor scale, as declared by the
    source quantization config.
    """

    # Class-level flag: the diagnostic log fires at most once across ALL
    # instances in a process. See Fp8OnlineLinearMethod for the assumption.
    _apply_logged: bool = False

    @property
    def required_checkpoint_parameters(self) -> Tuple[str, ...]:
        required = ["weight", "weight_scale"]
        if not self.quant_config.activation_dynamic:
            required.append("input_scale")
        return tuple(required)

    def create_weights(
        self,
        layer,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **kwargs,
    ):
        weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.register_parameter("weight", weight)

        weight_scale = nn.Parameter(
            torch.ones(1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("weight_scale", weight_scale)

        input_scale = nn.Parameter(
            torch.ones(1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("input_scale", input_scale)

    def validate_runtime_device(self, device: torch.device) -> None:
        _select_fp8_runtime_backend(device, "per_tensor")

    def apply(
        self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if getattr(self, "_runtime_backend", None) not in (
            _CUDA_SCALED_MM,
            _ROCM_SCALED_MM,
        ):
            raise RuntimeError("FP8 per-tensor backend was not initialized")
        out_dtype = x.dtype
        input_2d = x.reshape(-1, x.shape[-1])
        if not input_2d.is_contiguous():
            input_2d = input_2d.contiguous()
        out_features = layer.weight.shape[0]
        output_shape = list(x.shape[:-1]) + [out_features]

        input_scale = (
            None if self.quant_config.activation_dynamic else layer.input_scale
        )
        qinput, x_scale = _resolve_per_tensor_quant()(input_2d, input_scale)

        if not Fp8LinearMethod._apply_logged:
            logger.info(
                "[Fp8LinearMethod] FIRST forward via torch._scaled_mm: "
                "prefix=%r, x.shape=%s, weight.shape=%s",
                getattr(layer, "prefix", "?"),
                tuple(x.shape),
                tuple(layer.weight.shape),
            )
            Fp8LinearMethod._apply_logged = True

        output = torch._scaled_mm(
            qinput,
            layer.weight.t(),
            scale_a=x_scale,
            scale_b=layer.weight_scale,
            bias=bias,
            out_dtype=out_dtype,
        )
        if isinstance(output, tuple):
            output = output[0]
        return output.view(*output_shape)

    def process_weights_after_loading(self, layer):
        self._runtime_backend = _select_fp8_runtime_backend(
            layer.weight.device, "per_tensor"
        )
        fp8_weight, scale = _requant_per_tensor_to_runtime_fp8(
            layer.weight.data, layer.weight_scale.data
        )
        del layer.weight
        layer.register_parameter(
            "weight", nn.Parameter(fp8_weight, requires_grad=False)
        )
        del layer.weight_scale
        layer.register_parameter(
            "weight_scale", nn.Parameter(scale, requires_grad=False)
        )
        if hasattr(layer, "input_scale") and layer.input_scale.dim() == 0:
            layer.input_scale = nn.Parameter(
                layer.input_scale.reshape(1), requires_grad=False
            )
        if not self.quant_config.activation_dynamic:
            input_scale = layer.input_scale.detach().float().reshape(-1)
            if input_scale.numel() != 1:
                raise ValueError(
                    "Static FP8 per-tensor input_scale must contain one value"
                )
            if not bool(torch.isfinite(input_scale).all()) or bool(
                (input_scale <= 0).any()
            ):
                raise ValueError(
                    "Static FP8 per-tensor input_scale must be finite and positive"
                )


@register_quant_method("fp8_online", "FP8_DYNAMIC_PER_TENSOR")
class Fp8OnlineLinearMethod(QuantizeMethodBase):
    """Online FP8 per-tensor quantization.

    Loads BF16/FP16 weights from ckpt, quantizes to float8_e4m3fn at load time,
    and runs forward via real FP8 GEMM (torch._scaled_mm) with dynamic per-tensor
    activation quantization.
    """

    # Class-level flags: the diagnostic log fires at most once across ALL
    # instances in a process. Safe under RTP-LLM's process model where each
    # backend process loads the model once and does not reload. If dynamic
    # reload (e.g. hot-swap, multi-model serving) is ever added, move these to
    # __init__ so each instance can log once.
    _quant_logged: bool = False
    _apply_logged: bool = False
    _quant_count: int = 0
    required_checkpoint_parameters = ("weight",)

    def create_weights(
        self,
        layer,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **kwargs,
    ):
        # Stage as params_dtype so existing param.data.copy_(bf16_tensor) in
        # linear.py's load_weights() works. process_weights_after_loading will
        # replace this Parameter with a fp8 one.
        weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("weight", weight)

        layer.register_parameter(
            "weight_scale",
            nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False),
        )

    def validate_runtime_device(self, device: torch.device) -> None:
        _select_fp8_runtime_backend(device, "per_tensor")

    def process_weights_after_loading(self, layer):
        weight = layer.weight.data
        if weight.ndim != 2:
            raise ValueError(f"expected 2D weight, got {weight.shape}")

        self._runtime_backend = _select_fp8_runtime_backend(weight.device, "per_tensor")
        fp8_weight, scale = _resolve_per_tensor_quant()(weight)

        # nn.Parameter dtype is immutable; rebind both attributes as a
        # consistent accelerator-resident FP8/FP32 runtime pair.
        del layer.weight
        layer.register_parameter(
            "weight", nn.Parameter(fp8_weight, requires_grad=False)
        )
        del layer.weight_scale
        layer.register_parameter(
            "weight_scale",
            nn.Parameter(scale.view(1).contiguous(), requires_grad=False),
        )

        Fp8OnlineLinearMethod._quant_count += 1
        if not Fp8OnlineLinearMethod._quant_logged:
            logger.info(
                "[Fp8OnlineLinearMethod] online-quantized FIRST weight: "
                "prefix=%r, weight.shape=%s, weight.dtype=%s, "
                "scale=%.6f, weight.device=%s",
                getattr(layer, "prefix", "?"),
                tuple(fp8_weight.shape),
                fp8_weight.dtype,
                float(scale.view(1)[0].item()),
                fp8_weight.device,
            )
            Fp8OnlineLinearMethod._quant_logged = True

    def apply(
        self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if getattr(self, "_runtime_backend", None) not in (
            _CUDA_SCALED_MM,
            _ROCM_SCALED_MM,
        ):
            raise RuntimeError("FP8 per-tensor backend was not initialized")
        out_dtype = x.dtype
        input_2d = x.reshape(-1, x.shape[-1])
        if not input_2d.is_contiguous():
            input_2d = input_2d.contiguous()
        out_features = layer.weight.shape[0]
        output_shape = list(x.shape[:-1]) + [out_features]

        qinput, x_scale = _resolve_per_tensor_quant()(input_2d)

        if not Fp8OnlineLinearMethod._apply_logged:
            logger.info(
                "[Fp8OnlineLinearMethod] FIRST forward via torch._scaled_mm: "
                "prefix=%r, x.dtype=%s, x.shape=%s, qinput.dtype=%s, "
                "weight.dtype=%s, weight.shape=%s, total_quanted_weights=%d",
                getattr(layer, "prefix", "?"),
                x.dtype,
                tuple(x.shape),
                qinput.dtype,
                layer.weight.dtype,
                tuple(layer.weight.shape),
                Fp8OnlineLinearMethod._quant_count,
            )
            Fp8OnlineLinearMethod._apply_logged = True

        output = torch._scaled_mm(
            qinput,
            layer.weight.t(),
            scale_a=x_scale,
            scale_b=layer.weight_scale,
            bias=bias,
            out_dtype=out_dtype,
        )
        if isinstance(output, tuple):
            output = output[0]

        return output.view(*output_shape)


@register_quant_method("fp8_per_channel_online")
class Fp8PerChannelOnlineLinearMethod(QuantizeMethodBase):
    """Online FP8 per-channel quantization for the new loader.

    Loads BF16/FP16 weights from ckpt, quantizes to float8_e4m3fn at load time
    with one fp32 scale per output channel (per-row of [N, K]), and runs forward
    via torch._scaled_mm with online per-token activation quantization
    (one fp32 scale per token-row of [M, K]).

    Same `per_token_quant_fp8` CUDA op is reused for both weight quant
    (rows of [N, K]) and activation quant (rows of [M, K]).
    """

    # Class-level flags: the diagnostic log fires at most once across ALL
    # instances in a process. Safe under RTP-LLM's process model where each
    # backend process loads the model once and does not reload. If dynamic
    # reload (e.g. hot-swap, multi-model serving) is ever added, move these to
    # __init__ so each instance can log once.
    _quant_logged: bool = False
    _apply_logged: bool = False
    _quant_count: int = 0
    required_checkpoint_parameters = ("weight",)

    def create_weights(
        self,
        layer,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **kwargs,
    ):
        # Stage as params_dtype (bf16) so existing param.data.copy_ flows in
        # linear.py work; process_weights_after_loading replaces with fp8.
        weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("weight", weight)

        layer.register_parameter(
            "weight_scale",
            nn.Parameter(
                torch.ones(output_size, 1, dtype=torch.float32),
                requires_grad=False,
            ),
        )

    def validate_runtime_device(self, device: torch.device) -> None:
        _select_fp8_runtime_backend(
            device,
            "per_channel",
            self.quant_config.hw_kernel_config,
        )

    def process_weights_after_loading(self, layer):
        weight = layer.weight.data
        if weight.ndim != 2:
            raise ValueError(f"expected 2D weight, got {weight.shape}")
        self._runtime_backend = _select_fp8_runtime_backend(
            weight.device,
            "per_channel",
            self.quant_config.hw_kernel_config,
        )

        # Treat weight rows as "tokens" -> per-output-channel quant.
        # fp8_weight: [N, K] float8_e4m3fn, scale: [N, 1] float32.
        is_hip = self._runtime_backend in (
            _ROCM_AITER_PTPC,
            _ROCM_HIPBLASLT_PTPC,
        )
        quant_input = (
            weight.to(torch.bfloat16)
            if is_hip and weight.dtype != torch.bfloat16
            else weight
        )
        fp8_weight, scale = _resolve_per_token_quant()(quant_input)
        if is_hip:
            self._rocm_executor = _prepare_rocm_fp8_ptpc_executor(
                layer,
                fp8_weight,
                scale,
                self._runtime_backend,
            )
        else:
            # torch._scaled_mm consumes scale_b as [1, N]. Materialize that
            # layout once after loading instead of transposing every forward.
            scale = scale.view(1, -1).contiguous()

            del layer.weight
            layer.register_parameter(
                "weight", nn.Parameter(fp8_weight.contiguous(), requires_grad=False)
            )
            del layer.weight_scale
            layer.register_parameter(
                "weight_scale",
                nn.Parameter(scale, requires_grad=False),
            )

        Fp8PerChannelOnlineLinearMethod._quant_count += 1
        if not Fp8PerChannelOnlineLinearMethod._quant_logged:
            logger.info(
                "[Fp8PerChannelOnlineLinearMethod] online-quantized FIRST weight: "
                "prefix=%r, weight.shape=%s, weight.dtype=%s, "
                "scale.shape=%s, scale.mean=%.6f, weight.device=%s",
                getattr(layer, "prefix", "?"),
                tuple(fp8_weight.shape),
                fp8_weight.dtype,
                tuple(scale.shape),
                float(scale.mean().item()),
                fp8_weight.device,
            )
            Fp8PerChannelOnlineLinearMethod._quant_logged = True

    def apply(
        self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if getattr(self, "_runtime_backend", None) not in (
            _CUDA_SCALED_MM,
            _ROCM_AITER_PTPC,
            _ROCM_HIPBLASLT_PTPC,
        ):
            raise RuntimeError("FP8 per-channel backend was not initialized")
        out_dtype = x.dtype
        input_2d = x.reshape(-1, x.shape[-1])
        if not input_2d.is_contiguous():
            input_2d = input_2d.contiguous()
        N = layer.output_size
        output_shape = list(x.shape[:-1]) + [N]

        if not Fp8PerChannelOnlineLinearMethod._apply_logged:
            logger.info(
                "[Fp8PerChannelOnlineLinearMethod] FIRST forward via %s: "
                "prefix=%r, x.dtype=%s, x.shape=%s, weight.dtype=%s, "
                "weight.shape=%s, weight_scale.shape=%s, total_quanted_weights=%d",
                _rocm_fp8_ptpc_executor_name(
                    self._runtime_backend,
                    getattr(self, "_rocm_executor", None),
                ),
                getattr(layer, "prefix", "?"),
                x.dtype,
                tuple(x.shape),
                layer.weight.dtype,
                tuple(layer.weight.shape),
                tuple(layer.weight_scale.shape),
                Fp8PerChannelOnlineLinearMethod._quant_count,
            )
            Fp8PerChannelOnlineLinearMethod._apply_logged = True

        if self._runtime_backend in (
            _ROCM_AITER_PTPC,
            _ROCM_HIPBLASLT_PTPC,
        ):
            output = _run_rocm_fp8_ptpc_executor(
                getattr(self, "_rocm_executor", None),
                self._runtime_backend,
                input_2d,
                layer.weight,
                layer.weight_scale,
                layer.output_size,
            )
            if bias is not None:
                output = output + bias.to(output.dtype)
        else:
            # Per-token activation quant: [M, K] -> fp8 + [M, 1] fp32.
            qinput, x_scale = _resolve_per_token_quant()(input_2d)
            output = torch._scaled_mm(
                qinput,
                layer.weight.t(),
                scale_a=x_scale,
                scale_b=layer.weight_scale,
                bias=bias,
                out_dtype=out_dtype,
            )
            if isinstance(output, tuple):
                output = output[0]

        return output.view(*output_shape)


@register_quant_method(
    "fp8_per_channel",
    "FP8_PER_CHANNEL_COMPRESSED",
    "FP8_PER_CHANNEL_QUARK",
)
class Fp8PerChannelLinearMethod(QuantizeMethodBase):
    """Already-quantized FP8 per-channel ckpt loader (compressed-tensors / Quark).

    ckpt provides:
      - weight: float8_e4m3fn [N, K]
      - weight_scale: fp32 [N] or [N, 1]

    On ROCm, forward uses the same AITer per-token quantization, shuffled
    weight layout and gemm_a8w8_bpreshuffle kernel as the legacy loader.  This
    is required for old/new loader numerical parity.  Other platforms use
    torch._scaled_mm.
    """

    # Class-level flag: the diagnostic log fires at most once across ALL
    # instances in a process. See Fp8OnlineLinearMethod for the assumption.
    _apply_logged: bool = False
    required_checkpoint_parameters = ("weight", "weight_scale")

    def create_weights(
        self,
        layer,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **kwargs,
    ):
        weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.register_parameter("weight", weight)

        layer.register_parameter(
            "weight_scale",
            nn.Parameter(
                torch.ones(output_size, 1, dtype=torch.float32),
                requires_grad=False,
            ),
        )

    def validate_runtime_device(self, device: torch.device) -> None:
        _select_fp8_runtime_backend(
            device,
            "per_channel",
            self.quant_config.hw_kernel_config,
        )

    def process_weights_after_loading(self, layer):
        # ckpt scale may arrive as [N] or [N, 1]. ROCm fp8 kernels expect the
        # platform runtime dtype (e4m3fnuz on MI308X), so convert already-FP8
        # checkpoint weights once after loading and store scale as [N, 1].
        self._runtime_backend = _select_fp8_runtime_backend(
            layer.weight.device,
            "per_channel",
            self.quant_config.hw_kernel_config,
        )
        fp8_weight, scale = _requant_per_channel_to_runtime_fp8(
            layer.weight.data, layer.weight_scale.data
        )
        is_hip = self._runtime_backend in (
            _ROCM_AITER_PTPC,
            _ROCM_HIPBLASLT_PTPC,
        )
        if is_hip:
            self._rocm_executor = _prepare_rocm_fp8_ptpc_executor(
                layer,
                fp8_weight,
                scale,
                self._runtime_backend,
            )
        else:
            # torch._scaled_mm consumes scale_b as [1, N]. Materialize that
            # layout once after loading instead of transposing every forward.
            scale = scale.t().contiguous()
            del layer.weight
            layer.register_parameter(
                "weight", nn.Parameter(fp8_weight, requires_grad=False)
            )
            del layer.weight_scale
            layer.register_parameter(
                "weight_scale",
                nn.Parameter(scale, requires_grad=False),
            )

    def apply(
        self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if getattr(self, "_runtime_backend", None) not in (
            _CUDA_SCALED_MM,
            _ROCM_AITER_PTPC,
            _ROCM_HIPBLASLT_PTPC,
        ):
            raise RuntimeError("FP8 per-channel backend was not initialized")
        out_dtype = x.dtype
        input_2d = x.reshape(-1, x.shape[-1])
        if not input_2d.is_contiguous():
            input_2d = input_2d.contiguous()
        N = layer.output_size
        output_shape = list(x.shape[:-1]) + [N]

        if not Fp8PerChannelLinearMethod._apply_logged:
            logger.info(
                "[Fp8PerChannelLinearMethod] FIRST forward via %s: "
                "prefix=%r, x.shape=%s, weight.shape=%s, weight_scale.shape=%s",
                _rocm_fp8_ptpc_executor_name(
                    self._runtime_backend,
                    getattr(self, "_rocm_executor", None),
                ),
                getattr(layer, "prefix", "?"),
                tuple(x.shape),
                tuple(layer.weight.shape),
                tuple(layer.weight_scale.shape),
            )
            Fp8PerChannelLinearMethod._apply_logged = True

        if self._runtime_backend in (
            _ROCM_AITER_PTPC,
            _ROCM_HIPBLASLT_PTPC,
        ):
            output = _run_rocm_fp8_ptpc_executor(
                getattr(self, "_rocm_executor", None),
                self._runtime_backend,
                input_2d,
                layer.weight,
                layer.weight_scale,
                layer.output_size,
            )
            if bias is not None:
                output = output + bias.to(output.dtype)
        else:
            qinput, x_scale = _resolve_per_token_quant()(input_2d)
            output = torch._scaled_mm(
                qinput,
                layer.weight.t(),
                scale_a=x_scale,
                scale_b=layer.weight_scale,
                bias=bias,
                out_dtype=out_dtype,
            )
            if isinstance(output, tuple):
                output = output[0]
        return output.view(*output_shape)


@register_quant_method("fp8_block_online")
class Fp8BlockOnlineLinearMethod(QuantizeMethodBase):
    """Online FP8 per-block (128x128) quantization for the new loader.

    Loads BF16/FP16 weights from ckpt, quantizes to float8_e4m3fn at load time
    using DeepSeek-style 128x128 block scales. CUDA runs DeepGEMM and ROCm
    runs the existing AITer blockscale kernel with online group activation
    quantization.
    """

    BLOCK: int = 128

    # Class-level flags: the diagnostic log fires at most once across ALL
    # instances in a process. Safe under RTP-LLM's process model where each
    # backend process loads the model once and does not reload. If dynamic
    # reload (e.g. hot-swap, multi-model serving) is ever added, move these to
    # __init__ so each instance can log once.
    _quant_logged: bool = False
    _apply_logged: bool = False
    _quant_count: int = 0
    required_checkpoint_parameters = ("weight",)

    def __init__(self, quant_config=None):
        super().__init__(quant_config)
        self._runtime_backend = None
        self._use_deep_gemm = False

    def validate_runtime_device(self, device: torch.device) -> None:
        _select_fp8_runtime_backend(device, "block")

    def create_weights(
        self,
        layer,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **kwargs,
    ):
        configured_block_size = list(
            getattr(
                layer.quant_config,
                "weight_block_size",
                [self.BLOCK, self.BLOCK],
            )
        )
        if configured_block_size != [self.BLOCK, self.BLOCK]:
            raise ValueError(
                "Online FP8 block quantization supports only "
                f"{[self.BLOCK, self.BLOCK]}, got {configured_block_size}"
            )
        # Stage as params_dtype (bf16) so existing param.data.copy_ flows in
        # linear.py work; process_weights_after_loading replaces with fp8.
        backing = torch.empty(input_size, output_size, dtype=params_dtype)
        weight = nn.Parameter(backing.T, requires_grad=False)
        layer.register_parameter("weight", weight)

        n_blocks = (output_size + self.BLOCK - 1) // self.BLOCK
        k_blocks = (input_size + self.BLOCK - 1) // self.BLOCK
        layer.register_parameter(
            "weight_scale",
            nn.Parameter(
                torch.ones(n_blocks, k_blocks, dtype=torch.float32),
                requires_grad=False,
            ),
        )

    def process_weights_after_loading(self, layer):
        weight = layer.weight.data
        self._runtime_backend = _select_fp8_runtime_backend(weight.device, "block")
        self._use_deep_gemm = self._runtime_backend == _CUDA_DEEP_GEMM
        if weight.ndim != 2:
            raise ValueError(f"expected 2D weight, got {weight.shape}")

        logical_n, logical_k = weight.shape
        padded_k = (logical_k + self.BLOCK - 1) // self.BLOCK * self.BLOCK
        shard_names = getattr(layer, "shard_names", None)
        num_shards = len(shard_names) if shard_names else 1

        if num_shards > 1:
            if hasattr(layer, "q_size") and hasattr(layer, "kv_size"):
                shard_sizes = [layer.q_size, layer.kv_size, layer.kv_size]
            elif logical_n % num_shards == 0:
                shard_sizes = [logical_n // num_shards] * num_shards
            else:
                raise ValueError(
                    f"cannot infer FP8 shard sizes for merged output {logical_n} "
                    f"with {num_shards} shards in {getattr(layer, 'prefix', '?')}"
                )
            if len(shard_sizes) != num_shards or sum(shard_sizes) != logical_n:
                raise ValueError(
                    f"FP8 shard sizes {shard_sizes} do not match merged output "
                    f"{logical_n} in {getattr(layer, 'prefix', '?')}"
                )
            quant_shards = []
            scale_shards = []
            runtime_shard_sizes = []
            for shard, shard_n in zip(
                torch.split(weight, shard_sizes, dim=0), shard_sizes
            ):
                if shard_n % self.BLOCK != 0:
                    raise ValueError(
                        f"FP8 block quantization requires merged shard rows to align "
                        f"to {self.BLOCK}; got {shard_n} in "
                        f"{getattr(layer, 'prefix', '?')}"
                    )
                padded_shard_n = (shard_n + self.BLOCK - 1) // self.BLOCK * self.BLOCK
                quant_shard, scale_shard = per_block_quant_like_legacy(
                    shard, self.BLOCK
                )
                quant_shard = torch.nn.functional.pad(
                    quant_shard,
                    (0, padded_k - logical_k, 0, padded_shard_n - shard_n),
                )
                quant_shards.append(quant_shard)
                scale_shards.append(scale_shard)
                runtime_shard_sizes.append(padded_shard_n)
            fp8_weight = torch.cat(quant_shards, dim=0)
            scale = torch.cat(scale_shards, dim=0)
            runtime_n = sum(runtime_shard_sizes)
            logical_output_n = logical_n
        else:
            fp8_weight, scale = per_block_quant_like_legacy(weight, self.BLOCK)
            runtime_n = (logical_n + self.BLOCK - 1) // self.BLOCK * self.BLOCK
            if runtime_n != logical_n or padded_k != logical_k:
                fp8_weight = torch.nn.functional.pad(
                    fp8_weight,
                    (0, padded_k - logical_k, 0, runtime_n - logical_n),
                )
            logical_output_n = logical_n

        expected_scale_shape = (
            runtime_n // self.BLOCK,
            padded_k // self.BLOCK,
        )
        if scale.shape != expected_scale_shape:
            raise ValueError(
                f"FP8 scale shape {tuple(scale.shape)} != {expected_scale_shape}"
            )

        _validate_fp8_block_scales(fp8_weight, scale, self.BLOCK, self.BLOCK)
        if self._runtime_backend == _ROCM_AITER_BLOCK:
            fp8_weight, scale = _prepare_rocm_fp8_block_weight(fp8_weight, scale)
        elif is_deep_gemm_e8m0_used():
            fp8_weight, scale = _resolve_requant_weight_ue8m0()(fp8_weight, scale)

        del layer.weight
        layer.register_parameter(
            "weight", nn.Parameter(fp8_weight.contiguous(), requires_grad=False)
        )
        layer._fp8_logical_input_size = logical_k
        layer._fp8_logical_output_size = logical_output_n
        del layer.weight_scale
        layer.register_parameter(
            "weight_scale",
            nn.Parameter(scale.contiguous(), requires_grad=False),
        )

        Fp8BlockOnlineLinearMethod._quant_count += 1
        if not Fp8BlockOnlineLinearMethod._quant_logged:
            logger.info(
                "[Fp8BlockOnlineLinearMethod] online-quantized FIRST weight: "
                "prefix=%r, weight.shape=%s, weight.dtype=%s, "
                "scale.shape=%s, scale.dtype=%s, scale.mean=%.6f, "
                "weight.device=%s",
                getattr(layer, "prefix", "?"),
                tuple(fp8_weight.shape),
                fp8_weight.dtype,
                tuple(scale.shape),
                scale.dtype,
                float(scale.float().mean().item()),
                fp8_weight.device,
            )
            Fp8BlockOnlineLinearMethod._quant_logged = True

    def apply(
        self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self._runtime_backend not in (_CUDA_DEEP_GEMM, _ROCM_AITER_BLOCK):
            raise RuntimeError("FP8 block backend was not initialized during loading")

        out_dtype = x.dtype
        input_2d = x.reshape(-1, x.shape[-1])
        if not input_2d.is_contiguous():
            input_2d = input_2d.contiguous()
        M, logical_k = input_2d.shape
        logical_n = getattr(layer, "_fp8_logical_output_size", layer.weight.shape[0])
        padded_k = layer.weight.shape[1]
        padded_n = layer.weight.shape[0]
        if logical_k > padded_k:
            raise ValueError(f"FP8 input K={logical_k} exceeds weight K={padded_k}")
        if logical_k < padded_k:
            input_2d = torch.nn.functional.pad(input_2d, (0, padded_k - logical_k))
        output_shape = list(x.shape[:-1]) + [logical_n]

        if self._runtime_backend == _ROCM_AITER_BLOCK:
            output = _apply_rocm_fp8_block(
                input_2d,
                layer.weight,
                layer.weight_scale,
                out_dtype,
                self.BLOCK,
            )
        else:
            scale_ue8m0 = layer.weight_scale.dtype == torch.int32
            qinput, x_scales = _resolve_sgl_per_token_group_quant()(
                input_2d,
                group_size=self.BLOCK,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=scale_ue8m0,
            )
            output = torch.empty(M, padded_n, dtype=out_dtype, device=input_2d.device)
            _resolve_fp8_gemm_nt()(
                (qinput, x_scales),
                (layer.weight, layer.weight_scale),
                output,
                c=None,
                disable_ue8m0_cast=not scale_ue8m0,
            )

        output = output[:, :logical_n]

        if not Fp8BlockOnlineLinearMethod._apply_logged:
            logger.info(
                "[Fp8BlockOnlineLinearMethod] FIRST forward via %s: "
                "prefix=%r, x.dtype=%s, x.shape=%s, weight.dtype=%s, weight.shape=%s, "
                "weight_scale.shape=%s, total_quanted_weights=%d",
                self._runtime_backend,
                getattr(layer, "prefix", "?"),
                x.dtype,
                tuple(x.shape),
                layer.weight.dtype,
                tuple(layer.weight.shape),
                tuple(layer.weight_scale.shape),
                Fp8BlockOnlineLinearMethod._quant_count,
            )
            Fp8BlockOnlineLinearMethod._apply_logged = True

        if bias is not None:
            output = output + bias.to(out_dtype)
        return output.view(*output_shape)


@register_quant_method("fp8_block", "FP8_PER_BLOCK")
class Fp8BlockLinearMethod(Fp8BlockOnlineLinearMethod):
    """Already-quantized FP8 per-block (128x128) ckpt loader.

    ckpt provides:
      - weight: float8_e4m3fn [N, K]
      - weight_scale_inv: fp32 [ceil(N/128), ceil(K/128)]

    Unlike the online sibling (which loads BF16 and quantizes at load time),
    the weight is ALREADY fp8 + block-scaled. create_weights allocates the fp8
    weight plus a `weight_scale_inv` parameter; the parallel-linear load path
    (linear.py) TP-slices / shard-merges that block grid. process_weights_after_
    loading normalizes the checkpoint dtype/layout and renames
    `weight_scale_inv` -> `weight_scale`. The inherited apply() dispatches to
    CUDA DeepGEMM or ROCm AITer using the same runtime state as the online path.
    """

    _create_logged: bool = False
    required_checkpoint_parameters = ("weight", "weight_scale_inv")

    def create_weights(
        self,
        layer,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **kwargs,
    ):
        weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.register_parameter("weight", weight)

        block_n, block_k = getattr(
            layer.quant_config, "weight_block_size", [self.BLOCK, self.BLOCK]
        )
        for name, value in (("block_n", block_n), ("block_k", block_k)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"FP8 {name} must be a positive integer, got {value!r}"
                )
        if [block_n, block_k] != [self.BLOCK, self.BLOCK]:
            raise ValueError(
                "FP8 block runtime supports only "
                f"{[self.BLOCK, self.BLOCK]}, got {[block_n, block_k]}"
            )
        if not Fp8BlockLinearMethod._create_logged:
            logger.info(
                "[Fp8BlockLinearMethod] create_weights prefix=%r quant_type=%s "
                "source_config=%s weight_block_size=%s input=%d output=%d",
                getattr(layer, "prefix", "?"),
                getattr(layer.quant_config, "quant_type", None),
                (
                    type(getattr(layer.quant_config, "source_config", None)).__name__
                    if getattr(layer.quant_config, "source_config", None) is not None
                    else None
                ),
                getattr(layer.quant_config, "weight_block_size", None),
                input_size,
                output_size,
            )
            Fp8BlockLinearMethod._create_logged = True
        n_blocks = (output_size + block_n - 1) // block_n
        k_blocks = (input_size + block_k - 1) // block_k
        layer.register_parameter(
            "weight_scale_inv",
            nn.Parameter(
                torch.ones(n_blocks, k_blocks, dtype=torch.float32),
                requires_grad=False,
            ),
        )

    def process_weights_after_loading(self, layer):
        block_size = getattr(
            layer.quant_config, "weight_block_size", [self.BLOCK, self.BLOCK]
        )
        self._runtime_backend = _select_fp8_runtime_backend(
            layer.weight.device, "block"
        )
        self._use_deep_gemm = self._runtime_backend == _CUDA_DEEP_GEMM
        weight = layer.weight.data
        scale = layer.weight_scale_inv.data
        _validate_fp8_block_scales(weight, scale, block_size[0], block_size[1])
        if self._runtime_backend == _ROCM_AITER_BLOCK:
            weight, scale = _prepare_rocm_fp8_block_weight(weight, scale)
        if self._runtime_backend == _CUDA_DEEP_GEMM and is_deep_gemm_e8m0_used():
            weight, scale = _resolve_requant_weight_ue8m0()(weight, scale)
        del layer.weight
        layer.register_parameter(
            "weight", nn.Parameter(weight.contiguous(), requires_grad=False)
        )
        del layer.weight_scale_inv
        layer.register_parameter(
            "weight_scale",
            nn.Parameter(scale.contiguous(), requires_grad=False),
        )

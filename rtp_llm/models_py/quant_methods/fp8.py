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


def _shuffle_rocm_fp8_weight(weight: torch.Tensor) -> torch.Tensor:
    from aiter.ops.shuffle import shuffle_weight

    return shuffle_weight(weight, layout=(16, 16))


def _apply_rocm_fp8_per_channel(
    input_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    import aiter

    from rtp_llm.models_py.kernels.rocm.fp8_kernel import rocm_per_token_quant_fp8

    original_dtype = input_2d.dtype
    input_bf16 = (
        input_2d if original_dtype == torch.bfloat16 else input_2d.to(torch.bfloat16)
    )
    qinput, x_scale = rocm_per_token_quant_fp8(input_bf16, eps=1e-10)
    output = aiter.gemm_a8w8_bpreshuffle(
        qinput,
        weight,
        x_scale.to(torch.float32),
        weight_scale,
        None,
        torch.bfloat16,
    )
    return output if original_dtype == torch.bfloat16 else output.to(out_dtype)


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
        per_block_cast_to_fp8,
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
    per_block_cast_to_fp8 = None
    requant_weight_ue8m0 = None
    scaled_fp8_per_tensor_quant = None
    scaled_fp8_per_token_quant = None
    sgl_per_token_group_quant_fp8 = None

try:
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
        fp8_gemm_nt,
        has_deep_gemm,
        is_deep_gemm_e8m0_used,
    )
except ImportError as e:  # pragma: no cover
    logger.warning(
        "deepgemm_wrapper imports unavailable: %s (will fall back to lazy import)", e
    )
    fp8_gemm_nt = None

    def has_deep_gemm() -> bool:  # type: ignore[no-redef]
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
        if scale.numel() != 1 or not scale.is_floating_point():
            raise ValueError("Static FP8 per-tensor scale must be one floating value")
        scale = scale.to(device=input.device, dtype=torch.float32).reshape(1)
        if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError("Static FP8 per-tensor scale must be finite and positive")
        static_per_tensor_quant(output, input, scale)
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


def _resolve_per_block_cast():
    """Return per_block_cast_to_fp8, lazy-importing on a hoist miss."""
    global per_block_cast_to_fp8
    if per_block_cast_to_fp8 is None:
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            per_block_cast_to_fp8 as _fn,
        )

        per_block_cast_to_fp8 = _fn
    return per_block_cast_to_fp8


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

    Forward: torch._scaled_mm with online dynamic per-tensor activation quant.
    The ckpt's static input_scale is intentionally unused — using runtime-
    computed activation scale gives equivalent numerical accuracy without
    needing to plumb static-vs-dynamic through the LinearMethod construction.
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

    def apply(
        self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out_dtype = x.dtype
        input_2d = x.reshape(-1, x.shape[-1])
        if not input_2d.is_contiguous():
            input_2d = input_2d.contiguous()
        out_features = layer.weight.shape[0]
        output_shape = list(x.shape[:-1]) + [out_features]

        qinput, x_scale = _resolve_per_tensor_quant()(input_2d)

        if not Fp8LinearMethod._apply_logged:
            logger.info(
                "[Fp8LinearMethod] FIRST forward via torch._scaled_mm: "
                "prefix=%r, x.shape=%s, weight.shape=%s, weight_scale=%.6f, "
                "x_scale=%.6f",
                getattr(layer, "prefix", "?"),
                tuple(x.shape),
                tuple(layer.weight.shape),
                float(layer.weight_scale.view(1)[0].item()),
                float(x_scale.view(1)[0].item()),
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

    def process_weights_after_loading(self, layer):
        weight = layer.weight.data
        if weight.ndim != 2:
            raise ValueError(f"expected 2D weight, got {weight.shape}")

        if not weight.is_cuda:
            raise RuntimeError(
                "Online FP8 per-tensor postprocessing requires weights on the "
                "configured accelerator device"
            )
        fp8_weight, scale = _resolve_per_tensor_quant()(weight)

        # nn.Parameter dtype is immutable; rebind both attributes so the
        # post-load model.to(device) sees a consistent (cuda, fp8/fp32) pair.
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
                "weight.dtype=%s, weight.shape=%s, weight_scale=%.6f, "
                "x_scale=%.6f, total_quanted_weights=%d",
                getattr(layer, "prefix", "?"),
                x.dtype,
                tuple(x.shape),
                qinput.dtype,
                layer.weight.dtype,
                tuple(layer.weight.shape),
                float(layer.weight_scale.view(1)[0].item()),
                float(x_scale.view(1)[0].item()),
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

    def process_weights_after_loading(self, layer):
        weight = layer.weight.data
        if not weight.is_cuda:
            raise RuntimeError(
                "Online FP8 per-channel postprocessing requires weights on the "
                "configured accelerator device"
            )
        if weight.ndim != 2:
            raise ValueError(f"expected 2D weight, got {weight.shape}")

        # Treat weight rows as "tokens" -> per-output-channel quant.
        # fp8_weight: [N, K] float8_e4m3fn, scale: [N, 1] float32.
        is_hip = _is_hip_runtime()
        quant_input = (
            weight.to(torch.bfloat16)
            if is_hip and weight.dtype != torch.bfloat16
            else weight
        )
        fp8_weight, scale = _resolve_per_token_quant()(quant_input)
        if is_hip:
            fp8_weight = _shuffle_rocm_fp8_weight(fp8_weight)
            scale = scale.view(-1, 1).to(torch.float32).contiguous()
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
        out_dtype = x.dtype
        input_2d = x.reshape(-1, x.shape[-1])
        if not input_2d.is_contiguous():
            input_2d = input_2d.contiguous()
        N = layer.weight.shape[0]
        output_shape = list(x.shape[:-1]) + [N]

        if not Fp8PerChannelOnlineLinearMethod._apply_logged:
            logger.info(
                "[Fp8PerChannelOnlineLinearMethod] FIRST forward via %s: "
                "prefix=%r, x.dtype=%s, x.shape=%s, weight.dtype=%s, "
                "weight.shape=%s, weight_scale.shape=%s, total_quanted_weights=%d",
                (
                    "AITer gemm_a8w8_bpreshuffle"
                    if _is_hip_runtime()
                    else "torch._scaled_mm"
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

        if _is_hip_runtime():
            output = _apply_rocm_fp8_per_channel(
                input_2d,
                layer.weight,
                layer.weight_scale,
                out_dtype,
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

    def process_weights_after_loading(self, layer):
        # ckpt scale may arrive as [N] or [N, 1]. ROCm fp8 kernels expect the
        # platform runtime dtype (e4m3fnuz on MI308X), so convert already-FP8
        # checkpoint weights once after loading and store scale as [N, 1].
        fp8_weight, scale = _requant_per_channel_to_runtime_fp8(
            layer.weight.data, layer.weight_scale.data
        )
        is_hip = _is_hip_runtime()
        if is_hip:
            fp8_weight = _shuffle_rocm_fp8_weight(fp8_weight)
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
        out_dtype = x.dtype
        input_2d = x.reshape(-1, x.shape[-1])
        if not input_2d.is_contiguous():
            input_2d = input_2d.contiguous()
        N = layer.weight.shape[0]
        output_shape = list(x.shape[:-1]) + [N]

        if not Fp8PerChannelLinearMethod._apply_logged:
            logger.info(
                "[Fp8PerChannelLinearMethod] FIRST forward via %s: "
                "prefix=%r, x.shape=%s, weight.shape=%s, weight_scale.shape=%s",
                (
                    "AITer gemm_a8w8_bpreshuffle"
                    if _is_hip_runtime()
                    else "torch._scaled_mm"
                ),
                getattr(layer, "prefix", "?"),
                tuple(x.shape),
                tuple(layer.weight.shape),
                tuple(layer.weight_scale.shape),
            )
            Fp8PerChannelLinearMethod._apply_logged = True

        if _is_hip_runtime():
            output = _apply_rocm_fp8_per_channel(
                input_2d,
                layer.weight,
                layer.weight_scale,
                out_dtype,
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
    using DeepSeek-style 128x128 block scales, and runs forward via DeepGEMM
    fp8_gemm_nt with online per-token-group (128) activation quantization.
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
        if not has_deep_gemm():
            del layer.weight_scale
            expected_stride = (1, layer.weight.shape[0])
            if layer.weight.stride() != expected_stride:
                raise RuntimeError(
                    f"Online FP8 block fallback weight stride "
                    f"{layer.weight.stride()} does not match {expected_stride}"
                )
            logger.warning(
                "[Fp8BlockOnlineLinearMethod] DeepGEMM unavailable; keeping %r "
                "in its loaded dtype for F.linear fallback",
                getattr(layer, "prefix", "?"),
            )
            return
        if not weight.is_cuda:
            raise RuntimeError(
                "Online FP8 block postprocessing requires weights on the "
                "configured accelerator device"
            )
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

        if is_deep_gemm_e8m0_used():
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
        if not has_deep_gemm():
            if not hasattr(layer, "weight_scale"):
                return torch.nn.functional.linear(x, layer.weight, bias)
            raise RuntimeError(
                "Fp8BlockOnlineLinearMethod requires DeepGEMM at forward time; "
                "install the `deep_gemm` package or load fp8_block weights "
                "through the no-DeepGEMM dequant fallback."
            )

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

        # Online per-token-group activation quant (group_size=128).
        scale_ue8m0 = getattr(layer, "weight_scale", None) is not None and (
            layer.weight_scale.dtype == torch.int32
        )
        qinput, x_scales = _resolve_sgl_per_token_group_quant()(
            input_2d,
            group_size=self.BLOCK,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=scale_ue8m0,
        )

        output = torch.empty(M, padded_n, dtype=out_dtype, device=input_2d.device)
        # Keep runtime selection aligned with the legacy FP8_PER_BLOCK linear,
        # which uses DeepGEMM for both prefill and decode. Switching only the
        # M < 32 decode path to FlashInfer changes accumulation enough to make
        # old/new loader outputs diverge despite byte-identical weights.
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
                "[Fp8BlockOnlineLinearMethod] FIRST forward via deep_gemm.fp8_gemm_nt: "
                "prefix=%r, x.dtype=%s, x.shape=%s, qinput.dtype=%s, qinput.shape=%s, "
                "x_scales.shape=%s, weight.dtype=%s, weight.shape=%s, "
                "weight_scale.shape=%s, total_quanted_weights=%d",
                getattr(layer, "prefix", "?"),
                x.dtype,
                tuple(x.shape),
                qinput.dtype,
                tuple(qinput.shape),
                tuple(x_scales.shape),
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
    loading simply renames `weight_scale_inv` -> `weight_scale` so the inherited
    apply() (DeepGEMM fp8_gemm_nt, which reads `weight` + `weight_scale`) runs
    identically to the online path — see TestFp8BlockForward.
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
        # The weight is already fp8. With DeepGEMM available, expose the block
        # scale under the name the inherited apply() expects (`weight_scale`).
        # ROCm/test containers do not provide CUDA DeepGEMM, so dequantize once
        # at load time and use a plain bf16 linear fallback for smoke coverage.
        block_size = getattr(
            layer.quant_config, "weight_block_size", [self.BLOCK, self.BLOCK]
        )
        if not has_deep_gemm():
            if list(block_size) == [self.BLOCK, self.BLOCK]:
                weight_dequant = _dequant_block_to_bf16(
                    layer.weight.data, layer.weight_scale_inv.data, self.BLOCK
                )
            else:
                weight_dequant = _dequant_rectangular_blocks_to_bf16(
                    layer.weight.data,
                    layer.weight_scale_inv.data,
                    block_size[0],
                    block_size[1],
                )
            del layer.weight
            layer.register_parameter(
                "weight", nn.Parameter(weight_dequant.contiguous(), requires_grad=False)
            )
            del layer.weight_scale_inv
            logger.info(
                "[Fp8BlockLinearMethod] DeepGEMM unavailable; dequantized %r to bf16 for fallback linear",
                getattr(layer, "prefix", "?"),
            )
            return

        weight = layer.weight.data
        if list(block_size) != [self.BLOCK, self.BLOCK]:
            weight_dequant = _dequant_rectangular_blocks_to_bf16(
                weight,
                layer.weight_scale_inv.data,
                block_size[0],
                block_size[1],
            )
            weight, scale = _resolve_per_block_cast()(weight_dequant, use_ue8m0=False)
        else:
            scale = layer.weight_scale_inv.data
        if is_deep_gemm_e8m0_used():
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


def _dequant_block_to_bf16(
    weight: torch.Tensor, scale_inv: torch.Tensor, block: int = 128
) -> torch.Tensor:
    """Dequantize a DeepSeek FP8 per-block (128x128) weight [N,K] to bf16.

    scale_inv is the standard [ceil(N/128), ceil(K/128)] block grid. Expand it
    to full [N, K] (cropping any partial trailing block) and scale.
    """
    return _dequant_rectangular_blocks_to_bf16(weight, scale_inv, block, block)


def _dequant_rectangular_blocks_to_bf16(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block_n: int,
    block_k: int,
) -> torch.Tensor:
    if block_n <= 0 or block_k <= 0:
        raise ValueError(f"Invalid FP8 block size {(block_n, block_k)}")
    rows, columns = weight.shape
    expected = (
        (rows + block_n - 1) // block_n,
        (columns + block_k - 1) // block_k,
    )
    if tuple(scale_inv.shape) != expected:
        raise ValueError(
            f"FP8 block scale shape must be {expected}, got {tuple(scale_inv.shape)}"
        )
    if not bool(torch.isfinite(scale_inv).all()) or bool((scale_inv <= 0).any()):
        raise ValueError("FP8 block scales must be finite and positive")
    scales = scale_inv.to(torch.float32)
    scales = scales.repeat_interleave(block_n, dim=0).repeat_interleave(block_k, dim=1)
    return (weight.to(torch.float32) * scales[:rows, :columns]).to(torch.bfloat16)

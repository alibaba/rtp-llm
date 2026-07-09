import logging
from typing import List, Optional, Tuple

import torch
from torch import nn

from rtp_llm.models_py.quant_methods.base import (
    QuantizeMethodBase,
    register_quant_method,
)

logger = logging.getLogger(__name__)

_FP8_MIN_SCALE = 1e-12


def _runtime_fp8_dtype() -> torch.dtype:
    if getattr(torch.version, "hip", None) is None:
        return torch.float8_e4m3fn
    try:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm._utils import (
            get_rocm_fp8_dtype,
        )

        return get_rocm_fp8_dtype()
    except Exception:
        return torch.float8_e4m3fn


def _requant_per_tensor_to_runtime_fp8(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    runtime_dtype = _runtime_fp8_dtype()
    if weight.dtype == runtime_dtype:
        return weight.contiguous(), scale.view(1).contiguous()

    scale_scalar = scale.float().view(-1)[0]
    deq = weight.float() * scale_scalar
    fp8_max = float(torch.finfo(runtime_dtype).max)
    new_scale = deq.abs().amax().clamp_min(_FP8_MIN_SCALE) / fp8_max
    requant = (deq / new_scale).to(runtime_dtype)
    return requant.contiguous(), new_scale.reshape(1).to(torch.float32).contiguous()


def _requant_per_channel_to_runtime_fp8(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    runtime_dtype = _runtime_fp8_dtype()
    scale_rows = scale.float().view(-1, 1)
    if weight.dtype == runtime_dtype:
        return weight.contiguous(), scale_rows.view(1, -1).contiguous()

    if scale_rows.shape[0] != weight.shape[0]:
        raise ValueError(
            f"FP8 per-channel scale/weight mismatch: "
            f"weight={tuple(weight.shape)} scale={tuple(scale.shape)}"
        )
    deq = weight.float() * scale_rows
    fp8_max = float(torch.finfo(runtime_dtype).max)
    new_scale = (
        deq.abs().amax(dim=1, keepdim=True).clamp_min(_FP8_MIN_SCALE) / fp8_max
    )
    requant = (deq / new_scale).to(runtime_dtype)
    return requant.contiguous(), new_scale.view(1, -1).to(torch.float32).contiguous()

# Hoist kernel imports to module scope. apply() is on the per-token decode hot
# path: doing the import inside apply() (even though sys.modules caches it)
# still costs a sys.modules lookup + LOAD_ATTR on every call. Importing once at
# module load amortizes that cost across the lifetime of the process.
#
# Each method that uses a kernel still falls back to a lazy `_resolve_*()` helper
# below when the module-level symbol is None, since `fp8.py` is loaded eagerly
# from `quant_methods/__init__.py` and may resolve before the kernel package
# finishes initialization (the kernel package's __init__ calls load_all_configs()
# which can take long enough for an interleaving import to observe a partially
# initialized module — that is reported with a logger.warning here so the
# fallback isn't silent).
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
    )
except ImportError as e:  # pragma: no cover
    logger.warning(
        "deepgemm_wrapper imports unavailable: %s (will fall back to lazy import)", e
    )
    fp8_gemm_nt = None

    def has_deep_gemm() -> bool:  # type: ignore[no-redef]
        return False


def _resolve_per_tensor_quant():
    """Return scaled_fp8_per_tensor_quant, lazy-importing on a hoist miss."""
    global scaled_fp8_per_tensor_quant
    if scaled_fp8_per_tensor_quant is None:
        try:
            from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
                scaled_fp8_per_tensor_quant as _fn,
            )
        except ImportError as e:
            raise ImportError(
                "fp8 kernel not available: scaled_fp8_per_tensor_quant is required"
            ) from e
        if _fn is None:
            raise ImportError(
                "fp8 kernel not available: scaled_fp8_per_tensor_quant is None"
            )
        scaled_fp8_per_tensor_quant = _fn
    return scaled_fp8_per_tensor_quant


def cpu_per_tensor_quant_like_legacy(
    input: torch.Tensor, output: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CPU fallback matching old loader's dynamic FP8 per-tensor quantization."""
    device = input.device
    from rtp_llm.model_loader.dynamic_fp8_quant_weight import (
        quantize_weight_to_fp8 as legacy_quantize_weight_to_fp8,
    )

    quant, scale = legacy_quantize_weight_to_fp8(input.detach().to("cpu"))
    quant = quant.to(device)
    scale = scale.reshape(1).to(device)
    if output is not None:
        output.copy_(quant)
        quant = output
    return quant, scale


def _resolve_per_token_quant():
    """Return scaled_fp8_per_token_quant, lazy-importing on a hoist miss."""
    global scaled_fp8_per_token_quant
    if scaled_fp8_per_token_quant is None:
        try:
            from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
                scaled_fp8_per_token_quant as _fn,
            )
        except ImportError as e:
            raise ImportError(
                "fp8 kernel not available: scaled_fp8_per_token_quant is required"
            ) from e
        if _fn is None:
            raise ImportError(
                "fp8 kernel not available: scaled_fp8_per_token_quant is None"
            )
        scaled_fp8_per_token_quant = _fn
    return scaled_fp8_per_token_quant


def _resolve_per_block_cast():
    """Return per_block_cast_to_fp8, lazy-importing on a hoist miss."""
    global per_block_cast_to_fp8
    if per_block_cast_to_fp8 is None:
        try:
            from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
                per_block_cast_to_fp8 as _fn,
            )
        except ImportError as e:
            raise ImportError(
                "fp8 kernel not available: per_block_cast_to_fp8 is required"
            ) from e
        if _fn is None:
            raise ImportError("fp8 kernel not available: per_block_cast_to_fp8 is None")
        per_block_cast_to_fp8 = _fn
    return per_block_cast_to_fp8


def _normalize_weight_block_size(layer, default_block: int) -> List[int]:
    block_size = getattr(
        getattr(layer, "quant_config", None), "weight_block_size", None
    )
    if block_size is None:
        block_size = [default_block, default_block]
    if len(block_size) != 2:
        raise ValueError(f"weight_block_size must have 2 values, got {block_size}")
    block_n, block_k = int(block_size[0]), int(block_size[1])
    if block_n <= 0 or block_k <= 0:
        raise ValueError(f"weight_block_size values must be positive, got {block_size}")
    return [block_n, block_k]


def _require_cuda_weight_for_online_quant(
    layer, weight: torch.Tensor, method: str, allow_force_cpu: bool = False
):
    if weight.is_cuda:
        return weight
    if allow_force_cpu and bool(
        getattr(layer, "_new_loader_force_cpu_load_weights", False)
    ):
        return weight
    cpu_hint = (
        " or enable force_cpu_load_weights for CPU-side post-load quantization"
        if allow_force_cpu
        else " or use an already-quantized/non-online quant method for CPU load"
    )
    raise RuntimeError(
        f"{method} requires weights to be on CUDA/ROCm before post-load "
        f"quantization, got device={weight.device} for prefix="
        f"{getattr(layer, 'prefix', '?')!r}. Set LoadConfig.device to a CUDA "
        f"device{cpu_hint}."
    )


def _resolve_requant_weight_ue8m0():
    """Return requant_weight_ue8m0, lazy-importing on a hoist miss."""
    global requant_weight_ue8m0
    if requant_weight_ue8m0 is None:
        try:
            from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
                requant_weight_ue8m0 as _fn,
            )
        except ImportError as e:
            raise ImportError(
                "fp8 kernel not available: requant_weight_ue8m0 is required"
            ) from e
        if _fn is None:
            raise ImportError("fp8 kernel not available: requant_weight_ue8m0 is None")
        requant_weight_ue8m0 = _fn
    return requant_weight_ue8m0


def _resolve_sgl_per_token_group_quant():
    """Return sgl_per_token_group_quant_fp8, lazy-importing on a hoist miss."""
    global sgl_per_token_group_quant_fp8
    if sgl_per_token_group_quant_fp8 is None:
        try:
            from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
                sgl_per_token_group_quant_fp8 as _fn,
            )
        except ImportError as e:
            raise ImportError(
                "fp8 kernel not available: sgl_per_token_group_quant_fp8 is required"
            ) from e
        if _fn is None:
            raise ImportError(
                "fp8 kernel not available: sgl_per_token_group_quant_fp8 is None"
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


@register_quant_method(
    "fp8",
    "FP8_PER_TENSOR_COMPRESSED",
    "FP8_DYNAMIC_PER_TENSOR",
)
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

    def create_weights(
        self,
        layer,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **kwargs,
    ):
        weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=_runtime_fp8_dtype()),
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


@register_quant_method("fp8_online")
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

        weight = _require_cuda_weight_for_online_quant(
            layer, weight, type(self).__name__, allow_force_cpu=True
        )
        if bool(getattr(layer, "_new_loader_force_cpu_load_weights", False)):
            fp8_weight, scale = cpu_per_tensor_quant_like_legacy(weight)
        else:
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
        weight = _require_cuda_weight_for_online_quant(
            layer, weight, type(self).__name__
        )
        if weight.ndim != 2:
            raise ValueError(f"expected 2D weight, got {weight.shape}")

        # Treat weight rows as "tokens" → per-output-channel quant.
        # fp8_weight: [N, K] float8_e4m3fn, scale: [N, 1] float32.
        fp8_weight, scale = _resolve_per_token_quant()(weight)

        del layer.weight
        layer.register_parameter(
            "weight", nn.Parameter(fp8_weight.contiguous(), requires_grad=False)
        )
        # Store scale already shaped [1, N] contiguous so apply() can pass
        # layer.weight_scale directly to torch._scaled_mm as scale_b without
        # per-call reshape+contiguous (was a measurable per-token decode cost).
        del layer.weight_scale
        layer.register_parameter(
            "weight_scale",
            nn.Parameter(scale.view(1, -1).contiguous(), requires_grad=False),
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

        # Per-token activation quant: [M, K] bf16 -> [M, K] fp8 + [M, 1] fp32.
        qinput, x_scale = _resolve_per_token_quant()(input_2d)

        # Weight scale was already shaped [1, N] contiguous in
        # process_weights_after_loading — use it directly.
        weight_scale_b = layer.weight_scale

        if not Fp8PerChannelOnlineLinearMethod._apply_logged:
            logger.info(
                "[Fp8PerChannelOnlineLinearMethod] FIRST forward via torch._scaled_mm: "
                "prefix=%r, x.dtype=%s, x.shape=%s, qinput.dtype=%s, qinput.shape=%s, "
                "x_scale.shape=%s, weight.dtype=%s, weight.shape=%s, "
                "weight_scale.shape=%s, scale_b.shape=%s, total_quanted_weights=%d",
                getattr(layer, "prefix", "?"),
                x.dtype,
                tuple(x.shape),
                qinput.dtype,
                tuple(qinput.shape),
                tuple(x_scale.shape),
                layer.weight.dtype,
                tuple(layer.weight.shape),
                tuple(layer.weight_scale.shape),
                tuple(weight_scale_b.shape),
                Fp8PerChannelOnlineLinearMethod._quant_count,
            )
            Fp8PerChannelOnlineLinearMethod._apply_logged = True

        output = torch._scaled_mm(
            qinput,
            layer.weight.t(),
            scale_a=x_scale,
            scale_b=weight_scale_b,
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

    Forward: torch._scaled_mm with online per-token activation quantization
    (one fp32 scale per token-row of [M, K]), reusing
    Fp8PerChannelOnlineLinearMethod's apply path.
    """

    # Class-level flag: the diagnostic log fires at most once across ALL
    # instances in a process. See Fp8OnlineLinearMethod for the assumption.
    _apply_logged: bool = False

    def create_weights(
        self,
        layer,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **kwargs,
    ):
        weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=_runtime_fp8_dtype()),
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
        # checkpoint weights once after loading and store scale as [1, N].
        fp8_weight, scale = _requant_per_channel_to_runtime_fp8(
            layer.weight.data, layer.weight_scale.data
        )
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

        qinput, x_scale = _resolve_per_token_quant()(input_2d)
        # weight_scale was already shaped [1, N] contiguous in
        # process_weights_after_loading — use it directly.
        weight_scale_b = layer.weight_scale

        if not Fp8PerChannelLinearMethod._apply_logged:
            logger.info(
                "[Fp8PerChannelLinearMethod] FIRST forward via torch._scaled_mm: "
                "prefix=%r, x.shape=%s, weight.shape=%s, weight_scale.shape=%s",
                getattr(layer, "prefix", "?"),
                tuple(x.shape),
                tuple(layer.weight.shape),
                tuple(layer.weight_scale.shape),
            )
            Fp8PerChannelLinearMethod._apply_logged = True

        output = torch._scaled_mm(
            qinput,
            layer.weight.t(),
            scale_a=x_scale,
            scale_b=weight_scale_b,
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
        weight = _require_cuda_weight_for_online_quant(
            layer, weight, type(self).__name__
        )
        if weight.ndim != 2:
            raise ValueError(f"expected 2D weight, got {weight.shape}")

        # Match the legacy W8A8 per-block online loader exactly. The CUDA helper
        # below is also DeepGEMM-derived, but the smoke baseline was produced by
        # the Python loader-side quantizer.
        from rtp_llm.model_loader.per_block_fp8_quant_weight import (
            per_block_cast_to_fp8 as legacy_per_block_cast_to_fp8,
        )

        fp8_weight, scale = legacy_per_block_cast_to_fp8(weight, self.BLOCK)

        del layer.weight
        layer.register_parameter(
            "weight", nn.Parameter(fp8_weight.contiguous(), requires_grad=False)
        )
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
                float(scale.mean().item()),
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
        M, K = input_2d.shape
        N = layer.weight.shape[0]
        output_shape = list(x.shape[:-1]) + [N]

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

        output = torch.empty(M, N, dtype=out_dtype, device=input_2d.device)
        _resolve_fp8_gemm_nt()(
            (qinput, x_scales),
            (layer.weight, layer.weight_scale),
            output,
            c=None,
            disable_ue8m0_cast=not scale_ue8m0,
        )

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

        block_n, block_k = _normalize_weight_block_size(layer, self.BLOCK)
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
        block_size = _normalize_weight_block_size(layer, self.BLOCK)
        if not has_deep_gemm():
            if list(block_size) == [self.BLOCK, self.BLOCK]:
                weight_dequant = _dequant_block_to_bf16(
                    layer.weight.data, layer.weight_scale_inv.data, self.BLOCK
                )
            else:
                from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
                    block_quant_dequant,
                )

                weight_dequant = block_quant_dequant(
                    layer.weight.data,
                    layer.weight_scale_inv.data,
                    list(block_size),
                    torch.bfloat16,
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

        if list(block_size) != [self.BLOCK, self.BLOCK]:
            from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
                block_quant_dequant,
            )

            weight_dequant = block_quant_dequant(
                layer.weight.data,
                layer.weight_scale_inv.data,
                list(block_size),
                torch.bfloat16,
            )
            weight, scale = _resolve_per_block_cast()(weight_dequant, use_ue8m0=False)
            del layer.weight
            layer.register_parameter(
                "weight", nn.Parameter(weight.contiguous(), requires_grad=False)
            )
        else:
            scale = layer.weight_scale_inv.data
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
    n, k = weight.shape
    s = scale_inv.to(torch.float32)
    s = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    s = s[:n, :k]
    return (weight.to(torch.float32) * s).to(torch.bfloat16)


@register_quant_method("fp8_block_dequant")
class Fp8BlockDequantLinearMethod(Fp8BlockLinearMethod):
    """Already-quantized FP8 per-block ckpt, DEQUANTIZED to bf16 at load.

    Same ckpt contract as Fp8BlockLinearMethod (fp8 weight + weight_scale_inv,
    block-aware TP / shard-merge handled in linear.py), but
    process_weights_after_loading expands the block scale and converts the
    weight back to bf16, so apply() is a plain F.linear (no DeepGEMM).

    For models whose downstream math needs bf16 weights or that must match a
    bf16 reference — e.g. DeepSeek-V3.2's MLA, where the absorb path derives
    kc/vc via torch.bmm (no fp8 kernel) and fp8 GEMM would diverge from the
    validated bf16 baseline. The routed experts keep fp8 separately.
    """

    def process_weights_after_loading(self, layer):
        block_size = _normalize_weight_block_size(layer, self.BLOCK)
        if list(block_size) == [self.BLOCK, self.BLOCK]:
            bf16 = _dequant_block_to_bf16(
                layer.weight.data, layer.weight_scale_inv.data, self.BLOCK
            )
        else:
            from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
                block_quant_dequant,
            )

            bf16 = block_quant_dequant(
                layer.weight.data,
                layer.weight_scale_inv.data,
                list(block_size),
                torch.bfloat16,
            )
        del layer.weight
        layer.register_parameter(
            "weight", nn.Parameter(bf16.contiguous(), requires_grad=False)
        )
        del layer.weight_scale_inv

    def apply(self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None):
        return torch.nn.functional.linear(x, layer.weight, bias)

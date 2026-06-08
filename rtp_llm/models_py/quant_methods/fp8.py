import logging
from typing import Optional

import torch
from torch import nn

from rtp_llm.models_py.quant_methods.base import (
    QuantizeMethodBase,
    register_quant_method,
)

logger = logging.getLogger(__name__)

# Hoist kernel imports to module scope. apply() is on the per-token decode hot
# path: doing the import inside apply() (even though sys.modules caches it)
# still costs a sys.modules lookup + LOAD_ATTR on every call. Importing once at
# module load amortizes that cost across the lifetime of the process.
try:
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
        per_block_cast_to_fp8,
        scaled_fp8_per_tensor_quant,
        scaled_fp8_per_token_quant,
        sgl_per_token_group_quant_fp8,
    )
except ImportError:  # pragma: no cover - kernel package may be absent on CPU-only setups
    per_block_cast_to_fp8 = None
    scaled_fp8_per_tensor_quant = None
    scaled_fp8_per_token_quant = None
    sgl_per_token_group_quant_fp8 = None

try:
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
        fp8_gemm_nt,
        has_deep_gemm,
    )
except ImportError:  # pragma: no cover
    fp8_gemm_nt = None

    def has_deep_gemm() -> bool:  # type: ignore[no-redef]
        return False


@register_quant_method("fp8")
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
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            scaled_fp8_per_tensor_quant,
        )

        out_dtype = x.dtype
        input_2d = x.view(-1, x.shape[-1])
        out_features = layer.weight.shape[0]
        output_shape = list(x.shape[:-1]) + [out_features]

        qinput, x_scale = scaled_fp8_per_tensor_quant(input_2d)

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
        if layer.weight_scale.dim() == 0:
            layer.weight_scale = nn.Parameter(
                layer.weight_scale.reshape(1), requires_grad=False
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
        # _load_via_scratch loads ckpt to CPU; quant kernel needs CUDA.
        if not weight.is_cuda:
            weight = weight.to(f"cuda:{torch.cuda.current_device()}")
        assert weight.ndim == 2, f"expected 2D weight, got {weight.shape}"

        fp8_weight, scale = scaled_fp8_per_tensor_quant(weight)

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
        input_2d = x.view(-1, x.shape[-1])
        out_features = layer.weight.shape[0]
        output_shape = list(x.shape[:-1]) + [out_features]

        qinput, x_scale = scaled_fp8_per_tensor_quant(input_2d)

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
        if not weight.is_cuda:
            weight = weight.to(f"cuda:{torch.cuda.current_device()}")
        assert weight.ndim == 2, f"expected 2D weight, got {weight.shape}"

        # Treat weight rows as "tokens" → per-output-channel quant.
        # fp8_weight: [N, K] float8_e4m3fn, scale: [N, 1] float32.
        fp8_weight, scale = scaled_fp8_per_token_quant(weight)

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
        input_2d = x.view(-1, x.shape[-1])
        if not input_2d.is_contiguous():
            input_2d = input_2d.contiguous()
        N = layer.weight.shape[0]
        output_shape = list(x.shape[:-1]) + [N]

        # Per-token activation quant: [M, K] bf16 -> [M, K] fp8 + [M, 1] fp32.
        qinput, x_scale = scaled_fp8_per_token_quant(input_2d)

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


@register_quant_method("fp8_per_channel")
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
        # ckpt scale may arrive as [N] or [N, 1]; normalize to [1, N] contig
        # once here so apply() can pass layer.weight_scale directly to
        # _scaled_mm without per-call reshape+contiguous.
        scale = layer.weight_scale.data
        scale = scale.view(1, -1).contiguous()
        del layer.weight_scale
        layer.register_parameter(
            "weight_scale",
            nn.Parameter(scale, requires_grad=False),
        )

    def apply(
        self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out_dtype = x.dtype
        input_2d = x.view(-1, x.shape[-1])
        if not input_2d.is_contiguous():
            input_2d = input_2d.contiguous()
        N = layer.weight.shape[0]
        output_shape = list(x.shape[:-1]) + [N]

        qinput, x_scale = scaled_fp8_per_token_quant(input_2d)
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
        if not weight.is_cuda:
            weight = weight.to(f"cuda:{torch.cuda.current_device()}")
        assert weight.ndim == 2, f"expected 2D weight, got {weight.shape}"

        # per_block_cast_to_fp8 returns:
        #   fp8_weight: [N, K] float8_e4m3fn (sliced back to original shape)
        #   scale:      [ceil(N/128), ceil(K/128)] float32
        fp8_weight, scale = per_block_cast_to_fp8(weight, use_ue8m0=False)

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
            raise RuntimeError(
                "Fp8BlockOnlineLinearMethod requires DeepGEMM at forward time; "
                "install the `deep_gemm` package or fall back to "
                "QUANTIZATION=FP8_DYNAMIC_PER_TENSOR."
            )

        out_dtype = x.dtype
        input_2d = x.view(-1, x.shape[-1])
        if not input_2d.is_contiguous():
            input_2d = input_2d.contiguous()
        M, K = input_2d.shape
        N = layer.weight.shape[0]
        output_shape = list(x.shape[:-1]) + [N]

        # Online per-token-group activation quant (group_size=128).
        qinput, x_scales = sgl_per_token_group_quant_fp8(
            input_2d,
            group_size=self.BLOCK,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=False,
        )

        output = torch.empty(M, N, dtype=out_dtype, device=input_2d.device)
        fp8_gemm_nt(
            (qinput, x_scales),
            (layer.weight, layer.weight_scale),
            output,
            c=None,
            disable_ue8m0_cast=True,
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

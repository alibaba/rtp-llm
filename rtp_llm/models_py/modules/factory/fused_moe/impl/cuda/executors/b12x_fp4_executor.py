import logging
import math
import os
import threading
from typing import Any, Dict, Optional

import torch

from rtp_llm.device.flashinfer_b12x_adapter import (
    DISABLE_CUDA12_9_COMPAT_ENV,
    convert_b12x_blockscale_to_mma_layout,
    create_b12x_wrappers,
    get_b12x_kernel_tile_n,
    get_disable_cuda12_9_compat,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)

_runtime_config_logged = False
_runtime_config_lock = threading.Lock()
_ZEROED_ENERGY_LIMIT = 0.001
_ZEROED_ENERGY_LIMIT_ENV = "RTP_LLM_B12X_ZEROED_ENERGY_LIMIT"
_E4M3_MIN_NORMAL = 2.0**-6
_B12X_TOPK_IDS_DTYPE = torch.int32
_B12X_TOPK_WEIGHTS_DTYPE = torch.float32


def _get_zeroed_energy_limit() -> float:
    raw_limit = os.getenv(_ZEROED_ENERGY_LIMIT_ENV)
    if not raw_limit:
        return _ZEROED_ENERGY_LIMIT
    try:
        limit = float(raw_limit)
    except ValueError as error:
        raise ValueError(
            f"{_ZEROED_ENERGY_LIMIT_ENV} must be a float in [0, 1], got "
            f"{raw_limit!r}"
        ) from error
    if not math.isfinite(limit) or not 0 <= limit <= 1:
        raise ValueError(
            f"{_ZEROED_ENERGY_LIMIT_ENV} must be a finite float in [0, 1], "
            f"got {raw_limit!r}"
        )
    return limit


def _log_runtime_config_once(
    zeroed_energy_limit: float, disable_cuda12_9_compat: bool
) -> None:
    global _runtime_config_logged
    with _runtime_config_lock:
        if _runtime_config_logged:
            return
        logger.info(
            "b12x FP4 runtime config: %s=%s, %s=%s; checkpoint input_scale "
            "is not used and activation global scales are fixed to 1",
            _ZEROED_ENERGY_LIMIT_ENV,
            zeroed_energy_limit,
            DISABLE_CUDA12_9_COMPAT_ENV,
            disable_cuda12_9_compat,
        )
        _runtime_config_logged = True


def _validate_b12x_weight_shapes(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_sf: torch.Tensor,
    w2_sf: torch.Tensor,
    w1_scale_2: torch.Tensor,
    w2_scale_2: torch.Tensor,
    num_experts: int,
    kernel_tile_n: int,
) -> tuple[int, int]:
    """Validate packed weights and swizzled blockscales before layout conversion."""
    for name, weight in (("w1", w1), ("w2", w2)):
        if weight.dtype is not torch.uint8:
            raise ValueError(
                f"b12x FP4 {name} must contain packed uint8 weights, got "
                f"{weight.dtype}"
            )
    for name, scale in (("w1", w1_sf), ("w2", w2_sf)):
        if scale.dtype is not torch.float8_e4m3fn:
            raise ValueError(
                f"b12x FP4 {name} blockscales must use torch.float8_e4m3fn, "
                f"got {scale.dtype}"
            )
    for name, scale in (("w1", w1_scale_2), ("w2", w2_scale_2)):
        if scale.dtype is not torch.float32:
            raise ValueError(
                f"b12x FP4 {name} weight_scale_2 must use torch.float32, got "
                f"{scale.dtype}"
            )

    tensors = {
        "w1": w1,
        "w2": w2,
        "w1 blockscale": w1_sf,
        "w2 blockscale": w2_sf,
        "w1 weight_scale_2": w1_scale_2,
        "w2 weight_scale_2": w2_scale_2,
    }
    expected_device = w1.device
    mismatched_devices = {
        name: str(tensor.device)
        for name, tensor in tensors.items()
        if tensor.device != expected_device
    }
    if mismatched_devices:
        raise ValueError(
            "b12x FP4 weights and scales must share one device; "
            f"w1 is on {expected_device}, mismatches: {mismatched_devices}"
        )

    if w1.ndim != 3 or w2.ndim != 3:
        raise ValueError(
            f"b12x FP4 weights must be rank 3, got w1={tuple(w1.shape)}, "
            f"w2={tuple(w2.shape)}"
        )

    weight_experts, two_i, h_half = w1.shape
    w2_experts, hidden_size, i_half = w2.shape
    if weight_experts != num_experts or w2_experts != num_experts:
        raise ValueError(
            "b12x FP4 kernel indexes weights with global expert ids and has no "
            f"local-expert remapping: w1/w2 hold {weight_experts}/{w2_experts} "
            f"experts but the model has {num_experts} (EP-sharded weights are "
            "not supported)"
        )
    if two_i % 2 != 0:
        raise ValueError(f"b12x FP4 w13 rows must be 2*I, got {two_i}")

    intermediate_size = two_i // 2
    if h_half * 2 != hidden_size:
        raise ValueError(
            f"b12x FP4 w13 last dim {h_half} * 2 must equal hidden "
            f"size {hidden_size}"
        )
    if i_half * 2 != intermediate_size:
        raise ValueError(
            f"b12x FP4 w2 last dim {i_half} * 2 must equal intermediate "
            f"size {intermediate_size}"
        )
    if intermediate_size % kernel_tile_n != 0:
        raise ValueError(
            "b12x FP4 needs an intermediate size that is a multiple of "
            f"FlashInfer's gate/up tile width {kernel_tile_n}, got "
            f"{intermediate_size} (moe_inter_size split over tp_size)"
        )
    if two_i % 128 != 0:
        raise ValueError(
            "b12x FP4 needs 2*intermediate_size to be a multiple of 128 "
            "because the swizzled blockscale pads weight rows to 128, got "
            f"{two_i}"
        )
    if hidden_size % 128 != 0:
        raise ValueError(
            "b12x FP4 needs hidden_size to be a multiple of 128 so both w1 K "
            "and w2 M satisfy FlashInfer's MMA blockscale alignment, got "
            f"{hidden_size}"
        )

    expected_w1_sf = (num_experts, two_i, hidden_size // 16)
    expected_w2_sf = (num_experts, hidden_size, intermediate_size // 16)
    if tuple(w1_sf.shape) != expected_w1_sf:
        raise ValueError(
            "b12x FP4 w1 blockscale shape must match the aligned, swizzled "
            f"[E, 2*I, H/16] layout: expected {expected_w1_sf}, got "
            f"{tuple(w1_sf.shape)}"
        )
    if tuple(w2_sf.shape) != expected_w2_sf:
        raise ValueError(
            "b12x FP4 w2 blockscale shape must match the aligned, swizzled "
            f"[E, H, I/16] layout: expected {expected_w2_sf}, got "
            f"{tuple(w2_sf.shape)}"
        )
    # Swizzling is a permutation, so shape alone cannot distinguish a swizzled
    # scale tensor. The production weight-preparation path owns that contract.
    for name, scale in (("w1", w1_scale_2), ("w2", w2_scale_2)):
        if (
            scale.ndim == 0
            or scale.shape[0] != num_experts
            or scale.numel() != num_experts
        ):
            raise ValueError(
                f"b12x FP4 {name} weight_scale_2 must contain one scalar per "
                f"expert ({num_experts} values), got shape {tuple(scale.shape)}"
            )

    return intermediate_size, hidden_size


def _validate_folded_blockscale(
    name: str,
    product: torch.Tensor,
    folded: torch.Tensor,
    zeroed_energy_limit: float,
) -> tuple[torch.Tensor, float, float]:
    """Validate the e4m3 fold and return statistics used for diagnostics."""
    folded_f32 = folded.to(torch.float32)
    if not bool(torch.isfinite(folded_f32).all()):
        raise ValueError(
            f"b12x FP4: {name} blockscale overflowed e4m3 while folding "
            "weight_scale_2; the checkpoint's scales are out of range"
        )

    sf_nonzero = product != 0
    zeroed = (folded_f32 == 0) & sf_nonzero
    total_energy = (product**2).sum().item()
    if total_energy == 0:
        raise ValueError(
            f"b12x FP4: {name} blockscales have zero total scale energy after "
            "folding weight_scale_2; the checkpoint scale is missing, zero, "
            "or paired with the wrong weight tensor"
        )
    lost_energy = (product[zeroed] ** 2).sum().item() / total_energy
    if lost_energy > zeroed_energy_limit:
        raise ValueError(
            f"b12x FP4: folding weight_scale_2 underflowed "
            f"{int(zeroed.sum())}/{zeroed.numel()} {name} blockscales to "
            f"zero, dropping {lost_energy:.2%} of the total scale energy from "
            f"the GEMM (configured limit: {zeroed_energy_limit:.2%}). SM12X "
            "has no alternative single-GPU FP4 MoE backend; use non-FP4 MoE "
            f"weights or temporarily raise {_ZEROED_ENERGY_LIMIT_ENV}."
        )

    subnormal_frac = (
        ((folded_f32.abs() < _E4M3_MIN_NORMAL) & sf_nonzero & ~zeroed)
        .float()
        .mean()
        .item()
    )
    return zeroed, lost_energy, subnormal_frac


def _validate_execute_inputs(
    expert_x: Optional[torch.Tensor],
    topk_ids: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    expected_top_k: int,
    expected_hidden_size: int,
    activation: str,
    expert_map: Optional[torch.Tensor],
    a2_scale: Optional[torch.Tensor],
    apply_router_weight_on_input: bool,
    extra_expert_args: Optional[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate b12x runtime invariants without invoking a CUDA kernel."""
    if expert_x is None:
        raise ValueError("b12x requires expert activations")
    if expert_x.dtype is not torch.bfloat16:
        raise ValueError(
            f"b12x consumes bf16 activations directly, got {expert_x.dtype}"
        )
    if expert_x.ndim != 2 or expert_x.size(1) != expected_hidden_size:
        raise ValueError(
            "b12x activations must have shape [num_tokens, hidden_size] with "
            f"hidden_size={expected_hidden_size}, got {tuple(expert_x.shape)}"
        )
    if topk_ids is None or topk_weights is None:
        raise ValueError("b12x requires router top-k ids and weights")
    if topk_ids.dtype is not _B12X_TOPK_IDS_DTYPE:
        raise ValueError(
            "b12x requires top-k ids with dtype "
            f"{_B12X_TOPK_IDS_DTYPE}, got {topk_ids.dtype}"
        )
    if topk_weights.dtype is not _B12X_TOPK_WEIGHTS_DTYPE:
        raise ValueError(
            "b12x requires top-k weights with dtype "
            f"{_B12X_TOPK_WEIGHTS_DTYPE}, got {topk_weights.dtype}"
        )
    expected_router_shape = (expert_x.size(0), expected_top_k)
    if tuple(topk_ids.shape) != expected_router_shape:
        raise ValueError(
            "b12x top-k ids must have shape [num_tokens, top_k]: expected "
            f"{expected_router_shape}, got {tuple(topk_ids.shape)}"
        )
    if tuple(topk_weights.shape) != expected_router_shape:
        raise ValueError(
            "b12x top-k weights must have shape [num_tokens, top_k]: expected "
            f"{expected_router_shape}, got {tuple(topk_weights.shape)}"
        )
    if topk_ids.device != expert_x.device or topk_weights.device != expert_x.device:
        raise ValueError(
            "b12x activations, top-k ids, and top-k weights must share one "
            f"device, got {expert_x.device}, {topk_ids.device}, "
            f"{topk_weights.device}"
        )
    if apply_router_weight_on_input:
        raise ValueError(
            "b12x applies router weights inside the kernel; pre-applying them "
            "would weight the output twice"
        )
    if expert_map is not None:
        raise ValueError(
            "b12x indexes weights with global expert ids and does not support "
            "EP local-expert remapping"
        )
    if a2_scale is not None:
        raise ValueError(
            "b12x performs its own intermediate activation quantization and "
            "does not support an external a2_scale"
        )
    if extra_expert_args:
        raise ValueError(
            "b12x does not support extra expert arguments, got "
            f"{sorted(extra_expert_args)}"
        )
    # "siglu" is the normalized form of this repository's existing "SiGLU".
    if activation.lower() not in ("silu", "swiglu", "siglu"):
        raise ValueError(f"b12x MoE supports gated SiLU/SwiGLU only, got {activation}")
    return expert_x, topk_ids, topk_weights


class B12xFp4Executor(FusedMoeExpertExecutor):
    """flashinfer b12x CuTe DSL fused NVFP4 MoE executor for sm_120/sm_121."""

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.B12X_FP4

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return _B12X_TOPK_IDS_DTYPE

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )
        from rtp_llm.models_py.utils.arch import is_sm12x

        resolver = MoeConfigResolver()
        checker.check(resolver.is_bf16(config))
        checker.check(
            resolver.has_quantization(config)
            and resolver.get_quant_method(config) == "modelopt_fp4"
        )
        checker.check(is_sm12x())
        checker.check(resolver.is_single_gpu(config))

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        if config.enable_cuda_graph and config.ll_num_max_token <= 0:
            raise ValueError(
                "b12x FP4 CUDA Graph support requires ll_num_max_token > 0, got "
                f"{config.ll_num_max_token}"
            )

        self.w1 = weights.get(W.moe_w1, None)  # [E, 2*I, H//2] uint8, up-first
        self.w2 = weights.get(W.moe_w2, None)  # [E, H, I//2] uint8
        w1_sf = weights.get(W.moe_s1, None)  # fp8_e4m3, swizzled blockscale
        w2_sf = weights.get(W.moe_s2, None)  # fp8_e4m3, swizzled blockscale

        w1_scale_2 = weights.get(W.moe_w1_s2, None)  # [E] weight_scale_2 (w13)
        w2_scale_2 = weights.get(W.moe_w2_s2, None)  # [E] weight_scale_2 (w2)

        if self.w1 is None or self.w2 is None:
            raise ValueError("b12x FP4 needs moe_w1/moe_w2")
        if w1_sf is None or w2_sf is None:
            raise ValueError("b12x FP4 needs moe_s1/moe_s2")
        if w1_scale_2 is None or w2_scale_2 is None:
            raise ValueError("b12x FP4 needs weight_scale_2")

        self.num_experts = config.expert_num
        self.top_k = config.moe_k
        self.intermediate_size, self.hidden_size = _validate_b12x_weight_shapes(
            self.w1,
            self.w2,
            w1_sf,
            w2_sf,
            w1_scale_2,
            w2_scale_2,
            self.num_experts,
            get_b12x_kernel_tile_n(),
        )
        E, two_i, _ = self.w1.shape

        # Fold weight_scale_2 into the (already swizzled) block scale factors so
        # the kernel's per-block scales carry the full weight scale and the global
        # alphas can be 1 (a per-expert scalar multiply commutes with the swizzle
        # permutation). Passing weight_scale_2 through w1_alpha instead is NOT
        # equivalent: the sm12x dispatch feeds the same w1_alpha tensor to the
        # static/dynamic kernels as the activation-quantization global scale
        # (input_gs), so a non-unit alpha would change how activations are
        # quantized, not just rescale the FC1 output. Then convert to the 6D MMA
        # layout the kernel consumes; m = weight rows (2*I for w13, H for w2),
        # k = contraction dim.
        zeroed_energy_limit = _get_zeroed_energy_limit()
        _log_runtime_config_once(zeroed_energy_limit, get_disable_cuda12_9_compat())

        # Folding requantizes to e4m3. Measured behavior on sm_120:
        # - Overflow becomes NaN (torch e4m3 cast does not saturate): fatal.
        # - Exact underflow to zero (below ~2^-10) drops the whole 16-element
        #   weight block from the GEMM. Count it by the ENERGY those blocks
        #   carry, not by block count: real checkpoints legitimately hold a few
        #   percent of near-zero blocks whose loss is negligible (measured:
        #   5.98% zeroed blocks -> 0.00016% energy -> cosine 0.986 vs
        #   reference), while checkpoints that are genuinely too small for
        #   this path lose orders of magnitude more (0.33%..100%).
        # - SUBNORMAL folded scales are benign: the kernel reads them
        #   correctly (weights at randn*0.02 fold to 100% subnormal scales yet
        #   score cosine 0.98 vs reference when activations are healthy); the
        #   only cost is reduced scale mantissa precision. Output degradation
        #   observed with tiny weights AND tiny activations comes from the
        #   intermediate-activation dynamic quantization, a runtime property
        #   that cannot be checked against weights at load time.
        def fold_blockscale(
            name: str, blockscale: torch.Tensor, scale_2: torch.Tensor
        ) -> torch.Tensor:
            product = blockscale.to(torch.float32) * scale_2.reshape(E, 1, 1).to(
                torch.float32
            )
            folded = product.to(torch.float8_e4m3fn)
            zeroed, lost_energy, subnormal_frac = _validate_folded_blockscale(
                name, product, folded, zeroed_energy_limit
            )
            if zeroed.any():
                logger.warning(
                    "b12x FP4: %d/%d %s blockscale entries underflowed e4m3 "
                    "to zero while folding weight_scale_2 (%.4f%% of scale "
                    "energy; near-zero blocks, negligible precision impact).",
                    int(zeroed.sum()),
                    zeroed.numel(),
                    name,
                    lost_energy * 100,
                )
            if subnormal_frac > 0.5:
                logger.warning(
                    "b12x FP4: %.1f%% of %s blockscales are e4m3-subnormal "
                    "after folding weight_scale_2 (benign, but scale mantissa "
                    "precision is reduced for those blocks).",
                    subnormal_frac * 100,
                    name,
                )
            return folded

        w1_sf_folded = fold_blockscale("w1", w1_sf, w1_scale_2)
        self.w1_sf_mma = convert_b12x_blockscale_to_mma_layout(
            w1_sf_folded.reshape(-1).contiguous(),
            m=two_i,
            k=self.hidden_size,
            num_groups=E,
        )
        # ModelWeights owns this dictionary for the model lifetime. Replacing
        # the source scale here releases the old swizzled tensor after layer
        # construction instead of retaining both layouts for every layer.
        weights[W.moe_s1] = self.w1_sf_mma
        del w1_sf_folded, w1_sf, w1_scale_2

        w2_sf_folded = fold_blockscale("w2", w2_sf, w2_scale_2)
        self.w2_sf_mma = convert_b12x_blockscale_to_mma_layout(
            w2_sf_folded.reshape(-1).contiguous(),
            m=self.hidden_size,
            k=self.intermediate_size,
            num_groups=E,
        )
        weights[W.moe_s2] = self.w2_sf_mma
        weights.pop(W.moe_w1_s2, None)
        weights.pop(W.moe_w2_s2, None)
        del w2_sf_folded, w2_sf, w2_scale_2

        # Global scales are 1: weight scale is folded into the block factors, and
        # activation/intermediate quantization relies on per-block e4m3 scales.
        device = self.w1.device
        self.w1_alpha = torch.ones(E, dtype=torch.float32, device=device)
        self.w2_alpha = torch.ones(E, dtype=torch.float32, device=device)
        self.fc2_input_scale = torch.ones(1, dtype=torch.float32, device=device)

        # flashinfer-python 0.6.12rc1+rtp.260523 checks CUDA>=13 directly in
        # B12xMoEWrapper.__init__. Keeping the compatibility patch around the
        # pinned wrapper construction leaves forward execution and kernel JIT
        # under the real CUDA version.
        wrapper_args = dict(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            max_num_tokens=config.ll_num_max_token,
            output_dtype=torch.bfloat16,
            device=str(device),
            activation="silu",
            quant_mode="nvfp4",
            source_format="modelopt",
        )
        self._b12x_moe, self._b12x_moe_eager = create_b12x_wrappers(
            wrapper_args, config.enable_cuda_graph
        )

    @property
    def local_num_experts(self) -> int:
        assert self.w1 is not None
        return self.w1.size(0)

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        expert_x, topk_ids, topk_weights = _validate_execute_inputs(
            payload.expert_x,
            payload.expert_topk_ids,
            payload.expert_topk_weights,
            self.top_k,
            self.hidden_size,
            activation,
            expert_map,
            a2_scale,
            apply_router_weight_on_input,
            extra_expert_args,
        )

        wrapper = self._b12x_moe
        if (
            self._b12x_moe_eager is not None
            and expert_x.size(0) > self._b12x_moe.max_num_tokens
        ):
            wrapper = self._b12x_moe_eager

        output = wrapper.run(
            x=expert_x,  # [T, H] bf16
            w1_weight=self.w1,
            w1_weight_sf=self.w1_sf_mma,
            w2_weight=self.w2,
            w2_weight_sf=self.w2_sf_mma,
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
            w1_alpha=self.w1_alpha,
            w2_alpha=self.w2_alpha,
            fc2_input_scale=self.fc2_input_scale,
        )

        return CombineForwardPayload(fused_expert_output=output)

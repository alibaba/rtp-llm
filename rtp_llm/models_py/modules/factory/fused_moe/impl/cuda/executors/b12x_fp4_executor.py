import logging
import threading
from typing import Any, Dict, Optional

import torch

from rtp_llm.config.moe_config import (
    B12X_DISABLE_CUDA12_9_COMPAT_ENV,
    B12X_ZEROED_ENERGY_LIMIT_ENV,
)
from rtp_llm.device.flashinfer_b12x_adapter import (
    create_b12x_wrappers,
    get_b12x_kernel_tile_n,
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
    NVFP4_BLOCK_SIZE,
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)

_runtime_config_logged = False
_runtime_config_lock = threading.Lock()
_B12X_TOPK_IDS_DTYPE = torch.int32
_B12X_TOPK_WEIGHTS_DTYPE = torch.float32


def _validate_b12x_topology(config: MoEConfigAdapter) -> None:
    if config.ep_size != 1:
        raise ValueError(
            "b12x FP4 requires ep_size=1 because the kernel indexes full "
            f"weights with global expert ids, got ep_size={config.ep_size}"
        )


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
            B12X_ZEROED_ENERGY_LIMIT_ENV,
            zeroed_energy_limit,
            B12X_DISABLE_CUDA12_9_COMPAT_ENV,
            disable_cuda12_9_compat,
        )
        _runtime_config_logged = True


def _validate_b12x_weight_shapes(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_sf_mma: torch.Tensor,
    w2_sf_mma: torch.Tensor,
    num_experts: int,
    kernel_tile_n: int,
) -> tuple[int, int]:
    """Validate packed weights and loader-prepared MMA blockscales."""
    for name, weight in (("w1", w1), ("w2", w2)):
        if weight.dtype is not torch.uint8:
            raise ValueError(
                f"b12x FP4 {name} must contain packed uint8 weights, got "
                f"{weight.dtype}"
            )
    for name, scale in (("w1", w1_sf_mma), ("w2", w2_sf_mma)):
        if scale.dtype is not torch.float8_e4m3fn:
            raise ValueError(
                f"b12x FP4 {name} MMA blockscales must use torch.float8_e4m3fn, "
                f"got {scale.dtype}"
            )

    tensors = {
        "w1": w1,
        "w2": w2,
        "w1 MMA blockscale": w1_sf_mma,
        "w2 MMA blockscale": w2_sf_mma,
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

    expected_w1_sf = (
        32,
        4,
        two_i // 128,
        4,
        hidden_size // (4 * NVFP4_BLOCK_SIZE),
        num_experts,
    )
    expected_w2_sf = (
        32,
        4,
        hidden_size // 128,
        4,
        intermediate_size // (4 * NVFP4_BLOCK_SIZE),
        num_experts,
    )
    if tuple(w1_sf_mma.shape) != expected_w1_sf:
        raise ValueError(
            "b12x FP4 w1 blockscale must use the loader-prepared MMA layout: "
            f"expected {expected_w1_sf}, got {tuple(w1_sf_mma.shape)}"
        )
    if tuple(w2_sf_mma.shape) != expected_w2_sf:
        raise ValueError(
            "b12x FP4 w2 blockscale must use the loader-prepared MMA layout: "
            f"expected {expected_w2_sf}, got {tuple(w2_sf_mma.shape)}"
        )

    return intermediate_size, hidden_size


def _validate_execute_options(
    *,
    activation: str,
    expert_map: Optional[torch.Tensor],
    a2_scale: Optional[torch.Tensor],
    apply_router_weight_on_input: bool,
    extra_expert_args: Optional[dict[str, Any]],
) -> None:
    """Reject optional execution modes that the B12X kernel cannot represent."""
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


def _validate_execute_payload(
    payload: ExpertForwardPayload,
    *,
    expected_top_k: int,
    expected_hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate per-call tensors without invoking a CUDA kernel."""
    expert_x = payload.expert_x
    expert_x_scale = payload.expert_x_scale
    topk_ids = payload.expert_topk_ids
    topk_weights = payload.expert_topk_weights
    if expert_x is None:
        raise ValueError("b12x requires expert activations")
    if expert_x_scale is not None:
        raise ValueError(
            "b12x quantizes activations internally and does not support an "
            "external expert_x_scale"
        )
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
    if payload.expert_ids_are_local:
        raise ValueError("b12x requires global expert ids, got local expert ids")
    return expert_x, topk_ids, topk_weights


class B12xFp4Executor(FusedMoeExpertExecutor):
    """FlashInfer B12x NVFP4 MoE executor for sm_120/sm_121."""

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

        _validate_b12x_topology(config)
        expected_block_shape = [NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE]
        if quant_config.block_shape != expected_block_shape:
            raise ValueError(
                "b12x FP4 requires NVFP4 block_shape "
                f"{expected_block_shape}, got {quant_config.block_shape}"
            )

        if config.enable_cuda_graph and config.ll_num_max_token <= 0:
            raise ValueError(
                "b12x FP4 CUDA Graph support requires ll_num_max_token > 0, got "
                f"{config.ll_num_max_token}"
            )

        self.w1 = weights.get(W.moe_w1, None)  # [E, 2*I, H//2] uint8, up-first
        self.w2 = weights.get(W.moe_w2, None)  # [E, H, I//2] uint8
        # The model loader folds weight_scale_2 and converts the source scales
        # into FlashInfer's strided 6D MMA views before executor construction.
        self.w1_sf_mma = weights.get(W.moe_s1, None)
        self.w2_sf_mma = weights.get(W.moe_s2, None)

        if self.w1 is None or self.w2 is None:
            raise ValueError("b12x FP4 needs moe_w1/moe_w2")
        if self.w1_sf_mma is None or self.w2_sf_mma is None:
            raise ValueError("b12x FP4 needs loader-prepared moe_s1/moe_s2")

        self.num_experts = config.expert_num
        self.top_k = config.moe_k
        self.intermediate_size, self.hidden_size = _validate_b12x_weight_shapes(
            self.w1,
            self.w2,
            self.w1_sf_mma,
            self.w2_sf_mma,
            self.num_experts,
            get_b12x_kernel_tile_n(),
        )
        E = self.w1.size(0)
        device = self.w1.device
        _log_runtime_config_once(
            config.b12x_zeroed_energy_limit, config.b12x_disable_cuda12_9_compat
        )

        # Global scales are 1: weight scale is folded into the block factors, and
        # activation/intermediate quantization relies on per-block e4m3 scales.
        w1_alpha = torch.ones(E, dtype=torch.float32, device=device)
        w2_alpha = torch.ones(E, dtype=torch.float32, device=device)
        fc2_input_scale = torch.ones(1, dtype=torch.float32, device=device)

        # flashinfer-python 0.6.9 checks CUDA>=13 directly in
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
        )
        b12x_moe, b12x_moe_eager = create_b12x_wrappers(
            wrapper_args,
            config.enable_cuda_graph,
            config.b12x_disable_cuda12_9_compat,
        )

        self.w1_alpha = w1_alpha
        self.w2_alpha = w2_alpha
        self.fc2_input_scale = fc2_input_scale
        self._b12x_moe = b12x_moe
        self._b12x_moe_eager = b12x_moe_eager
        self._graph_max_num_tokens = (
            config.ll_num_max_token if config.enable_cuda_graph else None
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
        # Construction enforces ep_size=1 and this executor is paired with
        # PureTpRouterFp4PerGroup. Its recompute interval therefore covers all
        # experts [0, num_experts), so upstream top-k ids stay valid global ids.
        # Do not add a per-forward CUDA min/max here: that would synchronize the
        # B12x hot path solely to recheck the router/executor contract.
        _validate_execute_options(
            activation=activation,
            expert_map=expert_map,
            a2_scale=a2_scale,
            apply_router_weight_on_input=apply_router_weight_on_input,
            extra_expert_args=extra_expert_args,
        )
        expert_x, topk_ids, topk_weights = _validate_execute_payload(
            payload,
            expected_top_k=self.top_k,
            expected_hidden_size=self.hidden_size,
        )

        wrapper = self._b12x_moe
        if (
            self._b12x_moe_eager is not None
            and self._graph_max_num_tokens is not None
            and expert_x.size(0) > self._graph_max_num_tokens
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

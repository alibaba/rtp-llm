from typing import Any, Dict, Optional

import torch
import triton.language as tl

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
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul
from rtp_llm.models_py.triton_kernels.moe.fixed_order_route_reduce import (
    fixed_order_fp32_route_reduce,
)
from rtp_llm.models_py.triton_kernels.moe.fused_moe_kernel import (
    get_default_config,
    invoke_fused_moe_kernel,
    moe_align_block_size_torch,
)
from rtp_llm.utils.model_weight import W

from .decode_capabilities import DecodeCapabilities, decode_capabilities


class TritonFusedMoeExecutor(FusedMoeExpertExecutor):
    _batch_invariant = False

    @classmethod
    def executor_type(cls):
        return ExecutorType.FUSED_MOE

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(not resolver.has_quantization(config))

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.num_experts = config.expert_num
        assert self.num_experts % self.ep_size == 0
        self.num_experts_per_partition = self.num_experts // self.ep_size
        self.start_expert_id = self.ep_rank * self.num_experts_per_partition
        self.end_expert_id = self.start_expert_id + self.num_experts_per_partition - 1
        self.top_k = config.moe_k

        # w1: [E, 2*inter_size, hidden_size], w2: [E, hidden_size, inter_size]
        self.w1 = weights[W.moe_w1]
        self.w2 = weights[W.moe_w2]
        self.E, self.N, self.K = self.w1.shape
        self.inter_size = self.N // 2
        self._batch_invariant = getattr(config, "batch_invariant", False) is True
        if self._batch_invariant:
            capabilities = DecodeCapabilities(
                alignment=False, tuned_tiles=False, small_gemv=False
            )
        else:
            capabilities = decode_capabilities(
                self.w1,
                self.top_k,
                self.ep_size,
                getattr(getattr(config, "model_config", None), "model_type", ""),
            )
        self._decode_alignment = capabilities.alignment
        self._decode_tiles = capabilities.tuned_tiles
        self._small_moe_gemv = capabilities.small_gemv

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int32

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        hidden_states = payload.expert_x  # [M, K]
        topk_ids = payload.expert_topk_ids  # [M, top_k]
        topk_weights = payload.expert_topk_weights  # [M, top_k]

        assert topk_ids is not None
        assert topk_weights is not None

        if (
            self._small_moe_gemv
            and activation in ("silu", "SiGLU")
            and expert_map is None
            and a2_scale is None
            and not apply_router_weight_on_input
            and not extra_expert_args
        ):
            from rtp_llm.models_py.triton_kernels.moe.small_moe_gemv import (
                small_moe_gemv,
            )

            output = small_moe_gemv(
                hidden_states, self.w1, self.w2, topk_ids, topk_weights
            )
            if output is not None:
                return CombineForwardPayload(fused_expert_output=output)

        if self.ep_size > 1:
            local_ids = topk_ids - self.start_expert_id
            local_mask = (local_ids >= 0) & (local_ids < self.num_experts_per_partition)
            topk_ids = local_ids.clamp(min=0, max=self.num_experts_per_partition - 1)
            topk_weights = topk_weights * local_mask

        if self._batch_invariant:
            if activation not in ("silu", "SiGLU") or apply_router_weight_on_input:
                raise ValueError(
                    "batch-invariant experts require SwiGLU and output weighting"
                )
            if expert_map is not None or a2_scale is not None or extra_expert_args:
                raise ValueError(
                    "batch-invariant experts require unquantized inputs without expert remapping"
                )
            output = self._batch_invariant_experts(
                hidden_states, topk_ids, topk_weights
            )
            return CombineForwardPayload(fused_expert_output=output)

        M, K = hidden_states.shape
        top_k = topk_ids.size(1)
        # Config selection for both GEMMs
        config1 = get_default_config(M, self.E, self.N, K, top_k)
        config2 = get_default_config(M, self.E, K, self.inter_size, top_k)
        if self._decode_tiles:
            from rtp_llm.models_py.triton_kernels.moe.decode_alignment import (
                sm120_decode_gemm_configs,
            )

            config1, config2 = sm120_decode_gemm_configs(M, config1, config2)

        align = moe_align_block_size_torch
        if (
            self._decode_alignment
            and 0 < M <= 32
            and topk_ids.dtype == torch.int32
            and topk_ids.is_contiguous()
        ):
            from rtp_llm.models_py.triton_kernels.moe.decode_alignment import (
                moe_align_decode,
            )

            align = moe_align_decode
        routes = self._project_experts(
            hidden_states, topk_ids, topk_weights, config1, config2, align
        )
        output = routes.view(M, top_k, K).sum(dim=1)
        return CombineForwardPayload(fused_expert_output=output)

    def _projection_plan(self, hidden_states, topk_ids, topk_weights):
        from rtp_llm.models_py.determinism.batch_invariant import (
            batch_invariant_moe_plan,
        )

        return batch_invariant_moe_plan(
            hidden_states,
            self.w1,
            self.w2,
            topk_ids,
            topk_weights,
            ep_size=self.ep_size,
        )

    def _batch_invariant_experts(
        self, hidden_states, topk_ids, topk_weights
    ) -> torch.Tensor:
        tile, chunk_size = self._projection_plan(hidden_states, topk_ids, topk_weights)
        output = hidden_states.new_empty(hidden_states.shape)
        with torch.cuda.device_of(hidden_states):
            for start in range(0, hidden_states.size(0), chunk_size):
                chunk = slice(start, start + chunk_size)
                routes = self._project_experts(
                    hidden_states[chunk],
                    topk_ids[chunk].contiguous(),
                    topk_weights[chunk].contiguous(),
                    tile,
                    tile,
                    moe_align_block_size_torch,
                )
                fixed_order_fp32_route_reduce(routes, output[chunk], topk_ids.size(1))
                del routes  # Release this chunk before projecting the next.
        return output

    def _project_experts(
        self, hidden_states, topk_ids, topk_weights, config1, config2, align
    ) -> torch.Tensor:
        """Return one weighted down projection for each token and routing slot."""
        M, K = hidden_states.shape
        top_k = topk_ids.size(1)
        compute_type = (
            tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16
        )
        flat_topk_weights = topk_weights.view(-1)
        flat_topk_ids = topk_ids.view(-1)
        block_m1 = config1["BLOCK_SIZE_M"]
        block_m2 = config2["BLOCK_SIZE_M"]
        aligned1 = align(topk_ids, block_m1, self.E)
        aligned2 = (
            aligned1 if block_m1 == block_m2 else align(topk_ids, block_m2, self.E)
        )

        up_gate = hidden_states.new_empty((M * top_k, self.N))
        self._invoke_projection(
            hidden_states,
            self.w1,
            up_gate,
            flat_topk_weights,
            flat_topk_ids,
            *aligned1,
            mul_routed_weight=False,
            top_k=top_k,
            config=config1,
            compute_type=compute_type,
        )

        activated = self._activate(up_gate)
        del up_gate

        routes = hidden_states.new_empty((M * top_k, K))
        self._invoke_projection(
            activated,
            self.w2,
            routes,
            flat_topk_weights,
            flat_topk_ids,
            *aligned2,
            mul_routed_weight=True,
            top_k=1,
            config=config2,
            compute_type=compute_type,
        )

        return routes

    def _activate(self, up_gate):
        activated = up_gate.new_empty((up_gate.size(0), self.inter_size))
        return silu_and_mul(activated, up_gate)

    def _invoke_projection(self, *args, **kwargs):
        invoke_fused_moe_kernel(*args, **kwargs)


class BatchInvariantMXFP4Executor(TritonFusedMoeExecutor):
    """Native DeepSeek E2M1 expert weights with per-token FP8 activations."""

    @classmethod
    def check_conditions(cls, checker, config):
        checker.check(config.moe_quant_method == "FP8_FP4")
        checker.check(config.model_type in ("deepseek_v4", "deepseek_v41"))
        checker.check(config.data_type == torch.bfloat16)

    def __init__(self, config, quant_config, weights):
        super().__init__(config, quant_config, weights)
        self.K *= 2
        self.group_size = 32 if config.model_type == "deepseek_v41" else 128
        if (
            not self._batch_invariant
            or config.model_type not in ("deepseek_v4", "deepseek_v41")
            or config.tp_size != config.ep_size
            or config.dp_size != 1
            or config.tp_rank != config.ep_rank
            or self.E != self.num_experts_per_partition
            or self.K != config.hidden_size
            or self.inter_size != config.moe_inter_dim
            or self.N != 2 * self.inter_size
            or self.K % self.group_size
            or self.inter_size % self.group_size
            or self.w2.shape != (self.E, self.K, self.inter_size // 2)
        ):
            raise ValueError(
                "DeepSeek MXFP4 requires matching TP/EP groups and local expert shapes"
            )
        self.w1_scales = self._weight_scales(self.w1, weights[W.moe_s1])
        self.w2_scales = self._weight_scales(self.w2, weights[W.moe_s2])
        self.tile = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
        }

    @staticmethod
    def _weight_scales(weight, scales):
        if (
            weight.dtype != torch.int8
            or scales.dtype not in (torch.uint8, torch.float8_e8m0fnu)
            or weight.device != scales.device
            or scales.shape != (*weight.shape[:-1], weight.size(-1) // 16)
        ):
            raise ValueError(
                "DeepSeek routed weights require packed E2M1 and group-32 UE8M0 scales"
            )
        return scales.view(torch.uint8)

    def _projection_plan(self, hidden_states, topk_ids, topk_weights):
        if hidden_states.dtype != torch.bfloat16 or hidden_states.size(1) != self.K:
            raise ValueError(
                "MXFP4 experts require BF16 hidden states of the checkpoint width"
            )
        # Limit routed BF16 intermediates and FP8 activation buffers during prefill.
        bytes_per_token = topk_ids.size(1) * (6 * self.inter_size + 2 * self.K)
        return self.tile, max(1, (64 * 1024 * 1024) // bytes_per_token)

    def _activate(self, gate_up):
        from rtp_llm.models_py.triton_kernels.moe.silu_mul_split import silu_mul_split

        first, second = gate_up.chunk(2, dim=-1)
        gate, up = (
            (first, second)
            if self.config.moe_w1_layout == "gate_up"
            else (second, first)
        )
        return silu_mul_split(
            gate.float().contiguous(), up.float().contiguous(), self.config.swiglu_limit
        ).to(gate_up.dtype)

    def _invoke_projection(self, x, weight, *args, **kwargs):
        from rtp_llm.models_py.determinism.quantized_linear import quantize_fp8_blocks

        quantized, scales = quantize_fp8_blocks(
            x,
            True,
            self.group_size,
            eps=torch.finfo(torch.float32).tiny if self.group_size == 32 else 1e-4,
        )
        invoke_fused_moe_kernel(
            quantized,
            weight,
            *args,
            **kwargs,
            A_scale=scales,
            B_scale=self.w1_scales if weight is self.w1 else self.w2_scales,
            mxfp4_activation_group_size=self.group_size,
        )

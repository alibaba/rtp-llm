"""
Adapter to provide a unified interface from individual config objects.
This allows Router and Executor classes to work with specific config objects.
"""

from typing import TYPE_CHECKING, Optional, Union

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.ops import MoeConfig, ParallelismConfig

if TYPE_CHECKING:
    from rtp_llm.config.quant_config import QuantizationConfig


class _UnsetQuantConfig:
    pass


_UNSET_QUANT_CONFIG = _UnsetQuantConfig()


class MoEConfigAdapter:
    """
    Adapter class that provides a unified interface
    from individual configuration objects.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        moe_config: Optional[MoeConfig] = None,
        # Omitted means inherit the model-level config; explicit None means
        # this layer is intentionally excluded from quantization.
        quant_config: Union["QuantizationConfig", None, _UnsetQuantConfig] = (
            _UNSET_QUANT_CONFIG
        ),
        enable_cuda_graph: bool = False,
    ):
        if not isinstance(enable_cuda_graph, bool):
            raise TypeError("enable_cuda_graph must be a bool")
        self.model_config = model_config
        self.parallelism_config = parallelism_config
        self.moe_config = moe_config if moe_config is not None else MoeConfig()
        self.quant_config = (
            model_config.quant_config
            if quant_config is _UNSET_QUANT_CONFIG
            else quant_config
        )

        # Provide shortcut access to commonly used attributes
        self.ep_size = parallelism_config.ep_size
        self.ep_rank = parallelism_config.ep_rank
        # tp_size/tp_rank reflect the attention/MoE-input view: when CP is
        # enabled, get_attn_tp_size() returns 1, so MoE input slicing
        # (deepep narrow/allgather) stays a no-op. Router selectors that
        # need the physical TP topology (e.g. pure_cp_router) read raw
        # parallelism_config.tp_size via is_cp_equal_ep().
        self.tp_size = parallelism_config.get_attn_tp_size()
        self.tp_rank = parallelism_config.get_attn_tp_rank()
        self.dp_size = parallelism_config.dp_size
        self.dp_rank = parallelism_config.dp_rank
        self.world_size = parallelism_config.world_size
        self.world_rank = parallelism_config.world_rank
        # Calculate local_rank from world_rank and local_world_size
        self.local_rank = parallelism_config.local_rank

        self.expert_num = model_config.expert_num
        self.physical_expert_num = int(
            model_config.eplb_config.phy_exp_num(self.expert_num)
        )
        self.has_redundant_experts = self.physical_expert_num != self.expert_num
        self.moe_k = model_config.moe_k
        self.moe_topk_group = model_config.moe_topk_group
        self.hidden_size = model_config.hidden_size
        self.dim = self.hidden_size
        # The generic adapter is not layer-bound. Executors with per-layer
        # state receive a typed runtime config from their layer implementation.
        self.layer_id = -1
        self.moe_inter_dim = int(model_config.moe_inter_size)
        self.moe_w1_layout = model_config.moe_w1_layout
        self.n_routed_experts = self.expert_num
        self.n_activated_experts = self.moe_k
        self.route_scale = float(model_config.routed_scaling_factor)
        self.n_shared_experts = int(model_config.n_shared_experts)
        if self.n_shared_experts == 0 and model_config.moe_style == 2:
            routed_inter_size = int(model_config.moe_inter_size)
            shared_inter_size = int(model_config.inter_size)
            if (
                routed_inter_size <= 0
                or shared_inter_size <= 0
                or shared_inter_size % routed_inter_size != 0
            ):
                raise ValueError(
                    "moe_style=2 requires explicit n_shared_experts or legacy "
                    "shared-expert metadata with positive, divisible dimensions; "
                    f"got n_shared_experts=0, inter_size={shared_inter_size}, "
                    f"moe_inter_size={routed_inter_size}"
                )
            self.n_shared_experts = shared_inter_size // routed_inter_size
        self.has_shared_expert_gate = False
        self.swiglu_limit = float(model_config.swiglu_limit)
        if self.physical_expert_num % max(self.ep_size, 1) != 0:
            raise ValueError(
                f"physical_expert_num={self.physical_expert_num} must be "
                f"divisible by ep_size={self.ep_size}"
            )
        self.n_local_experts = self.physical_expert_num // max(self.ep_size, 1)
        self.local_expert_start = self.ep_rank * self.n_local_experts
        self.local_expert_end = self.local_expert_start + self.n_local_experts
        # Decode concurrency and prefill sequence length are independent
        # capacity requirements. A configured low-latency decode limit must
        # not shrink buffers needed by a longer prefill request.
        self.decode_max_tokens_per_rank = int(self.moe_config.ll_num_max_token or 0)
        prefill_capacity = model_config.moe_prefill_max_tokens_per_rank
        self.prefill_max_tokens_per_rank = int(
            model_config.max_seq_len if prefill_capacity is None else prefill_capacity
        )
        self.max_tokens_per_rank = max(
            self.decode_max_tokens_per_rank,
            self.prefill_max_tokens_per_rank,
            1,
        )
        # Generic execution is not chunked, so JIT warmup only needs the
        # request-visible bucket representatives rather than the capacity cap.
        self.warmup_include_capacity = False
        effective_quant_config = (
            quant_config if quant_config is not None else model_config.quant_config
        )
        self.moe_quant_method = (
            effective_quant_config.get_method()
            if effective_quant_config is not None
            else None
        )
        self.data_type = model_config.data_type
        self.head_num = model_config.attn_config.head_num
        self.ll_num_max_token = self.moe_config.ll_num_max_token
        self.masked_max_token_num = self.moe_config.masked_max_token_num
        self.moe_strategy = self.moe_config.moe_strategy
        self.use_mori_ep = self.moe_config.use_mori_ep
        self.use_deepep_moe = self.moe_config.use_deepep_moe
        self.enable_cuda_graph = enable_cuda_graph

    @property
    def activation_type(self):
        """Access activation_type from model_config when needed."""
        return self.model_config.activation_type

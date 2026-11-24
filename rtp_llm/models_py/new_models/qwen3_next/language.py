"""Qwen3-Next core model for the streaming newloader."""

from __future__ import annotations

import logging
import math
from numbers import Real
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.downstream_heads import DecoderHandlerOwner, needs_generation_projection
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.layers.embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    output_vocab_head_kwargs,
)
from rtp_llm.models_py.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.layers.norm import RMSNorm, RMSResNorm
from rtp_llm.models_py.model_desc.generic_moe import (
    GenericMoeLayer,
    use_unified_moe_tp_reduce,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextMetadata, Qwen3NextModel
from rtp_llm.models_py.model_desc.qwen3_next_mtp import Qwen3NextMTPModel
from rtp_llm.models_py.module_base import CheckpointLoadTarget, RtpModule
from rtp_llm.models_py.modules import MultimodalEmbeddingInjector
from rtp_llm.models_py.new_models.model_base import required_config_value
from rtp_llm.models_py.new_models.qwen3.language import (
    Qwen3Attention,
    _positive_int,
    _validate_supported_parallelism,
)
from rtp_llm.models_py.new_models.qwen3_next.gdn import Qwen3NextGatedDeltaNet
from rtp_llm.models_py.new_models.qwen3_next.gdn_parameters import _finite_positive_real
from rtp_llm.models_py.new_models.qwen_visual_namespace import is_visual_checkpoint_name
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.read_spec import unwrap_checkpoint_tensor
from rtp_llm.models_py.weight_mapper import WeightsMapper
from rtp_llm.ops import HybridAttentionType

if TYPE_CHECKING:
    from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, PyModelInputs

logger = logging.getLogger(__name__)


def split_q_gate(ts: List[torch.Tensor], head_num: int, part: int):
    """Split q_gate tensor into q or gate part."""
    if len(ts) != 1 or ts[0].dim() != 2:
        raise ValueError("q_gate transform expects one two-dimensional tensor")
    if part not in (0, 1):
        raise ValueError(f"q_gate part must be 0 or 1, got {part}")
    dim0, dim1 = ts[0].shape
    if dim0 % (head_num * 2) != 0:
        raise ValueError(
            f"q_gate rows {dim0} must be divisible by 2 * heads ({head_num})"
        )
    new_head_dim = dim0 // (head_num * 2)
    t = ts[0].reshape(head_num, 2, new_head_dim, dim1)
    return t[:, part, :, :].reshape(-1, dim1).contiguous()


def _plus_one(tensor: torch.Tensor) -> torch.Tensor:
    """Add 1 for gemma_rms_norm."""
    return tensor + 1


# ================================================================== #
#  Config extraction
# ================================================================== #


def _config_bool(model_config: ModelConfig, name: str, default: bool) -> bool:
    value = getattr(model_config, name, default)
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _validate_partition(size: Any, rank: Any, name: str) -> Tuple[int, int]:
    size = _positive_int(size, f"{name} size")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < size:
        raise ValueError(f"Invalid {name} partition: rank={rank}, size={size}")
    return size, rank


def _extract_config_values(
    model_config: ModelConfig,
    load_config: Any,
    *,
    expects_mtp: bool = False,
) -> Dict[str, Any]:
    """Validate the typed runtime configuration used by Qwen3-Next."""
    if not isinstance(model_config, ModelConfig):
        raise TypeError(
            "Qwen3-Next newloader requires a typed ModelConfig; normalize raw "
            "Hugging Face config.json data at the BaseModel boundary"
        )

    hidden_size = _positive_int(
        required_config_value(model_config, "hidden_size"), "hidden_size"
    )
    num_layers = _positive_int(
        required_config_value(model_config, "num_layers"), "num_layers"
    )
    vocab_size = _positive_int(
        required_config_value(model_config, "vocab_size"), "vocab_size"
    )
    inter_size = required_config_value(model_config, "inter_size")
    if (
        isinstance(inter_size, bool)
        or not isinstance(inter_size, int)
        or inter_size < 0
    ):
        raise ValueError("inter_size must be a non-negative integer")

    attn_config = required_config_value(model_config, "attn_config")
    num_heads = _positive_int(
        required_config_value(attn_config, "head_num"), "num_heads"
    )
    num_kv_heads = required_config_value(attn_config, "kv_head_num")
    if num_kv_heads == -1:
        num_kv_heads = num_heads
    num_kv_heads = _positive_int(num_kv_heads, "num_kv_heads")
    head_dim = _positive_int(
        required_config_value(attn_config, "size_per_head"), "head_dim"
    )
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}"
        )

    rms_norm_eps = _finite_positive_real(
        required_config_value(model_config, "layernorm_eps"), "layernorm_eps"
    )
    partial_rotary_factor = _finite_positive_real(
        required_config_value(model_config, "partial_rotary_factor"),
        "partial_rotary_factor",
    )
    if partial_rotary_factor > 1.0:
        raise ValueError("partial_rotary_factor cannot exceed 1.0")

    linear_attention_config = required_config_value(
        model_config, "linear_attention_config"
    )
    linear_fields = {
        "linear_conv_kernel_dim": "linear_conv_kernel_dim",
        "linear_key_head_dim": "linear_key_head_dim",
        "linear_num_key_heads": "linear_num_key_heads",
        "linear_num_value_heads": "linear_num_value_heads",
        "linear_value_head_dim": "linear_value_head_dim",
    }
    linear_values = {
        key: _positive_int(required_config_value(linear_attention_config, field), field)
        for key, field in linear_fields.items()
    }
    if (
        linear_values["linear_num_value_heads"] % linear_values["linear_num_key_heads"]
        != 0
    ):
        raise ValueError(
            "linear_num_value_heads must be divisible by linear_num_key_heads"
        )

    hybrid_config = required_config_value(model_config, "hybrid_attention_config")
    enabled = getattr(hybrid_config, "enable_hybrid_attention", False)
    if not isinstance(enabled, bool) or not enabled:
        raise ValueError("Qwen3-Next requires enabled hybrid attention")
    hybrid_types = list(required_config_value(hybrid_config, "hybrid_attention_types"))
    if len(hybrid_types) != num_layers:
        raise ValueError(
            "hybrid_attention_types must contain exactly one entry per layer: "
            f"expected {num_layers}, got {len(hybrid_types)}"
        )
    supported_types = (HybridAttentionType.NONE, HybridAttentionType.LINEAR)
    invalid_types = [value for value in hybrid_types if value not in supported_types]
    if invalid_types:
        raise ValueError(f"Unsupported Qwen3-Next attention types: {invalid_types}")

    moe_layer_index = list(required_config_value(model_config, "moe_layer_index"))
    if len(set(moe_layer_index)) != len(moe_layer_index):
        raise ValueError("moe_layer_index contains duplicate layers")
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < num_layers
        for index in moe_layer_index
    ):
        raise ValueError(f"Invalid moe_layer_index {moe_layer_index!r}")
    # Qwen3.5 Dense uses the same hybrid-attention blocks, but its normalized
    # config has no routed experts. Do not require (or synthesize) an MoE
    # topology merely to construct its ordinary gate/up/down MLP owners.
    expert_num = required_config_value(model_config, "expert_num")
    if (
        isinstance(expert_num, bool)
        or not isinstance(expert_num, int)
        or expert_num < 0
    ):
        raise ValueError("expert_num must be a non-negative integer")
    if not inter_size and len(moe_layer_index) < num_layers:
        raise ValueError("dense Qwen layers need a positive FFN intermediate size")
    shared_inter_size = 0
    if moe_layer_index or expert_num:
        expert_num = _positive_int(expert_num, "expert_num")
        moe_k = _positive_int(required_config_value(model_config, "moe_k"), "moe_k")
        if moe_k > expert_num:
            raise ValueError(f"moe_k={moe_k} cannot exceed expert_num={expert_num}")
        moe_inter_size = _positive_int(
            required_config_value(model_config, "moe_inter_size"), "moe_inter_size"
        )
        moe_style = required_config_value(model_config, "moe_style")
        if moe_style not in (1, 2):
            raise ValueError(f"Invalid Qwen routed MoE style {moe_style!r}")
        if moe_style == 2:
            shared_inter_size = _positive_int(
                inter_size, "shared expert intermediate size"
            )
    else:
        moe_k = 0
        moe_inter_size = 0
    has_moe_norm = _config_bool(model_config, "has_moe_norm", True)
    tie_word_embeddings = _config_bool(model_config, "tie_word_embeddings", False)
    if not isinstance(expects_mtp, bool):
        raise TypeError("expects_mtp must be a bool")
    is_mtp = _config_bool(model_config, "is_mtp", False)
    if is_mtp != expects_mtp:
        expected = "MTP" if expects_mtp else "core"
        raise ValueError(f"{expected} Qwen3-Next loader received is_mtp={is_mtp}")

    tp_size, tp_rank = _validate_partition(
        required_config_value(load_config, "tp_size"),
        required_config_value(load_config, "tp_rank"),
        "TP",
    )
    attn_tp_size, attn_tp_rank = _validate_partition(
        required_config_value(load_config, "attn_tp_size"),
        required_config_value(load_config, "attn_tp_rank"),
        "attention TP",
    )
    ffn_tp_size, ffn_tp_rank = _validate_partition(
        required_config_value(load_config, "ffn_tp_size"),
        required_config_value(load_config, "ffn_tp_rank"),
        "FFN TP",
    )
    lm_head_tp_size, lm_head_tp_rank = _validate_partition(
        required_config_value(load_config, "lm_head_tp_size"),
        required_config_value(load_config, "lm_head_tp_rank"),
        "LM head TP",
    )
    ep_size, ep_rank = _validate_partition(
        required_config_value(load_config, "ep_size"),
        required_config_value(load_config, "ep_rank"),
        "EP",
    )
    parallelism_config = required_config_value(load_config, "parallelism_config")
    context_parallel = parallelism_config.prefill_cp_config.is_enabled()
    if context_parallel:
        if (attn_tp_size, attn_tp_rank, ffn_tp_size, ffn_tp_rank) != (1, 0, 1, 0):
            raise ValueError(
                "Qwen3-Next Context parallelism requires replicated attention "
                "and shared/dense FFN parameters (TP=1/rank=0)"
            )
    elif (attn_tp_size, attn_tp_rank) != (tp_size, tp_rank):
        raise ValueError(
            "Context parallelism is not supported by the Qwen3-Next core slice"
        )
    if not context_parallel and (ffn_tp_size, ffn_tp_rank) != (tp_size, tp_rank):
        raise ValueError(
            "Independent FFN TP is not supported by the Qwen3-Next core slice"
        )
    if (lm_head_tp_size, lm_head_tp_rank) != (tp_size, tp_rank):
        raise ValueError(
            "A separate LM head topology is not supported by the Qwen3-Next core slice"
        )
    if expert_num % ep_size != 0:
        raise ValueError(
            f"expert_num={expert_num} must be divisible by ep_size={ep_size}"
        )
    if linear_values["linear_num_key_heads"] % attn_tp_size != 0:
        raise ValueError("linear_num_key_heads must be divisible by attention TP")
    if linear_values["linear_num_value_heads"] % attn_tp_size != 0:
        raise ValueError("linear_num_value_heads must be divisible by attention TP")

    _validate_supported_parallelism(parallelism_config)
    moe_config = required_config_value(load_config, "moe_config")
    dp_size, dp_rank = _validate_partition(
        getattr(parallelism_config, "dp_size", None),
        getattr(parallelism_config, "dp_rank", None),
        "DP",
    )
    del dp_size, dp_rank
    eplb_config = getattr(model_config, "eplb_config", None)
    enable_eplb = getattr(eplb_config, "enable_eplb", False)
    if callable(enable_eplb):
        enable_eplb = enable_eplb()
    if not isinstance(enable_eplb, bool):
        raise TypeError("eplb_config.enable_eplb must be a bool")

    params_dtype = required_config_value(load_config, "compute_dtype")
    if not isinstance(params_dtype, torch.dtype):
        raise TypeError("compute_dtype must be a torch.dtype")
    enable_fp32_lm_head = _config_bool(model_config, "enable_fp32_lm_head", True)
    return dict(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        vocab_size=vocab_size,
        rms_norm_eps=rms_norm_eps,
        hybrid_types=hybrid_types,
        moe_k=moe_k,
        expert_num=expert_num,
        moe_inter_size=moe_inter_size,
        inter_size=inter_size,
        shared_inter_size=shared_inter_size,
        moe_layer_index=moe_layer_index,
        has_moe_norm=has_moe_norm,
        partial_rotary_factor=partial_rotary_factor,
        tie_word_embeddings=tie_word_embeddings,
        tp_size=tp_size,
        tp_rank=tp_rank,
        attn_tp_size=attn_tp_size,
        attn_tp_rank=attn_tp_rank,
        ffn_tp_size=ffn_tp_size,
        ffn_tp_rank=ffn_tp_rank,
        lm_head_tp_size=lm_head_tp_size,
        lm_head_tp_rank=lm_head_tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        quant_config=getattr(load_config, "quant_config", None),
        params_dtype=params_dtype,
        lm_head_params_dtype=torch.float32 if enable_fp32_lm_head else params_dtype,
        model_config=model_config,
        linear_attention_config=linear_attention_config,
        parallelism_config=parallelism_config,
        moe_config=moe_config,
        **linear_values,
    )


class Qwen3NextAttention(Qwen3Attention):
    """Standard MHA attention with a gate projection.

    Q-Gate fusion: the checkpoint q_proj.weight contains both Q and Gate
    interleaved per-head. During load_weights, split_q_gate separates them.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_idx: int,
        tp_size: int,
        tp_rank: int,
        rms_norm_eps: float,
        params_dtype: torch.dtype,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            layer_idx=layer_idx,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=prefix,
            params_dtype=params_dtype,
            rms_norm_eps=rms_norm_eps,
        )
        self.gate = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=num_heads * head_dim,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
            bias=False,
            params_dtype=params_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        gate = self.gate(hidden_states)
        input_shape = hidden_states.shape[:-1]
        qkv = self.qkv_proj(hidden_states)
        qkv = self._apply_qk_norm(qkv)
        attn_output = fmha_impl.forward(qkv, kv_cache, self.layer_idx)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)


# ================================================================== #
#  MLP — Dense and MoE with shared expert
# ================================================================== #


class Qwen3NextMLP(RtpModule):
    """Dense MLP (SiGLU activation)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int,
        tp_rank: int,
        params_dtype: torch.dtype,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        self.tp_size = tp_size
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_size=2 * intermediate_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            bias=False,
            # rtp_llm fused silu_and_mul consumes [gate, up].
            shard_names=["gate_proj", "up_proj"],
            params_dtype=params_dtype,
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            bias=False,
            params_dtype=params_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from rtp_llm.models_py.layers.activation import silu_and_mul

        gate_up = self.gate_up_proj(x)
        act = silu_and_mul(gate_up)
        return self.down_proj(act)


class Qwen3NextSharedExpert(RtpModule):
    """Shared expert FFN (always active alongside routed experts).

    Note: gate_up_proj / down_proj are placed directly on this module (not
    nested inside a sub-MLP) so that the RtpModule redirect mechanism can
    match checkpoint keys 'shared_expert.gate_proj.weight' etc. directly.
    """

    def __init__(
        self,
        hidden_size: int,
        inter_size: int,
        tp_size: int,
        tp_rank: int,
        params_dtype: torch.dtype,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        self.tp_size = tp_size
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_size=2 * inter_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            bias=False,
            shard_names=["gate_proj", "up_proj"],
            params_dtype=params_dtype,
        )
        self.down_proj = RowParallelLinear(
            input_size=inter_size,
            output_size=hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            bias=False,
            params_dtype=params_dtype,
            reduce_output=False,
        )

    def forward(self, x: torch.Tensor, skip_allreduce: bool = False) -> torch.Tensor:
        from rtp_llm.models_py.layers.activation import silu_and_mul

        gate_up = self.gate_up_proj(x)
        act = silu_and_mul(gate_up)
        x = self.down_proj(act)
        if not skip_allreduce and self.tp_size > 1:
            x = all_reduce(x, group=Group.TP)
        return x


class Qwen3NextMoEBlock(RtpModule):
    """MoE block with routed experts + shared expert + shared_expert_gate.

    moe_style=2: shared expert + routed experts (Qwen3-Next style).
    """

    def __init__(
        self,
        hidden_size: int,
        moe_inter_size: int,
        expert_num: int,
        top_k: int,
        shared_inter_size: int,
        layer_idx: int,
        tp_size: int,
        tp_rank: int,
        ep_size: int,
        ep_rank: int,
        model_config: ModelConfig,
        parallelism_config: Any,
        moe_config: Any,
        quant_config: Optional[QuantizationConfig],
        params_dtype: torch.dtype,
        prefix: str,
        has_shared_expert_gate: bool = True,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.tp_size = tp_size
        self.ep_size = ep_size
        self.top_k = top_k
        self.hidden_size = hidden_size

        if model_config.expert_num != expert_num or model_config.moe_k != top_k:
            raise ValueError(
                "Qwen3-Next router dimensions disagree with ModelConfig: "
                f"experts={model_config.expert_num}/{expert_num}, "
                f"top_k={model_config.moe_k}/{top_k}"
            )
        if not isinstance(has_shared_expert_gate, bool):
            raise TypeError("has_shared_expert_gate must be a bool")
        self.gate = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=expert_num,
            tp_size=1,
            tp_rank=0,
            # Keep routing logits in the checkpoint dtype.  The legacy
            # online-FP8 loader deliberately excludes W.moe_gate from its
            # quantization allowlist: quantizing this small projection can
            # change top-k expert selection even when the expert weights
            # themselves are quantized correctly.
            quant_config=None,
            prefix=f"{prefix}.gate",
            bias=False,
            params_dtype=params_dtype,
        )
        self._router_config = model_config
        self.select_topk = None
        fake_balance = getattr(moe_config, "fake_balance_expert", False)
        if not isinstance(fake_balance, bool):
            raise TypeError("moe_config.fake_balance_expert must be a bool")
        self._fake_balance_config = (
            dict(
                expert_num=expert_num,
                moe_k=top_k,
                dp_rank=parallelism_config.dp_rank,
                dp_size=parallelism_config.dp_size,
                ep_size=ep_size,
            )
            if fake_balance
            else None
        )
        self.fake_balance_expert = None

        # Routed experts (reuse Qwen3Experts)
        from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts

        self.experts = BaseMoEExperts(
            num_experts=expert_num,
            top_k=top_k,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_inter_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            params_dtype=params_dtype,
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            quant_config=quant_config,
            layer_idx=layer_idx,
            prefix=f"{prefix}.experts",
        )

        # Shared expert
        self.shared_expert = (
            Qwen3NextSharedExpert(
                hidden_size,
                shared_inter_size,
                tp_size,
                tp_rank,
                params_dtype,
                quant_config,
                prefix=f"{prefix}.shared_expert",
            )
            if shared_inter_size
            else None
        )

        # Shared expert gate
        if self.shared_expert is not None and has_shared_expert_gate:
            self.shared_expert_gate = ColumnParallelLinear(
                input_size=hidden_size,
                output_size=1,
                tp_size=1,
                tp_rank=0,
                # Match the legacy FP8 allowlist.  This sigmoid gate is not a
                # GEMM worth quantizing and its numerical value directly
                # scales the shared-expert contribution.
                quant_config=None,
                prefix=f"{prefix}.shared_expert_gate",
                bias=False,
                params_dtype=params_dtype,
            )
        else:
            self.shared_expert_gate = None
        self.experts.has_shared_expert_gate = self.shared_expert_gate is not None
        self.sigmoid_gate_scale_add = None
        self.use_ep_shared_allreduce = (
            self.shared_expert is not None and self.tp_size > 1 and self.ep_size > 1
        )

    @property
    def use_unified_tp_allreduce(self) -> bool:
        # The expert executor (and hence its actual collective) is selected
        # after loading, not in the decoder constructor.
        router = getattr(self.experts.fused_moe, "router", None)
        return use_unified_moe_tp_reduce(
            has_shared_expert=self.shared_expert is not None,
            ffn_tp_size=self.tp_size,
            ep_size=self.ep_size,
            router=router,
        )

    def _bind_router(self) -> None:
        if self.select_topk is not None:
            return
        from rtp_llm.models_py.modules import SelectTopk

        selector = SelectTopk(config=self._router_config)
        if self.shared_expert_gate is not None:
            from rtp_llm.models_py.modules import SigmoidGateScaleAdd

            self.sigmoid_gate_scale_add = SigmoidGateScaleAdd()
        if self._fake_balance_config is not None:
            from rtp_llm.models_py.modules import FakeBalanceExpert

            self.fake_balance_expert = FakeBalanceExpert(**self._fake_balance_config)
        self.select_topk = selector

    def validate_runtime_device(self, device: torch.device) -> None:
        # A CPU staging load needs no accelerator routing kernel. Preserve
        # load-time failure for any device that will execute this model.
        if device.type != "cpu":
            self._bind_router()

    def _forward_routed(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.experts.fused_moe is None:
            raise RuntimeError(
                "Qwen3-Next MoE executor is not initialized; load and "
                "postprocess all expert weights before forward"
            )
        if hidden_states.dim() != 2:
            raise ValueError(
                "Qwen3-Next MoE expects a two-dimensional token matrix, got "
                f"{tuple(hidden_states.shape)}"
            )
        num_tokens = hidden_states.shape[0]
        router_logits = self.gate(hidden_states).float()

        topk_weights = torch.empty(
            (num_tokens, self.top_k), dtype=torch.float32, device=hidden_states.device
        )
        topk_ids = torch.empty(
            (num_tokens, self.top_k),
            dtype=self.experts.topk_ids_dtype,
            device=hidden_states.device,
        )
        self._bind_router()
        self.select_topk(router_logits, topk_ids, topk_weights)
        if self.fake_balance_expert is not None:
            self.fake_balance_expert(topk_ids, topk_weights)

        return self.experts(
            hidden_states,
            topk_weights,
            topk_ids,
            skip_tp_allreduce=self.use_unified_tp_allreduce,
        )

    # Checkpoint parameters stay in this owner. Reuse the inference runtime's
    # EP/TP reductions and single-rounding sigmoid gate, including the
    # non-unified branch for executors which must finalize their own output.
    _merge_shared_expert_output = GenericMoeLayer._merge_shared_expert_output
    _gate_shared_expert_output = GenericMoeLayer._gate_shared_expert_output
    _add_shared_expert = GenericMoeLayer._add_shared_expert
    forward = GenericMoeLayer.forward


# ================================================================== #
#  Decoder Layer
# ================================================================== #


class Qwen3NextDecoderLayer(RtpModule):

    def __init__(
        self,
        cfg: Dict[str, Any],
        layer_idx: int,
        model_config: ModelConfig,
        prefix: str,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        hidden_size = cfg["hidden_size"]
        rms_eps = cfg["rms_norm_eps"]

        self.layer_type = cfg["hybrid_types"][layer_idx]

        # Attention — checkpoint uses 'linear_attn.*' for Linear Attention layers
        # and 'self_attn.*' for standard MHA layers, so the attribute name must
        # match the checkpoint key prefix.
        if self.layer_type == HybridAttentionType.LINEAR:
            self.linear_attn = Qwen3NextGatedDeltaNet(
                parallelism_config=cfg["parallelism_config"],
                norm_activation=getattr(
                    model_config, "linear_attn_norm_activation", None
                )
                or "silu",
                linear_attn_config=model_config.linear_attention_config,
                hidden_size=hidden_size,
                rms_norm_eps=rms_eps,
                attn_tp_size=cfg["attn_tp_size"],
                attn_tp_rank=cfg["attn_tp_rank"],
                params_dtype=cfg["params_dtype"],
                quant_config=cfg["quant_config"],
                prefix=f"{prefix}.linear_attn",
            )
            self.linear_attn.layer_idx = layer_idx
        else:
            self.self_attn = Qwen3NextAttention(
                hidden_size=hidden_size,
                num_heads=cfg["num_heads"],
                num_kv_heads=cfg["num_kv_heads"],
                head_dim=cfg["head_dim"],
                layer_idx=layer_idx,
                tp_size=cfg["attn_tp_size"],
                tp_rank=cfg["attn_tp_rank"],
                rms_norm_eps=rms_eps,
                params_dtype=cfg["params_dtype"],
                quant_config=cfg["quant_config"],
                prefix=f"{prefix}.self_attn",
            )

        # MLP / MoE
        if layer_idx in cfg["moe_layer_index"]:
            self.mlp = Qwen3NextMoEBlock(
                hidden_size=hidden_size,
                moe_inter_size=cfg["moe_inter_size"],
                expert_num=cfg["expert_num"],
                top_k=cfg["moe_k"],
                shared_inter_size=cfg["shared_inter_size"],
                layer_idx=layer_idx,
                tp_size=cfg["ffn_tp_size"],
                tp_rank=cfg["ffn_tp_rank"],
                ep_size=cfg["ep_size"],
                ep_rank=cfg["ep_rank"],
                model_config=model_config,
                parallelism_config=cfg["parallelism_config"],
                moe_config=cfg["moe_config"],
                quant_config=cfg["quant_config"],
                params_dtype=cfg["params_dtype"],
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=hidden_size,
                intermediate_size=cfg["inter_size"],
                tp_size=cfg["ffn_tp_size"],
                tp_rank=cfg["ffn_tp_rank"],
                params_dtype=cfg["params_dtype"],
                quant_config=cfg["quant_config"],
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSResNorm(
            hidden_size, eps=rms_eps, params_dtype=cfg["params_dtype"]
        )
        self.post_attention_layernorm = RMSResNorm(
            hidden_size, eps=rms_eps, params_dtype=cfg["params_dtype"]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        attn_meta: Optional[Qwen3NextMetadata] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.layer_type == HybridAttentionType.LINEAR:
            hidden_states = self.linear_attn(
                hidden_states, fmha_impl, kv_cache, attention_inputs, attn_meta
            )
        else:
            hidden_states = self.self_attn(hidden_states, fmha_impl, kv_cache)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# ================================================================== #
#  Model classes
# ================================================================== #


class _StageLayers(nn.ModuleDict):
    """Dispatch by global checkpoint layer ID; execute in stage-local order."""

    def __iter__(self):
        return iter(self.values())


class Qwen3NextForCausalLM(DecoderHandlerOwner, Qwen3NextModel, RtpModule):
    """Qwen3-Next Dense + Linear Attention (prefix model.)."""

    supports_pp_global_weight_aliases = True
    _expects_mtp = False

    WEIGHTS_MAPPER = WeightsMapper(
        prefix_mapping={
            "model.language_model.": "",
            "model.": "",
        }
    )

    def _is_core_checkpoint_weight(self, name: str) -> bool:
        if self.config.model_type in ("qwen35_dense", "qwen35_moe"):
            return not (name.startswith("mtp.") or is_visual_checkpoint_name(name))
        return not name.startswith(("mtp.", "model.visual."))

    def checkpoint_weight_name_filter(self) -> Optional[Callable[[str], bool]]:
        return lambda name: self._is_core_checkpoint_weight(
            name
        ) and self._owns_checkpoint_name(name)

    def _owns_checkpoint_name(self, name):
        mapped = self.WEIGHTS_MAPPER.map_name(name)
        if mapped.startswith("layers."):
            return self._owns_mapped_layer(mapped)
        if name in getattr(self, "_handler_checkpoint_names", ()):
            return True
        if mapped == "lm_head.weight" and self.lm_head is None:
            return False
        if not self._pipeline_partitioned:
            return True
        if mapped == "embed_tokens.weight":
            return self.embed_tokens is not None or (
                self.lm_head is not None and self.tie_word_embeddings
            )
        if mapped.startswith("norm."):
            return self.norm is not None
        if mapped.startswith("lm_head."):
            return self.lm_head is not None
        return True

    def _owns_mapped_layer(self, mapped):
        parts = mapped.split(".", 2)
        if len(parts) == 3 and parts[0] == "layers" and parts[2]:
            index = parts[1]
            if index.isascii() and index.isdigit() and str(int(index)) == index:
                limit = getattr(
                    self.config, "_checkpoint_num_layers_before_override", self.layer_num
                )
                if int(index) < limit:
                    return int(index) in self._owned_layer_ids
        # An unknown namespace or an invalid layer is not another stage's weight.
        # Let the strict mapper reject it instead of silently skipping it.
        return True

    def owns_lora_checkpoint_target(self, mapped_name):
        return self._owns_mapped_layer(mapped_name)

    def resolve_checkpoint_load_plan(self, name, shape):
        if name in getattr(getattr(self, "handler_heads", None), "sources", ()):
            return None, None
        if (
            name == "embed_tokens.weight"
            and self.embed_tokens is None
            and self.lm_head is not None
            and self.tie_word_embeddings
        ):
            return (
                CheckpointLoadTarget(self.lm_head, "weight", True),
                self.lm_head.tied_checkpoint_read_spec("weight", shape),
            )
        return super().resolve_checkpoint_load_plan(name, shape)

    def allow_checkpoint_read_spec(self, checkpoint_name: str) -> bool:
        # q_proj contains both q and the linear-attention gate in this source
        # checkpoint. ``load_weights`` must split those logical tensors before
        # either consumer can declare its own TP slice.
        return not any(
            checkpoint_name.endswith(f"self_attn.q_proj.{suffix}")
            for suffix in (
                "weight",
                "weight_scale",
                "weight_scale_inv",
                "input_scale",
            )
        )

    def load_weights(self, weights):
        if isinstance(weights, dict):
            weights_iter = iter(weights.items())
        else:
            weights_iter = weights
        has_lm_head = False

        def _transform(it):
            cfg = self._cfg
            for name, tensor in it:
                if self._is_norm_weight(name):
                    tensor = _plus_one(tensor)
                if any(
                    name.endswith(f"self_attn.q_proj.{suffix}")
                    for suffix in (
                        "weight",
                        "weight_scale",
                        "weight_scale_inv",
                        "input_scale",
                    )
                ):
                    yield from self._split_q_gate_yield(name, tensor, cfg)
                    continue
                yield name, tensor

        mapped_iter = self.WEIGHTS_MAPPER.apply(_transform(self._capture_handler_inputs(weights_iter)))

        def _track_mapped_weights(it):
            nonlocal has_lm_head
            for name, tensor in it:
                if name == "embed_tokens.weight" and self.embed_tokens is None:
                    value, spec = unwrap_checkpoint_tensor(tensor)
                    self.lm_head.load_tied_checkpoint_weight(value, spec)
                    continue
                if name == "lm_head.weight":
                    has_lm_head = True
                yield name, tensor

        super().load_weights(_track_mapped_weights(mapped_iter))

        if (
            not has_lm_head
            and self.tie_word_embeddings
            and self.lm_head is not None
            and self.embed_tokens is not None
        ):
            logger.info(
                "[Qwen3NextForCausalLM] lm_head not found; tying to embed_tokens"
            )
            self.lm_head._copy_local_tied_weight(self.embed_tokens.weight.data)

    def process_weights_after_loading(self) -> None:
        if self._lm_head_postprocessed or self.lm_head is None:
            return
        processed = self.lm_head.weight
        if self.normalize_lm_head_weight:
            processed = torch.nn.functional.normalize(processed, dim=1)
        if self.logit_scale != 1.0:
            processed = processed * self.logit_scale
        if processed is not self.lm_head.weight:
            with torch.no_grad():
                self.lm_head.weight.copy_(processed)
        self._lm_head_postprocessed = True

    @staticmethod
    def _is_norm_weight(name: str) -> bool:
        # Linear Attention norm uses 'identity' (NOT plus_one) in the old loader,
        # so we must exclude it from the plus_one transform.
        if "linear_attn" in name:
            return False
        # Standard norm weights: input_layernorm, post_attention_layernorm,
        # q_norm, k_norm, final norm — all use plus_one (gemma_rms_norm).
        return name.endswith(
            (
                "norm.weight",
                "pre_fc_norm_embedding.weight",
                "pre_fc_norm_hidden.weight",
            )
        )

    @staticmethod
    def _split_q_gate_yield(
        name: str, tensor: torch.Tensor, cfg: Dict[str, Any]
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Split q_proj weight/scale into Q (part 0) and Gate (part 1)."""
        head_num = cfg["num_heads"]
        head_dim = cfg["head_dim"]
        suffix = name.rsplit(".", 1)[-1]
        base = name.rsplit(".", 1)[0]
        parent, projection = base.rsplit(".", 1)
        if projection != "q_proj":
            raise ValueError(f"Expected q_proj tensor, got {name!r}")
        if suffix == "input_scale" or (
            suffix == "weight_scale" and tensor.numel() == 1
        ):
            q_part = gate_part = tensor
        else:
            if suffix == "weight":
                expected_rows = 2 * head_num * head_dim
            elif suffix == "weight_scale":
                expected_rows = 2 * head_num * head_dim
            elif suffix == "weight_scale_inv":
                block_n = getattr(cfg["quant_config"], "weight_block_size", [128, 128])[
                    0
                ]
                block_n = _positive_int(block_n, "FP8 output block size")
                if head_dim % block_n:
                    raise ValueError(
                        f"Q-gate head_dim={head_dim} must align to FP8 block {block_n}"
                    )
                expected_rows = 2 * head_num * (head_dim // block_n)
            else:
                raise RuntimeError(f"Unsupported q_gate tensor {name!r}")
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(1)
            if tensor.dim() != 2 or tensor.shape[0] != expected_rows:
                raise ValueError(
                    f"{name} must have {expected_rows} rows, got {tuple(tensor.shape)}"
                )
            q_part = split_q_gate([tensor], head_num=head_num, part=0)
            gate_part = split_q_gate([tensor], head_num=head_num, part=1)
        qkv_name = f"{parent}.qkv_proj.q_proj.{suffix}"
        gate_name = f"{parent}.gate.{suffix}"
        yield qkv_name, q_part
        yield gate_name, gate_part

    def __init__(self, model_config: ModelConfig, load_config: Any):
        cfg = _extract_config_values(
            model_config, load_config, expects_mtp=self._expects_mtp
        )
        parallelism_config = cfg["parallelism_config"]
        fmha_config = getattr(load_config, "fmha_config", None)

        GptModelBase.__init__(
            self,
            config=model_config,
            parallelism_config=parallelism_config,
            weight=None,
            max_generate_batch_size=getattr(load_config, "max_generate_batch_size", 0),
            fmha_config=fmha_config,
            py_hw_kernel_config=getattr(load_config, "py_hw_kernel_config", None),
        )

        self._owned_layer_ids = self.layer_ids()
        self._pipeline_partitioned = (
            int(getattr(parallelism_config, "pp_size", 1)) > 1 and not self._expects_mtp
        )
        owns_embedding = 0 in self._owned_layer_ids
        owns_head = cfg["num_layers"] - 1 in self._owned_layer_ids
        if not self._expects_mtp:
            self._init_decoder_handlers(load_config, cfg["hidden_size"])
        self._cfg = cfg
        self.tie_word_embeddings = cfg["tie_word_embeddings"]
        self.normalize_lm_head_weight = _config_bool(
            model_config, "normalize_lm_head_weight", False
        )
        raw_logit_scale = getattr(model_config, "logit_scale", 1.0)
        if isinstance(raw_logit_scale, bool) or not isinstance(raw_logit_scale, Real):
            raise TypeError("logit_scale must be a finite real number")
        self.logit_scale = float(raw_logit_scale)
        if not math.isfinite(self.logit_scale):
            raise ValueError("logit_scale must be finite")
        self._lm_head_postprocessed = False
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()

        self.embed_tokens = (
            VocabParallelEmbedding(
                vocab_size=cfg["vocab_size"],
                embedding_dim=cfg["hidden_size"],
                tp_size=cfg["attn_tp_size"],
                tp_rank=cfg["attn_tp_rank"],
                params_dtype=cfg["params_dtype"],
                allow_vocab_padding=True,
            )
            if owns_embedding
            else None
        )
        owned = {
            str(i): Qwen3NextDecoderLayer(cfg, i, model_config, prefix=f"layers.{i}")
            for i in self._owned_layer_ids
        }
        self.layers = (
            _StageLayers(owned)
            if self._pipeline_partitioned
            else nn.ModuleList(owned.values())
        )
        self.norm = (
            RMSResNorm(
                cfg["hidden_size"],
                eps=cfg["rms_norm_eps"],
                params_dtype=cfg["params_dtype"],
            )
            if owns_head
            else None
        )
        self.lm_head = (
            ParallelLMHead(
                vocab_size=cfg["vocab_size"],
                hidden_size=cfg["hidden_size"],
                tp_size=cfg["lm_head_tp_size"],
                tp_rank=cfg["lm_head_tp_rank"],
                params_dtype=cfg["lm_head_params_dtype"],
                **output_vocab_head_kwargs(model_config),
            )
            if owns_head and (self._expects_mtp or needs_generation_projection(model_config))
            else None
        )
        if (
            self.tie_word_embeddings
            and self.embed_tokens is not None
            and self.lm_head is not None
        ):
            self.embed_tokens.tie_output_head(self.lm_head)

        self._init_capture_context(
            self._capture_canonical_layer, self._capture_canonical_final
        )

    def load_ft_checkpoint(self, checkpoint):
        from .ft_checkpoint import load_qwen_next_ft

        load_qwen_next_ft(self, checkpoint)

    def runtime_weight_view(self) -> dict[str, torch.Tensor]:
        heads = getattr(self, "handler_heads", None)
        result = heads.exports() if heads is not None else {}
        if self.embed_tokens is not None:
            result["embedding"] = self.embed_tokens.weight
        if self.norm is not None:
            result["final_layernorm.gamma"] = self.norm.weight
        if self.lm_head is not None:
            result["lm_head"] = self.lm_head.weight
        return result

    def word_embedding(self, inputs: PyModelInputs) -> torch.Tensor:
        input_ids = inputs.input_ids
        embedding_inputs = inputs.embedding_inputs
        text_tokens_mask = (
            None if embedding_inputs is None else embedding_inputs.text_tokens_mask
        )
        multimodal_inputs = inputs.multimodal_inputs
        features = (
            []
            if multimodal_inputs is None
            else multimodal_inputs.multimodal_features or []
        )
        locations = (
            None if multimodal_inputs is None else multimodal_inputs.mm_features_locs
        )

        if features and text_tokens_mask is None:
            raise ValueError(
                "Qwen3.5 received multimodal features without text_tokens_mask"
            )
        if features and locations is None:
            raise ValueError(
                "Qwen3.5 received multimodal features without mm_features_locs"
            )

        if text_tokens_mask is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            if text_tokens_mask.numel() != input_ids.numel():
                raise ValueError(
                    "text_tokens_mask must contain one entry per input token"
                )
            text_mask = text_tokens_mask.to(
                device=input_ids.device, dtype=torch.bool
            ).view_as(input_ids)
            safe_input_ids = torch.where(text_mask, input_ids, 0)
            hidden_states = self.embed_tokens(safe_input_ids)
            hidden_states = hidden_states * text_mask.unsqueeze(-1)

        if multimodal_inputs is None:
            return hidden_states
        return self.multimodal_embedding_injector(hidden_states, features, locations)


class Qwen3NextMTPForCausalLM(Qwen3NextForCausalLM):
    """Single-layer Qwen3-Next draft model backed by ``mtp.*`` weights."""

    _expects_mtp = True
    _expected_moe_layer_index = (0,)
    WEIGHTS_MAPPER = WeightsMapper(
        prefix_mapping={
            "model.language_model.": "",
            "model.": "",
            "mtp.": "",
        }
    )
    _EMBEDDING_WEIGHT_NAMES = frozenset(
        {
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
        }
    )

    @classmethod
    def _is_mtp_checkpoint_weight(cls, name: str) -> bool:
        return (
            (name.startswith("mtp.") and len(name) > len("mtp."))
            or name == "lm_head.weight"
            or name in cls._EMBEDDING_WEIGHT_NAMES
        )

    def checkpoint_weight_name_filter(self) -> Callable[[str], bool]:
        return self._is_mtp_checkpoint_weight

    def __init__(self, model_config: ModelConfig, load_config: Any):
        super().__init__(model_config, load_config)
        cfg = self._cfg
        if cfg["num_layers"] != 1:
            raise ValueError(
                f"Qwen3-Next MTP requires exactly one layer, got {cfg['num_layers']}"
            )
        if cfg["hybrid_types"] != [HybridAttentionType.NONE]:
            raise ValueError("Qwen3-Next MTP requires one standard-attention layer")
        if cfg["moe_layer_index"] != list(self._expected_moe_layer_index):
            raise ValueError(
                f"{type(self).__name__} requires MoE layer index "
                f"{list(self._expected_moe_layer_index)}"
            )

        self.pre_fc_norm_embedding = RMSNorm(
            cfg["hidden_size"],
            eps=cfg["rms_norm_eps"],
            params_dtype=cfg["params_dtype"],
        )
        self.pre_fc_norm_hidden = RMSNorm(
            cfg["hidden_size"],
            eps=cfg["rms_norm_eps"],
            params_dtype=cfg["params_dtype"],
        )
        self.fc = ColumnParallelLinear(
            input_size=cfg["hidden_size"] * 2,
            output_size=cfg["hidden_size"],
            tp_size=1,
            tp_rank=0,
            quant_config=None,
            prefix="fc",
            bias=False,
            params_dtype=cfg["params_dtype"],
        )

    forward = Qwen3NextMTPModel.forward
    _get_fmha_group_tags = Qwen3NextMTPModel._get_fmha_group_tags


class Qwen35MoeMTPForCausalLM(Qwen3NextMTPForCausalLM):
    """Qwen3.5 MoE MTP alias with the same runtime layout."""


class Qwen35DenseMTPForCausalLM(Qwen3NextMTPForCausalLM):
    """Qwen3.5 dense draft: full attention and dense MLP, never routed experts."""

    _expected_moe_layer_index = ()

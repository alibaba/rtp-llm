"""Qwen3-Next core model for the streaming newloader."""

import logging
import math
from numbers import Real
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.layers.embedding import ParallelLMHead, VocabParallelEmbedding
from rtp_llm.models_py.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.layers.norm import RMSResNorm
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.module_base import RtpModule, copy_weight_
from rtp_llm.models_py.modules import FakeBalanceExpert, SelectTopk
from rtp_llm.models_py.new_models.model_base import (
    required_config_value,
    select_block_map_for_layer,
)
from rtp_llm.models_py.new_models.qwen3.language import (
    Qwen3Attention,
    _positive_int,
    _validate_supported_parallelism,
)
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.weight_mapper import WeightsMapper
from rtp_llm.ops import HybridAttentionType, LinearAttentionConfig
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.util import to_torch_dtype

logger = logging.getLogger(__name__)


class Qwen3NextMetadata:
    def __init__(self, prefill_conv1d_meta=None, is_target_verify: bool = False):
        self.prefill_conv1d_meta = prefill_conv1d_meta
        self.is_target_verify = is_target_verify

    def get_prefill_conv1d_meta(self):
        return self.prefill_conv1d_meta


def _build_qwen3_next_metadata(
    inputs: PyModelInputs, hidden_states: torch.Tensor
) -> Qwen3NextMetadata:
    attention_inputs = inputs.attention_inputs
    if attention_inputs is None:
        raise ValueError("Qwen3-Next requires attention inputs")
    prefill_conv1d_meta = None
    if attention_inputs.is_prefill and not attention_inputs.is_target_verify:
        from rtp_llm.models_py.triton_kernels.causal_conv1d import (
            prepare_causal_conv1d_metadata,
        )

        prefill_conv1d_meta = prepare_causal_conv1d_metadata(
            query_start_loc=attention_inputs.cu_seqlens_device,
            device=hidden_states.device,
        )
    return Qwen3NextMetadata(
        prefill_conv1d_meta=prefill_conv1d_meta,
        is_target_verify=attention_inputs.is_target_verify,
    )


# ================================================================== #
#  Weight transformation functions (ported from qwen3_next_weight.py)
# ================================================================== #


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


def reorder_ba(ts: List[torch.Tensor], la_cfg: LinearAttentionConfig):
    """Reorder ba weight: [head_num_k, group_v*2, hidden] -> [b_all, a_all, hidden].T"""
    if len(ts) != 1 or ts[0].dim() != 2:
        raise ValueError("in_proj_ba transform expects one two-dimensional tensor")
    t = ts[0]
    hidden_size = t.shape[-1]
    head_num_k = la_cfg.linear_num_key_heads
    head_num_v = la_cfg.linear_num_value_heads
    if head_num_v % head_num_k != 0:
        raise ValueError(
            f"linear value heads {head_num_v} must divide into key heads {head_num_k}"
        )
    group_v = head_num_v // head_num_k
    if t.shape[0] != head_num_v * 2:
        raise ValueError(f"in_proj_ba rows must be {head_num_v * 2}, got {t.shape[0]}")
    t = t.reshape(head_num_k, group_v * 2, t.shape[-1])
    b, a = t.split([group_v, group_v], dim=1)
    return torch.cat([b.reshape(-1, hidden_size), a.reshape(-1, hidden_size)], dim=0)


def reorder_qkvz(ts: List[torch.Tensor], la_cfg: LinearAttentionConfig):
    """Reorder qkvz weight: split q,k,v,z per-head then concatenate."""
    if len(ts) != 1 or ts[0].dim() not in (1, 2):
        raise ValueError("in_proj_qkvz transform expects one row-oriented tensor")
    t = ts[0]
    if t.dim() == 1:
        t = t.unsqueeze(1)
    head_num_k = la_cfg.linear_num_key_heads
    head_num_v = la_cfg.linear_num_value_heads
    head_k_dim = la_cfg.linear_key_head_dim
    head_v_dim = la_cfg.linear_value_head_dim
    if head_num_v % head_num_k != 0:
        raise ValueError(
            f"linear value heads {head_num_v} must divide into key heads {head_num_k}"
        )
    group_v = head_num_v // head_num_k
    dim0, dim1 = t.shape
    qkvz_size = head_num_k * head_k_dim * 2 + head_num_v * head_v_dim * 2
    if dim0 == qkvz_size:
        t = t.reshape(head_num_k, head_k_dim * 2 + head_v_dim * group_v * 2, dim1)
        q, k, v, z = torch.split(
            t,
            [head_k_dim, head_k_dim, head_v_dim * group_v, head_v_dim * group_v],
            dim=1,
        )
        return torch.cat(
            [
                q.reshape(-1, dim1),
                k.reshape(-1, dim1),
                v.reshape(-1, dim1),
                z.reshape(-1, dim1),
            ],
            dim=0,
        )
    raise ValueError(f"in_proj_qkvz rows must be {qkvz_size}, got {dim0}")


def reorder_qkvz_scale(
    tensor: torch.Tensor,
    la_cfg: LinearAttentionConfig,
    block_n: int,
) -> torch.Tensor:
    """Reorder a fused per-block QKVZ scale grid without guessing layout."""
    head_k_dim = la_cfg.linear_key_head_dim
    head_v_dim = la_cfg.linear_value_head_dim
    head_num_k = la_cfg.linear_num_key_heads
    head_num_v = la_cfg.linear_num_value_heads
    if head_num_v % head_num_k != 0:
        raise ValueError("linear value heads must be divisible by key heads")
    if head_k_dim % block_n != 0 or head_v_dim % block_n != 0:
        raise ValueError(
            "Qwen3-Next FP8 QKVZ head dimensions must align to the output "
            f"block size {block_n}: key={head_k_dim}, value={head_v_dim}"
        )
    group_v = head_num_v // head_num_k
    per_group = (head_k_dim * 2 + head_v_dim * group_v * 2) // block_n
    expected_rows = head_num_k * per_group
    if tensor.dim() != 2 or tensor.shape[0] != expected_rows:
        raise ValueError(
            f"QKVZ scale grid must have {expected_rows} rows, got {tuple(tensor.shape)}"
        )
    reshaped = tensor.reshape(head_num_k, per_group, tensor.shape[1])
    q_rows = head_k_dim // block_n
    value_rows = head_v_dim * group_v // block_n
    q, k, v, z = torch.split(reshaped, [q_rows, q_rows, value_rows, value_rows], dim=1)
    return torch.cat(
        [
            q.reshape(-1, tensor.shape[1]),
            k.reshape(-1, tensor.shape[1]),
            v.reshape(-1, tensor.shape[1]),
            z.reshape(-1, tensor.shape[1]),
        ],
        dim=0,
    ).contiguous()


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


def _finite_positive_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite positive real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return result


def _validate_partition(size: Any, rank: Any, name: str) -> Tuple[int, int]:
    size = _positive_int(size, f"{name} size")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < size:
        raise ValueError(f"Invalid {name} partition: rank={rank}, size={size}")
    return size, rank


def _extract_config_values(
    model_config: ModelConfig, load_config: Any
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
    inter_size = _positive_int(
        required_config_value(model_config, "inter_size"), "inter_size"
    )

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

    expert_num = _positive_int(
        required_config_value(model_config, "expert_num"), "expert_num"
    )
    moe_k = _positive_int(required_config_value(model_config, "moe_k"), "moe_k")
    if moe_k > expert_num:
        raise ValueError(f"moe_k={moe_k} cannot exceed expert_num={expert_num}")
    moe_inter_size = _positive_int(
        required_config_value(model_config, "moe_inter_size"), "moe_inter_size"
    )
    if required_config_value(model_config, "moe_style") != 2:
        raise ValueError("Qwen3-Next newloader requires shared+routed MoE style 2")
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
    has_moe_norm = _config_bool(model_config, "has_moe_norm", True)
    tie_word_embeddings = _config_bool(model_config, "tie_word_embeddings", False)
    if _config_bool(model_config, "is_mtp", False):
        raise ValueError("MTP is not part of the Qwen3-Next core newloader slice")

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
    if (attn_tp_size, attn_tp_rank) != (tp_size, tp_rank):
        raise ValueError(
            "Context parallelism is not supported by the Qwen3-Next core slice"
        )
    if (ffn_tp_size, ffn_tp_rank) != (tp_size, tp_rank):
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

    parallelism_config = required_config_value(load_config, "parallelism_config")
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
    if enable_eplb:
        raise ValueError("EPLB is not supported by the Qwen3-Next core slice")

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


class Qwen3NextGatedDeltaNet(RtpModule):
    """Linear Attention (Gated Delta Network) for Qwen3-Next.

    Weight loading handles:
      - reorder_qkvz: fused per-head Q/K/V/Z rows
      - reorder_ba: fused per-head B/A rows
      - transpose: linear-attention projections into their runtime layouts
    Forward delegates to the same Triton kernels used by the old model_desc.
    """

    def __init__(
        self,
        linear_attn_config: LinearAttentionConfig,
        hidden_size: int,
        rms_norm_eps: float,
        attn_tp_size: int,
        attn_tp_rank: int,
        params_dtype: torch.dtype,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        self.linear_attn_config = linear_attn_config
        self.hidden_size = hidden_size
        self.rms_norm_eps = _finite_positive_real(rms_norm_eps, "layernorm_eps")
        self.attn_tp_size = attn_tp_size
        self.attn_tp_rank = attn_tp_rank
        self.tp_size = attn_tp_size
        self.head_k_dim = linear_attn_config.linear_key_head_dim
        self.head_v_dim = linear_attn_config.linear_value_head_dim
        if self.head_k_dim != self.head_v_dim:
            raise ValueError(
                "Qwen3-Next linear attention requires equal key/value head "
                f"dimensions, got {self.head_k_dim}/{self.head_v_dim}"
            )
        num_k_heads = linear_attn_config.linear_num_key_heads
        num_v_heads = linear_attn_config.linear_num_value_heads
        if num_k_heads % attn_tp_size or num_v_heads % attn_tp_size:
            raise ValueError(
                "Linear attention head counts must be divisible by attention TP"
            )
        self.local_num_k_heads = num_k_heads // attn_tp_size
        self.local_num_v_heads = num_v_heads // attn_tp_size
        if self.local_num_v_heads % self.local_num_k_heads:
            raise ValueError("Local value heads must be divisible by local key heads")
        self.num_key_value_heads = self.local_num_v_heads // self.local_num_k_heads
        self.linear_conv_kernel_dim = linear_attn_config.linear_conv_kernel_dim
        self.qkvz_size = (
            self.head_k_dim * self.local_num_k_heads * 2
            + self.head_v_dim * self.local_num_v_heads * 2
        )
        self.qkv_size = (
            self.head_k_dim * self.local_num_k_heads * 2
            + self.head_v_dim * self.local_num_v_heads
        )
        self.ba_size = self.local_num_v_heads * 2

        # These projections are already rank-local after the model-specific
        # Q/K/V/Z transform. The common Linear layer still owns quantization,
        # completeness validation, backend preflight and the post-load hook.
        self.in_proj_qkvz = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=self.qkvz_size,
            tp_size=1,
            tp_rank=0,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_qkvz",
            bias=False,
            params_dtype=params_dtype,
        )
        self.in_proj_ba_w = nn.Parameter(
            torch.empty(hidden_size, self.ba_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.conv1d_w = nn.Parameter(
            torch.empty(self.qkv_size, self.linear_conv_kernel_dim, dtype=params_dtype),
            requires_grad=False,
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.local_num_v_heads, dtype=params_dtype),
            requires_grad=False,
        )
        self.a_log = nn.Parameter(
            torch.empty(self.local_num_v_heads, dtype=params_dtype),
            requires_grad=False,
        )
        self.norm_w = nn.Parameter(
            torch.empty(self.head_v_dim, dtype=params_dtype), requires_grad=False
        )
        from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated

        self.norm_gated = RmsNormGated(
            self.norm_w,
            eps=self.rms_norm_eps,
            group_size=self.head_v_dim,
        )
        self.out_proj = ColumnParallelLinear(
            input_size=self.local_num_v_heads * self.head_v_dim,
            output_size=hidden_size,
            tp_size=1,
            tp_rank=0,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
            bias=False,
            params_dtype=params_dtype,
        )
        self.ssm_state_dtype = to_torch_dtype(linear_attn_config.ssm_state_dtype)
        self.conv_state_dtype = to_torch_dtype(linear_attn_config.conv_state_dtype)
        self._qkvz_layouts: Dict[str, str] = {}
        self._qkvz_split_parts: Dict[str, Set[str]] = {}
        self._ba_layout: Optional[str] = None
        self._ba_split_parts: Set[str] = set()

    def validate_runtime_device(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise RuntimeError(
                "Qwen3-Next linear attention requires a CUDA or ROCm accelerator"
            )

    @staticmethod
    def _block_size(layer: ColumnParallelLinear) -> Tuple[int, int]:
        value = getattr(layer.quant_config, "weight_block_size", [128, 128])
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"Invalid FP8 weight_block_size {value!r}")
        block_n = _positive_int(value[0], "FP8 output block size")
        block_k = _positive_int(value[1], "FP8 input block size")
        return block_n, block_k

    def _split_qkvz_rows(self, tensor: torch.Tensor, unit: int) -> torch.Tensor:
        global_k = self.head_k_dim * self.linear_attn_config.linear_num_key_heads
        global_v = self.head_v_dim * self.linear_attn_config.linear_num_value_heads
        if global_k % unit or global_v % unit:
            raise ValueError(
                f"QKVZ group sizes {global_k}/{global_v} must align to unit {unit}"
            )
        q_rows = global_k // unit
        v_rows = global_v // unit
        expected_rows = q_rows * 2 + v_rows * 2
        if tensor.dim() < 1 or tensor.shape[0] != expected_rows:
            raise ValueError(
                f"QKVZ tensor must have {expected_rows} rows, got {tuple(tensor.shape)}"
            )
        q, k, v, z = torch.split(tensor, [q_rows, q_rows, v_rows, v_rows], dim=0)
        if q_rows % self.attn_tp_size or v_rows % self.attn_tp_size:
            raise ValueError("QKVZ groups must be divisible by attention TP")
        local_q = q_rows // self.attn_tp_size
        local_v = v_rows // self.attn_tp_size
        q_start = self.attn_tp_rank * local_q
        v_start = self.attn_tp_rank * local_v
        return torch.cat(
            [
                q.narrow(0, q_start, local_q),
                k.narrow(0, q_start, local_q),
                v.narrow(0, v_start, local_v),
                z.narrow(0, v_start, local_v),
            ],
            dim=0,
        ).contiguous()

    def _split_out_weight(self, tensor: torch.Tensor) -> torch.Tensor:
        global_input = self.head_v_dim * self.linear_attn_config.linear_num_value_heads
        expected = (self.hidden_size, global_input)
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"out_proj.weight must have shape {expected}, got {tuple(tensor.shape)}"
            )
        local_input = global_input // self.attn_tp_size
        start = self.attn_tp_rank * local_input
        return tensor.narrow(1, start, local_input).contiguous()

    def _split_out_block_scale(self, tensor: torch.Tensor) -> torch.Tensor:
        block_n, block_k = self._block_size(self.out_proj)
        global_input = self.head_v_dim * self.linear_attn_config.linear_num_value_heads
        local_input = global_input // self.attn_tp_size
        start = self.attn_tp_rank * local_input
        expected = (
            math.ceil(self.hidden_size / block_n),
            math.ceil(global_input / block_k),
        )
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"out_proj.weight_scale_inv must have shape {expected}, "
                f"got {tuple(tensor.shape)}"
            )
        if self.attn_tp_size == 1:
            return tensor.contiguous()
        if start % block_k or local_input % block_k:
            raise ValueError(
                "out_proj TP input shard must align to FP8 input blocks: "
                f"start={start}, width={local_input}, block={block_k}"
            )
        return tensor.narrow(1, start // block_k, local_input // block_k).contiguous()

    def _split_ba(self, tensor: torch.Tensor) -> torch.Tensor:
        reordered = reorder_ba([tensor], self.linear_attn_config)
        global_heads = self.linear_attn_config.linear_num_value_heads
        b, a = torch.split(reordered, [global_heads, global_heads], dim=0)
        local = self.local_num_v_heads
        start = self.attn_tp_rank * local
        return (
            torch.cat([b.narrow(0, start, local), a.narrow(0, start, local)], dim=0)
            .t()
            .contiguous()
        )

    @staticmethod
    def _claim_layout(layouts: Dict[str, str], parameter: str, layout: str) -> None:
        current = layouts.get(parameter)
        if current is not None and current != layout:
            raise RuntimeError(
                f"Cannot mix {current} and {layout} checkpoint layouts for {parameter}"
            )
        layouts[parameter] = layout

    def _split_qkv_or_z_rows(
        self, tensor: torch.Tensor, component: str, unit: int
    ) -> Tuple[torch.Tensor, int]:
        global_k = self.head_k_dim * self.linear_attn_config.linear_num_key_heads
        global_v = self.head_v_dim * self.linear_attn_config.linear_num_value_heads
        if global_k % unit or global_v % unit:
            raise ValueError(
                f"QKV/Z group sizes {global_k}/{global_v} must align to unit {unit}"
            )
        q_rows = global_k // unit
        v_rows = global_v // unit
        if component == "qkv":
            expected_rows = q_rows * 2 + v_rows
        elif component == "z":
            expected_rows = v_rows
        else:
            raise ValueError(f"Unknown QKV/Z component {component!r}")
        if tensor.dim() < 1 or tensor.shape[0] != expected_rows:
            raise ValueError(
                f"in_proj_{component} tensor must have {expected_rows} rows, "
                f"got {tuple(tensor.shape)}"
            )
        if q_rows % self.attn_tp_size or v_rows % self.attn_tp_size:
            raise ValueError("QKV/Z groups must be divisible by attention TP")
        local_q = q_rows // self.attn_tp_size
        local_v = v_rows // self.attn_tp_size
        q_start = self.attn_tp_rank * local_q
        v_start = self.attn_tp_rank * local_v
        if component == "qkv":
            q, k, v = torch.split(tensor, [q_rows, q_rows, v_rows], dim=0)
            local_tensor = torch.cat(
                [
                    q.narrow(0, q_start, local_q),
                    k.narrow(0, q_start, local_q),
                    v.narrow(0, v_start, local_v),
                ],
                dim=0,
            )
            offset = 0
        else:
            local_tensor = tensor.narrow(0, v_start, local_v)
            if self.qkv_size % unit:
                raise ValueError(
                    f"Local QKV size {self.qkv_size} must align to unit {unit}"
                )
            offset = self.qkv_size // unit
        return local_tensor.contiguous(), offset

    def _load_split_qkvz(
        self, component: str, parameter: str, tensor: torch.Tensor
    ) -> None:
        if parameter == "input_scale":
            raise ValueError(
                "Split QKV/Z input_scale cannot be represented by one fused runtime "
                "scale; use a fused checkpoint or dynamic activation quantization"
            )
        if parameter == "weight_scale" and tensor.numel() == 1:
            raise ValueError(
                "Split QKV/Z scalar weight scales cannot be fused without "
                "dequantization and requantization"
            )
        target = getattr(self.in_proj_qkvz, parameter, None)
        if not isinstance(target, nn.Parameter):
            raise RuntimeError(
                f"Unsupported split QKV/Z tensor {self.in_proj_qkvz.prefix}.{parameter}"
            )
        self._claim_layout(self._qkvz_layouts, parameter, "split")
        loaded_parts = self._qkvz_split_parts.setdefault(parameter, set())
        if component in loaded_parts:
            raise RuntimeError(f"Duplicate split QKV/Z tensor {component}.{parameter}")

        unit = 1
        if parameter == "weight":
            if tensor.dim() != 2 or tensor.shape[1] != self.hidden_size:
                raise ValueError(
                    f"in_proj_{component}.weight must have input width "
                    f"{self.hidden_size}, got {tuple(tensor.shape)}"
                )
        elif parameter == "weight_scale_inv":
            block_n, block_k = self._block_size(self.in_proj_qkvz)
            unit = block_n
            expected_columns = math.ceil(self.hidden_size / block_k)
            if tensor.dim() != 2 or tensor.shape[1] != expected_columns:
                raise ValueError(
                    f"in_proj_{component}.weight_scale_inv must have "
                    f"{expected_columns} input blocks, got {tuple(tensor.shape)}"
                )
        elif parameter == "weight_scale":
            if tensor.dim() not in (1, 2):
                raise ValueError(
                    f"in_proj_{component}.weight_scale must be row-oriented, "
                    f"got {tuple(tensor.shape)}"
                )
        else:
            raise RuntimeError(f"Unsupported split QKV/Z parameter {parameter!r}")

        local, offset = self._split_qkv_or_z_rows(tensor, component, unit)
        target_slice = target.data.narrow(0, offset, local.shape[0])
        if local.shape != target_slice.shape and local.numel() == target_slice.numel():
            local = local.reshape(target_slice.shape)
        copy_weight_(
            target_slice,
            local,
            f"{self.in_proj_qkvz.prefix}.{component}.{parameter}",
        )
        loaded_parts.add(component)
        if loaded_parts == {"qkv", "z"}:
            self.in_proj_qkvz._record_parameter_loaded(parameter)

    def _load_split_ba(self, component: str, tensor: torch.Tensor) -> None:
        if self._ba_layout is not None and self._ba_layout != "split":
            raise RuntimeError("Cannot mix fused and split checkpoint layouts for B/A")
        self._ba_layout = "split"
        if component in self._ba_split_parts:
            raise RuntimeError(f"Duplicate split B/A tensor in_proj_{component}.weight")
        global_heads = self.linear_attn_config.linear_num_value_heads
        expected = (global_heads, self.hidden_size)
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"in_proj_{component}.weight must have shape {expected}, "
                f"got {tuple(tensor.shape)}"
            )
        start = self.attn_tp_rank * self.local_num_v_heads
        local = tensor.narrow(0, start, self.local_num_v_heads).t().contiguous()
        offset = 0 if component == "b" else self.local_num_v_heads
        copy_weight_(
            self.in_proj_ba_w.data.narrow(1, offset, self.local_num_v_heads),
            local,
            f"in_proj_{component}.weight",
        )
        self._ba_split_parts.add(component)
        if self._ba_split_parts == {"a", "b"}:
            self._mark_weight_loaded("in_proj_ba_w")

    def _split_head_tensor(self, tensor: torch.Tensor, label: str) -> torch.Tensor:
        expected = self.linear_attn_config.linear_num_value_heads
        if tensor.dim() != 1 or tensor.shape[0] != expected:
            raise ValueError(
                f"{label} must have shape {(expected,)}, got {tensor.shape}"
            )
        start = self.attn_tp_rank * self.local_num_v_heads
        return tensor.narrow(0, start, self.local_num_v_heads).contiguous()

    def _split_conv1d(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 3:
            if tensor.shape[1] != 1:
                raise ValueError(
                    f"conv1d.weight middle dimension must be 1, got {tensor.shape}"
                )
            tensor = tensor.squeeze(1)
        global_k = self.head_k_dim * self.linear_attn_config.linear_num_key_heads
        global_v = self.head_v_dim * self.linear_attn_config.linear_num_value_heads
        expected = (global_k * 2 + global_v, self.linear_conv_kernel_dim)
        if tuple(tensor.shape) == expected[::-1]:
            tensor = tensor.t().contiguous()
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"conv1d.weight must have shape {expected}, got {tuple(tensor.shape)}"
            )
        q, k, v = torch.split(tensor, [global_k, global_k, global_v], dim=0)
        local_k = global_k // self.attn_tp_size
        local_v = global_v // self.attn_tp_size
        k_start = self.attn_tp_rank * local_k
        v_start = self.attn_tp_rank * local_v
        return torch.cat(
            [
                q.narrow(0, k_start, local_k),
                k.narrow(0, k_start, local_k),
                v.narrow(0, v_start, local_v),
            ],
            dim=0,
        ).contiguous()

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        for name, tensor in weights.items():
            if name == "in_proj_qkvz.weight":
                self._claim_layout(self._qkvz_layouts, "weight", "fused")
                reordered = reorder_qkvz([tensor], self.linear_attn_config)
                self.in_proj_qkvz.load_weights(
                    {"weight": self._split_qkvz_rows(reordered, 1)}
                )
            elif name == "in_proj_qkvz.weight_scale_inv":
                self._claim_layout(self._qkvz_layouts, "weight_scale_inv", "fused")
                block_n, _ = self._block_size(self.in_proj_qkvz)
                reordered = reorder_qkvz_scale(tensor, self.linear_attn_config, block_n)
                self.in_proj_qkvz.load_weights(
                    {"weight_scale_inv": self._split_qkvz_rows(reordered, block_n)}
                )
            elif name == "in_proj_qkvz.weight_scale":
                self._claim_layout(self._qkvz_layouts, "weight_scale", "fused")
                if tensor.numel() == 1:
                    local = tensor
                else:
                    local = self._split_qkvz_rows(
                        reorder_qkvz([tensor], self.linear_attn_config), 1
                    )
                self.in_proj_qkvz.load_weights({"weight_scale": local})
            elif name == "in_proj_qkvz.input_scale":
                self._claim_layout(self._qkvz_layouts, "input_scale", "fused")
                self.in_proj_qkvz.load_weights({"input_scale": tensor})
            elif name.startswith("in_proj_qkv."):
                self._load_split_qkvz("qkv", name.removeprefix("in_proj_qkv."), tensor)
            elif name.startswith("in_proj_z."):
                self._load_split_qkvz("z", name.removeprefix("in_proj_z."), tensor)
            elif name == "in_proj_ba.weight":
                if self._ba_layout is not None and self._ba_layout != "fused":
                    raise RuntimeError(
                        "Cannot mix split and fused checkpoint layouts for B/A"
                    )
                self._ba_layout = "fused"
                if not self._assign_weight(
                    self, "in_proj_ba_w", self._split_ba(tensor)
                ):
                    raise RuntimeError("Failed to load in_proj_ba.weight")
            elif name == "in_proj_b.weight":
                self._load_split_ba("b", tensor)
            elif name == "in_proj_a.weight":
                self._load_split_ba("a", tensor)
            elif name == "conv1d.weight":
                if not self._assign_weight(
                    self, "conv1d_w", self._split_conv1d(tensor)
                ):
                    raise RuntimeError("Failed to load conv1d.weight")
            elif name == "dt_bias":
                if not self._assign_weight(
                    self, "dt_bias", self._split_head_tensor(tensor, name)
                ):
                    raise RuntimeError("Failed to load dt_bias")
            elif name == "A_log":
                if not self._assign_weight(
                    self, "a_log", self._split_head_tensor(tensor, name)
                ):
                    raise RuntimeError("Failed to load A_log")
            elif name == "norm.weight":
                if tuple(tensor.shape) != tuple(self.norm_w.shape):
                    raise ValueError(
                        f"linear_attn.norm.weight must have shape "
                        f"{tuple(self.norm_w.shape)}, got {tuple(tensor.shape)}"
                    )
                if not self._assign_weight(self, "norm_w", tensor.contiguous()):
                    raise RuntimeError("Failed to load linear attention norm")
            elif name == "out_proj.weight":
                self.out_proj.load_weights({"weight": self._split_out_weight(tensor)})
            elif name == "out_proj.weight_scale_inv":
                self.out_proj.load_weights(
                    {"weight_scale_inv": self._split_out_block_scale(tensor)}
                )
            elif name in ("out_proj.weight_scale", "out_proj.input_scale"):
                self.out_proj.load_weights({name.rsplit(".", 1)[-1]: tensor})
            else:
                raise RuntimeError(f"Unsupported Qwen3-Next linear tensor {name!r}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        attn_meta: Optional[Qwen3NextMetadata] = None,
    ) -> torch.Tensor:
        """Run Gated Delta Network linear attention.

        Delegates to the existing Triton kernels from triton_kernels/fla/.
        """
        from rtp_llm.models_py.triton_kernels.causal_conv1d import (
            causal_conv1d_fn,
            causal_conv1d_update,
            prepare_causal_conv1d_metadata,
        )
        from rtp_llm.models_py.triton_kernels.common.scatter_qkv import scatter_qkv
        from rtp_llm.models_py.triton_kernels.fla.chunk import (
            chunk_gated_delta_rule,
            chunk_gated_delta_rule_flydsl_with_cache_store,
            is_flydsl_chunk_gdn_enabled,
            is_flydsl_chunk_gdn_shape_supported,
        )
        from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule,
        )
        from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating
        from rtp_llm.models_py.utils.typed_storage_view import LinearCacheConverter

        if attention_inputs is None:
            raise ValueError("Qwen3-Next linear attention requires attention inputs")

        projected_qkvz = self.in_proj_qkvz(hidden_states)
        projected_ba = torch.nn.functional.linear(hidden_states, self.in_proj_ba_w.T)

        # Split qkvz into qkv and z; split ba into b and a
        split_qkvz = [
            self.head_k_dim * self.local_num_k_heads * 2
            + self.head_v_dim * self.local_num_v_heads,
            self.head_v_dim * self.local_num_v_heads,
        ]
        mixed_qkv, z = torch.split(projected_qkvz, split_qkvz, dim=1)
        b, a = torch.split(
            projected_ba, [self.local_num_v_heads, self.local_num_v_heads], dim=1
        )

        # Determine prefill vs decode
        is_prefill = (
            attention_inputs.is_prefill and not attention_inputs.is_target_verify
        )

        kv_cache_tensor = None
        seq_size_per_block = 1
        if kv_cache is not None:
            kv_cache_tensor = kv_cache.kv_cache_base.reshape(
                kv_cache.kv_cache_base.shape[0], -1
            )
            seq_size_per_block = kv_cache.seq_size_per_block

        # Setup cache converter
        ssm_state_dtype = self.ssm_state_dtype
        linear_cache_converter = LinearCacheConverter(
            local_num_v_heads=self.local_num_v_heads,
            head_v_dim=self.head_v_dim,
            head_k_dim=self.head_k_dim,
            ssm_state_dtype=ssm_state_dtype,
            linear_conv_kernel_dim=self.linear_conv_kernel_dim,
            qkv_size=self.qkv_size,
            conv_state_dtype=self.conv_state_dtype,
        )

        if is_prefill:
            attn_out = self._forward_prefill(
                mixed_qkv,
                b,
                a,
                attention_inputs,
                kv_cache,
                kv_cache_tensor,
                seq_size_per_block,
                causal_conv1d_fn,
                prepare_causal_conv1d_metadata,
                chunk_gated_delta_rule,
                chunk_gated_delta_rule_flydsl_with_cache_store,
                is_flydsl_chunk_gdn_enabled,
                is_flydsl_chunk_gdn_shape_supported,
                fused_gdn_gating,
                scatter_qkv,
                linear_cache_converter,
                attn_meta,
            )
        else:
            attn_out = self._forward_decode(
                mixed_qkv,
                b,
                a,
                attention_inputs,
                kv_cache,
                kv_cache_tensor,
                seq_size_per_block,
                causal_conv1d_update,
                fused_recurrent_gated_delta_rule,
                fused_gdn_gating,
                linear_cache_converter,
            )

        attn_output = self.norm_gated(
            attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
        )
        attn_output = attn_output.reshape(-1, self.local_num_v_heads * self.head_v_dim)

        attn_output = self.out_proj(attn_output)

        if self.tp_size > 1:
            attn_output = all_reduce(attn_output, group=Group.TP)
        return attn_output

    def _forward_prefill(
        self,
        mixed_qkv,
        b,
        a,
        attn_inputs,
        kv_cache,
        kv_cache_tensor,
        seq_size_per_block,
        causal_conv1d_fn,
        prepare_meta,
        chunk_gdn,
        chunk_gdn_flydsl,
        is_flydsl_enabled,
        is_flydsl_supported,
        fused_gdn_gating,
        scatter_qkv,
        linear_cache_converter,
        attn_meta: Optional[Qwen3NextMetadata],
    ):
        from rtp_llm.models_py.triton_kernels.fla.block import (
            load_initial_state_from_block_map,
            store_ssm_state_to_block_map,
        )
        from rtp_llm.ops.compute_ops import rtp_llm_ops as compute_ops

        cu_seqlens = attn_inputs.cu_seqlens_device
        conv_states = (
            linear_cache_converter.get_conv_state_tensor(kv_cache_tensor).transpose(
                1, 2
            )
            if kv_cache_tensor is not None
            else None
        )
        conv1d_meta = (
            attn_meta.get_prefill_conv1d_meta() if attn_meta is not None else None
        )
        if conv1d_meta is None and cu_seqlens is not None:
            conv1d_meta = prepare_meta(
                query_start_loc=cu_seqlens, device=mixed_qkv.device
            )

        mixed_qkv = causal_conv1d_fn(
            x=mixed_qkv.transpose(0, 1),
            weight=self.conv1d_w.data,
            bias=None,
            conv_states=conv_states,
            query_start_loc=cu_seqlens,
            block_map=attn_inputs.kv_cache_kernel_block_id_device,
            seq_size_per_block=seq_size_per_block,
            prefix_lengths=attn_inputs.prefix_lengths_device,
            metadata=conv1d_meta,
        ).transpose(0, 1)

        g, beta = fused_gdn_gating(self.a_log, a, b, self.dt_bias)
        ssm_states = (
            linear_cache_converter.get_ssm_state_tensor(kv_cache_tensor)
            if kv_cache_tensor is not None
            else None
        )
        context_batch_size = attn_inputs.input_lengths.shape[0]
        initial_states = None
        if ssm_states is not None:
            initial_states = torch.empty(
                context_batch_size,
                self.local_num_v_heads,
                self.head_v_dim,
                self.head_k_dim,
                device=mixed_qkv.device,
                dtype=self.ssm_state_dtype,
            )
            load_initial_state_from_block_map(
                attn_inputs.prefix_lengths_device,
                attn_inputs.kv_cache_kernel_block_id_device,
                ssm_states,
                initial_states,
                seq_size_per_block,
            )

        if mixed_qkv.shape[0] >= 2048 and self.head_k_dim == self.head_v_dim:
            query, key, value = scatter_qkv(
                mixed_qkv,
                self.local_num_k_heads,
                self.local_num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
            )
        else:
            query, key, value = torch.split(
                mixed_qkv,
                [
                    self.local_num_k_heads * self.head_k_dim,
                    self.local_num_k_heads * self.head_k_dim,
                    self.local_num_v_heads * self.head_v_dim,
                ],
                dim=-1,
            )
            query = query.view(
                1, query.shape[0], self.local_num_k_heads, self.head_k_dim
            )
            key = key.view(1, key.shape[0], self.local_num_k_heads, self.head_k_dim)
            value = value.view(
                1, value.shape[0], self.local_num_v_heads, self.head_v_dim
            )

        use_flydsl = is_flydsl_enabled() and is_flydsl_supported(
            query, key, value, beta
        )
        if use_flydsl:
            need_final = ssm_states is None
            attn_out, _ = chunk_gdn_flydsl(
                query,
                key,
                value,
                g,
                beta,
                prefix_lengths=(
                    attn_inputs.prefix_lengths_device
                    if ssm_states is not None
                    else None
                ),
                block_map=(
                    attn_inputs.kv_cache_kernel_block_id_device
                    if ssm_states is not None
                    else None
                ),
                ssm_states=ssm_states,
                seq_size_per_block=(
                    seq_size_per_block if ssm_states is not None else None
                ),
                initial_state=initial_states,
                output_final_state=need_final,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            attn_out, h, final_state = chunk_gdn(
                query,
                key,
                value,
                g,
                beta,
                initial_state=initial_states,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
            if ssm_states is not None:
                store_ssm_state_to_block_map(
                    h,
                    final_state,
                    attn_inputs.prefix_lengths_device,
                    cu_seqlens,
                    attn_inputs.kv_cache_kernel_block_id_device,
                    ssm_states,
                    seq_size_per_block,
                    chunk_size=64,
                )

        if kv_cache is not None:
            compute_ops.write_cache_store(
                attn_inputs.input_lengths,
                attn_inputs.prefix_lengths,
                attn_inputs.kv_cache_block_id,
                attn_inputs.cache_store_inputs,
                kv_cache,
            )
        return attn_out.squeeze_(0)

    def _forward_decode(
        self,
        mixed_qkv,
        b,
        a,
        attn_inputs,
        kv_cache,
        kv_cache_tensor,
        seq_size_per_block,
        causal_conv1d_update,
        fused_recurrent_gated_delta_rule,
        fused_gdn_gating,
        linear_cache_converter,
    ):
        is_target_verify = attn_inputs.is_target_verify
        token, _ = mixed_qkv.shape
        if not is_target_verify:
            batch, seq = token, 1
        else:
            if attn_inputs.prefix_lengths.size(0) <= 0:
                raise ValueError("Target verification requires a non-empty batch")
            batch = attn_inputs.prefix_lengths.size(0)
            if token % batch:
                raise ValueError(
                    f"Target verification token count {token} is not divisible "
                    f"by batch size {batch}"
                )
            seq = token // batch

        if kv_cache_tensor is None:
            raise ValueError("Qwen3-Next decode requires an allocated linear KV cache")

        origin_shape = mixed_qkv.shape
        mixed_qkv = mixed_qkv.reshape(batch, seq, -1).transpose(1, 2)
        conv_states = linear_cache_converter.get_conv_state_tensor(kv_cache_tensor)
        mixed_qkv = (
            causal_conv1d_update(
                mixed_qkv,
                conv_states.transpose(1, 2),
                self.conv1d_w.data,
                bias=None,
                activation="silu",
                cache_seqlens=None,
                block_map=attn_inputs.kv_cache_kernel_block_id_device,
                seq_size_per_block=seq_size_per_block,
                sequence_lengths=attn_inputs.sequence_lengths_plus_1_device,
            )
            .transpose(1, 2)
            .reshape(origin_shape)
        )

        mixed_qkv = mixed_qkv.reshape(
            batch,
            seq,
            self.local_num_k_heads * 2 + self.local_num_v_heads,
            self.head_k_dim,
        )
        query, key, value = torch.split(
            mixed_qkv,
            [self.local_num_k_heads, self.local_num_k_heads, self.local_num_v_heads],
            dim=2,
        )

        g, beta = fused_gdn_gating(self.a_log, a, b, self.dt_bias)
        g = g.view(batch, seq, self.local_num_v_heads)
        beta = beta.view(batch, seq, self.local_num_v_heads)
        ssm_states = linear_cache_converter.get_ssm_state_tensor(kv_cache_tensor)
        core_attn_out, _ = fused_recurrent_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            scale=None,
            initial_state=ssm_states,
            inplace_final_state=True,
            block_map=attn_inputs.kv_cache_kernel_block_id_device,
            seq_size_per_block=seq_size_per_block,
            sequence_lengths=attn_inputs.sequence_lengths_plus_1_device,
            use_qk_l2norm_in_kernel=True,
        )
        res = core_attn_out.reshape(-1, core_attn_out.shape[2], core_attn_out.shape[3])
        return res


# ================================================================== #
#  Standard Attention with Gate
# ================================================================== #


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
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
            bias=False,
            params_dtype=params_dtype,
        )
        self.select_topk = SelectTopk(config=model_config)
        fake_balance = getattr(moe_config, "fake_balance_expert", False)
        if not isinstance(fake_balance, bool):
            raise TypeError("moe_config.fake_balance_expert must be a bool")
        self.fake_balance_expert = (
            FakeBalanceExpert(
                expert_num=expert_num,
                moe_k=top_k,
                dp_rank=parallelism_config.dp_rank,
                dp_size=parallelism_config.dp_size,
                ep_size=ep_size,
            )
            if fake_balance
            else None
        )

        # Routed experts (reuse Qwen3Experts)
        from rtp_llm.models_py.new_models.qwen3_moe.language import Qwen3Experts

        self.experts = Qwen3Experts(
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
        self.shared_expert = Qwen3NextSharedExpert(
            hidden_size,
            shared_inter_size,
            tp_size,
            tp_rank,
            params_dtype,
            quant_config,
            prefix=f"{prefix}.shared_expert",
        )

        # Shared expert gate
        if has_shared_expert_gate:
            self.shared_expert_gate = ColumnParallelLinear(
                input_size=hidden_size,
                output_size=1,
                tp_size=1,
                tp_rank=0,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_expert_gate",
                bias=False,
                params_dtype=params_dtype,
            )
        else:
            self.shared_expert_gate = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
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
            dtype=self.experts.fused_moe.topk_ids_dtype,
            device=hidden_states.device,
        )
        self.select_topk(router_logits, topk_ids, topk_weights)
        if self.fake_balance_expert is not None:
            self.fake_balance_expert(topk_ids, topk_weights)

        experts_output = self.experts(hidden_states, topk_weights, topk_ids)

        use_ep_shared_allreduce = self.tp_size > 1 and self.ep_size > 1
        shared_output = self.shared_expert(
            hidden_states, skip_allreduce=use_ep_shared_allreduce
        )
        if self.shared_expert_gate is not None:
            gate_output = self.shared_expert_gate(hidden_states)
            shared_contribution = torch.sigmoid(gate_output) * shared_output
            if use_ep_shared_allreduce:
                shared_contribution = all_reduce(shared_contribution, group=Group.TP)
            experts_output = experts_output + shared_contribution
        else:
            if use_ep_shared_allreduce:
                shared_output = all_reduce(shared_output, group=Group.TP)
            experts_output = experts_output + shared_output

        return experts_output


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
                shared_inter_size=cfg["inter_size"],
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


class Qwen3NextForCausalLM(GptModelBase):
    """Qwen3-Next Dense + Linear Attention (prefix model.)."""

    WEIGHTS_MAPPER = WeightsMapper(
        prefix_mapping={
            "model.language_model.": "",
            "model.": "",
        }
    )

    @staticmethod
    def _is_core_checkpoint_weight(name: str) -> bool:
        return not (name.startswith("mtp.") or name.startswith("model.visual."))

    def checkpoint_weight_name_filter(self) -> Optional[Callable[[str], bool]]:
        return self._is_core_checkpoint_weight

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

        mapped_iter = self.WEIGHTS_MAPPER.apply(_transform(weights_iter))

        def _track_mapped_weights(it):
            nonlocal has_lm_head
            for name, tensor in it:
                if name == "lm_head.weight":
                    has_lm_head = True
                yield name, tensor

        super().load_weights(_track_mapped_weights(mapped_iter))

        if not has_lm_head and self.tie_word_embeddings:
            logger.info(
                "[Qwen3NextForCausalLM] lm_head not found; tying to embed_tokens"
            )
            self.lm_head._copy_local_tied_weight(self.embed_tokens.weight.data)

    def process_weights_after_loading(self) -> None:
        if self._lm_head_postprocessed:
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
        return name.endswith("norm.weight")

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
        cfg = _extract_config_values(model_config, load_config)
        parallelism_config = cfg["parallelism_config"]
        fmha_config = getattr(load_config, "fmha_config", None)
        device_resource_config = getattr(load_config, "device_resource_config", None)

        super().__init__(
            config=model_config,
            parallelism_config=parallelism_config,
            weight=None,
            max_generate_batch_size=0,
            fmha_config=fmha_config,
            device_resource_config=device_resource_config,
        )

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

        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["hidden_size"],
            tp_size=cfg["attn_tp_size"],
            tp_rank=cfg["attn_tp_rank"],
            params_dtype=cfg["params_dtype"],
        )
        self.layers = nn.ModuleList(
            [
                Qwen3NextDecoderLayer(cfg, i, model_config, prefix=f"layers.{i}")
                for i in range(cfg["num_layers"])
            ]
        )
        self.norm = RMSResNorm(
            cfg["hidden_size"],
            eps=cfg["rms_norm_eps"],
            params_dtype=cfg["params_dtype"],
        )
        self.lm_head = ParallelLMHead(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            tp_size=cfg["lm_head_tp_size"],
            tp_rank=cfg["lm_head_tp_rank"],
            params_dtype=cfg["lm_head_params_dtype"],
        )

    def runtime_weight_view(self) -> Dict[str, torch.Tensor]:
        return {
            "embedding": self.embed_tokens.weight,
            "final_layernorm.gamma": self.norm.weight,
            "lm_head": self.lm_head.weight,
        }

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        hidden_states = self.embed_tokens(inputs.input_ids)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        attn_meta = _build_qwen3_next_metadata(inputs, hidden_states)
        residual = torch.zeros_like(hidden_states)
        for i, layer in enumerate(self.layers):
            select_block_map_for_layer(inputs.attention_inputs, i)
            hidden_states, residual = layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                attention_inputs=inputs.attention_inputs,
                attn_meta=attn_meta,
            )
        hidden_states, residual = self.norm(hidden_states, residual)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)

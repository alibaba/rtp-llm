"""Qwen3-Next / Qwen3.5 series for the new (vLLM-style) weight loader.

Implements 6 model_type variants:
  qwen3_next       — Dense + Linear Attention (prefix model.)
  qwen35_dense     — Dense (prefix model.language_model.)
  qwen35_moe       — MoE + Linear Attention + multimodal (prefix model.language_model.)
  qwen3_next_mtp   — MTP draft head for qwen3_next (prefix mtp.)
  qwen35_moe_mtp   — MTP draft head for qwen35_moe (prefix mtp.)
  qwen_3_moe_eagle3 — Eagle3 speculative draft on Qwen3-MoE architecture

Core architectural feature: **Linear Attention** (Gated Delta Network) used in
hybrid layers alongside standard MHA. Weight loading must handle:
  - plus_one: gemma_rms_norm adds 1 to all norm weights
  - split_q_gate: q_proj weight is split into Q (part 0) and Gate (part 1)
  - reorder_qkvz / reorder_ba: Linear Attention weight reordering
  - merge_qkvz / merge_ba: Qwen3.5 uses separate qkv+z / b+a files
"""

import functools
import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather, all_reduce
from rtp_llm.models_py.layers.embedding import (
    HiddenParallelEmbedding,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from rtp_llm.models_py.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.layers.norm import RMSNorm
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules import FusedMoeFactory, SelectTopk
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.new_models.qwen3.language import Qwen3Attention
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.weight_mapper import WeightsMapper
from rtp_llm.ops import HybridAttentionType, LinearAttentionConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W
from rtp_llm.utils.util import to_torch_dtype

logger = logging.getLogger(__name__)

_ACT_DUMPED: set[str] = set()
_ACT_COUNTS: Dict[str, int] = {}


def _should_dump_layer(layer_idx: int) -> bool:
    return layer_idx == 0 or os.environ.get("DUMP_QWEN_NEXT_ALL_LAYERS") == "1"


def _dump_activation(name: str, tensor: torch.Tensor):
    dump_dir = os.environ.get("DUMP_ACTIVATIONS_DIR")
    if not dump_dir:
        return
    max_dumps = int(os.environ.get("DUMP_ACTIVATIONS_STEPS", "1"))
    count = _ACT_COUNTS.get(name, 0)
    if count >= max_dumps:
        return
    _ACT_COUNTS[name] = count + 1
    dump_name = name if max_dumps == 1 else f"{name}.step{count}"
    try:
        import json

        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        t = tensor.detach()
        flat = t.reshape(-1)
        n = flat.numel()
        idx = sorted(
            set(
                list(range(min(8, n)))
                + list(range(max(0, n // 2 - 4), min(n, n // 2 + 4)))
                + list(range(max(0, n - 8), n))
            )
        )
        samples = flat[idx].to(torch.float32).cpu().tolist() if n else []
        entry = {
            "name": dump_name,
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "idx": idx,
            "samples": samples,
        }
        if n:
            tf = flat.to(torch.float32)
            abs_tf = tf.abs()
            argmax = int(abs_tf.argmax().item())
            entry.update(
                {
                    "sum": float(tf.sum().item()),
                    "abs_sum": float(abs_tf.sum().item()),
                    "abs_max": float(abs_tf[argmax].item()),
                    "argmax": argmax,
                }
            )
        if "conv_states" in name:
            entry.update(
                {
                    "nonzero": int(torch.count_nonzero(tf).item()) if n else 0,
                }
            )
        os.makedirs(dump_dir, exist_ok=True)
        with open(os.path.join(dump_dir, f"rank{rank}.jsonl"), "a") as f:
            f.write(json.dumps(entry, separators=(",", ":"), sort_keys=True) + "\n")
    except Exception:
        logger.exception("failed to dump activation %s", name)


class Qwen3NextMetadata:
    def __init__(self, prefill_conv1d_meta=None, is_target_verify: bool = False):
        self.prefill_conv1d_meta = prefill_conv1d_meta
        self.is_target_verify = is_target_verify

    def get_prefill_conv1d_meta(self):
        return self.prefill_conv1d_meta


def _build_qwen3_next_metadata(
    inputs: PyModelInputs, hidden_states: torch.Tensor
) -> Qwen3NextMetadata:
    attn_meta = getattr(inputs, "attn_meta", None)
    if attn_meta is not None:
        return attn_meta

    attention_inputs = inputs.attention_inputs
    prefill_conv1d_meta = None
    if attention_inputs.is_prefill and not attention_inputs.is_target_verify:
        from rtp_llm.models_py.triton_kernels.causal_conv1d import (
            prepare_causal_conv1d_metadata,
        )

        prefill_conv1d_meta = prepare_causal_conv1d_metadata(
            query_start_loc=attention_inputs.cu_seqlens,
            device=hidden_states.device,
        )
    return Qwen3NextMetadata(
        prefill_conv1d_meta=prefill_conv1d_meta,
        is_target_verify=attention_inputs.is_target_verify,
    )


# ================================================================== #
#  Weight transformation functions (ported from qwen3_next_weight.py)
# ================================================================== #


def split_q_gate(ts: List[torch.Tensor], head_num: int, head_dim: int, part: int):
    """Split q_gate tensor into q or gate part."""
    dim0, dim1 = ts[0].shape
    assert dim0 % (head_num * 2) == 0
    new_head_dim = dim0 // (head_num * 2)
    t = ts[0].reshape(head_num, 2, new_head_dim, dim1)
    if part == 0:
        return t[:, 0, :, :].reshape(-1, dim1)
    else:
        return t[:, 1, :, :].reshape(-1, dim1)


def reorder_ba(ts: List[torch.Tensor], la_cfg: LinearAttentionConfig):
    """Reorder ba weight: [head_num_k, group_v*2, hidden] -> [b_all, a_all, hidden].T"""
    t = ts[0]
    hidden_size = t.shape[-1]
    head_num_k = la_cfg.linear_num_key_heads
    head_num_v = la_cfg.linear_num_value_heads
    group_v = head_num_v // head_num_k
    t = t.reshape(head_num_k, group_v * 2, t.shape[-1])
    b, a = t.split([group_v, group_v], dim=1)
    return torch.cat([b.reshape(-1, hidden_size), a.reshape(-1, hidden_size)], dim=0)


def reorder_qkvz(ts: List[torch.Tensor], la_cfg: LinearAttentionConfig):
    """Reorder qkvz weight: split q,k,v,z per-head then concatenate."""
    t = ts[0]
    head_num_k = la_cfg.linear_num_key_heads
    head_num_v = la_cfg.linear_num_value_heads
    head_k_dim = la_cfg.linear_key_head_dim
    head_v_dim = la_cfg.linear_value_head_dim
    group_v = head_num_v // head_num_k
    dim0, dim1 = t.shape
    qkvz_size = head_num_k * head_k_dim * 2 + head_num_v * head_v_dim * 2
    BLOCK_SIZE = 128
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
    elif dim0 == qkvz_size // BLOCK_SIZE:
        t = t.reshape(
            head_num_k, (head_k_dim * 2 + head_v_dim * group_v * 2) // BLOCK_SIZE, dim1
        )
        q, k, v, z = torch.split(
            t,
            [
                head_k_dim // BLOCK_SIZE,
                head_k_dim // BLOCK_SIZE,
                head_v_dim * group_v // BLOCK_SIZE,
                head_v_dim * group_v // BLOCK_SIZE,
            ],
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
    else:
        raise ValueError(
            f"Invalid qkvz shape: {t.shape}, expected: {qkvz_size} or {qkvz_size // BLOCK_SIZE}"
        )


def merge_qkvz_transpose_reorder(ts: List[torch.Tensor], la_cfg: LinearAttentionConfig):
    """Merge separate qkv + z into qkvz then transpose (Qwen3.5 MoE format)."""
    qkv = ts[0]
    z = ts[1]
    return torch.cat([qkv, z], dim=0).T


def merge_ba_transpose_reorder(ts: List[torch.Tensor], la_cfg: LinearAttentionConfig):
    """Merge separate b + a then transpose (Qwen3.5 MoE format)."""
    b = ts[0]
    a = ts[1]
    return torch.cat([b, a], dim=0).T


def _transpose_weight(tensor: torch.Tensor) -> torch.Tensor:
    """Transpose 2D weight [out, in] -> [in, out] (or vice versa)."""
    return tensor.T.contiguous()


def _plus_one(tensor: torch.Tensor) -> torch.Tensor:
    """Add 1 for gemma_rms_norm."""
    return tensor + 1


# ================================================================== #
#  Config extraction
# ================================================================== #


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_config_values(model_config: Any, load_config: Any) -> Dict[str, Any]:
    """Pull all fields needed to build Qwen3Next layers.

    Tolerates either ModelConfig (C++ pybind) or HF dict.
    """
    hidden_size = _get(model_config, "hidden_size", 2048)
    num_layers = _get(
        model_config, "num_layers", _get(model_config, "num_hidden_layers", 48)
    )
    vocab_size = _get(model_config, "vocab_size", 151936)

    attn_config = _get(model_config, "attn_config", None)
    if attn_config is not None:
        num_heads = _get(attn_config, "head_num", 32)
        num_kv_heads = _get(attn_config, "kv_head_num", num_heads)
        head_dim = _get(attn_config, "size_per_head", hidden_size // num_heads)
    else:
        num_heads = _get(model_config, "num_attention_heads", 32)
        num_kv_heads = _get(model_config, "num_key_value_heads", num_heads)
        head_dim = _get(model_config, "head_dim", hidden_size // num_heads)

    rms_norm_eps = _get(
        model_config, "layernorm_eps", _get(model_config, "rms_norm_eps", 1e-6)
    )

    # Linear attention config
    la_cfg = _get(model_config, "linear_attention_config", None)
    linear_conv_kernel_dim = _get(la_cfg, "linear_conv_kernel_dim", 4) if la_cfg else 4
    linear_key_head_dim = _get(la_cfg, "linear_key_head_dim", 192) if la_cfg else 192
    linear_num_key_heads = _get(la_cfg, "linear_num_key_heads", 4) if la_cfg else 4
    linear_num_value_heads = (
        _get(la_cfg, "linear_num_value_heads", 16) if la_cfg else 16
    )
    linear_value_head_dim = (
        _get(la_cfg, "linear_value_head_dim", 192) if la_cfg else 192
    )

    # Hybrid attention
    hybrid_cfg = _get(model_config, "hybrid_attention_config", None)
    hybrid_types = []
    if hybrid_cfg is not None:
        hybrid_types = _get(hybrid_cfg, "hybrid_attention_types", []) or []

    # MoE config
    moe_k = _get(model_config, "moe_k", 0)
    expert_num = _get(model_config, "expert_num", _get(model_config, "num_experts", 0))
    moe_inter_size = _get(
        model_config, "moe_inter_size", _get(model_config, "moe_intermediate_size", 0)
    )
    inter_size = _get(
        model_config, "inter_size", _get(model_config, "intermediate_size", 0)
    )
    moe_style = _get(model_config, "moe_style", 0)
    moe_layer_index = _get(model_config, "moe_layer_index", None)
    has_moe_norm = _get(
        model_config, "has_moe_norm", _get(model_config, "norm_topk_prob", True)
    )

    # RoPE
    partial_rotary_factor = _get(model_config, "partial_rotary_factor", 1.0)

    # Tie embeddings
    tie_word_embeddings = _get(model_config, "tie_word_embeddings", False)

    # is_mtp
    is_mtp = _get(model_config, "is_mtp", False)

    # Parallelism
    tp_size = getattr(load_config, "tp_size", 1)
    tp_rank = getattr(load_config, "tp_rank", 0)
    ep_size = getattr(load_config, "ep_size", 1)
    ep_rank = getattr(load_config, "ep_rank", 0)
    quant_config = getattr(load_config, "quant_config", None)
    params_dtype = getattr(load_config, "compute_dtype", torch.bfloat16)
    enable_fp32_lm_head = _get(model_config, "enable_fp32_lm_head", True)
    parallelism_config = getattr(load_config, "parallelism_config", None)
    moe_config = getattr(load_config, "moe_config", None)

    return dict(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        vocab_size=vocab_size,
        rms_norm_eps=rms_norm_eps,
        linear_conv_kernel_dim=linear_conv_kernel_dim,
        linear_key_head_dim=linear_key_head_dim,
        linear_num_key_heads=linear_num_key_heads,
        linear_num_value_heads=linear_num_value_heads,
        linear_value_head_dim=linear_value_head_dim,
        hybrid_types=hybrid_types,
        moe_k=moe_k,
        expert_num=expert_num,
        moe_inter_size=moe_inter_size,
        inter_size=inter_size,
        moe_style=moe_style,
        moe_layer_index=moe_layer_index,
        has_moe_norm=has_moe_norm,
        partial_rotary_factor=partial_rotary_factor,
        tie_word_embeddings=tie_word_embeddings,
        is_mtp=is_mtp,
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        quant_config=quant_config,
        params_dtype=params_dtype,
        lm_head_params_dtype=torch.float32 if enable_fp32_lm_head else params_dtype,
        model_config=model_config,
        parallelism_config=parallelism_config,
        moe_config=moe_config,
    )


# ================================================================== #
#  GemmaRMSNorm — RMSNorm with +1 on weight (plus_one transform)
# ================================================================== #


class GemmaRMSNorm(RtpModule):
    """RMSNorm variant where the stored weight has +1 applied (gemma_rms_norm).

    In the old loader, plus_one is applied during weight loading. Here we
    intercept the weight in the model's load_weights and add 1 before dispatch.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        params_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=params_dtype), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            try:
                from rtp_llm.ops.compute_ops import rtp_llm_ops

                orig_shape = x.shape
                x2d = x.reshape(-1, orig_shape[-1]).contiguous()
                out = torch.empty_like(x2d)
                stream_id = torch.cuda.current_stream().cuda_stream
                rtp_llm_ops.rmsnorm(out, x2d, self.weight.data, self.eps, stream_id)
                return out.reshape(orig_shape)
            except Exception:
                pass
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class GemmaRMSResNorm(RtpModule):
    """Residual RMSNorm with gemma +1 convention (fused add+rmsnorm)."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        params_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=params_dtype), requires_grad=False
        )

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.is_cuda:
            try:
                from rtp_llm.ops.compute_ops import rtp_llm_ops

                stream_id = torch.cuda.current_stream().cuda_stream
                rtp_llm_ops.fused_add_rmsnorm(
                    hidden_states, residual, self.weight.data, self.eps, stream_id
                )
                return hidden_states, residual
            except Exception:
                pass
        # Fallback
        residual = residual + hidden_states
        input_dtype = residual.dtype
        x = residual.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype), residual


class Qwen3NextLinearAttnLinear(RtpModule):
    """Linear projection used by Qwen3-Next linear attention.

    The qkvz/out projections in FP8 checkpoints are already block-quantized.
    Keep them in FP8 form and reuse the common quant_method forward path
    instead of dequantizing to BF16 and calling torch linear.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.prefix = prefix
        if quant_config is None:
            quant_config = QuantizationConfig(quant_type="none")
        self.quant_config = quant_config
        self.quant_method = quant_config.get_quant_method(self, prefix)
        self.quant_method.create_weights(
            layer=self,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
        )

    def load_weight(
        self,
        weight: torch.Tensor,
        weight_scale_inv: Optional[torch.Tensor] = None,
    ):
        weight = weight.contiguous()
        if weight.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            del self.weight
            self.register_parameter("weight", nn.Parameter(weight, requires_grad=False))
            if weight_scale_inv is not None:
                if hasattr(self, "weight_scale_inv"):
                    del self.weight_scale_inv
                self.register_parameter(
                    "weight_scale_inv",
                    nn.Parameter(
                        weight_scale_inv.to(torch.float32).contiguous(),
                        requires_grad=False,
                    ),
                )
            return
        self.weight.data.copy_(weight)

    def process_weights_after_loading(self):
        if getattr(self, "_post_load_done", False):
            return
        self.quant_method.process_weights_after_loading(self)
        self._post_load_done = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            return self.quant_method.apply(self, x, None)
        return torch.nn.functional.linear(x, self.weight)


# ================================================================== #
#  Linear Attention — Gated Delta Network
# ================================================================== #


class Qwen3NextGatedDeltaNet(RtpModule):
    """Linear Attention (Gated Delta Network) for Qwen3-Next.

    Weight loading handles:
      - reorder_qkvz: single file (qwen3_next) or merge qkv+z (qwen35_moe)
      - reorder_ba: single file (qwen3_next) or merge b+a (qwen35_moe)
      - transpose: all linear projections transposed for storage
    Forward delegates to the same Triton kernels used by the old model_desc.
    """

    def __init__(
        self,
        linear_attn_config: LinearAttentionConfig,
        hidden_size: int,
        parallelism_config: ParallelismConfig,
        tp_size: int,
        tp_rank: int,
        rms_norm_eps: float,
        params_dtype: torch.dtype,
        quant_config: Optional[QuantizationConfig] = None,
        merge_format: bool = False,
    ):
        super().__init__()
        self.linear_attn_config = linear_attn_config
        self.parallelism_config = parallelism_config
        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.merge_format = merge_format  # True for qwen35 (separate qkv+z files)

        self.head_k_dim = linear_attn_config.linear_key_head_dim
        self.head_v_dim = linear_attn_config.linear_value_head_dim
        assert (
            self.head_k_dim == self.head_v_dim
        ), "head_k_dim and head_v_dim must be the same"

        attn_tp_size = (
            parallelism_config.get_attn_tp_size() if parallelism_config else tp_size
        )
        attn_tp_rank = (
            parallelism_config.get_attn_tp_rank() if parallelism_config else tp_rank
        )
        self.attn_tp_size = attn_tp_size
        self.attn_tp_rank = attn_tp_rank
        self.local_num_k_heads = linear_attn_config.linear_num_key_heads // attn_tp_size
        self.local_num_v_heads = (
            linear_attn_config.linear_num_value_heads // attn_tp_size
        )
        self.num_key_value_heads = self.local_num_v_heads // self.local_num_k_heads
        self.linear_conv_kernel_dim = linear_attn_config.linear_conv_kernel_dim

        # Compute sizes
        self.qkvz_size = (
            self.head_k_dim * self.local_num_k_heads * 2
            + self.head_v_dim * self.local_num_v_heads * 2
        )
        self.qkv_size = (
            self.head_k_dim * self.local_num_k_heads * 2
            + self.head_v_dim * self.local_num_v_heads
        )
        self.ba_size = self.local_num_v_heads * 2

        self.in_proj_qkvz = Qwen3NextLinearAttnLinear(
            input_size=hidden_size,
            output_size=self.qkvz_size,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix="in_proj_qkvz",
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
            torch.zeros(self.local_num_v_heads, dtype=params_dtype), requires_grad=False
        )
        self.a_log = nn.Parameter(
            torch.zeros(self.local_num_v_heads, dtype=params_dtype), requires_grad=False
        )
        self.norm_w = nn.Parameter(
            torch.ones(self.head_v_dim, dtype=params_dtype), requires_grad=False
        )
        self.out_proj = Qwen3NextLinearAttnLinear(
            input_size=self.local_num_v_heads * self.head_v_dim,
            output_size=hidden_size,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix="out_proj",
        )

        self.ssm_state_dtype = to_torch_dtype(linear_attn_config.ssm_state_dtype)
        self.conv_state_dtype = to_torch_dtype(linear_attn_config.conv_state_dtype)

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        """Handle weight transformations for Linear Attention.

        Expects keys like 'linear_attn.in_proj_qkvz.weight' (already stripped of layer prefix).
        For merge_format (qwen35), expects 'linear_attn.in_proj_qkv.weight' + 'linear_attn.in_proj_z.weight'
        and 'linear_attn.in_proj_b.weight' + 'linear_attn.in_proj_a.weight'.
        """
        la_cfg = self.linear_attn_config
        for name, tensor in weights.items():
            param_name = name.rsplit(".", 1)[-1] if "." in name else name

            if param_name == "weight_scale_inv":
                if "in_proj_qkvz" in name:
                    self._pending_qkvz_scale = reorder_qkvz([tensor], la_cfg)
                    self._maybe_load_qkvz()
                elif "in_proj_qkv" in name:
                    self._pending_qkv_scale = tensor
                    self._maybe_merge_qkvz()
                elif "in_proj_z" in name:
                    self._pending_z_scale = tensor
                    self._maybe_merge_qkvz()
                elif "out_proj" in name:
                    self._pending_out_proj_scale = tensor
                    self._maybe_load_out_proj()
            elif param_name == "weight" and "in_proj_qkvz" in name:
                # Single-file format (qwen3_next)
                reordered = reorder_qkvz([tensor], la_cfg)
                self._pending_qkvz = reordered
                self._maybe_load_qkvz()
            elif (
                param_name == "weight"
                and "in_proj_qkv" in name
                and "in_proj_qkvz" not in name
            ):
                # Separate qkv file (qwen35_moe) — need to merge with z
                self._pending_qkv = tensor
                self._maybe_merge_qkvz()
            elif param_name == "weight" and "in_proj_z" in name:
                self._pending_z = tensor
                self._maybe_merge_qkvz()
            elif param_name == "weight" and "in_proj_ba" in name:
                reordered = reorder_ba([tensor], la_cfg)
                self.in_proj_ba_w.data.copy_(self._split_ba_t(reordered.T.contiguous()))
            elif (
                param_name == "weight"
                and "in_proj_b" in name
                and "in_proj_ba" not in name
            ):
                self._pending_b = tensor
                self._maybe_merge_ba()
            elif param_name == "weight" and "in_proj_a" in name:
                self._pending_a = tensor
                self._maybe_merge_ba()
            elif param_name == "weight" and "out_proj" in name:
                self._pending_out_proj = tensor
                self._maybe_load_out_proj()
            elif param_name == "weight" and "norm" in name:
                # Linear Attention norm uses identity (NOT plus_one) in old loader
                self.norm_w.data.copy_(self._normalize_norm_weight(tensor))
            elif param_name == "weight" and "conv1d" in name:
                conv = tensor.squeeze(1) if tensor.dim() == 3 else tensor
                self.conv1d_w.data.copy_(self._split_conv1d(conv))
            elif param_name in ("bias", "dt_bias") and "dt" in name:
                self.dt_bias.data.copy_(self._split_head_tensor(tensor))
            elif param_name in ("A_log", "a_log"):
                self.a_log.data.copy_(self._split_head_tensor(tensor))

    @staticmethod
    def _dequant_fp8_block(
        weight: torch.Tensor, scale_inv: torch.Tensor, block: int = 128
    ) -> torch.Tensor:
        if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            return weight
        n, k = weight.shape
        scale = scale_inv.to(torch.float32)
        scale = scale.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
        scale = scale[:n, :k]
        return (weight.to(torch.float32) * scale).to(torch.bfloat16)

    def _maybe_load_qkvz(self):
        if not hasattr(self, "_pending_qkvz"):
            return
        weight = self._pending_qkvz
        if weight.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            if not hasattr(self, "_pending_qkvz_scale"):
                return
            scale = self._split_qkvz_scale(self._pending_qkvz_scale)
        else:
            scale = None
        self.in_proj_qkvz.load_weight(
            self._split_qkvz_t(weight.T.contiguous()).T.contiguous(),
            scale,
        )
        del self._pending_qkvz
        if hasattr(self, "_pending_qkvz_scale"):
            del self._pending_qkvz_scale

    def _maybe_merge_qkvz(self):
        if hasattr(self, "_pending_qkv") and hasattr(self, "_pending_z"):
            if self._pending_qkv.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
                if not (
                    hasattr(self, "_pending_qkv_scale")
                    and hasattr(self, "_pending_z_scale")
                ):
                    return
                scale = torch.cat(
                    [self._pending_qkv_scale, self._pending_z_scale], dim=0
                )
                qkv = self._pending_qkv
                z = self._pending_z
            else:
                qkv = self._pending_qkv
                z = self._pending_z
                scale = None
            merged = merge_qkvz_transpose_reorder([qkv, z], self.linear_attn_config)
            self.in_proj_qkvz.load_weight(
                self._split_qkvz_t(merged).T.contiguous(),
                self._split_qkvz_scale(scale) if scale is not None else None,
            )
            del self._pending_qkv
            del self._pending_z
            if hasattr(self, "_pending_qkv_scale"):
                del self._pending_qkv_scale
            if hasattr(self, "_pending_z_scale"):
                del self._pending_z_scale

    def _maybe_merge_ba(self):
        if hasattr(self, "_pending_b") and hasattr(self, "_pending_a"):
            merged = merge_ba_transpose_reorder(
                [self._pending_b, self._pending_a], self.linear_attn_config
            )
            self.in_proj_ba_w.data.copy_(self._split_ba_t(merged))
            del self._pending_b
            del self._pending_a

    def _maybe_load_out_proj(self):
        if not hasattr(self, "_pending_out_proj"):
            return
        weight = self._pending_out_proj
        if weight.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            if not hasattr(self, "_pending_out_proj_scale"):
                return
            scale = self._split_out_proj_scale(self._pending_out_proj_scale)
        else:
            scale = None
        self.out_proj.load_weight(
            self._split_out_proj_t(weight.T.contiguous()).T.contiguous(),
            scale,
        )
        del self._pending_out_proj
        if hasattr(self, "_pending_out_proj_scale"):
            del self._pending_out_proj_scale

    def _split_qkvz_t(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape == (self.hidden_size, self.qkvz_size):
            return tensor
        q_size = self.head_k_dim * self.linear_attn_config.linear_num_key_heads
        k_size = q_size
        v_size = self.head_v_dim * self.linear_attn_config.linear_num_value_heads
        q, k, v, z = torch.split(tensor, [q_size, k_size, v_size, v_size], dim=1)
        local_k = self.head_k_dim * self.local_num_k_heads
        local_v = self.head_v_dim * self.local_num_v_heads
        k_start = local_k * self.attn_tp_rank
        v_start = local_v * self.attn_tp_rank
        return torch.cat(
            [
                q[:, k_start : k_start + local_k],
                k[:, k_start : k_start + local_k],
                v[:, v_start : v_start + local_v],
                z[:, v_start : v_start + local_v],
            ],
            dim=1,
        ).contiguous()

    def _split_qkvz_scale(self, scale: torch.Tensor) -> torch.Tensor:
        block_n, _ = getattr(
            self.in_proj_qkvz.quant_config, "weight_block_size", [128, 128]
        )
        q_size = self.head_k_dim * self.linear_attn_config.linear_num_key_heads
        k_size = q_size
        v_size = self.head_v_dim * self.linear_attn_config.linear_num_value_heads
        q_blocks = self._ceil_div(q_size, block_n)
        k_blocks = self._ceil_div(k_size, block_n)
        v_blocks = self._ceil_div(v_size, block_n)
        local_k_blocks = self._ceil_div(
            self.head_k_dim * self.local_num_k_heads, block_n
        )
        local_v_blocks = self._ceil_div(
            self.head_v_dim * self.local_num_v_heads, block_n
        )
        if scale.shape[0] == (local_k_blocks * 2 + local_v_blocks * 2):
            return scale.contiguous()
        q, k, v, z = torch.split(scale, [q_blocks, k_blocks, v_blocks, v_blocks], dim=0)
        k_start = local_k_blocks * self.attn_tp_rank
        v_start = local_v_blocks * self.attn_tp_rank
        return torch.cat(
            [
                q[k_start : k_start + local_k_blocks],
                k[k_start : k_start + local_k_blocks],
                v[v_start : v_start + local_v_blocks],
                z[v_start : v_start + local_v_blocks],
            ],
            dim=0,
        ).contiguous()

    def _split_ba_t(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape == self.in_proj_ba_w.shape:
            return tensor
        b, a = torch.split(
            tensor,
            [
                self.linear_attn_config.linear_num_value_heads,
                self.linear_attn_config.linear_num_value_heads,
            ],
            dim=1,
        )
        local = self.local_num_v_heads
        start = local * self.attn_tp_rank
        return torch.cat(
            [b[:, start : start + local], a[:, start : start + local]], dim=1
        ).contiguous()

    def _split_out_proj_t(self, tensor: torch.Tensor) -> torch.Tensor:
        local_out_proj_input = self.head_v_dim * self.local_num_v_heads
        if tensor.shape == (local_out_proj_input, self.hidden_size):
            return tensor
        out_dim = self.head_v_dim * self.linear_attn_config.linear_num_value_heads
        if tensor.shape[0] != out_dim:
            return tensor
        local = self.head_v_dim * self.local_num_v_heads
        start = local * self.attn_tp_rank
        return tensor[start : start + local, :].contiguous()

    def _split_out_proj_scale(self, scale: torch.Tensor) -> torch.Tensor:
        _, block_k = getattr(
            self.out_proj.quant_config, "weight_block_size", [128, 128]
        )
        local = self.head_v_dim * self.local_num_v_heads
        local_blocks = self._ceil_div(local, block_k)
        if scale.shape[1] == local_blocks:
            return scale.contiguous()
        start = local_blocks * self.attn_tp_rank
        return scale[:, start : start + local_blocks].contiguous()

    @staticmethod
    def _ceil_div(x: int, y: int) -> int:
        return (x + y - 1) // y

    def _split_head_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape == self.dt_bias.shape or tensor.shape == self.a_log.shape:
            return tensor
        if tensor.shape[0] != self.linear_attn_config.linear_num_value_heads:
            return tensor
        local = self.local_num_v_heads
        start = local * self.attn_tp_rank
        return tensor[start : start + local].contiguous()

    def _normalize_norm_weight(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape == self.norm_w.shape:
            return tensor
        if tensor.dim() == 2 and tensor.shape[-1] == self.head_v_dim:
            return tensor.reshape(-1, self.head_v_dim)[0].contiguous()
        return tensor

    def _split_conv1d(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape == self.conv1d_w.shape:
            return tensor
        if tensor.dim() != 2:
            return tensor
        if tensor.T.shape == self.conv1d_w.shape:
            return tensor.T.contiguous()

        q_size = self.head_k_dim * self.linear_attn_config.linear_num_key_heads
        k_size = q_size
        v_size = self.head_v_dim * self.linear_attn_config.linear_num_value_heads
        full_qkv = q_size + k_size + v_size

        # HF stores conv1d as [channels, kernel] (or occasionally transposed);
        # causal_conv1d_fn expects [local_channels, kernel].
        if tensor.shape[1] == full_qkv:
            tensor = tensor.T.contiguous()
        if tensor.shape[0] == q_size + k_size + v_size:
            q, k, v = torch.split(tensor, [q_size, k_size, v_size], dim=0)
            local_k = self.head_k_dim * self.local_num_k_heads
            local_v = self.head_v_dim * self.local_num_v_heads
            k_start = local_k * self.attn_tp_rank
            v_start = local_v * self.attn_tp_rank
            tensor = torch.cat(
                [
                    q[k_start : k_start + local_k],
                    k[k_start : k_start + local_k],
                    v[v_start : v_start + local_v],
                ],
                dim=0,
            )
        if tensor.shape == self.conv1d_w.shape:
            return tensor
        return tensor

    def process_weights_after_loading(self):
        self.in_proj_qkvz.process_weights_after_loading()
        self.out_proj.process_weights_after_loading()

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        attn_meta: Any = None,
    ) -> torch.Tensor:
        """Run Gated Delta Network linear attention.

        Delegates to the existing Triton kernels from triton_kernels/fla/.
        """
        from rtp_llm.models_py.triton_kernels.causal_conv1d import (
            CausalConv1dMetadata,
            causal_conv1d_fn,
            causal_conv1d_update,
            prepare_causal_conv1d_metadata,
        )
        from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated
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

        assert attention_inputs is not None

        projected_qkvz = self.in_proj_qkvz(hidden_states)
        projected_ba = torch.nn.functional.linear(hidden_states, self.in_proj_ba_w.T)
        if getattr(self, "layer_idx", -1) == 0:
            _dump_activation("new.la0.projected_qkvz", projected_qkvz)
            _dump_activation("new.la0.projected_ba", projected_ba)

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
        if getattr(self, "layer_idx", -1) == 0:
            _dump_activation("new.la0.mixed_qkv", mixed_qkv)
            _dump_activation("new.la0.z", z)
            _dump_activation("new.la0.b", b)
            _dump_activation("new.la0.a", a)

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

        # Norm gating
        norm = RmsNormGated(self.norm_w, eps=1e-6, group_size=self.head_v_dim)
        attn_output = norm(
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
        attn_meta,
    ):
        from rtp_llm.models_py.triton_kernels.fla.block import (
            load_initial_state_from_block_map,
            store_ssm_state_to_block_map,
        )
        from rtp_llm.ops.compute_ops import rtp_llm_ops as compute_ops

        cu_seqlens = attn_inputs.cu_seqlens
        conv_states = (
            linear_cache_converter.get_conv_state_tensor(kv_cache_tensor).transpose(
                1, 2
            )
            if kv_cache_tensor is not None
            else None
        )
        if getattr(self, "layer_idx", -1) == 0 and conv_states is not None:
            _dump_activation("new.la0.prefill.conv_states.before", conv_states)
        conv1d_meta = None
        if attn_meta is not None and hasattr(attn_meta, "get_prefill_conv1d_meta"):
            conv1d_meta = attn_meta.get_prefill_conv1d_meta()
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
            prefix_lengths=attn_inputs.prefix_lengths_d,
            metadata=conv1d_meta,
        ).transpose(0, 1)
        if getattr(self, "layer_idx", -1) == 0 and conv_states is not None:
            _dump_activation("new.la0.prefill.conv_states.after", conv_states)

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
                attn_inputs.prefix_lengths_d,
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
                    attn_inputs.prefix_lengths_d if ssm_states is not None else None
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
                    attn_inputs.prefix_lengths_d,
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
                attn_inputs.kv_cache_block_id_host,
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
            assert attn_inputs.prefix_lengths.size(0) > 0
            batch = attn_inputs.prefix_lengths.size(0)
            seq = token // batch

        origin_shape = mixed_qkv.shape
        mixed_qkv = mixed_qkv.reshape(batch, seq, -1).transpose(1, 2)
        conv_states = linear_cache_converter.get_conv_state_tensor(kv_cache_tensor)
        if getattr(self, "layer_idx", -1) == 0:
            _dump_activation("new.la0.decode.conv_states.before", conv_states)
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
                sequence_lengths=attn_inputs.sequence_lengths_plus_1_d,
            )
            .transpose(1, 2)
            .reshape(origin_shape)
        )
        if getattr(self, "layer_idx", -1) == 0:
            _dump_activation("new.la0.decode.conv_states.after", conv_states)
            _dump_activation("new.la0.decode.after_conv", mixed_qkv)

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
        if getattr(self, "layer_idx", -1) == 0:
            _dump_activation("new.la0.decode.g", g)
            _dump_activation("new.la0.decode.beta", beta)
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
            sequence_lengths=attn_inputs.sequence_lengths_plus_1_d,
            use_qk_l2norm_in_kernel=True,
        )
        res = core_attn_out.reshape(-1, core_attn_out.shape[2], core_attn_out.shape[3])
        if getattr(self, "layer_idx", -1) == 0:
            _dump_activation("new.la0.decode.core_attn_out", res)
        return res


# ================================================================== #
#  Standard Attention with Gate
# ================================================================== #


class Qwen3NextAttention(RtpModule):
    """Standard MHA attention with a gate projection.

    Q-Gate fusion: the checkpoint q_proj.weight contains both Q and Gate
    interleaved per-head. During load_weights, split_q_gate separates them.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        layer_idx,
        tp_size,
        tp_rank,
        rms_norm_eps,
        params_dtype,
        quant_config=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.tp_size = tp_size

        self.num_heads_per_partition = num_heads // tp_size
        self.num_kv_heads_per_partition = max(1, num_kv_heads // tp_size)
        self.q_size = self.num_heads_per_partition * head_dim
        self.kv_size = self.num_kv_heads_per_partition * head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix="qkv_proj",
            bias=False,
            params_dtype=params_dtype,
        )
        self.gate = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=num_heads * head_dim,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix="gate",
            bias=False,
            params_dtype=params_dtype,
        )
        self.q_norm = GemmaRMSNorm(
            head_dim, eps=rms_norm_eps, params_dtype=params_dtype
        )
        self.k_norm = GemmaRMSNorm(
            head_dim, eps=rms_norm_eps, params_dtype=params_dtype
        )
        self.o_proj = RowParallelLinear(
            input_size=num_heads * head_dim,
            output_size=hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix="o_proj",
            bias=False,
            params_dtype=params_dtype,
        )

    def _apply_qk_norm(self, qkv):
        prefix_shape = qkv.shape[:-1]
        q = qkv[..., : self.q_size].reshape(
            *prefix_shape, self.num_heads_per_partition, self.head_dim
        )
        k = qkv[..., self.q_size : self.q_size + self.kv_size].reshape(
            *prefix_shape, self.num_kv_heads_per_partition, self.head_dim
        )
        v = qkv[..., self.q_size + self.kv_size :]
        q = self.q_norm(q).reshape(*prefix_shape, self.q_size)
        k = self.k_norm(k).reshape(*prefix_shape, self.kv_size)
        return torch.cat([q, k, v], dim=-1)

    def forward(
        self,
        hidden_states,
        fmha_impl,
        kv_cache=None,
        attention_inputs=None,
        attn_meta=None,
    ):
        gate = self.gate(hidden_states)
        input_shape = hidden_states.shape[:-1]
        qkv = self.qkv_proj(hidden_states)
        qkv = self._apply_qk_norm(qkv)
        attn_output = fmha_impl.forward(qkv, kv_cache, self.layer_idx)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        output = self.o_proj(attn_output)
        if self.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output


# ================================================================== #
#  MLP — Dense and MoE with shared expert
# ================================================================== #


class Qwen3NextMLP(RtpModule):
    """Dense MLP (SiGLU activation)."""

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        tp_size,
        tp_rank,
        params_dtype,
        quant_config=None,
    ):
        super().__init__()
        self.tp_size = tp_size
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_size=2 * intermediate_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix="gate_up_proj",
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
            prefix="down_proj",
            bias=False,
            params_dtype=params_dtype,
        )

    def forward(self, x):
        from rtp_llm.models_py.layers.activation import silu_and_mul

        gate_up = self.gate_up_proj(x)
        act = silu_and_mul(gate_up)
        x = self.down_proj(act)
        if self.tp_size > 1:
            x = all_reduce(x, group=Group.TP)
        return x

    def process_weights_after_loading(self):
        if hasattr(self.gate_up_proj, "process_weights_after_loading"):
            self.gate_up_proj.process_weights_after_loading()
        if hasattr(self.down_proj, "process_weights_after_loading"):
            self.down_proj.process_weights_after_loading()


class Qwen3NextSharedExpert(RtpModule):
    """Shared expert FFN (always active alongside routed experts).

    Note: gate_up_proj / down_proj are placed directly on this module (not
    nested inside a sub-MLP) so that the RtpModule redirect mechanism can
    match checkpoint keys 'shared_expert.gate_proj.weight' etc. directly.
    """

    def __init__(
        self, hidden_size, inter_size, tp_size, tp_rank, params_dtype, quant_config=None
    ):
        super().__init__()
        self.tp_size = tp_size
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_size=2 * inter_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix="gate_up_proj",
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
            prefix="down_proj",
            bias=False,
            params_dtype=params_dtype,
        )

    def forward(self, x, skip_allreduce: bool = False):
        from rtp_llm.models_py.layers.activation import silu_and_mul

        gate_up = self.gate_up_proj(x)
        act = silu_and_mul(gate_up)
        x = self.down_proj(act)
        if not skip_allreduce and self.tp_size > 1:
            x = all_reduce(x, group=Group.TP)
        return x

    def process_weights_after_loading(self):
        if hasattr(self.gate_up_proj, "process_weights_after_loading"):
            self.gate_up_proj.process_weights_after_loading()
        if hasattr(self.down_proj, "process_weights_after_loading"):
            self.down_proj.process_weights_after_loading()


def _process_qwen35_fp8_mlp_layout(gate_up_proj, down_proj):
    """Make qwen3-next FP8 MLP tensors match legacy runtime semantics.

    Legacy ModelWeights stores FFN weights in [input, output] order. The new
    Linear layers store [output, input]. For qwen3.5 FP8 checkpoints the raw
    bytes otherwise match only as a flat buffer, which gives a wrong logical
    matrix unless we do the real transpose after the quant method has created
    weight_scale.
    """
    for linear in (gate_up_proj, down_proj):
        if hasattr(linear, "process_weights_after_loading"):
            linear.process_weights_after_loading()
        if getattr(linear, "_qwen35_fp8_layout_fixed", False):
            continue
        weight = getattr(linear, "weight", None)
        if weight is None or weight.dtype not in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ):
            continue
        input_size = getattr(linear, "input_size", None)
        output_size = getattr(linear, "output_size", None)
        if input_size is None or output_size is None:
            continue
        if weight.numel() != input_size * output_size:
            continue
        weight.data.copy_(weight.data.reshape(input_size, output_size).T.contiguous())

        scale = getattr(linear, "weight_scale", None)
        if scale is not None and scale.dim() == 2:
            block_n, block_k = getattr(
                linear.quant_config, "weight_block_size", [128, 128]
            )
            in_blocks = (input_size + block_k - 1) // block_k
            out_blocks = (output_size + block_n - 1) // block_n
            if scale.numel() == in_blocks * out_blocks:
                scale.data.copy_(
                    scale.data.reshape(in_blocks, out_blocks).T.contiguous()
                )
        linear._qwen35_fp8_layout_fixed = True


class Qwen3NextMoEBlock(RtpModule):
    """MoE block with routed experts + shared expert + shared_expert_gate.

    moe_style=2: shared expert + routed experts (Qwen3-Next style).
    """

    def __init__(
        self,
        hidden_size,
        moe_inter_size,
        expert_num,
        top_k,
        shared_inter_size,
        layer_idx,
        tp_size,
        tp_rank,
        ep_size,
        ep_rank,
        model_config,
        parallelism_config,
        moe_config,
        quant_config,
        params_dtype,
        has_shared_expert_gate=True,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.tp_size = tp_size
        self.ep_size = ep_size
        self.top_k = top_k
        self.hidden_size = hidden_size

        # Router gate (not TP-sharded)
        self.gate = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=expert_num,
            tp_size=1,
            tp_rank=0,
            quant_config=None,
            prefix="gate",
            bias=False,
            params_dtype=params_dtype,
        )
        self.select_topk = SelectTopk(config=model_config)

        # Routed experts (reuse Qwen3Experts)
        from rtp_llm.models_py.new_models.qwen3_moe.language import Qwen3Experts

        self.experts = Qwen3Experts(
            num_experts=expert_num,
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
        )

        # Shared expert
        self.shared_expert = Qwen3NextSharedExpert(
            hidden_size,
            shared_inter_size,
            tp_size,
            tp_rank,
            params_dtype,
            quant_config,
        )

        # Shared expert gate
        if has_shared_expert_gate:
            self.shared_expert_gate = ColumnParallelLinear(
                input_size=hidden_size,
                output_size=1,
                tp_size=1,
                tp_rank=0,
                quant_config=None,
                prefix="shared_expert_gate",
                bias=False,
                params_dtype=params_dtype,
            )
        else:
            self.shared_expert_gate = None

    def forward(self, hidden_states):
        num_tokens = hidden_states.shape[0]
        if self.layer_idx == 0:
            _dump_activation("new.l0.mlp.input", hidden_states)
        router_logits = self.gate(hidden_states).float()
        if self.layer_idx == 0:
            _dump_activation("new.l0.mlp.router_logits", router_logits)

        topk_weights = torch.empty(
            (num_tokens, self.top_k), dtype=torch.float32, device=hidden_states.device
        )
        topk_ids = torch.empty(
            (num_tokens, self.top_k),
            dtype=(
                self.experts.fused_moe.topk_ids_dtype
                if self.experts.fused_moe is not None
                else torch.int32
            ),
            device=hidden_states.device,
        )
        self.select_topk(router_logits, topk_ids, topk_weights)
        if self.layer_idx == 0:
            _dump_activation("new.l0.mlp.topk_ids", topk_ids)
            _dump_activation("new.l0.mlp.topk_weights", topk_weights)

        experts_output = self.experts(hidden_states, topk_weights, topk_ids)
        if self.layer_idx == 0:
            _dump_activation("new.l0.mlp.experts_output", experts_output)

        use_ep_shared_allreduce = self.tp_size > 1 and self.ep_size > 1
        shared_output = self.shared_expert(
            hidden_states, skip_allreduce=use_ep_shared_allreduce
        )
        if self.layer_idx == 0:
            _dump_activation("new.l0.mlp.shared_output", shared_output)
        if self.shared_expert_gate is not None:
            gate_output = self.shared_expert_gate(hidden_states)
            if self.layer_idx == 0:
                _dump_activation("new.l0.mlp.shared_gate", gate_output)
            shared_contribution = torch.sigmoid(gate_output) * shared_output
            if use_ep_shared_allreduce:
                shared_contribution = all_reduce(shared_contribution, group=Group.TP)
            if self.layer_idx == 0:
                _dump_activation("new.l0.mlp.shared_contribution", shared_contribution)
            experts_output = experts_output + shared_contribution
        else:
            if use_ep_shared_allreduce:
                shared_output = all_reduce(shared_output, group=Group.TP)
            if self.layer_idx == 0:
                _dump_activation("new.l0.mlp.shared_contribution", shared_output)
            experts_output = experts_output + shared_output

        if self.layer_idx == 0:
            _dump_activation("new.l0.mlp.output", experts_output)
        return experts_output


# ================================================================== #
#  Decoder Layer
# ================================================================== #


class Qwen3NextDecoderLayer(RtpModule):

    def __init__(
        self,
        cfg: Dict[str, Any],
        layer_idx: int,
        model_config: Any,
        load_config: Any,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        hidden_size = cfg["hidden_size"]
        rms_eps = cfg["rms_norm_eps"]

        # Determine layer type (LINEAR or NONE/standard)
        hybrid_types = cfg["hybrid_types"]
        if layer_idx < len(hybrid_types):
            self.layer_type = hybrid_types[layer_idx]
        else:
            self.layer_type = HybridAttentionType.NONE

        # Attention — checkpoint uses 'linear_attn.*' for Linear Attention layers
        # and 'self_attn.*' for standard MHA layers, so the attribute name must
        # match the checkpoint key prefix.
        if self.layer_type == HybridAttentionType.LINEAR:
            self.linear_attn = Qwen3NextGatedDeltaNet(
                linear_attn_config=model_config.linear_attention_config,
                hidden_size=hidden_size,
                parallelism_config=cfg["parallelism_config"],
                tp_size=cfg["tp_size"],
                tp_rank=cfg["tp_rank"],
                rms_norm_eps=rms_eps,
                params_dtype=cfg["params_dtype"],
                quant_config=cfg["quant_config"],
                merge_format=cfg.get("merge_format", False),
            )
            self.linear_attn.layer_idx = layer_idx
        else:
            self.self_attn = Qwen3NextAttention(
                hidden_size=hidden_size,
                num_heads=cfg["num_heads"],
                num_kv_heads=cfg["num_kv_heads"],
                head_dim=cfg["head_dim"],
                layer_idx=layer_idx,
                tp_size=cfg["tp_size"],
                tp_rank=cfg["tp_rank"],
                rms_norm_eps=rms_eps,
                params_dtype=cfg["params_dtype"],
                quant_config=cfg["quant_config"],
            )

        # MLP / MoE
        moe_layer_index = cfg["moe_layer_index"] or []
        is_moe_layer = layer_idx in moe_layer_index or cfg["moe_style"] == 2
        if is_moe_layer and cfg["expert_num"] > 0:
            self.mlp = Qwen3NextMoEBlock(
                hidden_size=hidden_size,
                moe_inter_size=cfg["moe_inter_size"],
                expert_num=cfg["expert_num"],
                top_k=cfg["moe_k"],
                shared_inter_size=cfg["inter_size"],
                layer_idx=layer_idx,
                tp_size=cfg["tp_size"],
                tp_rank=cfg["tp_rank"],
                ep_size=cfg["ep_size"],
                ep_rank=cfg["ep_rank"],
                model_config=model_config,
                parallelism_config=cfg["parallelism_config"],
                moe_config=cfg["moe_config"],
                quant_config=cfg["quant_config"],
                params_dtype=cfg["params_dtype"],
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=hidden_size,
                intermediate_size=cfg["inter_size"],
                tp_size=cfg["tp_size"],
                tp_rank=cfg["tp_rank"],
                params_dtype=cfg["params_dtype"],
                quant_config=cfg["quant_config"],
            )

        self.input_layernorm = GemmaRMSResNorm(
            hidden_size, eps=rms_eps, params_dtype=cfg["params_dtype"]
        )
        self.post_attention_layernorm = GemmaRMSResNorm(
            hidden_size, eps=rms_eps, params_dtype=cfg["params_dtype"]
        )

    def forward(
        self,
        hidden_states,
        residual,
        fmha_impl,
        kv_cache=None,
        attention_inputs=None,
        attn_meta=None,
    ):
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if _should_dump_layer(self.layer_idx):
            _dump_activation(f"new.l{self.layer_idx}.after_input_ln", hidden_states)
        if self.layer_type == HybridAttentionType.LINEAR:
            hidden_states = self.linear_attn(
                hidden_states, fmha_impl, kv_cache, attention_inputs, attn_meta
            )
        else:
            hidden_states = self.self_attn(hidden_states, fmha_impl, kv_cache)
        if _should_dump_layer(self.layer_idx):
            _dump_activation(f"new.l{self.layer_idx}.after_attn", hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if _should_dump_layer(self.layer_idx):
            _dump_activation(f"new.l{self.layer_idx}.after_post_ln", hidden_states)
        hidden_states = self.mlp(hidden_states)
        if _should_dump_layer(self.layer_idx):
            _dump_activation(f"new.l{self.layer_idx}.after_mlp", hidden_states)
            _dump_activation(f"new.l{self.layer_idx}.residual", residual)
        return hidden_states, residual


# ================================================================== #
#  Model classes
# ================================================================== #


class Qwen3NextForCausalLM(GptModelBase):
    """Qwen3-Next Dense + Linear Attention (prefix model.)."""

    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"model.": ""})

    def load_weights(self, weights):
        if isinstance(weights, dict):
            weights_iter = iter(weights.items())
        else:
            weights_iter = weights
        has_lm_head = False

        def _transform(it):
            nonlocal has_lm_head
            cfg = self._cfg
            for name, tensor in it:
                # Track lm_head
                if name == "lm_head.weight" or name.startswith("lm_head."):
                    has_lm_head = True
                # plus_one for norm weights
                if self._is_norm_weight(name):
                    tensor = _plus_one(tensor)
                # split_q_gate for standard attention q_proj and its FP8 block scales.
                if name.endswith("self_attn.q_proj.weight") or name.endswith(
                    "self_attn.q_proj.weight_scale_inv"
                ):
                    yield from self._split_q_gate_yield(name, tensor, cfg)
                    continue
                yield name, tensor

        mapped_iter = self.WEIGHTS_MAPPER.apply(_transform(weights_iter))
        super().load_weights(mapped_iter)

        if not has_lm_head and self._cfg["tie_word_embeddings"]:
            logger.info(
                "[Qwen3NextForCausalLM] lm_head not found; tying to embed_tokens"
            )
            self.lm_head.weight.data.copy_(self.embed_tokens.weight.data)

    @staticmethod
    def _is_norm_weight(name):
        # Linear Attention norm uses 'identity' (NOT plus_one) in the old loader,
        # so we must exclude it from the plus_one transform.
        if "linear_attn" in name:
            return False
        # Standard norm weights: input_layernorm, post_attention_layernorm,
        # q_norm, k_norm, final norm — all use plus_one (gemma_rms_norm).
        # MTP-specific norms: pre_fc_norm_embedding, pre_fc_norm_hidden.
        return (
            name.endswith("norm.weight")
            or name.endswith("q_norm.weight")
            or name.endswith("k_norm.weight")
            or "pre_fc_norm_embedding" in name
            or "pre_fc_norm_hidden" in name
        )

    @staticmethod
    def _split_q_gate_yield(name, tensor, cfg):
        """Split q_proj weight/scale into Q (part 0) and Gate (part 1)."""
        head_num = cfg["num_heads"]
        head_dim = cfg["head_dim"]
        q_part = split_q_gate([tensor], head_num=head_num, head_dim=head_dim, part=0)
        gate_part = split_q_gate([tensor], head_num=head_num, head_dim=head_dim, part=1)
        suffix = name.rsplit(".", 1)[-1]
        base = name.rsplit(".", 1)[0]  # e.g. "layers.0.self_attn.q_proj"
        qkv_name = base.replace("q_proj", "qkv_proj") + f".q_proj.{suffix}"
        gate_name = base.replace("q_proj", "gate") + f".{suffix}"
        yield qkv_name, q_part
        yield gate_name, gate_part

    def _prepare_config(self, model_config: Any, load_config: Any):
        """Hook for subclasses to modify cfg before layers are created."""
        pass

    def __init__(self, model_config: Any, load_config: Any):
        parallelism_config = getattr(load_config, "parallelism_config", None)
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

        self._cfg = _extract_config_values(model_config, load_config)
        self._prepare_config(model_config, load_config)
        cfg = self._cfg

        embedding_cls = (
            HiddenParallelEmbedding
            if cfg.get("hidden_parallel_embedding", False)
            else VocabParallelEmbedding
        )
        self.embed_tokens = embedding_cls(
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["hidden_size"],
            tp_size=cfg["tp_size"],
            tp_rank=cfg["tp_rank"],
            params_dtype=cfg["params_dtype"],
        )
        self.layers = nn.ModuleList(
            [
                Qwen3NextDecoderLayer(cfg, i, model_config, load_config)
                for i in range(cfg["num_layers"])
            ]
        )
        self.norm = GemmaRMSResNorm(
            cfg["hidden_size"],
            eps=cfg["rms_norm_eps"],
            params_dtype=cfg["params_dtype"],
        )
        self.lm_head = ParallelLMHead(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            tp_size=cfg["tp_size"],
            tp_rank=cfg["tp_rank"],
            params_dtype=cfg["lm_head_params_dtype"],
        )

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
        _dump_activation("new.final_hidden", hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


class Qwen35MoeForCausalLM(Qwen3NextForCausalLM):
    """Qwen3.5 MoE + Linear Attention + multimodal (prefix model.language_model.).

    Uses separate qkv+z / b+a files for Linear Attention (merge_format=True).
    Also handles multimodal embedding injection.
    """

    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"model.language_model.": ""})

    def _prepare_config(self, model_config: Any, load_config: Any):
        self._cfg["merge_format"] = True
        self._cfg["hidden_parallel_embedding"] = True

    def __init__(self, model_config: Any, load_config: Any):
        super().__init__(model_config, load_config)
        from rtp_llm.models_py.modules import MultimodalEmbeddingInjector

        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids = inputs.input_ids
        position_ids = inputs.combo_position_ids
        token_type_ids = inputs.embedding_inputs.combo_tokens_type_ids
        text_tokens_mask = inputs.embedding_inputs.text_tokens_mask
        mm_features = inputs.multimodal_inputs.multimodal_features
        mm_feature_locs = inputs.multimodal_inputs.mm_features_locs

        inputs_embeds = self.embed_tokens(
            input_ids, position_ids, token_type_ids, text_tokens_mask
        )
        hidden_states = self.multimodal_embedding_injector(
            inputs_embeds, mm_features, mm_feature_locs
        )

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
        _dump_activation("new.final_hidden", hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


class Qwen35DenseForCausalLM(Qwen3NextForCausalLM):
    """Qwen3.5 Dense (prefix model.language_model., no MoE, no Linear Attention by default).

    Uses standard Dense MLP (no shared expert, no routed experts).
    """

    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"model.language_model.": ""})

    def _prepare_config(self, model_config: Any, load_config: Any):
        self._cfg["merge_format"] = True
        self._cfg["hidden_parallel_embedding"] = True
        self._cfg["moe_style"] = 0
        self._cfg["expert_num"] = 0

    def __init__(self, model_config: Any, load_config: Any):
        super().__init__(model_config, load_config)
        # Dense: no MoE, use inter_size for MLP
        self._cfg["moe_style"] = 0
        self._cfg["expert_num"] = 0


# ================================================================== #
#  MTP draft models
# ================================================================== #


class Qwen3NextMTPForCausalLM(Qwen3NextForCausalLM):
    """MTP draft head for Qwen3-Next (prefix mtp.).

    1-layer model with standard attention (no Linear Attention).
    Extra weights: pre_fc_norm_embedding, pre_fc_norm_hidden, fc, norm.
    """

    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"mtp.": ""})

    def _prepare_config(self, model_config: Any, load_config: Any):
        self._cfg["num_layers"] = 1
        self._cfg["hybrid_types"] = [HybridAttentionType.NONE]
        self._cfg["moe_layer_index"] = [0]
        self._cfg["is_mtp"] = True

    def load_weights(self, weights):
        if isinstance(weights, dict):
            weights_iter = iter(weights.items())
        else:
            weights_iter = weights
        has_lm_head = False
        has_embed = False

        def _transform(it):
            nonlocal has_lm_head, has_embed
            cfg = self._cfg
            for name, tensor in it:
                if name == "lm_head.weight" or name.startswith("lm_head."):
                    has_lm_head = True
                # model.embed_tokens.weight -> embed_tokens.weight
                if "embed_tokens.weight" in name and not name.startswith(
                    "embed_tokens"
                ):
                    has_embed = True
                    yield "embed_tokens.weight", tensor
                    continue
                if self._is_norm_weight(name):
                    tensor = _plus_one(tensor)
                if name.endswith("self_attn.q_proj.weight") or name.endswith(
                    "self_attn.q_proj.weight_scale_inv"
                ):
                    yield from self._split_q_gate_yield(name, tensor, cfg)
                    continue
                yield name, tensor

        # Note: MTP uses model. prefix for embed_tokens, mtp. for the rest
        # The WEIGHTS_MAPPER strips mtp. but model.embed_tokens needs separate handling
        mapped_iter = self.WEIGHTS_MAPPER.apply(_transform(weights_iter))
        super(Qwen3NextForCausalLM, self).load_weights(mapped_iter)

        if not has_lm_head and self._cfg["tie_word_embeddings"]:
            self.lm_head.weight.data.copy_(self.embed_tokens.weight.data)

    def __init__(self, model_config: Any, load_config: Any):
        super().__init__(model_config, load_config)
        cfg = self._cfg

        # MTP-specific modules
        from rtp_llm.models_py.layers.linear import ColumnParallelLinear
        from rtp_llm.models_py.layers.norm import RMSNorm

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

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        last_hidden_states = inputs.input_hiddens
        e_norm = self.pre_fc_norm_embedding(inputs_embeds)
        h_norm = self.pre_fc_norm_hidden(last_hidden_states)
        cat_hidden = torch.cat([e_norm, h_norm], -1)
        hidden_states = self.fc(cat_hidden)

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        # Match the legacy MTP loader: draft prefill CUDA graph capture must not
        # allocate causal-conv prefill metadata inside the capture region.
        attn_meta = Qwen3NextMetadata()
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


class Qwen35MoeMTPForCausalLM(Qwen3NextMTPForCausalLM):
    """MTP draft head for Qwen3.5 MoE (prefix mtp., embed from model.language_model.)."""

    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"mtp.": ""})

    def _prepare_config(self, model_config: Any, load_config: Any):
        super()._prepare_config(model_config, load_config)
        self._cfg["merge_format"] = True
        self._cfg["hidden_parallel_embedding"] = True


# ================================================================== #
#  Eagle3 — Qwen3-MoE architecture with Eagle3 speculative draft
# ================================================================== #


class Qwen3MoeEagle3ForCausalLM(GptModelBase):
    """Eagle3 speculative draft on Qwen3-MoE architecture.

    Reuses the standard Qwen3-MoE attention/MLP but adds Eagle3-specific
    weights: fc (projection), hidden_norm, input_norm.
    This is a 1-layer model used for speculative decoding.
    """

    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"model.": ""})

    def load_weights(self, weights):
        if isinstance(weights, dict):
            weights_iter = iter(weights.items())
        else:
            weights_iter = weights
        has_lm_head = False

        def _track(it):
            nonlocal has_lm_head
            for name, tensor in it:
                if name == "lm_head.weight" or name.startswith("lm_head."):
                    has_lm_head = True
                yield name, tensor

        mapped_iter = self.WEIGHTS_MAPPER.apply(_track(weights_iter))
        super().load_weights(mapped_iter)

        if not has_lm_head:
            logger.info("[Qwen3MoeEagle3] lm_head not found; tying to embed_tokens")
            self.lm_head.weight.data.copy_(self.embed_tokens.weight.data)

    def __init__(self, model_config: Any, load_config: Any):
        parallelism_config = getattr(load_config, "parallelism_config", None)
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

        # Reuse Qwen3-MoE config extraction
        from rtp_llm.models_py.new_models.qwen3_moe.language import (
            _extract_moe_config_values,
        )

        cfg = _extract_moe_config_values(model_config, load_config)
        self._cfg = cfg

        # Eagle3 is a 1-layer model
        assert (
            cfg["num_layers"] == 1
        ), f"Eagle3 expects 1 layer, got {cfg['num_layers']}"

        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["hidden_size"],
            tp_size=cfg["tp_size"],
            tp_rank=cfg["tp_rank"],
            params_dtype=cfg["params_dtype"],
        )
        # Single decoder layer (Qwen3-MoE style)
        from rtp_llm.models_py.new_models.qwen3_moe.language import Qwen3MoeDecoderLayer

        self.layers = nn.ModuleList(
            [
                Qwen3MoeDecoderLayer(
                    hidden_size=cfg["hidden_size"],
                    num_heads=cfg["num_heads"],
                    num_kv_heads=cfg["num_kv_heads"],
                    moe_intermediate_size=cfg["moe_intermediate_size"],
                    num_experts=cfg["num_experts"],
                    top_k=cfg["top_k"],
                    head_dim=cfg["head_dim"],
                    layer_idx=0,
                    tp_size=cfg["tp_size"],
                    tp_rank=cfg["tp_rank"],
                    ep_size=cfg["ep_size"],
                    ep_rank=cfg["ep_rank"],
                    model_config=cfg["model_config"],
                    parallelism_config=cfg["parallelism_config"],
                    moe_config=cfg["moe_config"],
                    quant_config=cfg["quant_config"],
                    params_dtype=cfg["params_dtype"],
                    rms_norm_eps=cfg["rms_norm_eps"],
                )
            ]
        )
        self.norm = RMSNorm(
            cfg["hidden_size"],
            eps=cfg["rms_norm_eps"],
            params_dtype=cfg["params_dtype"],
        )
        self.lm_head = ParallelLMHead(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            tp_size=cfg["tp_size"],
            tp_rank=cfg["tp_rank"],
            params_dtype=cfg["lm_head_params_dtype"],
        )

        # Eagle3-specific weights
        self.eagle3_fc = ColumnParallelLinear(
            input_size=cfg["hidden_size"],
            output_size=cfg["hidden_size"],
            tp_size=1,
            tp_rank=0,
            quant_config=None,
            prefix="eagle3_fc",
            bias=False,
            params_dtype=cfg["params_dtype"],
        )
        self.eagle3_fc_norm = RMSNorm(
            cfg["hidden_size"],
            eps=cfg["rms_norm_eps"],
            params_dtype=cfg["params_dtype"],
        )
        self.eagle3_input_norm = RMSNorm(
            cfg["hidden_size"],
            eps=cfg["rms_norm_eps"],
            params_dtype=cfg["params_dtype"],
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids = inputs.input_ids
        hidden_states = self.embed_tokens(input_ids)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)

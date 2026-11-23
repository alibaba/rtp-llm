"""Kimi Linear 48B hybrid KDA/MLA model for the streaming NewLoader.

Checkpoint tensors are dispatched directly into prefix-owned ``RtpModule``
children.  The only derived layout is the small ``W``-named MLA kernel view
required by the existing attention factory; checkpoint loading itself never
uses ``ModelWeightInfo``, ``ModelWeights`` or the legacy loader layout.
"""

from __future__ import annotations

import logging
import math
import weakref
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from rtp_llm.models_py.layers.embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    output_vocab_head_kwargs,
)
from rtp_llm.models_py.layers.norm import RMSResNorm
from rtp_llm.models_py.model_desc.generic_moe import DecodeLayerOutput
from rtp_llm.models_py.model_desc.kimi_linear import (
    KimiLinearKDADecode,
    KimiLinearKDAPrefill,
    KimiLinearMetadata,
    KimiLinearModel,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.deepseek_v3.attention import DeepSeekV32MlaAttention
from rtp_llm.models_py.new_models.deepseek_v3.language import (
    checkpoint_path,
    positive_int,
    read_config_json,
    validate_deepseek_mla_backend,
    validate_deepseek_newloader_eplb,
)
from rtp_llm.models_py.new_models.deepseek_v3.mlp import DeepSeekV32MLP
from rtp_llm.models_py.new_models.deepseek_v3.moe import DeepSeekV32MoEBlock
from .kda_parameters import KimiKDAParameters
from rtp_llm.models_py.weight_mapper import WeightsMapper
from rtp_llm.ops import HybridAttentionType, RopeStyle
from rtp_llm.utils.util import to_torch_dtype

if TYPE_CHECKING:
    from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs

logger = logging.getLogger(__name__)


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {value!r}")
    return value


def _positive_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return float(value)


def _one_based_layer_set(value: Any, name: str, layer_count: int) -> set[int]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    for layer in value:
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or not 1 <= layer <= layer_count
        ):
            raise ValueError(
                f"{name} entries must be integers in [1, {layer_count}], "
                f"got {layer!r}"
            )
    result = set(value)
    if len(result) != len(value):
        raise ValueError(f"{name} must not contain duplicate layers")
    return result


def _partition(size_value: Any, rank: Any, name: str) -> Tuple[int, int]:
    size = positive_int(size_value, f"{name}_size")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < size:
        raise ValueError(f"invalid {name} partition: rank={rank!r}, size={size}")
    return size, rank


def _read_kimi_config(model_config: Any) -> Dict[str, Any]:
    path = checkpoint_path(model_config)
    payload = read_config_json(path)
    if not payload:
        raise FileNotFoundError(
            "Kimi Linear newloader requires checkpoint config.json to preserve "
            "the hybrid KDA/MLA and MoE topology"
        )
    if payload.get("model_type") != "kimi_linear":
        raise ValueError(
            "Kimi Linear newloader expected model_type='kimi_linear', got "
            f"{payload.get('model_type')!r}"
        )
    return payload


def _extract_config_values(
    model_config: Any, load_config: Any, raw: Dict[str, Any]
) -> Dict[str, Any]:
    validate_deepseek_newloader_eplb(model_config, "Kimi Linear")
    attn_config = model_config.attn_config
    use_mla = _strict_bool(attn_config.use_mla, "attn_config.use_mla")
    if not use_mla:
        raise ValueError("Kimi Linear full-attention layers require MLA")
    validate_deepseek_mla_backend(model_config, use_mla, "Kimi Linear")

    hidden_size = positive_int(raw["hidden_size"], "hidden_size")
    checkpoint_layers = positive_int(raw["num_hidden_layers"], "num_hidden_layers")
    num_layers = positive_int(model_config.num_layers, "model_config.num_layers")
    if num_layers > checkpoint_layers:
        raise ValueError(
            f"model_config.num_layers={num_layers} exceeds checkpoint layers="
            f"{checkpoint_layers}"
        )
    vocab_size = positive_int(raw["vocab_size"], "vocab_size")
    dense_intermediate_size = positive_int(
        raw["intermediate_size"], "intermediate_size"
    )
    rms_norm_eps = _positive_float(raw.get("rms_norm_eps", 1e-5), "rms_norm_eps")
    tie_word_embeddings = _strict_bool(
        raw.get("tie_word_embeddings", False), "tie_word_embeddings"
    )

    topology_mismatches = []
    for name, actual, expected in (
        ("hidden_size", model_config.hidden_size, hidden_size),
        ("vocab_size", model_config.vocab_size, vocab_size),
        ("rms_norm_eps", model_config.layernorm_eps, rms_norm_eps),
        (
            "tie_word_embeddings",
            model_config.tie_word_embeddings,
            tie_word_embeddings,
        ),
        ("num_attention_heads", attn_config.head_num, raw["num_attention_heads"]),
        ("kv_lora_rank", attn_config.kv_lora_rank, raw["kv_lora_rank"]),
        ("qk_nope_head_dim", attn_config.nope_head_dim, raw["qk_nope_head_dim"]),
        ("qk_rope_head_dim", attn_config.rope_head_dim, raw["qk_rope_head_dim"]),
        ("v_head_dim", attn_config.v_head_dim, raw["v_head_dim"]),
    ):
        if actual != expected:
            topology_mismatches.append(
                f"{name}=ModelConfig({actual!r})/config.json({expected!r})"
            )
    if topology_mismatches:
        raise ValueError(
            "Kimi Linear topology mismatch: " + ", ".join(topology_mismatches)
        )

    q_lora = raw.get("q_lora_rank")
    if q_lora is not None or attn_config.q_lora_rank != 0:
        raise ValueError("Kimi Linear newloader currently requires q_lora_rank=null")
    if not _strict_bool(raw.get("mla_use_nope", False), "mla_use_nope"):
        raise ValueError("Kimi Linear checkpoint must use MLA NoPE")
    if raw.get("rope_scaling") is not None:
        raise ValueError("Kimi Linear MLA NoPE does not support rope_scaling")
    if attn_config.rope_config.style != RopeStyle.No:
        raise ValueError("Kimi Linear MLA NoPE requires RopeStyle.No")

    linear = raw.get("linear_attn_config")
    if not isinstance(linear, dict):
        raise TypeError("linear_attn_config must be a JSON object")
    kda_head_dim = positive_int(linear["head_dim"], "linear_attn_config.head_dim")
    kda_num_heads = positive_int(linear["num_heads"], "linear_attn_config.num_heads")
    conv_kernel = positive_int(
        linear.get("short_conv_kernel_size", 4),
        "linear_attn_config.short_conv_kernel_size",
    )
    if kda_head_dim != model_config.linear_attention_config.linear_key_head_dim:
        raise ValueError("KDA head_dim differs between config.json and ModelConfig")
    if kda_head_dim != model_config.linear_attention_config.linear_value_head_dim:
        raise ValueError(
            "KDA value head_dim differs between config.json and ModelConfig"
        )
    if kda_num_heads != model_config.linear_attention_config.linear_num_key_heads:
        raise ValueError("KDA num_heads differs between config.json and ModelConfig")
    if kda_num_heads != model_config.linear_attention_config.linear_num_value_heads:
        raise ValueError(
            "KDA value head count differs between config.json and ModelConfig"
        )
    if conv_kernel != model_config.linear_attention_config.linear_conv_kernel_dim:
        raise ValueError(
            "KDA short_conv_kernel_size differs between config.json and ModelConfig"
        )

    kda_layers = _one_based_layer_set(
        linear.get("kda_layers"), "linear_attn_config.kda_layers", checkpoint_layers
    )
    full_layers = _one_based_layer_set(
        linear.get("full_attn_layers"),
        "linear_attn_config.full_attn_layers",
        checkpoint_layers,
    )
    expected_layers = set(range(1, checkpoint_layers + 1))
    if kda_layers & full_layers:
        raise ValueError("KDA and full-attention layer sets must be disjoint")
    if kda_layers | full_layers != expected_layers:
        missing = sorted(expected_layers - (kda_layers | full_layers))
        extra = sorted((kda_layers | full_layers) - expected_layers)
        raise ValueError(
            f"Kimi hybrid layer topology must cover every layer; missing={missing} "
            f"extra={extra}"
        )
    hybrid_types = [
        HybridAttentionType.LINEAR if i + 1 in kda_layers else HybridAttentionType.NONE
        for i in range(num_layers)
    ]
    configured_types = model_config.hybrid_attention_config.hybrid_attention_types
    if len(configured_types) < num_layers:
        raise ValueError("ModelConfig hybrid layer topology is shorter than num_layers")
    configured_hybrid = [configured_types[i] for i in range(num_layers)]
    if configured_hybrid != hybrid_types:
        raise ValueError("Kimi hybrid layer topology differs from ModelConfig")

    num_experts = positive_int(raw["num_experts"], "num_experts")
    top_k = positive_int(raw["num_experts_per_token"], "num_experts_per_token")
    if top_k > num_experts:
        raise ValueError("num_experts_per_token exceeds num_experts")
    moe_intermediate_size = positive_int(
        raw["moe_intermediate_size"], "moe_intermediate_size"
    )
    num_shared_experts = positive_int(
        raw.get("num_shared_experts", 1), "num_shared_experts"
    )
    shared_expert_intermediate_size = num_shared_experts * moe_intermediate_size
    first_k_dense_replace = raw.get("first_k_dense_replace", 1)
    if (
        isinstance(first_k_dense_replace, bool)
        or not isinstance(first_k_dense_replace, int)
        or first_k_dense_replace < 0
    ):
        raise ValueError("first_k_dense_replace must be a non-negative integer")
    moe_layer_freq = positive_int(raw.get("moe_layer_freq", 1), "moe_layer_freq")
    moe_layer_index = [
        i
        for i in range(num_layers)
        if i >= first_k_dense_replace and i % moe_layer_freq == 0
    ]
    if list(model_config.moe_layer_index) != moe_layer_index:
        raise ValueError("Kimi MoE layer topology differs from ModelConfig")
    scoring_name = raw.get("moe_router_activation_func", "sigmoid")
    if scoring_name == "softmax":
        scoring_func = 0
    elif scoring_name == "sigmoid":
        scoring_func = 1
    else:
        raise ValueError(f"unsupported moe_router_activation_func={scoring_name!r}")
    if scoring_func != 1:
        raise ValueError("Kimi correction-bias routing requires sigmoid activation")
    routed_scaling_factor = _positive_float(
        raw["routed_scaling_factor"], "routed_scaling_factor"
    )
    n_group = positive_int(raw.get("num_expert_group", 1), "num_expert_group")
    topk_group = positive_int(raw.get("topk_group", 1), "topk_group")
    if num_experts % n_group:
        raise ValueError("num_experts must be divisible by num_expert_group")
    if topk_group > n_group:
        raise ValueError("topk_group exceeds num_expert_group")
    if top_k > topk_group * (num_experts // n_group):
        raise ValueError("num_experts_per_token exceeds grouped-router capacity")
    has_moe_norm = _strict_bool(raw.get("moe_renormalize", False), "moe_renormalize")

    routing_mismatches = []
    for name, actual, expected in (
        ("expert_num", model_config.expert_num, num_experts),
        ("moe_k", model_config.moe_k, top_k),
        ("moe_inter_size", model_config.moe_inter_size, moe_intermediate_size),
        ("moe_n_group", model_config.moe_n_group, n_group),
        ("moe_topk_group", model_config.moe_topk_group, topk_group),
        ("scoring_func", model_config.scoring_func, scoring_func),
        ("has_moe_norm", model_config.has_moe_norm, has_moe_norm),
    ):
        if actual != expected:
            routing_mismatches.append(
                f"{name}=ModelConfig({actual!r})/config.json({expected!r})"
            )
    if not math.isclose(
        float(model_config.routed_scaling_factor),
        routed_scaling_factor,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        routing_mismatches.append(
            "routed_scaling_factor=ModelConfig"
            f"({model_config.routed_scaling_factor!r})/config.json"
            f"({routed_scaling_factor!r})"
        )
    if routing_mismatches:
        raise ValueError(
            "Kimi routing config mismatch: " + ", ".join(routing_mismatches)
        )

    attn_tp_size, attn_tp_rank = _partition(
        load_config.attn_tp_size, load_config.attn_tp_rank, "attn_tp"
    )
    ffn_tp_size, ffn_tp_rank = _partition(
        load_config.ffn_tp_size, load_config.ffn_tp_rank, "ffn_tp"
    )
    lm_head_tp_size, lm_head_tp_rank = _partition(
        load_config.lm_head_tp_size, load_config.lm_head_tp_rank, "lm_head_tp"
    )
    ep_size, ep_rank = _partition(load_config.ep_size, load_config.ep_rank, "ep")
    if raw["num_attention_heads"] % attn_tp_size:
        raise ValueError("MLA head count must be divisible by attention TP")
    if kda_num_heads % attn_tp_size:
        raise ValueError("KDA head count must be divisible by attention TP")
    if num_experts % ep_size:
        raise ValueError("num_experts must be divisible by EP size")
    if tie_word_embeddings and (
        attn_tp_size != lm_head_tp_size or attn_tp_rank != lm_head_tp_rank
    ):
        raise ValueError("tied embeddings require matching attention/LM-head TP")

    params_dtype = load_config.compute_dtype
    if not isinstance(params_dtype, torch.dtype):
        raise TypeError("compute_dtype must be a torch.dtype")
    linear_runtime = model_config.linear_attention_config
    return {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "checkpoint_layers": checkpoint_layers,
        "vocab_size": vocab_size,
        "dense_intermediate_size": dense_intermediate_size,
        "rms_norm_eps": rms_norm_eps,
        "tie_word_embeddings": tie_word_embeddings,
        "num_heads": positive_int(raw["num_attention_heads"], "num_attention_heads"),
        "q_lora_rank": 0,
        "kv_lora_rank": positive_int(raw["kv_lora_rank"], "kv_lora_rank"),
        "nope_head_dim": positive_int(raw["qk_nope_head_dim"], "qk_nope_head_dim"),
        "rope_head_dim": positive_int(raw["qk_rope_head_dim"], "qk_rope_head_dim"),
        "v_head_dim": positive_int(raw["v_head_dim"], "v_head_dim"),
        "hybrid_types": hybrid_types,
        "kda_head_dim": kda_head_dim,
        "kda_num_heads": kda_num_heads,
        "conv_kernel": conv_kernel,
        "ssm_state_dtype": to_torch_dtype(linear_runtime.ssm_state_dtype),
        "conv_state_dtype": to_torch_dtype(linear_runtime.conv_state_dtype),
        "moe_layer_index": moe_layer_index,
        "num_experts": num_experts,
        "top_k": top_k,
        "moe_intermediate_size": moe_intermediate_size,
        "shared_expert_intermediate_size": shared_expert_intermediate_size,
        "scoring_func": scoring_func,
        "routed_scaling_factor": routed_scaling_factor,
        "n_group": n_group,
        "topk_group": topk_group,
        "has_moe_norm": has_moe_norm,
        "attn_tp_size": attn_tp_size,
        "attn_tp_rank": attn_tp_rank,
        "ffn_tp_size": ffn_tp_size,
        "ffn_tp_rank": ffn_tp_rank,
        "lm_head_tp_size": lm_head_tp_size,
        "lm_head_tp_rank": lm_head_tp_rank,
        "ep_size": ep_size,
        "ep_rank": ep_rank,
        "parallelism_config": load_config.parallelism_config,
        "moe_config": load_config.moe_config,
        "quant_config": load_config.quant_config,
        "params_dtype": params_dtype,
        "lm_head_params_dtype": (
            torch.float32 if model_config.enable_fp32_lm_head else params_dtype
        ),
        "model_config": model_config,
    }


class _OwnedKdaRuntime:
    """Run main's KDA executor against the owner's current buffers, also after .to()."""

    def __init__(self, owner):
        nn.Module.__init__(self)
        self._owner = weakref.ref(owner)
        self.head_k_dim = self.head_v_dim = owner.head_dim
        self.local_num_k_heads = self.local_num_v_heads = owner.local_num_heads
        self.ssm_state_dtype = owner.ssm_state_dtype
        self.linear_cache_converter = owner.cache_converter

    @property
    def conv_weights(self):
        return self._owner().conv1d.weight

    @property
    def dt_bias(self):
        return self._owner().dt_bias

    @property
    def alog(self):
        return self._owner().A_log


class _OwnedPrefill(_OwnedKdaRuntime, KimiLinearKDAPrefill):
    pass


class _OwnedDecode(_OwnedKdaRuntime, KimiLinearKDADecode):
    pass


class KimiLinearKDA(KimiKDAParameters):
    """Dispatch to the same KDA kernels and KV-cache semantics as main."""

    def __init__(self, cfg, prefix):
        super().__init__(cfg, prefix)
        self.prefill_kda = _OwnedPrefill(self)
        self.decode_kda = _OwnedDecode(self)

    def validate_runtime_device(self, device: torch.device) -> None:
        # CPU is a valid checkpoint destination, including for inspection and
        # later migration. Running the recurrent kernels is a separate contract.
        if device.type not in ("cpu", "cuda"):
            raise RuntimeError(f"Kimi Linear KDA has no runtime for {device}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        attn_meta: Optional[KimiLinearMetadata] = None,
    ) -> torch.Tensor:
        del fmha_impl
        if attention_inputs is None or attn_meta is None:
            raise ValueError("Kimi KDA requires attention inputs and metadata")
        if (
            attention_inputs.is_prefill
            and not attention_inputs.is_target_verify
            and attn_meta.prefill_conv1d_meta is None
        ):
            raise ValueError("Kimi KDA prefill requires causal-conv metadata")
        mixed_qkv = self.qkv_proj(hidden_states)
        beta = self.b_proj(hidden_states)
        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states))
        output_gate = self.g_b_proj(self.g_a_proj(hidden_states))
        if attention_inputs.is_prefill and not attention_inputs.is_target_verify:
            output = self.prefill_kda(
                mixed_qkv, forget_gate, beta, attention_inputs, kv_cache, attn_meta
            )
        else:
            output = self.decode_kda(
                mixed_qkv, forget_gate, beta, attention_inputs, kv_cache, attn_meta
            )
        output = self.o_norm(
            output.reshape(-1, self.head_dim),
            output_gate.reshape(-1, self.head_dim),
        )
        output = output.reshape(-1, self.local_num_heads * self.head_dim)
        return self.o_proj(output)


class KimiLinearDecoderLayer(RtpModule):
    def __init__(self, cfg: Dict[str, Any], layer_idx: int, prefix: str) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = cfg["hybrid_types"][layer_idx]
        self.is_linear = self.layer_type == HybridAttentionType.LINEAR
        if self.is_linear:
            self.self_attn = KimiLinearKDA(cfg, prefix=f"{prefix}.self_attn")
        else:
            self.self_attn = DeepSeekV32MlaAttention(
                hidden_size=cfg["hidden_size"],
                num_heads=cfg["num_heads"],
                q_lora_rank=cfg["q_lora_rank"],
                kv_lora_rank=cfg["kv_lora_rank"],
                nope_head_dim=cfg["nope_head_dim"],
                rope_head_dim=cfg["rope_head_dim"],
                v_head_dim=cfg["v_head_dim"],
                layer_idx=layer_idx,
                tp_size=cfg["attn_tp_size"],
                tp_rank=cfg["attn_tp_rank"],
                quant_config=cfg["quant_config"],
                params_dtype=cfg["params_dtype"],
                layernorm_eps=cfg["rms_norm_eps"],
                prefix=f"{prefix}.self_attn",
            )
        if layer_idx in cfg["moe_layer_index"]:
            self.block_sparse_moe = DeepSeekV32MoEBlock(
                hidden_size=cfg["hidden_size"],
                moe_intermediate_size=cfg["moe_intermediate_size"],
                num_experts=cfg["num_experts"],
                top_k=cfg["top_k"],
                layer_idx=layer_idx,
                tp_size=cfg["ffn_tp_size"],
                tp_rank=cfg["ffn_tp_rank"],
                ep_size=cfg["ep_size"],
                ep_rank=cfg["ep_rank"],
                model_config=cfg["model_config"],
                parallelism_config=cfg["parallelism_config"],
                moe_config=cfg["moe_config"],
                quant_config=cfg["quant_config"],
                params_dtype=cfg["params_dtype"],
                has_shared_expert=True,
                shared_expert_intermediate_size=cfg["shared_expert_intermediate_size"],
                scoring_func=cfg["scoring_func"],
                routed_scaling_factor=cfg["routed_scaling_factor"],
                n_group=cfg["n_group"],
                topk_group=cfg["topk_group"],
                topk_method="noaux_tc",
                has_moe_norm=cfg["has_moe_norm"],
                correction_bias=True,
                prefix=f"{prefix}.block_sparse_moe",
            )
            self.mlp = None
        else:
            self.mlp = DeepSeekV32MLP(
                hidden_size=cfg["hidden_size"],
                intermediate_size=cfg["dense_intermediate_size"],
                tp_size=cfg["ffn_tp_size"],
                tp_rank=cfg["ffn_tp_rank"],
                quant_config=cfg["quant_config"],
                params_dtype=cfg["params_dtype"],
                reduce_output=True,
                prefix=f"{prefix}.mlp",
            )
            self.block_sparse_moe = None
        self.input_layernorm = RMSResNorm(
            cfg["hidden_size"],
            eps=cfg["rms_norm_eps"],
            params_dtype=cfg["params_dtype"],
        )
        self.post_attention_layernorm = RMSResNorm(
            cfg["hidden_size"],
            eps=cfg["rms_norm_eps"],
            params_dtype=cfg["params_dtype"],
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache],
        attention_inputs: PyAttentionInputs,
        attn_meta: KimiLinearMetadata,
    ) -> DecodeLayerOutput:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.is_linear:
            hidden_states = self.self_attn(
                hidden_states,
                fmha_impl,
                kv_cache,
                attention_inputs,
                attn_meta,
            )
        else:
            hidden_states = self.self_attn(hidden_states, fmha_impl, kv_cache)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        ffn = self.block_sparse_moe if self.block_sparse_moe is not None else self.mlp
        if ffn is None:
            raise RuntimeError(f"Kimi layer {self.layer_idx} has no FFN")
        return DecodeLayerOutput(ffn(hidden_states), residual)


class _KimiMlaKernelWeightLayout:
    def __init__(self, weights: list[Dict[str, torch.Tensor]]) -> None:
        self.weights = weights

    def get_global_weight_or_none(self, name: str) -> Optional[torch.Tensor]:
        del name
        return None


class KimiLinearForCausalLM(KimiLinearModel, RtpModule):
    """Clean prefix-streamed Kimi Linear NewLoader implementation."""

    # Kimi names routed-expert projections w1/w3/w2 while the shared
    # NewLoader MoE leaf owns the semantic gate/up/down projection names.
    # Keep this as a name-only checkpoint mapping; no tensor layout is
    # transformed here.
    WEIGHTS_MAPPER = WeightsMapper(
        regex_mapping=[
            (
                r"^model\.(layers\.\d+\.block_sparse_moe\.experts\.\d+)\.w1(\..+)$",
                r"\1.gate_proj\2",
            ),
            (
                r"^model\.(layers\.\d+\.block_sparse_moe\.experts\.\d+)\.w3(\..+)$",
                r"\1.up_proj\2",
            ),
            (
                r"^model\.(layers\.\d+\.block_sparse_moe\.experts\.\d+)\.w2(\..+)$",
                r"\1.down_proj\2",
            ),
            (r"^model\.", ""),
        ]
    )

    @staticmethod
    def _checkpoint_layer_index(name: str) -> Optional[int]:
        prefix = "model.layers."
        if not name.startswith(prefix):
            return None
        value, separator, _ = name[len(prefix) :].partition(".")
        return int(value) if separator and value.isdigit() else None

    def checkpoint_weight_name_filter(self) -> Callable[[str], bool]:
        num_layers = len(self.layers)
        checkpoint_layers = self._checkpoint_num_layers

        def should_load(name: str) -> bool:
            if name.startswith("model.layers."):
                index = self._checkpoint_layer_index(name)
                if index is None or index < num_layers:
                    return True
                # A layer that exists in config.json but was removed by a
                # model_config layer override is intentionally skipped.  A
                # tensor beyond the checkpoint topology remains visible so
                # the normal unknown-weight validation rejects it.
                return index >= checkpoint_layers
            # Only a valid, intentionally truncated tail layer may be hidden
            # from the model tree.  Keep every other name visible so typoed or
            # newly introduced checkpoint tensors reach the normal strict
            # unknown-weight validation instead of being silently discarded.
            return True

        return should_load

    def __init__(self, model_config: Any, load_config: Any) -> None:
        raw = _read_kimi_config(model_config)
        cfg = _extract_config_values(model_config, load_config, raw)
        GptModelBase.__init__(
            self,
            config=model_config,
            parallelism_config=cfg["parallelism_config"],
            weight=None,
            max_generate_batch_size=load_config.max_generate_batch_size,
            fmha_config=load_config.fmha_config,
            py_hw_kernel_config=load_config.py_hw_kernel_config,
        )
        self._init_capture_context(self._capture_canonical_layer, self._capture_canonical_final)
        self._cfg = cfg
        self._checkpoint_num_layers = cfg["checkpoint_layers"]
        self.tie_word_embeddings = cfg["tie_word_embeddings"]
        self._mla_kernel_layout: Optional[_KimiMlaKernelWeightLayout] = None
        self._keep_mla_checkpoint_weights = load_config.keep_mla_checkpoint_weights
        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["hidden_size"],
            tp_size=cfg["attn_tp_size"],
            tp_rank=cfg["attn_tp_rank"],
            params_dtype=cfg["params_dtype"],
            allow_vocab_padding=True,
        )
        self.layers = nn.ModuleList(
            [
                KimiLinearDecoderLayer(cfg, i, prefix=f"layers.{i}")
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
            **output_vocab_head_kwargs(model_config),
        )
        if self.tie_word_embeddings:
            self.embed_tokens.tie_output_head(self.lm_head)

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse)
        if self._mla_kernel_layout is not None:
            # The kernel layout is a non-Module view. Refresh its tensor
            # references after a model-wide device or dtype migration.
            self._mla_kernel_layout = None
            self._ensure_mla_kernel_layout()
            self.weight = self._mla_kernel_layout
        return result

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        iterator = weights.items() if isinstance(weights, dict) else weights
        has_lm_head = False

        def track(items):
            nonlocal has_lm_head
            for name, tensor in items:
                if name == "lm_head.weight":
                    has_lm_head = True
                yield name, tensor

        super().load_weights(track(self.WEIGHTS_MAPPER.apply(iterator)))
        if not has_lm_head and self.tie_word_embeddings:
            self.lm_head._copy_local_tied_weight(self.embed_tokens.weight.data)

    def load_ft_checkpoint(self, checkpoint):
        from .ft_checkpoint import load_kimi_linear_ft

        load_kimi_linear_ft(self, checkpoint)

    def runtime_weight_view(self) -> Dict[str, torch.Tensor]:
        return {
            "embedding": self.embed_tokens.weight,
            "final_layernorm.gamma": self.norm.weight,
            "lm_head": self.lm_head.weight,
        }

    def _ensure_mla_kernel_layout(self) -> None:
        if self._mla_kernel_layout is not None:
            return
        weights = []
        for layer in self.layers:
            if layer.is_linear:
                weights.append({})
            else:
                weights.append(layer.self_attn._build_mla_kernel_weights())
        self._mla_kernel_layout = _KimiMlaKernelWeightLayout(weights)

    def initialize(self, init_resource):
        ok = super().initialize(init_resource)
        if not ok:
            return ok
        for layer in self.layers:
            if layer.is_linear:
                device = self.embed_tokens.weight.device
                if device.type == "cpu":
                    raise RuntimeError("Kimi Linear KDA execution requires CUDA")
                layer.self_attn.validate_runtime_device(device)
        self._ensure_mla_kernel_layout()
        if self._keep_mla_checkpoint_weights:
            logger.info(
                "Keeping Kimi MLA checkpoint-only weights for debugging; "
                "GPU memory usage will be higher"
            )
        else:
            for layer in self.layers:
                if not layer.is_linear:
                    layer.self_attn.release_checkpoint_only_weights()
        return ok

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False, cuda_graph_selection_mode=None):
        self._ensure_mla_kernel_layout()
        self.weight = self._mla_kernel_layout
        return super().prepare_fmha_impl(inputs, is_cuda_graph, cuda_graph_selection_mode)

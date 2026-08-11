"""
DeepSeek V3 MTP (Multi-Token Prediction) draft head for new-loader.

Top-level model: DeepSeekV32MTPForCausalLM
  - Standalone draft checkpoints use "model.layers.0.".
  - Full checkpoints use the appended "model.layers.{num_hidden_layers}.".
  - load_weights() strips the selected draft prefix and remaps keys:
      embed_tokens.weight        -> embed_tokens.weight
      shared_head.head.weight    -> lm_head.weight
      shared_head.norm.weight    -> norm.weight
      enorm.weight               -> mtp_block.e_norm.weight
      hnorm.weight               -> mtp_block.h_norm.weight
      eh_proj.weight             -> mtp_block.fc.weight
      self_attn.*                -> layers.0.self_attn.*
      mlp.*                      -> layers.0.mlp.*
      input_layernorm.*          -> layers.0.input_layernorm.*
      post_attention_layernorm.* -> layers.0.post_attention_layernorm.*
"""

from collections.abc import Callable
from typing import Any, Dict

import torch
import torch.nn as nn

from rtp_llm.models_py.layers.embedding import ParallelLMHead, VocabParallelEmbedding
from rtp_llm.models_py.layers.norm import RMSResNorm
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.new_models.deepseek_v3.language import (
    MlaRuntimeLayoutMixin,
    build_rope_cache,
    checkpoint_path,
    extract_config_values,
    nonnegative_int,
    positive_int,
    read_config_json,
)
from rtp_llm.models_py.new_models.deepseek_v3.model import DeepSeekV32DecoderLayer
from rtp_llm.models_py.new_models.model_base import select_block_map_for_layer
from rtp_llm.models_py.new_models.mtp import MTPBlock
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs

# ------------------------------------------------------------------ #
#  Key remapping helpers
# ------------------------------------------------------------------ #

# After the selected draft-layer prefix is stripped, these are the remaining
# key prefixes / full names that need to be renamed to match submodule paths.
_LAYER_PREFIXES = (
    "self_attn.",
    "mlp.",
    "input_layernorm.",
    "post_attention_layernorm.",
)

_KEY_REMAP = {
    "shared_head.head.weight": "lm_head.weight",
    "shared_head.norm.weight": "norm.weight",
    "enorm.weight": "mtp_block.e_norm.weight",
    "hnorm.weight": "mtp_block.h_norm.weight",
}

_EH_PROJ_KEY = "eh_proj.weight"
_EH_PROJ_MAPPED = "mtp_block.fc.weight"


def _remap_key(name: str) -> str:
    """Remap a stripped HF key to the submodule key used in this model."""
    if name == _EH_PROJ_KEY:
        return _EH_PROJ_MAPPED

    # simple renames
    if name in _KEY_REMAP:
        return _KEY_REMAP[name]

    # layer-level weights: prepend "layers.0."
    for prefix in _LAYER_PREFIXES:
        if name.startswith(prefix):
            return "layers.0." + name

    # embed_tokens.weight and anything else: pass through unchanged
    return name


def _draft_checkpoint_layer(config_json: Dict[str, Any]) -> int:
    """Resolve the one draft layer without guessing from checkpoint order."""
    architectures = config_json.get("architectures", [])
    if not isinstance(architectures, list) or not all(
        isinstance(name, str) for name in architectures
    ):
        raise TypeError("config.json architectures must be a list of strings")
    nextn_layers = nonnegative_int(
        config_json.get("num_nextn_predict_layers", 0),
        "num_nextn_predict_layers",
    )
    if any(name.endswith("ForCausalLMNextN") for name in architectures):
        if nextn_layers > 1:
            raise ValueError(
                "DeepSeek standalone MTP checkpoints must contain at most "
                f"one draft layer, got num_nextn_predict_layers={nextn_layers}"
            )
        return 0

    if nextn_layers != 1:
        raise ValueError(
            "DeepSeek full-model checkpoints must contain exactly one "
            "appended MTP "
            f"layer, got num_nextn_predict_layers={nextn_layers}"
        )
    return positive_int(config_json.get("num_hidden_layers"), "num_hidden_layers")


# ------------------------------------------------------------------ #
#  Top-level model
# ------------------------------------------------------------------ #


class DeepSeekV32MTPForCausalLM(MlaRuntimeLayoutMixin, GptModelBase):
    """DeepSeek V3 MTP draft head for new-loader.

    Single-layer MoE decoder + MTPBlock projection.
    """

    def checkpoint_weight_name_filter(self) -> Callable[[str], bool]:
        prefix = self._checkpoint_prefix
        return lambda name: name.startswith(prefix) and len(name) > len(prefix)

    def load_weights(self, weights):
        if isinstance(weights, dict):
            weights_iter = iter(weights.items())
        else:
            weights_iter = weights

        def _mapped(it):
            for name, tensor in it:
                if not name.startswith(self._checkpoint_prefix):
                    raise RuntimeError(
                        "DeepSeek MTP received a non-draft checkpoint tensor: "
                        f"{name!r}"
                    )
                stripped_name = name[len(self._checkpoint_prefix) :]
                yield _remap_key(stripped_name), tensor

        super().load_weights(_mapped(weights_iter))

    def __init__(
        self,
        model_config: Any,
        load_config: Any,
    ):
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
        self._keep_mla_checkpoint_weights = load_config.keep_mla_checkpoint_weights
        self._mla_kernel_layout = None

        ckpt_path = checkpoint_path(model_config)

        config_json = read_config_json(ckpt_path)
        if not config_json:
            raise FileNotFoundError(
                "DeepSeek MTP newloader requires checkpoint config.json"
            )
        cfg = extract_config_values(model_config, load_config, config_json)
        self._checkpoint_layer = _draft_checkpoint_layer(config_json)
        self._checkpoint_prefix = f"model.layers.{self._checkpoint_layer}."

        # The engine builds the draft ModelConfig from the full checkpoint, so
        # model_config.num_layers still describes all main-model layers.  This
        # class represents exactly the single appended MTP layer selected above.
        cfg["num_layers"] = 1
        self.layer_num = 1
        if cfg["num_experts"] <= 0 or not 0 < cfg["top_k"] <= cfg["num_experts"]:
            raise ValueError(
                "DeepSeek MTP requires routed experts with "
                f"top_k={cfg['top_k']} and num_experts={cfg['num_experts']}"
            )
        if cfg["num_experts"] % cfg["ep_size"]:
            raise ValueError(
                f"num_experts={cfg['num_experts']} must be divisible by "
                f"ep_size={cfg['ep_size']}"
            )

        # The single draft layer always uses the routed/shared MoE path.
        cfg["moe_layer_index"] = [0]

        # --- RoPE cache ---
        device = torch.device(getattr(load_config, "device", "cuda"))
        cos_sin_cache = build_rope_cache(
            config_json if config_json else cfg,
            cfg["max_seq_len"],
            device,
        )
        self.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)

        # --- Embedding ---
        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["hidden_size"],
            tp_size=cfg["attn_tp_size"],
            tp_rank=cfg["attn_tp_rank"],
            params_dtype=cfg["params_dtype"],
        )

        # --- MTP projection block (reverse_concat=True for DeepSeek style) ---
        # HF ckpt names: enorm.weight -> e_norm, hnorm.weight -> h_norm,
        # eh_proj.weight -> fc (mapped in load_weights above)
        self.mtp_block = MTPBlock(
            hidden_size=cfg["hidden_size"],
            rms_norm_eps=cfg["rms_norm_eps"],
            reverse_concat=True,
            bias=False,
            params_dtype=cfg["params_dtype"],
            prefix="mtp_block",
        )

        # --- Single MoE decoder layer ---
        self.layers = nn.ModuleList()
        layer = DeepSeekV32DecoderLayer(
            hidden_size=cfg["hidden_size"],
            num_heads=cfg["num_heads"],
            q_lora_rank=cfg["q_lora_rank"],
            kv_lora_rank=cfg["kv_lora_rank"],
            nope_head_dim=cfg["nope_head_dim"],
            rope_head_dim=cfg["rope_head_dim"],
            v_head_dim=cfg["v_head_dim"],
            layer_idx=0,
            attn_tp_size=cfg["attn_tp_size"],
            attn_tp_rank=cfg["attn_tp_rank"],
            ffn_tp_size=cfg["ffn_tp_size"],
            ffn_tp_rank=cfg["ffn_tp_rank"],
            ep_size=cfg["ep_size"],
            ep_rank=cfg["ep_rank"],
            params_dtype=cfg["params_dtype"],
            layernorm_eps=cfg["rms_norm_eps"],
            quant_config=cfg["quant_config"],
            model_config=cfg["model_config"],
            parallelism_config=cfg["parallelism_config"],
            moe_config=cfg["moe_config"],
            is_moe_layer=True,
            dense_intermediate_size=cfg["dense_intermediate_size"],
            moe_intermediate_size=cfg["moe_intermediate_size"],
            num_experts=cfg["num_experts"],
            top_k=cfg["top_k"],
            shared_expert_intermediate_size=cfg["shared_expert_intermediate_size"],
            has_shared_expert=cfg["shared_expert_intermediate_size"] > 0,
            scoring_func=cfg["scoring_func"],
            routed_scaling_factor=cfg["routed_scaling_factor"],
            n_group=cfg["n_group"],
            topk_group=cfg["topk_group"],
            topk_method=cfg["topk_method"],
            has_moe_norm=cfg["has_moe_norm"],
            correction_bias=cfg["has_e_score_correction"],
            is_sparse=cfg["is_sparse"],
            index_n_heads=cfg["indexer_head_num"],
            index_head_dim=cfg["indexer_head_dim"],
            index_topk=cfg["indexer_topk"],
            indexer_is_neox_style=cfg["indexer_is_neox_style"],
            cos_sin_cache=cos_sin_cache,
            blocksize=cfg["blocksize"],
            prefix="layers.0",
        )
        self.layers.append(layer)

        # --- Final norm (from shared_head.norm) ---
        self.norm = RMSResNorm(
            cfg["hidden_size"],
            eps=cfg["rms_norm_eps"],
            params_dtype=cfg["params_dtype"],
        )

        # --- LM head (from shared_head.head) ---
        self.lm_head = ParallelLMHead(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            tp_size=cfg["lm_head_tp_size"],
            tp_rank=cfg["lm_head_tp_rank"],
            params_dtype=cfg["lm_head_params_dtype"],
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        if inputs.input_hiddens is None:
            raise ValueError("DeepSeek MTP requires input_hiddens")
        if inputs.attention_inputs is None:
            raise ValueError("DeepSeek MTP requires attention_inputs")
        # inputs_embeds from current token ids
        inputs_embeds = self.embed_tokens(inputs.input_ids)
        if inputs.input_hiddens.shape != inputs_embeds.shape:
            raise ValueError(
                "DeepSeek MTP input_hiddens must match token embeddings: "
                f"{tuple(inputs.input_hiddens.shape)} != {tuple(inputs_embeds.shape)}"
            )
        # MTP block: combine embed with last hidden states from the main model
        hidden_states = self.mtp_block(inputs_embeds, inputs.input_hiddens)
        residual = torch.zeros_like(hidden_states)

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        for i, layer in enumerate(self.layers):
            select_block_map_for_layer(inputs.attention_inputs, i)
            hidden_states, residual = layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return PyModelOutputs(hidden_states)

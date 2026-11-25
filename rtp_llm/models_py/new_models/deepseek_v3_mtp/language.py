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

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from rtp_llm.ops.compute_ops import PyModelInputs

import torch
import torch.nn as nn

from rtp_llm.models_py.layers.embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    output_vocab_head_kwargs,
)
from rtp_llm.models_py.layers.norm import RMSResNorm
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.module_base import RtpModule, _is_allowed_dropped_weight
from rtp_llm.models_py.new_models.deepseek_v3.language import (
    DeepSeekV32ForCausalLM,
    build_rope_cache,
    checkpoint_path,
    extract_config_values,
    nonnegative_int,
    positive_int,
    read_config_json,
)
from rtp_llm.models_py.weight_mapper import WeightsMapper

from .mtp_block import MTPBlock

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
    "shared_head.norm.weight": "norm.weight",
    "enorm.weight": "mtp_block.e_norm.weight",
    "hnorm.weight": "mtp_block.h_norm.weight",
}

_EH_PROJ_KEY = "eh_proj.weight"
_EH_PROJ_MAPPED = "mtp_block.fc.weight"


def _draft_checkpoint_layer(config_json: Dict[str, Any], ckpt_path: str) -> int:
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

    depth = positive_int(config_json.get("num_hidden_layers"), "num_hidden_layers")
    # Older one-layer drafts are served by explicit model type and do not
    # necessarily advertise the NextN architecture (or a nonzero nextn
    # count). A declared appended layer on a one-layer score is ambiguous:
    # distinguish the two using weight names, never weight values or order.
    if depth == 1 and nextn_layers == 0:
        return 0
    if depth == 1 and nextn_layers == 1:
        from rtp_llm.models_py.weight_mapper import (
            discover_ckpt_files,
            safetensor_index_entries,
        )

        def markers(names):
            return {i for i in (0, 1) if f"model.layers.{i}.eh_proj.weight" in names}

        entries = safetensor_index_entries(ckpt_path)
        if entries is not None:
            found = markers({name for name, _ in entries})
        else:
            from safetensors import safe_open

            found = set()
            for filename in discover_ckpt_files(ckpt_path):
                if filename.endswith(".safetensors"):
                    with safe_open(filename, framework="pt", device="cpu") as handle:
                        found.update(markers(set(handle.keys())))
                else:
                    # OldLoader's one-layer unindexed PyTorch draft is at
                    # layer 0. Inspect metadata on meta to also distinguish a
                    # one-layer score bundled with its draft, without a CPU
                    # allocation of the actual weights.
                    state = torch.load(filename, map_location="meta", weights_only=True)
                    found.update(markers(state))
        if len(found) > 1:
            raise ValueError("ambiguous NextN checkpoint contains two EH projections")
        if found:
            return found.pop()
    if nextn_layers != 1:
        raise ValueError(
            "DeepSeek full-model checkpoints must contain exactly one "
            "appended MTP "
            f"layer, got num_nextn_predict_layers={nextn_layers}"
        )
    return depth


# ------------------------------------------------------------------ #
#  Top-level model
# ------------------------------------------------------------------ #


class DeepSeekV32MTPForCausalLM(DeepSeekV32ForCausalLM):
    """DeepSeek V3 MTP draft head for new-loader.

    Single-layer MoE decoder + MTPBlock projection.
    """

    draft_checkpoint_layer = staticmethod(_draft_checkpoint_layer)
    reverse_mtp_concat = True

    def _vocab_mapping(self, prefix):
        return {
            prefix + "embed_tokens.weight": "embed_tokens.weight",
            prefix + "shared_head.head.weight": "lm_head.weight",
        }

    def checkpoint_to_runtime_expert_layer(self, index: int) -> int:
        if index != self._checkpoint_layer:
            raise ValueError(f"not a NextN expert layer: {index}")
        return 0

    def checkpoint_weight_name_filter(self) -> Callable[[str], bool]:
        prefix = self._checkpoint_prefix
        combined = self._checkpoint_layer != 0

        def keep(name):
            if name in self._vocab_keys:
                return True
            if name.startswith(prefix):
                return True
            if combined:
                if name in (
                    "model.embed_tokens.weight",
                    "model.norm.weight",
                    "lm_head.weight",
                ):
                    return False
                layer_idx = DeepSeekV32ForCausalLM._checkpoint_layer_index(name)
                if layer_idx is not None and layer_idx < self._checkpoint_layer:
                    return False
            # Do not silently drop misspelled or unexpected draft tensors.
            return True

        return keep

    def load_weights(self, weights):
        items = weights.items() if isinstance(weights, Mapping) else weights

        def selected():
            for name, tensor in items:
                if _is_allowed_dropped_weight(name):
                    continue
                if name not in self._vocab_keys and not name.startswith(self._checkpoint_prefix):
                    raise ValueError(f"unexpected NextN checkpoint tensor: {name!r}")
                yield name, tensor

        RtpModule.load_weights(self, self.WEIGHTS_MAPPER.apply(selected()))

    def __init__(
        self,
        model_config: Any,
        load_config: Any,
    ):
        parallelism_config = getattr(load_config, "parallelism_config", None)
        fmha_config = getattr(load_config, "fmha_config", None)

        GptModelBase.__init__(
            self,
            config=model_config,
            parallelism_config=parallelism_config,
            weight=None,
            max_generate_batch_size=load_config.max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=load_config.py_hw_kernel_config,
        )
        self._init_capture_context(
            self._capture_canonical_layer, self._capture_canonical_final
        )
        self._mla_kernel_layout = None

        ckpt_path = checkpoint_path(model_config)

        config_json = read_config_json(ckpt_path)
        if not config_json:
            raise FileNotFoundError(
                "DeepSeek MTP newloader requires checkpoint config.json"
            )
        cfg = extract_config_values(model_config, load_config, config_json)
        self._checkpoint_layer = self.draft_checkpoint_layer(config_json, ckpt_path)
        self._checkpoint_prefix = f"model.layers.{self._checkpoint_layer}."

        prefix = self._checkpoint_prefix
        vocab = self._vocab_mapping(prefix)
        self._vocab_keys = frozenset(vocab)
        self.WEIGHTS_MAPPER = WeightsMapper(
            exact_mapping={
                **{prefix + name: dest for name, dest in _KEY_REMAP.items()},
                prefix + _EH_PROJ_KEY: _EH_PROJ_MAPPED,
                **vocab,
            },
            prefix_mapping={
                prefix + name: "layers.0." + name for name in _LAYER_PREFIXES
            },
        )
        # The raw file's layer count locates the source; the runtime always
        # owns one block, including when the score has a different depth.
        self.layer_num = 1
        if cfg["num_experts"] <= 0 or not 0 < cfg["top_k"] <= cfg["num_experts"]:
            raise ValueError(
                "DeepSeek MTP requires routed experts with "
                f"top_k={cfg['top_k']} and num_experts={cfg['num_experts']}"
            )


        # The single draft layer always uses the routed/shared MoE path.
        cfg["moe_layer_index"] = [0]

        # --- RoPE cache ---
        device = torch.device(getattr(load_config, "device", "cuda"))
        cos_sin_cache = build_rope_cache(
            config_json,
            cfg["max_seq_len"],
            device,
            model_config=model_config,
        )
        self.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)

        # --- Embedding ---
        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["hidden_size"],
            tp_size=cfg["attn_tp_size"],
            tp_rank=cfg["attn_tp_rank"],
            params_dtype=cfg["params_dtype"],
            allow_vocab_padding=True,
        )

        # --- MTP projection block (reverse_concat=True for DeepSeek style) ---
        # HF ckpt names: enorm.weight -> e_norm, hnorm.weight -> h_norm,
        # eh_proj.weight -> fc (mapped in load_weights above)
        self.mtp_block = MTPBlock(
            hidden_size=cfg["hidden_size"],
            rms_norm_eps=cfg["rms_norm_eps"],
            reverse_concat=self.reverse_mtp_concat,
            bias=False,
            params_dtype=cfg["params_dtype"],
            prefix="mtp_block",
        )

        # --- Single MoE decoder layer ---
        self.layers = nn.ModuleList()
        layer = self.decoder_layer_cls(
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
            hw_kernel_config=load_config.py_hw_kernel_config,
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
            **output_vocab_head_kwargs(model_config),
        )

    def embedding(self, inputs: PyModelInputs) -> torch.Tensor:
        if inputs.input_hiddens is None:
            raise ValueError("DeepSeek NextN requires target hidden states")
        return self.mtp_block(self.embed_tokens(inputs.input_ids), inputs.input_hiddens)

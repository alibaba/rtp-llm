"""Qwen3 DSpARK parameters loaded directly into the inference module tree."""

import torch

from rtp_llm.models_py.layers.linear import ColumnParallelLinear
from rtp_llm.models_py.layers.norm import RMSNorm
from rtp_llm.models_py.model_desc.qwen3_dspark_runtime import Qwen3DSparkRuntime
from rtp_llm.models_py.module_base import _mark_loaded
from rtp_llm.models_py.new_models.qwen3.language import (
    Qwen3ForCausalLM,
    _extract_config_values,
)
from rtp_llm.models_py.new_models.qwen3_dspark_weights import Qwen3DSparkMarkovHead
from rtp_llm.models_py.read_spec import unwrap_checkpoint_tensor
from rtp_llm.models_py.speculative.qwen3_dspark_checkpoint import (
    absolute_d2t,
    is_inference_weight,
)
from rtp_llm.models_py.weight_mapper import WeightsMapper


class Qwen3DSparkForCausalLM(Qwen3DSparkRuntime, Qwen3ForCausalLM):
    def load_ft_checkpoint(self, checkpoint):
        from .ft_checkpoint import load_dspark_ft_checkpoint

        load_dspark_ft_checkpoint(self, checkpoint)

    runtime_parameter_paths = {
        "embedding": "embed_tokens.weight",
        "lm_head": "lm_head.weight",
    }
    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"model.": ""})

    @staticmethod
    def _dense_config(model_config, load_config):
        cfg = _extract_config_values(model_config, load_config)
        value = (
            model_config.get
            if isinstance(model_config, dict)
            else lambda name, default=None: getattr(model_config, name, default)
        )
        cfg["input_vocab_size"] = value("input_vocab_size") or cfg["vocab_size"]
        if cfg["input_vocab_size"] != cfg["vocab_size"] and value(
            "tie_word_embeddings", False
        ):
            raise ValueError(
                "DSpARK distinct input/output vocabularies cannot tie embedding and head"
            )
        return cfg

    def __init__(self, model_config, load_config):
        quant = getattr(load_config, "quant_config", None)
        if quant is not None and quant.quant_type != "none":
            raise ValueError("Qwen3 DSpARK requires unquantized QKV context weights")
        if getattr(model_config, "enable_output_vocab_pruning", False):
            raise ValueError(
                "DSpARK draft-local d2t vocabulary cannot use target output_vocab_ids pruning"
            )
        # The PP executor runs the full draft on the last target stage.
        # ModelFactory marks draft configs before loading; local_layer_range
        # then ignores the target's stage counts rather than slicing the draft.
        if getattr(load_config.parallelism_config, "pp_size", 1) > 1 and not getattr(
            model_config, "is_mtp", False
        ):
            raise ValueError("Qwen3 DSpARK on a pipeline requires a draft configuration")
        super().__init__(model_config, load_config)
        cfg = _extract_config_values(model_config, load_config)
        value = lambda name, default=None: (
            model_config.get(name, default)
            if isinstance(model_config, dict)
            else getattr(model_config, name, default)
        )
        input_vocab = value("input_vocab_size") or cfg["vocab_size"]
        target_layers = value("dspark_target_layer_ids")
        if (
            not isinstance(target_layers, (list, tuple))
            or not target_layers
            or any(type(i) is not int or i < 0 for i in target_layers)
        ):
            raise ValueError("DSpARK requires non-negative target layer ids")
        width = value("gen_num_per_cycle")
        self.init_dspark_proposer(
            width=width,
            query_width=width + int(not value("dspark_sample_from_anchor", True)),
            noise_token_id=value("dspark_noise_token_id"),
            aux_feature_dim=len(target_layers) * cfg["hidden_size"],
            hidden_dim=cfg["hidden_size"],
        )
        if self._dspark_noise_token_id >= input_vocab:
            raise ValueError("DSpARK mask token is outside the input vocabulary")
        self.attn_configs = model_config.getAttentionConfigs(cfg["attn_tp_size"])
        if self.attn_configs.is_causal:
            raise ValueError("Qwen3 DSpARK proposal attention must be non-causal")
        self.fc = ColumnParallelLinear(
            self._dspark_aux_feature_dim,
            cfg["hidden_size"],
            tp_size=1,
            tp_rank=0,
            params_dtype=cfg["params_dtype"],
            prefix="fc",
            bias=False,
        )
        self.hidden_norm = RMSNorm(
            cfg["hidden_size"],
            eps=cfg["rms_norm_eps"],
            params_dtype=cfg["params_dtype"],
        )
        self.markov_head = Qwen3DSparkMarkovHead(
            input_vocab, cfg["vocab_size"], value("dspark_markov_rank")
        )
        self.context_rope = None
        self.register_buffer("d2t", None)
        self._input_vocab_size = input_vocab

    def checkpoint_weight_name_filter(self):
        # The draft owns every layer even on a target pipeline. Its optional
        # training-only confidence head has no consumer in the proposer.
        return is_inference_weight

    def prepare_borrowed_parameters(self, aliases):
        if set(aliases) - {"embed_tokens.weight", "lm_head.weight"}:
            raise ValueError("Qwen3 DSpARK only borrows target embedding/head")
        if self.tie_word_embeddings:
            raise ValueError(
                "Borrowed DSpARK globals require explicit untied destinations"
            )
        for destination, binding in aliases.items():
            leaf = self.get_submodule(destination.rsplit(".", 1)[0])
            owner_leaf = binding.owner.get_submodule(
                binding.source_path.rsplit(".", 1)[0]
            )
            if (leaf.tp_size, leaf.tp_rank) != (owner_leaf.tp_size, owner_leaf.tp_rank):
                raise ValueError(
                    "DSpARK parameter loan requires the same local TP partition"
                )
        if "lm_head.weight" in aliases:
            # The target head has already undergone its final normalization and
            # scale. Borrowing is an identity operation, never a second transform.
            if self.normalize_lm_head_weight or self.logit_scale != 1.0:
                raise ValueError(
                    "Borrowed DSpARK head must not request another postprocess"
                )
            self._lm_head_postprocessed = True

    def process_weights_after_loading(self):
        super().process_weights_after_loading()
        # Construct the actual context-RoPE backend during GPU loading so an
        # unavailable implementation fails loading, not the first commit call.
        if self.embed_tokens.weight.device.type == "cuda":
            self.initialize_context_rope()

    def load_weights(self, weights):
        weights = weights.items() if isinstance(weights, dict) else weights

        def mapped():
            for name, payload in weights:
                key = self.WEIGHTS_MAPPER.map_name(name)
                if key == "d2t":
                    tensor, _ = unwrap_checkpoint_tensor(payload)
                    if self.d2t is not None:
                        raise ValueError("Duplicate DSpARK d2t map")
                    self._buffers["d2t"] = absolute_d2t(
                        name, tensor, self.vocab_size, self._input_vocab_size
                    )
                    _mark_loaded(self, "d2t")
                    continue
                yield key, payload

        # Borrowed destinations explicitly reject tied-head fallback in the
        # opt-in contract; ordinary checkpoints retain dense Qwen tying.
        Qwen3ForCausalLM.load_weights(self, mapped())

    def validate_weights_loaded(self, loaded_tensor_ids=None):
        super().validate_weights_loaded(loaded_tensor_ids)
        if self.vocab_size != self._input_vocab_size and self.d2t is None:
            raise ValueError("Distinct DSpARK draft vocabulary requires d2t")

        if "lm_head.weight" in getattr(self, "_rtp_borrowed_parameter_bindings", {}):
            if self.d2t is not None and not torch.equal(
                self.d2t, torch.arange(self.vocab_size, device=self.d2t.device)
            ):
                raise ValueError(
                    "Borrowed DSpARK lm_head requires identity draft token order"
                )

    def runtime_weight_view(self):
        weights = super().runtime_weight_view()
        weights.update(self.markov_head.runtime_weight_view())
        if self.d2t is not None:
            weights["multi_tokens_predict_d2t_map"] = self.d2t
        return weights

    def _context_kv_rows(self, hidden, kv_heads, head_dim):
        # Views of the live fused QKV rows (no second projection parameter;
        # post-load updates remain visible on the next commit).
        for layer in self.layers:
            attention = layer.self_attn
            kv = torch.nn.functional.linear(
                hidden, attention.qkv_proj.weight[attention.q_size :]
            )
            key, value = kv.reshape(-1, 2, kv_heads, head_dim).unbind(1)
            key = attention.k_norm(key.reshape(-1, head_dim)).view(
                -1, kv_heads, head_dim
            )
            yield key, value

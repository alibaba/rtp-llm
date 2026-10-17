"""DiffusionGemma proposer (DSpark draft) model registration.

The proposer shares the main model's backbone modules in-process (registry
lookup) and loads no per-layer weights of its own; only the embedding / lm_head
globals (aliased from the target) and the zero Markov matrices (required by the
DSpARK executor's shape checks, unused in diffusion mode) are declared. Its
ModelConfig describes a one-layer tiny cache so the engine's MTP module cache
costs kilobytes — the canvas attention reads the MAIN cache through the main
model reference.
"""

import json
import os
from typing import Any, List

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.ops import CacheGroupType, DataType, KVCacheSpecDesc, KVCacheSpecType
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    transpose,
)
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight

from rtp_llm.models.diffusion_gemma import DiffusionGemmaConfig

DGM_PYTHON_FIELDS_PROP = (
    "dgm_canvas_length",
    "dgm_max_denoising_steps",
    "dgm_t_min",
    "dgm_t_max",
    "dgm_entropy_bound",
    "dgm_confidence_threshold",
    "dgm_stability_threshold",
    "dgm_final_logit_softcapping",
    "dgm_diffusion_canvas",
)

# Extend the python-field allowlist once (shared class attribute).
_base_fields = set(getattr(ModelConfig, "_python_fields", set()))
_needed = set(DGM_PYTHON_FIELDS_PROP) | {
    "dgm_head_dim", "dgm_kv_head_num", "dgm_global_head_dim", "dgm_global_kv_head_num",
    "dgm_global_partial_rope", "dgm_swa_rope_theta", "dgm_global_rope_theta",
    "dgm_sliding_window", "dgm_layer_types", "dgm_final_logit_softcapping",
}
if not _needed <= _base_fields:
    ModelConfig._python_fields = _base_fields | _needed


class DiffusionGemmaProposerWeightInfo(ModelDeployWeightInfo):
    def _get_weight_info(self) -> ModelWeightInfo:
        vocab = int(self.model_config.vocab_size)
        weights: List = [
            # Aliased from the target (skipped at load when the alias owner
            # is attached by ModelFactory).
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("model.decoder.embed_tokens.weight")],
                identity,
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("model.decoder.embed_tokens.weight")],
                identity,
            ),
            # DSpARK executor shape checks require [vocab, rank] Markov
            # matrices; diffusion mode never consumes them.
            AtomicWeight(
                W.dspark_markov_w1,
                [],
                lambda ts, v=vocab: torch.zeros((v, 8), dtype=torch.float32),
            ),
            AtomicWeight(
                W.dspark_markov_w2,
                [],
                lambda ts, v=vocab: torch.zeros((v, 8), dtype=torch.float32),
            ),
        ]
        # One tiny MHA layer keeps the engine's MTP module cache allocation
        # near zero; canvas attention never touches it.
        tiny = KVCacheSpecDesc()
        tiny.tag = "swa_kv"
        tiny.cache_type = KVCacheSpecType.MHA
        tiny.mha_kv_head_num = 1
        tiny.mha_size_per_head = 64
        tiny.dtype = DataType.TYPE_BF16
        layer_weights = [[]]
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


class DiffusionGemmaProposer(BaseModel):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {ckpt_path}")
        with open(config_path) as reader:
            config_json = json.loads(reader.read())
        text = config_json.get("text_config", config_json)
        generation = {}
        gen_path = os.path.join(ckpt_path, "generation_config.json")
        if os.path.exists(gen_path):
            try:
                with open(gen_path) as reader:
                    generation = json.loads(reader.read())
            except json.JSONDecodeError:
                generation = {}

        config = DiffusionGemmaConfig()
        config.ckpt_path = ckpt_path
        # Tiny draft cache; the proposer reads the main cache.
        config.num_layers = 1
        config.attn_config.head_num = 1
        config.attn_config.kv_head_num = 1
        config.attn_config.size_per_head = 64
        config.attn_config.is_causal = False
        config.hidden_size = int(text["hidden_size"])
        config.vocab_size = int(text["vocab_size"])
        config.max_seq_len = int(text["max_position_embeddings"])
        config.tie_word_embeddings = True
        config.layernorm_eps = float(text["rms_norm_eps"])
        config.norm_type = "rmsnorm"
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.is_mtp = True
        config.gen_num_per_cycle = int(config_json.get("canvas_length", 256))

        # DSpark metadata consumed by ModelFactory._setup_dspark_configs.
        config.dspark_noise_token_id = 0  # input tokens are ignored by the proposer
        config.dspark_target_layer_ids = [max(0, int(text["num_hidden_layers"]) - 1)]
        config.dspark_markov_rank = 8
        config.dspark_sample_from_anchor = True

        # Diffusion canvas + EB sampler parameters.
        config.dgm_canvas_length = int(config_json.get("canvas_length", 256))
        config.dgm_max_denoising_steps = int(generation.get("max_denoising_steps", 48))
        t_max = float(generation.get("t_max", 0.8))
        t_min = float(generation.get("t_min", 0.4))
        config.dgm_t_min = t_min
        config.dgm_t_max = t_max
        sampler_cfg = generation.get("sampler_config") or {}
        config.dgm_entropy_bound = float(sampler_cfg.get("entropy_bound", 0.1))
        config.dgm_confidence_threshold = float(
            generation.get("confidence_threshold", 0.005)
        )
        config.dgm_stability_threshold = int(generation.get("stability_threshold", 1))
        config.dgm_final_logit_softcapping = float(
            text.get("final_logit_softcapping", 30.0)
        )
        config.dgm_diffusion_canvas = True

        tiny = KVCacheSpecDesc()
        tiny.tag = "swa_kv"
        tiny.cache_type = KVCacheSpecType.MHA
        tiny.mha_kv_head_num = 1
        tiny.mha_size_per_head = 64
        tiny.dtype = DataType.TYPE_BF16
        config.kv_cache_spec_descs = [[tiny]]
        return config

    @staticmethod
    def get_weight_cls():
        return DiffusionGemmaProposerWeightInfo

    @classmethod
    def speculative_weight_alias_names(cls, target_model, draft_model_config):
        if not hasattr(target_model, "py_model") or target_model.py_model is None:
            raise TypeError("DiffusionGemma proposer requires a DiffusionGemma target owner")
        target_config = target_model.model_config
        for name in ("vocab_size", "hidden_size"):
            if getattr(target_config, name) != getattr(draft_model_config, name):
                raise ValueError(
                    "DiffusionGemma proposer cannot alias incompatible target: "
                    f"{name}={getattr(target_config, name)!r} vs "
                    f"{getattr(draft_model_config, name)!r}"
                )
        return (W.embedding, W.lm_head)

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.diffusion_gemma_proposer import (
            DiffusionGemmaProposer as ProposerModel,
        )

        self.py_model = ProposerModel(
            self.model_config, self.parallelism_config, self.model_config.ckpt_path
        )
        return self.py_model

    def support_cuda_graph(self) -> bool:
        return False


register_model("diffusion_gemma_proposer", DiffusionGemmaProposer)

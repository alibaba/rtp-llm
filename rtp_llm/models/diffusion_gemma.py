"""DiffusionGemma main (target/encoder) model registration.

HF: google/diffusiongemma-26B-A4B-it, architecture
DiffusionGemmaForBlockDiffusion. One safetensors set holds the decoder-side
backbone (tied with the encoder), per-mode layer scalars, self-conditioning
module and the vision tower. The engine target model runs the encoder role
(causal prefill + canvas commit verify); the proposer model type
``diffusion_gemma_proposer`` runs the decoder role (see
models/diffusion_gemma_proposer.py).
"""

import json
import os
from typing import Any, List

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.diffusion_gemma_weight import DiffusionGemmaWeightInfo
from rtp_llm.ops import (
    CacheGroupType,
    CacheTailPolicyDesc,
    DataType,
    KVCacheSpecDesc,
    KVCacheSpecType,
)
from rtp_llm.utils.util import get_config_from_path

# Python-only ModelConfig fields for DiffusionGemma.
DGM_PYTHON_FIELDS = (
    "dgm_head_dim",
    "dgm_kv_head_num",
    "dgm_global_head_dim",
    "dgm_global_kv_head_num",
    "dgm_global_partial_rope",
    "dgm_swa_rope_theta",
    "dgm_global_rope_theta",
    "dgm_sliding_window",
    "dgm_layer_types",
    "dgm_canvas_length",
    "dgm_final_logit_softcapping",
    "dgm_max_denoising_steps",
    "dgm_t_min",
    "dgm_t_max",
    "dgm_entropy_bound",
    "dgm_confidence_threshold",
    "dgm_stability_threshold",
    "dgm_diffusion_canvas",
)


def _extend_python_fields() -> None:
    base = getattr(ModelConfig, "_python_fields", None)
    if isinstance(base, (set, frozenset)) and not all(f in base for f in DGM_PYTHON_FIELDS):
        merged = set(base)
        merged.update(DGM_PYTHON_FIELDS)
        ModelConfig._python_fields = merged


_extend_python_fields()


class DiffusionGemmaConfig(ModelConfig):
    """ModelConfig with DiffusionGemma python-only fields."""

    _python_fields = set(getattr(ModelConfig, "_python_fields", set())) | set(DGM_PYTHON_FIELDS)


def build_diffusion_gemma_kv_spec_descs(config: "DiffusionGemmaConfig") -> list:
    """Per-layer cache spec descs: sliding layers use an MHA SWA pool shaped
    by the global attn_config; the 5 global-attention layers need their own
    (kv 2 x 512) pool, declared through the per-tag MHA shape override."""
    descs = []
    for layer_idx in range(config.num_layers):
        layer_type = config.dgm_layer_types[layer_idx]
        if layer_type == "sliding_attention":
            swa = KVCacheSpecDesc()
            swa.tag = "swa_kv"
            swa.cache_type = KVCacheSpecType.MHA
            swa.group_type = CacheGroupType.SWA
            swa.dtype = DataType.TYPE_BF16
            # SWACacheManager materializes only the active tail plus the
            # reserve region and fills the remaining table slots with
            # NULL_BLOCK_IDX (-1). Without this policy the effective tail is
            # ~2 blocks, so a 278-token prefill's block table arrives as
            # [-1, -1, -1, 1, 2, ...] and every position->page mapping (KV
            # write, paged prefix read) indexes block -1 -> illegal memory
            # access (run 37 write-geometry dump). Materialize the full
            # window: ceil(window / kernel_page) + 1, matching the engine's
            # own kernel-block-table width for this group. Descs are built
            # before config finalization fills the block sizes, so fall back
            # to the project's 64-token page.
            page = (
                int(config.attn_config.kernel_tokens_per_block)
                or int(config.attn_config.tokens_per_block)
                or 64
            )
            tail = CacheTailPolicyDesc()
            tail.active_tail_blocks = (
                int(config.attn_config.sliding_window) + page - 1
            ) // page + 1
            swa.tail = tail
            descs.append([swa])
        else:
            g = KVCacheSpecDesc()
            g.tag = "global_kv"
            g.cache_type = KVCacheSpecType.MHA
            g.mha_kv_head_num = int(config.dgm_global_kv_head_num)
            g.mha_size_per_head = int(config.dgm_global_head_dim)
            g.dtype = DataType.TYPE_BF16
            descs.append([g])
    return descs


class DiffusionGemma(BaseModel):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {ckpt_path}")
        with open(config_path) as reader:
            config_json = json.loads(reader.read())
        text = config_json.get("text_config", config_json)
        if not isinstance(text, dict):
            raise TypeError("diffusion_gemma config.json requires a text_config object")

        config = DiffusionGemmaConfig()
        config.ckpt_path = ckpt_path

        config.attn_config.head_num = int(text["num_attention_heads"])
        config.attn_config.kv_head_num = int(text["num_key_value_heads"])
        config.attn_config.size_per_head = int(text["head_dim"])
        config.attn_config.is_causal = True
        config.attn_config.sliding_window = int(text["sliding_window"])
        config.attn_config.rope_config.style = 1  # RopeStyle.Base
        config.attn_config.dtype = torch.bfloat16
        config.num_layers = int(text["num_hidden_layers"])
        config.hidden_size = int(text["hidden_size"])
        config.vocab_size = int(text["vocab_size"])
        config.max_seq_len = int(text["max_position_embeddings"])
        config.tie_word_embeddings = bool(text.get("tie_word_embeddings", True))
        config.layernorm_eps = float(text["rms_norm_eps"])
        config.norm_type = "rmsnorm"
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.qk_norm = True
        config.activation_type = "gated-gelu"

        # MoE: shared expert MLP + routed experts.
        config.inter_size = int(text["intermediate_size"])
        config.moe_inter_size = int(text["moe_intermediate_size"])
        config.expert_num = int(text["num_experts"])
        config.moe_k = int(text["top_k_experts"])
        config.n_shared_experts = 1
        config.moe_style = 2  # shared expert without sigmoid gate
        config.has_moe_norm = True  # router renormalizes top-k weights
        config.moe_layer_index = list(range(config.num_layers))
        config.gen_num_per_cycle = int(config_json.get("canvas_length", 256))

        # DiffusionGemma python-only fields.
        config.dgm_head_dim = int(text["head_dim"])
        config.dgm_kv_head_num = int(text["num_key_value_heads"])
        config.dgm_global_head_dim = int(text.get("global_head_dim", text["head_dim"]))
        config.dgm_global_kv_head_num = int(text.get("num_global_key_value_heads", 0))
        rope_params = text.get("rope_parameters", {})
        swa_rope = rope_params.get("sliding_attention", {})
        glob_rope = rope_params.get("full_attention", {})
        config.dgm_swa_rope_theta = float(swa_rope.get("rope_theta", 10000.0))
        config.dgm_global_rope_theta = float(glob_rope.get("rope_theta", 1000000.0))
        config.dgm_global_partial_rope = float(glob_rope.get("partial_rotary_factor", 1.0))
        config.dgm_sliding_window = int(text["sliding_window"])
        layer_types = text.get("layer_types")
        if not layer_types:
            from rtp_llm.models_py.model_desc.diffusion_gemma import diffusion_gemma_layer_types

            layer_types = diffusion_gemma_layer_types(config.num_layers)
        config.dgm_layer_types = list(layer_types)
        config.dgm_canvas_length = int(config_json.get("canvas_length", 256))
        config.dgm_final_logit_softcapping = float(text.get("final_logit_softcapping", 30.0))

        config.kv_cache_spec_descs = build_diffusion_gemma_kv_spec_descs(config)

        # Multimodal: vision tower + embed_vision live in the same checkpoint
        # (model.encoder.vision_tower.* / model.encoder.embed_vision.*) and are
        # consumed by the diffusion_gemma multimodal mixin.
        if "vision_config" in config_json:
            config.mm_model_config.is_multimodal = True
            config.mm_model_config.mm_sep_tokens = [
                [
                    int(config_json.get("boi_token_id", 255999)),
                    int(config_json.get("eoi_token_id", 258882)),
                ]
            ]
            config.mm_related_params.special_tokens.update(
                {"default_mm_token": "<|image>"}
            )
            config.mm_related_params.config["ckpt_path"] = ckpt_path
        return config

    @staticmethod
    def get_weight_cls():
        return DiffusionGemmaWeightInfo

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.diffusion_gemma import DiffusionGemmaModel

        self.py_model = DiffusionGemmaModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        from rtp_llm.models.diffusion_gemma_registry import register_main_model

        register_main_model(self.py_model, self.model_config.ckpt_path)
        return self.py_model

    def support_cuda_graph(self) -> bool:
        return False  # canvas denoising is adaptive-length per request


register_model(
    "diffusion_gemma",
    DiffusionGemma,
    ["DiffusionGemmaForBlockDiffusion"],
    ["google/diffusiongemma-26B-A4B-it"],
)

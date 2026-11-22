"""DeepSeek-VL2 Tiny: bias-free Q/K/V, dense and shared/routed expert layers."""

import json
from pathlib import Path

import torch

from rtp_llm.models_py.modules.base.common.multimodal_embedding import (
    MultimodalEmbeddingInjector,
)
from rtp_llm.models_py.new_models.glm4_moe.checkpoint_layout import (
    Glm4CheckpointLayout,
    has_correction_bias,
)
from rtp_llm.models_py.new_models.glm4_moe.language import GqaMoeForCausalLM
from rtp_llm.models_py.new_models.glm4_moe.parameters import install_gqa_moe_parameters
from rtp_llm.models_py.weight_mapper import WeightsMapper


class DeepSeekVLV2ForCausalLM(GqaMoeForCausalLM):
    WEIGHTS_MAPPER = WeightsMapper(
        prefix_mapping={"language.model.": "", "language.": ""}
    )

    @staticmethod
    def _install_parameters(owner, cfg, *, layer_ids, partitioned):
        config = cfg["model_config"]
        path = Path(config.ckpt_path)
        hf = json.loads((path / "config.json").read_text())["language_config"]
        if hf.get("use_mla", "kv_lora_rank" in hf):
            raise ValueError(
                "DeepSeek-VL2 Small/Full use MLA; the supported Tiny checkpoint "
                "uses separate Q/K/V projections"
            )
        depth = hf["num_hidden_layers"]
        layout = Glm4CheckpointLayout(
            num_layers=config.num_layers,
            non_target_layers=depth - config.num_layers,
            routed_layers=frozenset(config.moe_layer_index),
            dense_width=hf["intermediate_size"],
            shared_width=hf["n_shared_experts"] * config.moe_inter_size,
            use_qk_norm=False,
            correction_bias=has_correction_bias(path, depth, prefix="language."),
        )
        install_gqa_moe_parameters(
            owner, cfg, layer_ids=layer_ids, partitioned=partitioned,
            layout=layout, qkv_bias=False,
        )

    def __init__(self, model_config, load_config):
        super().__init__(model_config, load_config)
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()

    def checkpoint_weight_name_filter(self):
        stage_owns = super().checkpoint_weight_name_filter()
        return lambda name: (
            name.startswith("language.")
            and self._layout.target_owns(name.removeprefix("language."))
            and stage_owns(name)
        )

    def _input_embeddings(self, inputs):
        mm = getattr(inputs, "multimodal_inputs", None)
        features = getattr(mm, "multimodal_features", None)
        if not features:
            return self.embed_tokens(inputs.input_ids)
        mask = inputs.embedding_inputs.text_tokens_mask
        ids = inputs.input_ids
        if mask is not None and mask.numel():
            mask = mask.to(device=ids.device, dtype=torch.bool)
            ids = torch.where(mask, ids, 0)
        embedded = self.embed_tokens(ids)
        if mask is not None and mask.numel():
            embedded = embedded * mask.unsqueeze(-1)
        return self.multimodal_embedding_injector(embedded, features, mm.mm_features_locs)

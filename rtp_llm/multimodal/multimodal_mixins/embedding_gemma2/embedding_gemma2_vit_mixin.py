"""EmbeddingGemma 2 vision weights through the validated Gemma4 image stack."""

import torch

from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import BaseVitWeights
from rtp_llm.multimodal.multimodal_mixins.embedding_gemma2.embedding_gemma2_media import (
    EmbeddingGemma2MediaEmbedding,
    EmbeddingGemma2PromptExpander,
)
from rtp_llm.multimodal.multimodal_mixins.gemma4.gemma4_vit_mixin import (
    Gemma4DeployWeightInfo,
    Gemma4Mixin,
)


class EmbeddingGemma2VisionWeightInfo(Gemma4DeployWeightInfo):
    @staticmethod
    def _ckpt_name(name):
        if name.startswith("audio."):
            return name.removeprefix("audio.")
        name = name.removeprefix("visual.")
        # Same Gemma4 vision module naming; this composite checkpoint has no
        # leading "model." on vision_tower/embed_vision tensors.
        return Gemma4DeployWeightInfo._ckpt_name(name).removeprefix("model.")


class EmbeddingGemma2VisionMixin(Gemma4Mixin):
    def _init_multimodal(self):
        self.mm_part = EmbeddingGemma2MediaEmbedding(self.mm_related_params)
        self.mm_part.output_dtype = self.compute_dtype
        self.mm_related_params.vit_weights = BaseVitWeights(
            {"visual": self.mm_part.visual, "audio": self.mm_part.audio}
        )

    def load_mm_weight(self, ctype, device):
        super().load_mm_weight(ctype, device)
        tower = self.mm_part.visual.vision_tower
        spatial_dim = tower.head_dim // 2
        # The checkpoint reference constructs this buffer on CPU in FP32.
        # GPU pow differs slightly and can change the final BF16 features.
        theta = self.mm_related_params.config["rope_parameters"]["rope_theta"]
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, spatial_dim, 2, dtype=torch.float32, device="cpu")
                / spatial_dim
            )
        )
        tower.rotary_emb.inv_freq = inv_freq.to(device)
        self.mm_part.audio.to(device=device, dtype=self.compute_dtype)

    @classmethod
    def get_multimodal_mixin_weight_info(cls):
        return EmbeddingGemma2VisionWeightInfo

    @classmethod
    def _get_mm_module(cls, mm_related_params, vit_config):
        media = EmbeddingGemma2MediaEmbedding(mm_related_params)
        return torch.nn.ModuleList([media.visual, media.audio])

    @classmethod
    def create_prompt_expander(cls, mm_related_params):
        return EmbeddingGemma2PromptExpander(mm_related_params.config["ckpt_path"])


register_multimodal_mixin("embedding_gemma2", EmbeddingGemma2VisionMixin)

import torch

from rtp_llm.models_py.model_desc.multimodal_embedding import (
    embed_with_multimodal_features,
)
from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextModel
from rtp_llm.ops.compute_ops import PyModelInputs


class Qwen35VLNextModel(Qwen3NextModel):
    def _embed_input_ids(self, inputs: PyModelInputs) -> torch.Tensor:
        return embed_with_multimodal_features(self.embed_tokens, inputs)


__all__ = ["Qwen35VLNextModel"]

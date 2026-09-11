"""Checkpoint loading for the shared DeepSeek vision encoder."""

import torch

from rtp_llm.models.multimodal.deepseek_vision import RMSNorm
from rtp_llm.models.multimodal.multimodal_mixin import BaseVitWeights


class DeepSeekVisionWeights(BaseVitWeights):
    def __init__(self, vision_parts):
        super().__init__(vision_parts, with_prefix=True)
        self.weight_dtypes = {
            f"vision.{name}.weight": torch.float32
            for name, module in vision_parts["vision"].named_modules()
            if isinstance(module, RMSNorm)
        }

    def _set_weight_prefix(self):
        self._ckpt_prefix = ""
        self._ft_prefix = "self.mm_part."

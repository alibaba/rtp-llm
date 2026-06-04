from typing import Dict

import torch

from rtp_llm.models.qwen_v2_audio.configuration_qwen2_audio import (
    Qwen2AudioEncoderConfig,
)
from rtp_llm.models.qwen_v2_audio.modeling_qwen2_audio import Qwen2AudioEncoder


class OmniAudioProcessor:
    def __init__(self, audio_config_dict: Dict):
        encoder_config = Qwen2AudioEncoderConfig.from_dict(audio_config_dict)
        self.audio_tower = Qwen2AudioEncoder._from_config(encoder_config)

    @staticmethod
    def validate_config(audio_config: Dict) -> None:
        required = [
            "d_model",
            "encoder_attention_heads",
            "encoder_layers",
            "num_mel_bins",
        ]
        for key in required:
            if key not in audio_config:
                raise ValueError(f"Missing required audio config key: {key}")

    @property
    def device(self):
        return next(self.audio_tower.parameters()).device

    @torch.inference_mode()
    def encode(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.audio_tower(input_features, attention_mask=attention_mask)

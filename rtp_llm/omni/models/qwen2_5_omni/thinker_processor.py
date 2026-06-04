from typing import Dict

import torch

from rtp_llm.config.model_config import VitParameters
from rtp_llm.models.multimodal.multimodal_common import AudioEmbeddingInterface
from rtp_llm.omni.models.qwen2_5_omni.thinker_audio import OmniAudioProcessor
from rtp_llm.omni.models.qwen2_5_omni.thinker_vision import OmniVisionProcessor
from rtp_llm.utils.util import get_config_from_path


class OmniThinkerProcessor(AudioEmbeddingInterface):
    def __init__(self, mm_related_params: VitParameters, ckpt_path: str):
        self.mm_related_params = mm_related_params
        config_json = get_config_from_path(ckpt_path)
        thinker_cfg = config_json.get("thinker_config", {})

        audio_config = thinker_cfg.get("audio_config", {})
        if audio_config:
            self.audio_processor = OmniAudioProcessor(audio_config)
        else:
            self.audio_processor = None

        vision_config = thinker_cfg.get("vision_config", {})
        if vision_config:
            self.vision_processor = OmniVisionProcessor(vision_config)
        else:
            self.vision_processor = None

    @property
    def _device(self):
        if self.audio_processor:
            return self.audio_processor.device
        return torch.device("cpu")

    @torch.inference_mode()
    def audio_embedding(
        self, features_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.audio_processor is None:
            raise RuntimeError("Audio processor not initialized")
        input_features = features_dict["input_features"].to(self._device)
        attention_mask = features_dict.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
        return self.audio_processor.encode(input_features, attention_mask)

    @torch.inference_mode()
    def vision_embedding(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor = None
    ) -> torch.Tensor:
        if self.vision_processor is None:
            raise RuntimeError("Vision processor not initialized")
        return self.vision_processor.encode(pixel_values, grid_thw)

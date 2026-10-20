"""Whisper A0+A1 processor: audio URL -> log-mel -> encoder hidden states.

A0 (mel) reuses transformers `WhisperFeatureExtractor` exactly like the
qwen2_audio mixin (processor.py:6-8,33) — 16 kHz mono via librosa, 128 mel
bins, 30 s zero-padded window. A1 runs the WhisperEncoderTower and returns
the LAST encoder hidden state as request-level side state: the engine flag
`mm_model_config.mm_skip_token_expansion` keeps it out of the token sequence
(whisper's decoder consumes it via cross-attention, not embedding splice).

Faithfulness notes (read-whisper-ref §5):
- the reference encoder ignores attention_mask entirely (padding is silence);
  we keep the full 1500 frames and never mask cross attention.
- no projector: the decoder owns encoder_attn.{k,v}_proj weights.
"""

from typing import Dict

import librosa
import torch
from transformers.models.whisper.feature_extraction_whisper import (
    WhisperFeatureExtractor,
)

from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    AudioEmbeddingInterface,
    VitConfig,
    get_bytes_io_from_url,
)
from rtp_llm.multimodal.multimodal_mixins.whisper.modeling_whisper import (
    WhisperEncoderTower,
)
from rtp_llm.ops import MultimodalInput
from rtp_llm.utils.base_model_datatypes import VitParameters
from rtp_llm.utils.util import get_config_from_path


class Processor(AudioEmbeddingInterface):
    def __init__(self, mm_related_params: VitParameters):
        self.mm_related_params = mm_related_params
        ckpt_path = mm_related_params.config["ckpt_path"]
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(ckpt_path)
        config_json = get_config_from_path(ckpt_path)
        self.encoder = WhisperEncoderTower(
            num_mel_bins=config_json.get("num_mel_bins", 128),
            d_model=config_json.get("d_model", 1280),
            num_layers=config_json.get("encoder_layers", 32),
            num_heads=config_json.get("encoder_attention_heads", 20),
            ffn_dim=config_json.get("encoder_ffn_dim", 5120),
            max_source_positions=config_json.get("max_source_positions", 1500),
            layer_norm_eps=1e-5,
        )

    @property
    def _data_type(self):
        return self.encoder.conv1.weight.dtype

    @property
    def _device(self):
        return self.encoder.conv1.weight.device

    @staticmethod
    def preprocess_input(
        mm_inputs,
        vit_config: VitConfig,
        feature_extractor=None,
        **kwargs,
    ):
        assert len(mm_inputs) == 1
        data = get_bytes_io_from_url(
            mm_inputs[0].url,
            vit_config.download_headers,
            max_file_size_kb=vit_config.mm_image_max_file_size_kb,
        )
        audio = librosa.load(data, sr=feature_extractor.sampling_rate)[0]
        features_dict = feature_extractor(
            [audio],
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
        )
        return features_dict

    def get_preprocess_params(self):
        return {
            "feature_extractor": self.feature_extractor,
        }

    @torch.inference_mode()
    def embedding(self, feature_dict: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        return self.audio_embedding(feature_dict)

    @torch.inference_mode()
    def audio_embedding(self, features_dict: Dict[str, torch.Tensor]):
        input_features = features_dict["input_features"].to(self._device).to(self._data_type)
        encoder_hidden = self.encoder(input_features)
        assert encoder_hidden.shape[0] == 1, "audio_feature_dim0 != 1"
        # [1, L, D] -> [L, D]: the engine treats each row as one mm feature row
        # but with mm_skip_token_expansion rows never enter the token stream.
        encoder_hidden = encoder_hidden[0]
        return encoder_hidden, None

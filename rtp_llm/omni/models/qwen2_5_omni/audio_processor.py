"""Audio processor for Qwen2.5-Omni: feature extraction + encoder + BOS/EOS wrapping.

Follows the pattern of rtp_llm/models/qwen_v2_audio/processor.py but adapted for the
Qwen2.5-Omni audio encoder which has built-in projection and audio_bos_eos_token.
"""

from io import BytesIO
from typing import Dict, Tuple

import librosa
import torch
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor

from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    AudioEmbeddingInterface,
    timeout_decorator,
)
from rtp_llm.omni.models.qwen2_5_omni.audio_encoder import (
    AudioEncoderConfig,
    Qwen2_5OmniAudioEncoder,
)
from rtp_llm.utils.util import get_config_from_path


class Processor(AudioEmbeddingInterface):
    def __init__(self, mm_related_params, ckpt_path: str):
        self.mm_related_params = mm_related_params
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(ckpt_path)

        config_json = get_config_from_path(ckpt_path)
        thinker_config = config_json.get("thinker_config", config_json)
        audio_config_dict = thinker_config["audio_config"]
        config = AudioEncoderConfig.from_dict(audio_config_dict)
        self.audio_tower = Qwen2_5OmniAudioEncoder(config)

    @property
    def _data_type(self):
        return self.audio_tower.conv1.weight.dtype

    @property
    def _device(self):
        return self.audio_tower.conv1.weight.device

    @timeout_decorator(30)
    def _mm_preprocess(self, data: BytesIO, **kwargs) -> Dict[str, torch.Tensor]:
        audio = librosa.load(data, sr=self.feature_extractor.sampling_rate)[0]
        features_dict = self.feature_extractor(
            [audio],
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return features_dict

    @torch.inference_mode()
    def audio_embedding(
        self, features_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        input_features = (
            features_dict["input_features"].to(self._device).to(self._data_type)
        )
        feature_attention_mask = features_dict["attention_mask"].to(self._device)

        feature_lens = feature_attention_mask.sum(-1).long()

        # Flatten features using mask (batch=1 expected)
        batch_size = input_features.shape[0]
        assert batch_size == 1, "audio_embedding expects batch_size=1"

        # input_features: [1, mel_bins, T] — extract valid frames
        flat_features = input_features[0, :, :int(feature_lens[0])]  # [mel_bins, valid_T]

        audio_feat_lengths, audio_output_lengths = (
            self.audio_tower._get_feat_extract_output_lengths(feature_lens)
        )

        audio_features = self.audio_tower(
            flat_features, feature_lens=feature_lens
        )

        # Trim to expected output length
        expected_len = int(audio_output_lengths.sum())
        audio_features = audio_features[:expected_len]

        # Wrap with BOS/EOS tokens from audio_bos_eos_token
        bos_embed = self.audio_tower.audio_bos_eos_token(
            torch.tensor([0], device=self._device)
        )
        eos_embed = self.audio_tower.audio_bos_eos_token(
            torch.tensor([1], device=self._device)
        )
        audio_features = torch.cat(
            [bos_embed, audio_features, eos_embed], dim=0
        )

        return audio_features

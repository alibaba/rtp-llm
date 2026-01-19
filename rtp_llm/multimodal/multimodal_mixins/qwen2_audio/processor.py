from io import BytesIO
from typing import Dict, List

import librosa
import torch
from transformers.models.whisper.feature_extraction_whisper import (
    WhisperFeatureExtractor,
)

from rtp_llm.config.model_config import VitParameters
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    AudioEmbeddingInterface,
    MultimodalInput,
    VitConfig,
    get_bytes_io_from_url,
    timeout_decorator,
)
from rtp_llm.multimodal.multimodal_mixins.qwen2_audio.configuration_qwen2_audio import (
    Qwen2AudioConfig,
    Qwen2AudioEncoderConfig,
)
from rtp_llm.multimodal.multimodal_mixins.qwen2_audio.modeling_qwen2_audio import (
    Qwen2AudioEncoder,
    Qwen2AudioMultiModalProjector,
)
from rtp_llm.utils.util import get_config_from_path


class Processor(AudioEmbeddingInterface):
    def __init__(self, mm_related_params: VitParameters, ckpt_path: str):
        self.mm_related_params = mm_related_params
        dtype = self._data_type
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(ckpt_path)
        config_json = get_config_from_path(ckpt_path)
        audio_config_json = config_json["audio_config"]
        # audio_tower
        audio_config = Qwen2AudioEncoderConfig.from_dict(audio_config_json)
        self.audio_tower = Qwen2AudioEncoder._from_config(audio_config)
        # projector
        model_config = Qwen2AudioConfig.from_dict(config_json)
        self.multi_modal_projector = Qwen2AudioMultiModalProjector(model_config)

    @property
    def _device(self):
        return self.audio_tower.device

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput],
        vit_config: VitConfig,
        feature_extractor,
        **kwargs,
    ):
        assert len(mm_inputs) == 1
        data = get_bytes_io_from_url(mm_inputs[0].url, vit_config.download_headers)
        audio = librosa.load(data, sr=feature_extractor.sampling_rate)[0]
        features_dict = feature_extractor(
            [audio],
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return features_dict

    def get_preprocess_params(self):
        return {
            "feature_extractor": self.feature_extractor,
        }

    @torch.inference_mode()
    def embedding(
        self, feature_dict: Dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        return self.audio_embedding(feature_dict)

    @torch.inference_mode()
    def audio_embedding(self, features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_features = (
            features_dict["input_features"].to(self._device).to(self._data_type)
        )
        feature_attention_mask = (
            features_dict["attention_mask"].to(self._device).to(self._data_type)
        )
        audio_feat_lengths, audio_output_lengths = (
            self.audio_tower._get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)
            )
        )
        batch_size, _, max_mel_seq_len = input_features.shape
        max_seq_len = (max_mel_seq_len - 2) // 2 + 1
        # Create a sequence tensor of shape (batch_size, max_seq_len)
        seq_range = (
            torch.arange(0, max_seq_len, dtype=self._data_type, device=self._device)
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)

        # Create mask
        padding_mask = seq_range >= lengths_expand

        audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
            batch_size, 1, max_seq_len, max_seq_len
        )
        audio_attention_mask = audio_attention_mask_.to(
            dtype=self.audio_tower.conv1.weight.dtype,
            device=self.audio_tower.conv1.weight.device,
        )
        audio_attention_mask[audio_attention_mask_] = float("-inf")
        audio_outputs = self.audio_tower(
            input_features, attention_mask=audio_attention_mask
        )
        selected_audio_feature = audio_outputs.last_hidden_state
        # ensure input always batch=1
        assert selected_audio_feature.shape[0] == 1, "audio_feature_dim0 != 1"
        selected_audio_feature = selected_audio_feature[0][: int(audio_output_lengths)]
        audio_features = self.multi_modal_projector(selected_audio_feature)
        return audio_features, None

import os
import json
import torch

import numpy as np
from typing import List, Any, Dict, Tuple, Union
from torchaudio.transforms import Resample
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from transformers.models.whisper.processing_whisper import WhisperProcessor

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.models.base_model import BaseModel
from maga_transformer.models.multimodal.multimodal_mixin import MultiModalMixin, BaseVitWeights
from maga_transformer.models.multimodal.multimodal_common import AudioEmbeddingInterface
from maga_transformer.distribute.worker_info import ParallelInfo, g_parallel_info
from maga_transformer.model_factory_register import register_model
from maga_transformer.models.whisper_weight import WhisperWeightInfo

class WhisperAudioEmbedding(AudioEmbeddingInterface):
    def __init__(self, processor: WhisperProcessor, encoder: WhisperEncoder, embedding_length: int):
        self.processor: WhisperProcessor = processor
        self.sampling_rate: int = self.processor.feature_extractor.sampling_rate
        self.encoder = encoder.half().cuda()
        self.embedding_length = embedding_length

    @torch.no_grad()
    def audio_embedding(self, audio: Tuple[torch.Tensor, int], device):
        audio_data, sample_rate = audio
        if sample_rate != self.sampling_rate:
            resampler = Resample(orig_freq = sample_rate, new_freq = self.sampling_rate)
            audio_data = resampler(audio_data)
        features = self.processor(np.array(audio_data), sampling_rate=self.sampling_rate, return_tensors="pt").input_features
        features = self.encoder(features.to(device).half())
        # features type is BaseModelOutput
        res = features.last_hidden_state
        if res.shape[0] > 1:
            raise Exception("Cannot deal with multichannel input")
        if res.shape[1] != self.embedding_length:
            raise Exception(f"Wrong shape embedding for audio input dim 1, expect {self.embedding_length}, but get {res.shape[1]}")
        return res

class Whisper(BaseModel, MultiModalMixin):
    def __init__(self, config: GptInitModelParameters):
        if g_parallel_info.tp_rank == 0:
            with torch.device(g_parallel_info.device):
                ckpt_path = config.ckpt_path
                self.mm_part = WhisperAudioEmbedding(WhisperProcessor.from_pretrained(ckpt_path), WhisperEncoder.from_pretrained(ckpt_path), config.cross_attn_input_len)
            config.mm_related_params.vit_weights = BaseVitWeights({}, False)
        BaseModel.__init__(self, config)

    @staticmethod
    def get_weight_cls():
        return WhisperWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = GptInitModelParameters(
            head_num=0,
            size_per_head=0,
            layer_num=0,
            max_seq_len=0,
            vocab_size=0,
            ckpt_path=ckpt_path,
            activation_type='gelu',
            norm_type='layernorm',
            rotary_embedding_dim=0,
            rotary_embedding_style=0,
            has_post_decoder_layernorm=True,
            has_positional_encoding=True,
        )

        config_path = os.path.join(ckpt_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path) as reader:
                content = reader.read()
                content = content.replace("LlamaForCausalLM", "LLaMAForCausalLM")
                config_json = json.loads(content)
            Whisper._from_hf(config, config_json)
        return config

    @staticmethod
    def _from_hf(config: GptInitModelParameters, config_json: Dict[str, Any]):
        config.head_num = config_json['decoder_attention_heads']
        config.hidden_size = config_json['d_model']
        config.size_per_head = config.hidden_size // config.head_num
        config.layer_num = config_json['decoder_layers']
        config.inter_size = config_json['decoder_ffn_dim']
        config.activation_type = config_json['activation_function']
        config.use_cross_attn = True
        config.cross_attn_input_len = config_json["max_source_positions"]
        config.vocab_size = config_json['vocab_size']

    @staticmethod
    def multimodal_modify_prompt_plugin(prompt: Union[List[Dict[str, Any]], str], images: List[str],
                                        img_token: str, **kwargs: Any) -> Tuple[str, List[Any]]:
        return "", images

    @staticmethod
    def process_encode_plugin(prompt: str, generate_config: Dict[str, Any], special_tokens: Any, tokenizer, **kwargs: Any) -> List[int]:
        # temporary use '<|startoftranscript|><|en|><|transcribe|><|notimestamps|>' as input prompt
        # as the origin tokenizer encode will add extra special tokens
        # besides whipser not accept any input prompt
        return [50258, 50259, 50360, 50364]

    def multimodal_embedding(
        self, input_ids: torch.Tensor, image_features: List[torch.Tensor], token_type_ids: torch.Tensor
    ):
        if len(image_features) > 1:
            raise Exception('Whisper can only accept single audio')

        return self.word_embedding(input_ids)

register_model('whisper', Whisper)
import json
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from transformers import AutoTokenizer

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters, TemplateType
from maga_transformer.distribute.worker_info import ParallelInfo, g_parallel_info
from maga_transformer.model_factory_register import register_model
from maga_transformer.models.eva2clip_vit import EVA2CLIPImageEmbedding
from maga_transformer.models.cogvlm2_weight import CogVLM2WeightInfo, CogVLM2VitWeights
from maga_transformer.models.llama import Llama
from maga_transformer.models.multimodal.multimodal_mixin import MultiModalMixin
from maga_transformer.utils.util import to_torch_dtype

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1

class CogVLM2(Llama, MultiModalMixin):
    def __init__(self, config: GptInitModelParameters):
        quant_algo = config.quant_algo
        if quant_algo.isGptq() or quant_algo.isAwq() or quant_algo.isSmoothQuant() or quant_algo.isOmniQuant():
            raise Exception("CogVLM2 only support FP32, BF16, FP16, INT8, not support other quant algorithm")
        super().__init__(config)
        
    def _init_multimodal(self, config: GptInitModelParameters):        
        self.mm_part = EVA2CLIPImageEmbedding(config)
        config.mm_related_params.vit_weights = CogVLM2VitWeights(
            {"vit": self.mm_part.vit}
        )

    def load(self, device: str):
        if os.environ.get("VIT_TRT", "0") == "1":
            weights_info = self.get_weight_cls()(self.config, g_parallel_info.tp_size, g_parallel_info.tp_rank)
            self.init_mm_trt(
                weights_info, self.config.ckpt_path,
                self.config.mm_related_params, device, to_torch_dtype(self.config.data_type)
            )
        super().load(device=device)

    @staticmethod
    def _create_config(ckpt_path):
        config = GptInitModelParameters(
            head_num=0,
            size_per_head=0,
            layer_num=0,
            max_seq_len=0,
            vocab_size=0,
            ckpt_path=ckpt_path,
            activation_type="SiGLU",
            norm_type="rmsnorm",
            rotary_embedding_dim=128,
            rotary_embedding_style=1,
            rotary_embedding_base=500000,
            has_post_decoder_layernorm=True
        )
        # hugggingface
        config_path = os.path.join(ckpt_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as reader:
                content = reader.read()
                config_json = json.loads(content)
            CogVLM2.from_huggingface(config, config_json)
        else:
            raise Exception("cogvlm2 parameter from unkown source")
        config.tp_split_emb_and_lm_head = False  # cogvlm2 embedding can't tp
        return config

    @staticmethod
    def get_weight_cls():
        return CogVLM2WeightInfo

    @staticmethod
    def from_huggingface(config: GptInitModelParameters, config_json: Dict[str, Any]):
        config.use_expert_attention = True
        config.add_special_tokens = False
        config.head_num = config_json["num_attention_heads"]
        config.head_num_kv = 8
        config.hidden_size = config_json["hidden_size"]
        config.size_per_head = (
            config_json["hidden_size"] // config_json["num_attention_heads"]
        )
        config.layer_num = config_json["num_hidden_layers"]
        config.max_seq_len = config_json.get("max_position_embeddings", 8192)
        config.vocab_size = config_json["vocab_size"]
        config.layernorm_eps = config_json.get("rms_norm_eps", 1e-05)
        config.inter_size = config_json["intermediate_size"]
        config.rotary_embedding_dim = config.size_per_head
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
        config.build_position_ids = True
        try:
            template_type_str = config_json.get("template_version", "chat")
            config.template_type = TemplateType[template_type_str]
        except KeyError:
            raise Exception(f"unknown template_type: {template_type_str}")

        config.reserve_runtime_mem_mb = 2048

        vit_config = config_json["vision_config"]
        config.mm_related_params.config.update(vit_config)
        # use vision hidden size for linear_proj and conv layer in eva2clip
        config.mm_related_params.config['use_vision_hidden_size'] = True
        config.special_tokens.bos_token_id = config_json["bos_token_id"]
        config.special_tokens.pad_token_id = config_json["pad_token_id"]

        if isinstance(config_json['eos_token_id'], list):
            config.special_tokens.eos_token_id = config_json['eos_token_id'][0]
            config.special_tokens.stop_words_id_list = [[x] for x in config_json['eos_token_id']]
        else:
            config.special_tokens.eos_token_id = config_json['eos_token_id']

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path, trust_remote_code=True
        )
        tokenizer.pad_token_id = 128002  # llama3 adapt for cogvlm
        return tokenizer

    def extend_generate_combo_token_types(self, combo_tokens: List[int]) -> List[int]:
        return [LANGUAGE_TOKEN_TYPE] * len(combo_tokens)

    def extend_context_combo_token_types(self, token_types: List[int]) -> List[int]:
        return token_types

    def extend_generate_position_ids(
        self, generate_batch_size: int, num_beams: int,
        vision_token_length: List[int], seq_lengths_list: List[int]
    ) -> List[int]:
        position_ids = []
        for i in range(generate_batch_size):
            start_idx = i * num_beams
            if vision_token_length[start_idx] == 0:
                # only text
                position_ids.append(seq_lengths_list[start_idx] - 1)
            else:
                # ccontain image
                position_ids.append(seq_lengths_list[start_idx] - vision_token_length[start_idx] + 2)
        return position_ids

    def extend_context_position_ids(
        self, context_begin_position: int, context_end_position: int,
        token_type_ids: torch.Tensor, token_ids: torch.Tensor
    ) -> List[int]:
        # construct position ids for rotary embedding, assuming the token_type_ids is [T, V, V, V, V, V, T, T, T]
        # the expected position ids is [0, 1, 2, 2, 2, 3, 4, 5, 6]
        # see https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B/blob/main/modeling_cogvlm.py#318
        tmp = token_type_ids.numpy().copy()
        is_boi_eoi = np.zeros_like(tmp, dtype=bool)
        is_boi_eoi[1:] |= (tmp[1:] == VISION_TOKEN_TYPE) & (tmp[:-1] == LANGUAGE_TOKEN_TYPE)
        is_boi_eoi[0] |= tmp[0] == VISION_TOKEN_TYPE
        is_boi_eoi[:-1] |= (tmp[:-1] == VISION_TOKEN_TYPE) & (tmp[1:] == LANGUAGE_TOKEN_TYPE)
        is_boi_eoi[-1] |= tmp[-1] == VISION_TOKEN_TYPE
        tmp[is_boi_eoi] = LANGUAGE_TOKEN_TYPE
        y = np.zeros_like(tmp, dtype=np.int32)
        y[1:] = (tmp[1:] == LANGUAGE_TOKEN_TYPE) | (
            (tmp[1:] == VISION_TOKEN_TYPE) & (tmp[:-1] == LANGUAGE_TOKEN_TYPE)
        )
        y = y.cumsum()
        return y.tolist()

    def expand_token_id(
        self, token_ids: List[int], images: List[torch.Tensor]
    ) -> Tuple[List[int], List[torch.Tensor], List[int]]:
        if len(images) > 1:
            raise Exception("CogVLM2 support processes one image at a time")

        input_ids = [self.tokenizer.bos_token_id]
        token_type_ids = [LANGUAGE_TOKEN_TYPE]

        if len(images) > 0:
            patch_size: int = self.config.mm_related_params.config["patch_size"]
            image_size: int = self.config.mm_related_params.config["image_size"]

            vision_token_num = (image_size // patch_size // 2) * (
                image_size // patch_size // 2
            ) + 2

            input_ids += [self.tokenizer.pad_token_id] * vision_token_num
            token_type_ids += [VISION_TOKEN_TYPE] * vision_token_num

        input_ids += token_ids
        token_type_ids += [LANGUAGE_TOKEN_TYPE] * len(token_ids)

        return input_ids, images, token_type_ids

    def multimodal_embedding(
        self,
        input_ids: torch.Tensor,
        image_features: List[torch.Tensor],
        token_type_ids: torch.Tensor,
    ):
        token_type_ids = token_type_ids.reshape(1, -1)
        input_embeds = self.word_embedding(input_ids)
        if len(image_features) > 0:
            image_features = torch.stack(image_features, dim=0)
            image_features = rearrange(image_features, "b n d -> (b n) d")
            image_features = image_features.to(
                dtype=input_embeds.dtype, device=input_embeds.device
            )
            input_embeds = input_embeds.index_put(
                [token_type_ids == VISION_TOKEN_TYPE], image_features
            )
        return input_embeds

register_model(
    "cogvlm2",
    CogVLM2,
    ["CogVLMForCausalLM"],
    [
        "THUDM/cogvlm2-llama3-chat-19B",
        "THUDM/cogvlm2-llama3-chinese-chat-19B"
    ],
)

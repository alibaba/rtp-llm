import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from transformers import AutoTokenizer

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters, TemplateVersion
from maga_transformer.model_factory_register import register_model
from maga_transformer.models.cogvlm2_vit import CogVLM2ImageEmbedding
from maga_transformer.models.cogvlm2_weight import CogVLM2VitWeights, CogVLM2WeightInfo
from maga_transformer.models.llama import Llama
from maga_transformer.models.multimodal_mixin import MultiModalMixin
from maga_transformer.ops.comm.nccl_op import NcclOp

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1


class CogVLM2(Llama, MultiModalMixin):
    def __init__(self, config: GptInitModelParameters):
        self.visual = CogVLM2ImageEmbedding(config)
        self.nccl_op_ = NcclOp()
        config.vit_related_params.vit_weights = CogVLM2VitWeights(
            {"vit": self.visual.vit}
        )
        Llama.__init__(self, config)

    @classmethod
    def is_multimodal(cls) -> bool:
        return True

    @staticmethod
    def multimodal_modify_prompt_plugin(
        prompt: Union[List[Dict[str, Any]], str],
        images: List[str],
        img_token: str,
        **kwargs: Any,
    ) -> Tuple[str, List[Any]]:
        prompt, images = MultiModalMixin.multimodal_modify_prompt_plugin(
            prompt, images, img_token, **kwargs
        )
        if img_token in prompt:
            return prompt, images
        else:
            return prompt + (img_token + "\n") * len(images), images

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
            rotary_embedding_style=5,
            rotary_embedding_base=500000,
            has_post_decoder_layernorm=True,
            is_multimodal=True,
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

        try:
            template_version_str = config_json.get("template_version", "chat")
            config.template_version = TemplateVersion[template_version_str]
        except KeyError:
            raise Exception(f"unknown template_version: {template_version_str}")

        config.reserve_runtime_mem_mb = 2048

        vit_config = config_json["vision_config"]
        config.vit_related_params.config.update(vit_config)
        config.special_tokens.bos_token_id = config_json["bos_token_id"]
        config.special_tokens.pad_token_id = config_json["pad_token_id"]

        if isinstance(config_json['eos_token_id'], list):            
            config.special_tokens.eos_token_id = config_json['eos_token_id'][0]
            config.special_tokens.stop_words_list = [[x] for x in config_json['eos_token_id']]
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
        self, context_begin_position: int, context_end_position: int, token_type_ids: torch.Tensor
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

    def async_input_word_embedding(
        self,
        inputs: torch.Tensor,
        images: List[torch.Tensor],
        token_type_ids: torch.Tensor,
    ):
        return MultiModalMixin.async_input_word_embedding(
            self, inputs, images, token_type_ids
        )

    @torch.no_grad()
    def expand_token_id(
        self, token_ids: List[int], images: List[torch.Tensor]
    ) -> Tuple[List[int], List[torch.Tensor], List[int]]:
        if len(images) > 1:
            raise Exception("CogVLM2 support processes one image at a time")

        input_ids = [self.tokenizer.bos_token_id]
        token_type_ids = [LANGUAGE_TOKEN_TYPE]

        if len(images) > 0:
            patch_size: int = self.config.vit_related_params.config["patch_size"]
            image_size: int = self.config.vit_related_params.config["image_size"]

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

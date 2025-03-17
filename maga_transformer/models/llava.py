import os
import json
import torch
import re
from typing import List, Any, Dict, Tuple, Union
from transformers import AutoConfig, CLIPVisionConfig, AutoTokenizer

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.models.llava_weight import LlavaWeightInfo
from maga_transformer.models.llama import Llama
from maga_transformer.models.multimodal.multimodal_mixin import MultiModalMixin, BaseVitWeights
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.models.llava_vit import LlavaImageEmbedding
from maga_transformer.utils.util import to_torch_dtype
from maga_transformer.model_factory_register import register_model

class LlavaTokenizer(object):
    def __init__(self,
                 tokenzier_path: str,
                 mm_use_im_patch_token: bool,
                 mm_use_im_start_end: bool,
                 special_token_ids: Dict[str, Any],
                 special_tokens: Dict[str, Any],
                 bos_id: int = 1):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenzier_path)
        self.mm_use_im_patch_token = mm_use_im_patch_token
        self.mm_use_im_start_end = mm_use_im_start_end

        extra_tokens: List[str] = []
        if self.mm_use_im_patch_token:
            extra_tokens.extend(["<im_patch>"])
        if self.mm_use_im_start_end:
            extra_tokens.extend(["<im_start>", "<im_end>"])
        self.tokenizer.add_tokens(extra_tokens, special_tokens=True)

        self.image_token_index: int = special_token_ids["image_token_index"]
        self.ignore_token_index: int = special_token_ids["ignore_token_index"]
        self.default_image_token = special_tokens["default_mm_token"]
        self.default_im_start_token = special_tokens["default_im_start_token"]
        self.default_im_end_token = special_tokens["default_im_end_token"]
        self.bos_id = bos_id

    def encode(self, s: str, **kwargs) -> List[int]:
        replace_token = self.default_image_token
        if self.mm_use_im_start_end:
            replace_token = self.default_im_start_token + replace_token + self.default_im_end_token
        s = s.replace(self.default_image_token, replace_token)

        prompt_chunks: List[List[int]] = [self.tokenizer.encode(chunk) for chunk in s.split(self.default_image_token)]

        images = len(prompt_chunks) - 1

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        t: List[int] = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == self.bos_id:
            offset = 1
            t.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [self.image_token_index] * (offset + 1)):
            t.extend(x[offset:])

        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)

    def apply_chat_template(self, messages, **kwargs):
        return self.tokenizer.apply_chat_template(messages, **kwargs)

class Llava(Llama, MultiModalMixin):
    def _init_multimodal(self, config: GptInitModelParameters):
        self.mm_part = LlavaImageEmbedding(config)
        vit_weight_dict: Dict[str, Any] = {"mm_projector": self.mm_part.mm_projector}
        if config.mm_related_params.config["unfreeze_mm_vision_tower"] or \
            "mm_vision_tower" in config.mm_related_params.config["mm_tunable_parts"]:
            vit_weight_dict["vision_tower"] = self.mm_part.vision_tower
        if "unpad" in config.mm_related_params.config.get("mm_patch_merge_type", "flat"):
            vit_weight_dict["image_newline"] = self.mm_part.image_newline
        config.mm_related_params.vit_weights = BaseVitWeights(vit_weight_dict, True)

    @staticmethod
    def multimodal_modify_prompt_plugin(prompt: Union[List[Dict[str, Any]], str], images: List[str],
                                        img_token: str, **kwargs: Any) -> Tuple[str, List[Any]]:
        prompt, mm_inputs = MultiModalMixin.multimodal_modify_prompt_plugin(prompt, images, img_token, **kwargs)
        if img_token in prompt:
            return prompt, mm_inputs
        else:
            return prompt + (img_token + "\n") * len(images), mm_inputs

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
            has_post_decoder_layernorm=True
        )
        # hugggingface
        config_path = os.path.join(ckpt_path, "config.json")
        param_path = os.path.join(ckpt_path, "params.json")
        if os.path.exists(config_path):
            with open(config_path) as reader:
                content = reader.read()
                content = content.replace("LlavaForCausalLM", "LLaVAForCausalLM")
                config_json = json.loads(content)
            Llava.from_huggingface(config, config_json)
        else:
            raise Exception("llava parameter from unkown source")
        return config

    @staticmethod
    def get_weight_cls():
        return LlavaWeightInfo

    @staticmethod
    def from_huggingface(config: GptInitModelParameters, config_json: Dict[str, Any]):
        if "text_config" in config_json:
            text_config = config_json["text_config"]
            # if text_config.get("_name_or_path", "") != "":
            #     text_config = AutoConfig.from_pretrained(text_config["_name_or_path"]).to_dict()
            Llama.from_huggingface(config, text_config)

            vision_config = config_json["vision_config"]
            config.mm_related_params.config["vision_config"] = CLIPVisionConfig(vision_config)

        else:
            Llama.from_huggingface(config, config_json)

            mm_related_params_list = [
                ("mm_use_im_patch_token", False),
                ("mm_use_im_start_end", False),
                ("image_aspect_ratio", None),
                ("tune_mm_mlp_adapter", False),
                ("image_grid_pinpoints", []),
                ("mm_projector_type", "linear"),
                ("mm_patch_merge_type", "flat"),
                ("hidden_size", 0),
                ("mm_vision_select_layer", None),
                ("mm_vision_select_feature", "patch"),
                ("unfreeze_mm_vision_tower", False),
                ("mm_tunable_parts", ""),
                ("add_faster_video", False),
                ("mm_newline_position", "grid"),
                ("mm_spatial_pool_mode", "bilinear")
            ]

            for param_name, default_value in mm_related_params_list:
                config.mm_related_params.config[param_name] = config_json.get(param_name, default_value)

            config.mm_related_params.config["mm_hidden_size"] = config_json.get("mm_hidden_size", config_json["hidden_size"])
            config.mm_related_params.special_token_ids.update({"ignore_token_index": -100, "image_token_index": -200})
            config.mm_related_params.special_tokens.update({
                "default_mm_token": "<image>",
                "default_im_start_token": "<im_start>",
                "default_im_end_token": "<im_end>"
            })

            vis_tower_name = config_json.get("mm_vision_tower", config_json.get("vision_tower", None))
            img_expand_match = re.search("patch(\d+)-(\d+)", vis_tower_name)
            if img_expand_match:
                patch_size = int(img_expand_match.group(1))
                img_size = int(img_expand_match.group(2))
                config.mm_related_params.config["patch_size"] = patch_size
                config.mm_related_params.config["image_size"] = img_size
            config.mm_related_params.config["vit_tower_path"] = vis_tower_name
            config.mm_sep_tokens = [[-200]] # image_token_index

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        return LlavaTokenizer(config.tokenizer_path,
                              config.mm_related_params.config["mm_use_im_patch_token"],
                              config.mm_related_params.config["mm_use_im_start_end"],
                              config.mm_related_params.special_token_ids,
                              config.mm_related_params.special_tokens,
                              config.special_tokens.bos_token_id)

register_model("llava", Llava, ["LlavaLlamaForCausalLM"])

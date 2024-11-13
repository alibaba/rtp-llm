
import torch
import os
import json
from typing import List, Any, Tuple, Dict, Union
from transformers import AutoTokenizer

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.models.qwen import QWen
from maga_transformer.models.qwen_vl_weight import QWenVLWeightInfo, QwenVLVitWeight
from maga_transformer.models.qwen_vl_vit import VisionTransformer as QWen_VL_ViT
from maga_transformer.models.base_model import BaseModel, MultimodalInput
from maga_transformer.models.multimodal.multimodal_mixin import MultiModalMixin
from maga_transformer.models.multimodal.multimodal_common import ImageEmbeddingInterface
from maga_transformer.model_factory_register import register_model
from maga_transformer.utils.util import to_torch_dtype

class QwenVLImageEmbedding(ImageEmbeddingInterface):
    def __init__(self, config: GptInitModelParameters):
        self.vit = QWen_VL_ViT(**config.mm_related_params.config)
        self.config = config

    @property
    def _device(self):
        return self.vit.device

    @torch.no_grad()
    def image_embedding(self, images: List[Any]) -> torch.Tensor:
        images = self.vit.encode(images, self._device, self._data_type)
        return images

class QWen_VL(QWen, MultiModalMixin):
    def _init_multimodal(self, config: GptInitModelParameters):
        self.mm_part = QwenVLImageEmbedding(config)
        config.mm_related_params.vit_weights = QwenVLVitWeight({"vit": self.mm_part.vit})

    def load(self, device: str):
        if os.environ.get("VIT_TRT", "0") == "1":
            weights_info = self.get_weight_cls()(self.config, g_parallel_info.tp_size, g_parallel_info.tp_rank)
            self.init_mm_trt(
                weights_info, self.config.ckpt_path,
                self.config.mm_related_params, device, to_torch_dtype(self.config.data_type)
            )
        super().load(device=device)

    @staticmethod
    def multimodal_modify_prompt_plugin(prompt: Union[List[Dict[str, Any]], str], images: List[str],
                                        img_token: str, **kwargs: Any) -> Tuple[str, List[MultimodalInput]]:
        prompt, mm_inputs = MultiModalMixin.multimodal_modify_prompt_plugin(prompt, images, img_token, **kwargs)
        start_str = '<img>'
        end_str = '</img>'
        if img_token in prompt:
            split_prompts = prompt.split(img_token)
            if len(split_prompts) - 1 != len(images):
                raise Exception('num of ' + img_token + ' should equals to images num')
            res = split_prompts[0]
            idx = 0
            for split_prompt in split_prompts[1:]:
                res = res + start_str + images[idx] + end_str + split_prompt
                idx = idx + 1
            return res, mm_inputs
        else:
            prefix_prompt = ''
            if len(images) > 0:
                for i in range(len(images)):
                    prefix_prompt += 'Picture {i}:'.format(i = i + 1) + start_str + images[i] + end_str + '\n'

            tmp_prompt = prompt
            while start_str in tmp_prompt:
                start_idx = tmp_prompt.find(start_str)
                end_idx = tmp_prompt.find(end_str)
                if end_idx < start_idx:
                    raise Exception(f'unclosed tag <img> pair in {prompt}')
                images.append(tmp_prompt[start_idx + len(start_str): end_idx])
                tmp_prompt = tmp_prompt[end_idx + len(end_str):]

            return prefix_prompt + prompt, [MultimodalInput(image) for image in images]

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = GptInitModelParameters(
            head_num=0,
            size_per_head=0,
            layer_num=0,
            max_seq_len=0,
            vocab_size=0
        )
        QWen_VL._common_config(config, ckpt_path)
        return config

    @staticmethod
    def _common_config(config: GptInitModelParameters, ckpt_path: str) -> GptInitModelParameters:
        QWen._common_config(config, ckpt_path)
        QWen._from_hf(config, ckpt_path)
        QWen_VL._load_vit_param(config, ckpt_path)
        return config

    @staticmethod
    def _load_vit_param(config: GptInitModelParameters, ckpt_path: str):
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path) as reader:
            content = reader.read()
            config_json = json.loads(content)

        vit_config = config_json['visual']
        config.mm_related_params.config.update(vit_config)
        config.mm_related_params.special_token_ids.update({
            'image_start_id': vit_config['image_start_id'],
            'image_end_id': vit_config['image_start_id'] + 1,
            'image_pad_id': vit_config['image_start_id'] + 2})
        config.mm_related_params.special_tokens.update({'default_mm_token': '<img/>'})
        config.mm_sep_tokens = [[vit_config['image_start_id'], vit_config['image_start_id'] + 1]]

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        return AutoTokenizer.from_pretrained(config.tokenizer_path, trust_remote_code=True)

    @staticmethod
    def get_weight_cls():
        return QWenVLWeightInfo

    @staticmethod
    def eval_model_size(config: GptInitModelParameters):
        llm_size = BaseModel.eval_model_size(config)

        data_width = 4
        llm_size += QWen_VL.eval_vit_param_count(config) * data_width
        return llm_size

    @staticmethod
    def eval_vit_param_count(config: GptInitModelParameters):
        vit_config = config.mm_related_params.config
        embed_dim = vit_config["output_dim"]
        width = vit_config["width"]
        layers = vit_config["layers"]
        patch_size = vit_config["patch_size"]
        mlp_ratio = vit_config["mlp_ratio"]
        mlp_width = int(mlp_ratio * width)

        llm_size = (3 * width * patch_size ** 2 + width * 2)
        llm_size += (layers * (width * 2 * 2 + width ** 2 * 4 + width * 4 + mlp_width * width * 2 + mlp_width + width))
        llm_size += (width * embed_dim + embed_dim ** 2 + embed_dim + embed_dim * 2 * 3)
        return llm_size

    @staticmethod
    def eval_model_param_count(config: GptInitModelParameters):
        llm_param_count = BaseModel.eval_model_param_count(config)
        llm_param_count += QWen_VL.eval_vit_param_count(config)

        return llm_param_count

register_model('qwen_vl', QWen_VL, ["QWenMLMHeadModel"])

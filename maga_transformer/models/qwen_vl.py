
import torch
import os
import re
import json
import logging
from typing import List, Any, Tuple, Dict, Optional, Union
from PIL import Image

from transformers import AutoTokenizer

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.models.qwen import QWen
from maga_transformer.models.qwen_vl_weight import QWenVLWeightInfo, QwenVLVitWeight
from maga_transformer.models.qwen_vl_vit import VisionTransformer as QWen_VL_ViT
from maga_transformer.models.qwen_vl_vit_engine import VITEngine
from maga_transformer.models.base_model import BaseModel
from maga_transformer.models.multimodal_mixin import MultiModalMixin, BaseImageEmbedding
from maga_transformer.model_factory_register import register_model
from maga_transformer.utils.util import get_device, to_torch_dtype, get_mem_info
from maga_transformer.ops.comm.nccl_op import NcclOp
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.utils.model_weights_loader import get_model_weights_loader
from maga_transformer.utils.database import CkptDatabase

class QwenVLImageEmbedding(BaseImageEmbedding):
    def __init__(self, config: Dict[str, Any]):
        self.vit = QWen_VL_ViT(**config)
    
    def image_embedding(self, images: List[Any], device) -> torch.Tensor:
        if len(images) != 0:
            images = self.vit.encode(images)
            assert images.shape[0] == len(images)
        return images.to(device=device)

class QWen_VL(QWen, MultiModalMixin):
    def __init__(self, config: GptInitModelParameters):
        self.visual = QwenVLImageEmbedding(config.vit_related_params.config)
        self.nccl_op_ = NcclOp()
        config.vit_related_params.vit_weights = QwenVLVitWeight({"vit": self.visual.vit})
        
        QWen.__init__(self, config)

    @classmethod
    def is_multimodal(cls) -> bool:
        return True
    
    def load(self, device: Optional[Union[str, int, torch.device]] = 'cuda:0'):
        if os.environ.get("VIT_TRT", "0") == "1": # 这里还需要判断有没有tensorrt
            try:
                import tensorrt
                self.init_vit_trt()
            except Exception as e:
                logging.info(f"init vit trt error: {e}")
        super().load(device=device)
    
    def _prepare_model_weight_loader(self, device: Optional[Union[str, int, torch.device]] = 'cuda:0'):
        device = device or get_device()
        # Load weight only for self.visual
        compute_dtype = to_torch_dtype(self.config.data_type or self.dtype)
        weights_info = self.get_weight_cls()(self.config, g_parallel_info.tp_size, g_parallel_info.tp_rank)
        database = CkptDatabase(self.config.ckpt_path)
        weight_loader = get_model_weights_loader(weights_info, database, compute_dtype=compute_dtype)
        return weight_loader
    
    def init_vit_trt(self, device: Optional[Union[str, int, torch.device]] = 'cuda:0'):
        os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
        if VITEngine.should_generate_engine():
            assert type(self.visual) == QwenVLImageEmbedding
            weight_loader = self._prepare_model_weight_loader(device=device)
            ctype = to_torch_dtype(self.config.data_type or self.dtype)
            vit_weight = self.config.vit_related_params.vit_weights
            ckpt_prefix = vit_weight.ckpt_prefix
            ft_prefix = vit_weight.ft_prefix
            vit_weight_names = vit_weight.weight_names
            for vit_weight_name in vit_weight_names:
                ckpt_weight_name = ckpt_prefix + vit_weight_name
                param_name = ft_prefix + vit_weight_name
                param_name = re.sub(r'\.\d+\.', lambda x: '[' + x.group(0)[1:-1] + '].', param_name)
                tensor = weight_loader.load_tensor(ckpt_weight_name)[0]
                param = eval(param_name)
                param.data = tensor.reshape(param.data.shape).to(ctype).to('cuda:0')
        
        vit_visual = VITEngine(self.visual.vit, self.config.vit_related_params.config.get("image_size"))
        
        del self.visual
        self.visual = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        self.visual = vit_visual
        self.config.vit_related_params.vit_weights = None
    
    def load_vit_weight(self, ctype: str):
        if type(self.visual) == VITEngine:
            # No need to load weight for VITEngine, its weight is inside trt engine.
            return
        else:
            super().load_vit_weight(ctype=ctype)
    
    @staticmethod
    def multimodal_modify_prompt_plugin(prompt: str, **kwargs: Any) -> Tuple[str, List[Any]]:
        prompt, images = MultiModalMixin.multimodal_modify_prompt_plugin(prompt, **kwargs)
        img_token: str = kwargs.get('img_token')
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
            return res, images
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

            return prefix_prompt + prompt, images
    
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = GptInitModelParameters(
            head_num=0,
            size_per_head=0,
            layer_num=0,
            max_seq_len=0,
            vocab_size=0,
            is_multimodal=True
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
        config.vit_related_params.config.update(vit_config)
        config.vit_related_params.vit_special_token_ids.update({
            'image_start_id': vit_config['image_start_id'],
            'image_end_id': vit_config['image_start_id'] + 1,
            'image_pad_id': vit_config['image_start_id'] + 2})
        config.vit_related_params.vit_special_tokens.update({'default_image_token': '<img/>'})

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        return AutoTokenizer.from_pretrained(config.tokenizer_path, trust_remote_code=True)
    
    @staticmethod
    def get_weight_cls():
        return QWenVLWeightInfo

    def async_input_word_embedding(self, inputs: torch.Tensor, images: List[List[Any]]):
        return MultiModalMixin.async_input_word_embedding(self, inputs, images)

    def input_word_embedding(self, inputs: torch.Tensor, images: List[List[Any]]):
        return MultiModalMixin.input_word_embedding(self, inputs, images)
    
    def expand_token_id(self, token_ids: List[int], images: List[Image.Image]) -> Tuple[List[int], Union[torch.Tensor, List[torch.Tensor]]]:
        # if len(images) > 0:
        #     image_features = self.visual.image_embedding(images, self.device)
        return token_ids, images
    
    # QWen_VL tokenizer encode image urls into tokens, so that multimodal_embedding don't need images as input
    def multimodal_embedding(self, input_ids: torch.Tensor, images: List[List[Any]]):
        img_start_id: int = self.config.vit_related_params.vit_special_token_ids['image_start_id']
        img_end_id: int = self.config.vit_related_params.vit_special_token_ids['image_end_id']
        bos_pos = torch.where(input_ids == img_start_id)
        eos_pos = torch.where(input_ids == img_end_id)
        assert (bos_pos[0] == eos_pos[0]).all()
        img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)

        image_data = []
        for image in images:
            image_data.extend(image)

        if len(image_data) != 0:
            images = self.visual.image_embedding(image_data, self.device)
            assert images.shape[0] == len(images)

        input_embeds = self.word_embedding(input_ids)

        if images != []:
            for idx, (i, a, b) in enumerate(img_pos):
                input_embeds[i][a + 1: b] = images[idx]

        return input_embeds

    @staticmethod
    def eval_model_size(config: GptInitModelParameters):
        llm_size = BaseModel.eval_model_size(config)
        
        vit_config = config.vit_related_params.config
        embed_dim = vit_config["output_dim"]
        width = vit_config["width"]
        layers = vit_config["layers"]
        patch_size = vit_config["patch_size"]
        mlp_ratio = vit_config["mlp_ratio"]
        mlp_width = int(mlp_ratio * width)
        data_width = 4
        
        llm_size += (3 * width * patch_size ** 2 + width * 2) * data_width
        llm_size += (layers * (width * 2 * 2 + width ** 2 * 4 + width * 4 + mlp_width * width * 2 + mlp_width + width)) * data_width
        llm_size += (width * embed_dim + embed_dim ** 2 + embed_dim + embed_dim * 2 * 3) * data_width

        return llm_size
    
register_model('qwen_vl', QWen_VL)

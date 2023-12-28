import os
import time
import logging
import json
import functools
import torch
import re
import requests

from typing import List, Any, Dict, Tuple
from PIL import Image
from io import BytesIO
from transformers.models.llama.tokenization_llama import LlamaTokenizer

from maga_transformer.utils.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.models.llava_weight import LlavaWeightInfo, LlavaVitWeights
from maga_transformer.models.llama import Llama
from maga_transformer.models.base_model import BaseTokenizer, BaseModel
from maga_transformer.models.multimodal_mixin import MultiModalMixin
from maga_transformer.ops.comm.nccl_op import NcclOp
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.utils.util import to_cuda, to_cpu
from maga_transformer.models.llava_vit_encoder import build_vision_tower, process_batch_images
from maga_transformer.utils.util import to_torch_dtype
from maga_transformer.model_factory_register import register_model

class IdentityMap(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}
    
def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = config.get('mm_projector_type', 'linear')

    if projector_type == 'linear':
        return torch.nn.Linear(config['mm_hidden_size'], config['hidden_size'], device='cuda:0'), 1

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [torch.nn.Linear(config['mm_hidden_size'], config['hidden_size'], device='cuda:0')]
        for _ in range(1, mlp_depth):
            modules.append(torch.nn.GELU())
            modules.append(torch.nn.Linear(config['hidden_size'], config['hidden_size'], device='cuda:0'))
        return torch.nn.Sequential(*modules), mlp_depth

    if projector_type == 'identity':
        return IdentityMap(), 0

    raise ValueError(f'Unknown projector type: {projector_type}')
    
class LlavaTokenizer(BaseTokenizer):
    def __init__(self, 
                 tokenzier_path: str, 
                 mm_use_im_start_end: bool, 
                 image_expand: int, 
                 vit_special_token_ids: Dict[str, Any],
                 vit_special_tokens: Dict[str, Any]):
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenzier_path)
        self.mm_use_im_start_end = mm_use_im_start_end
        self.image_expand = image_expand
        self.image_token_index: int = vit_special_token_ids['image_token_index']
        self.ignore_token_index: int = vit_special_token_ids['ignore_token_index']
        self.default_image_token = vit_special_tokens['default_image_token']
        self.default_im_start_token = vit_special_tokens['default_im_start_token']
        self.default_im_end_token = vit_special_tokens['default_im_end_token']
        self.bos_id = self.tokenizer.sp_model.bos_id()
        

    def encode(self, s: str) -> List[int]:
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

        t.extend([self.ignore_token_index] * images * (self.image_expand - 1))

        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)

class Llava(Llama, MultiModalMixin):
    def __init__(self, config: GptInitModelParameters):
        self.vision_tower = config.vit_related_params.vision_tower
        self.mm_projector = config.vit_related_params.mm_projector
        self.nccl_op_ = NcclOp() 

        Llama.__init__(self, config)
    
    @staticmethod
    def multimodal_modify_prompt_plugin(prompt: str, **kwargs: Any) -> Tuple[str, List[Any]]:
        img_token: str = kwargs.get('img_token')
        prompt, images = MultiModalMixin.multimodal_modify_prompt_plugin(prompt, **kwargs)
        img = kwargs.get('image', [])
        if img_token in prompt:
            return prompt, images
        else:
            return prompt + (img_token + '\n') * len(img), images

    def load_vit_weight(self, ctype: str):
        config = self.config
        proj_layers = config.vit_related_params.proj_layers
        interval = config.vit_related_params.vit_layer_id_interval

        def _safe_load_from_module(param: torch.nn.Parameter, fname: str, ctype: torch.dtype):
            param.data = self.weight.steal_pytorch_weight(fname).reshape(param.data.shape).to(ctype).to('cuda:0')

        for i in range(0, proj_layers):
            w = LlavaVitWeights.vit_proj_w.format(i = i * interval)
            b = LlavaVitWeights.vit_proj_b.format(i = i * interval)
            _safe_load_from_module(self.mm_projector[i * interval].weight, w, ctype)
            _safe_load_from_module(self.mm_projector[i * interval].bias, b, ctype)

    @staticmethod
    def _create_config(ckpt_path):
        config = GptInitModelParameters(
            head_num=0,
            size_per_head=0,
            layer_num=0,
            max_seq_len=0,
            vocab_size=0,
            ckpt_path=ckpt_path,
            activation_type='SiGLU',
            use_gated_activation=True,
            norm_type='rmsnorm',
            rotary_embedding_dim=128,
            rotary_embedding_style=1,
            has_post_decoder_layernorm=True,
            is_multimodal=True
        )
        # hugggingface
        config_path = os.path.join(ckpt_path, 'config.json')
        param_path = os.path.join(ckpt_path, 'params.json')
        if os.path.exists(config_path):
            with open(config_path) as reader:
                content = reader.read()
                content = content.replace("LlavaForCausalLM", "LLaVAForCausalLM")
                config_json = json.loads(content)
            Llava.from_huggingface(config, config_json)
        elif os.path.exists(param_path):
            logging.info("llava not find config.json, use default config")
            with open(param_path) as reader:
                param_json = json.loads(reader.read())
            Llava.from_params(config, param_json)
        else:
            raise Exception("llava parameter from unkown source")
        return config

    @staticmethod
    def get_weight_cls():
        return LlavaWeightInfo
   
    @staticmethod
    def from_huggingface(config: GptInitModelParameters, config_json: Dict[str, Any]):
        Llama.from_huggingface(config, config_json)
        config.vit_related_params.mm_use_im_start_end = config_json.get('mm_use_im_start_end', False)
        config.vit_related_params.image_aspect_ratio = config_json.get('image_aspect_ratio', None)
        config.vit_related_params.tune_mm_mlp_adapter = config_json.get('tune_mm_mlp_adapter', False)
        config.vit_related_params.mm_projector_type = config_json.get('mm_projector_type', 'linear')
        config.vit_related_params.mm_hidden_size = config_json.get('mm_hidden_size', config_json['hidden_size'])
        config.vit_related_params.vit_layer_id_interval = 2

        config.vit_related_params.vision_tower = build_vision_tower(config_json)
        config.vit_related_params.mm_projector, config.vit_related_params.proj_layers = build_vision_projector(config_json)

        config.vit_related_params.vit_special_token_ids.update({'ignore_token_index': -100, 'image_token_index': -200})
        config.vit_related_params.vit_special_tokens.update({
            'default_image_token': '<image>', 
            'default_im_start_token': '<im_start>', 
            'default_im_end_token': '<im_end>'
        })

        vis_tower_name = config_json.get('mm_vision_tower', config_json.get('vision_tower', None))
        img_expand_match = re.search('patch(\d+)-(\d+)', vis_tower_name)
        if img_expand_match:
            patch_size = int(img_expand_match.group(1))
            img_size = int(img_expand_match.group(2))
            config.vit_related_params.img_expand_len = (img_size // patch_size) ** 2
        config.vit_related_params.vit_tower_path = vis_tower_name

    def load_tokenizer(self):
        self.tokenizer = LlavaTokenizer(self.config.tokenizer_path, 
                                        self.config.vit_related_params.mm_use_im_start_end, 
                                        self.config.vit_related_params.img_expand_len, 
                                        self.config.vit_related_params.vit_special_token_ids,
                                        self.config.vit_related_params.vit_special_tokens)

    def encode_images(self, images):
        if images.shape[0] == 0:
            return images
        image_features = self.vision_tower(images).to(device=self.device)
        image_features = self.mm_projector(image_features)
        return image_features
    
    def async_input_word_embedding(self, inputs: torch.Tensor, images: List[List[str]]):
        inputs = inputs.reshape(1, -1)
        if g_parallel_info.tp_size <= 1:
            return self.multimodal_embedding(inputs, images).squeeze(0)

        if g_parallel_info.tp_rank == 0:
            embedding_tensor = self.multimodal_embedding(inputs, images).squeeze(0)
        else:
            embedding_tensor = torch.zeros((inputs.shape[1], self.config.head_num * self.config.size_per_head), dtype=torch.float16, device="cuda:0")
        self.nccl_op_.broadcast_tp([embedding_tensor])
        return embedding_tensor
        
    def input_word_embedding(self, inputs: torch.Tensor, images: List[List[str]]):
        return self.multimodal_embedding(inputs, images)

    def multimodal_embedding(
        self, input_ids: torch.Tensor, images: List[List[str]]
    ):
        vision_tower = self.vision_tower
        image_token_index = self.config.vit_related_params.vit_special_token_ids['image_token_index']
        ignore_token_index = self.config.vit_related_params.vit_special_token_ids['ignore_token_index']

        if images == [] and (input_ids == image_token_index).sum() != 0:
            raise ValueError("Number of images does not match number of <image> tokens in prompt") 

        image_data = []
        for i in range(input_ids.shape[0]):
            if (input_ids[i] == image_token_index).sum() != len(images[i]):
                raise ValueError("Number of images does not match number of <image> tokens in prompt")

        for i in range(len(images)):    
            now_image_data = []
            for image in images[i]:
                if image.startswith('/'):
                    now_image_data.append(Image.open(open(image, 'rb')))
                else:
                    now_image_data.append(Image.open(BytesIO(requests.get(image).content)))
            image_data.append(now_image_data)
        
        images = process_batch_images(image_data, self.config.vit_related_params.image_aspect_ratio, self.vision_tower.image_processor, self.device)

        image_features = []
        for query_images in images:
            image_features.append(self.encode_images(query_images))

        new_input_embeds = []

        tune_mm_mlp_adapter = getattr(self.config.vit_related_params, 'tune_mm_mlp_adapter', False)
        mm_use_im_start_end = getattr(self.config.vit_related_params, 'mm_use_im_start_end', False)
        append_extra_tokens = tune_mm_mlp_adapter and mm_use_im_start_end

        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_input_ids = cur_input_ids[~(cur_input_ids == ignore_token_index)]
            image_token_indices = torch.where(cur_input_ids == image_token_index)[0]
            cur_new_input_embeds = []
            cur_image_idx = 0
            if len(image_features[batch_idx]) == 0:
                cur_new_input_embeds = self.word_embedding(cur_input_ids)
            else:
                while image_token_indices.numel() > 0:
                    cur_image_features = image_features[batch_idx][cur_image_idx]
                    image_token_start = image_token_indices[0]
                    if append_extra_tokens:
                        cur_new_input_embeds.append(self.word_embedding(cur_input_ids[:image_token_start-1]).detach())
                        cur_new_input_embeds.append(self.word_embedding(cur_input_ids[image_token_start-1:image_token_start]))
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_input_embeds.append(self.word_embedding(cur_input_ids[image_token_start+1:image_token_start+2]))
                    else:
                        cur_new_input_embeds.append(self.word_embedding(cur_input_ids[:image_token_start]))
                        cur_new_input_embeds.append(cur_image_features)
                    
                    cur_image_idx += 1
                    if append_extra_tokens:
                        cur_input_ids = cur_input_ids[image_token_start+2:]
                    else:
                        cur_input_ids = cur_input_ids[image_token_start+1:]
                    image_token_indices = torch.where(cur_input_ids == image_token_index)[0]
                
                if cur_input_ids.numel() > 0:
                    if append_extra_tokens:
                        cur_new_input_embeds.append(self.word_embedding(cur_input_ids).detach())
                    else:
                        cur_new_input_embeds.append(self.word_embedding(cur_input_ids))

                cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
                cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)
            max_input_len = max(x.shape[0] for x in new_input_ids)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)

        assert input_ids.shape[1] == new_input_embeds.shape[1]
        return new_input_embeds.type(to_torch_dtype(self.config.data_type))

    @staticmethod
    def eval_model_size(config: GptInitModelParameters):
        llm_size = BaseModel.eval_model_size(config)
        vit_json_path: str = config.vit_related_params.vit_tower_path + '/config.json'
        if os.path.exists(vit_json_path):
            with open(vit_json_path) as reader:
                content = reader.read()
                config_json = json.loads(content)
                vision_config_dict = config_json['vision_config_dict']

                hidden_size = vision_config_dict['hidden_size']
                patch_num = vision_config_dict['image_size'] // vision_config_dict['patch_size']
                conv_size = patch_num ** 2 * hidden_size * 3
                pos_emb_size = patch_num ** 2 * hidden_size
                ln_size = 2 * hidden_size * 2

                clip_encoder_size = vision_config_dict['num_hidden_layers'] * (hidden_size ** 2 * 4 + hidden_size * 2 * 2 + hidden_size * vision_config_dict['intermediate_size'] * 2)

                data_type = vision_config_dict['torch_dtype']
                if data_type == 'float32':
                    data_type_size = 4
                elif data_type == 'int8':
                    data_type_size = 1
                else:
                    data_type_size = 2
                llm_size += (conv_size + pos_emb_size + ln_size + clip_encoder_size) * data_type_size

        return llm_size
    
register_model('llava', Llava)
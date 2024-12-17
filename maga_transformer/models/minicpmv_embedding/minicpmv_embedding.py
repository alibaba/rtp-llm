import json
import os
from typing import Any, Dict, List, Tuple, Union

import torch
import math
from PIL import Image
from transformers import AutoTokenizer
from maga_transformer.config.gpt_init_model_parameters import \
    GptInitModelParameters
from maga_transformer.distribute.worker_info import ParallelInfo, g_parallel_info
from maga_transformer.model_factory_register import register_model
from maga_transformer.models.multimodal.multimodal_mixin import MultiModalMixin, BaseVitWeights
from maga_transformer.models.multimodal.multimodal_common import MultiModalEmbeddingInterface, mm_lock
from maga_transformer.utils.multimodal_util import MMUrlType
from transformers import LlamaTokenizer
# from maga_transformer.models.minicpmv.modeling_navit_siglip import SiglipVisionTransformer, SiglipVisionConfig
from maga_transformer.models.minicpmv_embedding.resampler import Resampler
from maga_transformer.models.multimodal.multimodal_mixin import BaseVitWeights, BaseMultiModalWeightInfo
from maga_transformer.utils.multimodal_util import MMUrlType, vit_emb_cache_, get_bytes_io_from_url
from torchvision import transforms
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from typing import Any, Dict, List, Type, Optional
from maga_transformer.models.downstream_modules.custom_module import CustomModule
from maga_transformer.models.downstream_modules.embedding.minicpmv_embedding_module import MiniCPMVModule, slice_image
from maga_transformer.models.llama_weight import LlamaWeightInfo
from maga_transformer.models.llama import Llama
from maga_transformer.models.minicpmv.minicpmv import encode_video

import timm
# for faster batch inference
from concurrent.futures import ThreadPoolExecutor

from maga_transformer.config.task_type import TaskType


class LlamaTokenizerWrapper(LlamaTokenizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.im_start = "<image>"
        self.im_end = "</image>"
        self.ref_start = "<ref>"
        self.ref_end = "</ref>"
        self.box_start = "<box>"
        self.box_end = "</box>"
        self.quad_start = "<quad>"
        self.quad_end = "</quad>"
        self.point_start = "<point>"
        self.point_end = "</point>"
        self.slice_start = "<slice>"
        self.slice_end = "</slice>"

    @property
    def eos_id(self):
        return self.sp_model.eos_id()

    @property
    def bos_id(self):
        return self.sp_model.bos_id()

    @property
    def unk_id(self):
        return self.sp_model.unk_id()

    @property
    def im_start_id(self):
        return self._convert_token_to_id(self.im_start)

    @property
    def im_end_id(self):
        return self._convert_token_to_id(self.im_end)


class ImageEmbeddingInterface(MultiModalEmbeddingInterface):

    def __init__(self, config: GptInitModelParameters):
        self.config = config
        config = config.mm_related_params.config
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN,
                                 std=IMAGENET_INCEPTION_STD),
        ])
        self.vision_encoder = config['vision_encoder']
        self.drop_vision_last_layer = config['drop_vision_last_layer']
        self.vpm = self.init_vision_module()
        self.vision_dim = self.vpm.embed_dim
        self.embed_dim = config['llm_hidden_size']
        self.query_num = config['query_num']
        self.max_slice_nums = config['max_slice_nums']
        self.scale_resolution = config['scale_resolution']
        self.patch_size = config['patch_size']
        self.slice_mode = config['slice_mode']

        self.resampler = Resampler(grid_size=int(math.sqrt(self.query_num)),
                                   embed_dim=self.embed_dim,
                                   num_heads=self.embed_dim // 128,
                                   kv_dim=self.vision_dim,
                                   adaptive=True)

    @property
    def _device(self):
        return next(self.vpm.parameters()).device

    def init_vision_module(self):
        model = timm.create_model(self.vision_encoder,
                                  pretrained=False,
                                  num_classes=0,
                                  dynamic_img_size=True,
                                  dynamic_img_pad=True)

        if isinstance(model, timm.models.VisionTransformer):
            if model.attn_pool is not None:
                model.attn_pool = torch.nn.Identity()

        if self.drop_vision_last_layer:
            model.blocks = model.blocks[:-1]

        return model

    @torch.inference_mode()
    def mm_embedding(self, url: str, mm_type: MMUrlType, **kwargs):
        dtype = self._data_type
        if g_parallel_info.tp_rank > 0:
            return torch.Tensor([])
        cached_res = vit_emb_cache_.check_cache(url)
        if cached_res is None:
            cached_url_res = get_bytes_io_from_url(url)
            cached_url_res = self._mm_preprocess(cached_url_res, mm_type)
            with mm_lock:
                features = self.mm_process(cached_url_res,
                                        mm_type=mm_type,
                                        **kwargs)
            if isinstance(features, list):
                features = torch.stack(features).to(dtype).contiguous()
            vit_emb_cache_.insert_cache(url, features)
            return (features, None)
        else:
            return (cached_res, None)
        
    def _mm_preprocess(self, data, type, **kwargs):
        if type == MMUrlType.IMAGE:
            return Image.open(data).convert("RGB")
        elif type == MMUrlType.VIDEO:
            return encode_video(data) 

    @torch.inference_mode()
    def mm_process(self, mm_input, **kwargs):
        mm_type = kwargs.get("mm_type")
        if mm_type == MMUrlType.DEFAULT:
            if isinstance(mm_input, list):
                return self.image_embedding(mm_input)
            else:
                return self.image_embedding([mm_input])
        elif mm_type == MMUrlType.IMAGE:
            if isinstance(mm_input, list):
                raise Exception("expect single image input, but get a list")
            return self.image_embedding([mm_input])
        elif mm_type == MMUrlType.VIDEO:
            if not isinstance(mm_input, list):
                raise Exception("expect video input, but get a single image")
            return self.image_embedding(mm_input)
        else:
            raise Exception("unknown mm url type")

    def get_vision_embedding(self, pixel_values):
        res = []
        dtype = self._data_type

        # first slice
        H, W = pixel_values[0].shape[-2:]
        tgt_size = (math.ceil(H / self.vpm.patch_embed.patch_size[0]),
                    math.ceil(W / self.vpm.patch_embed.patch_size[0]))

        vision_embedding = self.vpm.forward_features(
            pixel_values[0].unsqueeze(0).type(dtype))
        res.append(self.resampler(vision_embedding, tgt_size)[0])

        # remaining slices as a batch
        if len(pixel_values) > 1:

            H, W = pixel_values[1].shape[-2:]
            tgt_size = (math.ceil(H / self.vpm.patch_embed.patch_size[0]),
                        math.ceil(W / self.vpm.patch_embed.patch_size[0]))
            vision_embedding = self.vpm.forward_features(
                torch.stack(pixel_values[1:], dim=0).type(dtype))
            vision_embedding = self.resampler(vision_embedding, tgt_size)
            for i in range(len(pixel_values) - 1):
                res.append(vision_embedding[i])
        return res

    @torch.no_grad()
    def image_embedding(self, images: List[Any]) -> List[torch.Tensor]:
        new_images_list = []
        for image in images:
            if self.slice_mode:
                source_image, patches, best_grid = slice_image(
                    image,
                    self.max_slice_nums,
                    self.scale_resolution,
                    self.patch_size,
                )
                slice_images = [source_image]
                if len(patches) > 0:
                    for i in range(len(patches)):
                        for j in range(len(patches[0])):
                            slice_images.append(patches[i][j])
                new_images_list.append(slice_images)
            else:
                new_images_list.append([image])
        pixel_values_list = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            for img_batch in new_images_list:
                img_inps = list(executor.map(self.transform, img_batch))
                for i in range(len(img_inps)):
                    img_inps[i] = img_inps[i].to(self._device)
                pixel_values_list.append(img_inps if img_inps else [])
        vision_hidden_states = []
        for pixel_values in pixel_values_list:
            if len(pixel_values) > 0:
                vision_hidden_states.extend(
                    self.get_vision_embedding(pixel_values))
            else:
                vision_hidden_states.append([])
        return vision_hidden_states


class MiniCPMVVitWeight(BaseVitWeights):

    def _set_weight_prefix(self):
        self._ckpt_prefix = ""
        self._ft_prefix = "self.mm_part."


class MiniCPMVWeightInfo(LlamaWeightInfo, BaseMultiModalWeightInfo):

    def __init__(self, config, tp_size, tp_rank):
        LlamaWeightInfo.__init__(self, config, tp_size, tp_rank, prefix="llm.")
        BaseMultiModalWeightInfo.__init__(self, config)

    def _get_weight_info(self):
        llama_vl_weight = super()._get_weight_info()
        self._get_vit_info(llama_vl_weight)
        return llama_vl_weight


class MiniCPMVEmbedding(Llama, MultiModalMixin):

    def __init__(self, config: GptInitModelParameters):
        Llama.__init__(self, config)
        self.im_start = "<image>"
        self.im_end = "</image>"
        self.slice_start = "<slice>"
        self.slice_end = "</slice>"
        # self.im_start_id = self.tokenizer._convert_token_to_id(self.im_start)
        # self.im_end_id = self.tokenizer._convert_token_to_id(self.im_end)
        # self.slice_start_id = self.tokenizer._convert_token_to_id(self.slice_start)
        # self.slice_end_id = self.tokenizer._convert_token_to_id(self.slice_end)

        self.im_start_id = self.tokenizer.im_start_id
        self.im_end_id = self.tokenizer.im_end_id
        self.slice_start_id = self.tokenizer._convert_token_to_id(
            self.slice_start)
        self.slice_end_id = self.tokenizer._convert_token_to_id(self.slice_end)

        self.config.mm_sep_tokens = [[self.im_start_id, self.im_end_id]
                                     # [self.slice_start_id, self.slice_end_id]
                                     ]

    def _init_multimodal(self, config: GptInitModelParameters):
        self.mm_part = ImageEmbeddingInterface(config)
        config.mm_related_params.vit_weights = MiniCPMVVitWeight({
            "vpm":
            self.mm_part.vpm,
            "resampler":
            self.mm_part.resampler
        })

    @staticmethod
    def get_weight_cls():
        return MiniCPMVWeightInfo

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path,
                                                  verbose=False,
                                                  trust_remote_code=True,
                                                  use_fast=True)
        return tokenizer

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = GptInitModelParameters(
            head_num=0,
            size_per_head=0,
            layer_num=0,
            max_seq_len=0,
            vocab_size=0,
            ckpt_path=ckpt_path,
            activation_type='SiGLU',
            norm_type='rmsnorm',
            rotary_embedding_dim=128,
            rotary_embedding_style=1,
            has_post_decoder_layernorm=True,
        )
        config_path = os.path.join(ckpt_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path) as reader:
                content = reader.read()
                config_json = json.loads(content)
                Llama.from_huggingface(config, config_json)
                config.input_embedding_scalar = config_json.get("scale_emb", 1)
                config.residual_scalar = config_json.get("scale_depth", 1.4) / math.sqrt(config.layer_num)
                # config.activation_type = config_json["hidden_act"]
                MiniCPMVEmbedding._init_vit_params(config, config_json)
        else:
            raise Exception("no config.json found")
        return config

    @staticmethod
    def _init_vit_params(config: GptInitModelParameters,
                         config_json: Dict[str, Any]):
        # config.mm_related_params.config = config_json["vision_config"]
        config.mm_related_params.config["llm_hidden_size"] = config_json[
            "hidden_size"]
        config.mm_related_params.config["query_num"] = config_json["query_num"]
        config.mm_related_params.config["ckpt_path"] = config.ckpt_path
        config.mm_related_params.config["max_slice_nums"] = config_json[
            "max_slice_nums"]
        config.mm_related_params.config["scale_resolution"] = config_json[
            "scale_resolution"]
        config.mm_related_params.config["patch_size"] = config_json[
            "patch_size"]
        config.mm_related_params.config["slice_mode"] = config_json[
            "slice_mode"]
        config.mm_related_params.config["vision_encoder"] = config_json[
            "vision_encoder"]
        config.mm_related_params.config[
            "drop_vision_last_layer"] = config_json["drop_vision_last_layer"]

    def load_custom_module(self) -> Optional[CustomModule]:
        return MiniCPMVModule(self.config, self.tokenizer)
        # return super().load_custom_module()

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        return LlamaTokenizerWrapper.from_pretrained(config.tokenizer_path)


register_model('minicpmv_embedding', MiniCPMVEmbedding, ["MiniCPMVEmbedding"])

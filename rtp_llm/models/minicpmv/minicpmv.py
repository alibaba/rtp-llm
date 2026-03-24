import json
import os
from typing import Any, Dict, List

import torch
from PIL import Image
from transformers import AutoProcessor

from rtp_llm.config.model_config import ModelConfig, VitParameters
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.minicpmv.modeling_navit_siglip import (
    SiglipVisionConfig,
    SiglipVisionTransformer,
)
from rtp_llm.models.minicpmv.resampler import Resampler
from rtp_llm.models.multimodal.multimodal_common import (
    MultiModalEmbeddingInterface,
    mm_lock,
    timeout_decorator,
)
from rtp_llm.models.multimodal.multimodal_mixin import (
    BaseMultiModalWeightInfo,
    BaseVitWeights,
    MultiModalMixin,
)
from rtp_llm.models.qwen_v2 import QWenV2, QWenV2Weight

# minicpmv need to calculate num of frames to renderer input prompt, it must be preprocess first in frontend
from rtp_llm.openai.renderers.minicpmv_renderer import encode_video
from rtp_llm.ops import (
    DeviceResourceConfig,
    FMHAConfig,
    HWKernelConfig,
    KVCacheConfig,
    ModelSpecificConfig,
    MoeConfig,
    ParallelismConfig,
    RuntimeConfig,
)
from rtp_llm.utils.multimodal_util import (
    MMUrlType,
    get_bytes_io_from_url,
    vit_emb_cache_,
)


class ImageEmbeddingInterface(MultiModalEmbeddingInterface):

    def __init__(self, mm_related_params: VitParameters):
        self.mm_related_params = mm_related_params
        config = mm_related_params.config
        self.vision_config = SiglipVisionConfig(**config)
        self.processor = AutoProcessor.from_pretrained(
            config["ckpt_path"], trust_remote_code=True
        )
        self.vpm = SiglipVisionTransformer(self.vision_config)
        self.embed_dim = config["llm_hidden_size"]
        self.query_num = config["query_num"]
        self.vision_dim = self.vision_config.hidden_size
        self.resampler = Resampler(
            num_queries=self.query_num,
            embed_dim=self.embed_dim,
            num_heads=self.embed_dim // 128,
            kv_dim=self.vision_dim,
            adaptive=True,
        )

    @property
    def _device(self):
        return self.vpm.device

    @torch.inference_mode()
    def mm_embedding(self, url: str, mm_type: MMUrlType, **kwargs):
        dtype = self._data_type
        # Use global vit_emb_cache_ instead of parameter
        cached_res = vit_emb_cache_.check_cache(url)
        if cached_res is None:
            cached_url_res = get_bytes_io_from_url(url)
            cached_url_res = self._mm_preprocess(cached_url_res, mm_type)
            with mm_lock:
                features = self.mm_process(cached_url_res, mm_type=mm_type, **kwargs)
            if isinstance(features, list):
                features = torch.stack(features).to(dtype).contiguous()
            vit_emb_cache_.insert_cache(url, features)
            return (features, None)
        else:
            return (cached_res, None)

    @timeout_decorator(30)
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

    @torch.no_grad()
    def image_embedding(self, images: List[Any]) -> List[torch.Tensor]:
        data = self.processor.image_processor(images, return_tensors="pt")
        dtype = self._data_type
        tgt_sizes = data["tgt_sizes"]
        pixel_values_list = data["pixel_values"]
        vision_hidden_states = []
        all_pixel_values = []
        img_cnt = []
        for pixel_values in pixel_values_list:
            img_cnt.append(len(pixel_values))
            all_pixel_values.extend(
                [
                    i.flatten(end_dim=1).permute(1, 0).to(self._device)
                    for i in pixel_values
                ]
            )

        assert all_pixel_values
        # exist image
        if all_pixel_values:
            tgt_sizes = [
                tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)
            ]
            tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

            max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

            all_pixel_values = torch.nn.utils.rnn.pad_sequence(
                all_pixel_values, batch_first=True, padding_value=0.0
            )
            B, L, _ = all_pixel_values.shape
            all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

            patch_attn_mask = torch.zeros(
                (B, 1, max_patches), dtype=torch.bool, device=self._device
            )
            for i in range(B):
                patch_attn_mask[i, 0, : tgt_sizes[i][0] * tgt_sizes[i][1]] = True

            vision_batch_size = 16
            all_pixel_values = all_pixel_values.type(dtype)
            if B > vision_batch_size:
                hs = []
                for i in range(0, B, vision_batch_size):
                    start_idx = i
                    end_idx = i + vision_batch_size
                    tmp_hs = self.vpm(
                        all_pixel_values[start_idx:end_idx],
                        patch_attention_mask=patch_attn_mask[start_idx:end_idx],
                        tgt_sizes=tgt_sizes[start_idx:end_idx],
                    ).last_hidden_state
                    hs.append(tmp_hs)
                vision_embedding = torch.cat(hs, dim=0)
            else:
                vision_embedding = self.vpm(
                    all_pixel_values,
                    patch_attention_mask=patch_attn_mask,
                    tgt_sizes=tgt_sizes,
                ).last_hidden_state
            vision_embedding = self.resampler(vision_embedding, tgt_sizes)

            start = 0
            for pixel_values in pixel_values_list:
                img_cnt = len(pixel_values)
                if img_cnt > 0:
                    for i in range(img_cnt):
                        vision_hidden_states.append(vision_embedding[start + i])
                    start += img_cnt
                else:
                    vision_hidden_states.append([])
        # print('embedding:', vision_hidden_states)
        # print('embedding shape:', [v.shape for v in vision_hidden_states])
        return vision_hidden_states


class MiniCPMVVitWeight(BaseVitWeights):

    def _set_weight_prefix(self):
        self._ckpt_prefix = ""
        self._ft_prefix = "self.mm_part."


class MiniCPMVWeightInfo(QWenV2Weight, BaseMultiModalWeightInfo):

    def __init__(self, vit_weights, **kwargs):
        QWenV2Weight.__init__(self, prefix="llm.", **kwargs)
        BaseMultiModalWeightInfo.__init__(self, vit_weights=vit_weights, **kwargs)

    def _get_weight_info(self):
        weights = super()._get_weight_info()
        weights = self._get_vit_info(weights)
        return weights


class MiniCPMV(QWenV2, MultiModalMixin):

    def __init__(
        self,
        model_config,
        parallelism_config: ParallelismConfig,
        model_specific_config: ModelSpecificConfig,
        hw_kernel_config: HWKernelConfig,
        kv_cache_config: KVCacheConfig,
        fmha_config: FMHAConfig,
        moe_config: MoeConfig,
        runtime_config: RuntimeConfig,
        device_resource_config: DeviceResourceConfig,
        mm_model_config=None,
        vit_config=None,
        merge_lora=False,
    ):
        QWenV2.__init__(
            self,
            model_config=model_config,
            parallelism_config=parallelism_config,
            model_specific_config=model_specific_config,
            hw_kernel_config=hw_kernel_config,
            kv_cache_config=kv_cache_config,
            fmha_config=fmha_config,
            moe_config=moe_config,
            runtime_config=runtime_config,
            device_resource_config=device_resource_config,
            mm_model_config=mm_model_config,
            vit_config=vit_config,
            merge_lora=merge_lora,
        )
        self.model_config.mm_model_config.mm_sep_tokens = [
            [self.tokenizer.im_start_id, self.tokenizer.im_end_id],
            [self.tokenizer.slice_start_id, self.tokenizer.slice_end_id],
        ]

    def _init_multimodal(self, mm_model_config, vit_config):
        # mm_related_params is in model_config, not mm_model_config
        self.mm_part = ImageEmbeddingInterface(self.model_config.mm_related_params)
        self.model_config.mm_related_params.vit_weights = MiniCPMVVitWeight(
            {"vpm": self.mm_part.vpm, "resampler": self.mm_part.resampler}
        )

    @staticmethod
    def get_weight_cls():
        return MiniCPMVWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str):
        from rtp_llm.config.model_config import VitParameters

        config = ModelConfig()
        config.attn_config.head_num = 0
        config.attn_config.kv_head_num = 0
        config.attn_config.size_per_head = 0
        config.num_layers = 0
        config.inter_size = 0
        config.vocab_size = 0
        config.max_seq_len = 8192
        config.ckpt_path = ckpt_path
        config.attn_config.rope_config.dim = 128
        config.attn_config.rope_config.style = 1
        config.activation_type = "SiGLU"
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.norm_type = "rmsnorm"
        config_path = os.path.join(ckpt_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as reader:
                content = reader.read()
                config_json = json.loads(content)
                QWenV2._from_config_json(config, config_json)
                MiniCPMV._init_vit_params(config, config_json)
        else:
            raise Exception("no config.json found")
        return config

    @staticmethod
    def _init_vit_params(config: ModelConfig, config_json: Dict[str, Any]):
        if config.mm_related_params.config is None:
            config.mm_related_params.config = {}
        config.mm_related_params.config = config_json["vision_config"]
        config.mm_related_params.config["llm_hidden_size"] = config_json["hidden_size"]
        config.mm_related_params.config["query_num"] = config_json["query_num"]
        config.mm_related_params.config["ckpt_path"] = config.ckpt_path


register_model("minicpmv", MiniCPMV, ["MiniCPMV"])

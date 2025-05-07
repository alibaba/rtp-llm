import json
import os
from typing import Any, Dict, List

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from maga_transformer.config.gpt_init_model_parameters import \
    GptInitModelParameters
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.model_factory_register import register_model
from maga_transformer.models.qwen_v2 import QWenV2, QWenV2Weight
from maga_transformer.models.multimodal.multimodal_mixin import MultiModalMixin, BaseVitWeights
from maga_transformer.models.multimodal.multimodal_common import MultiModalEmbeddingInterface, mm_lock 
from maga_transformer.utils.multimodal_util import MMUrlType
from maga_transformer.models.minicpmv.modeling_navit_siglip import SiglipVisionTransformer, SiglipVisionConfig
from maga_transformer.models.minicpmv.resampler import Resampler
from maga_transformer.models.multimodal.multimodal_mixin import BaseVitWeights, BaseMultiModalWeightInfo
from maga_transformer.utils.multimodal_util import MMUrlType, vit_emb_cache_, get_bytes_io_from_url

try:
    from decord import VideoReader, cpu
except ModuleNotFoundError:
    VideoReader = None
    cpu = None

def encode_video(video_path, max_num_frames: int = 32):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    return frames

class ImageEmbeddingInterface(MultiModalEmbeddingInterface):

    def __init__(self, config: GptInitModelParameters):
        self.config = config
        config = config.mm_related_params.config
        self.vision_config = SiglipVisionConfig(**config)
        self.processor = AutoProcessor.from_pretrained(config['ckpt_path'],
                                                       trust_remote_code=True)
        self.vpm = SiglipVisionTransformer(self.vision_config)
        self.embed_dim = config['llm_hidden_size']
        self.query_num = config['query_num']
        self.vision_dim = self.vision_config.hidden_size
        self.resampler = Resampler(num_queries=self.query_num,
                                   embed_dim=self.embed_dim,
                                   num_heads=self.embed_dim // 128,
                                   kv_dim=self.vision_dim,
                                   adaptive=True)

    @property
    def _device(self):
        return self.vpm.device

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

    @torch.no_grad()
    def image_embedding(self, images: List[Any]) -> List[torch.Tensor]:
        data = self.processor.image_processor(images, return_tensors="pt")
        dtype = self._data_type
        tgt_sizes = data['tgt_sizes']
        pixel_values_list = data['pixel_values']
        vision_hidden_states = []
        all_pixel_values = []
        img_cnt = []
        for pixel_values in pixel_values_list:
            img_cnt.append(len(pixel_values))
            all_pixel_values.extend([
                i.flatten(end_dim=1).permute(1, 0).to(self._device)
                for i in pixel_values
            ])

        assert all_pixel_values
        # exist image
        if all_pixel_values:
            tgt_sizes = [
                tgt_size for tgt_size in tgt_sizes
                if isinstance(tgt_size, torch.Tensor)
            ]
            tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

            max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

            all_pixel_values = torch.nn.utils.rnn.pad_sequence(
                all_pixel_values, batch_first=True, padding_value=0.0)
            B, L, _ = all_pixel_values.shape
            all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(
                B, 3, -1, L)

            patch_attn_mask = torch.zeros((B, 1, max_patches),
                                          dtype=torch.bool,
                                          device=self._device)
            for i in range(B):
                patch_attn_mask[i,
                                0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True

            vision_batch_size = 16
            all_pixel_values = all_pixel_values.type(dtype)
            if B > vision_batch_size:
                hs = []
                for i in range(0, B, vision_batch_size):
                    start_idx = i
                    end_idx = i + vision_batch_size
                    tmp_hs = self.vpm(all_pixel_values[start_idx:end_idx],
                                      patch_attention_mask=patch_attn_mask[
                                          start_idx:end_idx],
                                      tgt_sizes=tgt_sizes[start_idx:end_idx]
                                      ).last_hidden_state
                    hs.append(tmp_hs)
                vision_embedding = torch.cat(hs, dim=0)
            else:
                vision_embedding = self.vpm(
                    all_pixel_values,
                    patch_attention_mask=patch_attn_mask,
                    tgt_sizes=tgt_sizes).last_hidden_state
            vision_embedding = self.resampler(vision_embedding, tgt_sizes)

            start = 0
            for pixel_values in pixel_values_list:
                img_cnt = len(pixel_values)
                if img_cnt > 0:
                    for i in range(img_cnt):
                        vision_hidden_states.append(vision_embedding[start +
                                                                     i])
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

    def __init__(self, config, tp_size, tp_rank):
        QWenV2Weight.__init__(self, config, tp_size, tp_rank, prefix="llm.")
        BaseMultiModalWeightInfo.__init__(self, config)

    def _get_weight_info(self):
        weights = super()._get_weight_info()
        weights = self._get_vit_info(weights)
        return weights


class MiniCPMV(QWenV2, MultiModalMixin):

    def __init__(self, config: GptInitModelParameters):
        QWenV2.__init__(self, config)
        self.config.mm_sep_tokens = [
            [self.tokenizer.im_start_id, self.tokenizer.im_end_id],
            [self.tokenizer.slice_start_id, self.tokenizer.slice_end_id]
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
        config = GptInitModelParameters(head_num=0,
                                        head_num_kv=0,
                                        size_per_head=0,
                                        layer_num=0,
                                        inter_size=0,
                                        vocab_size=0,
                                        max_seq_len=8192,
                                        ckpt_path=ckpt_path,
                                        rotary_embedding_dim=128,
                                        rotary_embedding_style=1,
                                        activation_type='SiGLU',
                                        has_pre_decoder_layernorm=False,
                                        has_post_decoder_layernorm=True,
                                        norm_type='rmsnorm')
        config_path = os.path.join(ckpt_path, 'config.json')
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
    def _init_vit_params(config: GptInitModelParameters,
                         config_json: Dict[str, Any]):
        config.mm_related_params.config = config_json["vision_config"]
        config.mm_related_params.config["llm_hidden_size"] = config_json[
            "hidden_size"]
        config.mm_related_params.config["query_num"] = config_json["query_num"]
        config.mm_related_params.config["ckpt_path"] = config.ckpt_path


register_model('minicpmv', MiniCPMV, ["MiniCPMV"])

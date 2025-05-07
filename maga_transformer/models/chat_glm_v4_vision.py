import os

import torch

from maga_transformer.config.gpt_init_model_parameters import \
    GptInitModelParameters
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.model_factory_register import register_model
from maga_transformer.models.chat_glm_v4 import ChatGlmV4
from maga_transformer.models.chat_glm_v4_vision_weight import (
    ChatGlmV4VisionVitWeights, ChatGlmV4VisionWeightInfo)
from maga_transformer.models.eva2clip_vit import EVA2CLIPImageEmbedding
from maga_transformer.models.multimodal.multimodal_mixin import MultiModalMixin
from maga_transformer.utils.util import get_config_from_path, to_torch_dtype


class ChatGlmV4VisionImageEmbedding(EVA2CLIPImageEmbedding):
    @torch.inference_mode()
    def mm_process(self, mm_input, **kwargs):
        embeddings = self.image_embedding([mm_input])[0]
        pos_ids = [1] * embeddings.shape[0]
        pos_ids[0] = 0
        pos_ids[-1] = 2
        return embeddings, torch.tensor(pos_ids, dtype=torch.int32)

class ChatGlmV4Vision(ChatGlmV4, MultiModalMixin):
    def _init_multimodal(self, config: GptInitModelParameters):
        self.mm_part = ChatGlmV4VisionImageEmbedding(config)
        config.mm_related_params.vit_weights = ChatGlmV4VisionVitWeights(
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

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = ChatGlmV4._create_config(ckpt_path)
        config_dict = get_config_from_path(ckpt_path)
        vit_config = config_dict["vision_config"]
        config.mm_related_params.config.update(vit_config)
        config.build_position_ids = True
        # use initial hidden size for linear_proj and conv layer in eva2clip
        config.mm_related_params.config['use_vision_hidden_size'] = False
        config.mm_related_params.config["boi_token_id"] = config_dict.get("boi_token_id", 0)
        config.mm_related_params.config["eoi_token_id"] = config_dict.get("eoi_token_id", 0)
        config.mm_sep_tokens = [[config_dict.get("boi_token_id", 0), config_dict.get("eoi_token_id", 0)]]
        config.include_sep_tokens = True
        config.mm_position_ids_style = 1
        return config

    @staticmethod
    def get_weight_cls():
        return ChatGlmV4VisionWeightInfo

register_model("chatglm4v", ChatGlmV4Vision, [], ["THUDM/glm-4v-9b"])

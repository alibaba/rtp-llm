from functools import partial
import os
from typing import List, Tuple, Union

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
from maga_transformer.ops.comm.nccl_op import NcclOp
from maga_transformer.utils.util import get_config_from_path, to_torch_dtype


class ChatGlmV4Vision(ChatGlmV4, MultiModalMixin):
    def __init__(self, config: GptInitModelParameters):
        self.mm_part = EVA2CLIPImageEmbedding(config)
        self.nccl_op_ = NcclOp()
        config.vit_related_params.vit_weights = ChatGlmV4VisionVitWeights(
            {"vit": self.mm_part.vit}
        )
        ChatGlmV4.__init__(self, config)

    @classmethod
    def is_multimodal(cls) -> bool:
        return True

    def load(self, device: Union[str, torch.device] = 'cuda:0'):
        if os.environ.get("VIT_TRT", "0") == "1":
            weights_info = self.get_weight_cls()(self.config, g_parallel_info.tp_size, g_parallel_info.tp_rank)
            self.init_mm_trt(
                "chatglm4v", g_parallel_info, weights_info, self.config.ckpt_path,
                self.config.vit_related_params, device, to_torch_dtype(self.config.data_type)
            )
        super().load(device=device)

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = ChatGlmV4._create_config(ckpt_path)
        config_dict = get_config_from_path(ckpt_path)
        vit_config = config_dict["vision_config"]
        config.vit_related_params.config.update(vit_config)
        config.build_position_ids = True
        # use initial hidden size for linear_proj and conv layer in eva2clip
        config.vit_related_params.config['use_vision_hidden_size'] = False
        config.vit_related_params.config["boi_token_id"] = config_dict.get("boi_token_id", 0)
        config.vit_related_params.config["eoi_token_id"] = config_dict.get("eoi_token_id", 0)
        config.tp_split_emb_and_lm_head = False  # chatglmv4 embedding can't tp
        return config

    @staticmethod
    def get_weight_cls():
        return ChatGlmV4VisionWeightInfo

    def async_input_word_embedding(
        self,
        inputs: torch.Tensor,
        images: List[torch.Tensor],
        token_type_ids: torch.Tensor,
    ):
        return MultiModalMixin.async_input_word_embedding(
            self, inputs, images, token_type_ids
        )

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
                position_ids.append(seq_lengths_list[start_idx] - vision_token_length[start_idx] + 2)
        return position_ids

    def extend_context_position_ids(
        self, context_begin_position: int, context_end_position: int,
        token_type_ids: torch.Tensor, token_ids: torch.Tensor
    ) -> List[int]:

        img_start_token_id: int = self.config.vit_related_params.config["boi_token_id"]
        img_end_token_id: int = self.config.vit_related_params.config["eoi_token_id"]

        bos_pos = torch.where(token_ids == img_start_token_id)[0]
        eos_pos = torch.where(token_ids == img_end_token_id)[0]
        assert bos_pos.shape[0] <= 1
        if bos_pos.shape[0] == 0:
            return list(range(0, token_ids.shape[0]))

        vision_token_num = eos_pos[0] - bos_pos[0] + 1
        img_begin_position = bos_pos[0].item() + 1

        # construct position ids for rotary embedding, assuming the token ids' type is [T, V, V, V, V, V, T, T, T]
        # the expected position ids is [0, 1, 2, 2, 2, 3, 4, 5, 6]. Here [0, 1] is the list(range(0, img_begin_position)) part,
        # [2, 2, 2] is the [img_begin_position] * (vision_token_num - 2) part, and [3, 4, 5, 6] is the last part
        # see https://huggingface.co/THUDM/glm-4v-9b/blob/main/modeling_chatglm.py#862
        position_ids = list(range(0, img_begin_position)) + \
            [img_begin_position] * (vision_token_num - 2) + \
            list(range(img_begin_position + 1, img_begin_position + 1 + token_ids.shape[0] - eos_pos[0]))

        return position_ids

    def expand_token_id(
        self, token_ids: List[int], images: List[torch.tensor]
    ) -> Tuple[List[int], List[torch.Tensor], List[int]]:
        if len(images) > 1:
            raise Exception("ChatGLM4V support processes one image at a time")

        img_start_token_id: int = self.config.vit_related_params.config["boi_token_id"]
        img_end_token_id: int = self.config.vit_related_params.config["eoi_token_id"]

        img_start_positions = [i for i, x in enumerate(token_ids) if x == img_start_token_id]
        img_end_positions = [i for i, x in enumerate(token_ids) if x == img_end_token_id]
        assert len(img_start_positions) == len(img_end_positions) and len(img_start_positions) <= 1

        # only text
        if len(img_start_positions) == 0:
            return token_ids, images, []

        # add placehold tokens for image
        patch_size: int = self.config.vit_related_params.config["patch_size"]
        image_size: int = self.config.vit_related_params.config["image_size"]

        vision_token_num = (image_size // patch_size // 2) ** 2 + 2
        img_pad_token_id = self.config.special_tokens.pad_token_id

        new_token_ids = token_ids[:img_start_positions[0] + 1] + [img_pad_token_id] * (vision_token_num - 2) + \
            token_ids[img_end_positions[0]:]

        return new_token_ids, images, []

    def multimodal_embedding(
        self,
        input_ids: torch.Tensor,
        images_embedding: List[torch.Tensor],
        token_type_ids: torch.Tensor,
    ):

        img_start_token_id: int = self.config.vit_related_params.config["boi_token_id"]
        img_end_token_id: int = self.config.vit_related_params.config["eoi_token_id"]

        bos_pos = torch.where(input_ids == img_start_token_id)
        eos_pos = torch.where(input_ids == img_end_token_id)
        assert (bos_pos[0] == eos_pos[0]).all()
        img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)

        input_embeds = self.word_embedding(input_ids)

        # replace virtual placeholder embedding with real image embedding
        if images_embedding != []:
            for idx, (i, a, b) in enumerate(img_pos):
                input_embeds[i][a:b + 1] = images_embedding[idx]

        return input_embeds

register_model("chatglm4v", ChatGlmV4Vision, [], ["THUDM/glm-4v-9b"])

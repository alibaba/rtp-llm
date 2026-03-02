from rtp_llm.model_factory_register import register_model
from rtp_llm.models.chat_glm_v4 import ChatGlmV4
from rtp_llm.models.glm_v2_weight import GlmV2WeightInfo
from rtp_llm.utils.util import get_config_from_path


class ChatGlmV4Vision(ChatGlmV4):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = ChatGlmV4._create_config(ckpt_path)
        config_dict = get_config_from_path(ckpt_path)
        vit_config = config_dict["vision_config"]
        config.mm_related_params.config.update(vit_config)
        # use initial hidden size for linear_proj and conv layer in eva2clip
        config.mm_related_params.config["use_vision_hidden_size"] = False
        config.mm_related_params.config["llm_hidden_size"] = config.hidden_size
        config.mm_related_params.config["llm_inter_size"] = config.inter_size
        config.mm_related_params.config["boi_token_id"] = config_dict.get(
            "boi_token_id", 0
        )
        config.mm_related_params.config["eoi_token_id"] = config_dict.get(
            "eoi_token_id", 0
        )
        config.mm_model_config.mm_sep_tokens = [
            [config_dict.get("boi_token_id", 0), config_dict.get("eoi_token_id", 0)]
        ]
        config.mm_model_config.include_sep_tokens = True
        config.mm_model_config.mm_position_ids_style = 1
        config.mm_model_config.is_multimodal = True
        return config

    @staticmethod
    def get_weight_cls():
        return GlmV2WeightInfo


register_model("chatglm4v", ChatGlmV4Vision, [], ["THUDM/glm-4v-9b"])

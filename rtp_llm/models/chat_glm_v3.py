from rtp_llm.model_factory_register import register_model
from rtp_llm.models.chat_glm_v2 import ChatGlmV2
from rtp_llm.utils.util import get_config_from_path


class ChatGlmV3(ChatGlmV2):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config_dict = get_config_from_path(ckpt_path)
        if config_dict is not None:
            config = cls.from_huggingface(config_dict)
        else:
            config = ChatGlmV2.default_config()
        config = ChatGlmV2.modify_config(config)

        return config

    @staticmethod
    def get_rotary_embedding_scale(config, config_json):
        config.attn_config.rope_config.base = config_json.get("rope_theta", 10000) * int(
            config_json.get("rope_ratio", 1)
        )
        return config


register_model(
    "chatglm3",
    ChatGlmV3,
    [],
    ["THUDM/chatglm3-6b", "THUDM/chatglm3-6b-base", "THUDM/chatglm3-6b-32k"],
)
register_model("chat_glm_3", ChatGlmV3)

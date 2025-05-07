
from maga_transformer.utils.util import get_config_from_path
from maga_transformer.models.chat_glm_v2 import ChatGlmV2
from maga_transformer.tokenizer.tokenization_chatglm3 import ChatGLMTokenizer
from maga_transformer.model_factory_register import register_model
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters

class ChatGlmV3(ChatGlmV2):
    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        return ChatGLMTokenizer.from_pretrained(config.tokenizer_path, encode_special_tokens=True)

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
        config.rotary_embedding_base = config_json.get('rope_theta', 10000) * int(config_json.get("rope_ratio", 1))
        return config

register_model('chatglm3', ChatGlmV3, [], ["THUDM/chatglm3-6b", "THUDM/chatglm3-6b-base", "THUDM/chatglm3-6b-32k"])
register_model('chat_glm_3', ChatGlmV3)

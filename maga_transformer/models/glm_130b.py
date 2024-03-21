
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.models.chat_glm import ChatGlm
from maga_transformer.model_factory_register import register_model

class Glm130B(ChatGlm):
    def __init__(self, config: GptInitModelParameters):
        super().__init__(config)

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = GptInitModelParameters(
            head_num=96,
            size_per_head=128,
            inter_size=32768,
            activation_type='geglu',
            norm_type='layernorm',
            rotary_embedding_dim=128,
            rotary_embedding_style=1,
            add_bias_linear=True,
            layer_num=70,
            max_seq_len=2048,
            vocab_size=150528)
        config.special_tokens.eos_token_id = 20002
        return config

register_model('glm_130b', Glm130B)

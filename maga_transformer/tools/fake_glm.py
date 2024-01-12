from maga_transformer.tools.fake_model_base import *

def fake_glm():
    default_config = DefaultModelConfig()
    default_config.layer_num = 2
    default_config.head_num = 2
    default_config.head_kv_num = 2
    default_config.head_size = 64
    default_config.ffn_hidden_size = 4 * default_config.head_size * default_config.head_num
    default_config.ffn_inter_padding_size = default_config.ffn_hidden_size
    default_config.ffn_gate_active = False
    default_config.ffn_w1_w3_independ = False
    default_config.vocab_size = 130528

    fake_model("chatglm", default_config, default_save_config_func)

if __name__ == '__main__':
    fake_glm()

import os
import json
from maga_transformer.tools.fake_model_base import *

def save_config_func(dest_path: str,
                layer: int,
                head: int,
                head_kv: int,
                head_size: int,
                ffn_hidden_size: int,
                vocab_size: int):
    config = {
        "model_type": "chatglm",
        "architectures": [
            "ChatGLMModel"
        ],
        "auto_map": {
            "AutoConfig": "configuration_chatglm.ChatGLMConfig",
            "AutoModel": "modeling_chatglm.ChatGLMForConditionalGeneration",
            "AutoModelForCausalLM": "modeling_chatglm.ChatGLMForConditionalGeneration",
            "AutoModelForSeq2SeqLM": "modeling_chatglm.ChatGLMForConditionalGeneration"
        },
        "add_bias_linear": False,
        "add_qkv_bias": True,
        "apply_query_key_layer_scaling": True,
        "apply_residual_connection_post_layernorm": False,
        "attention_dropout": 0.0,
        "attention_softmax_in_fp32": True,
        "bias_dropout_fusion": True,
        "ffn_hidden_size": ffn_hidden_size,
        "fp32_residual_connection": False,
        "hidden_dropout": 0.0,
        "hidden_size": head * head_size,
        "kv_channels": head_size,
        "layernorm_epsilon": 1e-05,
        "multi_query_attention": True,
        "multi_query_group_num": head_kv,
        "num_attention_heads": head,
        "num_layers": layer,
        "original_rope": True,
        "padded_vocab_size": vocab_size,
        "post_layer_norm": True,
        "rmsnorm": True,
        "seq_length": 32768,
        "use_cache": True,
        "torch_dtype": "float16",
        "tie_word_embeddings": False,
        "eos_token_id": 2,
        "pad_token_id": 0
    }
    # save to config.json
    json.dump(config, open(os.path.join(dest_path, 'config.json'), 'w'), indent=2)

def fake_glm_v2():
    default_config = DefaultModelConfig()
    default_config.layer_num = 2
    default_config.head_num = 4
    default_config.head_kv_num = 2
    default_config.head_size = 64
    default_config.ffn_hidden_size = 1712
    default_config.vocab_size = 65024
    default_config.ffn_gate_active = True
    default_config.ffn_w1_w3_independ = False

    fake_model("chatglm2", default_config, save_config_func)

if __name__ == '__main__':
    fake_glm_v2()
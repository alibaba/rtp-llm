import os
import json
from maga_transformer.tools.fake_model_base import *

def save_config_func(model_type,
                     dest_path: str,
                     layer: int,
                     head: int,
                     head_kv: int,
                     head_size: int,
                     ffn_hidden_size: int,
                     ffn_inter_padding_size: int,
                     vocab_size: int):
    config = {
        "activation": "swiglu",
        "apply_residual_connection_post_layernorm": False,
        "architectures": [
            "QWenLMHeadModel"
        ],
        "attn_pdrop": 0.0,
        "auto_map": {
            "AutoConfig": "configuration_qwen.QWenConfig",
            "AutoModelForCausalLM": "modeling_qwen.QWenLMHeadModel"
        },
        "bf16": True,
        "bias_dropout_fusion": True,
        "bos_token_id": 151643,
        "embd_pdrop": 0.0,
        "eos_token_id": 151643,
        "ffn_hidden_size": ffn_hidden_size,
        "ffn_inter_padding_size": ffn_hidden_size,
        "fp16": False,
        "fp32": False,
        "initializer_range": 0.02,
        "kv_channels": head_size,
        "layer_norm_epsilon": 1e-06,
        "n_embd": head * head_size,
        "n_head": head,
        "n_layer": layer,
        "n_positions": 6144,
        "no_bias": True,
        "padded_vocab_size": 151936,
        "params_dtype": "torch.bfloat16",
        "pos_emb": "rotary",
        "resid_pdrop": 0.1,
        "rotary_emb_base": 10000,
        "rotary_pct": 1.0,
        "scale_attn_weights": True,
        "seq_length": 2048,
        "tie_word_embeddings": False,
        "tokenizer_type": "QWenTokenizer",
        "torch_dtype": "bfloat16",
        "transformers_version": "4.39.3",
        "use_cache": True,
        "use_dynamic_ntk": True,
        "use_flash_attn": True,
        "use_logn_attn": True,
        "vocab_size": vocab_size
    }
    # save to config.json
    json.dump(config, open(os.path.join(dest_path, 'config.json'), 'w'), indent=2)

def fake_qwen():
    default_config = DefaultModelConfig()
    default_config.layer_num = 2
    default_config.head_num = 2
    default_config.head_kv_num = 2
    default_config.head_size = 128
    default_config.ffn_hidden_size = 4 * default_config.head_size * default_config.head_num
    default_config.ffn_inter_padding_size = 4 * default_config.head_size * default_config.head_num
    default_config.ffn_gate_active = True
    default_config.ffn_w1_w3_independ = True
    default_config.vocab_size = 151936

    fake_model("qwen_7b", default_config, save_config_func)

if __name__ == '__main__':
    fake_qwen()

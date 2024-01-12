import os
import json
from maga_transformer.tools.fake_model_base import *


def save_config_func(
                model_type: str,
                dest_path: str,
                layer_num: int,
                head: int,
                head_kv_num: int,
                head_size: int,
                ffn_hidden_size: int,
                ffn_inter_padding_size: int,
                vocab_size: int):
    config = {
            "architectures": [
                "GPTNeoXForCausalLM"
            ],
            "auto_map": {
                "AutoModel": "model.GPTNeoXForCausalLM"
            },
            "bos_token_id": 2,
            "eos_token_id": 2,
            "hidden_act": "gelu",
            "hidden_size": head * head_size,
            "initializer_range": 0.02,
            "intermediate_size": ffn_hidden_size,
            "layer_norm_eps": 1e-05,
            "max_position_embeddings": 2048,
            "model_type": model_type,
            "num_attention_heads": head,
            "num_hidden_layers": layer_num,
            "rotary_emb_base": 10000,
            "rotary_pct": 1.0,
            "tie_word_embeddings": False,
            "torch_dtype": "float16",
            "transformers_version": "4.27.1",
            "use_cache": True,
            "use_parallel_residual": False,
            "vocab_size": vocab_size,
            "attention_dropout": 0,
            "hidden_dropout": 0,
            "classifier_dropout": 0.1,
            "rope_scaling": {
                "type": "dynamic",
                "factor": 2
            }
    }
    # save to config.json
    json.dump(config, open(os.path.join(dest_path, 'config.json'), 'w'), indent=2)


def fake_gpt_neox():
    default_config = DefaultModelConfig()
    default_config.layer_num = 2
    default_config.head_num = 4
    default_config.head_kv_num = 4
    default_config.head_size = 128
    default_config.ffn_hidden_size = 4 * default_config.head_size * default_config.head_num
    default_config.ffn_inter_padding_size = default_config.ffn_hidden_size
    default_config.ffn_gate_active = False
    default_config.ffn_w1_w3_independ = False
    default_config.vocab_size = 250752

    fake_model("gpt_neox", default_config, save_config_func)

if __name__ == '__main__':
    fake_gpt_neox()

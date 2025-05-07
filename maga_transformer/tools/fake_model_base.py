import torch
from typing import Dict, List
import argparse
import os
import json
from maga_transformer.tools.fake_util import generate_fake_model, copy_from_model
from maga_transformer.utils.util import load_ckpt

from maga_transformer.utils.model_weight import W
from maga_transformer.model_factory import ModelFactory

def default_save_config_func(model_type,
                dest_path: str,
                layer: int,
                head: int,
                head_kv: int,
                head_size: int,
                ffn_hidden_size: int,
                ffn_inter_padding_size: int,
                vocab_size: int):
    config = {
        "model_type": model_type,
        "add_bias_linear": False,
        "add_qkv_bias": True,
        "apply_query_key_layer_scaling": True,
        "apply_residual_connection_post_layernorm": False,
        "attention_dropout": 0.0,
        "attention_softmax_in_fp32": True,
        "bias_dropout_fusion": True,
        "ffn_hidden_size": ffn_hidden_size,
        "ffn_inter_padding_size": ffn_inter_padding_size,
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

def fake_model_impl(model_type: str,
                save_config_func,
                post_rewrite_func,
                dest_path: str,
                layer_num: int,
                head_num: int,
                head_kv_num: int,
                head_size: int,
                ffn_hidden_size: int,
                ffn_inter_padding_size: int,
                ffn_gate_active: bool,
                ffn_w1_w3_independ: bool,
                vocab_size: int,
                input_model: str | None):
    hidden_size = head_num * head_size
    qkv_hidden_size = hidden_size + head_kv_num * head_size * 2
    new_params: Dict[str, torch.Tensor] = {}

    model_weight_info = ModelFactory.get_weight_cls(model_type)(
                        hidden_size=hidden_size, inter_size=ffn_hidden_size * 2,
                        num_heads=head_num, num_heads_kv=head_kv_num, tp_size=1,
                        int8_mode=0, num_layers=layer_num)
    
    model_weight_info._lm_head = False
    model_weight_info._transformer_prefix = False
    weight_info = model_weight_info.get_weight_info()
    print(weight_info)

    weight_dict = { W.embedding : "", W.lm_head : "", W.pre_decoder_ln_gamma : "",
                    W.pre_decoder_ln_beta : "", W.final_ln_gamma : "", W.final_ln_beta : ""}

    for weight_item in weight_info.weights:
        if weight_item.name in weight_dict and len(weight_item.weights) > 0:
            weight_dict[weight_item.name] = weight_item.weights[0].name

    shape_map: Dict[str, List[int]] = {
        weight_dict[W.embedding]: [vocab_size, hidden_size],
        weight_dict[W.lm_head]: [vocab_size, hidden_size],
        weight_dict[W.pre_decoder_ln_gamma]: [hidden_size],
        weight_dict[W.pre_decoder_ln_beta]: [hidden_size],
        weight_dict[W.final_ln_gamma]: [hidden_size],
        weight_dict[W.final_ln_beta]: [hidden_size],
    }

    layer_weight_dict = {W.pre_ln_gamma: "", W.pre_ln_beta: "", W.attn_qkv_w: "", W.attn_qkv_b: "",
                        W.attn_ln_gamma: "", W.attn_ln_beta: "", W.attn_o_w: "",
                        W.attn_o_b: "", W.ffn_w1: "", W.ffn_b1: "", W.ffn_ln_gamma: "", W.ffn_ln_beta: "", W.ffn_w2: "", W.ffn_b2: "",
                        W.ffn_w3: "", W.ffn_b3: "", W.post_ln_gamma: "", W.post_ln_beta: ""}

    for layer_weight_item in weight_info.layer_weights:
        if layer_weight_item.name in layer_weight_dict and len(layer_weight_item.weights) > 0:
            layer_weight_dict[layer_weight_item.name] = layer_weight_item.weights[0].name

    for i in range(layer_num):
        shape_map[layer_weight_dict[W.pre_ln_gamma].format(i=str(i))] = [hidden_size]
        shape_map[layer_weight_dict[W.pre_ln_beta].format(i=str(i))] = [hidden_size]
        shape_map[layer_weight_dict[W.attn_qkv_w].format(i=str(i))] = [qkv_hidden_size, hidden_size]
        shape_map[layer_weight_dict[W.attn_qkv_b].format(i=str(i))] = [qkv_hidden_size]
        shape_map[layer_weight_dict[W.attn_ln_gamma].format(i=str(i))] = [hidden_size]
        shape_map[layer_weight_dict[W.attn_ln_beta].format(i=str(i))] = [hidden_size]
        shape_map[layer_weight_dict[W.attn_o_w].format(i=str(i))] = [hidden_size , hidden_size]
        shape_map[layer_weight_dict[W.attn_o_b].format(i=str(i))] = [hidden_size]

        if ffn_gate_active:
            if ffn_w1_w3_independ:
                shape_map[layer_weight_dict[W.ffn_w1].format(i=str(i))] = [ffn_hidden_size, hidden_size]
                shape_map[layer_weight_dict[W.ffn_b1].format(i=str(i))] = [ffn_hidden_size]
                shape_map[layer_weight_dict[W.ffn_w3].format(i=str(i))] = [ffn_hidden_size, hidden_size]
                shape_map[layer_weight_dict[W.ffn_b3].format(i=str(i))] = [ffn_hidden_size]
            else:
                shape_map[layer_weight_dict[W.ffn_w1].format(i=str(i))] = [ffn_hidden_size * 2, hidden_size]
                shape_map[layer_weight_dict[W.ffn_b1].format(i=str(i))] = [ffn_hidden_size * 2]
        else:
            shape_map[layer_weight_dict[W.ffn_w1].format(i=str(i))] = [ffn_hidden_size, hidden_size]
            shape_map[layer_weight_dict[W.ffn_b1].format(i=str(i))] = [ffn_hidden_size]

        shape_map[layer_weight_dict[W.ffn_ln_gamma].format(i=str(i))] = [ffn_hidden_size]
        shape_map[layer_weight_dict[W.ffn_ln_beta].format(i=str(i))] = [ffn_hidden_size]

        shape_map[layer_weight_dict[W.ffn_w2].format(i=str(i))] = [hidden_size, ffn_hidden_size]
        shape_map[layer_weight_dict[W.ffn_b2].format(i=str(i))] = [hidden_size]

        shape_map[layer_weight_dict[W.post_ln_gamma].format(i=str(i))] = [hidden_size]
        shape_map[layer_weight_dict[W.post_ln_beta].format(i=str(i))] = [hidden_size]

    if "" in shape_map:
        del shape_map[""]
    assert(len(shape_map) > 0)
    print("shape_map = ", shape_map)

    file_name = f"fake_{model_type}_{layer_num}_{head_num}_{head_kv_num}_{head_size}_{ffn_hidden_size}_{vocab_size}"
    if input_model:
        model = load_ckpt(input_model)
        model_weight_info.process_meta(model)
        new_params = copy_from_model(shape_map, model)
        file_name += "_copy"
    else:
        new_params = generate_fake_model(shape_map)

    file_name += ".pt"

    if not os.path.exists(dest_path):
        print(f'{dest_path} not exist, creating...')
        os.makedirs(dest_path)

    # save config
    print("saving config.json...")
    if save_config_func:
        save_config_func(model_type, dest_path, layer_num, head_num,
                        head_kv_num, head_size, ffn_hidden_size, ffn_inter_padding_size, vocab_size)

    # save model
    print("saving model...")
    file_name = os.path.join(dest_path, file_name)
    if post_rewrite_func:
        new_params = post_rewrite_func(new_params)

    print(new_params)

    torch.save(new_params, file_name)

    print(f"save finished, save path: {dest_path}")

class DefaultModelConfig:
    layer_num: int = 0
    head_num: int = 0
    head_kv_num: int = 0
    head_size: int = 0
    ffn_hidden_size: int = 0
    ffn_inter_padding_size: int = 0
    ffn_gate_active: bool = True
    ffn_w1_w3_independ: bool = False
    vocab_size: int = 0

def fake_model(model_type: str, default_values: DefaultModelConfig, save_config_func = None, post_rewrite_func = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', help='saved path',
                        required=True)
    parser.add_argument('--layer', '-l', help='layer number',
                        default=default_values.layer_num, type=int)
    parser.add_argument('--head', '-d', help='head number',
                        default=default_values.head_num, type=int)
    parser.add_argument('--head_kv', '-k', help='kv head number',
                        default=default_values.head_kv_num, type=int)
    parser.add_argument('--head_size', '-s', help='head size',
                        default=default_values.head_size, type=int)
    parser.add_argument('--ffn_hidden_size', '-f', help='ffn hidden size',
                        default=default_values.ffn_hidden_size, type=int)
    parser.add_argument('--ffn_inter_padding_size', '-e', help='ffn inter padding size',
                        default=default_values.ffn_inter_padding_size, type=int)
    parser.add_argument('--ffn_gate_active', '-g', help='ffn gate active',
                        default=default_values.ffn_gate_active, type=bool)
    parser.add_argument('--ffn_w1_w3_independ', '-r', help='ffn w1 w3 weight independ',
                        default=default_values.ffn_w1_w3_independ, type=bool)
    parser.add_argument('--vocab', '-v', help='vocab size',
                        default=default_values.vocab_size, type=int)
    parser.add_argument('--input', '-i', help='input model',
                        default=None, type=str)
    args = parser.parse_args()

    fake_model_impl(model_type, save_config_func, post_rewrite_func, args.path,
                    args.layer, args.head, args.head_kv,
                    args.head_size, args.ffn_hidden_size, args.ffn_inter_padding_size,
                    args.ffn_gate_active, args.ffn_w1_w3_independ,
                    args.vocab, args.input)
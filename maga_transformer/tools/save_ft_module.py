import argparse
import os
import torch
from typing import Any
import json
from maga_transformer.model_factory import _model_factory
from maga_transformer.utils.model_weights_loader import ModelWeightsLoader
from maga_transformer.distribute.worker_info import g_parallel_info

from typing import Dict, Type

class FTModelWeights:
    def __init__(self, num_layers: int) -> None:
        self.model: Dict[str, torch.Tensor] = {"ft_module": torch.ones([1]).half()}

    def append_pytorch_weight(self, name: str, tensor: torch.Tensor):
        self.model[name] = tensor

    def append_layer_weight(self, layer_id: int, name: str, tensor: torch.Tensor):
        self.model[f'layer.{layer_id}.{name}'] = tensor

def load_as_ft_module(ckpt_path: str, model_type: str, **kwargs: Any) -> Any:
    if model_type not in _model_factory:
        raise Exception(f"model {model_type} not registered!")
    model_cls = _model_factory[model_type]

    config = model_cls.create_config(ckpt_path, **kwargs)
    weight_cls = model_cls.get_weight_cls()

    weights_info = weight_cls(
        hidden_size=config.head_num * config.size_per_head,
        inter_size=config.inter_size,
        num_heads=config.head_num,
        num_heads_kv=config.head_num_kv,
        tp_size=g_parallel_info.tp_size,
        int8_mode=int8_mode,
        num_layers=config.layer_num
    )
    ckpt_loader = ModelWeightsLoader(ckpt_path=ckpt_path,
                             num_layers=config.layer_num,
                             tp_size=g_parallel_info.tp_size,
                             tp_rank=g_parallel_info.tp_rank,
                             pp_size=g_parallel_info.pp_size,
                             pp_rank=g_parallel_info.pp_rank,
                             weights_info=weights_info)
    weight = FTModelWeights(config.layer_num)

    ckpt_loader.load_weights_from_scratch(weight, config.quant_algo, 'cpu')
    return weight.model

def save_ft_module(ckpt_path: str, save_dir: str, model_type:str):
    module = load_as_ft_module(ckpt_path, model_type, tokenizer_path=ckpt_path)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output_name = f"{model_type}_optim.pt"

    torch.save(module, os.path.join(save_dir, output_name))

    # cp ckpt_path/config.json to save_dir/config.json
    if os.path.exists(os.path.join(ckpt_path, "config.json")):
        with open(os.path.join(ckpt_path, "config.json"), "r") as f_origin:
            config = json.load(f_origin)
            with open(os.path.join(save_dir, "config.json"), "w") as f_copy:
                json.dump(config, f_copy, indent=2)

    result = {
        "status": "ok",
        "model_result": save_dir,
        "model_type": model_type,
    }
    result_file = os.path.join(save_dir, "result.json")

    print(f"write final result to file {result_file}:{result} ")
    with open(result_file, "w") as f:
        f.write(json.dumps(result))

def main():
    ckpt_path = os.getenv('CHECKPOINT_PATH', None)
    model_type = os.getenv('MODEL_TYPE', None)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_path', '-i', type=str, required=False,
                        default=ckpt_path, help='input model path')
    parser.add_argument('--output_save_dir', '-o', type=str, required=True,
                        help='output save dir')
    parser.add_argument('--model_type', '-t', type=str, required=False,
                        default=model_type, help='model type')
    args = parser.parse_args()
    if args.input_model_path is None or args.model_type is None:
        raise ValueError("Please set CHECKPOINT_PATH and MODEL_TYPE env or pass --input_model_path/-i, --model_type/-t arguments.")
    save_ft_module(args.input_model_path, args.output_save_dir, args.model_type)


if __name__ == '__main__':
    main()

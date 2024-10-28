import copy
import json
import time
from typing import Dict, List
import torch
import os
import logging
from transformers import AutoModelForCausalLM, AutoConfig
import safetensors

from maga_transformer.tools.quant.base_quanter import QUANT_TYPE, BaseQuanter
'''
FP8_DEFAULT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "default": {"num_bits": (4, 3), "axis": None},
    },
    "algorithm": "max",
}
'''
'''
KV_CACHE_CFG = {
    "*.query_key_value.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.Wqkv.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.W_pack.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.c_attn.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.k_proj.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.v_proj.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
}

'''
class Fp8Quanter(BaseQuanter):
    FP8_DEFAULT_CFG = {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
            "*input_quantizer": {"num_bits": (4, 3), "axis": None},
            "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
            "default": {"num_bits": (4, 3), "axis": None},
        },
        "algorithm": "max",
    }
    KV_CACHE_CFG = {
        "*.query_key_value.output_quantizer": {
            "num_bits": 8,
            "axis": None,
            "enable": True
        },
        "*.Wqkv.output_quantizer": {
            "num_bits": 8,
            "axis": None,
            "enable": True
        },
        "*.W_pack.output_quantizer": {
            "num_bits": 8,
            "axis": None,
            "enable": True
        },
        "*.c_attn.output_quantizer": {
            "num_bits": 8,
            "axis": None,
            "enable": True
        },
        "*.k_proj.output_quantizer": {
            "num_bits": 8,
            "axis": None,
            "enable": True
        },
        "*.v_proj.output_quantizer": {
            "num_bits": 8,
            "axis": None,
            "enable": True
        },
    }
    def __init__(self, quantize_config: Dict[str, str], model_path: str, offload_folder: str):
        super().__init__()
        self.quantize_config = quantize_config
        
        max_memory = {}
        per_gpu_max_memory = int(torch.cuda.get_device_properties(torch.device('cuda:0')).total_memory*0.95/1024/1024/1024)
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        cuda_device_list = cuda_devices.split(',') if cuda_devices is not None else \
            [str(i) for i in range(torch.cuda.device_count())]
        max_memory.update({int(i): f'{per_gpu_max_memory}GIB' for i in range(len(cuda_device_list))})
        logging.info(f'max_memory: {max_memory}')

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            offload_folder=offload_folder,
            max_memory=max_memory)
        self.model = model.eval().half()

        self.quant_cfg = copy.deepcopy(Fp8Quanter.FP8_DEFAULT_CFG)
        kv_cache_dtype = quantize_config.get('kv_cache_dtype', None)
        if kv_cache_dtype is not None:
            if kv_cache_dtype == "fp8":
                for value in Fp8Quanter.KV_CACHE_CFG.values():
                    value.update({"num_bits": (4, 3)})  # type: ignore
            self.quant_cfg["quant_cfg"].update(Fp8Quanter.KV_CACHE_CFG)  # type: ignore
        


    def _quant(self, examples: List[Dict[str, torch.Tensor]]):
        examples = [_.get('input_ids').tolist() for _ in examples]
        import modelopt.torch.quantization as atq

        def calibrate_loop():
            if examples is None:
                return
            """Adjusts weights and scaling factors based on selected algorithms."""
            for idx, example in enumerate(examples):
                print(f"Calibrating batch {idx}")
                example = torch.cat(example, dim=0)
                # model might be mapped to different device because the device_map is auto
                self.model(example.to(next(self.model.parameters()).device))


        print("Starting quantization...")
        start_time = time.time()
        atq.quantize(self.model, self.quant_cfg, forward_loop=calibrate_loop)
        end_time = time.time()
        print("Quantization done. Total time used: {:.2f} s.".format(end_time -
                                                                    start_time))
    @classmethod
    def quant_type(cls):
        return QUANT_TYPE.FP8

    def _save_quantized(self, output_path: str):
        with torch.inference_mode():
            if model_type is None:
                print(
                    f"Unknown model type {type(model).__name__}. Continue exporting..."
                )
                model_type = f"unknown:{type(model).__name__}"

            export_path = output_path
            start_time = time.time()
            from modelopt.torch.export import export_tensorrt_llm_checkpoint
            export_tensorrt_llm_checkpoint(self.model,
                                        model_type,
                                        getattr(torch, dtype),
                                        export_dir=export_path,
                                        inference_tensor_parallel=1,
                                        inference_pipeline_parallel=1)

            with open(f"{export_path}/config.json", "r") as f:
                tensorrt_llm_config = json.load(f)

            # Workaround for MOE router quantization
            if "moe_num_experts" in tensorrt_llm_config and qformat != "full_prec":
                if "exclude_modules" not in tensorrt_llm_config["quantization"]:
                    # Append router and lm_head because we need both excluded
                    tensorrt_llm_config["quantization"]["exclude_modules"] = [
                        "router", "lm_head"
                    ]
                else:
                    tensorrt_llm_config["quantization"]["exclude_modules"].append(
                        "router")

            with open(f"{export_path}/config.json", "w") as f:
                json.dump(tensorrt_llm_config, f, indent=4)

            # Workaround for Modelopt 0.9.x fp8_kv_cache knob issue
            if qformat == 'fp8' and kv_cache_dtype is None:
                with open(f"{export_path}/config.json", "r") as f:
                    tensorrt_llm_config = json.load(f)
                tensorrt_llm_config["quantization"]["kv_cache_quant_algo"] = None
                with open(f"{export_path}/config.json", "w") as f:
                    json.dump(tensorrt_llm_config, f, indent=4)

            # Workaround for share_embedding_table
            if pp_size == 1:
                with safetensors.safe_open(f"{export_path}/rank0.safetensors",
                                        framework='pt',
                                        device='cpu') as f:
                    share_embedding_table = 'lm_head.weight' not in f.keys()
                if share_embedding_table:
                    with open(f"{export_path}/config.json", "r") as f:
                        tensorrt_llm_config = json.load(f)
                    tensorrt_llm_config["share_embedding_table"] = True
                    with open(f"{export_path}/config.json", "w") as f:
                        json.dump(tensorrt_llm_config, f, indent=4)

            # Workaround for gpt2 position embedding
            if model_type == 'gpt2':
                for rank in range(tp_size):
                    weights = {}
                    with safetensors.safe_open(
                            f"{export_path}/rank{rank}.safetensors",
                            framework='pt',
                            device='cpu') as f:
                        for key in f.keys():
                            weights[key] = f.get_tensor(key)
                    if 'transformer.positional_embedding.weight' in weights:
                        weights[
                            'transformer.position_embedding.weight'] = weights.pop(
                                'transformer.positional_embedding.weight')
                    safetensors.torch.save_file(
                        weights, f"{export_path}/rank{rank}.safetensors")

            # Workaround for qwen version
            if model_type == 'qwen':
                with open(f"{export_path}/config.json", "r") as f:
                    tensorrt_llm_config = json.load(f)
                qwen_config = AutoConfig.from_pretrained(model_dir,
                                                        trust_remote_code=True)
                tensorrt_llm_config["qwen_type"] = qwen_config.model_type
                tensorrt_llm_config[
                    "intermediate_size"] = qwen_config.intermediate_size
                with open(f"{export_path}/config.json", "w") as f:
                    json.dump(tensorrt_llm_config, f, indent=4)

            torch.cuda.empty_cache(
            )  # otherwise torch is keeping using GPU, other routine like build engine has less free GPU to use
            end_time = time.time()
            print(
                "Quantized model exported to {} \nTotal time used {:.2f} s.".format(
                    export_path, end_time - start_time))


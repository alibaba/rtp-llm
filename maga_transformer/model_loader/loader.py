import gc
import logging
import os
import torch
from collections import OrderedDict
import safetensors

from typing import Optional
from maga_transformer.utils.time_util import timer_wrapper
from maga_transformer.utils.util import check_with_info
from maga_transformer.utils.model_weight import WeightStyle
from maga_transformer.utils.database import BaseDatabase
from maga_transformer.lora.lora_weights import LoRAWeights
from maga_transformer.device import get_current_device
from maga_transformer.model_loader.load_config import LoadConfig
from maga_transformer.model_loader.model_weight_info import ModelDeployWeightInfo, ModelWeightInfo, ModelWeights


class ModelLoader:
    def __init__(self,
                 weights_info: ModelDeployWeightInfo,
                 compute_dtype: torch.dtype,
                 database: BaseDatabase):
        self._weights_info = weights_info
        self._model_weights_info: Optional[ModelWeightInfo] = self._weights_info.create_model_weight_info(database)

        use_fp32 = os.environ.get("USE_FLOAT32", None) is not None
        if use_fp32:
            compute_dtype = torch.float32

        self._load_config: LoadConfig = self._weights_info.create_load_config(compute_dtype, database, get_current_device())

    @timer_wrapper(description="load weights")
    @torch.inference_mode()
    def load_weights(self, device: str):
        if self._weights_info.weight_style == WeightStyle.RTP_LLM_STYLE:
            return self._load_from_ft_style(device)

        weights = self._create_model_weights(device)
        convert_device = self._choose_weight_convert_device(device)  # choose convert device to avoid out of mem
        logging.info(f"load weight by device: {convert_device}")

        for (layer_id, name, tensor) in self.prepare_weights(convert_device):
            if convert_device != device:
                tensor = tensor.to(device)
            if layer_id is not None and self._load_config.vit_separation != 1:
                weights.set_layer_weight(layer_id, name, tensor)
            else:
                weights.set_global_weight(name, tensor)

        return weights

    def load_raw_tensor(self, name: str, device: str, datatype: torch.dtype = torch.float16):
        tensors = self._load_config.database.load_tensor(name, datatype)
        if len(tensors) != 1:
            raise Exception(f"load tensor {name} failed, get len=={len(tensors)}")
        loaded_tensor = tensors[0].to(device)
        return loaded_tensor

    def load_lora_weights(self, adapter_name: str, lora_path: str, device: str = 'cpu'):
        lora_weights = LoRAWeights(self._load_config.num_layers)
        # set lora rank
        self._load_config.database.load_lora(adapter_name, lora_path)
        lora_config = self._load_config.database.get_lora_config(adapter_name)
        lora_alpha = lora_config.lora_alpha
        rank = lora_config.rank
        lora_weights.set_lora_rank(rank)
        logging.info(f"load lora weight for adapter {adapter_name}, lora_rank:{rank}")
        if self._weights_info.weight_style == WeightStyle.RTP_LLM_STYLE:
            raise ValueError("load_lora_weights only support non-ft-style weight")

        for id in range(self._load_config.num_layers):
            result = self._load_layer_lora_weights(adapter_name, id, device)
            for name, tensor in result.items():
                lora_weights.set_layer_weight(False, id, name, tensor)

        lora_weights.apply_scale(lora_alpha / rank) # apply scale
        self._load_config.database.remove_lora(adapter_name)    
        return lora_weights
        
    def dump_weight_as_ft_style(self, device: str, output_dir: str):
        check_with_info(not self._load_config.is_ft_style_weight, "dump_weight_as_ft_style only support non-ft-style weight")
        tp_rank = self._load_config.tp_rank
        dp_rank = self._load_config.dp_rank
        ep_rank = self._load_config.ep_rank
        weights = self._create_model_weights(device)

        filename_prefix = f"{output_dir}/model-{tp_rank:02d}-{dp_rank:02d}-"
        os.makedirs(output_dir, exist_ok=True)

        max_size = 6 * 1024**3  # 6GB
        part_idx = 0
        current_size = 0
        current_dict = OrderedDict()

        def maybe_save():
            nonlocal current_size, part_idx, current_dict
            if current_size >= max_size:
                filename = f"{filename_prefix}part-{part_idx:05d}.safetensors"
                safetensors.torch.save_file(current_dict, filename)
                logging.info(f"Saved partition {part_idx} ({current_size/1024**3:.2f}GB)")
                # 显式释放内存
                del current_dict
                current_dict = OrderedDict()
                part_idx += 1
                current_size = 0

        for (layer_id, name, tensor) in self.prepare_weights(device):
            if layer_id is not None:
                tensor_name = f"{weights.layer_weight_prefix(tp_rank, dp_rank, ep_rank)}{layer_id}.{name}"
            else:
                tensor_name = f"{weights.global_weight_prefix(tp_rank,dp_rank, ep_rank)}{name}"
            tensor_size = tensor.numel() * tensor.element_size()
            current_dict[tensor_name] = tensor.cpu().contiguous()
            current_size += tensor_size
            maybe_save()
            # self.force_clean_cuda_memory()

        # 保存最后剩余部分
        if current_dict:
            filename = f"{filename_prefix}part-{part_idx:05d}.safetensors"
            safetensors.torch.save_file(current_dict, filename)
            logging.info(f"Saved final partition {part_idx} ({current_size/1024**3:.2f}GB)")
            del current_dict

    @timer_wrapper(description="load_from_ft_style")
    def _load_from_ft_style(self, device: str):
        num_layers = self._load_config.num_layers
        tp_rank = self._load_config.tp_rank
        dp_rank = self._load_config.dp_rank
        ep_rank = self._load_config.ep_rank

        model_weights = ModelWeights(num_layers, device, self._load_config.compute_dtype)
        layer_weight_prefix = ModelWeights.layer_weight_prefix(tp_rank, dp_rank, ep_rank)
        global_weight_prefix = ModelWeights.global_weight_prefix(tp_rank, dp_rank, ep_rank)
        direct_io = self._load_config.exported_device.support_dio_load
        # 清空现有的权重
        weights = [ {} for _ in range(num_layers)]
        global_weights = {}
        # 重新构建权重
        all_tensors = self._load_config.database.load_tensors_by_prefix((layer_weight_prefix, global_weight_prefix), device, direct_io=direct_io)
        for key, tensor in all_tensors.items():
            if key.startswith(layer_weight_prefix):
                # 解析键名，例如 "layers.0.weight"
                parts = key[len(layer_weight_prefix):].split(".")
                layer_id = int(parts[0])
                name = ".".join(parts[1:])
                # 将张量移动到设备，并设置到对应的层
                check_with_info(len(tensor) == 1, f"{name} have {len(tensor)} tensor)")
                weights[layer_id][name] = tensor[0].to(device)
            elif key.startswith(global_weight_prefix):
                name = key[len(global_weight_prefix):]
                check_with_info(len(tensor) == 1, f"{name} have {len(tensor)} tensor)")
                global_weights[name] = tensor[0].to(device)
        model_weights.weights = weights
        model_weights.global_weights = global_weights
        model_weights.is_ft_style_weight = True
        return model_weights

    def prepare_weights(self, device: str):
        if self._load_config.vit_separation != 1:
            for id in range(self._load_config.num_layers):
                results =  self._load_layer_weights(id, device)
                for (name, tensor) in results.items():
                    yield (id, name, tensor)

        for weight in self._model_weights_info.weights:
            weights = weight.load(self._load_config.database, None, device, self._load_config)
            for (name, tensor) in weights.items():
                yield (None, name, tensor)


    @staticmethod
    def force_clean_cuda_memory():
        """安全清理显存，避免残留引用"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _create_model_weights(self, device):
        return ModelWeights(self._load_config.num_layers, device, self._load_config.compute_dtype)

    def _choose_weight_convert_device(self, current_device):
        if "FORCE_CPU_LOAD_WEIGHTS" in os.environ:
            return "cpu"
        model_size = self._weights_info.config.eval_model_size()
        device_mem_info = self._load_config.exported_device.get_mem_info()
        if device_mem_info is None:
            import psutil
            vmem = psutil.virtual_memory()
            free_mem = vmem.free / (1024.0 ** 2) / self._tp_size
            return "cpu"
        else:
            free_mem = device_mem_info.free / (1024.0 ** 2)
        model_mem = model_size / self._load_config.tp_size / (1024.0 ** 2)
        return current_device if free_mem * 0.8 > model_mem else "cpu"

    def _load_layer_weights(self, layer_id: int, device: str):
        assert isinstance(self._model_weights_info.layer_weights[0], list)
        layer_weights = self._model_weights_info.layer_weights[layer_id]
        weights = {}
        for weight in layer_weights:
            res = weight.load(self._load_config.database, layer_id, device, self._load_config)
            weights.update(res)
        return weights

    def _load_layer_lora_weights(self, lora_name: str, layer_id: int, device: str):
        assert isinstance(self._model_weights_info.layer_weights[0], list)
        layer_weights = self._model_weights_info.layer_weights[layer_id]
        weights = {}
        for weight in layer_weights:
            res = weight.load_lora(self._load_config.database, layer_id, device, self._load_config, lora_name)
            weights.update(res)
        return weights

def get_model_loader(weights_info: ModelDeployWeightInfo,
                     compute_dtype: torch.dtype,
                     database: BaseDatabase) -> ModelLoader:
    if weights_info._head_num % weights_info.tp_size != 0:
            raise Exception('invalid tp_size %d for config.head_num %d' \
                        % (weights_info.tp_size, weights_info._head_num))
    if weights_info._head_num_kv % weights_info.tp_size != 0 and weights_info._head_num_kv != 1:
        raise Exception('invalid tp_size %d for config.head_num_kv %d' \
                        % (weights_info.tp_size, weights_info._head_num_kv))
    return ModelLoader(weights_info, compute_dtype, database)
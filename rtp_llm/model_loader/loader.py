import gc
import logging
import os
import torch
from collections import OrderedDict
import safetensors
import torch.nn.functional as F

from typing import List, Optional
from rtp_llm.utils.fuser import fetch_remote_file_to_local
from rtp_llm.utils.time_util import timer_wrapper
from rtp_llm.utils.util import check_with_info
from rtp_llm.utils.model_weight import WeightStyle, W
from rtp_llm.utils.database import BaseDatabase, CkptDatabase
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.config.task_type import TaskType
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.eplb.ep_balancer import ExpertBalancer
from rtp_llm.lora.lora_weights import LoRAWeights
from rtp_llm.device import get_current_device
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.model_weight_info import ModelDeployWeightInfo, ModelWeightInfo, ModelWeights


class ModelLoader:
    def __init__(self,
                 task_type: TaskType,
                 weights_info: ModelDeployWeightInfo,
                 misc_weights_info: Optional[CustomAtomicWeight],
                 compute_dtype: torch.dtype,
                 database: BaseDatabase):
        self._task_type = task_type
        self._weights_info = weights_info
        self._misc_weights_info: Optional[CustomAtomicWeight] = misc_weights_info
        self._model_weights_info: Optional[ModelWeightInfo] = self._weights_info.create_model_weight_info(database)

        use_fp32 = os.environ.get("USE_FLOAT32", None) is not None
        if use_fp32:
            compute_dtype = torch.float32

        self._init_eplb_config(self._weights_info, compute_dtype)

        self._load_config: LoadConfig = self._weights_info.create_load_config(compute_dtype, database, get_current_device())

    @property
    def weights_info(self):
        return self._weights_info

    @timer_wrapper(description="load weights")
    @torch.inference_mode()
    def load_weights(self, device: str):
        if self._load_config.is_ft_style_weight:
            weights = self._load_from_ft_style(device)
        else:
            weights = self._load_from_scratch(device)

        # load dynamic weight
        self._load_dynamic_weights(weights, device)
        # load eplb weight
        self._init_eplb_weight(weights, device)
        return weights

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
                save_max_retry_times = 2   # maybe fuse is unstable
                for i in range(save_max_retry_times):
                    try:
                        safetensors.torch.save_file(current_dict, filename)
                        logging.info(f"Saved partition {part_idx} ({current_size/1024**3:.2f}GB)")
                        break
                    except Exception as e:
                        logging.error(f"Failed to save partition {part_idx}: {e}")
                        if i == save_max_retry_times - 1:
                            raise e
                        else:
                            logging.info(f"Failed to save partition {part_idx}: {e}, Retrying...")
                            continue
                # release gpu memory
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
            self.force_clean_cuda_memory()

        # save last partition
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

        for weight in self._misc_weights_info:
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

    def _load_from_scratch(self, device: str):
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


    def _load_dynamic_weights(self, weight: ModelWeights, device: str):
        assert weight is not None, "weight is None"

        embedding_weight = weight.global_weights.get(W.embedding, None)
        if embedding_weight != None:
            self._weights_info.config.embedding_size = embedding_weight.shape[0]
            logging.info(f"embedding_size is {self._weights_info.config.embedding_size}, vocab size is {self._weights_info.config.vocab_size}")

        if self._load_config.vit_separation != 1:
            if self._task_type == TaskType.LANGUAGE_MODEL:
                lm_head_w = weight.steal_global_weight(W.lm_head)
                if lm_head_w == None:
                    lm_head_w = weight.global_weights[W.embedding]
                if self._weights_info.config.normalize_lm_head_weight:
                    lm_head_w = F.normalize(lm_head_w)
                if self._weights_info.config.logit_scale != 1.0:
                    lm_head_w = self._weights_info.config.scale_logit * lm_head_w
                weight.set_global_weight(W.lm_head, lm_head_w)
            else:
                # Some LLM can be used for other tasks, e.g. classification, in which case lm_head is not needed
                weight.steal_global_weight(W.lm_head)

            pos_weight = weight.global_weights.get(W.positional_embedding, None)
            if pos_weight != None:
                if pos_weight.shape[0] < self._weights_info.config.max_seq_len:
                    raise Exception(f"positon_weight has shape: {pos_weight.shape}, but max_seq_len is: {self._weights_info.config.max_seq_len} > {pos_weight.shape[0]}")
                pos_weight = pos_weight[:self._weights_info.config.max_seq_len].to(device)
                weight.set_global_weight(W.positional_embedding, pos_weight)

            dynamic_weights = self._weights_info.create_dynamic_weights()
            if dynamic_weights:
                for dynamic_weight in dynamic_weights:
                    dynamic_w = dynamic_weight.load(self._load_config.database, None, device, self._load_config)
                    weight.set_global_weight(dynamic_weight.name, dynamic_w.get(dynamic_weight.name))


    def _init_redundant_expert(self, config: GptInitModelParameters):
        if config.expert_num == 0:
            return

        expert_num = config.expert_num
        ep_size = config.ep_size
        layer_num = config.layer_num
        phy_exp_num = config.phy_exp_num

        phy2log = LoadConfig.create_redundant_expert(layer_num=layer_num,
                                                           expert_num=expert_num,
                                                           phy_exp_num=phy_exp_num,
                                                           ep_size=ep_size,
                                                           num_nodes=config.num_nodes)
        config.phy2log = phy2log

    def _init_eplb_config(self, weights_info: ModelDeployWeightInfo, compute_dtype: torch.dtype):
        self._init_redundant_expert(weights_info.config)
        if weights_info.config.enable_eplb:
            model_path = None
            if weights_info.config.is_mtp:
                model_path = weights_info.config.ckpt_path
            else:
                model_path = fetch_remote_file_to_local(
                    os.environ.get(
                        "ORIGINAL_CHECKPOINT_PATH", weights_info.config.ckpt_path
                    )
                )

            ep_lb_database = CkptDatabase(model_path)
            self.ep_balancer = ExpertBalancer(
                weights_info=weights_info,
                compute_dtype=compute_dtype,
                phy2log=weights_info.config.phy2log,
                database=ep_lb_database
            )
            weights_info.config.py_eplb = self.ep_balancer

    def _init_eplb_weight(self, weight: ModelWeights, device: str):
        expert_num = self._load_config.expert_num
        redundant_expert = self._load_config.phy_exp_num - expert_num
        layer_num = self._load_config.num_layers
        phy2log = self._load_config.phy2log

        if expert_num == 0 or (not self._weights_info.config.enable_eplb and redundant_expert == 0):
            logging.info("don't need to init eplb weight, skip...")
            return

        # init logic_expert_cnt and log2phy
        for layer_id in range(layer_num):
            logic_expert_cnt = torch.zeros((expert_num,), dtype=torch.int32)
            log2phy = torch.empty((expert_num, redundant_expert + 1), dtype=torch.int32).fill_(-1)
            layer_phy2log = phy2log[layer_id]

            for phy_exp_id, expert_id in enumerate(layer_phy2log):
                cnt = logic_expert_cnt[expert_id]
                log2phy[expert_id, cnt] = phy_exp_id
                logic_expert_cnt[expert_id] += 1

            weight.weights[layer_id][W.logic_expert_cnt] = logic_expert_cnt.contiguous().to(device)
            weight.weights[layer_id][W.log2phy] = log2phy.contiguous().to(device)


def get_model_loader(task_type: TaskType,
                     weights_info: ModelDeployWeightInfo,
                     misc_weights_info: Optional[CustomAtomicWeight],
                     compute_dtype: torch.dtype,
                     database: BaseDatabase) -> ModelLoader:
    if weights_info._head_num % weights_info.tp_size != 0:
            raise Exception('invalid tp_size %d for config.head_num %d' \
                        % (weights_info.tp_size, weights_info._head_num))
    if weights_info._head_num_kv % weights_info.tp_size != 0 and weights_info._head_num_kv != 1:
        raise Exception('invalid tp_size %d for config.head_num_kv %d' \
                        % (weights_info.tp_size, weights_info._head_num_kv))
    return ModelLoader(task_type, weights_info, misc_weights_info, compute_dtype, database)

import os
import re
import gc
import logging
import multiprocessing
import torch
import traceback
import torch.serialization
from typing import List, Set, Optional, Tuple, Any
from typing_extensions import Self
from itertools import repeat
from maga_transformer.device import get_current_device
from maga_transformer.utils.model_weight import ModelDeployWeightInfo, ModelWeightInfo, \
    WeightInfo, W, ModelWeights
from maga_transformer.utils.util import check_with_info
from maga_transformer.lora.lora_weights import LoRAWeights
from maga_transformer.utils.time_util import timer_wrapper
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.utils.database import BaseDatabase, LoraConfig, CkptDatabase

class WeightLog:
    """
    record how many tensor had been loaded, accessed and missed.
    """
    accessed_tensors:   Set[str]
    loaded_tensors:     Set[str]
    missed_tensors:     Set[str]

    def __init__(self) -> None:

        self.accessed_tensors   = set([])
        self.loaded_tensors     = set([])
        self.missed_tensors     = set([])

    def record_accessed_tensor(self, name: str) -> None:
        self.accessed_tensors.add(name)

    def record_loaded_tensor(self, name: str) -> None:
        self.loaded_tensors.add(name)

    def record_missed_tensor(self, names: Set[str]) -> None:
        self.missed_tensors.update(names - self.accessed_tensors)

    def update(self, other: Self) -> None:
        self.loaded_tensors.update(other.loaded_tensors)
        self.accessed_tensors.update(other.accessed_tensors)
        self.missed_tensors.update(other.missed_tensors)

    def dump(self) -> None:
        logging.debug(f"""
            You have loaded {len(self.loaded_tensors)} tensors
            The loaded tensor is:
            {self.loaded_tensors}
            You have accessed {len(self.accessed_tensors)} tensors
            The accessed tensor is:
            {self.accessed_tensors}
            You have missed {len(self.missed_tensors)} tensors
            The missed tensor is:
            {self.missed_tensors}
        """)


class ModelWeightsLoader:

    def __init__(self, weights_info: ModelDeployWeightInfo, database: BaseDatabase):
        self._num_layers = weights_info._num_layers
        self._tp_size = weights_info.tp_size
        self._tp_rank = weights_info.tp_rank
        self._ep_size = weights_info.ep_size
        self._ep_rank = weights_info.ep_rank
        self._dp_size = weights_info.dp_size
        self._dp_rank = weights_info.dp_rank
        self._ffn_tp_rank = weights_info.ffn_tp_rank
        self._ffn_tp_size = weights_info.ffn_tp_size
        self._tp_split_emb_and_lm_head = weights_info.tp_split_emb_and_lm_head
        self._weights_info = weights_info
        self._weight_log: WeightLog = WeightLog()
        self._lora_log: WeightLog = WeightLog()
        self._database: BaseDatabase = database
        self._merge_lora = False
        self._static_lora_adapter_name = None
        self._use_expert_attention = weights_info.use_expert_attention
        self._exported_device = get_current_device()
        self._is_ft_style_weight = weights_info.is_ft_style_weight
        self._vit_separation = weights_info.vit_separation
        self._disable_merge_w13 = os.getenv('DISALBE_MERGE_W13', '0').lower() == '1'
        logging.info(f"DISALBE_MERGE_W13 {self._disable_merge_w13}")

        if isinstance(self._database, CkptDatabase):
            self._weights_info.process_meta_from_ckpt(self._database.PretrainFileList)
            self._weights_info.process_meta_from_ckpt(self._database.FinetuneFileList)
            self._model_weights_info: Optional[ModelWeightInfo] = None
            if not self._is_ft_style_weight:
                self._model_weights_info: Optional[ModelWeightInfo] = self._weights_info.get_weight_info()
                self._merge_lora = self._model_weights_info.has_lora_weight() and self._database.has_lora() and bool(os.environ.get("MERGE_LORA", 1))
                if self._merge_lora:
                    static_lora_config: LoraConfig = list(self._database.LoraCkpt.LoraFileList.keys())[0]
                    self._static_lora_adapter_name = static_lora_config.name if self._merge_lora else None

        else:
            raise Exception("Unknown database class")
        logging.info(f"merge lora is enable ? : {self._merge_lora}")

    @property
    def is_merge_lora(self):
        return self._merge_lora
    @property
    def static_lora_adapter_name(self):
        return self._static_lora_adapter_name

    @staticmethod
    def force_clean_cuda_memory():
        """安全清理显存，避免残留引用"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def set_data_type(self, data_type):
        self._data_type = data_type

    def show_warns(self, lora_name: str = "", only_dump_lora: bool = False):

        if not only_dump_lora:
            self._weight_log.record_missed_tensor(
                set(self._database.get_pretrain_tensor_names()))
            self._weight_log.dump()

        if lora_name != "":
            self._lora_log.record_missed_tensor(
                set(self._database.get_lora_tensor_names(lora_name)))
            self._lora_log.dump()

    def load_from_ft_style_weight(self, device):
        model_weights = ModelWeights(self._num_layers, device, self._data_type)
        layer_weight_prefix = ModelWeights.layer_weight_prefix(self._tp_rank, self._dp_rank, self._ep_rank)
        global_weight_prefix = ModelWeights.global_weight_prefix(self._tp_rank, self._dp_rank, self._ep_rank)
        direct_io = self._exported_device.support_dio_load
        # 清空现有的权重
        weights = [ {} for id in range(self._num_layers)]
        global_weights = {}
        # 重新构建权重
        all_tensors = self._database.load_tensors_by_prefix((layer_weight_prefix, global_weight_prefix), device, direct_io=direct_io)
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

        if not self._disable_merge_w13:
            update_weights = [ {} for id in range(self._num_layers)]
            compaction_weights = []
            for layer_id, layer_weight_dict in enumerate(weights):
                for weight_name, tensor in layer_weight_dict.items():
                    if weight_name in W.ffn_weights_1:
                        pair_weight_name = W.ffn_pair_weight_name_dict[weight_name]
                        pair_tensor = None
                        for weight_name2, tensor2 in layer_weight_dict.items():
                            if weight_name2 == pair_weight_name:
                                pair_tensor = tensor2
                                break
                        compaction_weights.append((layer_id, W.ffn_merge_weight_name_dict[weight_name], [tensor, pair_tensor]))
                    elif weight_name not in W.ffn_weights_3:
                        update_weights[layer_id][weight_name] = tensor
            for (layer_id, name, [gate_tensor, up_tensor]) in compaction_weights:
                merged_tensor = torch.concat([gate_tensor, up_tensor], dim=-1).continous()
                update_weights[layer_id][name] = merged_tensor
                logging.info(f"merge weight {layer_id}, {name}, gate tensor shape: {gate_tensor.shape}, up tensor shape: {up_tensor.shape}, merge tensor shape:  {merged_tensor.shape}")
                gate_tensor = gate_tensor.cpu()
                up_tensor = up_tensor.cpu()
            weights = update_weights

        model_weights.weights = weights
        model_weights.global_weights = global_weights
        model_weights.is_ft_style_weight = True
        return model_weights

    @timer_wrapper(description="load_weights_from_scratch")
    def load_weights_from_scratch(self, device: str):
        if self._is_ft_style_weight:
            return self.load_from_ft_style_weight(device)
        weights = self.create_model_weights(device)
        convert_device = self._choose_weight_convert_device(device)  # choose convert device to avoid out of mem
        for (layer_id, name, tensor) in self.prepare_weights_from_scratch(convert_device):
            if convert_device != device:
                tensor = tensor.to(device)
            if layer_id is not None and self._vit_separation != 1:
                weights.set_layer_weight(layer_id, name, tensor)
            else:
                weights.set_global_weight(name, tensor)

            self.force_clean_cuda_memory()

        return weights


    def _choose_weight_convert_device(self, current_device):
        if "FORCE_CPU_LOAD_WEIGHTS" in os.environ:
            return "cpu"
        model_size = self._weights_info.config.eval_model_size()
        device_mem_info = self._exported_device.get_mem_info()
        if device_mem_info is None:
            import psutil
            vmem = psutil.virtual_memory()
            free_mem = vmem.free / (1024.0 ** 2) / self._tp_size
            return "cpu"
        else:
            free_mem = device_mem_info.free / (1024.0 ** 2)
        model_mem = model_size / self._tp_size / (1024.0 ** 2)
        return current_device if free_mem * 0.8 > model_mem else "cpu"

    @torch.no_grad()
    def prepare_weights_from_scratch(self, device):
        if self._vit_separation != 1:
            for id in range(self._num_layers):
                results, logs, lora_logs =  self._load_layer_weight(id, device)
                self._weight_log.update(logs)
                if self._merge_lora:
                    self._lora_log.update(lora_logs)
                for (layer_id, name, tensor) in results:
                    tensor = self._exported_device.maybe_rewrite_weight_by_key(name, tensor)
                    yield (layer_id, name, tensor)

        for weight in self._model_weights_info.weights:
            tensor = self._load_and_convert_tensor(weight, datatype=self._data_type, device=device)
            tensor = self._split_and_sanitize_tensor(tensor, weight)
            tensor = self._exported_device.maybe_rewrite_weight_by_key(weight.name, tensor)
            tensor = tensor.to(device)
            yield (None, weight.name, tensor)

    def create_model_weights(self, device):
        return ModelWeights(self._num_layers, device, self._data_type)

    def load_lora_weights_from_scratch(self, lora_name: str, lora_path: str, device: str, num_process=1):
        lora_weights = LoRAWeights(self._num_layers)
        # set lora rank
        self._database.load_lora(lora_name, lora_path)
        lora_config = self._database.get_lora_config(lora_name)
        lora_alpha = lora_config.lora_alpha
        rank = lora_config.rank
        lora_weights.set_lora_rank(rank)

        all_results = [self._load_lora_layer_weight(id, lora_name, device)
                        for id in range(self._num_layers)]
        for results, logs in all_results:
            self._lora_log.update(logs)
            for (int8_flag, layer_id, name, tensor) in results:
                lora_weights.set_layer_weight(
                    int8_flag, layer_id, name, tensor)

        lora_weights.apply_scale(lora_alpha / rank) # apply scale
        self._database.remove_lora(lora_name)
        return lora_weights


    def _load_lora_layer_weight(self, layer_id: int, lora_name: str, device: str):
        use_fp32 = os.environ.get("USE_FLOAT32", None) is not None
        results = []
        layer_weights = self._model_weights_info.lora_weights
        for weight in layer_weights:
            try:
                tensor = self._load_and_convert_lora_tensor(weight, lora_name, layer_id, self._data_type)
                if tensor is None:
                    continue
                tensor = self._split_and_sanitize_tensor(tensor, weight).to(device)
                if use_fp32:
                    tensor = tensor.float()
                results.append((False, layer_id, weight.name, tensor))
            except Exception as e:
                logging.error(f'load {weight.name} in layer {layer_id} failed: {e}')
                raise e
        return results, self._lora_log

    def _load_groupwise_layer_weight(self, layer_weights, layer_id: int, device: str):
        if self._merge_lora:
            raise Exception("lora in groupwise is not implemented yet")
        results = []
        is_moe = self._weights_info.expert_num_ > 0 and layer_id in self._weights_info.moe_layer_index_
        is_gated_activation = self._weights_info._is_gated_activation
        def convert_weight(weight_lists, apply_func):
            for weight_list in weight_lists:
                if not isinstance(weight_list[0], list):
                    qweight = [weight for weight in layer_weights if weight.name == weight_list[0]]
                    qzero  = [weight for weight in layer_weights if weight.name == weight_list[1]]
                    qscale  = [weight for weight in layer_weights if weight.name == weight_list[2]]
                    if len(qweight) == 0:
                        if self._weights_info._is_sparse_head:
                            continue
                        else:
                            raise Exception(f"not found weight {weight_list[0]} in layer {layer_id}")
                    elif len(qweight) > 1:
                        raise Exception(f"found more than one weight {weight_list[0]} in layer {layer_id}")
                    try:
                        qweight_tensor = self._load_and_convert_tensor(qweight[0], layer_id=layer_id, datatype=torch.int32)
                        qzero_tensor = self._load_and_convert_tensor(qzero[0], layer_id=layer_id, datatype=torch.int32)
                        qscale_tensor = self._load_and_convert_tensor(qscale[0], layer_id=layer_id)
                        qweight_tensor = self._split_tensor(qweight[0].name, qweight_tensor)
                        qzero_tensor = self._split_tensor(qzero[0].name, qzero_tensor, bits=self._weights_info._quant_algo.getWeightBits())
                        qscale_tensor = self._split_tensor(qscale[0].name, qscale_tensor)
                        weight, zero, scale = apply_func(qweight_tensor, qzero_tensor, qscale_tensor, device,
                                                        self._weights_info._quant_algo.isGptq(),
                                                        self._weights_info._quant_algo.isAwq(),
                                                        self._weights_info._quant_algo.getWeightBits())
                        results.append((layer_id, qweight[0].name, weight))
                        results.append((layer_id, qzero[0].name, zero))
                        results.append((layer_id, qscale[0].name, scale))
                    except Exception as e:
                        logging.error(f'load groupwise layer_weight in layer {layer_id}.{qweight[0].name} failed: {e} {traceback.format_exc(e)}')
                        raise e
                else:
                    if len(weight_list) != 3:
                        raise Exception(f"groupwise layer_weight in layer {layer_id} should have 3 weight_list")
                    weight_list1 = weight_list[0]
                    weight_list2 = weight_list[1] 
                    merged_weight_list = weight_list[2]

                    qweight1 = [weight for weight in layer_weights if weight.name == weight_list1[0]]
                    qzero1  = [weight for weight in layer_weights if weight.name == weight_list1[1]]
                    qscale1  = [weight for weight in layer_weights if weight.name == weight_list1[2]]

                    qweight2 = [weight for weight in layer_weights if weight.name == weight_list2[0]]
                    qzero2  = [weight for weight in layer_weights if weight.name == weight_list2[1]]
                    qscale2  = [weight for weight in layer_weights if weight.name == weight_list2[2]]

                    merged_qweight = [weight for weight in layer_weights if weight.name == merged_weight_list[0]]

                    if len(qweight1) == 0:
                        if self._weights_info._is_sparse_head:
                            continue
                        else:
                            raise Exception(f"not found weight {weight_list1[0]} in layer {layer_id}")
                    elif len(qweight1) > 1:
                        raise Exception(f"found more than one weight {weight_list1[0]} in layer {layer_id}")
                    
                    if len(qweight2) == 0:
                        if self._weights_info._is_sparse_head:
                            continue
                        else:
                            raise Exception(f"not found weight {weight_list2[0]} in layer {layer_id}")
                    elif len(qweight2) > 1:
                        raise Exception(f"found more than one weight {weight_list2[0]} in layer {layer_id}")

                    if len(merged_qweight) > 0:
                        raise Exception(f"found merged weight {merged_weight_list[0]} in layer {layer_id}")
                    try:
                        qweight_tensor1 = self._load_and_convert_tensor(qweight1[0], layer_id=layer_id, datatype=torch.int32)
                        qzero_tensor1 = self._load_and_convert_tensor(qzero1[0], layer_id=layer_id, datatype=torch.int32)
                        qscale_tensor1 = self._load_and_convert_tensor(qscale1[0], layer_id=layer_id)
                        qweight_tensor1 = self._split_tensor(qweight1[0].name, qweight_tensor1)
                        qzero_tensor1 = self._split_tensor(qzero1[0].name, qzero_tensor1, bits=self._weights_info._quant_algo.getWeightBits())
                        qscale_tensor1 = self._split_tensor(qscale1[0].name, qscale_tensor1)


                        qweight_tensor2 = self._load_and_convert_tensor(qweight2[0], layer_id=layer_id, datatype=torch.int32)
                        qzero_tensor2 = self._load_and_convert_tensor(qzero2[0], layer_id=layer_id, datatype=torch.int32)
                        qscale_tensor2 = self._load_and_convert_tensor(qscale2[0], layer_id=layer_id)
                        qweight_tensor2 = self._split_tensor(qweight2[0].name, qweight_tensor2)
                        qzero_tensor2 = self._split_tensor(qzero2[0].name, qzero_tensor2, bits=self._weights_info._quant_algo.getWeightBits())
                        qscale_tensor2 = self._split_tensor(qscale2[0].name, qscale_tensor2)


                        merged_qweight_tensor = torch.concat([qweight_tensor1, qweight_tensor2], dim=-1).contiguous()
                        merged_qzero_tensor = torch.concat([qzero_tensor1, qzero_tensor2], dim=-1).contiguous()
                        merged_qscale_tensor = torch.concat([qscale_tensor1, qscale_tensor2], dim=-1).contiguous()

                        weight, zero, scale = apply_func(merged_qweight_tensor, merged_qzero_tensor, merged_qscale_tensor, device,
                                                        self._weights_info._quant_algo.isGptq(),
                                                        self._weights_info._quant_algo.isAwq(),
                                                        self._weights_info._quant_algo.getWeightBits())
                        results.append((layer_id, merged_weight_list[0], weight))
                        results.append((layer_id, merged_weight_list[1], zero))
                        results.append((layer_id, merged_weight_list[2], scale))
                    except Exception as e:
                        logging.error(f'load groupwise layer_weight in layer {layer_id}.{merged_weight_list[0]} failed: {e} {traceback.format_exc(e)}')
                        raise e

        convert_weight(W.groupwise_attn_weights, self._exported_device.preprocess_groupwise_weight_params)

        if is_gated_activation:
            ffn_weight_lists = W.groupwise_ffn_weights_3 if self._disable_merge_w13 else W.groupwise_ffn_weights
        else:
            ffn_weight_lists = W.groupwise_ffn_weights_2

        if self._weights_info.moe_style_ == 2:
            if is_moe:
                convert_weight(W.groupwise_partial_moe_weights, self._exported_device.preprocess_moe_groupwise_weight_params)
            convert_weight(ffn_weight_lists, self._exported_device.preprocess_groupwise_weight_params)
        else:
            if is_moe:
                convert_weight(ffn_weight_lists, self._exported_device.preprocess_moe_groupwise_weight_params)
            else:
                convert_weight(ffn_weight_lists, self._exported_device.preprocess_groupwise_weight_params)
        # load act_scales
        if self._weights_info.need_ffn_act_scale:
            try:
                ffn_act_scales_weight = [weight for weight in layer_weights if weight.name == W.ffn_act_s]
                if len(ffn_act_scales_weight) == 1:
                    ffn_act_scales_tensor = self._load_and_convert_tensor(ffn_act_scales_weight[0], layer_id=layer_id, datatype=torch.float16)
                    ffn_act_scales_tensor = self._split_tensor(ffn_act_scales_weight[0].name, ffn_act_scales_tensor).contiguous().clone().to(device)
                    results.append((layer_id, ffn_act_scales_weight[0].name, ffn_act_scales_tensor))
            except Exception as e:
                logging.error(f'load ffn_act_scales_weight in layer {layer_id}.{W.ffn_act_s} failed:{e}')
                raise e
        return results

    def _load_int8_layer_weight(self, layer_weights, layer_id: int, device: str):
        if self._merge_lora:
            raise Exception("lora in sq is not implemented yet")
        results = []
        def load_weight(weight_list, datatype):
            for quant_weight in weight_list:
                qweight = [weight for weight in layer_weights if weight.name == quant_weight]
                if len(qweight) == 0:
                    if self._weights_info._is_sparse_head:
                        continue
                    else:
                        logging.info(f"not found weight {quant_weight} in layer {layer_id}")
                        continue
                elif len(qweight) > 1:
                    raise Exception(f"found more than one weight {quant_weight} in layer {layer_id}")
                try:

                    qweight_tensor = self._load_and_convert_tensor(qweight[0], layer_id=layer_id, datatype=datatype)
                    qweight_tensor = self._split_tensor(qweight[0].name, qweight_tensor).contiguous().clone().to(device)
                    # int4
                    if (qweight_tensor.dim() == 2):
                        qweight_tensor = qweight_tensor.reshape(qweight_tensor.shape[-1], -1)
                    results.append((layer_id, qweight[0].name, qweight_tensor))

                    logging.debug(f"load qweight tensor {quant_weight} in layer {layer_id} and shape is {qweight_tensor.shape}, dtype:{qweight_tensor.dtype}, datatype:{datatype}")

                except Exception as e:
                    logging.error(f'load quant layer_weight in layer {layer_id} {qweight[0].name} failed: {e}')
                    raise e

        if self._weights_info._quant_algo.isFp8() and self._weights_info._quant_algo.isGroupwise():
            weight_list = W.int8_attn_weights + W.int8_ffn_weights + W.int8_ffn_weights_2 + W.int8_partial_moe_weights_2 + W.int8_partial_moe_weights
            weight_list = [[_[0],_[1]]for _ in set([(_[0],_[1]) for _ in weight_list ])]
            logging.info(f"load weight: {weight_list}")
            load_weight([_[0] for _ in weight_list], torch.float8_e4m3fn)
            load_weight([_[1] for _ in weight_list], torch.float32)

            return results
        elif self._weights_info._quant_algo.isFp8():
            qkv_w_weight = [weight for weight in layer_weights if weight.name == W.attn_qkv_w][0]
            qkv_w_weight_name = qkv_w_weight.weights[0].tensor_name(0)
            tensor_type = self._database.get_tensor_type(qkv_w_weight_name)
            if tensor_type == torch.float8_e4m3fn:
                logging.info(f"fp8 per tensor load type: float8_e4m3fn")
                load_weight(W.sq_quant_weights, torch.float8_e4m3fn)
            else:
                logging.info(f"fp8 per tensor load type: int8")
                load_weight(W.sq_quant_weights, torch.int8)
            load_weight(W.sq_quant_scales, torch.float32)
            return results

        load_weight(W.sq_quant_weights, torch.int8)
        load_weight(W.sq_quant_scales, torch.float32)
        if self._weights_info._quant_algo.isOmniQuant():
            load_weight(W.sq_quant_shifts, torch.float32)
        elif self._weights_info._quant_algo.isPerTensorQuant():
            load_weight(W.static_quant_scales, torch.float32)

        return results

    def _load_layer_weight_and_apply_int8(self, layer_weights, layer_id: int, device: str):
        results = []
        is_moe = self._weights_info.expert_num_ > 0 and layer_id in self._weights_info.moe_layer_index_
        is_gated_activation = self._weights_info._is_gated_activation
        def convert_weight(weight_lists, apply_func, datatype):
            for weight_list in weight_lists:
                if not isinstance(weight_list[0], list):
                    qweight = [weight for weight in layer_weights if weight.name == weight_list[0]]
                    scale_name = weight_list[1]
                    if len(qweight) == 0:
                        continue
                    elif len(qweight) > 1:
                        raise Exception(f"found more than one weight {weight_list[0]} in layer {layer_id}")
                    try:
                        qweight_tensor = self._load_and_convert_tensor(qweight[0], layer_id=layer_id, datatype=datatype)
                        if self._merge_lora:
                            qweight_tensor = self.apply_lora(qweight_tensor, qweight[0], layer_id)
                        qweight_tensor = self._split_and_sanitize_tensor(qweight_tensor, qweight[0])

                        weight, scale = apply_func(qweight_tensor, device)
                        results.append((layer_id, qweight[0].name, weight))
                        results.append((layer_id, scale_name, scale))
                    except Exception as e:
                        logging.error(f'load int8 layer_weight {weight_list[0]} in layer {layer_id} failed: {e}')
                        raise e
                else:
                    weight_list1 = weight_list[0]
                    weight_list2 = weight_list[1]
                    weight_list3 = weight_list[2]
                    qweight1 = [weight for weight in layer_weights if weight.name == weight_list1[0]]
                    scale1 = weight_list1[1]
                    qweight2 = [weight for weight in layer_weights if weight.name == weight_list2[0]]
                    scale2 = weight_list2[1]
                    if len(qweight1) == 0:
                        continue
                    elif len(qweight1) > 1:
                        raise Exception(f"found more than one weight {weight_list1[0]} in layer {layer_id}")
                    if len(qweight2) == 0:
                        continue
                    elif len(qweight2) > 1:
                        raise Exception(f"found more than one weight {weight_list2[0]} in layer {layer_id}")
                    try:
                        qweight_tensor1 = self._load_and_convert_tensor(qweight1[0], layer_id=layer_id, datatype=datatype)
                        if self._merge_lora:
                            qweight_tensor1 = self.apply_lora(qweight_tensor1, qweight1[0], layer_id)
                        qweight_tensor1 = self._split_and_sanitize_tensor(qweight_tensor1, qweight1[0])

                        qweight_tensor2 = self._load_and_convert_tensor(qweight2[0], layer_id=layer_id, datatype=datatype)
                        if self._merge_lora:
                            qweight_tensor2 = self.apply_lora(qweight_tensor2, qweight2[0], layer_id)
                        qweight_tensor2 = self._split_and_sanitize_tensor(qweight_tensor2, qweight2[0])

                        merge_qweight = torch.concat([qweight_tensor1, qweight_tensor2], dim=-1).contiguous()

                        merge_weight, merge_scale = apply_func(merge_qweight, device)
                        results.append((layer_id, weight_list3[0], merge_weight))
                        results.append((layer_id, weight_list3[1], merge_scale))
                    except Exception as e:
                        logging.error(f'load int8 layer_weight {weight_list[0]} in layer {layer_id} failed: {e}')
                        raise e

        if self._use_expert_attention:
            # for CogVLM2, moe is not supported and gated activation is enabled
            assert not is_moe and is_gated_activation, (
                "CogVLM2 shouldn't use moe mode and gated activation."
            )
        else:
            # for other models, we empty the list of vision weights
            W.int8_attn_vision_weights.clear()
            W.int8_vision_ffn_weights.clear()

        convert_weight(W.int8_attn_weights, self._exported_device.apply_int8, self._data_type)
        convert_weight(W.int8_attn_vision_weights, self._exported_device.apply_int8, self._data_type)

        if is_gated_activation:
            if is_moe:
                ffn_weight_lists = W.int8_partial_moe_weights
            elif self._disable_merge_w13:
                ffn_weight_lists = W.int8_ffn_weights_3
            else:
                ffn_weight_lists = W.int8_ffn_weights
        else:
            ffn_weight_lists = W.int8_ffn_weights_2 if is_moe == False else W.int8_partial_moe_weights_2

        if is_moe:
            convert_weight(ffn_weight_lists, self._exported_device.moe_apply_int8, self._data_type)
        else:
            convert_weight(ffn_weight_lists, self._exported_device.apply_int8, self._data_type)
            convert_weight(W.int8_vision_ffn_weights, self._exported_device.apply_int8, self._data_type)

        if self._weights_info.moe_style_ == 2:
            if self._disable_merge_w13:
                onvert_weight(W.int8_ffn_weights_3, self._exported_device.apply_int8, self._data_type)
            else:
                convert_weight(W.int8_ffn_weights, self._exported_device.apply_int8, self._data_type)


        return results

    def _is_quant_weight(self, w):
        quant_algo = self._weights_info._quant_algo
        if quant_algo.isWeightOnlyPerCol() or quant_algo.isSmoothQuant() or quant_algo.isOmniQuant() or quant_algo.isFp8():
            return w.name in W.quant_w
        if quant_algo.isGroupwise():
            return w.name in W.groupwise_quant_params or w.name in W.quant_w
        return False


    def _sort_layer_weight_by_read_order(self, layer_weights, layer_id):
        # 创建文件名到权重的映射
        weighted_entries = []
        # 创建文件名到权重的映射
        # 生成排序键的辅助逻辑
        def get_key(file_name):
            # 将文件名拆分为字母和数字部分
            parts = re.split(r'(\d+)', file_name)
            # 将数字部分转为整数，其他部分保持小写
            return tuple(int(p) if p.isdigit() else p.lower() for p in parts if p)

        for weight in layer_weights:
            file_keys = []
            file_indices = []
            orig_orders = []
            for ckpt_weight in weight.weights:
                tensor_name = ckpt_weight.tensor_name(layer_id)
                try:
                    file_name, idx = self._database.get_tensor_order(tensor_name)[0]
                    if file_name:  # 过滤无效文件名
                        file_keys.append(get_key(file_name))
                        file_indices.append(idx)
                except Exception as e:
                    logging.warning(f'layer: {layer_id} load tensor :{tensor_name} file meta failed')

            if file_keys:
                # 生成主排序键（取最小文件键）
                main_key = min(file_keys) if file_keys else (float('inf'),)
                main_idx = file_indices[file_keys.index(main_key)] if file_keys else 0

                weighted_entries.append( (main_key, main_idx, weight) )

        # 执行三级排序
        sorted_weights = sorted(
            weighted_entries,
            key=lambda x: (x[0], x[1], get_key(x[2].name))  # 文件名 > 索引 > 权重名
        )

        return [entry[2] for entry in sorted_weights]


    def _load_layer_weight(self, layer_id: int, device: str):
        use_fp32 = os.environ.get("USE_FLOAT32", None) is not None
        results = []
        if isinstance(self._model_weights_info.layer_weights[0], List):
            layer_weights = self._model_weights_info.layer_weights[layer_id]
        else:
            layer_weights = self._model_weights_info.layer_weights
        if self._weights_info.moe_style_ == 2 and layer_id not in self._weights_info.moe_layer_index_:
            layer_weights = self._trunc_layer_weights_for_partial_moe(layer_weights)

        layer_weights = self._sort_layer_weight_by_read_order(layer_weights, layer_id)

        for weight in layer_weights:
            try:
                if self._is_quant_weight(weight):
                    continue
                tensor = self._load_and_convert_tensor(weight, layer_id=layer_id, datatype=self._data_type, device=device)

                if self._merge_lora:
                    tensor = self.apply_lora(tensor, weight, layer_id)

                tensor = self._split_and_sanitize_tensor(tensor, weight)

                if use_fp32:
                    tensor = tensor.float()
                logging.debug(f"load qweight tensor {weight} in layer {layer_id} and shape is {tensor.shape}, dtype:{tensor.dtype}, datatype:{self._data_type}")
                results.append((layer_id, weight.name, tensor))
            except Exception as e:
                logging.error(f'load {weight.name} in layer {layer_id} failed: {e}')
                raise e
        quant_algo = self._weights_info._quant_algo
        if quant_algo.isGroupwise() and not quant_algo.isFp8():
            results.extend(self._load_groupwise_layer_weight(layer_weights, layer_id=layer_id, device=device))
        elif quant_algo.isSmoothQuant() or quant_algo.isOmniQuant() or quant_algo.isPerTensorQuant() or quant_algo.isFp8():
            results.extend(self._load_int8_layer_weight(layer_weights, layer_id=layer_id, device=device))
        elif quant_algo.isWeightOnlyPerCol():
            results.extend(self._load_layer_weight_and_apply_int8(layer_weights, layer_id=layer_id, device=device))
        
        if not self._disable_merge_w13:
            update_results = []
            compaction_weights = []
            for (layer_id, weight_name, tensor) in results:
                if weight_name in W.ffn_weights_1:
                    pair_weight_name = W.ffn_pair_weight_name_dict[weight_name]
                    pair_tensor = None
                    for (_, weight_name2, tensor2) in results:
                        if weight_name2 == pair_weight_name:
                            pair_tensor = tensor2
                            break
                    compaction_weights.append((layer_id, W.ffn_merge_weight_name_dict[weight_name], [tensor, pair_tensor]))
                elif weight_name not in W.ffn_weights_3:
                    update_results.append((layer_id, weight_name, tensor))
            for (layer_id, name, [gate_tensor, up_tensor]) in compaction_weights:
                merged_tensor = torch.concat([gate_tensor, up_tensor], dim=-1).contiguous()
                update_results.append((layer_id, name, merged_tensor))
                logging.info(f"merge weight {layer_id}, {name}, gate tensor shape: {gate_tensor.shape}, up tensor shape: {up_tensor.shape}, merge tensor shape:  {merged_tensor.shape}")
                gate_tensor = gate_tensor.cpu()
                up_tensor = up_tensor.cpu()
            results = update_results
        gc.collect()
        torch.cuda.empty_cache()
        return results, self._weight_log, self._lora_log

    def _trunc_layer_weights_for_partial_moe(self, layer_weights: List[WeightInfo]):
        truncated_layer_weights = []
        for weight in layer_weights:
            if weight.name not in W.partial_moe_w:
                truncated_layer_weights.append(weight)
        return truncated_layer_weights

    def apply_lora(self, tensor: torch.Tensor, weight: WeightInfo, layer_id: int):

        lora_a = self._model_weights_info.find_lora_a(weight)
        lora_b = self._model_weights_info.find_lora_b(weight)
        if lora_a is None or lora_b is None:
            return tensor
        lora_name = self._database.get_first_lora_name()
        if lora_name is None:
            raise Exception(f"invalid empty lora name")

        lora_a_tensor = self._load_and_convert_lora_tensor(lora_a, lora_name, layer_id)
        lora_b_tensor = self._load_and_convert_lora_tensor(lora_b, lora_name, layer_id)

        if lora_a_tensor is None or lora_b_tensor is None:
            return tensor

        scale = self._database.get_lora_config(lora_name).get_scale()
        # "addmm_impl_cpu_" not implemented for 'Half'
        if lora_b_tensor.dim() == 3 and lora_a_tensor.dim() == 2:
            lora_b_tensor = lora_b_tensor.reshape(lora_b_tensor.shape[0], lora_b_tensor.shape[1] * lora_b_tensor.shape[2])
            merge_tensor = (lora_a_tensor.type(torch.float32) @ lora_b_tensor.type(torch.float32) * scale).type(tensor.dtype).to(tensor.device)
        # moe
        elif lora_b_tensor.dim() == 3 and lora_a_tensor.dim() == 3:
            merge_tensor = torch.bmm(lora_a_tensor.type(torch.float32), lora_b_tensor.type(torch.float32) * scale).type(tensor.dtype).to(tensor.device)
        else:
            merge_tensor = (lora_a_tensor.type(torch.float32) @ lora_b_tensor.type(torch.float32) * scale).type(tensor.dtype).to(tensor.device)

        shape = tensor.shape
        tensor = tensor.reshape(tensor.nelement()) + merge_tensor.reshape(tensor.nelement())
        tensor = tensor.reshape(shape)

        del lora_a_tensor
        del lora_b_tensor
        return tensor

    def load_tensor(self, name: str, datatype: torch.dtype = torch.float16) -> List[torch.Tensor]:
        self._weight_log.record_accessed_tensor(name)
        return self._database.load_tensor(name, datatype)

    def load_lora_tensor(self, lora_name: str, tensor_name: str) -> List[torch.Tensor]:
        self._lora_log.record_accessed_tensor(tensor_name)
        return self._database.load_lora_tensor(lora_name, tensor_name)

    def _load_and_convert_lora_tensor(self, weight_info: WeightInfo, lora_name:str, layer_id: Optional[int] = None, datatype: torch.dtype = torch.float16):
        before_merge_tensors = []
        self._lora_log.record_loaded_tensor(weight_info.name)
        for ckpt_weight in weight_info.weights:
            ckpt_tensor_name = ckpt_weight.tensor_name(layer_id)
            ckpt_tensor = self.load_lora_tensor(lora_name, ckpt_tensor_name)

            hidden_size = self._weights_info._hidden_size
            rank = self._database.get_lora_config(lora_name).rank
            num_heads = self._weights_info._head_num
            num_key_value_heads = self._weights_info._head_num_kv
            head_dim = self._weights_info._size_per_head

            tensor= []
            # q
            if ckpt_tensor == [] and "q_proj" in ckpt_tensor_name:
                if (ckpt_weight.name.count("lora_A")):
                    # [head_num * head_dim, rank]
                    tensor.append(torch.zeros(rank, hidden_size))
                elif (ckpt_weight.name.count("lora_B")):
                     # [rank, kv_head_num * head_dim]
                    tensor.append(torch.zeros(num_heads*head_dim, rank))
                else:
                    raise Exception(f"invalid ckpt tensor name :{ckpt_weight.name}")
            # k
            elif ckpt_tensor == [] and "k_proj" in ckpt_tensor_name:
                if (ckpt_weight.name.count("lora_A")):
                    tensor.append(torch.zeros(rank, hidden_size))
                elif (ckpt_weight.name.count("lora_B")):
                    tensor.append(torch.zeros(num_key_value_heads*head_dim, rank))
                else:
                    raise Exception(f"invalid ckpt tensor name :{ckpt_weight.name}")
            # v
            elif ckpt_tensor == [] and "v_proj" in ckpt_tensor_name:
                if (ckpt_weight.name.count("lora_A")):
                    tensor.append(torch.zeros(rank, hidden_size))
                elif (ckpt_weight.name.count("lora_B")):
                    tensor.append(torch.zeros(num_key_value_heads*head_dim, rank))
                else:
                    raise Exception(f"invalid ckpt tensor name :{ckpt_weight.name}")

            elif ckpt_tensor == []:
                return None

            else:
                tensor = tensor + ckpt_tensor
            before_merge_tensors.append(ckpt_weight.merge_fun(tensor))

        after_merge_tensor = weight_info.process_fun(before_merge_tensors)

        return after_merge_tensor.to(datatype)

    def _load_and_convert_tensor(self, weight_info: WeightInfo, layer_id: Optional[int] = None, datatype: torch.dtype = torch.float16, device='cpu'):
        convert_type = datatype if weight_info.data_type is None else weight_info.data_type
        before_merge_tensors = []
        self._weight_log.record_loaded_tensor(weight_info.name)

        if weight_info.name in W.fp32_weights_list:
            convert_type = torch.float32

        for ckpt_weight in weight_info.weights:
            name = ckpt_weight.tensor_name(layer_id)
            try:
                before_merge_tensors.append(ckpt_weight.merge_fun([x.to(device) for x in self.load_tensor(name, convert_type)]))
            except Exception as e:
                raise Exception('load %s failed, except: %s' % (name, str(e)))

        after_merge_tensor = weight_info.process_fun(before_merge_tensors).to(convert_type)
        return after_merge_tensor

    def _split_and_sanitize_tensor(self, tensor: torch.Tensor, weight: WeightInfo):
        return self._sanitize(self._split_tensor(weight.name, tensor))

    def _split_tensor(self, name: str, tensor: torch.Tensor, bits=4) -> torch.Tensor:
        if self._tp_size <= 1 and self._ep_size <= 1 and self._dp_size <= 1:
            if name in [W.moe_w1, W.moe_w2]:
                return self._exported_device.shuffle_moe_weight(tensor, self._data_type, name)
            return tensor
        if (not self._tp_split_emb_and_lm_head and
            name in [W.lm_head, W.lm_head_b, W.embedding, W.positional_embedding, W.token_type_embedding]):
            return tensor
        if not self._model_weights_info.tp_strategy:
            raise Exception('this model not support TP')
        split_fun = self._model_weights_info.tp_strategy.get(name)
        if not split_fun:
            raise Exception('this model not support TP: ' + name)
        ts = split_fun(t=tensor,
                       tp=self._tp_size,
                       tp_rank=self._tp_rank,
                       ep=self._ep_size,
                       ep_rank=self._ep_rank,
                       dp=self._dp_size,
                       dp_rank=self._dp_rank,
                       ffn_tp_rank = self._ffn_tp_rank,
                       ffn_tp_size = self._ffn_tp_size,
                       hidden_size=self._weights_info._hidden_size,
                       head_num=self._weights_info._head_num,
                       head_num_kv=self._weights_info._head_num_kv,
                       size_per_head=self._weights_info._size_per_head,
                       bits=bits,
                       use_stack_weight=self._weights_info._use_stack_weight,
                       )
        if name in [W.moe_w1, W.moe_w2]:
                return self._exported_device.shuffle_moe_weight(ts, self._data_type, name)
        return ts

    # 避免被 storage 影响多用显存
    def _sanitize(self, t):
        return t.contiguous().clone()


def get_model_weights_loader(weights_info: ModelDeployWeightInfo, database: CkptDatabase, compute_dtype):
    if weights_info._head_num % weights_info.tp_size != 0:
        raise Exception('invalid tp_size %d for config.head_num %d' \
                        % (weights_info.tp_size, weights_info._head_num))
    if weights_info._head_num_kv % weights_info.tp_size != 0 and weights_info._head_num_kv != 1:
        raise Exception('invalid tp_size %d for config.head_num_kv %d' \
                        % (weights_info.tp_size, weights_info._head_num_kv))

    model_weights_loader = ModelWeightsLoader(weights_info, database)
    model_weights_loader.set_data_type(compute_dtype)
    return model_weights_loader

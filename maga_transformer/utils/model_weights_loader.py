import os
import re
import gc
import logging
import multiprocessing
import torch
import torch.serialization
from typing import List, Set, Optional, Tuple, Any
from typing_extensions import Self
from itertools import repeat
from maga_transformer.utils.model_weight import ModelDeployWeightInfo, ModelWeightInfo, \
    WeightInfo, W, ModelWeights, LoRAWeights
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.utils.database import BaseDatabase, CkptFileInfo, LoraConfig, ModuleDatabase, CkptDatabase, DictDatabase
from maga_transformer.utils.util import get_mem_info

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
        self._tp_split_emb_and_lm_head = weights_info.tp_split_emb_and_lm_head
        self._weights_info = weights_info
        self._database: BaseDatabase = database
        self._weight_log: WeightLog = WeightLog()
        self._lora_log: WeightLog = WeightLog()
        self._database: BaseDatabase = database
        self._merge_lora = False
        self._static_lora_adapter_name = None

        if isinstance(self._database, CkptDatabase):
            self._weights_info.process_meta_from_ckpt(self._database.PretrainFileList)
            self._weights_info.process_meta_from_ckpt(self._database.FinetuneFileList)
            self._model_weights_info: ModelWeightInfo = self._weights_info.get_weight_info()
            self._merge_lora = self._model_weights_info.has_lora_weight() and self._database.has_lora() and bool(os.environ.get("MERGE_LORA", 1))
            if self._merge_lora:
                static_lora_config: LoraConfig = list(self._database.LoraCkpt.LoraFileList.keys())[0]
                self._static_lora_adapter_name = static_lora_config.name if self._merge_lora else None
        elif isinstance(self._database, ModuleDatabase):
            self._weights_info.process_meta_from_dict(dict(self._database.ref_module.state_dict()))
            self._model_weights_info: ModelWeightInfo = self._weights_info.get_weight_info()
        elif isinstance(self._database, DictDatabase):
            self._weights_info.process_meta_from_dict(self._database.ref_dict)
            self._model_weights_info: ModelWeightInfo = self._weights_info.get_weight_info()
        else:
            raise Exception("Unknown database class")
        logging.info(f"merge lora is enable ? : {self._merge_lora}")

    @property
    def is_merge_lora(self):
        return self._merge_lora
    @property
    def static_lora_adapter_name(self):
        return self._static_lora_adapter_name
        
    def set_data_type(self, data_type):
        self._data_type = data_type

    def show_warns(self, lora_name: str = "", only_dump_lora: bool = False):
        if isinstance(self._database, ModuleDatabase):
            return
        
        if not only_dump_lora:
            self._weight_log.record_missed_tensor(
                set(self._database.get_pretrain_tensor_names()))
            self._weight_log.dump()

        if lora_name != "":
            self._lora_log.record_missed_tensor(
                set(self._database.get_lora_tensor_names(lora_name)))
            self._lora_log.dump()

    def load_weights_from_scratch(self, device: str='cuda:0', num_process=1):
        weights = ModelWeights(self._num_layers)
        if num_process > 1:
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(num_process) as pool:
                all_results = pool.starmap(
                    self._load_layer_weight,
                    zip(range(self._num_layers),
                        repeat(device)))
        else:
            all_results = [self._load_layer_weight(id, device)
                           for id in range(self._num_layers)]
            
        for results, logs, lora_logs in all_results:
            self._weight_log.update(logs)
            if self._merge_lora:
                self._lora_log.update(lora_logs)
            for (layer_id, name, tensor) in results:
                weights.append_layer_weight(layer_id, name, tensor)
        for weight in self._model_weights_info.weights:
            tensor = self._load_and_convert_tensor(weight)
            tensor = self._split_and_sanitize_tensor(tensor, weight)
            tensor = tensor.to('cuda')
            weights.append_pytorch_weight(weight.name, tensor)

        for name, tensor in self._load_medusa_weights(self._model_weights_info.medusa_weights):
            weights.append_pytorch_weight(name, tensor)

        return weights

    def _load_medusa_weights(self, medusa_weights: List[WeightInfo], device: str='cuda:0') -> List[Tuple[str, torch.Tensor]]:
        if len(medusa_weights) == 0:
            return []
        results: List[Tuple[str, torch.Tensor]] = []
        assert len(medusa_weights) == 1
        for weight in medusa_weights[0].weights:
            name = weight.tensor_name(None)
            results.append((name, self.load_tensor(name)[0]))
        return results

    def load_lora_weights_from_scratch(self, lora_name: str, device: str='cuda:0', num_process=1):
        lora_weights = LoRAWeights(self._num_layers)
        # set lora rank
        lora_config = self._database.get_lora_config(lora_name)
        lora_alpha = lora_config.lora_alpha
        rank = lora_config.rank
        lora_weights.set_lora_rank(rank)

        all_results = [self._load_lora_layer_weight(id, lora_name, device)
                        for id in range(self._num_layers)]
        for results, logs in all_results:
            self._lora_log.update(logs)
            for (int8_flag, layer_id, name, tensor) in results:
                lora_weights.append_layer_weight(
                    int8_flag, layer_id, name, tensor)

        lora_weights.apply_scale(lora_alpha / rank) # apply scale
        return lora_weights


    def _load_lora_layer_weight(self, layer_id: int, lora_name: str, device: str = "cuda:0"):
        use_fp32 = os.environ.get("USE_FLOAT32", None) is not None
        results = []
        layer_weights = self._model_weights_info.lora_weights
        for weight in layer_weights:
            try:
                tensor = self._load_and_convert_lora_tensor(weight, lora_name, layer_id)
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
                    qzero_tensor = self._split_tensor(qzero[0].name, qzero_tensor)
                    qscale_tensor = self._split_tensor(qscale[0].name, qscale_tensor)
                    weight, zero, scale = apply_func(qweight_tensor, qzero_tensor, qscale_tensor, device,
                                                     self._weights_info._quant_algo.isGptq(),
                                                     self._weights_info._quant_algo.isAwq(),
                                                     self._weights_info._quant_algo.getWeightBits())
                    results.append((layer_id, qweight[0].name, weight))
                    results.append((layer_id, qzero[0].name, zero))
                    results.append((layer_id, qscale[0].name, scale))
                except Exception as e:
                    logging.error(f'load groupwise layer_weight in layer {layer_id}.{qweight[0].name} failed: {e}')
                    raise e

        convert_weight(W.groupwise_attn_weights, self.preprocess_groupwise_weight_params)

        if is_gated_activation:
            ffn_weight_lists = W.groupwise_ffn_weights
        else:
            ffn_weight_lists = W.groupwise_ffn_weights_2

        if self._weights_info.moe_style_ == 2:
            if is_moe:
                convert_weight(W.groupwise_partial_moe_weights, self.preprocess_moe_groupwise_weight_params)
            convert_weight(ffn_weight_lists, self.preprocess_groupwise_weight_params)
        else:
            if is_moe:
                convert_weight(ffn_weight_lists, self.preprocess_moe_groupwise_weight_params)
            else:
                convert_weight(ffn_weight_lists, self.preprocess_groupwise_weight_params)
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
                    results.append((layer_id, qweight[0].name, qweight_tensor))

                except Exception as e:
                    logging.error(f'load smooth_quant layer_weight in layer {layer_id} failed: {e}')
                    raise e

        load_weight(W.sq_quant_weights, torch.int8)
        load_weight(W.sq_quant_scales, torch.float32)
        if self._weights_info._quant_algo.isOmniQuant():
            load_weight(W.sq_quant_shifts, torch.float32)
        return results

    def _load_layer_weight_and_apply_int8(self, layer_weights, layer_id: int, device: str):
        results = []
        is_moe = self._weights_info.expert_num_ > 0 and layer_id in self._weights_info.moe_layer_index_
        is_gated_activation = self._weights_info._is_gated_activation
        def convert_weight(weight_lists, apply_func):
            for weight_list in weight_lists:
                qweight = [weight for weight in layer_weights if weight.name == weight_list[0]]
                scale_name = weight_list[1]
                if len(qweight) == 0:
                    if self._weights_info._is_sparse_head:
                        continue
                    else:
                        raise Exception(f"not found weight {weight_list[0]} in layer {layer_id}")
                elif len(qweight) > 1:
                    raise Exception(f"found more than one weight {weight_list[0]} in layer {layer_id}")
                try:
                    qweight_tensor = self._load_and_convert_tensor(qweight[0], layer_id=layer_id)
                    if self._merge_lora:
                        qweight_tensor = self.apply_lora(qweight_tensor, qweight[0], layer_id)
                    qweight_tensor = self._split_and_sanitize_tensor(qweight_tensor, qweight[0])

                    weight, scale = apply_func(qweight_tensor, device)
                    results.append((layer_id, qweight[0].name, weight))
                    results.append((layer_id, scale_name, scale))
                except Exception as e:
                    logging.error(f'load int8 layer_weight {weight_list[0]} in layer {layer_id} failed: {e}')
                    raise e

        convert_weight(W.int8_attn_weights, self.apply_int8)

        if is_gated_activation:
            ffn_weight_lists = W.int8_ffn_weights if is_moe == False else W.int8_partial_moe_weights
        else:
            ffn_weight_lists = W.int8_ffn_weights_2 if is_moe == False else W.int8_partial_moe_weights_2

        if is_moe:
            convert_weight(ffn_weight_lists, self.moe_apply_int8)
        else:
            convert_weight(ffn_weight_lists, self.apply_int8)

        if self._weights_info.moe_style_ == 2:
            # convert_weight(W.int8_partial_moe_weights, self.moe_apply_int8)
            convert_weight(W.int8_ffn_weights, self.apply_int8)

        return results

    def _is_quant_weight(self, w):
        quant_algo = self._weights_info._quant_algo
        if quant_algo.isWeightOnlyPerCol() or quant_algo.isSmoothQuant() or quant_algo.isOmniQuant():
            return w.name in W.quant_w
        if quant_algo.isGroupwise():
            return w.name in W.groupwise_quant_params or w.name in W.quant_w
        return False

    def _load_layer_weight(self, layer_id: int, device: str = "cuda:0"):
        use_fp32 = os.environ.get("USE_FLOAT32", None) is not None
        results = []
        if isinstance(self._model_weights_info.layer_weights[0], List):
            layer_weights = self._model_weights_info.layer_weights[layer_id]
        else:
            layer_weights = self._model_weights_info.layer_weights
        if self._weights_info.moe_style_ == 2 and layer_id not in self._weights_info.moe_layer_index_:
            layer_weights = self._trunc_layer_weights_for_partial_moe(layer_weights)
        for weight in layer_weights:
            try:
                if self._is_quant_weight(weight):
                    continue
                tensor = self._load_and_convert_tensor(weight, layer_id=layer_id)

                if self._merge_lora:
                    tensor = self.apply_lora(tensor, weight, layer_id)

                tensor = self._split_and_sanitize_tensor(tensor, weight).to(device)

                if use_fp32:
                    tensor = tensor.float()

                results.append((layer_id, weight.name, tensor))
            except Exception as e:
                logging.error(f'load {weight.name} in layer {layer_id} failed: {e}')
                raise e
        quant_algo = self._weights_info._quant_algo
        if quant_algo.isGroupwise():
            results.extend(self._load_groupwise_layer_weight(layer_weights, layer_id=layer_id, device=device))
        elif quant_algo.isSmoothQuant() or quant_algo.isOmniQuant():
            results.extend(self._load_int8_layer_weight(layer_weights, layer_id=layer_id, device=device))
        elif quant_algo.isWeightOnlyPerCol():
            results.extend(self._load_layer_weight_and_apply_int8(layer_weights, layer_id=layer_id, device=device))
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

    def apply_int8(self, tensor: torch.Tensor, device: str):
        shape = tensor.shape
        int8_weight, int8_scale = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix( # type: ignore
            tensor.reshape([shape[0], -1]).cpu(), torch.int8)
        int8_weight = int8_weight.reshape(shape)
        return int8_weight.to(device), int8_scale.to(device)

    def moe_apply_int8(self, tensor: torch.Tensor, device: str):
        assert tensor.dim() == 3
        tensor_list = torch.chunk(tensor, tensor.shape[0], dim=0)
        int8_weights = []
        int8_scales = []
        for t in tensor_list:
            t = torch.squeeze(t).transpose(1,0).contiguous()
            shape = t.shape
            weight, scale = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix( # type: ignore
                t.reshape([shape[0], -1]).cpu(), torch.int8)
            int8_weights.append(weight)
            int8_scales.append(scale)
        int8_weight = torch.stack(int8_weights, dim=0)
        int8_scale = torch.stack(int8_scales, dim=0)
        return int8_weight.to(device), int8_scale.to(device)

    def unpack_int32_into_int16(self, w_packed: torch.Tensor, int8: bool):
        if int8:
            return w_packed.contiguous().view(torch.uint8).to(torch.int16)
        # unpack inputs packed in int32/float32 into uint4 and store them in int8 format
        w_packed_int4x2 = w_packed.contiguous().view(torch.uint8)
        w_unpacked = torch.zeros(w_packed_int4x2.shape[0],
                                 w_packed_int4x2.shape[1] * 2,
                                 dtype=torch.int8)
        w_unpacked[:, ::2] = w_packed_int4x2 % 16
        w_unpacked[:, 1::2] = w_packed_int4x2 // 16
        return w_unpacked.to(torch.int16).contiguous()

    def preprocess_moe_groupwise_weight_params(self, qweight_int32, qzeros_int32, scales_fp16, device: str, gptq: bool, awq: bool, weight_bits: int):
        assert qweight_int32.dim() == 3
        qweight_list = torch.chunk(tensor, qweight_int32.shape[0], dim=0)
        qzeros_list = torch.chunk(tensor, qzeros_int32.shape[0], dim=0)
        scales_list = torch.chunk(tensor, scales_fp16.shape[0], dim=0)
        processed_weights = []
        processed_zeros = []
        processed_scalses = []
        for w, z, s in zip(qweight_list, qzeros_list, scales_list):
            w = torch.squeeze(w)
            z = torch.squeeze(z)
            s = torch.squeeze(s)
            p_w, p_z, p_s = self.preprocess_groupwise_weight_params(w, z, s, device, gptq, awq, weight_bits)
            processed_weights.append(p_w)
            processed_zeros.append(p_z)
            processed_scalses.append(p_s)
        processed_weights = torch.stack(processed_weights, dim=0)
        processed_zeros = torch.stack(processed_zeros, dim=0)
        processed_scalses = torch.stack(processed_scalses, dim=0)
        return processed_weights, processed_zeros, processed_scalses

    def reverse_awq_order(self, ori_tensor: torch.Tensor):
        # AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

        assert ori_tensor.shape[-1] % 8 == 0
        reorder_tensor = ori_tensor.reshape(-1, 2,4).transpose(2,1).reshape(ori_tensor.shape)

        return reorder_tensor

    def preprocess_groupwise_weight_params(self, qweight_int32, qzeros_int32, scales_fp16, device: str,
                                           gptq: bool, awq: bool, weight_bits: int):
        GPTQ_FLAG = 1 if gptq == True else 0
        qweight = qweight_int32.reshape(qweight_int32.shape[0], -1).cpu()
        qzeros = qzeros_int32.reshape(qzeros_int32.shape[0], -1).cpu()
        scales_fp16 = scales_fp16.reshape(scales_fp16.shape[0], -1).cpu()
        packer = torch.ops.fastertransformer.pack_int8_tensor_to_packed_int4
        preprocessor = torch.ops.fastertransformer.preprocess_weights_for_mixed_gemm
        is_int8 = weight_bits == 8
        if is_int8:
            zero_shift = 128
            quant_type = torch.int8
        else:
            zero_shift = 8
            quant_type = torch.quint4x2

        if awq:
            qweight = self.unpack_int32_into_int16(qweight, is_int8).contiguous() - zero_shift
            qweight = self.reverse_awq_order(qweight)
        elif gptq:
            qweight = self.unpack_int32_into_int16(qweight.T, is_int8).T.contiguous() - zero_shift

        qweight = qweight.to(torch.int8)
        if not is_int8:
            qweight = packer(qweight)
        qweight_interleaved = preprocessor(qweight, quant_type)

        # zero = 0 if qzeros_int32 = -2004318072 torch.int32 for awq
        # zero = 0 if qzeros_int32 = 2004318071  torch.int32 for gptq
        qzeros = self.unpack_int32_into_int16(qzeros, is_int8)
        if awq:
            qzeros = self.reverse_awq_order(qzeros)

        # zeros = zeros * scales
        UINT_TO_INT_FLAG = 1
        zeros_x_scales_fp16 = (-qzeros + zero_shift * UINT_TO_INT_FLAG -
                               GPTQ_FLAG) * scales_fp16
        zeros_x_scales_fp16 = zeros_x_scales_fp16.half()

        # return processed interleaved weight, original scales and zeros * scales
        return qweight_interleaved.contiguous().to(device),  zeros_x_scales_fp16.contiguous().to(device), scales_fp16.contiguous().to(device)


    def load_tensor(self, name: str, datatype: torch.dtype = torch.float16) -> List[torch.Tensor]:
        self._weight_log.record_accessed_tensor(name)
        return self._database.load_tensor(name, datatype)

    def load_lora_tensor(self, lora_name: str, tensor_name: str) -> List[torch.Tensor]:
        self._lora_log.record_accessed_tensor(tensor_name)
        return self._database.load_lora_tensor(lora_name, tensor_name)

    def _load_and_convert_lora_tensor(self, weight_info: WeightInfo, lora_name:str, layer_id: Optional[int] = None):
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

        return after_merge_tensor

    def _load_and_convert_tensor(self, weight_info: WeightInfo, layer_id: Optional[int] = None, datatype: str = torch.float16):
        before_merge_tensors = []
        self._weight_log.record_loaded_tensor(weight_info.name)
        for ckpt_weight in weight_info.weights:
            name = ckpt_weight.tensor_name(layer_id)
            try:
                before_merge_tensors.append(ckpt_weight.merge_fun(self.load_tensor(name, datatype)))
            except Exception as e:
                raise Exception('load %s failed, except: %s' % (name, str(e)))

        after_merge_tensor = weight_info.process_fun(before_merge_tensors).to(datatype)
        return after_merge_tensor

    def _split_and_sanitize_tensor(self, tensor: torch.Tensor, weight: WeightInfo):
        return self._sanitize(self._split_tensor(weight.name, tensor))

    def _split_tensor(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if self._tp_size <= 1:
            return tensor
        if (not self._tp_split_emb_and_lm_head and
            name in [W.lm_head, W.lm_head_b, W.embedding, W.positional_embedding, W.token_type_embedding]):
            return tensor
        if not self._model_weights_info.tp_strategy:
            raise Exception('this model not support TP')
        split_fun = self._model_weights_info.tp_strategy.get(name)
        if not split_fun:
            raise Exception('this model not support TP: ' + name)
        kv_broadcast = self._weights_info._head_num_kv == 1
        qkv_hidden_size = self._weights_info._size_per_head *(self._weights_info._head_num + self._weights_info._head_num_kv * 2)
        ts = split_fun(tensor, self._tp_size, tp_rank=self._tp_rank, hidden_size=self._weights_info._hidden_size, qkv_hidden_size=qkv_hidden_size, kv_broadcast=kv_broadcast)
        return ts

    # 避免被 storage 影响多用显存
    def _sanitize(self, t):
        return t.contiguous().clone().to(self._data_type)

def estimate_load_parallel_num(config, tp_size):
    parallel_num = os.environ.get('LOAD_CKPT_NUM_PROCESS', None)
    if parallel_num is None:
        return 1
    parallel_num = int(parallel_num)
    if parallel_num > 0:
        logging.info(f'load weights by {parallel_num} process from env')
        return parallel_num
    model_size = config.eval_model_size()
    cuda_runtime_mem = 2
    weight_compute_mem = 2
    free_mem = get_mem_info().free / (1024.0 ** 3)
    model_mem = model_size / tp_size / (1024.0 ** 3)
    parallel_num = int((free_mem - model_mem) / (weight_compute_mem + cuda_runtime_mem))
    parallel_num = min(max(parallel_num, 1), 4) # 以防并发太多影响 io 效率
    # hippo mount 最大并发是16， 超过16就会file not found
    parallel_num = min(parallel_num, 16 // g_parallel_info.local_world_size)
    if model_mem < 1:
        parallel_num = 1 # 单元测试
    logging.info(f'free_mem: {free_mem:.2f} model_mem: {model_mem:.2f}, load weights by {parallel_num} process')
    return parallel_num

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

import os
import gc
import logging
import multiprocessing
import torch
import torch.serialization
from typing import List, Set, Optional, Tuple
from itertools import repeat
from maga_transformer.utils.model_weight import ModelDeployWeightInfo, ModelWeightInfo, \
    WeightInfo, W, ModelWeights, LoRAWeights, LoRAMap
from maga_transformer.utils.ckpt_database import CkptDatabase, CkptFileInfo

class ModelWeightsLoader:

    def __init__(self, weights_info: ModelDeployWeightInfo, database: CkptDatabase):
        self._num_layers = weights_info._num_layers
        self._tp_size = weights_info.tp_size
        self._tp_rank = weights_info.tp_rank
        self._tp_split_emb_and_lm_head = weights_info.tp_split_emb_and_lm_head
        self._weights_info = weights_info
        self._weight_access_log: Set[str] = set([])
        self._database: CkptDatabase = database
        self._ckpt_metas: List[CkptFileInfo] = []
        self._all_tensor_names: Set[str] = set([])
        for ckpt in self._ckpt_metas:
            self._all_tensor_names.update(ckpt.get_tensor_names())
        self._weights_info.process_meta(self._database.PretrainFileList)
        self._weights_info.process_meta(self._database.FinetuneFileList)
        self.preprocessed = 'ft_module' in self._all_tensor_names
        self._model_weights_info: ModelWeightInfo = self._weights_info.get_weight_info(self.preprocessed, self._all_tensor_names)
        self._merge_lora = self._model_weights_info.has_lora_weight() and self._database.has_lora()
        logging.info(f"merge lora {self._merge_lora}")
        
    def set_data_type(self, data_type):
        self._data_type = data_type

    def show_warns(self):
        not_access_set = self._all_tensor_names - self._weight_access_log
        if len(not_access_set) > 0:
            logging.warning('weights not access: %s', str(not_access_set))
        else:
            logging.info("all weights have been accessed")

    def _process_ckpt_metas(self):
        self._weights_info.process_meta(self._ckpt_metas)

    def load_weights_from_scratch(self, int8_mode: int, device: str='cuda:0'):
        weights = ModelWeights(self._num_layers)
        ctx = multiprocessing.get_context('spawn')
        num_process = int(os.environ.get('LOAD_CKPT_NUM_PROCESS', '1'))
        if num_process > 1:
            with ctx.Pool(num_process) as pool:
                all_results = pool.starmap(
                    self._load_layer_weight,
                    zip(range(self._num_layers),
                        repeat(int8_mode),
                        repeat(device)))
        else:
            all_results = [self._load_layer_weight(id, int8_mode, device)
                           for id in range(self._num_layers)]
        for results, logs in all_results:
            self._weight_access_log.update(logs)
            for (int8_flag, layer_id, name, tensor) in results:
                weights.append_layer_weight(
                    int8_flag, layer_id, name, tensor)
        for weight in self._model_weights_info.weights:
            tensor = self._load_and_convert_tensor(weight)
            tensor = self._split_and_sanitize_tensor(tensor, weight)
            tensor = tensor.to('cuda')
            weights.append_pytorch_weight(weight.name, tensor)
        
        for name, tensor in self._load_medusa_weights(self._model_weights_info.medusa_weights):
            weights.append_pytorch_weight(name, tensor)

        # dynamic lora
        if not self._merge_lora:
            for lora_config in self._database.LoraFileList.keys():
                lora_name = lora_config.name
                lora_weights = self.load_lora_weights_from_scratch(lora_name, int8_mode, device)
                # save lora_weight to lora_map to avoid free memory
                _ = weights.lora_map.add_lora_name(lora_name, lora_weights)
                # self.lora_map.padding_lora_rank(lora_weight.lora_rank)
        
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

    def load_lora_weights_from_scratch(self, lora_name: str, int8_mode: int, device: str='cuda:0'):
        lora_weights = LoRAWeights(self._num_layers)
        # set lora rank
        lora_alpha = self._database.get_lora_config(lora_name).lora_alpha
        rank = self._database.get_lora_config(lora_name).rank
        lora_weights.set_lora_rank(rank)

        all_results = [self._load_lora_layer_weight(id, lora_name, device)
                        for id in range(self._num_layers)]
        for results, logs in all_results:
            self._weight_access_log.update(logs)
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

        gc.collect()
        torch.cuda.empty_cache()
        return results, self._weight_access_log

    def _load_layer_weight(self, layer_id: int, int8_mode: int, device: str = "cuda:0"):
        use_fp32 = os.environ.get("USE_FLOAT32", None) is not None
        results = []
        if isinstance(self._model_weights_info.layer_weights[0], List):
            layer_weights = self._model_weights_info.layer_weights[layer_id]
        else:
            layer_weights = self._model_weights_info.layer_weights
        for weight in layer_weights:
            try:

                int8_flag = int8_mode == 1 and (weight.name in W.int8_quant_w)

                tensor = self._load_and_convert_tensor(weight, layer_id)
                
                if self._merge_lora:
                    tensor = self.apply_lora(tensor, weight, layer_id)

                tensor = self._split_and_sanitize_tensor(tensor, weight)

                if int8_flag:
                    tensor = self.apply_int8(tensor, device)
                else:
                    tensor = tensor.to(device)

                if use_fp32:
                    tensor = tensor.float()

                results.append((int8_flag, layer_id, weight.name, tensor))
            except Exception as e:
                logging.error(f'load {weight.name} in layer {layer_id} failed: {e}')
                raise e

        gc.collect()
        torch.cuda.empty_cache()
        return results, self._weight_access_log

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
        if len(lora_b_tensor.shape) == 3:
            lora_b_tensor = lora_b_tensor.reshape(lora_b_tensor.shape[0], lora_b_tensor.shape[1] * lora_b_tensor.shape[2])

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
        dummy = torch.IntTensor([0]).to(self._data_type)
        return (dummy.to(device), int8_weight.to(device), int8_scale.to(device))


    def load_tensor(self, name: str) -> List[torch.Tensor]:
        self._weight_access_log.add(name)
        return self._database.load_tensor(name)
    
    def load_lora_tensor(self, lora_name: str, tensor_name: str) -> List[torch.Tensor]:
        return self._database.load_lora_tensor(lora_name, tensor_name)

    def _load_and_convert_lora_tensor(self, weight_info: WeightInfo, lora_name:str, layer_id: Optional[int] = None):
        before_merge_tensors = []
        for ckpt_weight in weight_info.weights:
            tensor = self.load_lora_tensor(lora_name, ckpt_weight.tensor_name(layer_id))
            if tensor == []:
                return None
            before_merge_tensors.append(ckpt_weight.merge_fun(tensor))
            
        after_merge_tensor = weight_info.process_fun(before_merge_tensors)

        return after_merge_tensor


    def _load_and_convert_tensor(self, weight_info: WeightInfo, layer_id: Optional[int] = None):
        before_merge_tensors = []
        for ckpt_weight in weight_info.weights:
            before_merge_tensors.append(ckpt_weight.merge_fun(self.load_tensor(ckpt_weight.tensor_name(layer_id))))
            
        after_merge_tensor = weight_info.process_fun(before_merge_tensors)

        return after_merge_tensor

    def _split_and_sanitize_tensor(self, tensor: torch.Tensor, weight: WeightInfo):
        return self._sanitize(self._split_tensor(weight.name, tensor))

    def _split_tensor(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if self._tp_size <= 1:
            return tensor
        if (not self._tp_split_emb_and_lm_head and
            name in [W.lm_head, W.lm_head_b, W.embedding]):
            return tensor
        if not self._model_weights_info.tp_strategy:
            raise Exception('this model not support TP')
        split_fun = self._model_weights_info.tp_strategy.get(name)
        if not split_fun:
            raise Exception('this model not support TP: ' + name)
        kv_broadcast = self._weights_info._head_num_kv == 1
        ts = split_fun(tensor, self._tp_size, tp_rank=self._tp_rank, hidden_size=self._weights_info._hidden_size, kv_broadcast=kv_broadcast)
        return ts

    # 避免被 storage 影响多用显存
    def _sanitize(self, t):
        return t.contiguous().clone().to(self._data_type)

def load_weights(weights_info: ModelDeployWeightInfo, database: CkptDatabase, compute_dtype):
    if weights_info._head_num % weights_info.tp_size != 0:
        raise Exception('invalid tp_size %d for config.head_num %d' \
                        % (weights_info.tp_size, weights_info._head_num))
    if weights_info._head_num_kv % weights_info.tp_size != 0 and weights_info._head_num_kv != 1:
        raise Exception('invalid tp_size %d for config.head_num_kv %d' \
                        % (weights_info.tp_size, weights_info._head_num_kv))
    
    model_weights_loader = ModelWeightsLoader(weights_info, database)
    model_weights_loader.set_data_type(compute_dtype)
    weights = model_weights_loader.load_weights_from_scratch(weights_info._int8_mode)
    model_weights_loader.show_warns()
    return weights

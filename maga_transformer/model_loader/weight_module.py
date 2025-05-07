import functools
import logging
import torch
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type, Union
import inspect
import weakref
from maga_transformer.model_loader.load_config import LoadConfig
from maga_transformer.utils.database import BaseDatabase
from maga_transformer.utils.model_weight import CkptWeightInfo, W, WeightStyle, identity, sp_0, sp_head_lora, sp_id, sp_neg1
import traceback


class WeightModule(ABC):
    _registry = weakref.WeakValueDictionary()
    _cache = weakref.WeakKeyDictionary()
    lora_base_name = "base_model.model.{}.{}.weight"
    lora_A_suffix = 'lora_A'
    lora_B_suffix = 'lora_B'

    def __init__(self, 
                 name: str, 
                 lora_a_process_func: Optional[Callable]=None,
                 lora_b_process_func: Optional[Callable]=None, 
                 lora_a_split_func: Optional[Callable]=None,
                 lora_b_split_func: Optional[Callable]=None,
                 **kwargs: Any):
        self.name = name
        self.weight_style = kwargs.pop("weight_style", WeightStyle.NONE)
        self.lora_a_process_func: Optional[Callable] = lora_a_process_func
        self.lora_b_process_func: Optional[Callable] = lora_b_process_func
        self.lora_a_split_func: Optional[Callable] = lora_a_split_func
        self.lora_b_split_func: Optional[Callable] = lora_b_split_func
        self.lora_a: Optional['WeightModule'] = None
        self.lora_b: Optional['WeightModule'] = None
        self.is_lora = kwargs.pop("is_lora", False)

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    @property
    def lora_a_name(self):
        return f"{self.name}.{self.lora_A_suffix}"

    @property
    def lora_b_name(self):
        return f"{self.name}.{self.lora_B_suffix}"
    
    @classmethod
    def create(
        cls,
        weight_info: "WeightModule",
        quant_algo: Optional[Any] = None
    ) -> "WeightModule":
        if quant_algo is None or not quant_algo.isQuant():
            return weight_info

        if isinstance(weight_info, QuantWeight):
            return weight_info

        if isinstance(weight_info, AtomicWeight):
            valid_classes = [
                c for _, c in cls._registry.items()
                if c.support(quant_algo, weight_info)
            ]
            if not valid_classes:
                return weight_info
            if  len(valid_classes) > 1:
                raise ValueError(f"{weight_info.name} fit too many valid_classes:{valid_classes} with quant={quant_algo} for weight: {weight_info}")
            target_cls = valid_classes[0]

            params = cls.extract_params(target_cls, weight_info, quant_algo)
            return target_cls(**params)
        elif isinstance(weight_info, CompositeWeight):
            target_cls = weight_info.__class__
            params = target_cls.extract_params(target_cls, weight_info, quant_algo)
            return target_cls(**params)
        else:
            raise ValueError(f"Invalid weight_info type: {type(weight_info)}")

    @classmethod
    def from_params(cls, params):
        return cls(**params)

    @classmethod
    def extract_params(
        cls,
        target_cls: Type["WeightModule"],
        weight_info: "WeightModule",
        quant_algo: Any
    ) -> Dict[str, Any]:
        params = {}
        signature = inspect.signature(target_cls.__init__)
        need_var_key = False
        for param in list(signature.parameters.values())[1:]:  # Skip self
            if param.kind == inspect.Parameter.VAR_KEYWORD or param.kind == inspect.Parameter.VAR_POSITIONAL:
                need_var_key = True
                continue

            if param.name == "quant_algo":
                params[param.name] = quant_algo
                continue

            if param.name == 'src_weight_info':
                params['src_weight_info'] = weight_info
                continue

            if hasattr(weight_info, param.name):
                value = getattr(weight_info, param.name)
                # 递归创建子权重
                if param.name == "sub_weights" and isinstance(value, dict):
                    value = [
                        cls.create(v, quant_algo)
                        for _, v in value.items()
                    ]

                params[param.name] = value
            elif param.default != inspect.Parameter.empty:
                params[param.name] = param.default
            else:
                raise ValueError(f"target_cls: {target_cls} Missing required parameter: {param.name}")

        if need_var_key:
            for k, v in weight_info.__dict__.items():
                if isinstance(v, WeightModule):
                    continue
                if k in params:
                    continue
                params[k] = v

        return params

    @classmethod
    @abstractmethod
    def support(cls, quant_algo: Any, src_weight_info: 'WeightModule') -> bool:
        pass

    @torch.inference_mode()
    def load(self, database: BaseDatabase, layer_id: Optional[int], device: str, load_config: LoadConfig):
        raw_tensors = self._load_raw_tensor(database, layer_id, device, load_config)
        logging.debug(f"load weight: {self.name} layer_id: {layer_id}, res:{raw_tensors}")

        if load_config.merge_lora:
            logging.debug(f"merge lora weight: {self.name} layer_id: {layer_id}")
            merged_tensors = self._merge_lora(raw_tensors, database, layer_id, load_config)
        else:
            merged_tensors = raw_tensors

        split_tensors = self._split(merged_tensors, load_config)
        logging.debug(f"split weight: {self.name} layer_id: {layer_id}, res:{split_tensors}")

        processed_tensors = self._postprocess(split_tensors, device, load_config)
        logging.debug(f"postprocess weight: {self.name} layer_id: {layer_id}, res:{processed_tensors}")
        flat_res = {}
        def __extract_tensor(tensors):
            for k,v in tensors.items():
                if isinstance(v, dict):
                    __extract_tensor(v)
                else:
                    flat_res.update({k: v.to(device)})
        __extract_tensor(processed_tensors)
        shape_info = {k: (v.shape, v.dtype) for k, v in flat_res.items()}
        logging.debug(f"extract weight: {self.name} layer_id: {layer_id}, res:{shape_info}")
        return flat_res

    @torch.inference_mode()
    def load_lora(self, database: BaseDatabase, layer_id: Optional[int], device: str, load_config: LoadConfig, lora_name: str):
        try: 
            raw_loras = self._load_raw_lora(database, layer_id, device, load_config, lora_name)
        except Exception as e:
            logging.warning(f"load layer: {layer_id} lora tensor {self.lora_a} or {self.lora_b} failed: traceback: {traceback.format_exc()}")
            return {}
        if raw_loras is None:
            return {}

        if load_config.tp_size <= 1 and load_config.dp_size <= 1 and load_config.ep_size <= 1 :
            res = raw_loras
        else:
            res = self._split_lora(raw_loras, load_config)
        flat_res = {}
        def __extract_tensor(tensors):
            for k,v in tensors.items():
                if isinstance(v, dict):
                    __extract_tensor(v)
                elif v is not None:
                    flat_res.update({k: v.contiguous().clone().to(device)})
        __extract_tensor(res)
        return flat_res
        

    @abstractmethod
    def _load_raw_tensor(self, database: BaseDatabase, layer_id: Optional[int], device: str, load_config: LoadConfig):
        pass
    
    @abstractmethod
    def _split(self, tensor: torch.Tensor, load_config: LoadConfig):
        pass

    @abstractmethod
    def _postprocess(self, tensor: torch.Tensor, device: str, load_config: LoadConfig):
        return tensor

    @abstractmethod
    def _merge_lora(self, tensor: Dict[str, torch.Tensor], database: BaseDatabase, layer_id: Optional[int], load_config: LoadConfig):
        pass
                     
    @abstractmethod
    def _load_raw_lora(self, database: BaseDatabase, layer_id: Optional[int], device: str, load_config: LoadConfig, lora_name: str):
        pass

    @abstractmethod
    def _split_lora(self, tensor: Dict[str, torch.Tensor], load_config: LoadConfig):
        pass


class AtomicWeight(WeightModule):
    weights: List[CkptWeightInfo]
    process_fun: Callable[[List[torch.Tensor]], torch.Tensor]
    data_type: Optional[torch.dtype] = None
    split_func = None

    """原子权重（不可分割的单个权重）"""
    def __init__(
        self,
        name: str,
        weights: List[CkptWeightInfo],
        process_fun: Callable[[List[torch.Tensor]], torch.Tensor] = identity,
        data_type: Optional[torch.dtype] = None,
        **kwargs 
    ) -> None:
        self.name = name
        self.weights = weights
        self.process_fun = process_fun
        self.data_type = data_type
        super().__init__(name=name, **kwargs)

    def create_from(self, *args: Any, **kwargs: Any) -> 'AtomicWeight':
        return self.__class__(*args, **kwargs)

    @property
    def need_transpose(self) -> bool:
        if isinstance(self.process_fun, functools.partial) and self.process_fun.func.__name__ in ['transpose_pad', 'transpose']:
            return True
        else:
            return False
        
    def _load_raw_tensor(self, database: BaseDatabase, layer_id: Optional[int], device: str, load_config: LoadConfig):
        before_merge_tensors = []
        convert_type = self.data_type if self.data_type is not None else load_config.compute_dtype
        for ckpt_weight in self.weights:
            name = ckpt_weight.tensor_name(layer_id)
            try:
                before_merge_tensors.append(ckpt_weight.merge_fun([x.to(device) for x in database.load_tensor(name, convert_type)]))
            except Exception as e:
                logging.error(f"加载 {self.name}: {name} 失败，完整堆栈:\n{traceback.format_exc()}")
                raise e
        after_merge_tensor = self.process_fun(before_merge_tensors).to(device).to(convert_type)
        return {self.name: after_merge_tensor}

    def lora_tensor_name(self, layer_id: Optional[int], name: str):
        if layer_id is not None:
            return name.format(i=str(layer_id), i_1=str(layer_id + 1))
        return name


    def _load_raw_lora(self, database: BaseDatabase, layer_id: Optional[int], device: str, load_config: LoadConfig, lora_name: str):
        if self.lora_a_process_func is None or self.lora_b_process_func is None:
            return {}
        a_res = self._load_lora_a(database, layer_id, device, load_config, lora_name)
        b_res = self._load_lora_b(database, layer_id, device, load_config, lora_name)
        a_res.update(b_res)
        return a_res

    def _split_lora(self, tensor: Dict[str, torch.Tensor], load_config: LoadConfig) -> Dict[str, torch.Tensor]:
        if self.lora_a_split_func is None or self.lora_b_split_func is None or not tensor:
            return tensor
        lora_a_name: str = self.lora_a_name
        lora_b_name: str = self.lora_b_name
        return {
             lora_a_name: self.__split_tensor(self.lora_a_split_func, tensor.get(lora_a_name), load_config),
             lora_b_name: self.__split_tensor(self.lora_b_split_func, tensor.get(lora_b_name), load_config)
        }

    def __split_tensor(self, split_func: Callable, tensor: torch.Tensor, load_config: LoadConfig) -> torch:
        return split_func(t=tensor,
                        tp=load_config.tp_size,
                        tp_rank=load_config.tp_rank,
                        ep=load_config.ep_size,
                        ep_rank=load_config.ep_rank,
                        dp=load_config.dp_size,
                        dp_rank=load_config.dp_rank,
                        ffn_tp_rank=load_config.ffn_tp_rank,
                        ffn_tp_size=load_config.ffn_tp_size,
                        hidden_size=load_config.hidden_size,
                        head_num=load_config.head_num,
                        head_num_kv=load_config.head_num_kv,
                        size_per_head=load_config.size_per_head,
                        use_stack_weight=load_config.use_stack_weight,
                        bits=load_config.bit
        )
                        

    def _load_lora_a(self, database: BaseDatabase, layer_id: Optional[int], device: str, load_config: LoadConfig, lora_name: str):
        assert self.lora_a_process_func is not None
        before_merge_tensors = []
        for ckpt_weight in self.weights:
            ckpt_name = self.lora_base_name.format(ckpt_weight.name[:-len(".weight")], self.lora_A_suffix)
            tensor_name = self.lora_tensor_name(layer_id, ckpt_name)
            try: 
                before_merge_tensors.append(ckpt_weight.merge_fun([x for x in database.load_lora_tensor(lora_name, tensor_name)]))
            except:
                logging.warning(f"load {self.name} lora A failed: {tensor_name}, {traceback.format_exc()}")
                return {}
        after_merge_tensor = self.lora_a_process_func(before_merge_tensors)
        return {self.lora_a_name : after_merge_tensor}

    def _load_lora_b(self, database: BaseDatabase, layer_id: Optional[int], device: str, load_config: LoadConfig, lora_name: str):
        assert self.lora_b_process_func is not None
        before_merge_tensors = []
        for ckpt_weight in self.weights:
            ckpt_name = self.lora_base_name.format(ckpt_weight.name[:-len(".weight")], self.lora_B_suffix)
            tensor_name = self.lora_tensor_name(layer_id, ckpt_name)
            try: 
                before_merge_tensors.append(ckpt_weight.merge_fun([x for x in database.load_lora_tensor(lora_name, tensor_name)]))
            except:
                logging.warning(f"load {self.name} lora B failed: {tensor_name}, {traceback.format_exc()}")
                return {}
        after_merge_tensor = self.lora_b_process_func(before_merge_tensors)
        return {self.lora_b_name : after_merge_tensor}


    def _merge_lora(self, tensor: Union[torch.Tensor, Dict[str, torch.Tensor]], database: BaseDatabase, layer_id: Optional[int], load_config: LoadConfig, lora_name:Optional[str] = None):
        if self.lora_a_process_func is None or self.lora_b_process_func is None:
            return tensor
        lora_name = database.get_first_lora_name() if lora_name is None else lora_name
        assert lora_name is not None
        if lora_name is None:
            raise Exception(f"invalid empty lora name")
        
        try: 
            raw_loras = self._load_raw_lora(database, layer_id, device=load_config.exported_device, load_config=load_config, lora_name=lora_name)
            lora_a_tensor = raw_loras[self.lora_a_name]
            lora_b_tensor = raw_loras[self.lora_b_name]
        except Exception as e:
            logging.warning(f"load layer: {layer_id} lora tensor {self.lora_a} or {self.lora_b} failed: traceback: {traceback.format_exc()}")
            return tensor
        if lora_a_tensor is None or lora_b_tensor is None:
            return tensor

        raw_tensor = tensor if isinstance(tensor, torch.Tensor) else tensor[self.name]
                
        scale = database.get_lora_config(lora_name).get_scale()
        # "addmm_impl_cpu_" not implemented for 'Half'
        if lora_b_tensor.dim() == 3 and lora_a_tensor.dim() == 2:
            lora_b_tensor = lora_b_tensor.reshape(lora_b_tensor.shape[0], lora_b_tensor.shape[1] * lora_b_tensor.shape[2])
            merge_tensor = (lora_a_tensor.type(torch.float32) @ lora_b_tensor.type(torch.float32) * scale).type(raw_tensor.dtype).to(raw_tensor.device)
        # moe
        elif lora_b_tensor.dim() == 3 and lora_a_tensor.dim() == 3:
            merge_tensor = torch.bmm(lora_a_tensor.type(torch.float32), lora_b_tensor.type(torch.float32) * scale).type(raw_tensor.dtype).to(raw_tensor.device)
        else:
            merge_tensor = (lora_a_tensor.type(torch.float32) @ lora_b_tensor.type(torch.float32) * scale).type(raw_tensor.dtype).to(raw_tensor.device)

        
        shape = raw_tensor.shape
        raw_tensor = raw_tensor.reshape(raw_tensor.nelement()) + merge_tensor.reshape(raw_tensor.nelement())
        raw_tensor = raw_tensor.reshape(shape)

        del lora_a_tensor
        del lora_b_tensor
        return {self.name : raw_tensor}
    
        
    def _split(self, tensor: Union[torch.Tensor, Dict[str, torch.Tensor]], load_config: LoadConfig):
        raw_tensor = tensor if isinstance(tensor, torch.Tensor) else tensor[self.name]
        if load_config.tp_size <= 1 and load_config.dp_size <= 1 and load_config.ep_size <= 1 :
            return {self.name : raw_tensor}

        tp_split_emb_and_lm_head = load_config.tp_split_emb_and_lm_head

        if (not tp_split_emb_and_lm_head and
            self.name in [W.lm_head, W.lm_head_b, W.embedding, W.positional_embedding, W.token_type_embedding]):
            return {self.name : raw_tensor}

        split_func = self._get_split_func()

        ts = self.__split_tensor(split_func, raw_tensor, load_config).contiguous().clone()
        return {self.name: ts}

    def _postprocess(self, tensor: Union[torch.Tensor, Dict[str, torch.Tensor]], device:str, load_config: LoadConfig):
        raw_tensor = tensor.get(self.name) if isinstance(tensor, dict) else tensor
        return {self.name: load_config.exported_device.maybe_rewrite_weight_by_key(self.name, raw_tensor)}

    def _get_split_func(self):
        return W.gpt_style_tp_strategy[self.name]

    def get_components(self):
        return [self]
    @classmethod
    def support(cls, quant_algo: Any, src_weight_info: WeightModule) -> bool:
        return quant_algo is None or not quant_algo.isQuant()

    def get_ckpt_tensor_names(self) -> List[str]:
        if not bool(self.weights):
            return []
        return [ckpt.name for ckpt in self.weights]

    def __str__(self) -> str:
        return f"AtomicWeight[{self.name}]-{self.weight_style}-{self.weights}"

    def __repr__(self) -> str:
        return self.__str__()


class QuantWeight(WeightModule):
    def __init__(self, name: str, quant_algo, *args, **kwargs):
        super().__init__(name)
        self.quant_algo = quant_algo

class MMAtomicWeight(AtomicWeight):
    def __init__(
        self,
        name: str,
        weights: List[CkptWeightInfo],
        process_fun: Callable[[List[torch.Tensor]], torch.Tensor] = identity,
        data_type: Optional[torch.dtype] = None,
        split_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        **kwargs 
    ) -> None:
        super().__init__(name, weights, process_fun, data_type, **kwargs)
        self.split_func = split_func

    def _get_split_func(self):
        return self.split_func

class CompositeWeight(WeightModule):
    """复合权重组件（如MoE、FFN）"""
    def __init__(self, sub_weights: Dict[str, WeightModule], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sub_weights = self._init_sub_weights(sub_weights) if isinstance(sub_weights, list) else sub_weights

    def get_components(self):
        res = []
        for sub_weight in self.sub_weights.values():
            res.extend(sub_weight.get_components())
        return res

    def _init_sub_weights(self, sub_weights: List[WeightModule]):
        inited_sub_weights = {}
        for sub_weight in sub_weights:
            inited_sub_weights.update({sub_weight.name: sub_weight})
        return inited_sub_weights


    def __str__(self) -> str:
        return f"{self.__class__}[{self.name}]{self.sub_weights}"

    def __repr__(self) -> str:
        return self.__str__()


    def _load_raw_tensor(self, database: BaseDatabase, layer_id: Optional[int], device: str, load_config: LoadConfig):
        raw_tensors = {}
        for name, sub_weight in self.sub_weights.items():
            sub_tensors = sub_weight._load_raw_tensor(database, layer_id, device, load_config)
            if isinstance(sub_weight, AtomicWeight) and isinstance(sub_tensors, dict):
                raw_tensors.update(sub_tensors)
            else:
                raw_tensors.update({name:sub_tensors})
        return raw_tensors

    def _merge_lora(self, tensor: Union[torch.Tensor, Dict[str, torch.Tensor]], database: BaseDatabase, layer_id: Optional[int], load_config: LoadConfig):
        merged_tensors = {}
        for name, sub_weight in self.sub_weights.items():
            sub_tensors = tensor.get(name)
            sub_tensors = sub_weight._merge_lora(sub_tensors, database, layer_id, load_config)
            if isinstance(sub_weight, AtomicWeight) and isinstance(sub_tensors, dict):
                merged_tensors.update(sub_tensors)
            else:
                merged_tensors.update({name:sub_tensors})
        return merged_tensors

    def _load_raw_lora(self, database: BaseDatabase, layer_id: Optional[int], device: str, load_config: LoadConfig, lora_name: str):
        raw_tensors = {}
        for name, sub_weight in self.sub_weights.items():
            sub_tensors = sub_weight._load_raw_lora(database, layer_id, device, load_config, lora_name=lora_name)
            raw_tensors.update({name:sub_tensors})
        return raw_tensors

    def _split_lora(self, tensor: Union[torch.Tensor, Dict[str, torch.Tensor]], load_config: LoadConfig):
        split_tensors = {}
        for name, sub_weight in self.sub_weights.items():
            sub_tensors = tensor.get(name)
            sub_tensors = sub_weight._split_lora(sub_tensors, load_config)
            split_tensors.update({name:sub_tensors})
        return split_tensors

    def _split(self, tensor: Union[torch.Tensor, Dict[str, torch.Tensor]], load_config: LoadConfig):
        split_tensors = {}
        for name, sub_weight in self.sub_weights.items():
            sub_tensors = tensor.get(name)
            sub_tensors = sub_weight._split(sub_tensors, load_config)
            if isinstance(sub_weight, AtomicWeight) and isinstance(sub_tensors, dict):
                split_tensors.update(sub_tensors)
            else:
                split_tensors.update({name:sub_tensors})
        return split_tensors

    def _postprocess(self, tensor: Union[torch.Tensor, Dict[str, torch.Tensor]], device: str, load_config: LoadConfig) -> torch.Tensor:
        processed_tensors = {}
        for name, sub_weight in self.sub_weights.items():
            sub_tensors = tensor.get(name)
            sub_tensors = sub_weight._postprocess(sub_tensors, device, load_config)
            if isinstance(sub_weight, AtomicWeight) and isinstance(sub_tensors, dict):
                processed_tensors.update(sub_tensors)
            else:
                processed_tensors.update({name:sub_tensors})
        return processed_tensors

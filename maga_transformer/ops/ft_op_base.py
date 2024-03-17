import torch
from abc import abstractmethod
from typing import Any, Optional, Dict, List, Callable, Union

from maga_transformer.utils.model_weight import LoraResource

class FTWeightsBase:
    def __init__(self):
        self.weights: Union[Dict[str, torch.Tensor], List[torch.Tensor]] = []
        self.lora_resource: LoraResource = LoraResource()

    @abstractmethod
    def load(self, *args: List[Any], **kwargs: Any) -> bool:
        raise NotImplementedError

    @property
    def dtype(self):
        if isinstance(self.weights, dict):
            return list(self.weights.values())[0].dtype
        return self.weights[0].dtype

    @property
    def device(self):
        if isinstance(self.weights, dict):
            return list(self.weights.values())[0].device
        return self.weights[0].device

    def _map(self, func: Callable[[torch.Tensor], torch.Tensor]):
        if isinstance(self.weights, dict):
            raise Exception("weight based on map not support _map yet!")
        for i in range(len(self.weights)):
            if isinstance(self.weights[i], list):
                for j in range(len(self.weights[i])):
                    self.weights[i][j] = func(self.weights[i][j])
            else:
                self.weights[i] = func(self.weights[i])

    def float(self):
        if self.dtype == torch.float32:
            return
        self._map(lambda x: x.float())

    def half(self):
        if self.dtype == torch.float16:
            return
        self._map(lambda x: x.half())

    def bfloat16(self):
        if self.dtype == torch.bfloat16:
            return
        self._map(lambda x: x.bfloat16())

    def cuda(self, device: Optional[str]=None):
        self._map(lambda x: x.cuda(device))

    def to(self, device: Optional[str]=None):
        self._map(lambda x: x.to(device))

class FTOPBase:
    def __init__(self):
        self.weight: Optional[Any] = None
        self.ft_op: Optional[Any] = None

    @classmethod
    def from_config(cls, config: Any) -> Any:
        return cls(config)

    @abstractmethod
    def _initialize_op(self, force_init: bool=False) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args: List[Any], **kwargs: Any) -> Any:
        raise NotImplementedError

    def set_weight(self, weight: FTWeightsBase):
        old_weight_dtype = self.weight.dtype if self.weight is not None else None
        self.weight = weight
        if old_weight_dtype is None or old_weight_dtype != self.weight.dtype:
            self._initialize_op(force_init=True)

    @property
    def dtype(self):
        assert self.weight is not None
        return self.weight.dtype

    @property
    def device(self):
        assert self.weight is not None
        return self.weight.device

    def update_lora(self):
        if self.weight != None:
            for id in self.weight.lora_resource.to_remove_lora_id:
                self.ft_op.remove_lora(id)
            for id in self.weight.lora_resource.to_add_lora_id:
                lora_weight = self.weight.lora_resource.lora_map.weights_map[id]
                self.ft_op.add_lora(id, lora_weight.lora_a_weights, lora_weight.lora_b_weights)
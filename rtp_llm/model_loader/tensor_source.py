from typing import Any, Dict, Generator, List, Optional
from rtp_llm.utils.database import BaseDatabase

import torch

class TensorSource:
    def load_tensor(
        self, name: str, data_type: Optional[torch.dtype] = torch.float16
    ) -> List[torch.Tensor]:
        raise NotImplementedError

    def get_database(self) -> BaseDatabase:
        raise NotImplementedError


class DatabaseTensorSource(TensorSource):
    _database: BaseDatabase

    def __init__(self, database: BaseDatabase):
        self._database = database
    
    def load_tensor(self, name, data_type = torch.float16):
        return self._database.load_tensor(name, data_type)

    def get_database(self) -> BaseDatabase:
        return self._database


class TensorCollector(TensorSource):
    _target_keys: List[str]
    _tensors: Dict[str, torch.Tensor]
    _completed_once: bool
    _database: BaseDatabase

    def __init__(self, target_keys: List[str], database: BaseDatabase):
        self._target_keys = target_keys
        self._tensors = {}
        self._completed_once = False
        self._database = database
    
    def load_tensor(self, name, data_type = torch.float16):
        tensors = []
        t = self._tensors.get(name)
        if t is not None:
            tensors.append(self._tensors[name].to(data_type))
        return tensors

    def store_tensor(self, name: str, tensor: torch.Tensor) -> bool:
        if name not in self._target_keys:
            raise ValueError(f"Tensor name '{name}' not in target list.")
        self._tensors[name] = tensor
        self._check_completion()
        return self.is_collection_complete()
    
    def _check_completion(self):
        if self._target_keys.issubset(self._tensors.keys()):
            self._completed_once = True

    def clear(self):
        self._tensors.clear()

    def is_collection_complete(self) -> bool:
        return self._completed_once
    
    def get_database(self) -> BaseDatabase:
        return self._database

from typing import Callable, Dict, List, Optional, Set, Tuple

import torch

from rtp_llm.utils.database import BaseDatabase


class TensorSource:
    def load_tensor(
        self, name: str, data_type: Optional[torch.dtype] = torch.float16
    ) -> List[torch.Tensor]:
        raise NotImplementedError

    def has_tensor(self, name: str) -> bool:
        raise NotImplementedError

    def get_database(self) -> BaseDatabase:
        raise NotImplementedError

    def stacked_expert_index(self, name: str, expert_id: int) -> int:
        return expert_id


class StackSplitTensorSource(TensorSource):
    """Wraps a TensorSource to transparently split stacked MoE tensors by expert.

    When load_tensor is called with a per-expert logical key that maps to a stacked
    checkpoint key, the stacked tensor is loaded once, split by expert dimension,
    and the requested expert slice is returned. Non-mapped keys pass through to the
    base source.
    """

    def __init__(
        self,
        base: TensorSource,
        split_config: Dict[str, Tuple[str, int, Callable]],
    ):
        self._base = base
        self._split_config = split_config
        # Lifetime: each MoeAtomicWeight._load_raw_tensor call creates a fresh
        # StackSplitTensorSource, so the cache is GC'd after that method returns.
        self._stacked_cache: Dict[str, torch.Tensor] = {}

    def load_tensor(
        self, name: str, data_type: torch.dtype = torch.float16
    ) -> List[torch.Tensor]:
        config = self._split_config.get(name)
        if config is not None:
            stacked_key, expert_id, merge_fun = config
            if stacked_key not in self._stacked_cache:
                raw = self._base.load_tensor(stacked_key, data_type)
                self._stacked_cache[stacked_key] = merge_fun(raw)
            index = self._base.stacked_expert_index(stacked_key, expert_id)
            return [self._stacked_cache[stacked_key][index]]
        return self._base.load_tensor(name, data_type)

    def has_tensor(self, name: str) -> bool:
        config = self._split_config.get(name)
        if config is not None:
            return self._base.has_tensor(config[0])
        return self._base.has_tensor(name)

    def get_database(self):
        return self._base.get_database()


class DatabaseTensorSource(TensorSource):
    _database: BaseDatabase

    def __init__(self, database: BaseDatabase):
        self._database = database

    def load_tensor(self, name, data_type=torch.float16):
        return self._database.load_tensor(name, data_type)

    def has_tensor(self, name: str) -> bool:
        return self._database.has_tensor(name)

    def get_database(self) -> BaseDatabase:
        return self._database


class TensorCollector(TensorSource):
    _target_keys: Set[str]
    _tensors: Dict[str, torch.Tensor]
    _scales: Dict[str, torch.Tensor]
    _completed_once: bool
    _database: BaseDatabase

    def __init__(self, target_keys: Set[str], database: BaseDatabase, *, stacked_experts=None):
        self._target_keys = target_keys
        self._tensors = {}
        self._scales = {}
        self._completed_once = False
        self._database = database
        self._stacked_experts = {
            name: selected
            for name, selected in (stacked_experts or {}).items()
            if name in target_keys
        }

    def stacked_expert_index(self, name: str, expert_id: int) -> int:
        selected = self._stacked_experts.get(name)
        if selected is None:
            return expert_id
        if expert_id not in selected:
            raise ValueError(f"Expert {expert_id} is outside the received packed shard for {name}")
        return expert_id - selected.start

    def load_tensor(self, name, data_type=torch.float16):
        tensors = []
        t = self._tensors.get(name)
        if t is not None:
            if name in self._scales:
                tensors.append(t)
            else:
                tensors.append(t.to(data_type))
        return tensors

    def has_tensor(self, name: str) -> bool:
        return name in self._tensors

    def store_tensor(self, name: str, tensor: torch.Tensor) -> bool:
        if name not in self._target_keys:
            raise ValueError(f"Tensor name '{name}' not in target list.")
        self._tensors[name] = tensor
        self._check_completion()
        return self.is_collection_complete()

    def store_fp8_quantized(
        self, name: str, fp8_tensor: torch.Tensor, scale: torch.Tensor
    ) -> bool:
        if name not in self._target_keys:
            raise ValueError(f"Tensor name '{name}' not in target list.")
        self._tensors[name] = fp8_tensor
        self._scales[name] = scale
        self._check_completion()
        return self.is_collection_complete()

    def has_prequantized_scale(self, name: str) -> bool:
        return name in self._scales

    def get_scale(self, name: str) -> torch.Tensor:
        return self._scales[name]

    def _check_completion(self):
        if self._target_keys.issubset(self._tensors.keys()):
            self._completed_once = True

    def clear(self):
        self._tensors.clear()
        self._scales.clear()

    def is_collection_complete(self) -> bool:
        return self._completed_once

    def get_database(self) -> BaseDatabase:
        return self._database

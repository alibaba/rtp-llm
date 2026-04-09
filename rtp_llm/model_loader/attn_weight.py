from copy import copy
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from pydantic import BaseModel
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.utils.model_weight import CkptWeightInfo, identity, W


class AttnConfig(BaseModel):
    hidden_size: int = -1
    size_per_head: int = -1
    head_num: int = -1
    head_num_kv: int = -1
    need_post_ln: bool = False


class AttnAtomicWeight(AtomicWeight):
    def __init__(
        self,
        name: str,
        weights: List[CkptWeightInfo],
        process_fun: Callable[[List[torch.Tensor]], torch.Tensor] = identity,
        data_type: Optional[torch.dtype] = None,
        config: Optional[AttnConfig] = None,
        *args,
        **kwargs
    ):
        self.config = config
        super().__init__(name, weights, process_fun, data_type, *args, **kwargs)

    def _split(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        load_config: LoadConfig,
    ):
        """Override to use per-layer attention config for TP splitting.

        Models with heterogeneous attention (e.g. Gemma4 with sliding vs global
        layers that have different head_num / kv_head_num) need the TP split
        function to see the correct per-layer head counts, not the model-global
        defaults stored in load_config.
        """
        if self.config is not None:
            # Check if the per-layer config has different head counts
            cfg_head_num = getattr(self.config, 'head_num', -1)
            cfg_kv_head_num = getattr(self.config, 'kv_head_num',
                                       getattr(self.config, 'head_num_kv', -1))
            cfg_size_per_head = getattr(self.config, 'size_per_head', -1)

            needs_override = (
                (cfg_head_num > 0 and cfg_head_num != load_config.head_num) or
                (cfg_kv_head_num > 0 and cfg_kv_head_num != load_config.head_num_kv) or
                (cfg_size_per_head > 0 and cfg_size_per_head != load_config.size_per_head)
            )
            if needs_override:
                load_config = copy(load_config)
                if cfg_head_num > 0:
                    load_config.head_num = cfg_head_num
                if cfg_kv_head_num > 0:
                    load_config.head_num_kv = cfg_kv_head_num
                if cfg_size_per_head > 0:
                    load_config.size_per_head = cfg_size_per_head

        return super()._split(tensor, load_config)


class MlaConfig(BaseModel):
    head_num: int = -1
    nope_head_dim: int = -1
    rope_head_dim: int = -1
    kv_lora_rank: int = -1
    v_head_dim: int = -1
    use_mla: bool = False
    q_use_lora: bool = False


class MlaAttnAtomicWeight(AtomicWeight):
    def __init__(
        self,
        name: str,
        weights: List[CkptWeightInfo],
        process_fun: Callable[[List[torch.Tensor]], torch.Tensor] = identity,
        data_type: Optional[torch.dtype] = None,
        config: Optional[MlaConfig] = None,
        *args: Any,
        **kwargs: Any
    ):
        self.config = config
        super().__init__(name, weights, process_fun, data_type, *args, **kwargs)

    @property
    def head_num(self) -> int:
        return self.config.head_num

    @property
    def nope_head_dim(self) -> int:
        return self.config.nope_head_dim

    @property
    def rope_head_dim(self) -> int:
        return self.config.rope_head_dim

    @property
    def kv_lora_rank(self) -> int:
        return self.config.kv_lora_rank

    @property
    def v_head_dim(self) -> int:
        return self.config.v_head_dim

    @property
    def use_mla(self) -> bool:
        return self.config.use_mla

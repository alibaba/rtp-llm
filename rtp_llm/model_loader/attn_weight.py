from typing import Any, Callable, List, Optional, Union, Dict

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
    use_fp8_kv_cache: bool = False
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

    def _swizzle_gemm_weight(
        self,
        name: str,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        load_config: LoadConfig,
    ):
        if name not in (W.attn_qkv_w, W.attn_o_w):
            raise ValueError(f"unsupported swizzle name: {name}")
        if isinstance(tensor, dict):
            w = tensor.get(name)
            if isinstance(w, torch.Tensor):
                w = load_config.exported_device.swizzle_gemm_weight(w, w.dtype != torch.float8_e4m3fn)
                tensor[name] = w
            elif isinstance(w, dict):
                self._swizzle_gemm_weight(name, w, load_config)
            else:
                raise TypeError(f"unsupported type at key {name}: {type(w)}")

        elif isinstance(tensor, torch.Tensor):
            swizzled = load_config.exported_device.swizzle_gemm_weight(tensor, tensor.dtype != torch.float8_e4m3fn)
            return swizzled

    def _postprocess(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        device: str,
        load_config: LoadConfig
    ):
        if load_config.use_swizzleA:
            if isinstance(tensor, torch.Tensor):
                if getattr(self, "name", None) in (W.attn_qkv_w, W.attn_o_w):
                    tensor = self._swizzle_gemm_weight(self.name, tensor, load_config)
                return super()._postprocess(tensor, device, load_config)

            for key in (W.attn_qkv_w, W.attn_o_w):
                w = tensor.get(key)
                if isinstance(w, dict):
                    self._swizzle_gemm_weight(key, w, load_config)
                elif isinstance(w, torch.Tensor):
                    self._swizzle_gemm_weight(key, tensor, load_config)

        return super()._postprocess(tensor, device, load_config)

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

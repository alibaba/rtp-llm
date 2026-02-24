import functools
import logging
import traceback
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from pydantic import BaseModel

from rtp_llm.config.quant_config import QuantizationConfig
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CompositeWeight,
    QuantWeight,
    WeightModule,
)
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity
from rtp_llm.utils.util import check_with_info


class FfnConfig(BaseModel):
    is_gated_activation: bool = False
    # align_size is used for dynamic padding calculation
    align_size: int = 0  # 0 means no padding needed
    is_moe: bool = False
    need_post_ln: bool = False
    need_ffn_act_scale: bool = False


class FfnAtomicWeight(AtomicWeight):
    def __init__(
        self,
        name: str,
        weights: List[CkptWeightInfo],
        process_fun: Callable[[List[torch.Tensor]], torch.Tensor] = identity,
        data_type: Optional[torch.dtype] = None,
        config: FfnConfig = None,
        *args: Any,
        **kwargs: Any,
    ):
        self.config = config
        super().__init__(name, weights, process_fun, data_type, *args, **kwargs)

    @property
    def need_padding(self) -> bool:
        if isinstance(
            self.process_fun, functools.partial
        ) and self.process_fun.func.__name__ in ["transpose_pad", "pad"]:
            return True
        else:
            return False

    @property
    def pad_dim(self) -> Optional[int]:
        if not self.need_padding:
            return None
        return self.process_fun.keywords["dim"]


def w13_func_wrap(ts: List[torch.Tensor], origin_w1, origin_w3):
    w1_size = len(origin_w1.weights)
    w3_size = len(origin_w3.weights)
    assert len(ts) == w1_size + w3_size
    w1 = origin_w1.process_fun(ts[:w1_size])
    w3 = origin_w3.process_fun(ts[w1_size:])
    return torch.concat([w1, w3], dim=-1).contiguous()


def w13_lora_a_func_wrap(
    ts: torch.Tensor, origin_w1: FfnAtomicWeight, origin_w3: FfnAtomicWeight
):
    assert origin_w1.lora_a_process_func and origin_w3.lora_a_process_func
    w1, w3 = (
        torch.chunk(ts, 2, dim=-1) if isinstance(ts, torch.Tensor) else (ts[0], ts[1])
    )
    w1 = origin_w1.lora_a_process_func([w1])
    w3 = origin_w3.lora_a_process_func([w3])
    return torch.concat([w1, w3], dim=-1).contiguous()


def w13_lora_b_func_wrap(
    ts: torch.Tensor, origin_w1: FfnAtomicWeight, origin_w3: FfnAtomicWeight
):
    assert origin_w1.lora_b_process_func and origin_w3.lora_b_process_func
    w1, w3 = (
        torch.chunk(ts, 2, dim=-1) if isinstance(ts, torch.Tensor) else (ts[0], ts[1])
    )
    w1 = origin_w1.lora_b_process_func([w1])
    w3 = origin_w3.lora_b_process_func([w3])

    # For LoRA B, we need to create a block diagonal matrix to ensure proper matrix multiplication
    # The result should be [[w1, 0], [0, w3]] to match the LoRA A processing

    # Get dimensions
    w1_rows, w1_cols = w1.shape
    w3_rows, w3_cols = w3.shape

    # Create zero padding matrices
    zeros_top_right = torch.zeros(w1_rows, w3_cols, dtype=w1.dtype, device=w1.device)
    zeros_bottom_left = torch.zeros(w3_rows, w1_cols, dtype=w3.dtype, device=w3.device)

    # Create the block diagonal matrix [[w1, 0], [0, w3]]
    top_row = torch.cat([w1, zeros_top_right], dim=1)
    bottom_row = torch.cat([zeros_bottom_left, w3], dim=1)
    res = torch.cat([top_row, bottom_row], dim=0)

    return res


def w13_lora_a_split_func_wrap(
    ts: torch.Tensor, origin_w1: FfnAtomicWeight, origin_w3: FfnAtomicWeight
):
    assert origin_w1.lora_a_split_func and origin_w3.lora_a_split_func
    w1, w3 = torch.chunk(ts, 2, dim=-1)
    w1 = origin_w1.lora_a_split_func(w1)
    w3 = origin_w3.lora_a_split_func(w3)
    return torch.concat([w1, w3], dim=-1).contiguous()


def w13_lora_b_split_func_wrap(
    ts: torch.Tensor, origin_w1: FfnAtomicWeight, origin_w3: FfnAtomicWeight
):
    assert origin_w1.lora_b_split_func and origin_w3.lora_b_split_func
    w1, w3 = torch.chunk(ts, 2, dim=-1)
    w1 = origin_w1.lora_b_split_func(w1)
    w3 = origin_w3.lora_b_split_func(w3)

    # For LoRA B split function, we need to create a block diagonal matrix to ensure proper matrix multiplication
    # The result should be [[w1, 0], [0, w3]] to match the LoRA A processing

    # Get dimensions
    w1_rows, w1_cols = w1.shape
    w3_rows, w3_cols = w3.shape

    # Create zero padding matrices
    zeros_top_right = torch.zeros(w1_rows, w3_cols, dtype=w1.dtype, device=w1.device)
    zeros_bottom_left = torch.zeros(w3_rows, w1_cols, dtype=w3.dtype, device=w3.device)

    # Create the block diagonal matrix [[w1, 0], [0, w3]]
    top_row = torch.cat([w1, zeros_top_right], dim=1)
    bottom_row = torch.cat([zeros_bottom_left, w3], dim=1)
    res = torch.cat([top_row, bottom_row], dim=0)

    return res


def fix_merge_w13(sub_weight_dict: Dict[str, FfnAtomicWeight]):
    origin_w1 = sub_weight_dict[W.ffn_w1]
    origin_w3 = sub_weight_dict[W.ffn_w3]
    w_list = origin_w1.weights + origin_w3.weights
    lora_a_process_func = (
        functools.partial(
            w13_lora_a_func_wrap, origin_w1=origin_w1, origin_w3=origin_w3
        )
        if origin_w1.lora_a_process_func
        else None
    )
    lora_b_process_func = (
        functools.partial(
            w13_lora_b_func_wrap, origin_w1=origin_w1, origin_w3=origin_w3
        )
        if origin_w1.lora_b_process_func
        else None
    )
    lora_a_split_func = (
        functools.partial(
            w13_lora_a_split_func_wrap, origin_w1=origin_w1, origin_w3=origin_w3
        )
        if origin_w1.lora_a_split_func
        else None
    )
    lora_b_split_func = (
        functools.partial(
            w13_lora_b_split_func_wrap, origin_w1=origin_w1, origin_w3=origin_w3
        )
        if origin_w1.lora_b_split_func
        else None
    )
    w13 = FfnAtomicWeight(
        name=W.ffn_w13,
        weights=w_list,
        process_fun=functools.partial(
            w13_func_wrap, origin_w1=origin_w1, origin_w3=origin_w3
        ),
        lora_a_process_func=lora_a_process_func,
        lora_b_process_func=lora_b_process_func,
        lora_a_split_func=lora_a_split_func,
        lora_b_split_func=lora_b_split_func,
        data_type=origin_w1.data_type,
        config=origin_w1.config,
    )

    sub_weight_dict.pop(W.ffn_w1)
    sub_weight_dict.pop(W.ffn_w3)
    sub_weight_dict[W.ffn_w13] = w13
    return sub_weight_dict


def fix_merge_b13(sub_weight_dict: Dict[str, FfnAtomicWeight]):
    origin_b1 = sub_weight_dict[W.ffn_b1]
    origin_b3 = sub_weight_dict[W.ffn_b3]
    w_list = origin_b1.weights + origin_b3.weights
    lora_a_process_func = (
        functools.partial(
            w13_lora_a_func_wrap, origin_w1=origin_b1, origin_w3=origin_b3
        )
        if origin_b1.lora_a_process_func
        else None
    )
    lora_b_process_func = (
        functools.partial(
            w13_lora_b_func_wrap, origin_w1=origin_b1, origin_w3=origin_b3
        )
        if origin_b1.lora_b_process_func
        else None
    )
    lora_a_split_func = (
        functools.partial(
            w13_lora_a_split_func_wrap, origin_w1=origin_b1, origin_w3=origin_b3
        )
        if origin_b1.lora_a_split_func
        else None
    )
    lora_b_split_func = (
        functools.partial(
            w13_lora_b_split_func_wrap, origin_w1=origin_b1, origin_w3=origin_b3
        )
        if origin_b1.lora_b_split_func
        else None
    )

    b13 = FfnAtomicWeight(
        name=W.ffn_w13,
        weights=w_list,
        process_fun=functools.partial(
            FfnWeight.__w13_func_wrap, origin_w1=origin_b1, origin_w3=origin_b3
        ),
        lora_a_process_func=lora_a_process_func,
        lora_b_process_func=lora_b_process_func,
        lora_a_split_func=lora_a_split_func,
        lora_b_split_func=lora_b_split_func,
        data_type=origin_b1.data_type,
        config=origin_b1.config,
    )

    sub_weight_dict.pop(W.ffn_b1)
    sub_weight_dict.pop(W.ffn_b3)
    sub_weight_dict[W.ffn_b13] = b13
    return sub_weight_dict


class FfnWeight(CompositeWeight):

    def __init__(
        self,
        sub_weights: Union[
            Dict[str, FfnAtomicWeight], List[Union[FfnAtomicWeight, AtomicWeight]]
        ],
        config: FfnConfig,
        *args: Any,
        **kwargs: Any,
    ):
        self.name = W.ffn
        sub_weight_dict = {sub_weight.name: sub_weight for sub_weight in sub_weights}
        self.config = config
        if W.ffn_w1 in sub_weight_dict and W.ffn_w3 in sub_weight_dict:
            self.origin_w1 = sub_weight_dict[W.ffn_w1]
            self.origin_w3 = sub_weight_dict[W.ffn_w3]
            sub_weight_dict = fix_merge_w13(sub_weight_dict)
        if W.ffn_b1 in sub_weight_dict and W.ffn_b3 in sub_weight_dict:
            self.origin_b1 = sub_weight_dict[W.ffn_b1]
            self.origin_b3 = sub_weight_dict[W.ffn_b3]
            sub_weight_dict = fix_merge_b13(sub_weight_dict)

        kwargs["name"] = W.ffn

        super().__init__(sub_weight_dict, *args, **kwargs)

        self.w1 = self.sub_weights.get(W.ffn_w1)
        self.w2 = self.sub_weights.get(W.ffn_w2)
        self.w3 = self.sub_weights.get(W.ffn_w3)
        self.w13 = self.sub_weights.get(W.ffn_w13)
        self.b1 = self.sub_weights.get(W.ffn_b1)
        self.b2 = self.sub_weights.get(W.ffn_b2)
        self.b3 = self.sub_weights.get(W.ffn_b3)
        self.b13 = self.sub_weights.get(W.ffn_b13)

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        return False

    @torch.inference_mode()
    def update(
        self, tensor: torch.Tensor, device: str, load_config: LoadConfig, **kwargs
    ):
        if "module_name" in kwargs:
            name: str = kwargs["module_name"]
            if name not in self.sub_weights:
                raise KeyError(
                    f"can not find key: {name} in ffn weights, allow key names are {[name for name in self.sub_weights]}"
                )
            return self.sub_weights[name].update(tensor, device, load_config)
        else:
            return super().update(tensor, device, load_config)

    def _split(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        load_config: LoadConfig,
    ):
        if (
            load_config.tp_size <= 1
            and load_config.dp_size <= 1
            and load_config.ep_size <= 1
        ):
            if self.name not in [W.moe_w1, W.moe_w2]:
                return tensor
        return super()._split(tensor, load_config)


class MoeConfig(BaseModel):
    is_moe: bool = True
    expert_num: int = -1
    # align_size is used for dynamic padding calculation
    align_size: int = 0  # 0 means no padding needed (for MoE)
    weight_stack: bool = False


class MoeAtomicWeight(AtomicWeight):
    def __init__(
        self,
        name: str,
        weights: List[CkptWeightInfo],
        process_fun: Callable[[List[torch.Tensor]], torch.Tensor] = identity,
        data_type: Optional[torch.dtype] = None,
        config: MoeConfig = None,
        *args: Any,
        **kwargs: Any,
    ):
        self.config = config
        super().__init__(name, weights, process_fun, data_type, *args, **kwargs)

    def _load_raw_tensor(
        self,
        tensor_source: TensorSource,
        layer_id: Optional[int],
        device: str,
        load_config: LoadConfig,
    ):
        if self.config.weight_stack:
            return super()._load_raw_tensor(
                tensor_source, layer_id, device, load_config
            )

        # weight should be expand by experts
        before_merge_tensors = []
        convert_type = (
            self.data_type if self.data_type is not None else load_config.compute_dtype
        )
        for ckpt_weight in self.weights:
            selected_experts = load_config.get_selected_experts(
                layer_id, self.config.expert_num
            )
            for expert_id in selected_experts:
                name = ckpt_weight.name.format(
                    i=str(layer_id), i_1=str(layer_id + 1), expert_id=str(expert_id)
                )
                logging.debug("tensor name: %s", name)
                try:
                    before_merge_tensors.append(
                        ckpt_weight.merge_fun(
                            [
                                x.to(device)
                                for x in tensor_source.load_tensor(name, convert_type)
                            ]
                        )
                    )
                except Exception as e:
                    logging.error(
                        f"加载 {name} 失败，完整堆栈:\n{traceback.format_exc()}"
                    )
                    raise e

        after_merge_tensor = self.process_fun(before_merge_tensors).to(convert_type)
        logging.debug("load weight :%s, %s ", self.name, after_merge_tensor.shape)
        return {self.name: after_merge_tensor}

    def get_tensor_names(
        self, layer_id: Optional[int], load_config: LoadConfig
    ) -> set[str]:
        if self.config.weight_stack:
            return super().get_tensor_names(layer_id, load_config)
        names = set[str]()
        for ckpt_weight in self.weights:
            selected_experts = load_config.get_selected_experts(
                layer_id, self.config.expert_num
            )
            for expert_id in selected_experts:
                name = ckpt_weight.name.format(
                    i=str(layer_id), i_1=str(layer_id + 1), expert_id=str(expert_id)
                )
                names.add(name)
        return names


class MoeWeight(CompositeWeight):
    def __init__(
        self, sub_weights: List[MoeAtomicWeight], config: MoeConfig, **kwargs: Any
    ):
        self.config = config
        # check all is MoeAtomicWeight
        assert all(
            isinstance(sub_weight, MoeAtomicWeight)
            or isinstance(sub_weight, QuantWeight)
            for sub_weight in sub_weights
        )
        kwargs["name"] = W.moe
        super().__init__(sub_weights, **kwargs)

        self.moe_w1 = self.sub_weights[W.moe_w1]
        self.moe_w2 = self.sub_weights[W.moe_w2]
        self.moe_gate = self.sub_weights[W.moe_gate]

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        return False

    def _shuff_moe_weight(
        self,
        name: str,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        load_config: LoadConfig,
    ):
        w = tensor.get(name)
        if isinstance(w, torch.Tensor):
            w = load_config.exported_device.shuffle_moe_weight(
                w, load_config.compute_dtype, name
            )
            tensor[name] = w
        elif isinstance(w, dict):
            self._shuff_moe_weight(name, w, load_config)
        else:
            raise ValueError("unsupported type")

    def _postprocess(
        self, tensor: Dict[str, torch.Tensor], device: str, load_config: LoadConfig
    ):
        moe_w1 = tensor.get(W.moe_w1)
        moe_w2 = tensor.get(W.moe_w2)
        for weight, keys in [
            (moe_w1, [W.moe_w1, W.moe_s1]),
            (moe_w2, [W.moe_w2, W.moe_s2]),
        ]:
            if isinstance(weight, dict):
                for key in keys:
                    if key in weight:
                        self._shuff_moe_weight(key, weight, load_config)
            else:
                self._shuff_moe_weight(keys[0], tensor, load_config)
        return super()._postprocess(tensor, device, load_config)


class SharedMoeConfig(FfnConfig, MoeConfig):
    pass


class MoeWithSharedWeight(CompositeWeight):
    def __init__(
        self,
        sub_weights: List[Union[FfnAtomicWeight, MoeAtomicWeight]],
        config: SharedMoeConfig,
        **kwargs: Any,
    ):
        self.config = config
        check_with_info(
            all(
                isinstance(sub_weight, MoeAtomicWeight)
                or isinstance(sub_weight, FfnAtomicWeight)
                or isinstance(sub_weight, QuantWeight)
                for sub_weight in sub_weights
            ),
            f"{[sub_weight.__class__ for sub_weight in sub_weights]}, {sub_weights}",
        )
        kwargs["name"] = W.moe
        sub_weight_dict = {sub_weight.name: sub_weight for sub_weight in sub_weights}

        if W.ffn_w1 in sub_weight_dict and W.ffn_w3 in sub_weight_dict:
            self.origin_w1 = sub_weight_dict[W.ffn_w1]
            self.origin_w3 = sub_weight_dict[W.ffn_w3]
            sub_weight_dict = fix_merge_w13(sub_weight_dict)
        if W.ffn_b1 in sub_weight_dict and W.ffn_b3 in sub_weight_dict:
            self.origin_b1 = sub_weight_dict[W.ffn_b1]
            self.origin_b3 = sub_weight_dict[W.ffn_b3]
            sub_weight_dict = fix_merge_b13(sub_weight_dict)

        super().__init__(sub_weight_dict, **kwargs)

        self.moe_w1 = self.sub_weights.get(W.moe_w1)
        self.moe_w2 = self.sub_weights.get(W.moe_w2)
        self.moe_gate = self.sub_weights.get(W.moe_gate)
        self.ffn_w1 = self.sub_weights.get(W.ffn_w1)
        self.ffn_w2 = self.sub_weights.get(W.ffn_w2)
        self.ffn_w3 = self.sub_weights.get(W.ffn_w3)
        self.w13 = self.sub_weights.get(W.ffn_w13)
        self.ffn_b1 = self.sub_weights.get(W.ffn_b1)
        self.ffn_b2 = self.sub_weights.get(W.ffn_b2)
        self.ffn_b3 = self.sub_weights.get(W.ffn_b3)
        self.ffn_b13 = self.sub_weights.get(W.ffn_b13)
        self.shared_expert_gate = self.sub_weights.get(W.shared_expert_gate)

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        return False

    def _shuff_moe_weight(
        self,
        name: str,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        load_config: LoadConfig,
    ):
        w = tensor.get(name)
        if isinstance(w, torch.Tensor):
            w = load_config.exported_device.shuffle_moe_weight(
                w, load_config.compute_dtype, name
            )
            tensor[name] = w
        elif isinstance(w, dict):
            self._shuff_moe_weight(name, w, load_config)
        else:
            raise ValueError("unsupported type")

    def _split(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        load_config: LoadConfig,
    ):
        res = super()._split(tensor, load_config)

        return res

    def _postprocess(
        self, tensor: Dict[str, torch.Tensor], device: str, load_config: LoadConfig
    ):
        moe_w1 = tensor.get(W.moe_w1)
        moe_w2 = tensor.get(W.moe_w2)
        for weight, keys in [
            (moe_w1, [W.moe_w1, W.moe_s1]),
            (moe_w2, [W.moe_w2, W.moe_s2]),
        ]:
            if isinstance(weight, dict):
                for key in keys:
                    if key in weight:
                        self._shuff_moe_weight(key, weight, load_config)
            else:
                self._shuff_moe_weight(keys[0], tensor, load_config)
        return super()._postprocess(tensor, device, load_config)

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


def gate_up_func_wrap(ts: List[torch.Tensor], origin_up, origin_gate):
    up_size = len(origin_up.weights)
    gate_size = len(origin_gate.weights)
    assert len(ts) == up_size + gate_size
    up = origin_up.process_fun(ts[:up_size])
    gate = origin_gate.process_fun(ts[up_size:])
    return torch.concat([gate, up], dim=-1).contiguous()


def gate_up_lora_a_func_wrap(
    ts: torch.Tensor, origin_up: FfnAtomicWeight, origin_gate: FfnAtomicWeight
):
    assert origin_up.lora_a_process_func and origin_gate.lora_a_process_func
    gate, up = torch.chunk(ts, 2, dim=-1)
    up = origin_up.lora_a_process_func(up)
    gate = origin_gate.lora_a_process_func(gate)
    return torch.concat([gate, up], dim=-1).contiguous()


def gate_up_lora_b_func_wrap(
    ts: torch.Tensor, origin_up: FfnAtomicWeight, origin_gate: FfnAtomicWeight
):
    assert origin_up.lora_b_process_func and origin_gate.lora_b_process_func
    gate, up = torch.chunk(ts, 2, dim=-1)
    up = origin_up.lora_b_process_func(up)
    gate = origin_gate.lora_b_process_func(gate)
    return torch.concat([gate, up], dim=-1).contiguous()


def gate_up_lora_a_split_func_wrap(
    ts: torch.Tensor, origin_up: FfnAtomicWeight, origin_gate: FfnAtomicWeight
):
    assert origin_up.lora_a_split_func and origin_gate.lora_a_split_func
    gate, up = torch.chunk(ts, 2, dim=-1)
    up = origin_up.lora_a_split_func(up)
    gate = origin_gate.lora_a_split_func(gate)
    return torch.concat([gate, up], dim=-1).contiguous()


def gate_up_lora_b_split_func_wrap(
    ts: torch.Tensor, origin_up: FfnAtomicWeight, origin_gate: FfnAtomicWeight
):
    assert origin_up.lora_b_split_func and origin_gate.lora_b_split_func
    gate, up = torch.chunk(ts, 2, dim=-1)
    up = origin_up.lora_b_split_func(up)
    gate = origin_gate.lora_b_split_func(gate)
    return torch.concat([gate, up], dim=-1).contiguous()


def fix_merge_gate_up(sub_weight_dict: Dict[str, FfnAtomicWeight]):
    origin_up = sub_weight_dict[W.ffn_up]
    origin_gate = sub_weight_dict[W.ffn_gate]
    w_list = origin_up.weights + origin_gate.weights
    lora_a_process_func = (
        functools.partial(
            gate_up_lora_a_func_wrap, origin_up=origin_up, origin_gate=origin_gate
        )
        if origin_up.lora_a_process_func
        else None
    )
    lora_b_process_func = (
        functools.partial(
            gate_up_lora_b_func_wrap, origin_up=origin_up, origin_gate=origin_gate
        )
        if origin_up.lora_b_process_func
        else None
    )
    lora_a_split_func = (
        functools.partial(
            gate_up_lora_a_split_func_wrap,
            origin_up=origin_up,
            origin_gate=origin_gate,
        )
        if origin_up.lora_a_split_func
        else None
    )
    lora_b_split_func = (
        functools.partial(
            gate_up_lora_b_split_func_wrap,
            origin_up=origin_up,
            origin_gate=origin_gate,
        )
        if origin_up.lora_b_split_func
        else None
    )
    gate_up = FfnAtomicWeight(
        name=W.ffn_gate_up,
        weights=w_list,
        process_fun=functools.partial(
            gate_up_func_wrap, origin_up=origin_up, origin_gate=origin_gate
        ),
        lora_a_process_func=lora_a_process_func,
        lora_b_process_func=lora_b_process_func,
        lora_a_split_func=lora_a_split_func,
        lora_b_split_func=lora_b_split_func,
        data_type=origin_up.data_type,
        config=origin_up.config,
    )

    sub_weight_dict.pop(W.ffn_up)
    sub_weight_dict.pop(W.ffn_gate)
    sub_weight_dict[W.ffn_gate_up] = gate_up
    return sub_weight_dict


def fix_merge_gate_up_b(sub_weight_dict: Dict[str, FfnAtomicWeight]):
    origin_up_b = sub_weight_dict[W.ffn_up_b]
    origin_gate_b = sub_weight_dict[W.ffn_gate_b]
    w_list = origin_up_b.weights + origin_gate_b.weights
    lora_a_process_func = (
        functools.partial(
            gate_up_lora_a_func_wrap, origin_up=origin_up_b, origin_gate=origin_gate_b
        )
        if origin_up_b.lora_a_process_func
        else None
    )
    lora_b_process_func = (
        functools.partial(
            gate_up_lora_b_func_wrap, origin_up=origin_up_b, origin_gate=origin_gate_b
        )
        if origin_up_b.lora_b_process_func
        else None
    )
    lora_a_split_func = (
        functools.partial(
            gate_up_lora_a_split_func_wrap,
            origin_up=origin_up_b,
            origin_gate=origin_gate_b,
        )
        if origin_up_b.lora_a_split_func
        else None
    )
    lora_b_split_func = (
        functools.partial(
            gate_up_lora_b_split_func_wrap,
            origin_up=origin_up_b,
            origin_gate=origin_gate_b,
        )
        if origin_up_b.lora_b_split_func
        else None
    )

    gate_up_b = FfnAtomicWeight(
        name=W.ffn_gate_up_b,
        weights=w_list,
        process_fun=functools.partial(
            gate_up_func_wrap, origin_up=origin_up_b, origin_gate=origin_gate_b
        ),
        lora_a_process_func=lora_a_process_func,
        lora_b_process_func=lora_b_process_func,
        lora_a_split_func=lora_a_split_func,
        lora_b_split_func=lora_b_split_func,
        data_type=origin_up_b.data_type,
        config=origin_up_b.config,
    )

    sub_weight_dict.pop(W.ffn_up_b)
    sub_weight_dict.pop(W.ffn_gate_b)
    sub_weight_dict[W.ffn_gate_up_b] = gate_up_b
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
        if W.ffn_up in sub_weight_dict and W.ffn_gate in sub_weight_dict:
            self.origin_up = sub_weight_dict[W.ffn_up]
            self.origin_gate = sub_weight_dict[W.ffn_gate]
            sub_weight_dict = fix_merge_gate_up(sub_weight_dict)
        if W.ffn_up_b in sub_weight_dict and W.ffn_gate_b in sub_weight_dict:
            self.origin_up_b = sub_weight_dict[W.ffn_up_b]
            self.origin_gate_b = sub_weight_dict[W.ffn_gate_b]
            sub_weight_dict = fix_merge_gate_up_b(sub_weight_dict)

        kwargs["name"] = W.ffn

        super().__init__(sub_weight_dict, *args, **kwargs)

        self.up = self.sub_weights.get(W.ffn_up)
        self.down = self.sub_weights.get(W.ffn_down)
        self.gate = self.sub_weights.get(W.ffn_gate)
        self.gate_up = self.sub_weights.get(W.ffn_gate_up)
        self.up_b = self.sub_weights.get(W.ffn_up_b)
        self.down_b = self.sub_weights.get(W.ffn_down_b)
        self.gate_b = self.sub_weights.get(W.ffn_gate_b)
        self.gate_up_b = self.sub_weights.get(W.ffn_gate_up_b)

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
            if self.name not in [W.moe_gate_up, W.moe_down]:
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

        self.moe_gate_up = self.sub_weights[W.moe_gate_up]
        self.moe_down = self.sub_weights[W.moe_down]
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
        moe_gate_up = tensor.get(W.moe_gate_up)
        moe_down = tensor.get(W.moe_down)
        for weight, keys in [
            (moe_gate_up, [W.moe_gate_up, W.moe_gate_up_s]),
            (moe_down, [W.moe_down, W.moe_down_s]),
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

        if W.ffn_up in sub_weight_dict and W.ffn_gate in sub_weight_dict:
            self.origin_up = sub_weight_dict[W.ffn_up]
            self.origin_gate = sub_weight_dict[W.ffn_gate]
            sub_weight_dict = fix_merge_gate_up(sub_weight_dict)
        if W.ffn_up_b in sub_weight_dict and W.ffn_gate_b in sub_weight_dict:
            self.origin_up_b = sub_weight_dict[W.ffn_up_b]
            self.origin_gate_b = sub_weight_dict[W.ffn_gate_b]
            sub_weight_dict = fix_merge_gate_up_b(sub_weight_dict)

        super().__init__(sub_weight_dict, **kwargs)

        self.moe_gate_up = self.sub_weights.get(W.moe_gate_up)
        self.moe_down = self.sub_weights.get(W.moe_down)
        self.moe_gate = self.sub_weights.get(W.moe_gate)
        self.ffn_up = self.sub_weights.get(W.ffn_up)
        self.ffn_down = self.sub_weights.get(W.ffn_down)
        self.ffn_gate = self.sub_weights.get(W.ffn_gate)
        self.gate_up = self.sub_weights.get(W.ffn_gate_up)
        self.ffn_up_b = self.sub_weights.get(W.ffn_up_b)
        self.ffn_down_b = self.sub_weights.get(W.ffn_down_b)
        self.ffn_gate_b = self.sub_weights.get(W.ffn_gate_b)
        self.ffn_gate_up_b = self.sub_weights.get(W.ffn_gate_up_b)
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
        moe_gate_up = tensor.get(W.moe_gate_up)
        moe_down = tensor.get(W.moe_down)
        for weight, keys in [
            (moe_gate_up, [W.moe_gate_up, W.moe_gate_up_s]),
            (moe_down, [W.moe_down, W.moe_down_s]),
        ]:
            if isinstance(weight, dict):
                for key in keys:
                    if key in weight:
                        self._shuff_moe_weight(key, weight, load_config)
            else:
                self._shuff_moe_weight(keys[0], tensor, load_config)
        return super()._postprocess(tensor, device, load_config)

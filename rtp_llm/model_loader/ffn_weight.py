import functools
import logging
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from pydantic import BaseModel

from rtp_llm.config.quant_config import QuantizationConfig
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.tensor_source import StackSplitTensorSource, TensorSource
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CompositeWeight,
    QuantWeight,
    WeightModule,
)
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity


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
    w1, w3 = torch.chunk(ts, 2, dim=-1)
    w1 = origin_w1.lora_a_process_func(w1)
    w3 = origin_w3.lora_a_process_func(w3)
    return torch.concat([w1, w3], dim=-1).contiguous()


def w13_lora_b_func_wrap(
    ts: torch.Tensor, origin_w1: FfnAtomicWeight, origin_w3: FfnAtomicWeight
):
    assert origin_w1.lora_b_process_func and origin_w3.lora_b_process_func
    w1, w3 = torch.chunk(ts, 2, dim=-1)
    w1 = origin_w1.lora_b_process_func(w1)
    w3 = origin_w3.lora_b_process_func(w3)
    return torch.concat([w1, w3], dim=-1).contiguous()


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
    return torch.concat([w1, w3], dim=-1).contiguous()


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


class MoeAtomicWeight(AtomicWeight):
    def __init__(
        self,
        name: str,
        weights: List[CkptWeightInfo],
        process_fun: Callable[[List[torch.Tensor]], torch.Tensor] = identity,
        data_type: Optional[torch.dtype] = None,
        config: MoeConfig = None,
        stacked_ckpt_keys: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        self.config = config
        self.stacked_ckpt_keys = stacked_ckpt_keys
        super().__init__(name, weights, process_fun, data_type, *args, **kwargs)

    def _expert_key_pattern(self, idx: int) -> str:
        """Generate a logical per-expert key for the idx-th stacked weight."""
        return f"layers.{{i}}.moe.{self.name}.{{expert_id}}.{idx}"

    def _get_expert_weights(self) -> List[CkptWeightInfo]:
        """Generate per-expert CkptWeightInfo with logical keys for stacked weights."""
        return [
            CkptWeightInfo(self._expert_key_pattern(idx))
            for idx in range(len(self.weights))
        ]

    def _build_split_config(
        self, layer_id: Optional[int], load_config: LoadConfig
    ) -> Dict[str, Tuple[str, int, Callable]]:
        """Build per-expert-key -> (stacked_key, expert_id, merge_fun) mapping."""
        split_config = {}
        selected_experts = load_config.get_selected_experts(
            layer_id, self.config.expert_num
        )
        for idx, ckpt_weight in enumerate(self.weights):
            stacked_key = ckpt_weight.tensor_name(layer_id)
            pattern = self._expert_key_pattern(idx)
            for expert_id in selected_experts:
                per_expert_key = pattern.format(
                    i=str(layer_id), expert_id=str(expert_id)
                )
                split_config[per_expert_key] = (
                    stacked_key,
                    expert_id,
                    ckpt_weight.merge_fun,
                )
        return split_config

    def _load_raw_tensor(
        self,
        tensor_source: TensorSource,
        layer_id: Optional[int],
        device: str,
        load_config: LoadConfig,
    ):
        if self.stacked_ckpt_keys and tensor_source.has_tensor(
            self.weights[0].tensor_name(layer_id)
        ):
            tensor_source = StackSplitTensorSource(
                tensor_source,
                self._build_split_config(layer_id, load_config),
            )
        ckpt_weights = (
            self._get_expert_weights() if self.stacked_ckpt_keys else self.weights
        )

        convert_type = (
            self.data_type if self.data_type is not None else load_config.compute_dtype
        )
        selected_experts = load_config.get_selected_experts(
            layer_id, self.config.expert_num
        )
        num_experts = len(selected_experts)
        num_ckpt_weights = len(ckpt_weights)

        # Try GPU pre-allocate + direct copy path for large MoE weights
        if (
            num_experts > 1
            and torch.cuda.is_available()
            and self.process_fun.__name__
            in ("stack_moe_w1", "stack_", "stack_moe_w1_s2")
        ):
            return self._load_raw_tensor_gpu_preallocate(
                tensor_source,
                layer_id,
                device,
                load_config,
                ckpt_weights,
                selected_experts,
                convert_type,
            )

        # Fallback: original serial path
        before_merge_tensors = []
        for ckpt_weight in ckpt_weights:
            for expert_id in selected_experts:
                name = ckpt_weight.name.format(
                    i=str(layer_id), i_1=str(layer_id + 1), expert_id=str(expert_id)
                )
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
        return {self.name: after_merge_tensor}

    def _load_raw_tensor_gpu_preallocate(
        self,
        tensor_source,
        layer_id,
        device,
        load_config,
        ckpt_weights,
        selected_experts,
        convert_type,
    ):
        """Pre-allocate output tensor on GPU and copy each expert directly into position.
        Avoids expensive CPU stack of thousands of small tensors."""
        num_experts = len(selected_experts)
        num_ckpt_weights = len(ckpt_weights)
        gpu_device = "cuda:0"

        # Peek at first tensor to get shape
        first_name = ckpt_weights[0].name.format(
            i=str(layer_id),
            i_1=str(layer_id + 1),
            expert_id=str(selected_experts[0]),
        )
        first_tensor = ckpt_weights[0].merge_fun(
            tensor_source.load_tensor(first_name, convert_type)
        )
        expert_shape = first_tensor.shape  # e.g., [intermediate, hidden] for fp8

        is_w1 = self.process_fun.__name__ == "stack_moe_w1"
        is_w1_s2 = self.process_fun.__name__ == "stack_moe_w1_s2"

        if is_w1:
            # stack_moe_w1: gate[512] + up[512] → [512, 2*intermediate, hidden]
            # ckpt_weights has 2 entries (gate, up), each with 512 experts
            assert num_ckpt_weights == 2
            dim0 = expert_shape[0]  # intermediate_size
            dim1 = expert_shape[1] if len(expert_shape) > 1 else 1
            out = torch.empty(
                [num_experts, dim0 * 2, dim1],
                dtype=convert_type,
                device=gpu_device,
            )
            # Fill gate (first ckpt_weight) into [:, :dim0, :]
            # Fill up (second ckpt_weight) into [:, dim0:, :]
            for cw_idx, ckpt_weight in enumerate(ckpt_weights):
                row_offset = cw_idx * dim0
                for local_idx, expert_id in enumerate(selected_experts):
                    name = ckpt_weight.name.format(
                        i=str(layer_id),
                        i_1=str(layer_id + 1),
                        expert_id=str(expert_id),
                    )
                    if name == first_name:
                        t = first_tensor
                    else:
                        t = ckpt_weight.merge_fun(
                            tensor_source.load_tensor(name, convert_type)
                        )
                    out[local_idx, row_offset : row_offset + dim0, :].copy_(t)
        elif is_w1_s2:
            # stack_moe_w1_s2: same structure as w1 but for scale (max of gate/up scales)
            assert num_ckpt_weights == 2
            dim0 = expert_shape[0]
            dim1 = expert_shape[1] if len(expert_shape) > 1 else 1
            # Load gate and up scales, compute max, store
            out = torch.empty(
                [num_experts, dim0, dim1],
                dtype=convert_type,
                device=gpu_device,
            )
            gate_scales = []
            up_scales = []
            for ckpt_weight_idx, ckpt_weight in enumerate(ckpt_weights):
                target = gate_scales if ckpt_weight_idx == 0 else up_scales
                for expert_id in selected_experts:
                    name = ckpt_weight.name.format(
                        i=str(layer_id),
                        i_1=str(layer_id + 1),
                        expert_id=str(expert_id),
                    )
                    t = ckpt_weight.merge_fun(
                        tensor_source.load_tensor(name, convert_type)
                    )
                    target.append(t)
            for i in range(num_experts):
                out[i].copy_(torch.max(gate_scales[i], up_scales[i]))
            return {self.name: out}
        else:
            # stack_: simple stack → [512, *expert_shape]
            out = torch.empty(
                [num_experts] + list(expert_shape),
                dtype=convert_type,
                device=gpu_device,
            )
            for cw_idx, ckpt_weight in enumerate(ckpt_weights):
                for local_idx, expert_id in enumerate(selected_experts):
                    name = ckpt_weight.name.format(
                        i=str(layer_id),
                        i_1=str(layer_id + 1),
                        expert_id=str(expert_id),
                    )
                    if name == first_name:
                        t = first_tensor
                    else:
                        t = ckpt_weight.merge_fun(
                            tensor_source.load_tensor(name, convert_type)
                        )
                    out[local_idx].copy_(t)

        return {self.name: out}

    def get_tensor_names(
        self, layer_id: Optional[int], load_config: LoadConfig
    ) -> set[str]:
        ckpt_weights = (
            self._get_expert_weights() if self.stacked_ckpt_keys else self.weights
        )

        names = set[str]()
        for ckpt_weight in ckpt_weights:
            selected_experts = load_config.get_selected_experts(
                layer_id, self.config.expert_num
            )
            for expert_id in selected_experts:
                name = ckpt_weight.name.format(
                    i=str(layer_id), i_1=str(layer_id + 1), expert_id=str(expert_id)
                )
                names.add(name)
        return names


def iter_stacked_moe_weights(weight: WeightModule):
    """Yield all MoeAtomicWeight instances with stacked_ckpt_keys from a weight tree."""
    if isinstance(weight, MoeAtomicWeight) and weight.stacked_ckpt_keys:
        yield weight
    elif isinstance(weight, CompositeWeight):
        for sub_weight in weight.sub_weights.values():
            yield from iter_stacked_moe_weights(sub_weight)


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
        self.moe_gate = self.sub_weights.get(W.moe_gate)

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

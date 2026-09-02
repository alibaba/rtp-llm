import functools
import logging
import traceback
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from pydantic import BaseModel

from rtp_llm.config.quant_config import QuantizationConfig
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.tensor_source import (
    DatabaseTensorSource,
    StackSplitTensorSource,
    TensorSource,
)
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CompositeWeight,
    QuantWeight,
    WeightModule,
)
from rtp_llm.utils import model_weight as mw
from rtp_llm.utils.database import CkptDatabase
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


class PreShardedTensor(NamedTuple):
    """Marker for rank-local tensors that must skip the _split stage."""

    tensor: torch.Tensor


# Keyed by (weight name, process function, checkpoint tensor count).
_PURE_TP_LAYOUTS = {
    (W.moe_w2, mw.stack_, 1): (1, (0,), False, mw.sp_moe_neg1),
    (W.moe_s2, mw.stack_, 1): (1, (0,), False, mw.sp_moe_neg1),
    (W.moe_w1, mw.stack_moe_w1, 2): (0, (0,), False, mw.sp_moe_w1),
    (W.moe_s1, mw.stack_moe_w1, 2): (0, (0,), False, mw.sp_moe_w1),
    (W.moe_w1, mw.transpose_stack_moe_w1, 1): (0, (1, 0), True, mw.sp_moe_w1),
}


class MoeExpertLayout(NamedTuple):
    """Resolved checkpoint/source layout shared by MoE loading policies."""

    selected_experts: Tuple[int, ...]
    tensor_source: TensorSource
    ckpt_weights: Tuple[CkptWeightInfo, ...]
    uses_stacked_keys: bool
    source_contains_raw_stacked: bool


class MoeAtomicWeight(AtomicWeight):
    # A pre-sharded tensor reaching an online-quant clone crashes the load;
    # capable clones opt back in (see per_block_fp8_quant_weight).
    _CLONE_EXCLUDED = frozenset({"enable_pure_tp_preshard"})

    def __init__(
        self,
        name: str,
        weights: List[CkptWeightInfo],
        process_fun: Callable[[List[torch.Tensor]], torch.Tensor] = identity,
        data_type: Optional[torch.dtype] = None,
        config: MoeConfig = None,
        stacked_ckpt_keys: bool = False,
        enable_pure_tp_preshard: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        self.config = config
        self.stacked_ckpt_keys = stacked_ckpt_keys
        self.enable_pure_tp_preshard = enable_pure_tp_preshard
        # Pre-resolve function name for GPU preallocate path dispatch.
        # functools.partial objects have .func instead of __name__.
        if isinstance(process_fun, functools.partial):
            self._process_fun_name = process_fun.func.__name__
        else:
            self._process_fun_name = process_fun.__name__
        super().__init__(name, weights, process_fun, data_type, *args, **kwargs)

    def _split(self, tensor, load_config: LoadConfig):
        raw_tensor = tensor.get(self.name) if isinstance(tensor, dict) else tensor
        if isinstance(raw_tensor, PreShardedTensor):
            return {self.name: raw_tensor.tensor}
        return super()._split(tensor, load_config)

    def _expert_key_pattern(self, idx: int) -> str:
        """Generate a logical per-expert key for the idx-th stacked weight."""
        return f"layers.{{i}}.moe.{self.name}.{{expert_id}}.{idx}"

    def _ckpt_name(self, ckpt_weight: CkptWeightInfo, layer_id: int, expert_id: int):
        return ckpt_weight.name.format(
            i=str(layer_id), i_1=str(layer_id + 1), expert_id=str(expert_id)
        )

    def _get_expert_weights(self) -> List[CkptWeightInfo]:
        """Generate per-expert CkptWeightInfo with logical keys for stacked weights."""
        return [
            CkptWeightInfo(self._expert_key_pattern(idx))
            for idx in range(len(self.weights))
        ]

    @property
    def process_fun_name(self) -> str:
        return self._process_fun_name

    def raw_stacked_tensor_names(self, layer_id: Optional[int]) -> List[str]:
        """Return concrete raw stacked keys, or empty for per-expert templates."""

        if not self.stacked_ckpt_keys or not self.weights:
            return []
        names = []
        for ckpt_weight in self.weights:
            # A checkpoint template that names ``expert_id`` is already a
            # per-expert layout. Formatting it as a raw stacked key would
            # either raise KeyError or misclassify expert 0 as a stacked tensor.
            if "{expert_id" in ckpt_weight.name:
                return []
            names.append(ckpt_weight.tensor_name(layer_id))
        return names

    # Compatibility alias for existing in-class tests and callers. New code
    # should use the public layout resolver instead of composing private probes.
    _raw_stacked_tensor_names = raw_stacked_tensor_names

    def _has_raw_stacked_tensors(self, tensor_source, layer_id: Optional[int]) -> bool:
        """Return whether every raw tensor required by this atomic weight exists."""

        names = self.raw_stacked_tensor_names(layer_id)
        return bool(names) and all(tensor_source.has_tensor(name) for name in names)

    def uses_stacked_expert_keys(self, database, layer_id: Optional[int]) -> bool:
        """Return whether the checkpoint layout requires logical expert keys.

        This decision must use the immutable checkpoint database. A
        TensorCollector populated by AutoLoader contains logical expert keys
        even when the underlying checkpoint stores raw stacked tensors.
        """

        return self._has_raw_stacked_tensors(database, layer_id)

    def _has_logical_expert_tensors(
        self,
        tensor_source: TensorSource,
        layer_id: Optional[int],
        selected_experts: List[int],
    ) -> bool:
        """Return whether the source already contains every logical expert key."""

        if not self.stacked_ckpt_keys or not selected_experts:
            return False
        return all(
            tensor_source.has_tensor(
                self._expert_key_pattern(idx).format(
                    i=str(layer_id), expert_id=str(expert_id)
                )
            )
            for idx in range(len(self.weights))
            for expert_id in selected_experts
        )

    def _build_split_config(
        self, layer_id: Optional[int], load_config: LoadConfig
    ) -> Dict[str, Tuple[str, int, Callable]]:
        """Build per-expert-key -> (stacked_key, expert_id, merge_fun) mapping."""
        split_config = {}
        selected_experts = load_config.get_selected_experts(
            layer_id, self.config.expert_num
        )
        stacked_keys = self.raw_stacked_tensor_names(layer_id)
        for idx, (ckpt_weight, stacked_key) in enumerate(
            zip(self.weights, stacked_keys)
        ):
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

    def resolve_expert_layout(
        self,
        tensor_source: TensorSource,
        layer_id: Optional[int],
        load_config: LoadConfig,
    ) -> MoeExpertLayout:
        """Resolve selected experts, source wrapping, and logical ckpt keys once."""

        selected_experts = tuple(
            load_config.get_selected_experts(layer_id, self.config.expert_num)
        )
        uses_stacked_keys = self.uses_stacked_expert_keys(
            tensor_source.get_database(), layer_id
        ) or self._has_logical_expert_tensors(
            tensor_source, layer_id, list(selected_experts)
        )
        source_contains_raw_stacked = uses_stacked_keys and (
            self._has_raw_stacked_tensors(tensor_source, layer_id)
        )
        resolved_source = tensor_source
        if source_contains_raw_stacked:
            resolved_source = StackSplitTensorSource(
                tensor_source,
                self._build_split_config(layer_id, load_config),
            )
        ckpt_weights = (
            tuple(self._get_expert_weights())
            if uses_stacked_keys
            else tuple(self.weights)
        )
        return MoeExpertLayout(
            selected_experts=selected_experts,
            tensor_source=resolved_source,
            ckpt_weights=ckpt_weights,
            uses_stacked_keys=uses_stacked_keys,
            source_contains_raw_stacked=source_contains_raw_stacked,
        )

    def _postprocess(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        device: str,
        load_config: LoadConfig,
    ):
        raw_tensor = tensor.get(self.name) if isinstance(tensor, dict) else tensor
        # Scale (moe_s1/moe_s2) must also pass shuffle_moe_weight: it applies the
        # up/gate swap (do_shuffle=False skips the layout shuffle for scale).
        if self.name in [W.moe_w1, W.moe_w2, W.moe_s1, W.moe_s2]:
            raw_tensor = load_config.exported_device.shuffle_moe_weight(
                raw_tensor, load_config.compute_dtype, self.name
            )
        return {
            self.name: load_config.exported_device.maybe_rewrite_weight_by_key(
                self.name, raw_tensor
            )
        }

    def _load_raw_tensor(
        self,
        tensor_source: TensorSource,
        layer_id: Optional[int],
        device: str,
        load_config: LoadConfig,
    ):
        pre_sharded = self._load_pure_tp(tensor_source, layer_id, device, load_config)
        if pre_sharded is not None:
            return {self.name: PreShardedTensor(pre_sharded)}

        layout = self.resolve_expert_layout(tensor_source, layer_id, load_config)
        selected_experts = layout.selected_experts
        tensor_source = layout.tensor_source
        ckpt_weights = layout.ckpt_weights

        convert_type = (
            self.data_type if self.data_type is not None else load_config.compute_dtype
        )
        num_experts = len(selected_experts)
        num_ckpt_weights = len(ckpt_weights)

        # Try GPU pre-allocate + direct copy path for large MoE weights
        # Only when CUDA is available and target device is GPU
        target_device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        if (
            num_experts > 1
            and torch.cuda.is_available()
            and target_device.type == "cuda"
            and self._process_fun_name in ("stack_moe_w1", "stack_", "stack_moe_w1_s2")
        ):
            result = self._load_raw_tensor_gpu_preallocate(
                tensor_source,
                layer_id,
                device,
                load_config,
                ckpt_weights,
                selected_experts,
                convert_type,
            )
            if result is not None:
                return result

        # Fallback: original serial path
        before_merge_tensors = []
        for ckpt_weight in ckpt_weights:
            for expert_id in selected_experts:
                name = self._ckpt_name(ckpt_weight, layer_id, expert_id)
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

    def _load_pure_tp(
        self,
        tensor_source: TensorSource,
        layer_id: Optional[int],
        device: str,
        load_config: LoadConfig,
    ) -> Optional[torch.Tensor]:
        if not (load_config.moe_pure_tp_preshard and self.enable_pure_tp_preshard):
            return None

        layout = _PURE_TP_LAYOUTS.get((self.name, self.process_fun, len(self.weights)))
        database = (
            tensor_source.get_database()
            if isinstance(tensor_source, DatabaseTensorSource)
            else None
        )
        log_context = f"{self.name} layer {layer_id}: pre-shard"
        if (
            layout is None
            or not load_config.moe_pure_tp_mode
            or load_config.merge_lora
            or layer_id is None
            or not isinstance(database, CkptDatabase)
            or not database.is_safetensor
            or not all(weight.merge_fun is identity for weight in self.weights)
        ):
            logging.warning(f"{log_context} unavailable; using legacy read")
            return None

        split_dim, segments, requires_stacked, split_func = layout
        is_stacked = self.uses_stacked_expert_keys(database, layer_id)
        if (requires_stacked and not is_stacked) or (
            self._get_split_func() is not split_func
        ):
            logging.warning(f"{log_context} fallback: incompatible weight layout")
            return None

        experts = list(
            load_config.get_selected_experts(layer_id, self.config.expert_num)
        )
        if not experts:
            logging.warning(f"{log_context} fallback: no experts selected")
            return None

        expert_dims = slice(1, None) if is_stacked else slice(None)

        def shape_of(weight: CkptWeightInfo, expert: int) -> List[int]:
            name = self._ckpt_name(weight, layer_id, expert)
            return list(database.get_tensor_shape(name))[expert_dims]

        # Metadata-only reads: keep legacy's every-expert shape fail-fast.
        shapes = [shape_of(w, e) for w in self.weights for e in experts]
        expert_shape = shapes[0]
        if len(expert_shape) != 2 or any(shape != expert_shape for shape in shapes):
            logging.warning(f"{log_context} fallback: incompatible expert shapes")
            return None

        divisor = len(segments) * load_config.tp_size
        if expert_shape[split_dim] % divisor:
            logging.warning(
                f"{log_context} fallback: dimension {expert_shape[split_dim]} is not "
                f"divisible by tp_size x segments ({divisor})"
            )
            return None
        if layer_id == 0:
            logging.info(f"{self.name}: pure-TP pre-shard (tp={load_config.tp_size})")

        segment_size = expert_shape[split_dim] // len(segments)
        shard_size = segment_size // load_config.tp_size
        output_shape = expert_shape.copy()
        output_shape[split_dim] = shard_size * len(segments) * len(self.weights)
        dtype = self.data_type or load_config.compute_dtype
        output = torch.empty((len(experts), *output_shape), dtype=dtype, device=device)
        # Strided last-dim safetensors read: read full expert, slice host-side.
        read_full_expert = split_dim == 1
        dst = 0
        full = (slice(None), slice(None))
        for weight in self.weights:
            for segment in segments:
                start = segment * segment_size + load_config.tp_rank * shard_size
                cut = slice(start, start + shard_size)
                shard_slice = (slice(None), cut) if split_dim else (cut, slice(None))
                read_slice = full if read_full_expert else shard_slice
                for slot, expert in enumerate(experts):
                    name = self._ckpt_name(weight, layer_id, expert)
                    where = (*((expert,) if is_stacked else ()), *read_slice)
                    loaded = database.load_tensor_slice(name, where, dtype)
                    if read_full_expert:
                        loaded = loaded[shard_slice].contiguous()
                    output[slot].narrow(split_dim, dst, shard_size).copy_(loaded)
                dst += shard_size
        return output

    def load_expert_tensor(
        self,
        ckpt_weight,
        layer_id,
        expert_id,
        tensor_source,
        convert_type,
        first_name=None,
        first_tensor=None,
    ):
        """Load a single expert tensor with error handling."""
        name = self._ckpt_name(ckpt_weight, layer_id, expert_id)
        if first_name is not None and name == first_name:
            return name, first_tensor
        try:
            t = ckpt_weight.merge_fun(tensor_source.load_tensor(name, convert_type))
            return name, t
        except Exception as e:
            logging.error(f"加载 {name} 失败，完整堆栈:\n{traceback.format_exc()}")
            raise e

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
        gpu_device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )

        # Peek at first tensor to get shape
        first_name, first_tensor = self.load_expert_tensor(
            ckpt_weights[0],
            layer_id,
            selected_experts[0],
            tensor_source,
            convert_type,
        )
        expert_shape = first_tensor.shape  # e.g., [intermediate, hidden] for fp8

        is_w1 = self._process_fun_name == "stack_moe_w1"
        is_w1_s2 = self._process_fun_name == "stack_moe_w1_s2"

        if is_w1:
            # stack_moe_w1: gate[512] + up[512] → [512, 2*intermediate, hidden]
            # For non-2D tensors (e.g. per-tensor quant scales), fall back to
            # the normal serial path which handles all shapes.
            if len(expert_shape) != 2:
                return None
            assert num_ckpt_weights == 2
            dim0, dim1 = expert_shape
            out = torch.empty(
                [num_experts, dim0 * 2, dim1],
                dtype=convert_type,
                device=gpu_device,
            )
            for cw_idx, ckpt_weight in enumerate(ckpt_weights):
                row_offset = cw_idx * dim0
                for local_idx, expert_id in enumerate(selected_experts):
                    _, t = self.load_expert_tensor(
                        ckpt_weight,
                        layer_id,
                        expert_id,
                        tensor_source,
                        convert_type,
                        first_name,
                        first_tensor,
                    )
                    out[local_idx, row_offset : row_offset + dim0, :].copy_(t)
        elif is_w1_s2:
            # stack_moe_w1_s2: scale (max of gate/up scales per expert)
            assert num_ckpt_weights == 2
            out = torch.empty(
                [num_experts] + list(expert_shape),
                dtype=convert_type,
                device=gpu_device,
            )
            gate_scales = []
            up_scales = []
            for cw_idx, ckpt_weight in enumerate(ckpt_weights):
                target = gate_scales if cw_idx == 0 else up_scales
                for expert_id in selected_experts:
                    _, t = self.load_expert_tensor(
                        ckpt_weight,
                        layer_id,
                        expert_id,
                        tensor_source,
                        convert_type,
                        first_name,
                        first_tensor,
                    )
                    target.append(t)
            for i in range(num_experts):
                out[i].copy_(torch.max(gate_scales[i], up_scales[i]))
            return {self.name: out}
        else:
            # stack_: simple stack → [num_experts, *expert_shape]
            assert (
                num_ckpt_weights == 1
            ), f"stack_ fast path expects 1 ckpt_weight, got {num_ckpt_weights}"
            out = torch.empty(
                [num_experts] + list(expert_shape),
                dtype=convert_type,
                device=gpu_device,
            )
            ckpt_weight = ckpt_weights[0]
            for local_idx, expert_id in enumerate(selected_experts):
                _, t = self.load_expert_tensor(
                    ckpt_weight,
                    layer_id,
                    expert_id,
                    tensor_source,
                    convert_type,
                    first_name,
                    first_tensor,
                )
                out[local_idx].copy_(t)

        return {self.name: out}

    def get_tensor_names(
        self, layer_id: Optional[int], load_config: LoadConfig
    ) -> set[str]:
        has_stacked_tensor = self.uses_stacked_expert_keys(
            load_config.database, layer_id
        )
        ckpt_weights = (
            self._get_expert_weights() if has_stacked_tensor else self.weights
        )

        names = set[str]()
        for ckpt_weight in ckpt_weights:
            selected_experts = load_config.get_selected_experts(
                layer_id, self.config.expert_num
            )
            for expert_id in selected_experts:
                names.add(self._ckpt_name(ckpt_weight, layer_id, expert_id))
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
        # MoE weight shuffle is handled by MoeAtomicWeight._postprocess
        # (called via CompositeWeight._postprocess's recursive sub_weight loop).
        # Do NOT shuffle here to avoid double-shuffle.
        return super()._postprocess(tensor, device, load_config)

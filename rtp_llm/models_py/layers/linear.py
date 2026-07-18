import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.module_base import RtpModule, copy_weight_
from rtp_llm.models_py.quant_methods.base import QuantizationConfig


def _require_positive_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")


def _validate_parallel_partition(size: int, rank: int, label: str) -> None:
    _require_positive_int(size, f"{label} size")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < size:
        raise ValueError(f"Invalid {label} partition: rank={rank}, size={size}")


def _require_divisible(value: int, divisor: int, label: str) -> None:
    _require_positive_int(value, label)
    _require_positive_int(divisor, "divisor")
    if value % divisor != 0:
        raise ValueError(f"{label}={value} must be divisible by {divisor}")


def _projection_name(weight_name: str, shard_names: List[str]) -> Optional[str]:
    parts = weight_name.split(".")
    if len(parts) < 2:
        return None
    candidate = parts[-2]
    return candidate if candidate in shard_names else None


def _copy_checked(target: torch.Tensor, source: torch.Tensor, label: str) -> None:
    copy_weight_(target, source, label)


class LinearBase(RtpModule):

    # FP8 per-block (DeepSeek-style) quantization block size, used when
    # TP-slicing / shard-merging `weight_scale_inv` block grids.
    _FP8_BLOCK = 128

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        bias: bool = False,
        params_dtype: torch.dtype = torch.float16,
        **kwargs,
    ):
        super().__init__()
        _validate_parallel_partition(tp_size, tp_rank, "TP")
        _require_positive_int(input_size, "input_size")
        _require_positive_int(output_size, "output_size")
        if not isinstance(bias, bool):
            raise TypeError("bias must be a bool")
        if not isinstance(params_dtype, torch.dtype):
            raise TypeError("params_dtype must be a torch.dtype")
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.prefix = prefix
        self.params_dtype = params_dtype
        self._main_weight_loaded = False
        self._loaded_parameter_names = set()

        if quant_config is None:
            quant_config = QuantizationConfig(quant_type="none")
        self.quant_config = quant_config
        self.quant_method = quant_config.get_quant_method(self, prefix)

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(output_size, dtype=params_dtype), requires_grad=False
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 0 or x.shape[-1] != self.input_size:
            raise ValueError(
                f"{self.prefix or type(self).__name__} expected input width "
                f"{self.input_size}, got {tuple(x.shape)}"
            )
        output = self.quant_method.apply(self, x, self.bias)
        return output

    def validate_runtime_device(self, device: torch.device) -> None:
        self.quant_method.validate_runtime_device(device)

    def process_weights_after_loading(self):
        # Called once by NewModelLoader._run_post_load_hooks after model.to(device)
        # and after every weight shard is in place. Doing the work here (rather
        # than at the end of load_weights) is required for streaming dispatch:
        # MergedColumnParallelLinear / QKVParallelLinear receive their shards on
        # separate RtpModule.load_weights ticks, so a per-tick call would quantize
        # before all shards arrived (corrupt scale) and then re-quantize an
        # already-fp8 weight on the next tick (kernel dispatch failure).
        if getattr(self, "_post_load_done", False):
            return
        self.quant_method.process_weights_after_loading(self)
        self._post_load_done = True

    def validate_weights_loaded(self, loaded_tensor_ids=None) -> None:
        required = set(
            getattr(
                self.quant_method,
                "required_checkpoint_parameters",
                ("weight",),
            )
        )
        if self.bias is not None:
            required.add("bias")
        missing = [
            name for name in required if name not in self._loaded_parameter_names
        ]
        if missing:
            raise RuntimeError(
                f"{self.prefix} is missing checkpoint tensors {sorted(missing)}"
            )

    def _record_parameter_loaded(self, name: str) -> None:
        if name in self._loaded_parameter_names:
            raise RuntimeError(f"Duplicate checkpoint tensor {self.prefix}.{name}")
        self._loaded_parameter_names.add(name)
        self._mark_weight_loaded(name)
        if name == "weight":
            self._main_weight_loaded = True

    def _fp8_scale_block_size(self):
        value = getattr(
            self.quant_config, "weight_block_size", [self._FP8_BLOCK, self._FP8_BLOCK]
        )
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"Invalid FP8 weight_block_size {value!r}")
        block_n, block_k = value
        _require_positive_int(block_n, "FP8 output block size")
        _require_positive_int(block_k, "FP8 input block size")
        return block_n, block_k

    @staticmethod
    def _ceil_div(x: int, y: int) -> int:
        return (x + y - 1) // y


class ColumnParallelLinear(LinearBase):

    shard_names: List[str] = []

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        bias: bool = False,
        gather_output: bool = False,
        params_dtype: torch.dtype = torch.float16,
        **kwargs,
    ):
        _validate_parallel_partition(tp_size, tp_rank, "TP")
        if not isinstance(gather_output, bool):
            raise TypeError("gather_output must be a bool")
        if gather_output:
            raise ValueError("gather_output is not supported by ColumnParallelLinear")
        _require_divisible(output_size, tp_size, "output_size")
        self.output_size_per_partition = output_size // tp_size
        self.gather_output = gather_output

        super().__init__(
            input_size=input_size,
            output_size=self.output_size_per_partition,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=prefix,
            bias=bias,
            params_dtype=params_dtype,
            **kwargs,
        )

        self.quant_method.create_weights(
            layer=self,
            input_size=input_size,
            output_size=self.output_size_per_partition,
            params_dtype=params_dtype,
        )

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        for name, tensor in weights.items():
            param_name = self._get_param_name(name)
            if param_name is None:
                continue

            param = getattr(self, param_name, None)
            if param is None or not isinstance(param, nn.Parameter):
                raise RuntimeError(f"Unsupported tensor {self.prefix}.{name}")
            if param_name in self._loaded_parameter_names:
                raise RuntimeError(f"Duplicate checkpoint tensor {self.prefix}.{name}")

            if param_name == "weight":
                global_output = self.output_size_per_partition * self.tp_size
                expected = (global_output, self.input_size)
                if tuple(tensor.shape) == expected:
                    pass
                elif tuple(tensor.shape) == (self.input_size, global_output):
                    tensor = tensor.t().contiguous()
                else:
                    raise ValueError(
                        f"{self.prefix}.weight checkpoint shape must be {expected}, "
                        f"got {tuple(tensor.shape)}"
                    )
                tensor = self._split_weight(tensor, dim=0)
            elif param_name == "bias":
                tensor = self._split_weight(tensor, dim=0)
            elif param_name in ("weight_scale", "input_scale"):
                if (
                    tensor.numel() > 1
                    and tensor.shape[0] == self.output_size_per_partition * self.tp_size
                ):
                    tensor = self._split_weight(tensor, dim=0)
            elif param_name == "weight_scale_inv":
                if self.tp_size > 1:
                    block_n, _ = self._fp8_scale_block_size()
                    start_row = self.tp_rank * self.output_size_per_partition
                    if (
                        start_row % block_n != 0
                        or self.output_size_per_partition % block_n != 0
                    ):
                        raise ValueError(
                            f"{self.prefix}.weight_scale_inv TP output shard must "
                            f"align to FP8 block {block_n}: start={start_row}, "
                            f"rows={self.output_size_per_partition}"
                        )
                    tensor = self._split_weight(tensor, dim=0)
            if (
                param_name != "weight"
                and tensor.shape != param.shape
                and tensor.numel() == param.numel()
            ):
                tensor = tensor.reshape(param.shape)
            if tensor.shape != param.shape:
                if (
                    param_name == "weight"
                    and tensor.dim() == 2
                    and tensor.t().shape == param.shape
                ):
                    tensor = tensor.t().contiguous()
                else:
                    raise ValueError(
                        f"Shape mismatch for {self.prefix}.{param_name}: "
                        f"weight {tensor.shape} vs param {param.shape}"
                    )
            _copy_checked(param.data, tensor, f"{self.prefix}.{param_name}")
            self._record_parameter_loaded(param_name)

        # process_weights_after_loading is invoked by NewModelLoader's
        # post-load hook after every shard has landed; see LinearBase.

    def _split_weight(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if self.tp_size <= 1:
            return tensor
        _require_divisible(tensor.shape[dim], self.tp_size, "checkpoint shard")
        size_per_partition = tensor.shape[dim] // self.tp_size
        start = self.tp_rank * size_per_partition
        return tensor.narrow(dim, start, size_per_partition).contiguous()

    def _get_param_name(self, weight_name: str) -> Optional[str]:
        parts = weight_name.rsplit(".", 1)
        name = parts[-1] if len(parts) > 1 else weight_name
        return name


class RowParallelLinear(LinearBase):

    shard_names: List[str] = []

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        bias: bool = False,
        reduce_output: bool = True,
        params_dtype: torch.dtype = torch.float16,
        **kwargs,
    ):
        _validate_parallel_partition(tp_size, tp_rank, "TP")
        if not isinstance(reduce_output, bool):
            raise TypeError("reduce_output must be a bool")
        if bias and tp_size > 1 and not reduce_output:
            raise ValueError(
                "RowParallelLinear with bias and TP>1 requires reduce_output=True "
                "so bias can be added once after the collective"
            )
        _require_divisible(input_size, tp_size, "input_size")
        self.input_size_per_partition = input_size // tp_size
        self.reduce_output = reduce_output

        super().__init__(
            input_size=self.input_size_per_partition,
            output_size=output_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=prefix,
            bias=bias,
            params_dtype=params_dtype,
            **kwargs,
        )

        self.quant_method.create_weights(
            layer=self,
            input_size=self.input_size_per_partition,
            output_size=output_size,
            params_dtype=params_dtype,
        )

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        for name, tensor in weights.items():
            param_name = self._get_param_name(name)
            if param_name is None:
                continue

            param = getattr(self, param_name, None)
            if param is None or not isinstance(param, nn.Parameter):
                raise RuntimeError(f"Unsupported tensor {self.prefix}.{name}")
            if param_name in self._loaded_parameter_names:
                raise RuntimeError(f"Duplicate checkpoint tensor {self.prefix}.{name}")

            if param_name == "weight":
                global_input = self.input_size_per_partition * self.tp_size
                expected = (self.output_size, global_input)
                if tuple(tensor.shape) == expected:
                    pass
                elif tuple(tensor.shape) == (global_input, self.output_size):
                    tensor = tensor.t().contiguous()
                else:
                    raise ValueError(
                        f"{self.prefix}.weight checkpoint shape must be {expected}, "
                        f"got {tuple(tensor.shape)}"
                    )
                tensor = self._split_weight(tensor, dim=1)
            elif param_name in ("weight_scale", "input_scale"):
                if tensor.numel() == param.numel():
                    tensor = tensor.reshape(param.shape)
                elif (
                    tensor.numel() > 1
                    and tensor.shape[-1] == self.input_size_per_partition * self.tp_size
                ):
                    tensor = self._split_weight(tensor, dim=-1)
            elif param_name == "weight_scale_inv":
                if self.tp_size > 1:
                    _, block_k = self._fp8_scale_block_size()
                    start_column = self.tp_rank * self.input_size_per_partition
                    if (
                        start_column % block_k != 0
                        or self.input_size_per_partition % block_k != 0
                    ):
                        raise ValueError(
                            f"{self.prefix}.weight_scale_inv TP input shard must "
                            f"align to FP8 block {block_k}: start={start_column}, "
                            f"columns={self.input_size_per_partition}"
                        )
                    tensor = self._split_weight(tensor, dim=1)
            if (
                param_name != "weight"
                and tensor.shape != param.shape
                and tensor.numel() == param.numel()
            ):
                tensor = tensor.reshape(param.shape)
            if tensor.shape != param.shape:
                if (
                    param_name == "weight"
                    and tensor.dim() == 2
                    and tensor.t().shape == param.shape
                ):
                    tensor = tensor.t().contiguous()
                else:
                    raise ValueError(
                        f"Shape mismatch for {self.prefix}.{param_name}: "
                        f"weight {tensor.shape} vs param {param.shape}"
                    )
            _copy_checked(param.data, tensor, f"{self.prefix}.{param_name}")
            self._record_parameter_loaded(param_name)

        # process_weights_after_loading is invoked by NewModelLoader's
        # post-load hook after every shard has landed; see LinearBase.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 0 or x.shape[-1] != self.input_size:
            raise ValueError(
                f"{self.prefix or type(self).__name__} expected input width "
                f"{self.input_size}, got {tuple(x.shape)}"
            )
        output = self.quant_method.apply(self, x, None)
        if self.reduce_output and self.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        if self.bias is not None:
            output = output + self.bias
        return output

    def _split_weight(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if self.tp_size <= 1:
            return tensor
        _require_divisible(tensor.shape[dim], self.tp_size, "checkpoint shard")
        size_per_partition = tensor.shape[dim] // self.tp_size
        start = self.tp_rank * size_per_partition
        return tensor.narrow(dim, start, size_per_partition).contiguous()

    def _get_param_name(self, weight_name: str) -> Optional[str]:
        parts = weight_name.rsplit(".", 1)
        return parts[-1] if len(parts) > 1 else weight_name


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        bias: bool = False,
        shard_names: Optional[List[str]] = None,
        params_dtype: torch.dtype = torch.float16,
        **kwargs,
    ):
        if shard_names:
            self.shard_names = list(shard_names)
        if not self.shard_names:
            raise ValueError("MergedColumnParallelLinear requires shard_names")
        _require_divisible(output_size, len(self.shard_names), "output_size")
        _require_divisible(
            output_size // len(self.shard_names), tp_size, "merged shard size"
        )
        self._loaded_weight_shards = set()
        self._loaded_parameter_shards: Dict[str, set] = {}
        self._fused_loaded_parameters = set()
        self._merged_per_tensor_scales: Dict[str, Dict[int, float]] = {}

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=prefix,
            bias=bias,
            params_dtype=params_dtype,
            **kwargs,
        )

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        num_shards = len(self.shard_names) if self.shard_names else 1
        shard_size = self.output_size_per_partition // num_shards

        for full_name, tensor in weights.items():
            param_name = self._get_param_name(full_name)
            shard_id = self._get_shard_id(full_name)
            if param_name == "bias" and self.bias is None:
                raise RuntimeError(f"Unexpected bias tensor for {self.prefix}")
            self._ensure_parameter_can_load(param_name, shard_id)

            if param_name != "weight" and param_name != "bias":
                param = getattr(self, param_name, None)
                if param is None or not isinstance(param, nn.Parameter):
                    raise RuntimeError(f"Unsupported tensor {self.prefix}.{full_name}")

                if shard_id < 0:
                    if param_name == "weight_scale_inv" and tensor.shape != param.shape:
                        block_n, _ = self._fp8_scale_block_size()
                        global_shard_rows = shard_size * self.tp_size
                        if global_shard_rows % block_n or shard_size % block_n:
                            raise ValueError(
                                f"Fused {self.prefix}.{param_name} shards must align "
                                f"to FP8 output block {block_n}: global_rows="
                                f"{global_shard_rows}, local_rows={shard_size}"
                            )
                        global_shard_blocks = global_shard_rows // block_n
                        local_shard_blocks = shard_size // block_n
                        expected_blocks = global_shard_blocks * num_shards
                        if tensor.dim() != 2 or tensor.shape[0] != expected_blocks:
                            raise ValueError(
                                f"Fused {self.prefix}.{param_name} must have "
                                f"{expected_blocks} output blocks, got "
                                f"{tuple(tensor.shape)}"
                            )
                        pieces = []
                        for index in range(num_shards):
                            start = (
                                index * global_shard_blocks
                                + self.tp_rank * local_shard_blocks
                            )
                            pieces.append(
                                tensor.narrow(0, start, local_shard_blocks).contiguous()
                            )
                        tensor = torch.cat(pieces, dim=0)
                    elif (
                        tensor.shape != param.shape
                        and tensor.dim() > 0
                        and tensor.shape[0] == shard_size * self.tp_size * num_shards
                    ):
                        global_shard_rows = shard_size * self.tp_size
                        pieces = []
                        for index in range(num_shards):
                            start = (
                                index * global_shard_rows + self.tp_rank * shard_size
                            )
                            pieces.append(
                                tensor.narrow(0, start, shard_size).contiguous()
                            )
                        tensor = torch.cat(pieces, dim=0)
                    if tensor.shape != param.shape:
                        if tensor.numel() == param.numel():
                            tensor = tensor.reshape(param.shape)
                        else:
                            raise ValueError(
                                f"Shape mismatch for fused {self.prefix}.{param_name}: "
                                f"got {tuple(tensor.shape)}, expected {tuple(param.shape)}"
                            )
                    _copy_checked(param.data, tensor, f"{self.prefix}.{param_name}")
                    self._record_fused_parameter_loaded(param_name)
                    continue

                if param_name == "weight_scale_inv":
                    # FP8 per-block scale grid [out_blocks, in_blocks]. Each
                    # shard (gate/up) ships its own grid; place this shard's
                    # block-rows at its block offset. shard_size is in weight
                    # rows, so convert through the quant config's N block.
                    block_n, _ = self._fp8_scale_block_size()
                    start_row = self.tp_rank * shard_size
                    if start_row % block_n != 0 or shard_size % block_n != 0:
                        raise ValueError(
                            f"{self.prefix}.{param_name} merged TP shard must align "
                            f"to FP8 block {block_n}: start={start_row}, "
                            f"rows={shard_size}"
                        )
                    blocks_per_shard = self._ceil_div(shard_size, block_n)
                    if (
                        self.tp_size > 1
                        and tensor.shape[0] == blocks_per_shard * self.tp_size
                    ):
                        start = self.tp_rank * blocks_per_shard
                        tensor = tensor.narrow(0, start, blocks_per_shard).contiguous()
                    offset = shard_id * blocks_per_shard
                    expected = (blocks_per_shard,) + tuple(param.shape[1:])
                    if tuple(tensor.shape) != expected:
                        raise ValueError(
                            f"Shape mismatch for merged {self.prefix}.{param_name} "
                            f"shard={shard_id}: got {tuple(tensor.shape)}, "
                            f"expected {expected}"
                        )
                    _copy_checked(
                        param.data[offset : offset + blocks_per_shard],
                        tensor,
                        f"{self.prefix}.{param_name}/{self.shard_names[shard_id]}",
                    )
                    self._record_parameter_shard_loaded(param_name, shard_id)
                    continue

                # Per-shard non-weight param (e.g. per-channel weight_scale).
                if param.numel() == 1:
                    if tensor.numel() != 1:
                        raise ValueError(
                            f"{self.prefix}.{param_name}/{self.shard_names[shard_id]} "
                            "must be scalar"
                        )
                    scales = self._merged_per_tensor_scales.setdefault(param_name, {})
                    if shard_id in scales:
                        raise RuntimeError(
                            f"Duplicate {self.prefix}.{self.shard_names[shard_id]}."
                            f"{param_name} shard"
                        )
                    scales[shard_id] = float(tensor.flatten()[0].item())
                    if len(scales) == len(self.shard_names):
                        self._record_parameter_loaded(param_name)
                    continue

                # Per-channel/per-shard tensor: TP-slice + offset copy along dim 0.
                # Convention: ckpt tensor's dim-0 covers one shard's full output
                # (= shard_size * tp_size).
                if self.tp_size > 1 and tensor.shape[0] == shard_size * self.tp_size:
                    start = self.tp_rank * shard_size
                    tensor = tensor.narrow(0, start, shard_size).contiguous()
                offset = shard_id * shard_size
                if tensor.shape[0] != shard_size:
                    raise ValueError(
                        f"Shape mismatch for merged {self.prefix}.{param_name} "
                        f"shard={shard_id}: got dim-0={tensor.shape[0]}, "
                        f"expected {shard_size}"
                    )
                target = param.data[offset : offset + shard_size]
                if tuple(tensor.shape) != tuple(target.shape):
                    if tensor.numel() == target.numel():
                        tensor = tensor.reshape(target.shape)
                    else:
                        raise ValueError(
                            f"Shape mismatch for merged {self.prefix}.{param_name} "
                            f"shard={shard_id}: got {tuple(tensor.shape)}, "
                            f"expected {tuple(target.shape)}"
                        )
                _copy_checked(
                    target,
                    tensor,
                    f"{self.prefix}.{param_name}/{self.shard_names[shard_id]}",
                )
                self._record_parameter_shard_loaded(param_name, shard_id)
                continue

            if shard_id < 0:
                if param_name in ("weight", "bias"):
                    expected_rows = shard_size * self.tp_size * num_shards
                    if (
                        param_name == "weight"
                        and tensor.dim() == 2
                        and tuple(tensor.shape) != (expected_rows, self.input_size)
                        and tuple(tensor.shape) == (self.input_size, expected_rows)
                    ):
                        tensor = tensor.t().contiguous()
                    if tensor.shape[0] % num_shards != 0:
                        raise ValueError(
                            f"Fused {self.prefix}.{param_name} rows "
                            f"{tensor.shape[0]} must divide into {num_shards} shards"
                        )
                    global_shard_size = tensor.shape[0] // num_shards
                    _require_divisible(
                        global_shard_size, self.tp_size, "fused shard rows"
                    )
                    local_shard_size = global_shard_size // self.tp_size
                    pieces = []
                    for index in range(num_shards):
                        start = (
                            index * global_shard_size + self.tp_rank * local_shard_size
                        )
                        pieces.append(
                            tensor.narrow(0, start, local_shard_size).contiguous()
                        )
                    split_tensor = torch.cat(pieces, dim=0)
                if param_name == "weight":
                    if (
                        split_tensor.shape != self.weight.shape
                        and split_tensor.t().shape == self.weight.shape
                    ):
                        split_tensor = split_tensor.t().contiguous()
                    _copy_checked(
                        self.weight.data, split_tensor, f"{self.prefix}.weight"
                    )
                    self._record_fused_parameter_loaded("weight")
                elif param_name == "bias" and self.bias is not None:
                    if split_tensor.shape != self.bias.shape:
                        raise ValueError(
                            f"Shape mismatch for fused {self.prefix}.bias: "
                            f"got {tuple(split_tensor.shape)}, "
                            f"expected {tuple(self.bias.shape)}"
                        )
                    _copy_checked(self.bias.data, split_tensor, f"{self.prefix}.bias")
                    self._record_fused_parameter_loaded("bias")
                continue

            if param_name == "weight":
                global_shard = shard_size * self.tp_size
                expected = (global_shard, self.input_size)
                if tuple(tensor.shape) == expected:
                    pass
                elif tuple(tensor.shape) == (self.input_size, global_shard):
                    tensor = tensor.t().contiguous()
                else:
                    raise ValueError(
                        f"{self.prefix}.{self.shard_names[shard_id]}.weight rows "
                        f"must be {global_shard}, got {tensor.shape[0]}"
                    )
            if self.tp_size > 1:
                _require_divisible(
                    tensor.shape[0], self.tp_size, "checkpoint merged shard"
                )
                shard_output_size = tensor.shape[0] // self.tp_size
                start = self.tp_rank * shard_output_size
                tensor = tensor.narrow(0, start, shard_output_size).contiguous()

            if param_name == "weight":
                offset = shard_id * shard_size
                expected = tuple(
                    self.weight.data[offset : offset + shard_size, :].shape
                )
                if tuple(tensor.shape) != expected:
                    raise ValueError(
                        f"Shape mismatch for {self.prefix}."
                        f"{self.shard_names[shard_id]}.weight: "
                        f"got {tuple(tensor.shape)}, expected {expected}"
                    )
                _copy_checked(
                    self.weight.data[offset : offset + shard_size, :],
                    tensor,
                    f"{self.prefix}.{self.shard_names[shard_id]}.weight",
                )
                self._loaded_weight_shards.add(shard_id)
                self._record_parameter_shard_loaded("weight", shard_id)
            elif param_name == "bias" and self.bias is not None:
                offset = shard_id * shard_size
                if tuple(tensor.shape) != (shard_size,):
                    raise ValueError(
                        f"Shape mismatch for {self.prefix}."
                        f"{self.shard_names[shard_id]}.bias: got {tuple(tensor.shape)}, "
                        f"expected {(shard_size,)}"
                    )
                _copy_checked(
                    self.bias.data[offset : offset + shard_size],
                    tensor,
                    f"{self.prefix}.{self.shard_names[shard_id]}.bias",
                )
                self._record_parameter_shard_loaded("bias", shard_id)

        # process_weights_after_loading is invoked by NewModelLoader's
        # post-load hook after every shard has landed; see LinearBase.

    def _get_shard_id(self, weight_name: str) -> int:
        shard_name = _projection_name(weight_name, self.shard_names)
        return self.shard_names.index(shard_name) if shard_name is not None else -1

    def _record_fused_parameter_loaded(self, parameter_name: str) -> None:
        if self._loaded_parameter_shards.get(parameter_name):
            raise RuntimeError(
                f"Cannot mix fused and per-shard {self.prefix}.{parameter_name}"
            )
        self._fused_loaded_parameters.add(parameter_name)
        self._record_parameter_loaded(parameter_name)

    def _ensure_parameter_can_load(self, parameter_name: str, shard_id: int) -> None:
        if shard_id < 0:
            if parameter_name in self._fused_loaded_parameters:
                raise RuntimeError(
                    f"Duplicate checkpoint tensor {self.prefix}.{parameter_name}"
                )
            if self._loaded_parameter_shards.get(parameter_name):
                raise RuntimeError(
                    f"Cannot mix fused and per-shard {self.prefix}.{parameter_name}"
                )
            return

        if parameter_name in self._fused_loaded_parameters:
            raise RuntimeError(
                f"Cannot mix fused and per-shard {self.prefix}.{parameter_name}"
            )
        loaded = self._loaded_parameter_shards.get(parameter_name, set())
        scalar_scales = self._merged_per_tensor_scales.get(parameter_name, {})
        if shard_id in loaded or shard_id in scalar_scales:
            raise RuntimeError(
                f"Duplicate {self.prefix}.{self.shard_names[shard_id]}."
                f"{parameter_name} shard"
            )

    def _record_parameter_shard_loaded(
        self, parameter_name: str, shard_id: int
    ) -> None:
        if parameter_name in self._fused_loaded_parameters:
            raise RuntimeError(
                f"Cannot mix fused and per-shard {self.prefix}.{parameter_name}"
            )
        loaded = self._loaded_parameter_shards.setdefault(parameter_name, set())
        if shard_id in loaded:
            raise RuntimeError(
                f"Duplicate {self.prefix}.{self.shard_names[shard_id]}."
                f"{parameter_name} shard"
            )
        loaded.add(shard_id)
        if len(loaded) == len(self.shard_names):
            self._record_parameter_loaded(parameter_name)

    def process_weights_after_loading(self):
        if getattr(self, "_post_load_done", False):
            return
        self._merge_per_tensor_scales()
        self.quant_method.process_weights_after_loading(self)
        self._post_load_done = True

    def _merge_per_tensor_scales(self) -> None:
        expected_shards = set(range(len(self.shard_names)))
        for parameter_name, scales in self._merged_per_tensor_scales.items():
            if set(scales) != expected_shards:
                missing = expected_shards - set(scales)
                raise RuntimeError(
                    f"{self.prefix}.{parameter_name} is missing shards "
                    f"{[self.shard_names[index] for index in sorted(missing)]}"
                )
            parameter = getattr(self, parameter_name, None)
            if not isinstance(parameter, nn.Parameter) or parameter.numel() != 1:
                raise RuntimeError(
                    f"{self.prefix}.{parameter_name} is not a scalar parameter"
                )
            invalid = {
                shard_id: scale
                for shard_id, scale in scales.items()
                if scale <= 0 or not math.isfinite(scale)
            }
            if invalid:
                raise ValueError(
                    f"{self.prefix}.{parameter_name} has invalid shard scales {invalid}"
                )
            max_scale = max(scales.values())
            parameter.data.fill_(max_scale)
            if parameter_name == "weight_scale":
                shard_size = self.output_size_per_partition // len(self.shard_names)
                for shard_id, scale in scales.items():
                    if scale == max_scale:
                        continue
                    offset = shard_id * shard_size
                    weight = self.weight.data[offset : offset + shard_size]
                    weight.copy_(
                        (weight.float() * (scale / max_scale)).to(weight.dtype)
                    )
        self._merged_per_tensor_scales.clear()


class QKVParallelLinear(ColumnParallelLinear):

    shard_names: List[str] = ["q_proj", "k_proj", "v_proj"]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        bias: bool = False,
        params_dtype: torch.dtype = torch.float16,
        **kwargs,
    ):
        _validate_parallel_partition(tp_size, tp_rank, "TP")
        _require_positive_int(hidden_size, "hidden_size")
        _require_positive_int(num_heads, "num_heads")
        _require_positive_int(num_kv_heads, "num_kv_heads")
        _require_positive_int(head_dim, "head_dim")
        _require_divisible(num_heads, num_kv_heads, "num_heads")
        _require_divisible(num_heads, tp_size, "num_heads")
        if num_kv_heads % tp_size != 0 and tp_size % num_kv_heads != 0:
            raise ValueError(
                f"num_kv_heads={num_kv_heads} and tp_size={tp_size} must be "
                "mutually divisible so rank-local Q/KV head groups stay aligned"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.num_heads_per_partition = num_heads // tp_size
        kv_tp_size = math.gcd(num_kv_heads, tp_size)
        self.num_kv_heads_per_partition = num_kv_heads // kv_tp_size
        kv_replicas = tp_size // kv_tp_size
        self.kv_head_rank = tp_rank // kv_replicas
        _require_divisible(
            self.num_heads_per_partition,
            self.num_kv_heads_per_partition,
            "local num_heads",
        )

        self.q_size = self.num_heads_per_partition * head_dim
        self.kv_size = self.num_kv_heads_per_partition * head_dim

        total_output = self.q_size + 2 * self.kv_size

        LinearBase.__init__(
            self,
            input_size=hidden_size,
            output_size=total_output,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=prefix,
            bias=bias,
            params_dtype=params_dtype,
            **kwargs,
        )
        self.output_size_per_partition = total_output
        self.gather_output = False
        self._loaded_weight_shards = set()
        self._loaded_parameter_shards: Dict[str, set] = {}
        self._fused_loaded_parameters = set()

        self.quant_method.create_weights(
            layer=self,
            input_size=hidden_size,
            output_size=total_output,
            params_dtype=params_dtype,
        )

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        # Streaming-dispatch contract: this method may be called many times
        # with a single q/k/v shard at a time. Each call writes the shard
        # into its own offset slice in self.weight (and self.bias, etc.) so
        # that partial calls are safe — there's no "wait for all three" gate.
        # NewModelLoader._run_post_load_hooks invokes
        # process_weights_after_loading after every shard has landed; see
        # LinearBase.
        for full_name, tensor in weights.items():
            param_name = self._get_param_name(full_name)
            if param_name is None:
                continue

            projection = _projection_name(full_name, self.shard_names)
            qkv_key = {
                "q_proj": "q",
                "k_proj": "k",
                "v_proj": "v",
            }.get(projection)

            if qkv_key is None:
                # A fused checkpoint must be split into Q/K/V before applying
                # rank-local head selection; slicing the whole axis would mix
                # Q rows with K/V rows under TP.
                if param_name in ("weight", "bias"):
                    q_rows = self.num_heads * self.head_dim
                    kv_rows = self.num_kv_heads * self.head_dim
                    total_rows = q_rows + 2 * kv_rows
                    if (
                        param_name == "weight"
                        and tuple(tensor.shape) != (total_rows, self.hidden_size)
                        and tuple(tensor.shape) == (self.hidden_size, total_rows)
                    ):
                        tensor = tensor.t().contiguous()
                    if tensor.shape[0] != total_rows:
                        raise ValueError(
                            f"Fused QKV rows {tensor.shape[0]} do not match "
                            f"{total_rows}"
                        )
                    q, k, v = torch.split(tensor, [q_rows, kv_rows, kv_rows], dim=0)
                    for key, shard in zip(("q", "k", "v"), (q, k, v)):
                        self._dispatch_qkv_shard(key, param_name, shard)
                else:
                    param = getattr(self, param_name, None)
                    if not isinstance(param, nn.Parameter):
                        raise RuntimeError(
                            f"Unsupported tensor {self.prefix}.{full_name}"
                        )
                    if param.numel() == 1:
                        if tensor.numel() != 1:
                            raise ValueError(
                                f"Fused {self.prefix}.{param_name} must be scalar"
                            )
                        _copy_checked(
                            param.data,
                            tensor.reshape(param.shape),
                            f"{self.prefix}.{param_name}",
                        )
                        self._record_fused_parameter_loaded(param_name)
                    elif param_name == "weight_scale_inv":
                        block_n, _ = self._fp8_scale_block_size()
                        q_rows = self.num_heads * self.head_dim
                        kv_rows = self.num_kv_heads * self.head_dim
                        if q_rows % block_n or kv_rows % block_n:
                            raise ValueError(
                                f"Fused {self.prefix}.weight_scale_inv Q/K/V rows "
                                f"must align to FP8 block {block_n}"
                            )
                        q_blocks, kv_blocks = q_rows // block_n, kv_rows // block_n
                        if tensor.shape[0] != q_blocks + 2 * kv_blocks:
                            raise ValueError(
                                f"Fused {self.prefix}.weight_scale_inv rows "
                                f"{tensor.shape[0]} do not match "
                                f"{q_blocks + 2 * kv_blocks}"
                            )
                        q, k, v = torch.split(
                            tensor, [q_blocks, kv_blocks, kv_blocks], dim=0
                        )
                        for key, shard in zip(("q", "k", "v"), (q, k, v)):
                            self._dispatch_qkv_shard(key, param_name, shard)
                    elif (
                        tensor.dim() > 0
                        and tensor.shape[0]
                        == (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
                    ):
                        q_rows = self.num_heads * self.head_dim
                        kv_rows = self.num_kv_heads * self.head_dim
                        q, k, v = torch.split(tensor, [q_rows, kv_rows, kv_rows], dim=0)
                        for key, shard in zip(("q", "k", "v"), (q, k, v)):
                            self._dispatch_qkv_shard(key, param_name, shard)
                    else:
                        raise ValueError(
                            f"Unsupported fused layout for {self.prefix}.{param_name}: "
                            f"got {tuple(tensor.shape)}, expected {tuple(param.shape)}"
                        )
                continue

            self._dispatch_qkv_shard(qkv_key, param_name, tensor)

    def _dispatch_qkv_shard(self, qkv_key: str, param_name: str, tensor: torch.Tensor):
        """Write a single q/k/v shard into the merged parameter at its offset."""
        # Resolve per-shard size and offset in the merged QKV layout.
        if qkv_key == "q":
            num_heads, size, offset = self.num_heads, self.q_size, 0
        elif qkv_key == "k":
            num_heads, size, offset = self.num_kv_heads, self.kv_size, self.q_size
        else:  # "v"
            num_heads, size, offset = (
                self.num_kv_heads,
                self.kv_size,
                self.q_size + self.kv_size,
            )

        if param_name == "weight":
            self._ensure_parameter_shard_not_loaded(param_name, qkv_key)
            expected_rows = num_heads * self.head_dim
            expected = (expected_rows, self.hidden_size)
            if tuple(tensor.shape) == expected:
                pass
            elif tuple(tensor.shape) == (self.hidden_size, expected_rows):
                tensor = tensor.t().contiguous()
            else:
                raise ValueError(
                    f"{self.prefix}.{qkv_key}.weight shape must be "
                    f"{(expected_rows, self.hidden_size)}, got {tuple(tensor.shape)}"
                )
            split = self._split_qkv(tensor, num_heads, self.head_dim, qkv_key)
            _copy_checked(
                self.weight.data[offset : offset + size],
                split,
                f"{self.prefix}.{qkv_key}.weight",
            )
            self._loaded_weight_shards.add(qkv_key)
            self._record_parameter_shard_loaded(param_name, qkv_key)
            return

        if param_name == "bias":
            if self.bias is None:
                raise RuntimeError(f"Unexpected bias tensor for {self.prefix}")
            self._ensure_parameter_shard_not_loaded(param_name, qkv_key)
            split = self._split_qkv(tensor, num_heads, self.head_dim, qkv_key)
            _copy_checked(
                self.bias.data[offset : offset + size],
                split,
                f"{self.prefix}.{qkv_key}.bias",
            )
            self._record_parameter_shard_loaded(param_name, qkv_key)
            return

        # Auxiliary per-shard params (per-channel weight_scale, input_scale, ...).
        param = getattr(self, param_name, None)
        if param is None or not isinstance(param, nn.Parameter):
            raise RuntimeError(
                f"Unsupported tensor {self.prefix}.{qkv_key}.{param_name}"
            )

        if param_name == "weight_scale_inv":
            # FP8 per-block grid for this q/k/v shard: TP-slice along the
            # output(head)-block dim, then place at the shard's block offset
            # in the merged [total_blocks, in_blocks] grid.
            block_n, _ = self._fp8_scale_block_size()
            rank = self.tp_rank if qkv_key == "q" else self.kv_head_rank
            self._ensure_parameter_shard_not_loaded(param_name, qkv_key)
            if self.tp_size > 1:
                heads_pp = (
                    self.num_heads_per_partition
                    if qkv_key == "q"
                    else self.num_kv_heads_per_partition
                )
                rows = heads_pp * self.head_dim
                start_row = rank * rows
                if start_row % block_n != 0 or rows % block_n != 0:
                    raise ValueError(
                        f"{self.prefix}.{qkv_key}.weight_scale_inv TP shard "
                        f"must align to FP8 output block {block_n}: "
                        f"start={start_row}, rows={rows}"
                    )
                start_blk = start_row // block_n
                tensor = tensor.narrow(0, start_blk, rows // block_n).contiguous()
            if offset % block_n != 0:
                raise ValueError(
                    f"{self.prefix}.{qkv_key}.weight_scale_inv offset {offset} "
                    f"must align to FP8 output block {block_n}"
                )
            off_blk = offset // block_n
            target = param.data[off_blk : off_blk + tensor.shape[0]]
            if tuple(target.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"Shape mismatch for {self.prefix}.{qkv_key}.weight_scale_inv: "
                    f"got {tuple(tensor.shape)}, expected {tuple(target.shape)}"
                )
            _copy_checked(
                param.data[off_blk : off_blk + tensor.shape[0]],
                tensor,
                f"{self.prefix}.{qkv_key}.{param_name}",
            )
            self._record_parameter_shard_loaded(param_name, qkv_key)
            return

        if param.numel() == 1:
            # Per-tensor scalar: collect per-shard scales now and merge in
            # process_weights_after_loading. Naive max-merge here is wrong for
            # `weight_scale` because each q/k/v shard's fp8 weights are stored
            # in the representation of its OWN ckpt scale; if the merged scale
            # is the max across shards, the smaller-scale shards' fp8 values
            # decode to magnitudes inflated by max/own — silently corrupt.
            # The original loader's `merge_qkv_hf_fp8_with_scale` rescales each
            # shard's weight to share max_scale; we mirror that in post-load.
            if tensor.numel() != 1:
                raise ValueError(f"{self.prefix}.{qkv_key}.{param_name} must be scalar")
            val = float(tensor.reshape(-1)[0].item())
            if not hasattr(self, "_qkv_per_tensor_scales"):
                self._qkv_per_tensor_scales = {}
            scales = self._qkv_per_tensor_scales.setdefault(param_name, {})
            if qkv_key in scales:
                raise RuntimeError(
                    f"Duplicate {self.prefix}.{qkv_key}.{param_name} shard"
                )
            scales[qkv_key] = val
            if set(scales) == {"q", "k", "v"}:
                self._record_parameter_loaded(param_name)
            return

        # Per-channel along output dim: TP-slice and write into [offset:offset+size).
        split = self._split_qkv(tensor, num_heads, self.head_dim, qkv_key)
        if split.shape[0] != size:
            raise ValueError(
                f"Shape mismatch for QKV shard {self.prefix}.{param_name}/{qkv_key}: "
                f"got dim-0={split.shape[0]}, expected {size}"
            )
        self._ensure_parameter_shard_not_loaded(param_name, qkv_key)
        target = param.data[offset : offset + size]
        if tuple(target.shape) != tuple(split.shape):
            if split.numel() == target.numel():
                split = split.reshape(target.shape)
            else:
                raise ValueError(
                    f"Shape mismatch for QKV shard {self.prefix}.{param_name}/"
                    f"{qkv_key}: got {tuple(split.shape)}, "
                    f"expected {tuple(target.shape)}"
                )
        _copy_checked(target, split, f"{self.prefix}.{qkv_key}.{param_name}")
        self._record_parameter_shard_loaded(param_name, qkv_key)

    def process_weights_after_loading(self):
        # Merge per-tensor q/k/v scales and rescale fp8 weights so all three
        # shards share the same final scale. Must run BEFORE the quant_method
        # post-load hook (which may rebind weight/scale Parameters and freeze
        # dtypes), so we override here instead of letting LinearBase handle it.
        if getattr(self, "_post_load_done", False):
            return
        self._merge_qkv_per_tensor_scales()
        self.quant_method.process_weights_after_loading(self)
        self._post_load_done = True

    def _merge_qkv_per_tensor_scales(self):
        scales_by_param = getattr(self, "_qkv_per_tensor_scales", None)
        if not scales_by_param:
            return

        for param_name, scales in scales_by_param.items():
            param = getattr(self, param_name, None)
            if not isinstance(param, nn.Parameter) or param.numel() != 1:
                raise RuntimeError(
                    f"{self.prefix}.{param_name} is not a scalar parameter"
                )
            missing = {"q", "k", "v"} - set(scales)
            if missing:
                raise RuntimeError(
                    f"{self.prefix}.{param_name} is missing QKV shards "
                    f"{sorted(missing)}"
                )
            invalid = {
                shard_name: scale
                for shard_name, scale in scales.items()
                if scale <= 0 or not math.isfinite(scale)
            }
            if invalid:
                raise ValueError(
                    f"{self.prefix}.{param_name} has invalid shard scales {invalid}"
                )
            max_scale = max(scales.values())
            param.data.fill_(max_scale)
            # input_scale doesn't index into self.weight; only weight_scale
            # requires per-shard fp8 rescale.
            if param_name != "weight_scale":
                continue
            for qkv_key, scale_val in scales.items():
                if scale_val == max_scale:
                    continue
                offset, size = self._qkv_shard_offset_size(qkv_key)
                weight_slice = self.weight.data[offset : offset + size]
                ratio = scale_val / max_scale
                rescaled = (weight_slice.float() * ratio).to(weight_slice.dtype)
                self.weight.data[offset : offset + size].copy_(rescaled)

        del self._qkv_per_tensor_scales

    def _ensure_parameter_shard_not_loaded(
        self, parameter_name: str, qkv_key: str
    ) -> None:
        if parameter_name in self._fused_loaded_parameters:
            raise RuntimeError(
                f"Cannot mix fused and per-shard {self.prefix}.{parameter_name}"
            )
        if qkv_key in self._loaded_parameter_shards.get(parameter_name, set()):
            raise RuntimeError(
                f"Duplicate {self.prefix}.{qkv_key}.{parameter_name} shard"
            )

    def _record_parameter_shard_loaded(self, parameter_name: str, qkv_key: str) -> None:
        loaded = self._loaded_parameter_shards.setdefault(parameter_name, set())
        loaded.add(qkv_key)
        if loaded == {"q", "k", "v"}:
            self._record_parameter_loaded(parameter_name)

    def _record_fused_parameter_loaded(self, parameter_name: str) -> None:
        if self._loaded_parameter_shards.get(parameter_name):
            raise RuntimeError(
                f"Cannot mix fused and per-shard {self.prefix}.{parameter_name}"
            )
        self._fused_loaded_parameters.add(parameter_name)
        self._record_parameter_loaded(parameter_name)

    def _qkv_shard_offset_size(self, qkv_key: str):
        if qkv_key == "q":
            return 0, self.q_size
        if qkv_key == "k":
            return self.q_size, self.kv_size
        return self.q_size + self.kv_size, self.kv_size

    def _split_qkv(
        self,
        tensor: torch.Tensor,
        num_heads: int,
        head_dim: int,
        qkv_key: str,
    ) -> torch.Tensor:
        expected_rows = num_heads * head_dim
        if tensor.shape[0] != expected_rows:
            raise ValueError(
                f"{self.prefix}.{qkv_key} rows must be {expected_rows}, "
                f"got {tensor.shape[0]}"
            )
        if self.tp_size <= 1:
            return tensor
        if qkv_key == "q":
            heads_per_partition = self.num_heads_per_partition
            rank = self.tp_rank
        else:
            heads_per_partition = self.num_kv_heads_per_partition
            rank = self.kv_head_rank
        size_per_partition = heads_per_partition * head_dim
        start = rank * size_per_partition
        return tensor.narrow(0, start, size_per_partition).contiguous()

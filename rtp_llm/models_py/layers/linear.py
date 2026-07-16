from collections import defaultdict
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods.base import QuantizationConfig


def _validate_parallel_partition(size: int, rank: int, label: str) -> None:
    if size <= 0 or not 0 <= rank < size:
        raise ValueError(f"Invalid {label} partition: rank={rank}, size={size}")


def _require_divisible(value: int, divisor: int, label: str) -> None:
    if value % divisor != 0:
        raise ValueError(f"{label}={value} must be divisible by {divisor}")


def _parameter_name(weight_name: str) -> str:
    return weight_name.rsplit(".", 1)[-1]


def _projection_name(weight_name: str, shard_names: List[str]) -> Optional[str]:
    parts = weight_name.split(".")
    if len(parts) < 2:
        return None
    candidate = parts[-2]
    return candidate if candidate in shard_names else None


class LinearBase(RtpModule):
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
        params_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        _validate_parallel_partition(tp_size, tp_rank, "TP")
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.prefix = prefix
        self.params_dtype = params_dtype
        self.quant_config = quant_config or QuantizationConfig("none")
        self.quant_method = self.quant_config.get_quant_method(self, prefix)
        if bias:
            self.bias = nn.Parameter(
                torch.empty(output_size, dtype=params_dtype), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

    def _copy_parameter(self, name: str, tensor: torch.Tensor) -> None:
        if not self._assign_weight(self, name, tensor):
            raise RuntimeError(f"Unknown parameter {self.prefix}.{name}")

    def _copy_parameter_slice(
        self, name: str, offset: int, tensor: torch.Tensor
    ) -> None:
        target = getattr(self, name, None)
        if not isinstance(target, nn.Parameter):
            raise RuntimeError(f"Unknown parameter {self.prefix}.{name}")
        target_slice = target[offset : offset + tensor.shape[0]]
        if tuple(target_slice.shape) != tuple(tensor.shape):
            raise ValueError(
                f"Shape mismatch for {self.prefix}.{name} shard: expected "
                f"{tuple(target_slice.shape)}, got {tuple(tensor.shape)}"
            )
        if target_slice.dtype != tensor.dtype and not (
            target_slice.is_floating_point() and tensor.is_floating_point()
        ):
            raise TypeError(
                f"Dtype mismatch for {self.prefix}.{name} shard: expected "
                f"{target_slice.dtype}, got {tensor.dtype}"
            )
        with torch.no_grad():
            target_slice.copy_(tensor)

    def process_weights_after_loading(self) -> None:
        if getattr(self, "_post_load_done", False):
            return
        self.quant_method.process_weights_after_loading(self)
        self._post_load_done = True

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim == 0 or x.shape[-1] != self.input_size:
            raise ValueError(
                f"{self.prefix or type(self).__name__} expected input width "
                f"{self.input_size}, got shape {tuple(x.shape)}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        return self.quant_method.apply(self, x, self.bias)


class ColumnParallelLinear(LinearBase):
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
    ):
        _require_divisible(output_size, tp_size, "output_size")
        self.output_size_per_partition = output_size // tp_size
        self.gather_output = gather_output
        super().__init__(
            input_size,
            self.output_size_per_partition,
            tp_size,
            tp_rank,
            quant_config,
            prefix,
            bias,
            params_dtype,
        )
        self.quant_method.create_weights(
            self, input_size, self.output_size_per_partition, params_dtype
        )

    def _split_output(self, tensor: torch.Tensor) -> torch.Tensor:
        _require_divisible(tensor.shape[0], self.tp_size, "checkpoint output")
        rows = tensor.shape[0] // self.tp_size
        return tensor.narrow(0, self.tp_rank * rows, rows).contiguous()

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        for name, tensor in weights.items():
            parameter_name = _parameter_name(name)
            if parameter_name not in ("weight", "bias"):
                raise RuntimeError(f"Unsupported tensor {self.prefix}.{name}")
            if parameter_name == "bias" and self.bias is None:
                raise RuntimeError(f"Unexpected bias tensor {self.prefix}.{name}")
            self._copy_parameter(parameter_name, self._split_output(tensor))


class RowParallelLinear(LinearBase):
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
    ):
        _require_divisible(input_size, tp_size, "input_size")
        self.input_size_per_partition = input_size // tp_size
        self.reduce_output = reduce_output
        super().__init__(
            self.input_size_per_partition,
            output_size,
            tp_size,
            tp_rank,
            quant_config,
            prefix,
            bias,
            params_dtype,
        )
        self.quant_method.create_weights(
            self, self.input_size_per_partition, output_size, params_dtype
        )

    def _split_input(self, tensor: torch.Tensor) -> torch.Tensor:
        _require_divisible(tensor.shape[1], self.tp_size, "checkpoint input")
        columns = tensor.shape[1] // self.tp_size
        return tensor.narrow(1, self.tp_rank * columns, columns).contiguous()

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        for name, tensor in weights.items():
            parameter_name = _parameter_name(name)
            if parameter_name == "weight":
                self._copy_parameter("weight", self._split_input(tensor))
            elif parameter_name == "bias" and self.bias is not None:
                self._copy_parameter("bias", tensor)
            else:
                raise RuntimeError(f"Unsupported tensor {self.prefix}.{name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        output = self.quant_method.apply(self, x, None)
        if self.reduce_output and self.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        if self.bias is not None:
            output = output + self.bias
        return output


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
    ):
        self.shard_names = list(shard_names or [])
        if not self.shard_names:
            raise ValueError("MergedColumnParallelLinear requires shard_names")
        _require_divisible(output_size, len(self.shard_names), "output_size")
        global_shard_size = output_size // len(self.shard_names)
        _require_divisible(global_shard_size, tp_size, "merged shard size")
        self.global_shard_size = global_shard_size
        self.local_shard_size = global_shard_size // tp_size
        self._loaded_shards: Dict[str, Set[str]] = defaultdict(set)
        super().__init__(
            input_size,
            output_size,
            tp_size,
            tp_rank,
            quant_config,
            prefix,
            bias,
            False,
            params_dtype,
        )

    def _mark_shard(self, parameter_name: str, shard_name: str) -> None:
        loaded = self._loaded_shards[parameter_name]
        if shard_name in loaded:
            raise RuntimeError(
                f"Duplicate {self.prefix}.{shard_name}.{parameter_name} shard"
            )
        loaded.add(shard_name)
        if loaded == set(self.shard_names):
            self._mark_weight_loaded(parameter_name)

    def _copy_fused(self, parameter_name: str, tensor: torch.Tensor) -> None:
        if self._loaded_shards[parameter_name]:
            raise RuntimeError(
                f"Cannot mix fused and per-shard {self.prefix}.{parameter_name}"
            )
        expected_rows = self.global_shard_size * len(self.shard_names)
        if tensor.shape[0] != expected_rows:
            raise ValueError(
                f"Fused {self.prefix}.{parameter_name} rows must be "
                f"{expected_rows}, got {tensor.shape[0]}"
            )
        pieces = []
        for index in range(len(self.shard_names)):
            start = (
                index * self.global_shard_size + self.tp_rank * self.local_shard_size
            )
            pieces.append(tensor.narrow(0, start, self.local_shard_size).contiguous())
        self._copy_parameter(parameter_name, torch.cat(pieces, dim=0))
        self._loaded_shards[parameter_name] = set(self.shard_names)

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        for name, tensor in weights.items():
            parameter_name = _parameter_name(name)
            if parameter_name not in ("weight", "bias"):
                raise RuntimeError(f"Unsupported tensor {self.prefix}.{name}")
            if parameter_name == "bias" and self.bias is None:
                raise RuntimeError(f"Unexpected bias tensor {self.prefix}.{name}")
            shard_name = _projection_name(name, self.shard_names)
            if shard_name is None:
                self._copy_fused(parameter_name, tensor)
                continue
            if self._loaded_shards[parameter_name] == set(self.shard_names):
                raise RuntimeError(
                    f"Cannot mix fused and per-shard {self.prefix}.{parameter_name}"
                )
            if shard_name in self._loaded_shards[parameter_name]:
                raise RuntimeError(
                    f"Duplicate {self.prefix}.{shard_name}.{parameter_name} shard"
                )
            if tensor.shape[0] != self.global_shard_size:
                raise ValueError(
                    f"{self.prefix}.{shard_name}.{parameter_name} rows must be "
                    f"{self.global_shard_size}, got {tensor.shape[0]}"
                )
            shard_index = self.shard_names.index(shard_name)
            start = self.tp_rank * self.local_shard_size
            local = tensor.narrow(0, start, self.local_shard_size).contiguous()
            offset = shard_index * self.local_shard_size
            self._copy_parameter_slice(parameter_name, offset, local)
            self._mark_shard(parameter_name, shard_name)

    def validate_weights_loaded(self, loaded_tensor_ids=None) -> None:
        for parameter_name in ("weight", "bias"):
            if parameter_name == "bias" and self.bias is None:
                continue
            missing = set(self.shard_names) - self._loaded_shards[parameter_name]
            if missing:
                raise RuntimeError(
                    f"{self.prefix}.{parameter_name} is missing shards "
                    f"{sorted(missing)}"
                )
        super().validate_weights_loaded(loaded_tensor_ids)


class QKVParallelLinear(ColumnParallelLinear):
    shard_names = ["q_proj", "k_proj", "v_proj"]

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
    ):
        _validate_parallel_partition(tp_size, tp_rank, "TP")
        if num_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0:
            raise ValueError("QKV head counts and head_dim must be positive")
        _require_divisible(num_heads, num_kv_heads, "num_heads")
        _require_divisible(num_heads, tp_size, "num_heads")
        if num_kv_heads >= tp_size:
            _require_divisible(num_kv_heads, tp_size, "num_kv_heads")
            self.num_kv_heads_per_partition = num_kv_heads // tp_size
            self.kv_head_rank = tp_rank
        else:
            _require_divisible(tp_size, num_kv_heads, "tp_size")
            self.num_kv_heads_per_partition = 1
            replicas = tp_size // num_kv_heads
            self.kv_head_rank = tp_rank // replicas
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_heads_per_partition = num_heads // tp_size
        self.q_size = self.num_heads_per_partition * head_dim
        self.kv_size = self.num_kv_heads_per_partition * head_dim
        self._loaded_shards: Dict[str, Set[str]] = defaultdict(set)

        LinearBase.__init__(
            self,
            hidden_size,
            self.q_size + 2 * self.kv_size,
            tp_size,
            tp_rank,
            quant_config,
            prefix,
            bias,
            params_dtype,
        )
        self.output_size_per_partition = self.output_size
        self.gather_output = False
        self.quant_method.create_weights(
            self, hidden_size, self.output_size, params_dtype
        )

    def _local_qkv(self, shard_name: str, tensor: torch.Tensor) -> torch.Tensor:
        if shard_name == "q_proj":
            start_head = self.tp_rank * self.num_heads_per_partition
            head_count = self.num_heads_per_partition
        else:
            start_head = self.kv_head_rank * self.num_kv_heads_per_partition
            head_count = self.num_kv_heads_per_partition
        start = start_head * self.head_dim
        rows = head_count * self.head_dim
        return tensor.narrow(0, start, rows).contiguous()

    def _offset(self, shard_name: str) -> int:
        if shard_name == "q_proj":
            return 0
        if shard_name == "k_proj":
            return self.q_size
        return self.q_size + self.kv_size

    def _mark_shard(self, parameter_name: str, shard_name: str) -> None:
        loaded = self._loaded_shards[parameter_name]
        if shard_name in loaded:
            raise RuntimeError(
                f"Duplicate {self.prefix}.{shard_name}.{parameter_name} shard"
            )
        loaded.add(shard_name)
        if loaded == set(self.shard_names):
            self._mark_weight_loaded(parameter_name)

    def _load_shard(
        self, shard_name: str, parameter_name: str, tensor: torch.Tensor
    ) -> None:
        if shard_name in self._loaded_shards[parameter_name]:
            raise RuntimeError(
                f"Duplicate {self.prefix}.{shard_name}.{parameter_name} shard"
            )
        expected_rows = (
            self.num_heads if shard_name == "q_proj" else self.num_kv_heads
        ) * self.head_dim
        if tensor.shape[0] != expected_rows:
            raise ValueError(
                f"{self.prefix}.{shard_name}.{parameter_name} rows must be "
                f"{expected_rows}, got {tensor.shape[0]}"
            )
        local = self._local_qkv(shard_name, tensor)
        offset = self._offset(shard_name)
        self._copy_parameter_slice(parameter_name, offset, local)
        self._mark_shard(parameter_name, shard_name)

    def _load_fused(self, parameter_name: str, tensor: torch.Tensor) -> None:
        if self._loaded_shards[parameter_name]:
            raise RuntimeError(
                f"Cannot mix fused and per-shard {self.prefix}.{parameter_name}"
            )
        q_rows = self.num_heads * self.head_dim
        kv_rows = self.num_kv_heads * self.head_dim
        if tensor.shape[0] != q_rows + 2 * kv_rows:
            raise ValueError(
                f"Fused QKV rows {tensor.shape[0]} do not match "
                f"{q_rows + 2 * kv_rows}"
            )
        q, k, v = torch.split(tensor, [q_rows, kv_rows, kv_rows], dim=0)
        for shard_name, shard in zip(self.shard_names, (q, k, v)):
            self._load_shard(shard_name, parameter_name, shard)

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        for name, tensor in weights.items():
            parameter_name = _parameter_name(name)
            if parameter_name not in ("weight", "bias"):
                raise RuntimeError(f"Unsupported tensor {self.prefix}.{name}")
            if parameter_name == "bias" and self.bias is None:
                raise RuntimeError(f"Unexpected bias tensor {self.prefix}.{name}")
            shard_name = _projection_name(name, self.shard_names)
            if shard_name is None:
                self._load_fused(parameter_name, tensor)
            else:
                if self._loaded_shards[parameter_name] == set(self.shard_names):
                    raise RuntimeError(
                        f"Cannot mix fused and per-shard "
                        f"{self.prefix}.{parameter_name}"
                    )
                self._load_shard(shard_name, parameter_name, tensor)

    def validate_weights_loaded(self, loaded_tensor_ids=None) -> None:
        for parameter_name in ("weight", "bias"):
            if parameter_name == "bias" and self.bias is None:
                continue
            missing = set(self.shard_names) - self._loaded_shards[parameter_name]
            if missing:
                raise RuntimeError(
                    f"{self.prefix}.{parameter_name} is missing shards "
                    f"{sorted(missing)}"
                )
        super().validate_weights_loaded(loaded_tensor_ids)

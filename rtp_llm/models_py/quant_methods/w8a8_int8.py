"""Module-owned compressed W8A8 INT8 weights for the existing PPU executor.

Checkpoint parameters stay row-major INT8 with FP32 per-output-channel scales.
No weight re-quantization or legacy DeviceImpl conversion is performed.
"""

import importlib

import torch
from torch import nn

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.module_base import copy_weight_
from rtp_llm.models_py.quant_methods.base import (
    QuantizeMethodBase,
    register_quant_method,
)
from rtp_llm.models_py.read_spec import (
    ReadSpec,
    make_sharded_read_spec,
    unwrap_checkpoint_tensor,
)

W8A8_METHOD = "W8A8_INT8_PER_CHANNEL_COMPRESSED"


def resolved_backend_symbol(module, names):
    """Match the reused PPU wrapper's first-present alias resolution exactly."""
    for name in names:
        value = getattr(module, name, None)
        if value is not None:
            return value
    return None


def require_ppu_device(device):
    device = torch.device(device)
    if device.type != "cuda" or get_device_type() != DeviceType.Ppu:
        raise RuntimeError(f"Compressed W8A8 runtime requires PPU, got {device}")


def _dense_executor_class():
    """Import the actual consumer, not just the presence of a DeepGEMM wheel."""
    path = "rtp_llm.models_py.modules.factory.linear.ppu_impl.int8_deepgemm_linear"
    try:
        module = importlib.import_module(path)
        cls = getattr(module, "PpuInt8DeepGemmLinear")
        if not callable(cls) or not callable(module.per_token_quant_int8):
            raise AttributeError("dense executor or activation quantizer is missing")
        if not module.has_deep_gemm_int8_dense():
            raise RuntimeError("PPU DeepGEMM dense INT8 GEMM symbol is missing")
        native = importlib.import_module("deep_gemm")
        if not callable(
            resolved_backend_symbol(
                native, ("gemm_int8_int8_bf16_nt", "gemm_nt_i8i8bf16")
            )
        ):
            raise RuntimeError("PPU DeepGEMM dense INT8 GEMM symbol is not callable")
        return cls
    except (ImportError, OSError, RuntimeError, AttributeError) as error:
        raise RuntimeError(f"PPU W8A8 dense backend is unavailable: {error}") from error


def checked_channel_scale(tensor, rows, label):
    if tensor.dtype != torch.float32:
        raise TypeError(f"{label} requires FP32 scales, got {tensor.dtype}")
    if tuple(tensor.shape) not in ((rows,), (rows, 1)):
        raise ValueError(
            f"{label} expected channel scale [{rows}, 1], got {tensor.shape}"
        )
    if not torch.isfinite(tensor).all() or bool((tensor <= 0).any()):
        raise ValueError(f"{label} scales must be finite and positive")
    return tensor.reshape(rows, 1)


class Int8CheckpointLinearMethod(QuantizeMethodBase):
    """Shared INT8 parameter layout and TP loading, independent of the executor."""

    owns_checkpoint_loading = True
    required_checkpoint_parameters = ("weight", "weight_scale")

    def __init__(self, quant_config=None):
        super().__init__(quant_config)
        self._pieces = {}

    def create_weights(self, layer, input_size, output_size, params_dtype):
        if params_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("W8A8 compute dtype must be FP16 or BF16")
        layer.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.int8), requires_grad=False
        )
        layer.weight_scale = nn.Parameter(
            torch.empty(output_size, 1, dtype=torch.float32), requires_grad=False
        )

    @staticmethod
    def _layout(layer):
        # Each item: name, global rows, local rows, source shard rank, target row.
        if hasattr(layer, "num_kv_heads"):
            return (
                (
                    "q_proj",
                    layer.num_heads * layer.head_dim,
                    layer.q_size,
                    layer.tp_rank,
                    0,
                ),
                (
                    "k_proj",
                    layer.num_kv_heads * layer.head_dim,
                    layer.kv_size,
                    layer.kv_head_rank,
                    layer.q_size,
                ),
                (
                    "v_proj",
                    layer.num_kv_heads * layer.head_dim,
                    layer.kv_size,
                    layer.kv_head_rank,
                    layer.q_size + layer.kv_size,
                ),
            )
        if layer.shard_names:
            local = layer.output_size_per_partition // len(layer.shard_names)
            return tuple(
                (name, local * layer.tp_size, local, layer.tp_rank, index * local)
                for index, name in enumerate(layer.shard_names)
            )
        local = layer.output_size
        row = hasattr(layer, "input_size_per_partition")
        return (
            (
                "weight",
                local if row else local * layer.tp_size,
                local,
                0 if row else layer.tp_rank,
                0,
            ),
        )

    @classmethod
    def _selection(cls, layer, name):
        parts = name.split(".")
        field = parts[-1]
        if field not in ("weight", "weight_scale", "bias"):
            raise ValueError(f"Unsupported compressed W8A8 tensor {name!r}")
        if field == "bias" and layer.bias is None:
            raise ValueError(f"Unexpected W8A8 bias {name!r}")
        layout = cls._layout(layer)
        projection = parts[-2] if len(parts) > 1 else None
        chosen = tuple(piece for piece in layout if piece[0] == projection)
        if projection is not None and not chosen:
            raise ValueError(f"Unknown compressed W8A8 projection {projection!r}")
        return field, chosen or layout

    def checkpoint_read_spec(self, layer, name, shape):
        field, pieces = self._selection(layer, name)
        if len(pieces) != 1:
            # Disjoint Q/K/V and gate/up rows need a segmented ReadSpec. Keep
            # the correct full read until that shared protocol is installed.
            return ReadSpec.full(name, shape)
        _, global_rows, local_rows, rank, _ = pieces[0]
        row_parallel = hasattr(layer, "input_size_per_partition")
        if row_parallel:
            if field == "weight" and layer.tp_size > 1:
                return make_sharded_read_spec(
                    name, shape, 1, layer.tp_size, layer.tp_rank
                )
        elif global_rows != local_rows:
            return make_sharded_read_spec(
                name, shape, 0, global_rows // local_rows, rank
            )
        return ReadSpec.full(name, shape)

    def load_weights(self, layer, weights):
        for name, checkpoint_value in weights.items():
            tensor, read_spec = unwrap_checkpoint_tensor(checkpoint_value)
            field, pieces = self._selection(layer, name)
            loaded = self._pieces.setdefault(field, set())
            names = {piece[0] for piece in pieces}
            if loaded.intersection(names):
                raise RuntimeError(f"Duplicate compressed W8A8 tensor {name!r}")
            local = read_spec is not None and read_spec.is_local_shard
            if local and len(pieces) != 1:
                raise ValueError("Fused W8A8 checkpoint cannot use a single TP slice")
            row_parallel = hasattr(layer, "input_size_per_partition")
            rows = sum(
                piece[2] if local and not row_parallel else piece[1] for piece in pieces
            )
            if field == "weight":
                if tensor.dtype != torch.int8:
                    raise TypeError(f"{name} requires signed INT8 checkpoint storage")
                columns = layer.input_size
                if row_parallel and not local:
                    columns *= layer.tp_size
                if tuple(tensor.shape) != (rows, columns):
                    raise ValueError(
                        f"{name} shape {tensor.shape} != {(rows, columns)}"
                    )
            elif field == "weight_scale":
                tensor = checked_channel_scale(tensor, rows, name)
            elif tensor.shape != (rows,):
                raise ValueError(
                    f"{name} expected bias shape {(rows,)}, got {tensor.shape}"
                )
            target = getattr(layer, field)
            source_offset = 0
            for _, global_rows, local_rows, rank, target_offset in pieces:
                begin = source_offset if local else source_offset + rank * local_rows
                piece = tensor.narrow(0, begin, local_rows)
                if field == "weight" and row_parallel and not local:
                    piece = piece.narrow(
                        1, layer.tp_rank * layer.input_size, layer.input_size
                    )
                copy_weight_(
                    target.data.narrow(0, target_offset, local_rows), piece, name
                )
                source_offset += local_rows if local else global_rows
            loaded.update(names)
            if len(loaded) == len(self._layout(layer)):
                layer._record_parameter_loaded(field)


@register_quant_method(W8A8_METHOD)
class W8A8Int8LinearMethod(Int8CheckpointLinearMethod):
    def __init__(self, quant_config=None):
        super().__init__(quant_config)
        self._executor = None
        self._storage = None

    def validate_runtime_device(self, device):
        if torch.device(device).type == "cpu":
            # CPU loading is useful for conversion/validation, not INT8 inference.
            return
        require_ppu_device(device)
        _dense_executor_class()

    @staticmethod
    def _storage_signature(layer):
        return tuple(
            (value.device, value.data_ptr())
            for value in (layer.weight, layer.weight_scale)
        )

    def process_weights_after_loading(self, layer):
        layer.validate_weights_loaded()
        checked_channel_scale(layer.weight_scale, layer.weight.shape[0], layer.prefix)
        if layer.weight.device.type == "cpu":
            return
        self.validate_runtime_device(layer.weight.device)
        self.bind_resident_runtime(layer)

    def bind_resident_runtime(self, layer):
        if layer.weight.device.type == "cpu":
            return
        cls = _dense_executor_class()
        n, k = layer.weight.shape
        # Legacy executor accepts a [K,N] view of row-major [N,K] bytes. This
        # view shares the module-owned storage; it is deliberately not transpose.
        self._executor = cls(
            weight=layer.weight.reshape(k, n),
            weight_scales=layer.weight_scale,
            quant_config=self.quant_config.source_config,
            bias=None,
        )
        self._storage = self._storage_signature(layer)

    def apply(self, layer, x, bias=None):
        require_ppu_device(x.device)
        if x.device != layer.weight.device:
            raise ValueError("W8A8 activation and parameter devices differ")
        if self._executor is None or self._storage != self._storage_signature(layer):
            self.process_weights_after_loading(layer)
        result = self._executor(x)
        return result if bias is None else result + bias

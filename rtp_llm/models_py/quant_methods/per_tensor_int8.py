"""Calibrated decoder W8A8. Checkpoint matrices are [input, output]."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.models_py.module_base import copy_weight_
from rtp_llm.models_py.quant_methods.base import register_quant_method
from rtp_llm.models_py.quant_methods.startup_lora import merge_calibrated_projection
from rtp_llm.models_py.quant_methods.w8a8_int8 import W8A8Int8LinearMethod
from rtp_llm.models_py.read_spec import (
    LoadedCheckpointTensor,
    ReadSpec,
    make_sharded_read_spec,
    unwrap_checkpoint_tensor,
)


class StaticInt8Mapper:
    """Route the legacy non-dotted activation suffix before module dispatch."""

    def __init__(self, model_mapper):
        self.model_mapper = model_mapper

    @staticmethod
    def _activation_name(name):
        if not name.endswith("_input_scale"):
            return name
        owner = name.removesuffix("_input_scale")
        parent, sep, projection = owner.rpartition(".")
        # Old fused FFN descriptors use the first (gate) projection's step.
        # QKV calibration belongs to the common input of the three matrices.
        projection = {
            "qkv": "qkv_proj",
            "q_proj": "qkv_proj",
            "gate_proj": "gate_up_proj",
        }.get(projection, projection)
        return parent + sep + projection + ".input_scale"

    def map_name(self, name):
        return self.model_mapper.map_name(self._activation_name(name))

    def apply(self, weights):
        return self.model_mapper.apply(
            (self._activation_name(name), tensor) for name, tensor in weights
        )


@register_quant_method("pertensor_quant")
class PerTensorInt8LinearMethod(W8A8Int8LinearMethod):
    required_checkpoint_parameters = ("weight", "weight_scale", "input_scale")

    def merge_ft_lora(self, layer, update):
        merge_calibrated_projection(layer, update, per_row=True)

    def create_weights(self, layer, input_size, output_size, params_dtype):
        layer.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.int8), requires_grad=False
        )
        layer.weight_scale = nn.Parameter(
            torch.empty(output_size, 1, dtype=torch.float32), requires_grad=False
        )
        layer.input_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )

    @staticmethod
    def _canonical(name):
        head, dot, leaf = name.rpartition(".")
        if leaf not in ("qweight", "scales", "bias", "input_scale"):
            raise ValueError(f"Unexpected pertensor_quant tensor {name!r}")
        return (
            head + dot + {"qweight": "weight", "scales": "weight_scale"}.get(leaf, leaf)
        )

    def checkpoint_read_spec(self, layer, name, shape):
        canonical = self._canonical(name)
        if canonical == "input_scale":
            if math.prod(shape) != 1:
                raise ValueError(f"{layer.prefix}.{name} must be a scalar")
            return ReadSpec.full(name, shape)
        field, pieces = self._selection(layer, canonical)
        rows = sum(piece[1] for piece in pieces)
        row_parallel = hasattr(layer, "input_size_per_partition")
        columns = layer.input_size * (layer.tp_size if row_parallel else 1)
        dim = None
        if field == "weight" and tuple(shape) == (columns, rows):
            dim = 0 if row_parallel else 1
        elif field == "bias" and tuple(shape) == (rows,):
            dim = None if row_parallel else 0
        elif field == "weight_scale" and math.prod(shape) == 1:
            pass  # A scalar has no TP axis.
        elif field == "weight_scale" and math.prod(shape) == rows and rows in shape:
            dim = None if row_parallel else shape.index(rows)
        else:
            raise ValueError(f"Invalid {layer.prefix}.{name} shape {shape}")
        if dim is not None and len(pieces) == 1:
            _, total, local, source_rank, _ = pieces[0]
            count, rank = (
                (layer.tp_size, layer.tp_rank)
                if row_parallel
                else (total // local, source_rank)
            )
            if count > 1:
                return make_sharded_read_spec(name, shape, dim, count, rank)
        return ReadSpec.full(name, shape)

    def load_weights(self, layer, weights):
        for name, value in weights.items():
            tensor, spec = unwrap_checkpoint_tensor(value)
            canonical = self._canonical(name)
            if spec is None:
                self.checkpoint_read_spec(layer, name, tuple(tensor.shape))
            if canonical == "input_scale":
                if (
                    not tensor.is_floating_point()
                    or tensor.numel() != 1
                    or not torch.isfinite(tensor).all()
                    or not (tensor > 0).all()
                ):
                    raise ValueError(
                        f"{layer.prefix}.{name} requires a finite positive calibration step"
                    )
                copy_weight_(layer.input_scale.data, tensor.reshape(1), name)
                layer._record_parameter_loaded("input_scale")
                continue
            field, pieces = self._selection(layer, canonical)
            if field == "weight":
                if tensor.ndim != 2:
                    raise ValueError(f"{name} must be an [input, output] INT8 matrix")
                tensor = tensor.t()
            elif field == "weight_scale":
                rows = sum(
                    piece[2] if spec is not None else piece[1] for piece in pieces
                )
                if not tensor.is_floating_point() or tensor.numel() not in (1, rows):
                    raise ValueError(
                        f"{name} requires one scale or {rows} output-channel scales"
                    )
                tensor = tensor.reshape(-1).expand(rows).float()
            if spec is not None:
                tensor = LoadedCheckpointTensor(tensor, spec)
            super().load_weights(layer, {canonical: tensor})

    def validate_runtime_device(self, device):
        if torch.device(device).type not in ("cpu", "cuda"):
            raise ValueError(f"Unsupported static INT8 runtime: {device}")

    def process_weights_after_loading(self, layer):
        layer.validate_weights_loaded()

    def apply(self, layer, x, bias=None):
        n, k = layer.weight.shape
        a = (
            (x.reshape(-1, k).float() / layer.input_scale)
            .round()
            .clamp(-128, 127)
            .to(torch.int8)
        )
        m = a.shape[0]
        if not m:
            return x.new_empty((*x.shape[:-1], n))
        if x.is_cuda:
            # cuBLAS int8 GEMM has tile constraints even at decode batch=1.
            pk, pm, pn = -k % 32, -m % 32, -n % 8
            w = F.pad(layer.weight, (0, pk, 0, pn)) if pk or pn else layer.weight
            a = F.pad(a, (0, pk, 0, pm)) if pk or pm else a
            sums = torch._int_mm(a.contiguous(), w.t())[:m, :n]
        else:
            sums = a.int() @ layer.weight.int().t()
        result = sums.float() * layer.input_scale * layer.weight_scale.view(1, n)
        if bias is not None:
            result = result + bias
        return result.to(x.dtype).reshape(*x.shape[:-1], n)

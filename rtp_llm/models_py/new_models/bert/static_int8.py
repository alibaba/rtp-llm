"""Static W8A8 BERT: calibrated steps and signed input-/output-major matrices."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.models_py.layers.linear import QKVParallelLinear, RowParallelLinear
from rtp_llm.models_py.module_base import copy_weight_
from rtp_llm.models_py.quant_methods.base import QuantizationConfig, QuantizeMethodBase
from rtp_llm.models_py.read_spec import (
    ReadSpec,
    make_sharded_read_spec,
    unwrap_checkpoint_tensor,
)


class BertStaticInt8Config(QuantizationConfig):
    """The BERT checkpoint contract is not a generic decoder quantization format."""

    def __init__(self, quant_type, *, input_major=False):
        super().__init__(quant_type)
        self.input_major = input_major

    def get_quant_method(self, layer, prefix=""):
        return BertStaticInt8Linear(self)


class BertStaticInt8Linear(QuantizeMethodBase):
    owns_checkpoint_loading = True
    required_checkpoint_parameters = ("weight", "weight_scale", "input_scale")

    def create_weights(self, layer, input_size, output_size, params_dtype, **kwargs):
        for key, shape, dtype in (
            ("weight", (output_size, input_size), torch.int8),
            ("weight_scale", (output_size,), torch.float32),
            ("input_scale", (1,), torch.float32),
        ):
            layer.register_parameter(
                key, nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            )

    def _layout(self, layer, name):
        keys = name.split(".")
        key = keys[-1]
        row = isinstance(layer, RowParallelLinear)
        leaf = None
        if isinstance(layer, QKVParallelLinear) and key != "input_scale":
            if len(keys) != 2 or keys[0] not in layer.shard_names:
                raise ValueError(f"Expected a BERT query/key/value tensor: {name}")
            leaf = keys[0][0]
        elif len(keys) != 1:
            raise ValueError(f"Unknown BERT projection: {name}")
        offset, width = (
            layer._qkv_shard_offset_size(leaf) if leaf else (0, layer.output_size)
        )
        out = width if row else width * layer.tp_size
        inp = layer.input_size * layer.tp_size if row else layer.input_size
        return key, row, leaf, offset, width, out, inp

    def checkpoint_read_spec(self, layer, name, shape):
        key, row, _, _, _, out, inp = self._layout(layer, name)
        shape = tuple(shape)
        dim = None
        input_major = self.quant_config.input_major
        if key == "weight" and shape == ((inp, out) if input_major else (out, inp)):
            dim = 1 if row else 0
            if input_major:
                dim = 1 - dim
        elif key == "bias" and shape == (out,):
            dim = None if row else 0
        elif key == "input_scale" and math.prod(shape) == 1:
            pass
        elif key == "weight_scale" and math.prod(shape) == 1:
            pass
        elif (
            key == "weight_scale" and math.prod(shape) == out and shape.count(out) == 1
        ):
            dim = None if row else shape.index(out)
        else:
            raise ValueError(f"Invalid static INT8 {layer.prefix}.{name}: {shape}")
        if dim is None or layer.tp_size == 1:
            return ReadSpec.full(name, shape)
        return make_sharded_read_spec(name, shape, dim, layer.tp_size, layer.tp_rank)

    def load_weights(self, layer, weights):
        for name, value in weights.items():
            x, read = unwrap_checkpoint_tensor(value)
            key, _, leaf, offset, width, _, _ = self._layout(layer, name)
            if read is None or not read.is_local_shard:
                spec = self.checkpoint_read_spec(layer, name, x.shape)
                if spec.slices is not None:
                    x = x[spec.slices]
            if key == "weight":
                if x.dtype != torch.int8:
                    raise ValueError(
                        f"{layer.prefix}.{name} must contain signed INT8 weights"
                    )
                if self.quant_config.input_major:
                    x = x.t()  # Only the rank-local rectangle has been materialized.
            elif not x.is_floating_point():
                raise ValueError(f"{layer.prefix}.{name} must be floating point")
            target = getattr(layer, key)
            if leaf:
                layer._ensure_parameter_shard_not_loaded(key, leaf)
                target = target.narrow(0, offset, width)
            elif key in layer._loaded_parameter_names:
                raise RuntimeError(f"Duplicate checkpoint tensor {layer.prefix}.{name}")
            if key.endswith("scale"):
                x = x.reshape(-1)
                if not torch.isfinite(x).all() or not (x > 0).all():
                    raise ValueError(
                        f"{layer.prefix}.{name} must have finite positive scale"
                    )
                if x.numel() == 1:
                    x = x.expand_as(target)
            copy_weight_(target.data, x, f"{layer.prefix}.{name}")
            if leaf:
                layer._record_parameter_shard_loaded(key, leaf)
            else:
                layer._record_parameter_loaded(key)

    def process_weights_after_loading(self, layer):
        pass  # We retain the signed checkpoint representation.

    def apply(self, layer, x, bias=None):
        # Quantize with the stored activation step, accumulate exactly in int32,
        # and restore the output units before applying floating-point bias.
        a = (
            (x.reshape(-1, layer.input_size).float() / layer.input_scale)
            .round()
            .clamp(-128, 127)
            .to(torch.int8)
        )
        w = layer.weight
        m, n = a.shape[0], w.shape[0]
        if m == 0:
            return x.new_empty((*x.shape[:-1], n))
        if x.is_cuda:
            # The native int8 GEMM requires aligned dimensions, including for
            # small token batches. Padded signed zeros do not affect the sum.
            pk, pn, pm = (-a.shape[1]) % 32, (-n) % 8, (-m) % 32
            if pk or pm:
                a = F.pad(a, (0, pk, 0, pm))
            if pk or pn:
                w = F.pad(w, (0, pk, 0, pn))
        y = torch._int_mm(a.contiguous(), w.t())[:m, :n].float()
        y = y * layer.input_scale * layer.weight_scale
        if bias is not None:
            y = y + bias
        return y.to(x.dtype).reshape(*x.shape[:-1], n)

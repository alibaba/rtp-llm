"""Dense SmoothQuant/OmniQuant checkpoint layout and online activation quantization."""

import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.models_py.module_base import copy_weight_
from rtp_llm.models_py.quant_methods.base import register_quant_method
from rtp_llm.models_py.quant_methods.w8a8_int8 import Int8CheckpointLinearMethod
from rtp_llm.models_py.read_spec import (
    LoadedCheckpointTensor,
    ReadSpec,
    make_sharded_read_spec,
    unwrap_checkpoint_tensor,
)


def quantize_activation(x, smoother=None, shift=None):
    # Legacy perTokenQuantization divides, then adds, in activation dtype.
    if smoother is not None:
        x = x / smoother.to(x.dtype)
    if shift is not None:
        x = x + shift.to(x.dtype)
    peak = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6).float()
    q = (x.float() * (127.0 / peak)).round().clamp(-128, 127).to(torch.int8)
    return q, peak / 127.0


@register_quant_method("smooth_quant", "omni_quant")
class SmoothOmniLinearMethod(Int8CheckpointLinearMethod):
    """Reuse the direct INT8 owner and TP copy logic, not the PPU executor."""

    def create_weights(self, layer, input_size, output_size, params_dtype):
        super().create_weights(layer, input_size, output_size, params_dtype)
        leaf = layer.prefix.split(".")[-1]
        self.omni = self.quant_config.quant_type == "omni_quant"
        self.qkv = hasattr(layer, "num_kv_heads")
        self.row = hasattr(layer, "input_size_per_partition")
        # In the old unfused Omni Q/K/V descriptor each .scales also supplies
        # that projection's bias. A fused archive instead stores a real bias.
        if self.omni and (
            self.qkv or leaf in ("o_proj", "gate_up_proj", "gate_proj", "up_proj")
        ):
            if self.row and layer.tp_size > 1 and not layer.reduce_output:
                raise ValueError("OmniQuant row bias requires TP output reduction")
            if layer.bias is None:
                layer.bias = nn.Parameter(
                    torch.empty(output_size, dtype=params_dtype), requires_grad=False
                )
        extra = []
        if leaf == "o_proj" or (leaf == "down_proj" and not self.omni):
            extra.append("smoother")
        if leaf == "o_proj" and self.omni:
            extra.append("shift")
        for field in extra:
            layer.register_parameter(
                field,
                nn.Parameter(
                    torch.empty(input_size, dtype=torch.float32), requires_grad=False
                ),
            )
        self.required_checkpoint_parameters = ("weight", "weight_scale", *extra)

    def _describe(self, layer, name):
        parts = name.split(".")
        source_field = parts[-1]
        if (
            source_field in ("shift", "smoother")
            and name == source_field
            and source_field in self.required_checkpoint_parameters
        ):
            return source_field, (), False
        field = {
            "qweight": "weight",
            "qweight_out_in": "weight",
            "scales": "weight_scale",
            "bias": "bias",
        }.get(source_field)
        if field is None:
            raise ValueError(
                f"Unexpected {self.quant_config.quant_type} checkpoint tensor {name!r}"
            )
        parts[-1] = field
        canonical = ".".join(parts)
        _, pieces = self._selection(layer, canonical)
        transposed = source_field == "qweight" and not (
            self.omni and self.qkv and len(parts) > 1
        )
        return canonical, pieces, transposed

    def checkpoint_read_spec(self, layer, name, shape):
        canonical, pieces, transposed = self._describe(layer, name)
        if not pieces:
            width = layer.input_size * layer.tp_size
            if tuple(shape) not in ((width,), (1, width)):
                raise ValueError(f"{name}: expected one value per input channel")
            if layer.tp_size > 1:
                return make_sharded_read_spec(
                    name, shape, len(shape) - 1, layer.tp_size, layer.tp_rank
                )
        elif len(pieces) == 1:
            _, total, local, rank, _ = pieces[0]
            if self.row and canonical.endswith("weight") and layer.tp_size > 1:
                return make_sharded_read_spec(
                    name, shape, 0 if transposed else 1, layer.tp_size, layer.tp_rank
                )
            if not self.row and total != local:
                row_scale = canonical.endswith("weight_scale") and tuple(shape) == (
                    1,
                    total,
                )
                return make_sharded_read_spec(
                    name, shape, int(transposed or row_scale), total // local, rank
                )
        return ReadSpec.full(name, shape)

    def load_weights(self, layer, weights):
        for name, value in weights.items():
            canonical, pieces, transposed = self._describe(layer, name)
            tensor, spec = unwrap_checkpoint_tensor(value)
            if not pieces:
                if tensor.ndim == 2 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                if tensor.dtype != torch.float32 or tensor.ndim != 1:
                    raise ValueError(f"{name}: expected FP32 input-channel vector")
                if spec is None and layer.tp_size > 1:
                    if tensor.numel() != layer.input_size * layer.tp_size:
                        raise ValueError(f"{name}: wrong global input width")
                    tensor = tensor.narrow(
                        0, layer.tp_rank * layer.input_size, layer.input_size
                    )
                if not torch.isfinite(tensor).all() or (
                    name == "smoother" and (tensor <= 0).any()
                ):
                    raise ValueError(f"{name}: invalid activation transform")
                copy_weight_(getattr(layer, name).data, tensor, name)
                layer._record_parameter_loaded(name)
                continue
            if transposed:
                # The Qwen1 SQ exporter groups a fused MHA matrix as [K,3,H]
                # and its scales as [3,H]. Never flatten packed GPTQ/AWQ.
                if (
                    self.qkv
                    and name == "qweight"
                    and tensor.ndim == 3
                    and tensor.shape[1] == 3
                ):
                    tensor = tensor.flatten(1)
                if tensor.ndim != 2:
                    raise ValueError(f"{name}: expected matrix")
                tensor = tensor.t()
            elif (
                self.qkv
                and name == "scales"
                and tensor.ndim == 2
                and tensor.shape[0] == 3
            ):
                tensor = tensor.flatten()
            elif (
                canonical.endswith("weight_scale")
                and tensor.ndim == 2
                and tensor.shape[0] == 1
            ):
                tensor = tensor.squeeze(0)

            def wrap(t):
                return t if spec is None else LoadedCheckpointTensor(t, spec)

            super().load_weights(layer, {canonical: wrap(tensor)})
            if self.omni and self.qkv and canonical.endswith(".weight_scale"):
                bias = canonical.rsplit(".", 1)[0] + ".bias"
                super().load_weights(layer, {bias: wrap(tensor.reshape(-1))})

    def validate_runtime_device(self, device):
        if torch.device(device).type not in ("cpu", "cuda"):
            raise ValueError(f"Unsupported dense INT8 runtime: {device}")

    def process_weights_after_loading(self, layer):
        layer.validate_weights_loaded()

    def apply(self, layer, x, bias=None):
        n, k = layer.weight.shape
        rows = x.reshape(-1, k)
        if not rows.shape[0]:
            return x.new_empty((*x.shape[:-1], n))
        q, scale = quantize_activation(
            rows, getattr(layer, "smoother", None), getattr(layer, "shift", None)
        )
        # cuBLAS integer MM has alignment requirements even for one-token decode.
        # Only pad where necessary. The weight remains INT8; sums stay INT32.
        m = rows.shape[0]
        pad_k, pad_n = -k % 8, -n % 8
        padded_m = max(24, (m + 7) // 8 * 8) if x.is_cuda else m
        a = F.pad(q, (0, pad_k, 0, padded_m - m)) if pad_k or padded_m != m else q
        w = (
            F.pad(layer.weight, (0, pad_k, 0, pad_n))
            if pad_k or pad_n
            else layer.weight
        )
        sums = torch._int_mm(a, w.t())[:m, :n]
        result = (sums.float() * scale * layer.weight_scale.view(1, -1)).to(x.dtype)
        if bias is not None:
            result = result + bias
        return result.reshape(*x.shape[:-1], n)

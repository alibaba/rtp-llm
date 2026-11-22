"""AWQ/GPTQ packed checkpoint ownership for ordinary and fused linears."""

import math
from dataclasses import asdict

import torch
from torch import nn

from rtp_llm.models_py.module_base import copy_weight_
from rtp_llm.models_py.quant_methods.base import (
    QuantizeMethodBase,
    register_quant_method,
)
from rtp_llm.models_py.quant_methods.groupwise_checkpoint import (
    GroupwiseCheckpointLayout,
)
from rtp_llm.models_py.read_spec import (
    ReadSpec,
    make_sharded_read_spec,
    unwrap_checkpoint_tensor,
)


def groupwise_settings(config):
    source = config.source_config
    method = config.quant_type
    bits = getattr(source, "bits", None)
    size = getattr(source, "group_size", None)
    size = size() if callable(size) else size
    if method not in ("awq", "gptq") or isinstance(bits, bool) or bits not in (4, 8):
        raise ValueError("AWQ/GPTQ requires an explicit 4/8-bit source configuration")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("AWQ/GPTQ requires a positive quantization group size")
    quantized = getattr(source, "is_quanted", None)
    if callable(quantized) and not quantized():
        raise ValueError("AWQ/GPTQ requires an already-quantized checkpoint")
    return method, bits, size


def runtime_module():
    try:
        from rtp_llm.models_py.quant_methods import groupwise_runtime
    except ImportError as error:
        raise RuntimeError("AWQ/GPTQ native runtime adapter is unavailable") from error
    return groupwise_runtime


def checked_piece(value, name, expected):
    tensor, spec = unwrap_checkpoint_tensor(value)
    parameter = name.rsplit(".", 1)[-1]
    if parameter in ("qweight", "qzeros", "g_idx") and tensor.dtype != torch.int32:
        raise TypeError(f"Grouped packed tensor must be int32: {name}")
    if parameter in ("scales", "bias") and tensor.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise TypeError(f"Grouped scales/bias must be floating point: {name}")
    if spec is not None:
        if spec.for_checkpoint_name(name) != expected:
            raise ValueError(
                f"Grouped ReadSpec belongs to another projection/rank: {name}"
            )
    else:
        if tuple(tensor.shape) != expected.source_shape:
            raise ValueError(
                f"Grouped source shape mismatch for {name}: {tuple(tensor.shape)}"
            )
        if expected.slices is not None:
            tensor = tensor[expected.slices]
    return tensor


@register_quant_method("awq", "gptq")
class GroupwiseLinearMethod(QuantizeMethodBase):
    owns_checkpoint_loading = True
    required_checkpoint_parameters = ("qweight", "qzeros", "scales")
    requires_staged_device_postprocess = True

    def __init__(self, quant_config):
        super().__init__(quant_config)
        self.method, self.bits, self.group_size = groupwise_settings(quant_config)

    def validate_runtime_device(self, device):
        device = torch.device(device)
        if device.type != "cpu":
            runtime_module().validate_runtime_device(device, self.method, self.bits)

    def create_weights(self, layer, input_size, output_size, params_dtype, **kwargs):
        layout = GroupwiseCheckpointLayout(
            self.method, self.bits, self.group_size, input_size, output_size
        )
        layer._groupwise_checkpoint_layout = layout
        layer._groupwise_layout = asdict(layout)
        layer._groupwise_seen = set()
        for name, shape in layout.shapes.items():
            layer.register_parameter(
                name,
                nn.Parameter(
                    torch.empty(
                        shape, dtype=params_dtype if name == "scales" else torch.int32
                    ),
                    requires_grad=False,
                ),
            )
        tp, rank = layer.tp_size, layer.tp_rank
        sources = {}
        if hasattr(layer, "num_kv_heads"):
            kv_tp = math.gcd(layer.num_kv_heads, tp)
            kv_rank = rank // (tp // kv_tp)
            declarations = [
                ("q_proj", layer.num_heads * layer.head_dim, tp, rank),
                ("k_proj", layer.num_kv_heads * layer.head_dim, kv_tp, kv_rank),
                ("v_proj", layer.num_kv_heads * layer.head_dim, kv_tp, kv_rank),
            ]
            offset = 0
            for projection, width, shards, shard_rank in declarations:
                source = GroupwiseCheckpointLayout(
                    self.method, self.bits, self.group_size, input_size, width
                )
                local = source.local_layout(1, shards, shard_rank)
                sources[projection] = (source, 1, shards, shard_rank, offset)
                offset += local.output_size
        elif getattr(layer, "shard_names", ()):
            shards = tuple(layer.shard_names)
            if output_size % len(shards):
                raise ValueError("Grouped fused projections have unequal local widths")
            width = output_size // len(shards)
            for index, projection in enumerate(shards):
                source = GroupwiseCheckpointLayout(
                    self.method, self.bits, self.group_size, input_size, width * tp
                )
                source.local_layout(1, tp, rank)
                sources[projection] = (source, 1, tp, rank, index * width)
        else:
            dim = 0 if hasattr(layer, "input_size_per_partition") else 1
            source = GroupwiseCheckpointLayout(
                self.method,
                self.bits,
                self.group_size,
                input_size * tp if dim == 0 else input_size,
                output_size * tp if dim == 1 else output_size,
            )
            source.local_layout(dim, tp, rank)
            sources[""] = (source, dim, tp, rank, 0)
        layer._groupwise_sources = sources

    def _role(self, layer, name):
        parts = name.split(".")
        parameter = parts[-1]
        if parameter not in ("qweight", "qzeros", "scales", "bias", "g_idx"):
            raise ValueError(f"Unexpected grouped checkpoint tensor {name}")
        if len(parts) > 2 or (
            len(parts) == 2 and parts[0] not in layer._groupwise_sources
        ):
            raise ValueError(f"Unexpected grouped projection {name}")
        projection = parts[0] if len(parts) == 2 else ""
        if parameter == "bias" and layer.bias is None and self.method != "gptq":
            raise ValueError(f"Unexpected grouped bias {name}")
        return projection, parameter

    def is_validation_only_checkpoint_tensor(self, layer, name):
        _, parameter = self._role(layer, name)
        return parameter == "g_idx" or (
            self.method == "gptq" and parameter == "bias" and layer.bias is None
        )

    def _source_spec(self, name, shape, source, dim, shards, rank, parameter):
        if parameter in ("qweight", "qzeros", "scales"):
            return source.read_spec(name, shape, dim, shards, rank)
        expected = (
            (source.output_size,) if parameter == "bias" else (source.input_size,)
        )
        if tuple(shape) != expected:
            raise ValueError(
                f"Grouped {parameter} shape must be {expected}, got {shape}"
            )
        return (
            make_sharded_read_spec(name, expected, 0, shards, rank)
            if parameter == "bias" and dim == 1 and shards > 1
            else ReadSpec.full(name, expected)
        )

    def checkpoint_read_spec(self, layer, name, shape):
        projection, parameter = self._role(layer, name)
        if projection in layer._groupwise_sources:
            source, dim, shards, rank, _ = layer._groupwise_sources[projection]
            if parameter == "bias" and layer.bias is None:
                # GPTQ exporters may serialize an all-zero bias buffer even
                # when the original model linear has no bias. Validate the
                # complete buffer on every TP rank, not only its local slice.
                shards, rank = 1, 0
            return self._source_spec(name, shape, source, dim, shards, rank, parameter)
        # A fused source has disjoint TP ranges for its constituent projections.
        # Keep that source whole, then copy each logical range exactly once.
        first = next(iter(layer._groupwise_sources.values()))[0]
        source = GroupwiseCheckpointLayout(
            self.method,
            self.bits,
            self.group_size,
            first.input_size,
            sum(item[0].output_size for item in layer._groupwise_sources.values()),
        )
        return self._source_spec(name, shape, source, 1, 1, 0, parameter)

    def _copy_projection(self, layer, projection, parameter, value, name):
        key = (projection, parameter)
        if key in layer._groupwise_seen:
            raise RuntimeError(f"Duplicate grouped projection tensor {name}")
        source, dim, shards, rank, offset = layer._groupwise_sources[projection]
        tensor, spec = unwrap_checkpoint_tensor(value)
        shape = spec.source_shape if spec is not None else tuple(tensor.shape)
        expected = self._source_spec(name, shape, source, dim, shards, rank, parameter)
        tensor = checked_piece(value, name, expected)
        if parameter == "g_idx":
            identity = (
                torch.arange(source.input_size, device=tensor.device, dtype=torch.int32)
                // source.group_size
            )
            if tensor.dtype != torch.int32 or not torch.equal(tensor, identity):
                raise ValueError(
                    "GPTQ activation-order g_idx requires an explicit permutation backend"
                )
        else:
            local = source.local_layout(dim, shards, rank)
            if parameter == "bias":
                target = layer.bias.narrow(0, offset, local.output_size)
            else:
                pack = (
                    local.pack_factor
                    if parameter == "qzeros"
                    or (parameter == "qweight" and self.method == "awq")
                    else 1
                )
                target = getattr(layer, parameter).narrow(
                    1, offset // pack, local.shapes[parameter][1]
                )
            copy_weight_(target, tensor, name)
        layer._groupwise_seen.add(key)
        if parameter != "g_idx" and all(
            (part, parameter) in layer._groupwise_seen
            for part in layer._groupwise_sources
        ):
            layer._record_parameter_loaded(parameter)
            if parameter == "qweight":
                layer._main_weight_loaded = True

    def load_weights(self, layer, weights):
        if getattr(layer, "_groupwise_runtime_metadata", None) is not None and not all(
            self.is_validation_only_checkpoint_tensor(layer, name) for name in weights
        ):
            raise RuntimeError(
                "Grouped runtime weights require a fresh checkpoint staging module"
            )
        for name, value in weights.items():
            projection, parameter = self._role(layer, name)
            if parameter == "bias" and layer.bias is None:
                parts = (
                    (projection,)
                    if projection in layer._groupwise_sources
                    else layer._groupwise_sources
                )
                keys = {(part, parameter) for part in parts}
                if not keys.isdisjoint(layer._groupwise_seen):
                    raise RuntimeError(f"Duplicate grouped projection tensor {name}")
                tensor, spec = unwrap_checkpoint_tensor(value)
                shape = spec.source_shape if spec is not None else tuple(tensor.shape)
                tensor = checked_piece(
                    value, name, self.checkpoint_read_spec(layer, name, shape)
                )
                if torch.count_nonzero(tensor).item() != 0:
                    raise ValueError(
                        "GPTQ checkpoint bias for a biasless linear must be all zero: "
                        f"owner={layer.prefix}, tensor={name}"
                    )
                # This is checkpoint-only metadata, not an owned parameter or
                # an excuse to add a bias to the production model architecture.
                layer._groupwise_seen.update(keys)
                continue
            if projection in layer._groupwise_sources:
                self._copy_projection(layer, projection, parameter, value, name)
                continue
            tensor, spec = unwrap_checkpoint_tensor(value)
            shape = spec.source_shape if spec is not None else tuple(tensor.shape)
            tensor = checked_piece(
                value, name, self.checkpoint_read_spec(layer, name, shape)
            )
            if parameter == "g_idx":
                for part in layer._groupwise_sources:
                    self._copy_projection(
                        layer, part, parameter, tensor, part + "." + parameter
                    )
                continue
            pack = (
                layer._groupwise_checkpoint_layout.pack_factor
                if parameter == "qzeros"
                or (parameter == "qweight" and self.method == "awq")
                else 1
            )
            axis = 0 if parameter == "bias" else 1
            widths = [
                item[0].output_size // (1 if parameter == "bias" else pack)
                for item in layer._groupwise_sources.values()
            ]
            for part, piece in zip(
                layer._groupwise_sources, tensor.split(widths, dim=axis)
            ):
                self._copy_projection(
                    layer, part, parameter, piece, part + "." + parameter
                )

    def process_weights_after_loading(self, layer):
        layer.validate_weights_loaded()
        if layer.qweight.device.type != "cpu":
            runtime_module().finalize_linear(layer)

    def runtime_layout_is_ready(self, layer):
        # CPU staging must not suppress accelerator finalization after .to().
        # Runtime finalizers own their cache/idempotence checks.
        return layer.qweight.device.type == "cpu"

    def apply(self, layer, x, bias=None):
        if layer.qweight.device.type == "cpu":
            raise RuntimeError(
                "CPU grouped checkpoint staging is not a native inference backend"
            )
        return runtime_module().linear(layer, x, bias)

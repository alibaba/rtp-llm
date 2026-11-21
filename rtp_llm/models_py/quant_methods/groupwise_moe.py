"""AWQ/GPTQ expert parameter owners; packing remains checkpoint-native on CPU."""

from dataclasses import asdict

import torch
from torch import nn

from rtp_llm.models_py.module_base import copy_weight_
from rtp_llm.models_py.quant_methods.base import (
    FusedMoEMethodBase,
    register_moe_quant_method,
)
from rtp_llm.models_py.quant_methods.groupwise import (
    checked_piece,
    groupwise_settings,
    runtime_module,
)
from rtp_llm.models_py.quant_methods.groupwise_checkpoint import (
    GroupwiseCheckpointLayout,
)
from rtp_llm.models_py.read_spec import ReadSpec, unwrap_checkpoint_tensor


@register_moe_quant_method("awq", "gptq")
class GroupwiseMoEMethod(FusedMoEMethodBase):
    owns_checkpoint_loading = True
    requires_staged_device_postprocess = True

    def __init__(self, quant_config):
        super().__init__(quant_config)
        self.method, self.bits, self.group_size = groupwise_settings(quant_config)

    def validate_runtime_device(self, device):
        device = torch.device(device)
        if device.type != "cpu":
            runtime_module().validate_runtime_device(device, self.method, self.bits)

    def create_weights(
        self, layer, num_experts, hidden_size, intermediate_size, params_dtype, **kwargs
    ):
        layouts = {
            "w13": GroupwiseCheckpointLayout(
                self.method,
                self.bits,
                self.group_size,
                hidden_size,
                2 * intermediate_size,
            ),
            "w2": GroupwiseCheckpointLayout(
                self.method, self.bits, self.group_size, intermediate_size, hidden_size
            ),
        }
        layer._groupwise_checkpoint_layouts = layouts
        layer._groupwise_layout = {
            name: asdict(layout) for name, layout in layouts.items()
        }
        for projection, layout in layouts.items():
            for parameter, shape in layout.shapes.items():
                name = (
                    projection
                    if parameter == "qweight"
                    else projection + "_" + parameter
                )
                dtype = params_dtype if parameter == "scales" else torch.int32
                layer.register_parameter(
                    name,
                    nn.Parameter(
                        torch.empty((num_experts, *shape), dtype=dtype),
                        requires_grad=False,
                    ),
                )
        # Validate TP group boundaries before allocating/reading any checkpoint.
        for projection in layer.PROJ_NAMES:
            self._source(layer, projection)[0].local_layout(
                *self._source(layer, projection)[1:]
            )

    def _source(self, layer, projection):
        if projection not in layer.PROJ_NAMES:
            raise ValueError(f"Unsupported grouped expert projection {projection!r}")
        down = projection == "down_proj"
        layout = GroupwiseCheckpointLayout(
            self.method,
            self.bits,
            self.group_size,
            layer.moe_inter if down else layer.hidden_size,
            layer.hidden_size if down else layer.moe_inter,
        )
        return (
            layout,
            0 if down else 1,
            layer.moe_expert_tp_size,
            layer.moe_expert_tp_rank,
        )

    @staticmethod
    def _role(layer, name):
        parts = name.split(".")
        if len(parts) != 3 or not parts[0].isdigit():
            raise ValueError(
                f"Grouped experts require per-expert projection keys: {name}"
            )
        expert, projection, parameter = int(parts[0]), parts[1], parts[2]
        if not 0 <= expert < layer.num_experts or projection not in layer.PROJ_NAMES:
            raise ValueError(f"Invalid grouped expert source {name}")
        if parameter not in ("qweight", "qzeros", "scales", "g_idx"):
            raise ValueError(f"Unexpected grouped expert parameter {name}")
        return expert, projection, parameter

    def checkpoint_read_spec(self, layer, name, shape):
        expert, projection, parameter = self._role(layer, name)
        source, dim, shards, rank = self._source(layer, projection)
        if parameter == "g_idx":
            if tuple(shape) != (source.input_size,):
                raise ValueError(f"Invalid grouped expert g_idx shape for {name}")
            spec = ReadSpec.full(name, tuple(shape))
        else:
            spec = source.read_spec(name, tuple(shape), dim, shards, rank)
        return (
            spec
            if layer._remap_expert_ids(expert)
            else ReadSpec.skip(name, tuple(shape))
        )

    def required_aux_parameters(self):
        return ("qzeros", "scales")

    def load_weights(self, layer, weights):
        if getattr(layer, "_groupwise_runtime_metadata", None) is not None:
            raise RuntimeError(
                "Grouped expert runtime weights require a fresh checkpoint staging module"
            )
        for name, value in weights.items():
            expert, projection, parameter = self._role(layer, name)
            if name in layer._seen_checkpoint_names:
                raise RuntimeError(f"Duplicate grouped expert checkpoint tensor {name}")
            tensor, spec = unwrap_checkpoint_tensor(value)
            shape = spec.source_shape if spec is not None else tuple(tensor.shape)
            expected = self.checkpoint_read_spec(layer, name, shape)
            local_ids = layer._remap_expert_ids(expert)
            if not local_ids:
                layer._seen_checkpoint_names.add(name)
                continue
            tensor = checked_piece(value, name, expected)
            source, dim, shards, rank = self._source(layer, projection)
            if parameter == "g_idx":
                identity = (
                    torch.arange(
                        source.input_size, device=tensor.device, dtype=torch.int32
                    )
                    // source.group_size
                )
                if tensor.dtype != torch.int32 or not torch.equal(tensor, identity):
                    raise ValueError(
                        "GPTQ activation-order expert g_idx requires a permutation backend"
                    )
                layer._seen_checkpoint_names.add(name)
                continue
            local = source.local_layout(dim, shards, rank)
            stem = "w2" if projection == "down_proj" else "w13"
            attribute = stem if parameter == "qweight" else stem + "_" + parameter
            pack = (
                local.pack_factor
                if parameter == "qzeros"
                or (parameter == "qweight" and self.method == "awq")
                else 1
            )
            # BaseMoE's existing executor contract is up first, then gate.
            offset = layer.moe_inter_tp // pack if projection == "gate_proj" else 0
            width = local.shapes[parameter][1]
            for local_id in local_ids:
                key = (local_id, projection)
                if parameter == "qweight":
                    layer._ensure_projection_not_loaded(local_id, projection)
                elif (*key, parameter) in layer._loaded_aux_keys:
                    raise RuntimeError(
                        f"Duplicate grouped expert auxiliary tensor {name}"
                    )
                target = getattr(layer, attribute)[local_id].narrow(1, offset, width)
                copy_weight_(target, tensor, name)
                if parameter == "qweight":
                    layer._record_projection_loaded(local_id, projection)
                else:
                    layer._loaded_aux_keys.add((*key, parameter))
            layer._seen_checkpoint_names.add(name)
            projections = ("down_proj",) if stem == "w2" else ("up_proj", "gate_proj")
            complete = all(
                (
                    (local_id, proj) in layer._loaded_keys
                    if parameter == "qweight"
                    else (local_id, proj, parameter) in layer._loaded_aux_keys
                )
                for local_id in range(layer.num_local_experts)
                for proj in projections
            )
            if complete:
                layer._mark_weight_loaded(attribute)

    def process_weights_after_loading(self, layer):
        layer._check_load_complete()
        if layer.w13.device.type != "cpu":
            runtime_module().finalize_moe(layer)

    def build_moe_executor(self, layer):
        if layer.w13.device.type == "cpu":
            raise RuntimeError(
                "CPU grouped expert staging is not a native inference backend"
            )
        return runtime_module().build_moe_executor(layer)

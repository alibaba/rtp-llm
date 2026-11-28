"""Separate immutable decoder parameters from process-local execution objects.

The cold owner finalizes checkpoint-dependent tensors and adapters before T1.
The ordinary post-load pass builds executors *after* relocation in both workers.
"""

import json

import torch

from rtp_llm.models_py.layers.linear import LinearBase
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts


def _physical(value):
    return [list(value.shape), str(value.dtype)]


def _layout(leaf):
    method = leaf.quant_method
    facts = {
        "method": type(method).__name__,
        "params": {
            k: _physical(v) for k, v in leaf._parameters.items() if v is not None
        },
        "buffers": {
            k: _physical(v)
            for k, v in leaf._buffers.items()
            if v is not None and k not in leaf._non_persistent_buffers_set
        },
        "flags": {k: getattr(leaf, k) for k in _LEAF_FLAGS if hasattr(leaf, k)},
        "method_flags": {
            k: getattr(method, k) for k in _METHOD_FLAGS if hasattr(method, k)
        },
    }
    runtime = getattr(leaf, "_groupwise_runtime", None)
    if runtime is not None:
        from dataclasses import asdict

        states = runtime if isinstance(runtime, dict) else {"linear": runtime}
        facts["integer"] = {
            k: [asdict(v.layout), v.runtime_bits, v.promotion_reason]
            for k, v in states.items()
        }
    if isinstance(leaf, BaseMoEExperts):
        leaf.validate_weights_loaded()
        facts["expert"] = [
            leaf.num_local_experts,
            list(leaf.PROJ_NAMES),
            sorted(leaf._required_aux_param_names()),
            leaf._expert_layout_ready,
        ]
    else:
        facts["loaded"] = sorted(leaf._loaded_parameter_names)
    return facts


_LEAF_FLAGS = (
    "_fp8_gate_first",
    "_fp8_logical_input_size",
    "_fp8_logical_output_size",
    "output_size",
    "output_size_per_partition",
    "input_size",
    "input_size_per_partition",
)
_METHOD_FLAGS = ("_use_deep_gemm", "_runtime_backend")


def _restore_layout(leaf, facts, values, prefix):
    method = leaf.quant_method
    if facts["method"] != type(method).__name__:
        raise ValueError(f"Resident quantizer changed: {prefix}")
    for owner, flags, allowed in (
        (leaf, facts["flags"], _LEAF_FLAGS),
        (method, facts["method_flags"], _METHOD_FLAGS),
    ):
        for k, v in flags.items():
            if k not in allowed or not isinstance(v, (str, int, bool, type(None))):
                raise ValueError(f"Invalid resident quantizer field: {prefix}{k}")
            setattr(owner, k, v)
    for kind, declared in (("params", leaf._parameters), ("buffers", leaf._buffers)):
        fields = facts[kind]
        for name, desc in fields.items():
            value = values[prefix + name]
            if desc != _physical(value):
                raise ValueError(f"Invalid resident quantized layout: {prefix}{name}")
            if kind == "buffers":
                # Newly derived scales are registered by the quant method on cold
                # start; reject collisions with methods, Parameters or private state.
                if name not in declared and hasattr(leaf, name):
                    raise ValueError(f"Unexpected resident buffer: {prefix}{name}")
                leaf.register_buffer(name, value)
            else:
                if name not in declared:
                    raise ValueError(f"Undeclared resident Parameter: {prefix}{name}")
                if declared[name] is None:
                    leaf.register_parameter(
                        name, torch.nn.Parameter(value, requires_grad=False)
                    )
                else:
                    declared[name].data = value
        # A quantizer may release its scratch scales once packed.
        for name in set(declared) - fields.keys():
            if declared[name] is None or (
                kind == "buffers" and name in leaf._non_persistent_buffers_set
            ):
                continue
            if kind == "buffers" and name in getattr(
                method, "resident_released_buffers", ()
            ):
                delattr(leaf, name)
            else:
                raise ValueError(f"Missing resident {kind}: {prefix}{name}")
    if isinstance(leaf, BaseMoEExperts):
        count, projections, aux, ready = facts["expert"]
        if (
            count != leaf.num_local_experts
            or tuple(projections) != tuple(leaf.PROJ_NAMES)
            or aux != sorted(leaf._required_aux_param_names())
        ):
            raise ValueError(f"Resident expert placement changed: {prefix}")
        leaf._loaded_keys = {(i, p) for i in range(count) for p in projections}
        leaf._loaded_aux_keys = {(*key, a) for key in leaf._loaded_keys for a in aux}
        leaf._loaded_count = len(leaf._loaded_keys)
        leaf._expert_layout_ready = ready
    else:
        leaf._loaded_parameter_names = set(facts["loaded"])
        leaf._main_weight_loaded = bool(
            leaf._loaded_parameter_names & {"weight", "qweight"}
        )
        leaf._post_load_done = True
    if "integer" in facts:
        from rtp_llm.models_py.quant_methods.groupwise_runtime import (
            GroupwiseLayout,
            GroupwiseRuntimeState,
        )

        states = {}
        for key, (layout, bits, reason) in facts["integer"].items():
            if key == "linear" and isinstance(leaf, LinearBase):
                weight, scale, zero = leaf.qweight, leaf.scales, leaf.qzeros
            elif key in ("w13", "w2") and isinstance(leaf, BaseMoEExperts):
                raw = layout["method"] == "INT8"
                weight = getattr(leaf, key)
                scale = getattr(leaf, key + ("_int8_scale" if raw else "_scales"))
                zero = getattr(leaf, key + ("_int8_zero" if raw else "_qzeros"))
            else:
                raise ValueError(f"Invalid resident integer projection {key}")
            states[key] = GroupwiseRuntimeState(
                weight, scale, zero, GroupwiseLayout(**layout), bits, reason
            )
        if isinstance(leaf, LinearBase):
            leaf._groupwise_runtime = states["linear"]
            leaf._groupwise_runtime_metadata = states["linear"].metadata
        else:
            leaf._groupwise_runtime = states
            leaf._groupwise_runtime_metadata = {
                k: v.metadata for k, v in states.items()
            }


class ResidentDecoderOwner:
    resident_after_post_load = False

    def prepare_resident_weights(self):
        # This hook is the owner's head normalization/scaling, not a recursive
        # module walk. Applying it after attach would write into read-only VMM.
        finalize_head = getattr(self, "_finalize_resident_head", None)
        if finalize_head is not None:
            finalize_head()
        for leaf in self.modules():
            if getattr(leaf, "_rtp_borrowed_parameter_names", ()):
                continue
            if isinstance(leaf, BaseMoEExperts):
                if leaf.fused_moe is not None:
                    raise RuntimeError("Resident expert already has a live executor")
                leaf.process_weights_during_staged_load()
            elif isinstance(leaf, LinearBase):
                leaf.process_weights_after_loading()
            pack = getattr(leaf, "prepare_resident_projection", None)
            if pack is not None:
                pack()

    def resident_constructor_state(self):
        head = getattr(self, "lm_head", None)
        heads = getattr(self, "handler_heads", None)
        layouts, indices, owners = [], {}, {}
        for name, leaf in self.named_modules():
            if not isinstance(leaf, (LinearBase, BaseMoEExperts)) or getattr(
                leaf, "_rtp_borrowed_parameter_names", ()
            ):
                continue
            facts = _layout(leaf)
            # Most layers have exactly the same geometry. Store it once,
            # not hundreds of copies in the store's control-plane message.
            key = json.dumps(facts, sort_keys=True)
            if key not in indices:
                indices[key] = len(layouts)
                layouts.append(facts)
            owners[name] = indices[key]
        return {
            "ft_unknown_tied_vocab": getattr(self, "_ft_unknown_tied_vocab", False),
            "handlers": {} if heads is None else heads.resident_shapes(),
            "head": (
                None
                if head is None
                else [head._explicit_checkpoint_loaded, head._tied_checkpoint_loaded]
            ),
            "leaves": owners,
            "layouts": layouts,
            "projections": {
                n: m.resident_projection_facts()
                for n, m in self.named_modules()
                if hasattr(m, "resident_projection_facts")
            },
            "packed_gdn": [
                name
                for name, leaf in self.named_modules()
                if getattr(leaf, "_input_projection_packed", False)
            ],
        }

    def prepare_resident_checkpoint_restore(self, state, values):
        heads = getattr(self, "handler_heads", None)
        if heads is not None:
            heads.bind_resident_parameters_from_record(state["handlers"], values)
        self._ft_unknown_tied_vocab = state["ft_unknown_tied_vocab"]
        self._lm_head_postprocessed = True
        head = getattr(self, "lm_head", None)
        if (head is None) != (state["head"] is None):
            raise ValueError("Resident decoder head ownership changed")
        if head is not None:
            (
                self.lm_head._explicit_checkpoint_loaded,
                self.lm_head._tied_checkpoint_loaded,
            ) = state["head"]
        leaves = dict(self.named_modules())
        for name, spec in state["projections"].items():
            leaves[name].restore_resident_projection(spec, values, name + ".")
        expected = {
            name
            for name, leaf in leaves.items()
            if isinstance(leaf, (LinearBase, BaseMoEExperts))
            and not getattr(leaf, "_rtp_borrowed_parameter_names", ())
        }
        if expected != state["leaves"].keys():
            raise ValueError("Resident decoder parameter owners changed")
        for name, index in state["leaves"].items():
            if type(index) is not int or not 0 <= index < len(state["layouts"]):
                raise ValueError(f"Invalid resident layout index: {name}")
            _restore_layout(leaves[name], state["layouts"][index], values, name + ".")
        for name in state["packed_gdn"]:
            leaf = leaves[name]
            if not hasattr(leaf, "prepare_resident_projection"):
                raise ValueError("Invalid resident packed GDN owner")
            leaf._input_projection_packed = True

    def finish_resident_checkpoint_restore(self):
        # Validate actual registrations after bind, before constructing any kernel view.
        for leaf in self.modules():
            if (
                getattr(leaf, "_input_projection_packed", False)
                and leaf._packed_input_weight() is None
            ):
                raise ValueError("Resident GDN input projections lost storage sharing")

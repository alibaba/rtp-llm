"""Construction facts and direct runtime weight ownership for resident Qwen4."""

from pathlib import Path

import torch

from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from .nvfp4_experts import Qwen4Nvfp4Experts
from .ple_weights import Qwen4PLEHostTable


_PLAN_FIELDS = (
    "source_config",
    "prefix",
    "num_layers",
    "hidden_size",
    "hc_mult",
    "vocab_size",
    "layer_types",
    "ple_layers",
    "ple_prefetch_mb",
    "source_kind",
    "source_fp8",
    "dense_fp8",
    "vocab_fp8",
    "dense_storage",
    "vocab_storage",
)


class Qwen4ResidentOwner:
    """Recreate the target/draft construction plan without touching HF shards."""

    @staticmethod
    def _resident_plan(state, directory, config):
        if state is None:
            return None
        from .target_weights import Qwen4CheckpointPlan

        plan = Qwen4CheckpointPlan.__new__(Qwen4CheckpointPlan)
        facts = state["plan"]
        for key in _PLAN_FIELDS:
            setattr(plan, key, facts[key])
        plan.root, plan.model_config = Path(directory), config
        plan.text = plan.source_config.get("text_config", plan.source_config)
        plan.shapes = {k: tuple(v) for k, v in facts["shapes"].items()}
        return plan

    def resident_constructor_state(self):
        plan = self.checkpoint_plan
        facts = {key: getattr(plan, key) for key in _PLAN_FIELDS}
        facts["shapes"] = {
            k: value
            for k, value in plan.shapes.items()
            if k.endswith(
                (
                    "shared_expert.gate_proj.weight",
                    "shared_expert_gate.weight",
                    "input_mix_weight_down.weight",
                    "lm_head.weight",
                )
            )
        }
        tables, experts, marlin = {}, {}, {}
        for name, leaf in self.named_modules():
            if isinstance(leaf, Qwen4PLEHostTable):
                leaf.validate_weights_loaded()
                leaf.process_weights_after_loading()
                tables[name] = [
                    leaf.shard_rows,
                    sorted(leaf._scaled),
                    leaf._ft_predequantized,
                ]
            elif isinstance(leaf, Qwen4Nvfp4Experts):
                leaf.validate_weights_loaded()
                marlin[name] = sorted(leaf._loaded)
            elif isinstance(leaf, BaseMoEExperts):
                leaf.validate_weights_loaded()
                # Record the final quantized tensors, before any executor captures
                # their addresses. The ordinary post-load pass will bind it.
                if leaf.fused_moe is not None:
                    raise RuntimeError(
                        "Qwen4 resident relocation must precede MoE executor construction"
                    )
                leaf.process_weights_during_staged_load()
                experts[name] = {
                    "weights": len(leaf._loaded_keys),
                    "scales": len(leaf._loaded_aux_keys),
                    "ready": leaf._expert_layout_ready,
                }
        heads = getattr(self, "handler_heads", None)
        return {
            "plan": facts,
            "tables": tables,
            "experts": experts,
            "marlin": marlin,
            "handlers": {} if heads is None else heads.resident_shapes(),
        }

    def prepare_resident_checkpoint_restore(self, state, values):
        heads = getattr(self, "handler_heads", None)
        if heads is not None:
            heads.bind_resident_parameters_from_record(state["handlers"], values)
        for name, leaf in self.named_modules():
            if isinstance(leaf, Qwen4PLEHostTable):
                rows, scaled, predequantized = state["tables"][name]
                leaf._scaled = set(scaled)
                leaf._ft_predequantized = predequantized
                if leaf.source_fp8 and not predequantized:
                    leaf._seen.add("weight_scale")
                elif predequantized:
                    leaf.weight_scale = None
                dtype = torch.float8_e4m3fn if leaf.preserve_fp8 else leaf.compute_dtype
                for i in leaf.shard_indices:
                    field = f"shard_{i}"
                    value = values[name + "." + field]
                    if (
                        tuple(value.shape) != (rows, leaf.embedding_dim)
                        or value.dtype != dtype
                        or value.device.type != "cpu"
                    ):
                        raise ValueError(f"Invalid resident PLE shard: {name}.{field}")
                    # The constructor registered a zero-row Parameter. Do not allocate
                    # a second full-size table just to validate and bind this view.
                    leaf._parameters[field].data = value
                    leaf._seen.add(field + ".weight")
                leaf.shard_rows = rows
            elif isinstance(leaf, Qwen4Nvfp4Experts):
                leaf._loaded = set(state["marlin"][name])
            elif isinstance(leaf, BaseMoEExperts):
                facts = state["experts"][name]
                leaf._loaded_keys = {
                    (i, proj)
                    for i in range(leaf.num_local_experts)
                    for proj in leaf.PROJ_NAMES
                }
                leaf._loaded_aux_keys = {
                    (*key, aux)
                    for key in leaf._loaded_keys
                    for aux in leaf._required_aux_param_names()
                }
                leaf._loaded_count = len(leaf._loaded_keys)
                if (leaf._loaded_count, len(leaf._loaded_aux_keys)) != (
                    facts["weights"],
                    facts["scales"],
                ):
                    raise ValueError(
                        "Resident Qwen4 expert layout differs from this rank"
                    )
                leaf._expert_layout_ready = facts["ready"]
                if (
                    leaf._expert_layout_ready
                    and self.checkpoint_plan.source_kind != "bf16"
                ):
                    # TRTLLM and CUTLASS own packed shapes; register only slots
                    # declared by this quant method, never arbitrary manifest keys.
                    for field in ("w13", "w2"):
                        param = leaf._parameters[field]
                        packed = values[name + "." + field]
                        if (
                            packed.dtype != param.dtype
                            or packed.numel() != param.numel()
                        ):
                            raise ValueError(
                                f"Invalid packed NVFP4 expert geometry: {field}"
                            )
                        param.data = packed
                    for field in (
                        "w13_scale",
                        "w2_scale",
                        *leaf.quant_method._derived_names(),
                    ):
                        value = values[name + "." + field]
                        leaf.register_buffer(field, value)

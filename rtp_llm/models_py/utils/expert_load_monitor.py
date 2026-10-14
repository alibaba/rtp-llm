"""Instance-owned counters for routed physical expert activations.

The engine owns counter lifetime and clears/reduces the buffers on the model
execution stream. These helpers only add assignments, including routed padding.
"""

from __future__ import annotations

from typing import Optional

import torch


def record_expert_load(topk_ids: torch.Tensor, counts: Optional[torch.Tensor]) -> None:
    """Add a routing tensor's histogram to one layer's persistent CUDA buffer."""
    if counts is None:
        return
    # Native metadata validation and launch avoid per-layer Triton dispatch
    # overhead in eager forwards. Disabled monitoring never loads the binding.
    from rtp_llm.ops import compute_ops

    compute_ops.record_expert_load(topk_ids, counts)


def _bind_router(router, counts: torch.Tensor, model_tp_rank: int) -> None:
    config = router.config
    logical_experts = int(config.expert_num)
    physical_experts = int(getattr(config, "physical_expert_num", logical_experts))
    if physical_experts != logical_experts:
        raise NotImplementedError(
            "expert monitoring requires final physical routing IDs; this Python "
            "router does not implement redundant-expert routing"
        )
    if physical_experts != counts.numel():
        raise ValueError("monitor expert dimension does not match the router")
    module_name = type(router).__module__
    supported = {
        "pure_tp_router",
        "pure_dp_router",
        "pure_cp_router",
        "deepep_normal_router",
        "deepep_low_latency_router",
        "fp8_fp4_router",
    }
    batched_router = module_name.endswith(".impl.common.router.batched_data_router")
    cuda_router = (
        ".impl.cuda.routers." in module_name
        and module_name.rsplit(".", 1)[-1] in supported
    )
    if not (cuda_router or batched_router):
        raise NotImplementedError(
            f"expert monitoring is not implemented for {type(router).__name__}"
        )
    # Pure-TP inputs are replicated. The other routers count unique source
    # shards (before all-gather, or after DeepEP's TP slice).
    replicated = module_name.endswith(".pure_tp_router") or batched_router
    owner = not replicated or int(config.tp_rank) == 0
    if module_name.endswith(".fp8_fp4_router") and int(config.ep_size) == 1:
        # FP8/FP4 runtime config describes the executor and fixes tp_rank=0.
        # EP1 input replication belongs to the enclosing model's attention TP;
        # its effective rank is zero on every independent CP source.
        owner = model_tp_rank == 0
    router._expert_load_counts = counts if owner else None


def bind_expert_load_monitor(model, step_counts: torch.Tensor) -> None:
    """Bind once after model initialization and before graph capture.

    ``step_counts`` uses global model-layer and physical-expert indices. A
    different buffer cannot replace it: captured graphs retain its address.
    """
    if torch.version.hip is not None:
        raise NotImplementedError(
            "expert load monitoring currently supports NVIDIA CUDA only"
        )
    if (
        step_counts.ndim != 2
        or step_counts.dtype != torch.int64
        or not step_counts.is_contiguous()
        or not step_counts.is_cuda
        or min(step_counts.shape) <= 0
    ):
        raise ValueError("step_counts must be contiguous CUDA int64 [layers, experts]")
    previous = getattr(model, "_expert_load_monitor_step_counts", None)
    if previous is not None:
        if (
            previous.data_ptr() != step_counts.data_ptr()
            or previous.shape != step_counts.shape
            or previous.device != step_counts.device
        ):
            raise RuntimeError("expert monitor cannot be rebound to a different buffer")
        # A pybind cast may return a fresh Python wrapper for the same storage.
        # Revisit modules in case initialization rebuilt the decoder layers.

    # GenericMoE/Qwen3Next/KimiLinear put MoE children under model.layers but
    # GenericMoeLayer has no layer index itself. FP8/FP4 and DSV4.1 additionally
    # expose a layer_id on their MoE module.
    layers = getattr(model, "layers", None)
    roots = enumerate(layers) if layers is not None else [(None, model)]
    args = getattr(model, "_v4_args", getattr(model, "args", None))
    model_tp_rank = int(getattr(args, "tp_rank", 0))
    found = False
    seen = set()
    for index, layer in roots:
        default_index = getattr(layer, "layer_idx", index)
        for module in layer.modules():
            layer_index = getattr(module, "layer_id", default_index)
            fused_moe = getattr(module, "fused_moe", None)
            legacy = type(
                module
            ).__module__ == "rtp_llm.models_py.modules.dsv41.moe.moe_layer" and hasattr(
                module, "_strategy"
            )
            if fused_moe is None and not legacy:
                continue
            if layer_index is None:
                raise NotImplementedError(
                    "cannot infer MoE layer index without model.layers or layer_id"
                )
            layer_index = int(layer_index)
            if not 0 <= layer_index < step_counts.size(0):
                raise ValueError(
                    f"MoE layer index {layer_index} is outside monitor buffer"
                )
            counts = step_counts[layer_index]
            if fused_moe is not None:
                router = fused_moe.router
                if id(router) in seen:
                    continue
                _bind_router(router, counts, model_tp_rank)
                # Fused gate packing materializes IDs inside the executor.
                fused_moe.fused_experts._expert_load_counts = router._expert_load_counts
                seen.add(id(router))
            else:
                if int(module.n_routed_experts) != counts.numel():
                    raise ValueError("DSV4.1 expert dimension does not match monitor")
                # The legacy EP=1 path computes complete experts on replicated
                # attention-TP inputs. Its effective V4Args TP rank already
                # becomes zero on every CP source (distinct token shards).
                owner = int(module.ep_size) > 1 or model_tp_rank == 0
                module._expert_load_counts = counts if owner else None
                module._strategy._expert_load_counts = module._expert_load_counts
            found = True
    if not found:
        raise NotImplementedError(
            "no supported MoE modules found for expert monitoring"
        )
    model._expert_load_monitor_step_counts = step_counts

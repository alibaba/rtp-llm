"""Qwen checkpoint storage, compute copies and model-local linear assembly.

Storage owners remain alive with the compute view. Updating graph-bound FP8 or
copied parameters requires restarting the model; no view advertises live update.
"""

import copy

import torch

from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory
from rtp_llm.utils.fp8_constants import FP8_STORAGE_SCALE_SUFFIX
from rtp_llm.utils.model_weight import W


@dataclass(frozen=True)
class WeightView:
    name: str
    storage: torch.Tensor
    scale: torch.Tensor | None
    compute: torch.Tensor
    update_requires_restart: bool
    alias_owner: torch.Tensor | None = None


@dataclass(frozen=True)
class ComputeWeightInventory:
    source_format: str
    source_owner: object
    layers: tuple[tuple[WeightView, ...], ...]
    global_weights: tuple[WeightView, ...]


def summarize_compute_inventory(inventory: ComputeWeightInventory):
    """Describe resident weight storage without reading values or synchronizing CUDA.

    Count backing allocations, including padding, once across views and layers.
    This excludes allocator reserves and non-weight runtime workspaces.
    """
    entries = tuple(entry for layer in inventory.layers for entry in layer)
    entries += inventory.global_weights

    def storage_id(tensor):
        storage = tensor.untyped_storage()
        return str(tensor.device), storage.data_ptr(), storage.nbytes()

    source = {}
    for entry in entries:
        for tensor in (entry.storage, entry.scale, entry.alias_owner):
            if tensor is not None:
                source.setdefault(storage_id(tensor), tensor)
    extra_compute = {}
    for entry in entries:
        key = storage_id(entry.compute)
        if key not in source:
            extra_compute.setdefault(key, entry.compute)

    def describe(storages):
        groups = {}
        for (device, _, size), tensor in storages.items():
            key = (device, str(tensor.dtype))
            group = groups.setdefault(key, {"bytes": 0, "allocations": 0})
            group["bytes"] += size
            group["allocations"] += 1
        return [
            {"device": device, "dtype": dtype, **value}
            for (device, dtype), value in sorted(groups.items())
        ]

    roles = {
        W.moe_w1: "routed_up_gate",
        W.moe_w2: "routed_down",
        W.moe_s1: "routed_up_gate_scale",
        W.moe_s2: "routed_down_scale",
        W.qwen4_ple_ngram_scale: "ple_global_scale",
    }
    layouts = {}
    for entry in entries:
        role = roles.get(entry.name)
        if entry.name.startswith(W.qwen4_ple_ngram_shards + "."):
            role = "ple_shard"
        if role is None:
            continue
        tensor = entry.storage
        key = (role, str(tensor.device), str(tensor.dtype), tuple(tensor.shape))
        layouts[key] = layouts.get(key, 0) + 1
    return {
        "source_format": inventory.source_format,
        "layers": len(inventory.layers),
        "source_storage": describe(source),
        "extra_compute_storage": describe(extra_compute),
        "layouts": [
            {
                "role": role,
                "device": device,
                "dtype": dtype,
                "shape": list(shape),
                "tensors": count,
            }
            for (role, device, dtype, shape), count in sorted(layouts.items())
        ],
    }


class PreparedWeightDict(dict):
    """Compute weights whose copied projections require a model restart to update."""

    def __init__(self, owner=None):
        super().__init__(owner or {})
        self._storage_owner = owner if owner is not None else {}
        self.consumer_views = {}

    def __setitem__(self, name, value):
        original = self._storage_owner.get(name)
        if original is not None and original is not value:
            # WeightManager retains the owner dictionary. Rebinding a compute
            # entry must also prevent updates to its now detached source.
            original._rtp_requires_restart_for_update = True
            value._rtp_requires_restart_for_update = True
        super().__setitem__(name, value)


def prepare_compute_weights(weights):
    view = copy.copy(weights)
    view._fp8_storage_owner = weights

    def convert(items):
        result = PreparedWeightDict(items)
        for name, value in items.items():
            scale = items.get(name + FP8_STORAGE_SCALE_SUFFIX)
            compute = (
                (value.float() * scale).to(torch.bfloat16)
                if value.dtype == torch.float8_e4m3fn
                and value.ndim != 2
                and scale is not None
                else value
            )
            restart = value.dtype == torch.float8_e4m3fn or compute is not value
            if restart:
                value._rtp_requires_restart_for_update = True
                compute._rtp_requires_restart_for_update = True
            result[name] = compute
        return result

    view.weights = [convert(items) for items in weights.weights]
    view.global_weights = convert(weights.global_weights)
    return view


def mark_projection_for_restart(weights, module):
    """Mark model-local FP8 consumers and BF16 copies as requiring restart."""
    if not isinstance(weights, PreparedWeightDict) or module is None:
        return
    compute = (
        module.compute_weight if module.compute_weight is not None else module.weight
    )
    compute._rtp_requires_restart_for_update = True


def record_projection_consumer(weights, name, module):
    if not isinstance(weights, PreparedWeightDict) or module is None:
        return
    mark_projection_for_restart(weights, module)
    compute = module.compute_weight if module.compute_weight is not None else module.weight
    weights.consumer_views[name] = WeightView(
        name,
        weights._storage_owner.get(name, weights[name]),
        weights.get(name + FP8_STORAGE_SCALE_SUFFIX),
        compute,
        True,
        getattr(module.weight, "_base", None),
    )


def annotate_compute_inventory(weights):
    source_owner = getattr(weights, "_fp8_storage_owner", weights)

    def inventory(items, source):
        consumers = getattr(items, "consumer_views", {})
        return tuple(
            consumers[name]
            if name in consumers
            else WeightView(
                name,
                source.get(name, value),
                source.get(name + FP8_STORAGE_SCALE_SUFFIX),
                value,
                bool(getattr(value, "_rtp_requires_restart_for_update", False)),
                getattr(value, "_base", None),
            )
            for name, value in items.items()
        )

    weights.compute_inventory = ComputeWeightInventory(
        "FP8" if source_owner is not weights else "BF16",
        source_owner,
        tuple(inventory(items, source) for items, source in zip(weights.weights, source_owner.weights)),
        inventory(weights.global_weights, source_owner.global_weights),
    )
    return weights.compute_inventory


class Qwen4LinearFactory:
    @classmethod
    def create_linear_from_weights(
        cls,
        weights,
        weight_key,
        scale_key=None,
        bias_key=None,
        quant_config=None,
        hw_kernel_config=None,
        weight_scale_2_key=None,
        input_scale_key=None,
    ):
        storage_scale = weights.get(weight_key + FP8_STORAGE_SCALE_SUFFIX)
        if storage_scale is not None:
            module = Fp8Linear(
                weights[weight_key].t().contiguous(),
                storage_scale,
                weights.get(bias_key) if bias_key else None,
            )
            record_projection_consumer(weights, weight_key, module)
            return module
        return LinearFactory.create_linear_from_weights(
            weights,
            weight_key,
            scale_key,
            bias_key,
            quant_config,
            hw_kernel_config,
            weight_scale_2_key,
            input_scale_key,
        )

    create_linear = LinearFactory.create_linear
    create_merged_linear = LinearFactory.create_merged_linear

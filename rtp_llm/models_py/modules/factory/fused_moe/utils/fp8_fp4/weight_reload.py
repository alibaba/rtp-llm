"""Replay checkpoint-to-MoE adaptation without replacing resident storages."""

from __future__ import annotations

import weakref
from typing import Mapping

import torch

from rtp_llm.model_loader.weight_memory_saver import current_model_scope
from rtp_llm.model_loader.weight_memory_saver import is_enabled as sleep_enabled
from rtp_llm.model_loader.weight_memory_saver import suppress_weights_region
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.weight_adapter import (
    adapt_split_moe_weights,
)

_RELOADERS: weakref.WeakSet = weakref.WeakSet()


def copy_tensors_in_place(live: Mapping, fresh: Mapping) -> None:
    """Validate the whole tensor set before writing; never rebind graph aliases."""
    if live.keys() != fresh.keys():
        raise RuntimeError(
            f"MoE reload keys differ: live={sorted(live)}, fresh={sorted(fresh)}"
        )
    for name, destination in live.items():
        source = fresh[name]
        if (
            destination.shape != source.shape
            or destination.dtype != source.dtype
            or destination.device != source.device
        ):
            raise RuntimeError(
                f"MoE reload mismatch for {name}: "
                f"{destination.shape}/{destination.dtype}/{destination.device} vs "
                f"{source.shape}/{source.dtype}/{source.device}"
            )
    for name, destination in live.items():
        destination.copy_(fresh[name])


class SplitMoeWeightReload:
    """Own the model-specific naming adapter, not a second execution path."""

    def __init__(self, layer, live_weights, raw_names, required, inter_dim, shared):
        self.layer_id = layer.layer_id
        self._layer = weakref.ref(layer)
        self._live_weights = live_weights
        self._raw_names = dict(raw_names)
        self.required_names = frozenset(required)
        self._inter_dim = inter_dim
        self._shared = shared
        self._sleep_model_scope = current_model_scope()

    def reload_weights(self, raw: Mapping) -> set[str]:
        if raw.keys() != self.required_names:
            raise RuntimeError(
                f"MoE layer {self.layer_id} reload coverage mismatch: "
                f"missing={sorted(self.required_names - raw.keys())}, "
                f"extra={sorted(raw.keys() - self.required_names)}"
            )
        layer = self._layer()
        if layer is None:
            raise RuntimeError(f"MoE reload owner expired for layer {self.layer_id}")
        # Exactly the cold-load name/layout adapter, but scratch is not pausable.
        with suppress_weights_region(), torch.inference_mode():
            canonical = adapt_split_moe_weights(
                dict(raw), self._inter_dim, self._shared, self._raw_names
            )
            retained = {
                name: self._live_weights[name]
                for name in canonical
                if name in self._live_weights
            }
            copy_tensors_in_place(retained, {k: canonical[k] for k in retained})
            layer.reload_weights(canonical)
        return set(retained)


def register_split_moe_reload(
    layer, live_weights, raw_names, required, inter_dim, shared
) -> None:
    if not sleep_enabled():
        return
    reload = SplitMoeWeightReload(
        layer, live_weights, raw_names, required, inter_dim, shared
    )
    # Layer owns the callback; registry is weak and callback refers back weakly.
    layer._sleep_weight_reload = reload
    _RELOADERS.add(reload)


def iter_moe_reloaders() -> list:
    return sorted(list(_RELOADERS), key=lambda item: item.layer_id)

"""Opt-in 78 -> 105 GLM5 execution plan, with checkpoint weights shared.

GLM5_REPEAT_MOE_LAYERS=middle27 repeats original layers 27..53 immediately.
Alternatively supply exactly 27 distinct, zero-based MoE layer IDs as CSV.
The runtime config/cache uses logical IDs; checkpoint loading uses source IDs.
"""

import logging
import os
from types import FunctionType, MethodType

from rtp_llm.utils.dsa_indexing import dsa_layer_has_indexer


def configure_glm5_layer_repeat(config):
    selection = os.environ.get("GLM5_REPEAT_MOE_LAYERS", "").strip()
    if not selection or getattr(config, "is_mtp", False):
        return
    if getattr(config, "_glm5_layer_source_ids", None):
        if config.num_layers != 105:
            raise ValueError("Repeated GLM5 execution plan must have 105 layers")
        return
    if config.model_type != "glm_5" or config.num_layers != 78:
        raise ValueError("GLM5_REPEAT_MOE_LAYERS requires a 78-layer glm_5 target")
    if not config.attn_config.use_mla or not config.attn_config.is_sparse:
        raise ValueError("GLM5 layer repetition requires sparse MLA")
    if config.eplb_config.enable_eplb():
        raise ValueError("Repeated GLM5 weights do not support dynamic EPLB")
    moe_layers = list(config.moe_layer_index)
    if selection == "middle27":
        start = (len(moe_layers) - 27) // 2
        repeated = moe_layers[start : start + 27]
    else:
        try:
            repeated = [int(value.strip()) for value in selection.split(",")]
        except ValueError as error:
            raise ValueError(
                "Use middle27 or 27 comma-separated MoE layer IDs"
            ) from error
    if len(repeated) != 27 or len(set(repeated)) != 27:
        raise ValueError("Exactly 27 distinct MoE layers must be repeated")
    if not set(repeated).issubset(moe_layers):
        raise ValueError("Only existing MoE layer IDs may be repeated; no Dense layers")
    source_types = [
        "full" if dsa_layer_has_indexer(config, i) else "shared" for i in range(78)
    ]
    source_ids = [i for i in range(78) for _ in range(2 if i in repeated else 1)]
    config._glm5_checkpoint_num_layers = 78
    config._glm5_checkpoint_moe_layers = moe_layers
    config._glm5_checkpoint_indexer_types = source_types
    config._glm5_layer_source_ids = source_ids
    config.num_layers = len(source_ids)
    config.moe_layer_index = [
        i for i, source in enumerate(source_ids) if source in moe_layers
    ]
    # Materialize the original pattern before inserting layers. Applying the old
    # frequency to logical IDs would silently change which weights have indexers.
    config.indexer_types = [source_types[source] for source in source_ids]
    logging.info(
        "GLM5 repeated layers: source_ids=%s, logical_layers=%d",
        source_ids,
        config.num_layers,
    )


class _CheckpointConfigView:
    """Read checkpoint topology without mutating the live C++ runtime config."""

    def __init__(self, config):
        object.__setattr__(self, "_config", config)
        object.__setattr__(
            self,
            "_overrides",
            {
                "num_layers": config._glm5_checkpoint_num_layers,
                "moe_layer_index": config._glm5_checkpoint_moe_layers,
                "indexer_types": config._glm5_checkpoint_indexer_types,
            },
        )

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        value = getattr(self._config, name)
        # Python sizing helpers must also see the checkpoint's 78 layers.
        if isinstance(value, MethodType) and isinstance(value.__func__, FunctionType):
            return value.__func__.__get__(self)
        return value

    def __setattr__(self, name, value):
        if name in self._overrides:
            self._overrides[name] = value
        else:
            setattr(self._config, name, value)


def checkpoint_layer_config(config):
    if not getattr(config, "_glm5_layer_source_ids", None):
        return config
    return _CheckpointConfigView(config)


def expand_repeated_layer_weights(config, weights):
    source_ids = getattr(config, "_glm5_layer_source_ids", None)
    if not source_ids:
        return
    if len(weights.weights) != config._glm5_checkpoint_num_layers:
        raise ValueError(
            "Load original checkpoint layers before expanding shared weights"
        )
    source_weights = weights.weights
    # Alias dictionaries too: initialization can replace/pop transformed weights.
    weights.weights = [source_weights[source] for source in source_ids]


def build_execution_layers(config, create_layer, repeat_layer):
    source_ids = getattr(config, "_glm5_layer_source_ids", None)
    if not source_ids:
        return [create_layer(i) for i in range(config.num_layers)]
    if len(source_ids) != config.num_layers:
        raise ValueError("Layer execution plan does not match runtime num_layers")
    originals = {}
    layers = []
    for logical_id, source_id in enumerate(source_ids):
        if source_id in originals:
            layer = repeat_layer(originals[source_id], logical_id)
        else:
            layer = create_layer(logical_id)
            originals[source_id] = layer
        layers.append(layer)
    return layers

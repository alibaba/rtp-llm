"""Run a 28-layer GLM5 block twice with shared checkpoint weights.

The checkpoint's glm5_loop_layer_range contains zero-based [start, stop)
source layer IDs. Each logical pass stores its own attention KV because its
inputs differ. This ungated loop is for checkpoint-based experiments.
"""

import logging
from types import FunctionType, MethodType

from rtp_llm.utils.dsa_indexing import dsa_layer_has_indexer


def configure_glm5_layer_repeat(config):
    block = getattr(config, "_glm5_loop_layer_range", None)
    if block is None or getattr(config, "is_mtp", False):
        return
    source_ids = getattr(config, "_glm5_layer_source_ids", None)
    if source_ids is not None:
        if config.num_layers != len(source_ids):
            raise ValueError("GLM5 loop execution plan does not match num_layers")
        return
    if config.model_type != "glm_5" or config.num_layers != 78:
        raise ValueError("glm5_loop_layer_range requires a 78-layer glm_5 target")
    if not config.attn_config.use_mla or not config.attn_config.is_sparse:
        raise ValueError("The GLM5 loop requires sparse MLA")
    if config.eplb_config.enable_eplb():
        raise ValueError("The GLM5 loop does not support dynamic EPLB")
    if getattr(config, "kv_cache_spec_descs", None):
        raise ValueError("Configure the GLM5 loop before building KV cache specs")
    if (
        not isinstance(block, (list, tuple))
        or len(block) != 2
        or any(type(layer_id) is not int for layer_id in block)
    ):
        raise ValueError("glm5_loop_layer_range must be a pair of integer layer IDs")
    start, stop = block
    if not 0 <= start < stop <= config.num_layers or stop - start != 28:
        raise ValueError("The GLM5 loop must contain 28 consecutive checkpoint layers")
    moe_layers = list(config.moe_layer_index)
    moe_ids = set(moe_layers)
    block_ids = range(start, stop)
    if not moe_ids.issuperset(block_ids):
        raise ValueError("The GLM5 loop must contain only MoE layers")
    source_types = [
        "full" if dsa_layer_has_indexer(config, i) else "shared"
        for i in range(config.num_layers)
    ]
    if source_types[0] != "full" or source_types[start] != "full":
        raise ValueError(
            "The GLM5 stack and each loop pass must start with a full DSA indexer"
        )
    # Insert the second pass after the first entire block. Its first indexer
    # recomputes top-k from the second pass state, not the first pass's tail.
    source_ids = (*range(stop), *block_ids, *range(stop, config.num_layers))
    config._glm5_checkpoint_num_layers = config.num_layers
    config._glm5_checkpoint_moe_layers = moe_layers
    config._glm5_checkpoint_indexer_types = source_types
    config._glm5_layer_source_ids = source_ids
    config.num_layers = len(source_ids)
    config.moe_layer_index = [
        i for i, source in enumerate(source_ids) if source in moe_ids
    ]
    # Materialize the original pattern before inserting layers. Applying the old
    # frequency to logical IDs would silently change which weights have indexers.
    config.indexer_types = [source_types[source] for source in source_ids]
    logging.info(
        "GLM5 loop layer range [%d, %d), passes=2, logical_layers=%d",
        start,
        stop,
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

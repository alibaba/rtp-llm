"""Shared FT shard serialization, independent of either model loader."""

import logging
import os
from collections import OrderedDict

from safetensors.torch import save_file


def write_ft_shards(
    records,
    output_dir,
    tp_rank=0,
    dp_rank=0,
    ep_rank=0,
    max_size=6 * 1024**3,
    after_tensor=None,
    head_provenance=None,
):
    os.makedirs(output_dir, exist_ok=True)
    filename_prefix = f"{output_dir}/model-{tp_rank:02d}-{dp_rank:02d}-"
    rank_prefix = f"rank_{tp_rank:02d}_{dp_rank:02d}_{ep_rank:02d}."
    pending, size, part = OrderedDict(), 0, 0
    serialized_keys, pending_storage = set(), set()

    def save_partition():
        path = f"{filename_prefix}part-{part:05d}.safetensors"
        source = (head_provenance or {}).get("source")
        metadata = None
        if source is not None and rank_prefix + "global.lm_head" in pending:
            if source not in ("lm_head", "embedding"):
                raise ValueError("Unknown FT LM-head source")
            metadata = {"ft_lm_head_source": source}
        for attempt in range(2):
            try:
                save_file(pending, path, metadata=metadata)
                return
            except Exception:
                if attempt == 1:
                    raise
                logging.warning("Retry writing FT partition %s", path)

    for layer_id, suffix, tensor in records:
        key = (
            rank_prefix
            + (f"layers.{layer_id}." if layer_id is not None else "global.")
            + suffix
        )
        if key in serialized_keys:
            raise ValueError(f"Duplicate FT tensor {key}")
        serialized_keys.add(key)
        value = tensor.cpu().contiguous()
        storage = value.untyped_storage().data_ptr()
        if storage in pending_storage:
            # A tied CPU embedding/head remains aliased after cpu().contiguous().
            # FT stores both explicit tensors; safetensors forbids shared storage.
            # Copy only the additional alias, not every exported parameter.
            value = value.clone()
        pending_storage.add(value.untyped_storage().data_ptr())
        pending[key] = value
        size += tensor.numel() * tensor.element_size()
        if size >= max_size:
            save_partition()
            pending, size, part = OrderedDict(), 0, part + 1
            pending_storage.clear()
        if after_tensor is not None:
            after_tensor()
    if pending:
        save_partition()

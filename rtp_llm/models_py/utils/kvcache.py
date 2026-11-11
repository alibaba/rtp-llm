"""Cache-group tags: how a module publishes them, and how Python reads them.

Every module owns its own cache pools, so the hard requirement is only that a
module names each of its groups with a tag no other module publishes.  The
content of the tag is free: names such as ``full``, ``swa_kv`` or
``mtp_default`` merely describe the group's business semantics.  Draft (MTP /
DSpark) modules adopt the ``mtp_`` prefix because it keeps ``mtp_full``
readable next to the main model's ``full``.

The declaration belongs to the model description: each draft model class calls
:func:`declare_draft_cache_tags` from its own ``_post_build_model_config``, so
the published tag names are an explicit part of the Python model description.
Nothing on the C++ side infers or rewrites them; ``CacheConfig`` only validates
that two modules never publish the same tag.

Python consumers keep addressing a pool by its canonical name, so the module
prefix must not leak into the kernels: :class:`ModuleLocalKVCache` is the
boundary that reports canonical group tags and maps a canonical tag back onto
the published one.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import torch

from rtp_llm.ops.compute_ops import KVCache, LayerKVCache

DRAFT_CACHE_TAG_PREFIX = "mtp_"


def canonical_cache_tag(tag: str) -> str:
    """Strip the module prefix from a published cache-group tag.

    The main model publishes the canonical names unchanged, so this is the
    identity there; a draft module publishes ``<prefix><tag>``, and its
    Python consumers still address the pool by ``<tag>``.
    """
    tag = str(tag)
    if tag.startswith(DRAFT_CACHE_TAG_PREFIX):
        return tag[len(DRAFT_CACHE_TAG_PREFIX) :]
    return tag


def declare_draft_cache_tags(
    model_config: Any, prefix: str = DRAFT_CACHE_TAG_PREFIX
) -> None:
    """Publish this draft module's cache tags under ``prefix``.

    Idempotent: tags that already carry the prefix are left untouched.  Also
    records ``model_config.cache_tag_prefix`` so the model description states
    which prefix its groups were published under.
    """
    descs = getattr(model_config, "kv_cache_spec_descs", None)
    if not descs:
        return

    declared = []
    for layer_descs in descs:
        for desc in layer_descs:
            tag = str(desc.tag)
            if not tag.startswith(prefix):
                desc.tag = prefix + tag
            declared.append(str(desc.tag))

    model_config.cache_tag_prefix = prefix
    logging.info(
        "draft module declares module-scoped cache tags: %s",
        sorted(set(declared)),
    )


class ModuleLocalKVCache:
    """Module-local view of the framework ``KVCache`` handle.

    Each cache module publishes its own group tags. A draft module prefixes
    them with ``DRAFT_CACHE_TAG_PREFIX`` so the engine's tag -> module mapping
    stays bijective, while Python consumers keep addressing a module's pools by
    the canonical names (``swa_kv``, ``csa_kv``, ...). The view is installed
    once, where the handle enters Python, and translates at that boundary:
    reported group tags are canonical, and a canonical tag is mapped onto the
    published one before it reaches the engine.

    ``LayerKVCache.tag`` stays as published: the tensors are handed back to the
    C++ cache-store writer, which resolves the group on the engine side.
    """

    def __init__(self, kv_cache: KVCache) -> None:
        self._kv_cache = kv_cache
        # Topology is fixed for the allocator's lifetime, so resolve the two
        # names once instead of per layer per decode step.
        published = [str(tag) for tag in kv_cache.group_tags or []]
        self._group_tags = [canonical_cache_tag(tag) for tag in published]
        self._published_tags = {canonical_cache_tag(tag): tag for tag in published}

    @property
    def group_tags(self) -> list[str]:
        return self._group_tags

    @property
    def layer_count(self) -> int:
        return self._kv_cache.layer_count

    def _published_tag(self, tag: str) -> str:
        return self._published_tags.get(str(tag), str(tag))

    def get_layer_cache(self, layer_id: int, tag: str | None = None) -> LayerKVCache:
        if tag is None:
            return self._kv_cache.get_layer_cache(layer_id)
        return self._kv_cache.get_layer_cache(layer_id, self._published_tag(tag))

    def get_layer_cache_groups(self, layer_id: int) -> list[LayerKVCache]:
        return self._kv_cache.get_layer_cache_groups(layer_id)

    def get_seq_size_per_block(self, tag: str) -> int:
        return self._kv_cache.get_seq_size_per_block(self._published_tag(tag))

    def get_kernel_seq_size_per_block(self, tag: str) -> int:
        return self._kv_cache.get_kernel_seq_size_per_block(self._published_tag(tag))

    def get_entries_per_block(self, tag: str) -> int:
        return self._kv_cache.get_entries_per_block(self._published_tag(tag))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._kv_cache, name)


class SingleGroupKVCacheAdapter:
    """Small Python-owned cache used by standalone and CUDA-graph tests.

    Production whole-model caches are created only by C++ from a canonical
    GroupedCacheLayerLayout. This adapter intentionally exposes the same
    read-only query surface without pretending to own production topology.
    """

    def __init__(
        self,
        layer_tensors: Sequence[torch.Tensor],
        seq_size_per_block: int,
        *,
        tag: str = "default",
        layer_scale_tensors: Sequence[torch.Tensor | None] | None = None,
    ) -> None:
        if not layer_tensors:
            raise ValueError("SingleGroupKVCacheAdapter requires at least one layer")
        if seq_size_per_block <= 0:
            raise ValueError("seq_size_per_block must be positive")
        if not tag:
            raise ValueError("tag must not be empty")
        if layer_scale_tensors is not None and len(layer_scale_tensors) != len(
            layer_tensors
        ):
            raise ValueError("layer scale tensor count must match layer tensor count")
        self._layer_tensors = tuple(layer_tensors)
        self._layer_scale_tensors = (
            tuple(layer_scale_tensors)
            if layer_scale_tensors is not None
            else (None,) * len(layer_tensors)
        )
        self._seq_size_per_block = int(seq_size_per_block)
        self._tag = str(tag)

    @property
    def group_tags(self) -> list[str]:
        return [self._tag]

    @property
    def layer_count(self) -> int:
        return len(self._layer_tensors)

    def _validate_layer(self, layer_id: int) -> None:
        if layer_id < 0 or layer_id >= self.layer_count:
            raise RuntimeError(f"Invalid layer index: {layer_id}")

    def _validate_tag(self, tag: str) -> None:
        if tag != self._tag:
            raise RuntimeError(f"KV cache tag {tag!r} is not available")

    def get_layer_cache(self, layer_id: int, tag: str | None = None) -> LayerKVCache:
        self._validate_layer(layer_id)
        self._validate_tag(self._tag if tag is None else str(tag))
        return LayerKVCache(
            self._layer_tensors[layer_id],
            self._seq_size_per_block,
            layer_id=layer_id,
            tag=self._tag,
            kv_scale_base=self._layer_scale_tensors[layer_id],
        )

    def get_layer_cache_groups(self, layer_id: int) -> list[LayerKVCache]:
        return [self.get_layer_cache(layer_id)]

    def get_seq_size_per_block(self, tag: str) -> int:
        self._validate_tag(str(tag))
        return self._seq_size_per_block

    def get_kernel_seq_size_per_block(self, tag: str) -> int:
        self._validate_tag(str(tag))
        return self._seq_size_per_block

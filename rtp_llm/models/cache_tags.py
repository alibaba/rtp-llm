"""Declared cache-group tags for draft (MTP / DSpark) modules.

Every module owns its own cache pools, so the hard requirement is only that a
module names each of its groups with a tag no other module publishes.  The
content of the tag is free: existing names such as ``full``, ``swa_kv`` or
``mtp_default`` merely describe the group's business semantics.  Draft modules
here adopt the ``mtp_`` prefix because it keeps ``mtp_full`` readable next to
the main model's ``full``.

The declaration belongs to the model description: each draft model class calls
:func:`declare_draft_cache_tags` from its own ``_post_build_model_config``, so
the published tag names are an explicit part of the Python model description.
Nothing on the C++ side infers or rewrites them; ``CacheConfig`` only validates
that two modules never publish the same tag.

Python consumers keep addressing a pool by its canonical name: the handle
installed at the Python boundary
(:class:`rtp_llm.models_py.utils.kvcache.ModuleLocalKVCache`) reports canonical
group tags and maps them back onto the published ones, so the prefix never
leaks into the kernels.  :func:`canonical_cache_tag` is that mapping in its
prefix-stripping direction.
"""

import logging
from typing import Any

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
    records ``model_config.cache_tag_prefix`` so runtime consumers can map a
    canonical tag onto the tag this module published.
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

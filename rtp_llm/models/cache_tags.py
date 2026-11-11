"""Declared cache-group tags for draft (MTP / DSpark) modules.

A draft module owns its own cache pools, so the tag it publishes must not
collide with the main model's tags.  The rule is a plain naming principle: the
Python model description of a draft module publishes its cache groups under the
``mtp_`` prefix, while the main model keeps the canonical names.

The declaration belongs to the model description: each draft model class calls
:func:`declare_draft_cache_tags` from its own ``_post_build_model_config``, so
the published tag names are an explicit part of the Python model description.
Nothing on the C++ side infers or rewrites them; ``CacheConfig`` only validates
that two modules never publish the same tag.

``DeepSeekV41DSpark`` is the one deliberate exception: its builder already
declares draft-specific groups (``build_v41_kv_cache_spec_descs`` takes a
``draft`` flag), and the V4.1 runtime matches the literal ``swa_kv`` /
``decoder_swa_kv`` names in its pool helpers, so prefixing that family here
would silently retarget its pools.  A V4.1 draft that still collides with its
main model is reported by the ``CacheConfig`` validation instead of being
rewritten behind the model's back.
"""

import logging
from typing import Any

DRAFT_CACHE_TAG_PREFIX = "mtp_"


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

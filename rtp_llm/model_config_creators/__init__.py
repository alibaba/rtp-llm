"""Model configuration creators module.

This module provides configuration creation functions that are decoupled from
model classes, allowing configuration creation without GPU dependencies.
"""

from rtp_llm.model_config_creators.registry import (
    get_config_creator,
    register_config_creator,
)

# Import configuration creators to trigger registration
# These imports will execute the register_config_creator calls at module level
# Import order matters for models that depend on others (e.g., roberta depends on bert)
try:
    # Import base creators first
    # Note: Internal source models (deepseek_vl_v2, mixtbstars, flot, flot_vl, video_logics)
    # are now in internal_source/rtp_llm/model_config_creators and will be imported
    # via the internal_source import below
    # Note: tbstars2_5 is now in internal_source/rtp_llm/model_config_creators
    from rtp_llm.model_config_creators import bert  # noqa: F401
    from rtp_llm.model_config_creators import bloom  # noqa: F401
    from rtp_llm.model_config_creators import chatglm  # noqa: F401
    from rtp_llm.model_config_creators import chatglm_vision  # noqa: F401
    from rtp_llm.model_config_creators import cosyvoice_qwen  # noqa: F401
    from rtp_llm.model_config_creators import deepseek_v2  # noqa: F401
    from rtp_llm.model_config_creators import falcon  # noqa: F401
    from rtp_llm.model_config_creators import glm4_moe  # noqa: F401
    from rtp_llm.model_config_creators import gpt_neox  # noqa: F401
    from rtp_llm.model_config_creators import internvl  # noqa: F401
    from rtp_llm.model_config_creators import llama  # noqa: F401
    from rtp_llm.model_config_creators import llava  # noqa: F401
    from rtp_llm.model_config_creators import minicpmv  # noqa: F401
    from rtp_llm.model_config_creators import mixtral  # noqa: F401
    from rtp_llm.model_config_creators import mpt  # noqa: F401
    from rtp_llm.model_config_creators import phi  # noqa: F401
    from rtp_llm.model_config_creators import qwen  # noqa: F401
    from rtp_llm.model_config_creators import qwen2_vl  # noqa: F401
    from rtp_llm.model_config_creators import qwen3_next  # noqa: F401
    from rtp_llm.model_config_creators import qwen_audio  # noqa: F401
    from rtp_llm.model_config_creators import qwen_moe  # noqa: F401
    from rtp_llm.model_config_creators import qwen_vl  # noqa: F401
    from rtp_llm.model_config_creators import sgpt_bloom  # noqa: F401
    from rtp_llm.model_config_creators import starcoder  # noqa: F401

    # Import internal source config creators if available
    from rtp_llm.utils.import_util import has_internal_source

    if has_internal_source():
        try:
            import internal_source.rtp_llm.model_config_creators  # noqa: F401
        except ImportError as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to import internal config creators: {e}")
    # Add more imports as they are migrated
except ImportError as e:
    # Allow partial imports if some modules are missing
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to import some config creators: {e}")

__all__ = [
    "get_config_creator",
    "register_config_creator",
]

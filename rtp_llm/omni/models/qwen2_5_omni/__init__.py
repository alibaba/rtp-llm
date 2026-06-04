from rtp_llm.omni.config.pipeline_registry import OmniPipelineRegistry
from rtp_llm.omni.models.qwen2_5_omni.pipeline_config import QWEN2_5_OMNI_PIPELINE

import rtp_llm.omni.models.qwen2_5_omni.thinker  # noqa: F401
import rtp_llm.omni.models.qwen2_5_omni.talker  # noqa: F401
import rtp_llm.omni.models.qwen2_5_omni.token2wav  # noqa: F401

if OmniPipelineRegistry.get("qwen2_5_omni") is None:
    OmniPipelineRegistry.register(QWEN2_5_OMNI_PIPELINE)

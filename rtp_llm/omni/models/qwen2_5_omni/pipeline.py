from rtp_llm.model_factory_register import register_model
from rtp_llm.omni.config.pipeline_registry import OmniPipelineRegistry
from rtp_llm.omni.config.stage_config import (
    OmniPipelineConfig,
    OmniStageConfig,
    StageExecutionType,
)
from rtp_llm.omni.models.qwen2_5_omni.thinker import Qwen2_5OmniThinker

QWEN2_5_OMNI_PIPELINE = OmniPipelineConfig(
    model_type="qwen2_5_omni",
    model_arch="Qwen2_5OmniModel",
    stages=(
        OmniStageConfig(
            name="thinker",
            execution_type=StageExecutionType.LLM_AR,
            model_cls="Qwen2_5OmniThinker",
            model_type="qwen2_5_omni_thinker",
            next="talker",
            terminal=False,
            # Legacy compat
            stage_id=0,
            model_stage="thinker",
            final_output=True,
            final_output_type="text",
            requires_multimodal_data=True,
            engine_output_type="latent",
        ),
        OmniStageConfig(
            name="talker",
            execution_type=StageExecutionType.LLM_AR,
            model_cls="Qwen2_5OmniTalker",
            model_type="qwen2_5_omni_talker",
            next="token2wav",
            terminal=False,
            # Legacy compat
            stage_id=1,
            model_stage="talker",
            input_sources=(0,),
            engine_output_type="latent",
            stage_processor="rtp_llm.omni.models.qwen2_5_omni.stage_processors.thinker2talker",
        ),
        OmniStageConfig(
            name="token2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            model_cls="Qwen2_5OmniToken2Wav",
            model_type="qwen2_5_omni_token2wav",
            terminal=True,
            # Legacy compat
            stage_id=2,
            model_stage="token2wav",
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            stage_processor="rtp_llm.omni.models.qwen2_5_omni.stage_processors.talker2code2wav",
        ),
    ),
)

OmniPipelineRegistry.register(QWEN2_5_OMNI_PIPELINE)

register_model(
    "qwen2_5_omni",
    Qwen2_5OmniThinker,
    ["Qwen2_5OmniModel"],
)

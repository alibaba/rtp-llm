import logging

from rtp_llm.omni.engine.stage_connector import StageOutput
from rtp_llm.omni.engine.stage_processor_base import StageProcessorBase
from rtp_llm.omni.engine.stage_processor_registry import StageProcessorRegistry

logger = logging.getLogger(__name__)

CODEC_END_TOKEN = 8294


class Thinker2TalkerProcessor(StageProcessorBase):
    def process(self, source_output: StageOutput) -> StageOutput:
        return StageOutput(
            embeddings=source_output.embeddings,
            metadata={
                "source_token_ids": source_output.token_ids,
                "source_text": source_output.metadata.get("text", ""),
            },
        )


class Talker2Token2WavProcessor(StageProcessorBase):
    def process(self, source_output: StageOutput) -> StageOutput:
        token_ids = source_output.token_ids or []
        filtered = [t for t in token_ids if t != CODEC_END_TOKEN]
        return StageOutput(
            token_ids=filtered,
            metadata={"codec_token_count": len(filtered)},
        )


StageProcessorRegistry.register(
    "qwen2_5_omni.thinker2talker", Thinker2TalkerProcessor
)
StageProcessorRegistry.register(
    "qwen2_5_omni.talker2token2wav", Talker2Token2WavProcessor
)

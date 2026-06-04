from rtp_llm.omni.engine.stage_connector import StageOutput
from rtp_llm.omni.engine.stage_processor_base import StageProcessorBase


class Thinker2TalkerProcessor(StageProcessorBase):
    def process(self, source_output: StageOutput) -> StageOutput:
        return StageOutput(
            embeddings=source_output.embeddings,
            metadata={
                "source_token_ids": source_output.token_ids,
                "source_text": source_output.metadata.get("text", ""),
            },
        )


class Talker2Code2WavProcessor(StageProcessorBase):
    def process(self, source_output: StageOutput) -> StageOutput:
        return StageOutput(
            token_ids=source_output.token_ids,
            metadata={"from_talker": True},
        )

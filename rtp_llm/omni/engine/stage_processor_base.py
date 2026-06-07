from abc import ABC, abstractmethod

from rtp_llm.omni.engine.stage_connector import StageOutput


class StageProcessorBase(ABC):
    @abstractmethod
    def process(self, source_output: StageOutput) -> StageOutput:
        ...

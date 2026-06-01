from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class StageExecutionType(Enum):
    LLM_AR = "llm_ar"
    LLM_GENERATION = "llm_generation"
    DIFFUSION = "diffusion"


@dataclass(frozen=True)
class OmniStageConfig:
    stage_id: int
    model_stage: str
    execution_type: StageExecutionType
    model_cls: str
    input_sources: Tuple[int, ...] = ()
    final_output: bool = False
    final_output_type: Optional[str] = None
    requires_multimodal_data: bool = False
    engine_output_type: Optional[str] = None
    stage_processor: Optional[str] = None


@dataclass(frozen=True)
class OmniPipelineConfig:
    model_type: str
    model_arch: str
    stages: Tuple[OmniStageConfig, ...]

    def get_final_output_stages(self) -> list:
        return [s for s in self.stages if s.final_output]

    def get_stage(self, stage_id: int) -> OmniStageConfig:
        for s in self.stages:
            if s.stage_id == stage_id:
                return s
        raise KeyError(f"Stage {stage_id} not found in pipeline {self.model_type}")

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class StageExecutionType(Enum):
    LLM_AR = "llm_ar"
    LLM_GENERATION = "llm_generation"
    DIFFUSION = "diffusion"
    CPU_EXECUTOR = "cpu_executor"


@dataclass(frozen=True)
class OmniStageConfig:
    name: str
    execution_type: StageExecutionType
    model_cls: str
    factory: Optional[str] = None
    factory_args: Dict[str, Any] = field(default_factory=dict)
    gpu: Optional[int] = None
    tp_size: int = 1
    process: str = "pipeline"
    next: Union[str, Tuple[str, ...], None] = None
    stream_to: Tuple[str, ...] = ()
    wait_for: Tuple[str, ...] = ()
    merge_fn: Optional[str] = None
    project_payload: Dict[str, str] = field(default_factory=dict)
    terminal: bool = False
    can_accept_stream_before_payload: bool = False
    # Legacy fields for backward compatibility
    stage_id: Optional[int] = None
    model_type: Optional[str] = None
    model_stage: Optional[str] = None
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

    def get_stage_by_name(self, name: str) -> OmniStageConfig:
        for s in self.stages:
            if s.name == name:
                return s
        raise KeyError(f"Stage '{name}' not found in pipeline {self.model_type}")

    def get_terminal_stages(self) -> List[OmniStageConfig]:
        return [s for s in self.stages if s.terminal]

    def get_entry_stages(self) -> List[OmniStageConfig]:
        referenced: set = set()
        for s in self.stages:
            if isinstance(s.next, str):
                referenced.add(s.next)
            elif isinstance(s.next, tuple):
                referenced.update(s.next)
        return [s for s in self.stages if s.name not in referenced]

    def validate(self) -> None:
        names = set()
        for s in self.stages:
            if s.name in names:
                raise ValueError(
                    f"Duplicate stage name: '{s.name}' in pipeline {self.model_type}"
                )
            names.add(s.name)

        for s in self.stages:
            targets: List[str] = []
            if isinstance(s.next, str):
                targets.append(s.next)
            elif isinstance(s.next, tuple):
                targets.extend(s.next)
            targets.extend(s.stream_to)
            targets.extend(s.wait_for)

            for t in targets:
                if t == s.name:
                    raise ValueError(
                        f"Stage '{s.name}' has self-reference in pipeline {self.model_type}"
                    )
                if t not in names:
                    raise ValueError(
                        f"Stage '{s.name}' references '{t}' which does not exist "
                        f"in pipeline {self.model_type}"
                    )

        terminals = self.get_terminal_stages()
        if not terminals:
            raise ValueError(
                f"Pipeline {self.model_type} has no terminal stage"
            )

        entries = self.get_entry_stages()
        if not entries:
            raise ValueError(
                f"Pipeline {self.model_type} has no entry point "
                f"(every stage is referenced by another stage's 'next')"
            )

    # Legacy methods for backward compatibility
    def get_final_output_stages(self) -> List[OmniStageConfig]:
        return [s for s in self.stages if s.final_output]

    def get_stage(self, stage_id: int) -> OmniStageConfig:
        for s in self.stages:
            if s.stage_id == stage_id:
                return s
        raise KeyError(f"Stage {stage_id} not found in pipeline {self.model_type}")

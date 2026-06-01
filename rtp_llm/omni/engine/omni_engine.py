import logging
from typing import Any, Dict, Optional

from rtp_llm.omni.config.stage_config import OmniPipelineConfig
from rtp_llm.omni.engine.orchestrator import OmniOrchestrator
from rtp_llm.omni.engine.output_processor import OmniOutputProcessor
from rtp_llm.omni.engine.stage_connector import SharedMemoryConnector, StageConnector
from rtp_llm.omni.engine.stage_pool import OmniStagePool

logger = logging.getLogger(__name__)


class OmniEngine:
    def __init__(
        self,
        pipeline_config: OmniPipelineConfig,
        connector: Optional[StageConnector] = None,
        model_config: Any = None,
        engine_config: Any = None,
    ):
        self.pipeline_config = pipeline_config
        self.model_config = model_config
        self.engine_config = engine_config
        self.connector = connector or SharedMemoryConnector()
        self.output_processor = OmniOutputProcessor()

        self.stage_pools: Dict[int, OmniStagePool] = {}
        for stage_config in pipeline_config.stages:
            self.stage_pools[stage_config.stage_id] = OmniStagePool(
                stage_config=stage_config
            )

        self.orchestrator = OmniOrchestrator(
            pipeline_config=pipeline_config,
            connector=self.connector,
            stage_pools=self.stage_pools,
        )

        logger.info(
            f"OmniEngine created for {pipeline_config.model_type} "
            f"with {len(pipeline_config.stages)} stages"
        )

    @property
    def num_stages(self) -> int:
        return len(self.pipeline_config.stages)

    def get_final_output_types(self) -> Dict[str, int]:
        result = {}
        for stage in self.pipeline_config.stages:
            if stage.final_output and stage.final_output_type:
                result[stage.final_output_type] = stage.stage_id
        return result

    @classmethod
    def from_pipeline_config(
        cls,
        pipeline_config: OmniPipelineConfig,
        model_config: Any = None,
        engine_config: Any = None,
    ) -> "OmniEngine":
        return cls(
            pipeline_config=pipeline_config,
            model_config=model_config,
            engine_config=engine_config,
        )

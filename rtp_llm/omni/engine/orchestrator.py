import logging
from typing import Any, Dict, List, Optional

from rtp_llm.omni.config.stage_config import OmniPipelineConfig
from rtp_llm.omni.engine.stage_connector import StageConnector, StageOutput
from rtp_llm.omni.engine.stage_pool import OmniStagePool

logger = logging.getLogger(__name__)


class OmniRequestState:
    def __init__(self, request_id: str, num_stages: int):
        self.request_id = request_id
        self.current_stage = 0
        self._num_stages = num_stages
        self.stage_status: List[str] = ["pending"] * num_stages
        self.is_complete = False

    def advance(self) -> None:
        if self.is_complete:
            raise RuntimeError(
                f"Request {self.request_id} is already complete, cannot advance"
            )
        self.stage_status[self.current_stage] = "completed"
        self.current_stage += 1
        if self.current_stage >= self._num_stages:
            self.is_complete = True


class OmniOrchestrator:
    def __init__(
        self,
        pipeline_config: OmniPipelineConfig,
        connector: StageConnector,
        stage_pools: Dict[int, OmniStagePool],
    ):
        self._pipeline_config = pipeline_config
        self._connector = connector
        self._stage_pools = stage_pools
        self._requests: Dict[str, OmniRequestState] = {}

    def submit(self, request_id: str) -> OmniRequestState:
        if request_id in self._requests:
            raise ValueError(
                f"Request {request_id} already submitted to pipeline "
                f"{self._pipeline_config.model_type}"
            )
        state = OmniRequestState(
            request_id=request_id,
            num_stages=len(self._pipeline_config.stages),
        )
        self._requests[request_id] = state
        logger.info(
            f"Submitted request {request_id} to pipeline "
            f"{self._pipeline_config.model_type}"
        )
        return state

    def get_execution_order(self) -> List[int]:
        return [s.stage_id for s in self._pipeline_config.stages]

    def get_request_state(self, request_id: str) -> Optional[OmniRequestState]:
        return self._requests.get(request_id)

    def cleanup(self, request_id: str) -> None:
        self._connector.cleanup(request_id)
        self._requests.pop(request_id, None)
        logger.info(f"Cleaned up request {request_id}")

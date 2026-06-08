import logging
from typing import Dict, List, Optional

from rtp_llm.omni.config.stage_config import OmniPipelineConfig
from rtp_llm.omni.engine.stage_connector import StageConnector
from rtp_llm.omni.engine.stage_pool import OmniStagePool

logger = logging.getLogger(__name__)


class OmniRequestState:
    def __init__(self, request_id: str, stage_names: List[str]):
        self.request_id = request_id
        self._stage_status: Dict[str, str] = {name: "pending" for name in stage_names}

    @property
    def is_complete(self) -> bool:
        return all(s == "completed" for s in self._stage_status.values())

    def is_stage_complete(self, stage_name: str) -> bool:
        return self._stage_status.get(stage_name) == "completed"

    def mark_complete(self, stage_name: str) -> None:
        if stage_name not in self._stage_status:
            raise KeyError(f"Unknown stage: {stage_name}")
        self._stage_status[stage_name] = "completed"


class OmniOrchestrator:
    def __init__(
        self,
        pipeline_config: OmniPipelineConfig,
        connector: StageConnector,
        stage_pools: Dict[str, OmniStagePool],
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
        stage_names = [s.name for s in self._pipeline_config.stages]
        state = OmniRequestState(request_id, stage_names)
        self._requests[request_id] = state
        return state

    def get_execution_order(self) -> List[str]:
        stages = {s.name: s for s in self._pipeline_config.stages}
        in_edges: Dict[str, set] = {name: set() for name in stages}
        for s in self._pipeline_config.stages:
            targets = []
            if isinstance(s.next, str):
                targets.append(s.next)
            elif isinstance(s.next, tuple):
                targets.extend(s.next)
            for t in targets:
                in_edges[t].add(s.name)

        order = []
        ready = [name for name, deps in in_edges.items() if not deps]
        while ready:
            name = ready.pop(0)
            order.append(name)
            s = stages[name]
            targets = []
            if isinstance(s.next, str):
                targets.append(s.next)
            elif isinstance(s.next, tuple):
                targets.extend(s.next)
            for t in targets:
                in_edges[t].discard(name)
                if not in_edges[t]:
                    ready.append(t)
        return order

    def get_downstream(self, stage_name: str) -> List[str]:
        stage = self._pipeline_config.get_stage_by_name(stage_name)
        if stage.next is None:
            return []
        if isinstance(stage.next, str):
            return [stage.next]
        return list(stage.next)

    def get_request_state(self, request_id: str) -> Optional[OmniRequestState]:
        return self._requests.get(request_id)

    def cleanup(self, request_id: str) -> None:
        self._connector.cleanup(request_id)
        self._requests.pop(request_id, None)

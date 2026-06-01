import logging
import threading
from typing import Any, List

from rtp_llm.omni.config.stage_config import OmniStageConfig

logger = logging.getLogger(__name__)


class OmniStagePool:
    def __init__(self, stage_config: OmniStageConfig):
        self.stage_config = stage_config
        self.stage_id = stage_config.stage_id
        self._replicas: List[Any] = []
        self._index = 0
        self._lock = threading.Lock()

    @property
    def num_replicas(self) -> int:
        return len(self._replicas)

    def add_replica(self, pipeline: Any) -> None:
        with self._lock:
            self._replicas.append(pipeline)
            logger.info(
                f"Added replica for stage {self.stage_id} "
                f"({self.stage_config.model_stage}), "
                f"total replicas: {len(self._replicas)}"
            )

    def get_replica(self) -> Any:
        with self._lock:
            if not self._replicas:
                raise RuntimeError(
                    f"No replicas available for stage {self.stage_id} "
                    f"({self.stage_config.model_stage})"
                )
            replica = self._replicas[self._index % len(self._replicas)]
            self._index += 1
            return replica

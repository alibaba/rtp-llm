import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class StageOutput:
    token_ids: Optional[List[int]] = None
    embeddings: Optional[torch.Tensor] = None
    audio_waveform: Optional[torch.Tensor] = None
    image_tensor: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class StageConnector(ABC):
    @abstractmethod
    def put(self, request_id: str, stage_id: int, data: StageOutput) -> bool:
        ...

    @abstractmethod
    def get(self, request_id: str, stage_id: int) -> Optional[StageOutput]:
        ...

    @abstractmethod
    def cleanup(self, request_id: str) -> None:
        ...


class SharedMemoryConnector(StageConnector):
    def __init__(self):
        self._store: Dict[Tuple[str, int], StageOutput] = {}
        self._lock = threading.Lock()

    def put(self, request_id: str, stage_id: int, data: StageOutput) -> bool:
        key = (request_id, stage_id)
        with self._lock:
            self._store[key] = data
        return True

    def get(self, request_id: str, stage_id: int) -> Optional[StageOutput]:
        key = (request_id, stage_id)
        with self._lock:
            return self._store.get(key)

    def cleanup(self, request_id: str) -> None:
        with self._lock:
            keys_to_remove = [
                k for k in self._store if k[0] == request_id
            ]
            for k in keys_to_remove:
                del self._store[k]

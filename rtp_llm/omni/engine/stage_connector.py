import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from rtp_llm.omni.engine.stream_channel import StreamChannel

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
    def put(self, request_id: str, stage_name: str, data: StageOutput) -> bool:
        ...

    @abstractmethod
    def get(self, request_id: str, stage_name: str) -> Optional[StageOutput]:
        ...

    @abstractmethod
    def open_stream(
        self, request_id: str, source: str, target: str
    ) -> StreamChannel:
        ...

    @abstractmethod
    def cleanup(self, request_id: str) -> None:
        ...


class SharedMemoryConnector(StageConnector):
    def __init__(self, stream_maxsize: int = 256):
        self._store: Dict[Tuple[str, str], StageOutput] = {}
        self._streams: Dict[Tuple[str, str, str], StreamChannel] = {}
        self._lock = threading.Lock()
        self._stream_maxsize = stream_maxsize

    def put(self, request_id: str, stage_name: str, data: StageOutput) -> bool:
        key = (request_id, stage_name)
        with self._lock:
            self._store[key] = data
        return True

    def get(self, request_id: str, stage_name: str) -> Optional[StageOutput]:
        key = (request_id, stage_name)
        with self._lock:
            return self._store.get(key)

    def open_stream(
        self, request_id: str, source: str, target: str
    ) -> StreamChannel:
        key = (request_id, source, target)
        with self._lock:
            if key not in self._streams:
                self._streams[key] = StreamChannel(maxsize=self._stream_maxsize)
            return self._streams[key]

    def cleanup(self, request_id: str) -> None:
        streams_to_close = []
        with self._lock:
            keys_to_remove = [k for k in self._store if k[0] == request_id]
            for k in keys_to_remove:
                del self._store[k]
            stream_keys = [k for k in self._streams if k[0] == request_id]
            for k in stream_keys:
                streams_to_close.append(self._streams[k])
                del self._streams[k]
        for stream in streams_to_close:
            stream.close()

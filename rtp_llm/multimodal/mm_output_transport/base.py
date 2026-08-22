from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalInputsPB,
    MultimodalOutputPB,
)
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes

@dataclass
class MMOutputResult:
    receipt: MultimodalOutputPB
    transport: str
    payload_embedding_bytes: int
    payload_pos_bytes: int
    payload_extra_bytes: int


class MMTransportBackend(ABC):
    """ViT-side data-plane interface."""

    name: str

    @abstractmethod
    def supports(self, request: MultimodalInputsPB, res: MMEmbeddingRes) -> bool:
        """Return whether this backend can serve the request without transfer work."""

    @abstractmethod
    def try_transfer(
        self, request: MultimodalInputsPB, res: MMEmbeddingRes
    ) -> Optional[MMOutputResult]:
        """Return None when the manager should try the next backend."""

    def release(self, handles: List[str]) -> None:
        return None

    def close(self) -> None:
        return None


class MMTerminalBackend(MMTransportBackend):
    """Always-available backend that cannot request another fallback."""

    def supports(self, request: MultimodalInputsPB, res: MMEmbeddingRes) -> bool:
        return True

    @abstractmethod
    def transfer(
        self, request: MultimodalInputsPB, res: MMEmbeddingRes
    ) -> MMOutputResult:
        """Return a result without asking the manager to fall back again."""

    def try_transfer(
        self, request: MultimodalInputsPB, res: MMEmbeddingRes
    ) -> Optional[MMOutputResult]:
        return self.transfer(request, res)

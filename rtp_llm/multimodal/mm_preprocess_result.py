"""CPU-only optional sidecar; never sent through the frontend/ViT RPC."""

from dataclasses import dataclass
from typing import Any, List


@dataclass
class MMPreprocessResult:
    """Model CPU data plus the exact encoded media consumed by that decoder.

    The executor detaches inspection_sources before handing data to the GPU
    scheduler. Only inspection-enabled requests ask workers to retain bytes.
    """

    data: Any
    inspection_sources: List[bytes]

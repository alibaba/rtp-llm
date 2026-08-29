"""ViT-side multimodal output transport."""

from rtp_llm.multimodal.transport.factory import (
    create_mm_output_transport,
)
from rtp_llm.multimodal.transport.manager import MMOutputTransport

__all__ = ["MMOutputTransport", "create_mm_output_transport"]

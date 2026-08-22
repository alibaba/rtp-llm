"""ViT-side multimodal output transport."""

from rtp_llm.multimodal.mm_output_transport.factory import (
    create_mm_output_transport,
)
from rtp_llm.multimodal.mm_output_transport.manager import MMOutputTransport

__all__ = ["MMOutputTransport", "create_mm_output_transport"]

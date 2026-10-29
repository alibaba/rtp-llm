"""CPU-only optional sidecars; never sent through the frontend/ViT RPC."""

import hashlib
import re
from dataclasses import dataclass
from typing import Any, List, Optional


def is_single_image_input(mm_inputs) -> bool:
    from rtp_llm.utils.base_model_datatypes import MMUrlType

    return (
        len(mm_inputs) == 1
        and mm_inputs[0].mm_type == MMUrlType.IMAGE
        and (mm_inputs[0].tensor is None or mm_inputs[0].tensor.numel() == 0)
    )


def is_http_image_input(mm_inputs) -> bool:
    return is_single_image_input(mm_inputs) and mm_inputs[0].url.startswith(
        ("http://", "https://")
    )


@dataclass(frozen=True)
class ImageSourceIdentity:
    """Digest of the exact encoded bytes decoded by a trusted CPU worker.

    This is content identity, never an inspection verdict. It is deliberately
    absent from the public request/RPC schema.
    """

    sha256: str
    nbytes: int

    def __post_init__(self):
        if not isinstance(self.sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.sha256
        ):
            raise ValueError("Invalid image source digest")
        if type(self.nbytes) is not int or self.nbytes <= 0:
            raise ValueError("Invalid image source size")

    @classmethod
    def from_bytes(cls, source: bytes):
        return cls(hashlib.sha256(source).hexdigest(), len(source))


@dataclass
class MMPreprocessResult:
    """CPU data plus exact encoded media owned by inspection, and its digest.

    Worker inspection detaches the original bytes before IPC. The small digest
    sidecar can cross IPC independently, including when inspection is disabled.
    """

    data: Any
    inspection_sources: List[bytes]
    source_identity: Optional[ImageSourceIdentity] = None

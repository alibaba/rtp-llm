"""Model-owned request policies for the DashSc inference bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.response_format_compiler import ReasoningFormat
from rtp_llm.dash_sc.codec import (
    DashScRequestControls,
    SamplingParams,
    parse_multimodal_parts_from_request,
)
from rtp_llm.dash_sc.proto import predict_v2_pb2

if TYPE_CHECKING:
    from rtp_llm.ops import MultimodalInput


class PromptTokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class DashScRequestContext:
    request: predict_v2_pb2.ModelInferRequest
    input_ids: list[int]
    sampling: SamplingParams
    controls: DashScRequestControls
    generate_config: GenerateConfig
    tokenizer: PromptTokenizer | None
    mm_inputs: list[MultimodalInput] | None = None


@dataclass(frozen=True)
class PreparedDashScRequest:
    input_ids: list[int]
    mm_inputs: list[MultimodalInput] | None = None
    prompt_token_offset: int = 0
    reasoning_format: ReasoningFormat | None = None


class DashScRequestAdapter:
    """Customize request semantics before generic grammar compilation/enqueue.

    The default preserves token IDs, media and sampling configuration. Models
    may validate protocol fields, normalize prompt IDs and set GenerateConfig
    in prepare(). Replacing the ID list invalidates a pre-parsed input tensor.
    """

    def prepare(self, context: DashScRequestContext) -> PreparedDashScRequest:
        return PreparedDashScRequest(context.input_ids, context.mm_inputs)


def build_multimodal_inputs_from_request(
    request: predict_v2_pb2.ModelInferRequest,
) -> list[MultimodalInput]:
    """Convert DashSc message parts to the engine's generic multimodal inputs."""
    parts = parse_multimodal_parts_from_request(request)
    if not parts:
        return []

    from rtp_llm.ops import MMPreprocessConfig, MultimodalInput

    return [
        MultimodalInput(
            part.url,
            part.mm_type,
            torch.empty(0),
            MMPreprocessConfig(
                min_pixels=part.min_pixels,
                max_pixels=part.max_pixels,
                fps=part.fps,
                min_frames=part.min_frames,
                max_frames=part.max_frames,
            ),
        )
        for part in parts
    ]

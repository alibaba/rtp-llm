from __future__ import annotations
import torch
import typing
__all__: list[str] = ['EmbeddingCppOutput', 'MMPreprocessConfig', 'MultimodalFeature', 'MultimodalInput', 'MultimodalOutput', 'RtpEmbeddingOp', 'RtpLLMOp', 'TypedOutput']
class EmbeddingCppOutput:
    output: TypedOutput
    def __init__(self) -> None:
        ...
    def setMapOutput(self, arg0: list[dict[str, torch.Tensor]]) -> None:
        ...
    def setTensorOutput(self, arg0: torch.Tensor) -> None:
        ...
class MMPreprocessConfig:
    crop_positions: list[float]
    fps: int
    height: int
    max_frames: int
    max_pixels: int
    min_frames: int
    min_pixels: int
    mm_timeout_ms: int
    width: int
    def __init__(self, width: int, height: int, min_pixels: int, max_pixels: int, fps: int, min_frames: int, max_frames: int, crop_positions: list[float], mm_timeout_ms: int) -> None:
        ...
class MultimodalFeature:
    expanded_ids: ...
    features: list[torch.Tensor]
    inputs: list[MultimodalInput]
    locs: ...
    text_tokens_mask: ...
    def __init__(self) -> None:
        ...
class MultimodalInput:
    mm_type: int
    tensor: torch.Tensor
    url: str
    def __init__(self, url: str, tensor: torch.Tensor, mm_type: int) -> None:
        ...
class MultimodalOutput:
    mm_deepstack_embeds: list[torch.Tensor] | None
    mm_features: list[torch.Tensor]
    mm_position_ids: list[torch.Tensor] | None
    def __init__(self) -> None:
        ...
class RtpEmbeddingOp:
    def __init__(self) -> None:
        ...
    def decode(self, token_ids: torch.Tensor, token_type_ids: torch.Tensor, input_lengths: torch.Tensor, request_id: int, multimodal_inputs: list[MultimodalInput]) -> typing.Any:
        ...
    def init(self, model: typing.Any, engine_config: typing.Any, vit_config: typing.Any, mm_process_engine: typing.Any) -> None:
        ...
    def stop(self) -> None:
        ...
class RtpLLMOp:
    def __init__(self) -> None:
        ...
    def init(self, model: typing.Any, engine_config: typing.Any, vit_config: typing.Any, propose_model: typing.Any, token_processor: typing.Any, mm_process_engine: typing.Any) -> None:
        ...
    def start_http_server(self, model_weights_loader: typing.Any, lora_infos: typing.Any, world_info: typing.Any, tokenizer: typing.Any, render: typing.Any) -> None:
        ...
    def stop(self) -> None:
        ...
class TypedOutput:
    isTensor: bool
    def __init__(self) -> None:
        ...
    @property
    def map(self) -> typing.Any:
        ...
    @map.setter
    def map(self, arg1: list[dict[str, torch.Tensor]]) -> None:
        ...
    @property
    def t(self) -> typing.Any:
        ...
    @t.setter
    def t(self, arg1: torch.Tensor) -> None:
        ...

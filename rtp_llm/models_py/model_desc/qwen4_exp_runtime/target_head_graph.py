"""Own exact-row target logits captured with speculative verification.

The native target transaction activates its request key after replay. Head
consumption remains one-shot; TP gather and final FP32 conversion stay native.
The dynamic activation scale is computed over precisely the live target rows.
"""

import torch


class TargetHeadGraph:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.outputs: dict[int, tuple[torch.Tensor, int]] = {}
        self.active = None

    def clear_activation(self) -> None:
        self.active = None

    def stage(self, head, hidden: torch.Tensor, request_count: int) -> None:
        if not self.enabled:
            return
        if (
            type(request_count) is not int
            or request_count <= 0
            or hidden.ndim != 2
            or hidden.shape[0] <= 0
            or hidden.shape[0] % request_count
            or hidden.dtype != torch.bfloat16
            or not hidden.is_cuda
        ):
            raise RuntimeError("target head requires exact packed CUDA BF16 rows")
        logits = head(hidden)
        if (
            logits.ndim != 2
            or logits.shape[0] != hidden.shape[0]
            or logits.dtype != torch.bfloat16
            or logits.device != hidden.device
        ):
            raise RuntimeError("target graph head must produce local CUDA BF16 logits")
        self.outputs[request_count] = (logits, int(hidden.shape[1]))

    def activate(self, request_count: int) -> None:
        self.active = self.outputs.get(request_count) if self.enabled else None

    def take(self, hidden: torch.Tensor):
        selected, self.active = self.active, None
        if selected is None:
            return None
        logits, width = selected
        # A selected-row or otherwise changed post-layer consumer must compute
        # its own quantization scale. Never crop logits quantized over more rows.
        if (
            hidden.ndim != 2
            or tuple(hidden.shape) != (logits.shape[0], width)
            or hidden.dtype != torch.bfloat16
            or hidden.device != logits.device
        ):
            return None
        return logits

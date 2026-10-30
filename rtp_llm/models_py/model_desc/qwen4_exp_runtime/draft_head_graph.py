"""Own local BF16 draft logits produced by the selected model CUDA graph.

The native caller activates a key only after replay and consumes it once in
the ordinary post-layer head path. TP gathering and FP32 sampler outputs keep
their existing ownership and ordering.
"""

import torch


class DraftHeadGraph:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.outputs: dict[int, torch.Tensor] = {}
        self.active = None

    def stage(self, head, hidden: torch.Tensor) -> None:
        if not self.enabled:
            return
        logits = head(hidden.to(torch.bfloat16))
        if (
            logits.ndim != 2
            or logits.shape[0] != hidden.shape[0]
            or logits.dtype != torch.bfloat16
            or not logits.is_cuda
        ):
            raise RuntimeError("draft graph head must produce local CUDA BF16 logits")
        self.outputs[int(hidden.shape[0])] = logits

    def clear_activation(self) -> None:
        self.active = None

    def activate(self, graph_batch: int) -> None:
        self.active = None
        if not self.enabled:
            return
        if graph_batch not in self.outputs:
            raise RuntimeError("draft graph has no captured FP8 head for this batch")
        self.active = self.outputs[graph_batch]

    def take(self, hidden: torch.Tensor):
        logits, self.active = self.active, None
        if logits is None:
            return None
        if (
            hidden.ndim != 2
            or hidden.device != logits.device
            or not 0 < hidden.shape[0] <= logits.shape[0]
        ):
            raise RuntimeError(
                "draft graph head rows do not cover the live hidden rows"
            )
        # FP8 input quantization computes one scale over all captured rows.
        # A smaller live batch must recompute its head using only those rows;
        # cropping the captured logits would retain the dummy rows' scale.
        if hidden.shape[0] != logits.shape[0]:
            return None
        return logits

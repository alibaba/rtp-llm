"""Fork/join for independent inference branches, including CUDA Graph replay."""

import os

from typing import Callable, Optional, Sequence, Tuple, TypeVar

import torch

MainOutput = TypeVar("MainOutput")


class CmpMoeStreams:
    """Own a side stream initialized by eager warmup, never during capture.

    Both branches must use separate scratch storage and consume only the supplied
    ready inputs. The main branch retains the caller stream (and collectives).
    Instances may be reused by graph clones that execute sequentially.
    """

    @classmethod
    def from_env(cls):
        """Resolve once at model construction; off allocates no helper or stream."""
        if os.environ.get("RTP_LLM_GLM5_CMP_MOE_DUAL_STREAM", "1") != "1":
            return None
        return cls()

    def __init__(self) -> None:
        self._stream: Optional[torch.cuda.Stream] = None

    def run(
        self,
        main_branch: Callable[[], MainOutput],
        side_branch: Callable[[], torch.Tensor],
        inputs: Sequence[Optional[torch.Tensor]],
    ) -> Tuple[MainOutput, torch.Tensor]:
        anchor = inputs[0]
        assert anchor is not None
        if (
            not anchor.is_cuda
            or torch.is_grad_enabled()
            or any(t is not None and t.device != anchor.device for t in inputs)
        ):
            return main_branch(), side_branch()
        with torch.cuda.device(anchor.device):
            if self._stream is None:
                if torch.cuda.is_current_stream_capturing():
                    return main_branch(), side_branch()
                self._stream = torch.cuda.Stream(device=anchor.device)
            if self._stream.device != anchor.device:
                return main_branch(), side_branch()
            main = torch.cuda.current_stream(anchor.device)
            side = self._stream
            side.wait_stream(main)
            try:
                # Match SGLang: issue routed before the shared block so graph
                # capture can reuse the auxiliary stream across adjacent layers.
                main_output = main_branch()
                with torch.cuda.stream(side):
                    for tensor in inputs:
                        if tensor is not None:
                            tensor.record_stream(side)
                    side_output = side_branch()
            finally:
                # Also order already-enqueued work if either branch raises.
                main.wait_stream(side)
            side_output.record_stream(main)
            return main_output, side_output

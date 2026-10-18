"""V4.1 RGB transfer only: bounded pinned packing, one H2D, event handoff.

No decode, resize, normalization, model forward, cache or request scheduling is
performed here. One uploader belongs to one scheduler's upload thread. Each
batch owns its host/device storage; no shared mutable slot can cross requests.
"""

from dataclasses import dataclass

import torch

from rtp_llm.multimodal.multimodal_mixins.multimodal_common import PreparedMMInputs
from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate


@dataclass(frozen=True)
class UploadedImage:
    """Device uint8 RGB; shape metadata remains readable without a D2H sync."""

    rgb: torch.Tensor


class UploadedRGBBatch(PreparedMMInputs):
    def __init__(self, data_list, host, device_storage, event):
        super().__init__(data_list)
        self._host = host
        self._device_storage = device_storage
        self._event = event

    def wait(self):
        if self._event is None:
            raise RuntimeError("RGB upload batch already closed")
        with torch.cuda.device(
            self._device_storage.device
        ), cuda_graph_gate.operation():
            stream = torch.cuda.current_stream(self._device_storage.device)
            stream.wait_event(self._event)
            # The allocation originated on the copy stream; forward may still
            # read it after this envelope is released by CPU code.
            self._device_storage.record_stream(stream)

    def close(self):
        if self._event is not None:
            # Cancellation may occur before wait()/forward. Retain pinned bytes
            # until DMA is done, not merely until the async copy was enqueued.
            self._event.synchronize()
            self._event = None
        self._host = None
        self._device_storage = None
        super().close()


class RGBBatchUploader:
    def __init__(self, device, max_bytes):
        if max_bytes <= 0:
            raise ValueError("V41_GPU_PREFETCH_MAX_BYTES must be positive")
        self.device = torch.device(device)
        self.max_bytes = max_bytes
        self._stream = None
        self.batches = 0
        self.bytes = 0
        self.budget_fallbacks = 0

    @torch.inference_mode()
    @cuda_graph_gate.operation()
    def upload(self, groups, singletons):
        images = [rgb for group in groups for rgb in group]
        if not images:
            return None
        for rgb in images:
            if (
                rgb.device.type != "cpu"
                or rgb.dtype != torch.uint8
                or rgb.ndim != 3
                or rgb.shape[-1] != 3
                or not rgb.is_contiguous()
            ):
                raise ValueError("RGB staging requires contiguous CPU uint8 HWC RGB")
        total = sum(rgb.numel() for rgb in images)
        if total > self.max_bytes:
            self.budget_fallbacks += 1
            return None  # Ordinary per-image transfer, no partial upload or retry.

        with torch.profiler.record_function("v41_pin_rgb_batch"):
            host = torch.empty(total, dtype=torch.uint8, pin_memory=True)
            offset = 0
            for rgb in images:
                count = rgb.numel()
                host[offset : offset + count].copy_(rgb.view(-1))
                offset += count

        with torch.cuda.device(self.device), cuda_graph_gate.operation():
            if self._stream is None:
                self._stream = torch.cuda.Stream(device=self.device)
            try:
                with torch.cuda.stream(self._stream), torch.profiler.record_function(
                    "v41_h2d_rgb_batch"
                ):
                    storage = host.to(device=self.device, non_blocking=True)
                    done = torch.cuda.Event()
                    done.record(self._stream)
                offset = 0
                data_list = []
                for group, singleton in zip(groups, singletons):
                    uploaded = []
                    for rgb in group:
                        count = rgb.numel()
                        uploaded.append(
                            UploadedImage(
                                storage[offset : offset + count].view(rgb.shape)
                            )
                        )
                        offset += count
                    data_list.append(uploaded[0] if singleton else uploaded)
                result = UploadedRGBBatch(data_list, host, storage, done)
            except Exception:
                # Includes allocation/event failures after a copy was submitted.
                # Do not release host storage or silently retry on another path.
                self._stream.synchronize()
                raise
        self.batches += 1
        self.bytes += total
        return result

    def stats(self):
        return {
            "batches": self.batches,
            "bytes": self.bytes,
            "budget_fallbacks": self.budget_fallbacks,
            "max_bytes_per_batch": self.max_bytes,
        }

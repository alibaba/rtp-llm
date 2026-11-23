"""V4.1 RGB transfer only: bounded pinned packing, one H2D, event handoff.

No decode, resize, normalization, model forward, cache or request scheduling is
performed here. One uploader belongs to one scheduler's upload thread. Each
batch owns its host/device storage; no shared mutable slot can cross requests.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch

from rtp_llm.metrics import GaugeMetrics
from rtp_llm.models_py.modules.dsv41.vision_preprocess_tables import ResizeTables
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import PreparedMMInputs
from rtp_llm.multimodal.vit_metrics import VIT_PHASE_METRICS, report_vit_phase
from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate


def _pack_rgb_bytes(images, host):
    """Copy validated contiguous CPU uint8 images without ATen parallel work.

    NumPy views share the tensor storage; copyto performs only a byte copy and
    does not enlist the PyTorch intra-op pool for every image. In particular,
    the upload thread must not change process-wide torch/OMP thread settings.
    The caller owns host until its asynchronous H2D has completed.
    """
    destination = host.numpy()
    offset = 0
    for rgb in images:
        source = rgb.numpy().reshape(-1)
        count = source.size
        np.copyto(destination[offset : offset + count], source, casting="no")
        offset += count


@dataclass(frozen=True)
class UploadedImage:
    """Device uint8 RGB; shape metadata remains readable without a D2H sync."""

    rgb: torch.Tensor
    resize_tables: ResizeTables | None = None


class UploadedRGBBatch(PreparedMMInputs):
    def __init__(self, data_list, host, device_storage, event, h2d_start=None):
        super().__init__(data_list)
        self._host = host
        self._device_storage = device_storage
        self._event = event
        self._h2d_start = h2d_start

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
            if self._h2d_start is not None:
                try:
                    elapsed_us = self._h2d_start.elapsed_time(self._event) * 1000.0
                except RuntimeError:
                    logging.exception("Failed to time ViT RGB upload")
                else:
                    report_vit_phase(
                        GaugeMetrics.VIT_RGB_UPLOAD_PHASE_US_METRIC,
                        elapsed_us,
                        {"phase": "h2d_device"},
                    )
                self._h2d_start = None
            self._event = None
        self._host = None
        self._device_storage = None
        super().close()


class RGBBatchUploader:
    def __init__(self, device, max_bytes):
        if max_bytes <= 0:
            raise ValueError("RGB prefetch byte budget must be positive")
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

        pack_start = time.monotonic_ns() if VIT_PHASE_METRICS else None
        with torch.profiler.record_function("v41_pin_rgb_batch"):
            host = torch.empty(total, dtype=torch.uint8, pin_memory=True)
            _pack_rgb_bytes(images, host)
        if pack_start is not None:
            report_vit_phase(
                GaugeMetrics.VIT_RGB_UPLOAD_PHASE_US_METRIC,
                (time.monotonic_ns() - pack_start) / 1000.0,
                {"phase": "pin_and_pack"},
            )

        with torch.cuda.device(self.device), cuda_graph_gate.operation():
            if self._stream is None:
                self._stream = torch.cuda.Stream(device=self.device)
            try:
                with torch.cuda.stream(self._stream), torch.profiler.record_function(
                    "v41_h2d_rgb_batch"
                ):
                    h2d_start = None
                    if VIT_PHASE_METRICS:
                        h2d_start = torch.cuda.Event(enable_timing=True)
                        h2d_start.record(self._stream)
                    storage = host.to(device=self.device, non_blocking=True)
                    done = torch.cuda.Event(enable_timing=h2d_start is not None)
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
                result = UploadedRGBBatch(data_list, host, storage, done, h2d_start)
            except Exception:
                # Includes allocation/event failures after a copy was submitted.
                # Do not release host storage or silently retry on another path.
                self._stream.synchronize()
                raise
        self.batches += 1
        self.bytes += total
        if VIT_PHASE_METRICS:
            report_vit_phase(GaugeMetrics.VIT_RGB_UPLOAD_BYTES_METRIC, total, {})
        return result

    def stats(self):
        return {
            "batches": self.batches,
            "bytes": self.bytes,
            "budget_fallbacks": self.budget_fallbacks,
            "max_bytes_per_batch": self.max_bytes,
        }

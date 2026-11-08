"""Bounded registered CUDA transport for the existing CPU object codec.

The wire payload uses PACE GDR. Serialization and SHA256 validation still run
on CPU; this is not a claim that the tensor-tree codec is GPU-only.
"""

import logging
import threading

import torch

from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate


class RegisteredObjectTransport:
    transport_mode = "gdr"
    _GUARD_BYTES = 65536
    _quarantine_lock = threading.Lock()
    _quarantined_transports = []

    def __init__(self, factory, *, max_object_bytes, device):
        if max_object_bytes <= 0 or max_object_bytes > 1024**3:
            raise ValueError("invalid registered object capacity")
        self._lock = threading.Lock()
        self._closed = False
        self._poisoned = False
        self._capacity = max_object_bytes
        self._device = torch.device(device)
        if self._device.type != "cuda":
            raise ValueError("GDR transport requires a CUDA device")
        # A dedicated allocation pool isolates registered storage from the
        # model allocator. Reserve a nonzero MR-relative offset: PACE's normal
        # single-IOV interface treats a zero IOVA as an invalid address.
        # A failed transfer/registration may leave DMA outstanding. Refuse new
        # arenas after failure so retries cannot accumulate quarantined memory.
        with self._quarantine_lock:
            if self._quarantined_transports:
                raise RuntimeError("GDR transport is quarantined; restart the worker")
            try:
                with cuda_graph_gate.operation(), torch.cuda.device(self._device):
                    self._pool = torch.cuda.MemPool()
                    with torch.cuda.use_mem_pool(self._pool):
                        self._arena = torch.empty(
                            self._GUARD_BYTES + max_object_bytes,
                            dtype=torch.uint8,
                            device=self._device,
                        )
                    self._client = factory(gpu_tensors=(self._arena,))
            except Exception:
                self._poisoned = True
                self._quarantined_transports.append(self)
                raise
        self.instance_id = self._client.instance_id
        self.instance_group = self._client.instance_group

    def _check_available(self):
        if self._closed or self._poisoned:
            raise RuntimeError(
                "registered transport unavailable; restart worker after GDR failure"
            )

    def _quarantine(self):
        if not self._poisoned:
            self._poisoned = True
            with self._quarantine_lock:
                self._quarantined_transports.append(self)
            logging.error("GDR transfer failed; arena quarantined until worker restart")

    def _view(self, tensor):
        self._check_available()
        if (
            tensor.device.type != "cpu"
            or tensor.dtype != torch.uint8
            or tensor.ndim != 1
            or not tensor.is_contiguous()
            or not 0 < tensor.numel() <= self._capacity
        ):
            raise ValueError("invalid registered object buffer")
        return self._arena[self._GUARD_BYTES : self._GUARD_BYTES + tensor.numel()]

    def object_size(self, key, **kwargs):
        with self._lock:
            self._check_available()
            return self._client.object_size(key, **kwargs)

    def save_one(self, key, buffer, **kwargs):
        with self._lock, cuda_graph_gate.operation(), torch.cuda.device(self._device):
            transfer = self._view(buffer)
            try:
                transfer.copy_(buffer, non_blocking=False)
                torch.cuda.current_stream(self._device).synchronize()
                self._client.save_one(key, transfer, **kwargs)
                torch.cuda.current_stream(self._device).synchronize()
            except Exception:
                self._quarantine()
                raise

    def load_one(self, key, buffer, **kwargs):
        with self._lock, cuda_graph_gate.operation(), torch.cuda.device(self._device):
            transfer = self._view(buffer)
            try:
                self._client.load_one(key, transfer, **kwargs)
                torch.cuda.current_stream(self._device).synchronize()
                buffer.copy_(transfer, non_blocking=False)
                return buffer
            except Exception:
                self._quarantine()
                raise

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._poisoned:
                # A timeout is not proof that peer DMA stopped. Keep the native
                # client, registration, arena AND allocator pool alive together.
                # This transport cannot be reused or recreated in this worker.
                self._closed = True
                return
            try:
                self._client.close()
            except Exception:
                self._quarantine()
                raise
            self._closed = True
            self._arena = None
            self._pool = None

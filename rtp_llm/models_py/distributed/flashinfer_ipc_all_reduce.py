# SPDX-License-Identifier: Apache-2.0
"""Small TP2 PCIe reductions through FlashInfer's vLLM IPC implementation.

Use staging for eager/aliased calls; opt-in captured inputs may register directly. FlashInfer records rank pointer slots
during capture even for staging buffers, so finish_graph_capture must register
these slots before the first replay. Only the ordered communication stream owns
the shared staging buffers.
The algorithm and API are provided by flashinfer.comm.vllm_ar (Apache-2.0).
"""

import ctypes
import logging

import torch
import torch.distributed as dist


class FlashInferIpcAllReduce:
    def __init__(self, group, device, max_bytes=32 * 2560 * 2):
        self.group = group
        self.device = device
        self.rank = dist.get_rank(group)
        self.max_bytes = max_bytes
        self.capture_calls = 0
        self.registered_graph_inputs = True
        self.handle = 0
        self.rank_data = None
        self._allocated = []
        self._opened = []
        error = None
        handles = None
        try:
            from flashinfer.comm import cuda_ipc, vllm_ar

            self.ops = vllm_ar
            self.ipc = cuda_ipc
            self._close_ipc = cuda_ipc.cudart.lib.cudaIpcCloseMemHandle
            self._close_ipc.argtypes = [ctypes.c_void_p]
            self._close_ipc.restype = ctypes.c_int
            self.rank_data = torch.empty(
                8 * 1024 * 1024, device=device, dtype=torch.uint8
            )
            meta_bytes = vllm_ar.meta_size() + max_bytes
            for size in (meta_bytes, max_bytes):
                self._allocated.append(cuda_ipc.cudart.cudaMalloc(size))
            cuda_ipc.cudart.cudaMemset(self._allocated[0], 0, meta_bytes)
            handles = tuple(
                cuda_ipc.cudart.cudaIpcGetMemHandle(p) for p in self._allocated
            )
            torch.cuda.synchronize(device)
        except Exception as exc:
            error = exc
            self._release_local()
        # Local imports, allocation and memset finish before the first exchange.
        # All ranks vote even when only one rank fails; no dependency helper may
        # interleave its hidden collectives with these fallible local operations.
        peers = [None] * dist.get_world_size(group)
        dist.all_gather_object(peers, (error is None, handles), group=group)
        if not all(peer[0] for peer in peers):
            self._release_local()
            raise RuntimeError("IPC allocation failed on a TP rank") from error
        try:
            pointers = []
            for index in range(2):
                shared = []
                for rank, (_, peer_handles) in enumerate(peers):
                    if rank == self.rank:
                        pointer = self._allocated[index]
                    else:
                        pointer = self.ipc.cudart.cudaIpcOpenMemHandle(
                            peer_handles[index]
                        )
                        self._opened.append(pointer)
                    shared.append(pointer.value)
                pointers.append(shared)
            self.meta, self.buffer = pointers
            self.handle = self.ops.init_custom_ar(
                self.meta, self.rank_data, self.rank, False
            )
            self.ops.register_buffer(self.handle, self.buffer)
            torch.cuda.synchronize(device)
        except Exception as exc:
            error = exc
        dist.all_gather_object(peers, error is None, group=group)
        if not all(peers):
            self._release_local()
            raise RuntimeError("IPC registration failed on a TP rank") from error

    def _release_local(self):
        # Cleanup never contains a collective: partial construction is allowed.
        actions = []
        if self.handle:
            actions.append((self.ops.dispose, self.handle))
            self.handle = 0
        actions.extend((self._close_mapping, p) for p in self._opened)
        actions.extend((self.ipc.cudart.cudaFree, p) for p in self._allocated)
        self._opened = []
        self._allocated = []
        for function, pointer in actions:
            try:
                function(pointer)
            except Exception:
                logging.exception("Failed to release local IPC allocation")
        self.rank_data = None

    def _close_mapping(self, pointer):
        self.ipc.cudart.CUDART_CHECK(self._close_ipc(pointer))

    def is_buffer_size_sufficient(self, world_size, rows, hidden, dtype):
        return (
            world_size == 2
            and dtype == torch.bfloat16
            and hidden == 2560
            and (
                1 <= rows <= 32
                or (
                    getattr(self, "large_prefill", False)
                    and rows in (4093, 4094, 4095, 4096, 8192, 12288, 16384)
                    and not torch.cuda.is_current_stream_capturing()
                )
            )
            and rows * hidden * 2 <= self.max_bytes
        )

    def all_reduce(self, tensor, output=None):
        if output is None:
            output = torch.empty_like(tensor)
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing:
            self.capture_calls += 1
        # A distinct output keeps every registered input unchanged while peers
        # read it. Full/partial aliases, including independent C++ storage wrappers over
        # overlapping byte ranges, must retain staging.
        direct = (
            capturing
            and self.registered_graph_inputs
            and tensor.is_contiguous()
            and output.is_contiguous()
            and tensor.untyped_storage().data_ptr()
            != output.untyped_storage().data_ptr()
            and (
                tensor.data_ptr() + tensor.numel() * tensor.element_size()
                <= output.data_ptr()
                or output.data_ptr() + output.numel() * output.element_size()
                <= tensor.data_ptr()
            )
        )
        self.ops.all_reduce(
            self.handle,
            tensor,
            output,
            0 if direct else self.buffer[self.rank],
            self.max_bytes,
            36,
        )
        return output

    def finish_graph_capture(self):
        if not self.capture_calls:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("IPC pointer registration requires completed capture")
        handle, offsets = self.ops.get_graph_buffer_ipc_meta(self.handle)
        if len(offsets) != self.capture_calls:
            raise RuntimeError("IPC capture pointer count changed")
        peers = [None] * dist.get_world_size(self.group)
        dist.all_gather_object(peers, (handle, offsets), group=self.group)
        self.ops.register_graph_buffers(
            self.handle, [p[0] for p in peers], [p[1] for p in peers]
        )
        self.capture_calls = 0

    def destroy(self):
        if self.handle:
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.group)
            self._release_local()
            dist.barrier(group=self.group)

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

from rtp_llm.models_py.distributed.collective_utils import exchange_group_state


class IpcInitializationError(RuntimeError):
    """All ranks agreed that local IPC setup failed; NCCL is still usable."""


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
            raise IpcInitializationError(
                "IPC allocation failed on a TP rank"
            ) from error
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
            self._release_shared()
            raise IpcInitializationError(
                "IPC registration failed on a TP rank"
            ) from error

    def _close_local(self):
        closed = True
        if self.handle:
            try:
                self.ops.dispose(self.handle)
                self.handle = 0
            except Exception:
                closed = False
                logging.exception("Failed to dispose IPC workspace")
        pending = []
        for pointer in self._opened:
            try:
                self._close_mapping(pointer)
            except Exception:
                closed = False
                pending.append(pointer)
                logging.exception("Failed to close IPC mapping")
        self._opened = pending
        if not self.handle:
            self.rank_data = None
        return closed

    def _free_local(self):
        pending = []
        for pointer in self._allocated:
            try:
                self.ipc.cudart.cudaFree(pointer)
            except Exception:
                pending.append(pointer)
                logging.exception("Failed to free IPC allocation")
        self._allocated = pending

    def _release_local(self):
        # Before handles are exchanged, no peer can have imported this memory.
        if self._close_local():
            self._free_local()

    def _release_shared(self):
        # CUDA requires every importer to close before the exporter frees.
        # A barrier around the whole cleanup does not order these two phases.
        closed = self._close_local()
        peers = exchange_group_state(self.group, closed, "ipc_close_mappings")
        if all(peers):
            self._free_local()
        else:
            # Keep raw exports alive until process exit if any close failed.
            logging.error(
                "Retaining IPC allocations: a TP rank could not close mappings"
            )

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
        # Different ranks can receive views with different storage offsets.
        # Keep the agreed backend, staging unaligned inputs and copying aligned
        # kernel output back into an unaligned destination when necessary.
        source = tensor if tensor.data_ptr() % 16 == 0 else tensor.clone()
        result = output if output.data_ptr() % 16 == 0 else torch.empty_like(tensor)
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
            and tensor.data_ptr() % 16 == 0
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
            source,
            result,
            0 if direct else self.buffer[self.rank],
            self.max_bytes,
            36,
        )
        if result is not output:
            output.copy_(result)
        return output

    def finish_graph_capture(self):
        error = None
        handle, offsets = None, []
        try:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "IPC pointer registration requires completed capture"
                )
            if self.capture_calls:
                handle, offsets = self.ops.get_graph_buffer_ipc_meta(self.handle)
                if len(offsets) != self.capture_calls:
                    raise RuntimeError("IPC capture pointer count changed")
        except Exception as exc:
            error = str(exc)
        # Even a rank with zero calls or invalid local metadata must exchange
        # its verdict before returning, so peers cannot wait alone.
        # Handles, offsets and verdicts are host metadata. Exchange them on
        # the existing CPU store instead of copying objects through NCCL/GPU.
        peers = exchange_group_state(
            self.group,
            [error, self.capture_calls, handle, offsets],
            "ipc_graph_registration",
        )
        if any(p[0] is not None or p[1] != self.capture_calls for p in peers):
            raise RuntimeError("IPC capture metadata differs across TP ranks")
        if not self.capture_calls:
            return
        self.ops.register_graph_buffers(
            self.handle, [p[2] for p in peers], [p[3] for p in peers]
        )
        self.capture_calls = 0

    def destroy(self):
        if self.handle:
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.group)
            self._release_shared()
            dist.barrier(group=self.group)

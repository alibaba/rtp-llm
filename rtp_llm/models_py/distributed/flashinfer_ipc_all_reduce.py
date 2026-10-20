# SPDX-License-Identifier: Apache-2.0
"""Small TP2 PCIe reductions through FlashInfer's vLLM IPC implementation.

Use staging for eager/aliased calls; opt-in captured inputs may register directly. FlashInfer records rank pointer slots
during capture even for staging buffers, so finish_graph_capture must register
these slots before the first replay. Only the ordered communication stream owns
the shared staging buffers.
The algorithm and API are provided by flashinfer.comm.vllm_ar (Apache-2.0).
"""

import ctypes
import os

import torch
import torch.distributed as dist


class FlashInferIpcAllReduce:
    def __init__(self, group, device, max_bytes=32 * 2560 * 2):
        from flashinfer.comm import cuda_ipc, vllm_ar

        self.group = group
        self.rank = dist.get_rank(group)
        self.ops = vllm_ar
        self.ipc = cuda_ipc
        self.max_bytes = max_bytes
        self.capture_calls = 0
        self.registered_graph_inputs = (
            os.environ.get("QWEN4_TP_REGISTERED_INPUTS", "0") == "1"
        )
        self.meta = cuda_ipc.create_shared_buffer(
            vllm_ar.meta_size() + max_bytes, group
        )
        cuda_ipc.cudart.cudaMemset(
            ctypes.c_void_p(self.meta[self.rank]), 0, vllm_ar.meta_size() + max_bytes
        )
        self.buffer = cuda_ipc.create_shared_buffer(max_bytes, group)
        self.rank_data = torch.empty(8 * 1024 * 1024, device=device, dtype=torch.uint8)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        self.handle = vllm_ar.init_custom_ar(
            self.meta, self.rank_data, self.rank, False
        )
        vllm_ar.register_buffer(self.handle, self.buffer)

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
            self.ops.dispose(self.handle)
            self.handle = 0
            self.ipc.free_shared_buffer(self.meta, self.group)
            self.ipc.free_shared_buffer(self.buffer, self.group)
            self.rank_data = None

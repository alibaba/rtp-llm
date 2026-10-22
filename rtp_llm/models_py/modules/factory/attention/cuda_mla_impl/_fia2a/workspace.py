"""TP-group output storage; each graph keeps the generation it captured."""

import cutlass
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
import triton
import tvm_ffi

from .ops import CONTROL_SIZE
from .pull_merge import compile_pull_merge, warps_for


class _OutputBuffers:
    def __init__(self, heads, capacity, device, dtype, group=None):
        self.capacity = capacity
        self.world = dist.get_world_size(group) if group is not None else 1
        local_heads = heads // self.world
        allocate = symm.empty if group is not None else torch.empty
        self.output = allocate((self.world, capacity, local_heads, 512), dtype=dtype, device=device)
        self.lse = allocate((self.world, capacity, local_heads), dtype=torch.float32, device=device)
        if group is None:
            pointers = [(self.output.data_ptr(), self.lse.data_ptr())]
            self.output_ptrs = (cutlass.Int64(self.output.data_ptr()),)
        else:
            # The producer's last CTA publishes completion into each peer's
            # ready slot and waits for every peer's on its local copy.
            self.control = symm.empty((CONTROL_SIZE,), dtype=torch.int64, device=device)
            self.handles = [symm.rendezvous(t, group=group) for t in (self.output, self.lse, self.control)]
            self.control.zero_()
            # Every peer must finish initialization before publication can start.
            torch.cuda.synchronize()
            dist.barrier(group=group)
            pointers = list(zip(self.handles[0].buffer_ptrs, self.handles[1].buffer_ptrs))
            self.output_ptrs = tuple(cutlass.Int64(p) for p in self.handles[0].buffer_ptrs)
            self.control_peers = torch.tensor(self.handles[2].buffer_ptrs, device=device, dtype=torch.int64)
        self.pointers = torch.tensor(pointers, device=device, dtype=torch.int64)


class _AllToAllBuffers:
    """Direct O/LSE sources and replay epochs for a serial TP group.

    No consumption barrier: a layer's following TP collective orders each
    rank's next source write after every peer's reads of the current sources.
    """

    def __init__(self, heads, capacity, device, group):
        self.capacity = capacity
        self.world, self.rank = dist.get_world_size(group), dist.get_rank(group)
        self.local_heads = heads // self.world
        # Match local producer storage (world=1), while making its final O/LSE
        # readable by peers. Split partials themselves remain private FP32.
        self.output = symm.empty((1, capacity, heads, 512), dtype=torch.bfloat16, device=device)
        self.lse = symm.empty((1, capacity, heads), dtype=torch.float32, device=device)
        self.control = symm.empty((CONTROL_SIZE,), dtype=torch.int64, device=device)
        self.handles = [symm.rendezvous(t, group=group) for t in (self.output, self.lse, self.control)]
        pointers = list(zip(self.handles[0].buffer_ptrs, self.handles[1].buffer_ptrs))
        self.peers = torch.tensor(pointers, dtype=torch.int64, device=device)
        self.control_peers = torch.tensor(self.handles[2].buffer_ptrs, dtype=torch.int64, device=device)
        self.pointers = torch.tensor([[self.output.data_ptr(), self.lse.data_ptr()]], dtype=torch.int64, device=device)
        self.output_ptrs = (cutlass.Int64(self.output.data_ptr()),)
        self.sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        self.warps = warps_for(self.world)
        self.kernel = compile_pull_merge(self.world, self.local_heads, heads, self.warps)
        self.control.zero_()
        torch.cuda.synchronize()
        dist.barrier(group=group)

    def combine(self, rows):
        output = self.output.new_empty(self.local_heads, rows, 512)
        # At most one CTA per SM; each warp owns an output row.
        ctas = min(self.sm_count, triton.cdiv(rows * self.local_heads, self.warps))
        with tvm_ffi.use_torch_stream():
            self.kernel(self.peers, self.control_peers, output,
                        cutlass.Int32(rows), cutlass.Int32(self.rank), cutlass.Int32(ctas))
        return output


class FIA2AWorkspace:
    """Growing local/peer allocations per wire dtype for one serial TP group.

    Backend instances retain their allocation when a later eager step grows the
    group's capacity. Captured pointers and rendezvous handles then remain valid.
    No allocation is keyed by every historical batch size.
    """

    def __init__(self, communicator):
        self.communicator = communicator
        self.buffers = {}
        self.all_to_all_buffers = None
        self.prepared = set()

    def acquire_all_to_all(self, rows):
        current = self.all_to_all_buffers
        if current is None or current.capacity < rows:
            capacity = max(16, 1 << (rows - 1).bit_length())
            current = _AllToAllBuffers(self.communicator.heads, capacity,
                                      self.communicator.device, self.communicator.group)
            self.all_to_all_buffers = current
        return current

    def acquire(self, rows, splits, fused):
        dtype = torch.bfloat16 if splits == 1 else torch.float32
        key = (fused, dtype)
        current = self.buffers.get(key)
        required = rows * splits
        if current is None or current.capacity < required:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("FIA2A workspace must be prepared before graph capture")
            # Geometric growth avoids rendezvous on each eager batch-size change.
            capacity = max(16, 1 << (required - 1).bit_length())
            current = _OutputBuffers(
                self.communicator.heads, capacity, self.communicator.device, dtype,
                self.communicator.group if fused else None,
            )
            self.buffers[key] = current
        return current


_workspaces = {}


def get_workspace(communicator):
    device = communicator.device
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (communicator.group, index, communicator.heads,
           communicator.q_dim, communicator.latent_dim, communicator.dtype)
    if key not in _workspaces:
        _workspaces[key] = FIA2AWorkspace(communicator)
    return _workspaces[key]

"""TP-group output storage; each graph keeps the generation it captured."""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass.cute.runtime import from_dlpack
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
import triton

from .ops import _a2a_pull

@dsl_user_op
def read_globaltimer(*, loc=None, ip=None):
    # CuTe 4.4.2 has no arch.globaltimer wrapper. This register counts ns.
    return Int64(llvm.inline_asm(T.i64(), [], 'mov.u64 $0, %globaltimer;', '=l',
                                has_side_effects=True, asm_dialect=0, loc=loc, ip=ip))


@dsl_user_op
def signal_release(address: Int64, value: Int32, *, loc=None, ip=None):
    llvm.inline_asm(None, [address.ir_value(), value.ir_value()],
                    'red.release.sys.global.add.s32 [$0], $1;', 'l,r',
                    has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)


@dsl_user_op
def signal_acquire(address: Int64, *, loc=None, ip=None):
    return Int32(llvm.inline_asm(T.i32(), [address.ir_value()],
                                'ld.acquire.sys.global.s32 $0, [$1];', '=r,l',
                                has_side_effects=True, asm_dialect=0, loc=loc, ip=ip))


@dsl_user_op
def timeout_trap(*, loc=None, ip=None):
    llvm.inline_asm(None, [], 'trap;', '', has_side_effects=True,
                    asm_dialect=0, loc=loc, ip=ip)


class PeerBarrier:
    def __init__(self, world):
        self.world = world

    @cute.jit
    def __call__(self, peers, control, stream: cuda.CUstream):
        self.barrier(peers, control).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def barrier(self, peers, control):
        tid, _, _ = cute.arch.thread_idx()
        status = control[0] & 3
        phase = status & 1
        subtract = (status >> 1) != 0
        delta = Int32(1)
        target = Int32(self.world)
        if subtract:
            delta = Int32(-1)
            target = Int32(0)
        if tid < self.world:
            address = peers[tid] + Int64((1 + phase) * 4)
            signal_release(address, delta)
        cute.arch.sync_threads()
        if tid == 0:
            # Only this thread updates this rank's local invocation counter.
            control[0] = control[0] + 1
            address = control.iterator.toint() + Int64((1 + phase) * 4)
            start = read_globaltimer()
            while signal_acquire(address) != target:
                if read_globaltimer() - start > Int64(5000000000):
                    timeout_trap()


@functools.cache
def compile_peer_barrier(world):
    # CuTe unloads an unreachable compiled library during Python GC. Keeping
    # stateless code alive prevents that unload from invalidating a later graph
    # capture. Fake arguments retain no workspace tensors or peer mappings.
    peers = cute.runtime.make_fake_compact_tensor(Int64, (world,), stride_order=(0,), assumed_align=16)
    control = cute.runtime.make_fake_compact_tensor(Int32, (32,), stride_order=(0,), assumed_align=16)
    return cute.compile(PeerBarrier(world), peers, control, cute.runtime.make_fake_stream())


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
            self.control = symm.empty((32,), dtype=torch.int32, device=device)
            self.handles = [symm.rendezvous(t, group=group) for t in (self.output, self.lse, self.control)]
            self.control.zero_()
            # Every peer must finish initialization before publication can start.
            torch.cuda.synchronize()
            dist.barrier(group=group)
            pointers = list(zip(self.handles[0].buffer_ptrs, self.handles[1].buffer_ptrs))
            self.output_ptrs = tuple(cutlass.Int64(p) for p in self.handles[0].buffer_ptrs)
            self.control_peers = torch.tensor(self.handles[2].buffer_ptrs, device=device, dtype=torch.int64)
            self.barrier_args = tuple(from_dlpack(t, assumed_align=16) for t in (self.control_peers, self.control))
            self.barrier = compile_peer_barrier(self.world)
        self.pointers = torch.tensor(pointers, device=device, dtype=torch.int64)

    def synchronize_peers(self):
        self.barrier(*self.barrier_args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))


class _AllToAllBuffers:
    """Packed symmetric sources and local receive storage for a serial TP group."""

    def __init__(self, heads, capacity, device, group):
        self.capacity = capacity
        self.world, self.rank = dist.get_world_size(group), dist.get_rank(group)
        self.local_heads = heads // self.world
        self.send = symm.empty((capacity * heads * 514,), dtype=torch.bfloat16, device=device)
        self.received = torch.empty_like(self.send)
        self.control = symm.empty((32,), dtype=torch.int32, device=device)
        self.handles = [symm.rendezvous(t, group=group) for t in (self.send, self.control)]
        self.peers = torch.tensor(self.handles[0].buffer_ptrs, dtype=torch.int64, device=device)
        self.control_peers = torch.tensor(self.handles[1].buffer_ptrs, dtype=torch.int64, device=device)
        self.barrier_args = tuple(from_dlpack(t, assumed_align=16) for t in (self.control_peers, self.control))
        self.barrier = compile_peer_barrier(self.world)
        self.control.zero_()
        torch.cuda.synchronize()
        dist.barrier(group=group)

    def packed(self, rows):
        # The wire rank stride is the current token count, not arena capacity.
        shape = (self.world, rows, self.local_heads, 514)
        return self.send[:self.world * rows * self.local_heads * 514].view(shape)

    def warmup(self, rows):
        words = rows * self.local_heads * 257
        return _a2a_pull.warmup(
            self.peers, self.received.view(torch.int32), words, self.rank, self.world, 4096,
            num_warps=4, grid=(self.world * triton.cdiv(words, 4096),),
        )

    def synchronize(self):
        self.barrier(*self.barrier_args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))

    def exchange(self, rows):
        words = rows * self.local_heads * 257
        self.synchronize()
        _a2a_pull[(self.world * triton.cdiv(words, 4096),)](
            self.peers, self.received.view(torch.int32), words, self.rank, self.world, 4096,
            num_warps=4,
        )
        # Kernel completion precedes publication. Every peer finishes reading
        # before the next pack can overwrite its source; local receive reuse
        # follows the local combine on the same stream.
        self.synchronize()
        return self.received[:2 * self.world * words].view(self.world, rows, self.local_heads, 514)


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

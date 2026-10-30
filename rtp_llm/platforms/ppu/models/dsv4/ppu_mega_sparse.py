"""Opt-in sparse GPU-initiated RDMA transport for TP1 EP32 MegaMoE."""
import ctypes as C
import os
from importlib.util import find_spec
from pathlib import Path

import torch
import triton

from rtp_llm.platforms.ppu.models.dsv4.ppu_mega_collective import PpuMegaCommunicator
from .ppu_mega_sparse_kernels import pack_sparse, unpack_sparse, pack_return, unpack_return


def _transport_library(name):
    package = find_spec('deep_gemm_mega')
    if package is None or package.origin is None:
        raise RuntimeError('Sparse LargeEP requires the deep-gemm-mega wheel')
    library = Path(package.origin).parent / 'lib' / name
    if not library.is_file():
        raise RuntimeError(f'Missing {library}; rebuild the LargeEP deep-gemm-mega wheel')
    return C.CDLL(str(library))


class Allocation:
    def __init__(self, lib, words):
        self.lib = lib
        self.pointer = C.c_void_p()
        lib.pcclMemAlloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
        lib.pcclMemFree.argtypes = [C.c_void_p]
        rc = lib.pcclMemAlloc(C.byref(self.pointer), words * 4)
        if rc:
            raise RuntimeError(f"pcclMemAlloc failed: {rc}")
        self.__cuda_array_interface__ = {
            'shape': (words,), 'strides': None, 'typestr': '<i4',
            'data': (self.pointer.value, False), 'version': 3,
            'stream': torch.cuda.current_stream().cuda_stream or 1,
        }
        self.tensor = torch.as_tensor(self, device='cuda')
        self.tensor.zero_()


class Exchange:
    def __init__(self, communicator, capacity, row_bytes):
        self.comm = communicator
        self.stride = 16 + capacity * row_bytes
        self.tx = Allocation(communicator._lib, self.stride // 4 * communicator.size)
        self.rx = Allocation(communicator._lib, self.stride // 4 * communicator.size)
        self.native = _transport_library('libsparse_gin.so')
        self.native.sparse_gin_create.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p,
            C.c_size_t, C.c_size_t, C.c_int, C.c_int, C.POINTER(C.c_void_p)]
        self.native.sparse_gin_begin.argtypes = [C.c_void_p, C.c_void_p]
        self.native.sparse_gin_finish.argtypes = [C.c_void_p, C.c_void_p]
        self.native.sparse_gin_destroy.argtypes = [C.c_void_p]
        self.handle = C.c_void_p()
        self.check(self.native.sparse_gin_create(communicator._comm, self.tx.tensor.data_ptr(),
            self.rx.tensor.data_ptr(), self.stride, row_bytes,
            communicator.rank, communicator.size, C.byref(self.handle)))

    @staticmethod
    def check(result):
        if result:
            raise RuntimeError(f"Sparse GIN failure: {result}")

    def begin(self):
        self.check(self.native.sparse_gin_begin(self.handle, torch.cuda.current_stream().cuda_stream))

    def finish(self):
        self.check(self.native.sparse_gin_finish(self.handle, torch.cuda.current_stream().cuda_stream))

    def close(self):
        self.check(self.native.sparse_gin_destroy(self.handle))
        del self.tx.tensor, self.rx.tensor
        self.comm._check(self.comm._lib.pcclMemFree(self.tx.pointer))
        self.comm._check(self.comm._lib.pcclMemFree(self.rx.pointer))


class SignalExchange(Exchange):
    """Only for serial, paired dispatch/return; close is collective across peers."""

    def __init__(self, communicator, capacity, row_bytes):
        self.comm = communicator
        self.stride = 16 + capacity * row_bytes
        self.tx = Allocation(communicator._lib, self.stride // 4 * communicator.size)
        self.rx = Allocation(communicator._lib, self.stride // 4 * communicator.size)
        self.native = _transport_library('libsignal_gin.so')
        self.native.signal_gin_create.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p,
            C.c_size_t, C.c_size_t, C.c_int, C.c_int, C.POINTER(C.c_void_p)]
        self.native.signal_gin_exchange.argtypes = [C.c_void_p, C.c_void_p]
        self.native.signal_gin_destroy.argtypes = [C.c_void_p]
        self.handle = C.c_void_p()
        self.check(self.native.signal_gin_create(communicator._comm,
            self.tx.tensor.data_ptr(), self.rx.tensor.data_ptr(), self.stride,
            row_bytes, communicator.rank, communicator.size, C.byref(self.handle)))

    def begin(self):
        self.check(self.native.signal_gin_exchange(self.handle,
            torch.cuda.current_stream().cuda_stream))

    def finish(self):
        pass

    def close(self):
        if not self.handle.value:
            return
        self.check(self.native.signal_gin_destroy(self.handle))
        self.handle = C.c_void_p()
        del self.tx.tensor, self.rx.tensor
        self.comm._check(self.comm._lib.pcclMemFree(self.tx.pointer))
        self.comm._check(self.comm._lib.pcclMemFree(self.rx.pointer))


class SparseHierarchy:
    def __init__(self, base, capacity):
        self.base = base
        self.capacity, self.nodes, self.node = capacity, base.nodes, base.node
        self.dim, self.experts = base.dim, base.experts
        self.words = self.dim // 2 + 12
        self.comm = PpuMegaCommunicator(base.cross_group)
        signal_only = (self.nodes == 4
                       and os.environ.get('DSV4_PPU_MEGAMOE_SIGNAL_ONLY', '0') == '1')
        exchange = SignalExchange if signal_only else Exchange
        self.dispatch_exchange = exchange(self.comm, capacity, self.words * 4)
        self.return_exchange = exchange(self.comm, capacity, self.dim * 2)
        self.inverse = torch.empty((self.nodes, capacity), device='cuda', dtype=torch.int32)
        self.x = torch.empty((self.nodes * capacity, self.dim), device='cuda', dtype=torch.bfloat16)
        self.ids = torch.empty((self.nodes * capacity, 8), device='cuda', dtype=torch.int64)
        self.weights = torch.empty_like(self.ids, dtype=torch.float32)
        self.output = torch.empty((capacity, self.dim), device='cuda', dtype=torch.bfloat16)

    def start(self, x, weights, ids, active=None):
        if x.size(0) > self.capacity:
            raise ValueError('Sparse transport capacity exceeded')
        e = self.dispatch_exchange
        pack_sparse[(self.nodes, self.capacity)](x, ids, weights, active, e.tx.tensor,
            self.inverse, x.size(0), self.capacity, self.dim, self.words, e.stride // 4,
            self.experts, active is not None, triton.next_power_of_2(self.capacity),
            triton.next_power_of_2(self.dim), num_warps=4)
        e.begin()

    def unpack(self):
        e = self.dispatch_exchange
        unpack_sparse[(self.nodes, self.capacity)](e.rx.tensor, self.x, self.ids,
            self.weights, self.capacity, self.dim, self.words, e.stride // 4,
            self.node, self.experts, triton.next_power_of_2(self.dim), num_warps=4)
        return self.x, self.ids, self.weights

    def finish(self, y, count):
        d, e = self.dispatch_exchange, self.return_exchange
        pack_return[(self.nodes, self.capacity)](y, d.rx.tensor, e.tx.tensor,
            self.capacity, self.dim, d.stride // 4, e.stride // 4,
            triton.next_power_of_2(self.dim), num_warps=4)
        e.begin()
        e.finish()
        unpack_return[(self.capacity, triton.cdiv(self.dim, 256))](e.rx.tensor,
            self.inverse, self.output, self.capacity, self.dim, e.stride // 4,
            self.nodes, 256, num_warps=4)
        return self.output[:count]

    def forward(self, x, weights, ids, kernel, active=None):
        self.start(x, weights, ids, active)
        self.dispatch_exchange.finish()
        y = kernel(*self.unpack())
        return self.finish(y, x.size(0))

    def close(self):
        self.dispatch_exchange.close()
        self.return_exchange.close()
        self.comm.close()

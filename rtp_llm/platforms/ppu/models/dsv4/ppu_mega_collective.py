"""PCCL communicator ownership for sparse cross-host RDMA."""

import ctypes
from pathlib import Path

import torch
import torch.distributed as dist


class _UniqueId(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * 128)]


class PpuMegaCommunicator:
    """Owned by serial model resources, which must outlive captured graphs."""

    def __init__(self, group):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("MegaMoE communicator requires eager initialization")
        paths = list(Path("/usr/local/PPU_SDK/pccl").glob("lib*/libpccl.so"))
        if len(paths) != 1:
            raise RuntimeError(f"Expected one installed PCCL library: {paths}")
        self._lib = ctypes.CDLL(str(paths[0]))
        pointer, integer, count = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
        signatures = {
            "pcclGetUniqueId": [ctypes.POINTER(_UniqueId)],
            "pcclCommInitRank": [ctypes.POINTER(pointer), integer, _UniqueId, integer],
            "pcclGetErrorString": [integer],
            "pcclCommDestroy": [pointer],
        }
        for name, arguments in signatures.items():
            function = getattr(self._lib, name)
            function.argtypes, function.restype = arguments, integer
        self._lib.pcclGetErrorString.restype = ctypes.c_char_p
        self.rank, self.size = dist.get_rank(group), dist.get_world_size(group)
        self.device = torch.cuda.current_device()
        uid = _UniqueId()
        if self.rank == 0:
            self._check(self._lib.pcclGetUniqueId(ctypes.byref(uid)))
        objects = [bytes(uid)]
        dist.broadcast_object_list(objects, src=dist.get_global_rank(group, 0), group=group)
        self._comm = pointer()
        self._check(self._lib.pcclCommInitRank(
            ctypes.byref(self._comm), self.size,
            _UniqueId.from_buffer_copy(objects[0]), self.rank,
        ))

    def _check(self, result):
        if result:
            raise RuntimeError(self._lib.pcclGetErrorString(result).decode())

    def close(self):
        """Call only after all graphs using this communicator have been destroyed."""
        if self._comm.value is not None:
            torch.cuda.synchronize(self.device)
            self._check(self._lib.pcclCommDestroy(self._comm))
            self._comm = ctypes.c_void_p()

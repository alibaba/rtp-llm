"""Sealed memfd storages, optionally registered for CUDA reads in this process.

The mmap is itself the buffer owner retained by torch.frombuffer: no wrapper or
Tensor attribute has to keep it alive. It owns no tensors. File sizes are page
aligned; callers expose and checksum only the logical storage bytes. CUDA access
is requested by a consumer, never inferred merely from a tensor being on CPU.
"""

import ctypes
import fcntl
import mmap
import os
import sys
import warnings
from typing import List, Optional

import torch

# Linux UAPI; some supported Conda builds omit the corresponding Python names.
_F_ADD_SEALS, _F_GET_SEALS, _SEALS = 1033, 1034, 0x000F
_deferred_releases: List["CpuAllocation"] = []


class CpuAllocation(mmap.mmap):
    """One immutable CPU backing; every storage/view retains this actual buffer.

    Registered mappings follow VMM's release rule: wait for queued GPU readers,
    and defer destruction requested inside graph capture until a safe release.
    The store retains only an exported FD, not a mapping or CUDA registration.
    """

    @classmethod
    def create(cls, source: torch.UntypedStorage, device: int) -> "CpuAllocation":
        logical_bytes = source.nbytes()
        if not logical_bytes:
            raise ValueError("empty CPU storages need metadata, not an allocation")
        data = torch.empty(0, dtype=torch.uint8).set_(source)
        cuda_access = torch.cuda.is_available() and data.is_pinned()
        size = (logical_bytes + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE
        create = ctypes.CDLL(None, use_errno=True).memfd_create
        create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        create.restype = ctypes.c_int
        fd = create(b"rtp-resident-cpu", 3)  # CLOEXEC | ALLOW_SEALING
        if fd < 0:
            raise OSError(ctypes.get_errno(), "memfd_create")
        try:
            os.ftruncate(fd, size)
            with mmap.mmap(
                fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE
            ) as mapping:
                staging = torch.frombuffer(
                    mapping, dtype=torch.uint8, count=logical_bytes
                )
                try:
                    staging.copy_(data)
                finally:
                    del staging
            fcntl.fcntl(fd, _F_ADD_SEALS, _SEALS)
            # dup/SCM_RIGHTS preserve O_RDWR; reopen to obtain a read-only description.
            readonly = os.open(f"/proc/self/fd/{fd}", os.O_RDONLY | os.O_CLOEXEC)
            try:
                return cls.from_fd(readonly, size, device, cuda_access)
            finally:
                os.close(readonly)
        finally:
            os.close(fd)

    @classmethod
    def from_fd(
        cls, fd: int, size_bytes: int, device: int, cuda_access_required: bool = False
    ) -> "CpuAllocation":
        if os.fstat(fd).st_size != size_bytes or not size_bytes:
            raise ValueError("CPU backing size does not match its descriptor")
        if fcntl.fcntl(fd, _F_GET_SEALS) & _SEALS != _SEALS:
            raise ValueError("CPU backing must be sealed before importing it")
        owned = os.dup(fd)
        try:
            self = cls(owned, size_bytes, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
        except BaseException:
            os.close(owned)
            raise
        self.fd: Optional[int] = owned
        self.size_bytes = size_bytes
        self.cuda_access_required = False
        self._runtime = None
        self._registered_device: Optional[int] = None
        storage, _ = self.as_torch_storage(0, size_bytes)
        self.ptr = storage.data_ptr()
        del storage
        try:
            if cuda_access_required:
                self.ensure_cuda_access(torch.device("cuda", device))
        except BaseException:
            self._release()
            raise
        return self

    @property
    def allocation(self) -> "CpuAllocation":
        return self  # the buffer is also the allocation, unlike a CUDA array bridge

    def export_fd(self) -> int:
        return os.dup(self.fd)

    def as_torch_storage(self, offset_bytes: int, nbytes: int):
        # Torch has no readonly Tensor type; mmap/seals enforce our input contract.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The given buffer is not writable.*"
            )
            tensor = torch.frombuffer(
                self, dtype=torch.uint8, count=nbytes, offset=offset_bytes
            )
        return tensor.untyped_storage(), self

    def ensure_cuda_access(self, device: torch.device) -> None:
        self.cuda_access_required = (
            True  # a persisted requirement, not old runtime state
        )
        if self._registered_device is not None:
            return
        with torch.cuda.device(device):
            runtime = ctypes.CDLL(torch._C.__file__)
            attribute = runtime.cudaDeviceGetAttribute
            attribute.argtypes = [
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
            ]
            supported = ctypes.c_int()
            code = attribute(ctypes.byref(supported), 113, torch.cuda.current_device())
            if code:
                raise RuntimeError(
                    f"querying cudaDevAttrHostRegisterReadOnlySupported on cuda:"
                    f"{torch.cuda.current_device()} failed: CUDA {code}"
                )
            if not supported.value:
                raise RuntimeError(
                    f"resident CPU weights read by the GPU need read-only host registration "
                    f"(cudaDevAttrHostRegisterReadOnlySupported), which cuda:"
                    f"{torch.cuda.current_device()} lacks; {self.size_bytes} bytes were to be "
                    f"registered. Run without resident weights on this device"
                )
            register = runtime.cudaHostRegister
            register.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
            code = register(
                self.ptr, self.size_bytes, 11
            )  # Portable | Mapped | ReadOnly
            if code:
                raise RuntimeError(
                    f"resident CPU cudaHostRegister of {self.size_bytes} bytes on cuda:"
                    f"{torch.cuda.current_device()} failed: CUDA {code}"
                )
            runtime.cudaHostUnregister.argtypes = [ctypes.c_void_p]
            self._runtime = runtime
            self._registered_device = torch.cuda.current_device()

    def _release(self) -> None:
        if self.fd is None:
            return
        try:
            if self._runtime is not None:
                code = self._runtime.cudaHostUnregister(self.ptr)
                if code:
                    warnings.warn(
                        f"resident CPU cudaHostUnregister failed: CUDA {code}"
                    )
        finally:
            self.close()
            os.close(self.fd)
            self.fd = None

    def __del__(self):
        if sys.is_finalizing() or getattr(self, "fd", None) is None:
            return
        if self._registered_device is not None:
            with torch.cuda.device(self._registered_device):
                if torch.cuda.is_current_stream_capturing():
                    _deferred_releases.append(self)
                    return
                torch.cuda.synchronize(self._registered_device)
                self._release()
                while _deferred_releases:
                    _deferred_releases.pop()._release()
        else:
            self._release()

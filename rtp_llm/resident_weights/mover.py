"""Moves one resident record's GPU allocations between CUDA VMM and sealed host memfds.

The store spawns one mover per move, with ``CUDA_VISIBLE_DEVICES`` set to the record's GPU and
one end of a socket pair in ``RTP_LLM_RESIDENT_MOVER_FD``. It receives the job and the source
FDs, builds the complete destination, sends its FDs back and exits; the store then swaps them
in and closes the sources. A failure sends an error instead and leaves the sources untouched.
"""

import os
import socket
import sys
from typing import List, Sequence

import torch

from rtp_llm.resident_weights import store
from rtp_llm.resident_weights.cpu import CpuAllocation, sealed_memfd
from rtp_llm.resident_weights.vmm import VmmAllocation

MOVER_FD_ENV = "RTP_LLM_RESIDENT_MOVER_FD"
_DEVICE = 0  # CUDA_VISIBLE_DEVICES leaves only the record's GPU


def _register(tensor: torch.Tensor) -> None:
    """Page-lock a host mapping: unregistered copies run at a fraction of PCIe bandwidth."""
    code = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), tensor.numel(), 0)
    if int(code):
        raise RuntimeError(f"cudaHostRegister failed: CUDA {int(code)}")


def to_host(fds: Sequence[int], sizes: Sequence[int]) -> List[int]:
    """One sealed memfd holding each VMM allocation's bytes."""
    out: List[int] = []
    try:
        for fd, size in zip(fds, sizes):
            gpu = VmmAllocation.from_fd(fd, size, _DEVICE, writable=False)
            source, _bridge = gpu.as_torch_storage(0, size)
            src = torch.empty(0, dtype=torch.uint8, device=f"cuda:{_DEVICE}").set_(
                source
            )

            def copy(host: torch.Tensor) -> None:
                _register(host)
                try:
                    host.copy_(src)
                    torch.cuda.synchronize(_DEVICE)
                finally:
                    torch.cuda.cudart().cudaHostUnregister(host.data_ptr())

            out.append(sealed_memfd(size, copy, preallocate=True))
            del src, source, _bridge, gpu
    except BaseException:
        store._close(out)
        raise
    return out


def to_gpu(fds: Sequence[int], sizes: Sequence[int]) -> List[int]:
    """One exported VMM allocation holding each memfd's bytes."""
    # Allocate everything first: a GPU short of memory fails before any copy.
    gpus = [VmmAllocation.create(size, _DEVICE) for size in sizes]
    for fd, size, gpu in zip(fds, sizes, gpus):
        host = CpuAllocation.from_fd(fd, size, _DEVICE, cuda_access_required=True)
        source, _ = host.as_torch_storage(0, size)
        target, _bridge = gpu.as_torch_storage(0, size)
        torch.empty(0, dtype=torch.uint8, device=f"cuda:{_DEVICE}").set_(target).copy_(
            torch.empty(0, dtype=torch.uint8).set_(source)
        )
        torch.cuda.synchronize(_DEVICE)
        del source, target, _bridge, host
    out: List[int] = []
    try:
        for gpu in gpus:
            out.append(gpu.export_fd())
    except BaseException:
        store._close(out)
        raise
    return out


def main() -> None:
    # Host memory charged for a memfd does not count toward its owner's RSS, so out of memory
    # the kernel would pick another process; volunteer this one, which only loses the move.
    with open("/proc/self/oom_score_adj", "w") as f:
        f.write("1000")
    sock = socket.socket(fileno=int(os.environ[MOVER_FD_ENV]))
    job, fds = store.recv_message(sock)
    try:
        # The VMM driver calls need an initialized driver and a current context, which an
        # engine process already has from torch.
        torch.cuda.set_device(_DEVICE)
        move = to_host if job["to_host"] else to_gpu
        out = move(fds, job["sizes"])
    except BaseException as error:
        store.send_message(sock, {"error": f"{type(error).__name__}: {error}"})
        sys.exit(1)
    finally:
        store._close(fds)
    store.send_message(sock, {}, out)
    store._close(out)


if __name__ == "__main__":
    main()

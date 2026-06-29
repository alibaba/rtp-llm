"""Minimal pynccl for DSV4 CP gathers — direct ctypes ``ncclAllGather`` on a
caller stream, with NO torch c10d Work object / event sync.

Motivation (cp_comm_opt.md): DSV4 32k prefill is CPU-launch-bound — ~426 small
CP all-gathers/iter, each paying torch.distributed.all_gather_into_tensor's
Work-creation + event-sync overhead (~25-40us CPU/AG, and the event sync
inflates the kernel's effective stream time). SGLang fixes this with pynccl
(parallel_state.py::cp_all_gather_into_tensor_async). Microbench confirmed
pynccl cuts CPU-issue ~3x (11us vs 30-44us) and small-AG stream time ~3x
(18us vs 57us).

Correctness: this is a plain all_gather over a SECOND NCCL communicator built on
the SAME ranks as the given torch ProcessGroup → bit-identical, deterministic.
Cross-stream ordering is the caller's responsibility (record an event on the
gather stream after this call and wait it on the consumer stream — exactly what
CudaAsyncCPGatherImpl already does). Gated by DSV4_CP_PYNCCL=1.

Every DSV4 CP gather site goes through the dispatcher at the bottom
(:func:`cp_all_gather` / :func:`cp_all_gather_sync`) so the torch-vs-pynccl
choice lives in ONE place. NCCL symmetric-memory is a separate, additive backend
that plugs into the same two functions (kept on its own change so it can land
independently); the ``symm_variable`` argument and the ``symm_*`` return slot of
:func:`_select_cp_backend` are the forward-compatible seam for it.
"""
from __future__ import annotations

import ctypes
import glob
import os
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist

_ncclComm_t = ctypes.c_void_p
_cudaStream_t = ctypes.c_void_p
_buffer_t = ctypes.c_void_p
_ncclDataType_t = ctypes.c_int
_ncclResult_t = ctypes.c_int


class _ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


# torch dtype -> NCCL datatype enum. NCCL has no fp8 type, but all_gather is a
# pure byte move — fp8 (1 byte/elem) maps to ncclUint8 with the same element
# count → identical bytes gathered. (indexer-K quant is float8_e4m3fn.)
_NCCL_DT = {
    torch.int8: 0,
    torch.uint8: 1,
    torch.int32: 2,
    torch.int64: 4,
    torch.float16: 6,
    torch.float32: 7,
    torch.float64: 8,
    torch.bfloat16: 9,
}
for _fp8_name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
    _fp8 = getattr(torch, _fp8_name, None)
    if _fp8 is not None:
        _NCCL_DT[_fp8] = 1  # ncclUint8 (1-byte move)


class _NCCLLib:
    def __init__(self) -> None:
        cands = glob.glob(
            os.path.join(os.path.dirname(torch.__file__), "lib", "libnccl.so*")
        )
        self._lib_path = cands[0] if cands else "libnccl.so.2"
        self._lib = ctypes.CDLL(self._lib_path)
        self._lib.ncclGetErrorString.restype = ctypes.c_char_p
        self._lib.ncclGetErrorString.argtypes = [_ncclResult_t]
        self._lib.ncclGetUniqueId.restype = _ncclResult_t
        self._lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(_ncclUniqueId)]
        self._lib.ncclCommInitRank.restype = _ncclResult_t
        self._lib.ncclCommInitRank.argtypes = [
            ctypes.POINTER(_ncclComm_t),
            ctypes.c_int,
            _ncclUniqueId,
            ctypes.c_int,
        ]
        self._lib.ncclAllGather.restype = _ncclResult_t
        self._lib.ncclAllGather.argtypes = [
            _buffer_t,
            _buffer_t,
            ctypes.c_size_t,
            _ncclDataType_t,
            _ncclComm_t,
            _cudaStream_t,
        ]

    def _chk(self, r: int) -> None:
        if r != 0:
            raise RuntimeError(
                "pynccl: " + self._lib.ncclGetErrorString(r).decode()
            )

    def make_comm(self, pg, device: torch.device) -> _ncclComm_t:
        world = dist.get_world_size(pg)
        rank = dist.get_rank(pg)
        src = dist.get_global_rank(pg, 0)
        uid = _ncclUniqueId()
        cpu = torch.empty(128, dtype=torch.uint8)
        if rank == 0:
            self._chk(self._lib.ncclGetUniqueId(ctypes.byref(uid)))
            ctypes.memmove(cpu.data_ptr(), ctypes.addressof(uid), 128)
        g = cpu.to(device)
        dist.broadcast(g, src=src, group=pg)
        cpu = g.cpu().contiguous()
        ctypes.memmove(ctypes.addressof(uid), cpu.data_ptr(), 128)
        comm = _ncclComm_t()
        self._chk(self._lib.ncclCommInitRank(ctypes.byref(comm), world, uid, rank))
        return comm

    def all_gather(self, send_ptr, recv_ptr, count, dtype, comm, stream_ptr) -> None:
        self._chk(
            self._lib.ncclAllGather(
                _buffer_t(send_ptr),
                _buffer_t(recv_ptr),
                count,
                _NCCL_DT[dtype],
                comm,
                _cudaStream_t(stream_ptr),
            )
        )


_LIB: "_NCCLLib | None" = None
_COMMS: Dict[Any, Tuple["_NCCLLib", Any]] = {}


def pynccl_enabled() -> bool:
    return os.environ.get("DSV4_CP_PYNCCL", "0") == "1"


def _get_comm(process_group, device: torch.device):
    """Lazily build (once) a pynccl communicator over ``process_group``'s ranks.
    Collective (ncclCommInitRank) — must be reached rank-uniformly; the CP gather
    predicates already guarantee that."""
    global _LIB
    key = process_group
    cached = _COMMS.get(key)
    if cached is not None:
        return cached
    if _LIB is None:
        _LIB = _NCCLLib()
    comm = _LIB.make_comm(process_group, device)
    _COMMS[key] = (_LIB, comm)
    return _COMMS[key]


def pynccl_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor,
                                  stream, process_group) -> None:
    """``output[world*n] = all_gather(input[n])`` via direct ncclAllGather on
    ``stream`` (no Work/event). Caller fences cross-stream via its own event."""
    lib, comm = _get_comm(process_group, input.device)
    lib.all_gather(
        input.data_ptr(),
        output.data_ptr(),
        input.numel(),
        input.dtype,
        comm,
        stream.cuda_stream,
    )


# ---- Unified CP all-gather dispatcher -----------------------------------------
# Every DSV4 CP gather site funnels through these helpers so the backend choice
# lives in ONE place instead of a copy-pasted if/else block at each call site.
# Today the choice is torch.distributed vs pynccl; NCCL symmetric-memory is an
# additive backend that plugs into the same two functions (landed separately).
# Callers supply the input, a plain-output allocator, and the stream / process
# group / role; ``symm_variable`` is accepted for forward compatibility with the
# symmetric path and ignored on this (pynccl/torch) path.


def _select_cp_backend(role, rows, cols, dtype, device, process_group,
                       symm_variable):
    """Pick the CP all-gather backend. Returns ``(symm_out, symm_ok, use_pynccl)``
    — ``symm_out``/``symm_ok`` are the seam for the additive symmetric backend
    (always ``None`` / ``False`` on this path). Warms the pynccl comm (one-time
    ncclCommInitRank) OUTSIDE any gather stream — its broadcast/commInitRank must
    not be tangled into the gather stream."""
    use_pynccl = pynccl_enabled()
    if use_pynccl:
        _get_comm(process_group, device)
    return None, False, use_pynccl


def _launch_cp_all_gather(gathered, input, symm_ok, use_pynccl, stream,
                          process_group, profile_name, async_op):
    """Issue the gather on ``stream`` under a ``{profile_name}.launch[.backend]``
    range. Returns the torch Work (async torch path only) or None (pynccl
    stream-ordered path, or sync torch)."""
    from rtp_llm.models_py.modules.dsv4._profiler import record_function_range

    if use_pynccl:
        with record_function_range(f"{profile_name}.launch.pynccl"):
            pynccl_all_gather_into_tensor(gathered, input, stream, process_group)
        return None
    with record_function_range(f"{profile_name}.launch"):
        return torch.distributed.all_gather_into_tensor(
            gathered, input, group=process_group, async_op=async_op
        )


def cp_all_gather(input: torch.Tensor, alloc_plain, *, role: str, rows: int,
                  cols: int, process_group, gather_stream, profile_name: str,
                  symm_variable: bool = False):
    """Async CP all-gather. Picks the backend (pynccl when DSV4_CP_PYNCCL=1, else
    torch.distributed), allocates the output via ``alloc_plain(rows, cols,
    dtype)``, launches on ``gather_stream`` and records a completion event.
    Returns ``(gathered, work, completion_event)``; ``work`` is None on the
    pynccl path (stream-ordered — the caller fences via ``completion_event``,
    exactly as the torch path's event). The caller owns the wait_stream /
    record_stream edges around the input."""
    device = input.device
    dtype = input.dtype
    symm_out, symm_ok, use_pynccl = _select_cp_backend(
        role, rows, cols, dtype, device, process_group, symm_variable
    )
    with torch.cuda.stream(gather_stream):
        gathered = symm_out if symm_ok else alloc_plain(rows, cols, dtype)
        work = _launch_cp_all_gather(
            gathered, input, symm_ok, use_pynccl, gather_stream,
            process_group, profile_name, async_op=True,
        )
        try:
            completion_event = torch.cuda.Event()
            completion_event.record(gather_stream)
        except Exception:
            # Drain the in-flight NCCL Work before propagating; the caller never
            # sees the pending handle, so nothing else would wait it.
            if work is not None:
                work.wait()
            raise
    return gathered, work, completion_event


def cp_all_gather_sync(input: torch.Tensor, alloc_plain, torch_fallback, *,
                       role: str, rows: int, cols: int, process_group, stream,
                       profile_name: str, symm_variable: bool = False):
    """Synchronous CP all-gather (result consumed stream-ordered right after; no
    Work/event). When pynccl is active the gather writes into
    ``alloc_plain(rows, cols, dtype)`` on ``stream``; otherwise it defers to
    ``torch_fallback()`` — which keeps collective_torch.all_gather's own fast
    paths (torch symm_mem / rocm capture). Returns the gathered tensor."""
    device = input.device
    dtype = input.dtype
    symm_out, symm_ok, use_pynccl = _select_cp_backend(
        role, rows, cols, dtype, device, process_group, symm_variable
    )
    if not use_pynccl:
        return torch_fallback()
    with torch.cuda.stream(stream):
        gathered = symm_out if symm_ok else alloc_plain(rows, cols, dtype)
        _launch_cp_all_gather(
            gathered, input, symm_ok, use_pynccl, stream,
            process_group, profile_name, async_op=False,
        )
    return gathered

"""Minimal pynccl for DSV4 CP gathers — direct ctypes ``ncclAllGather`` on a
caller stream, with NO torch c10d Work object / event sync.

Motivation (cp_comm_opt.md): DSV4 32k prefill is CPU-launch-bound — ~426 small
CP all-gathers/iter, each paying torch.distributed.all_gather_into_tensor's
Work-creation + event-sync overhead (~25-40us CPU/AG, and the event sync
inflates the kernel's effective stream time). SGLang fixes this with pynccl
(parallel_state.py::cp_all_gather_into_tensor_async). Microbench (scripts/
microbench_pynccl.py) confirmed pynccl cuts CPU-issue ~3x (11us vs 30-44us) and
small-AG stream time ~3x (18us vs 57us).

Correctness: this is a plain all_gather over a SECOND NCCL communicator built on
the SAME ranks as the given torch ProcessGroup → bit-identical, deterministic.
Cross-stream ordering is the caller's responsibility (record an event on the
gather stream after this call and wait it on the consumer stream — exactly what
CudaAsyncCPGatherImpl already does). Gated by DSV4_CP_PYNCCL=1.
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
_ncclWindow_t = ctypes.c_void_p
_NCCL_WIN_COLL_SYMMETRIC = 1  # ncclWindowFlags


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
        # Symmetric-memory window registration (NCCL >= 2.27). Registering a
        # buffer as a SYMMETRIC window lets NCCL pick its low-latency symmetric
        # kernels — microbench (cp_comm_opt.md §8/§10): small all-gather 2-2.5x
        # faster, bit-exact. Only the OUTPUT buffer needs registration.
        self._has_symm = hasattr(self._lib, "ncclCommWindowRegister")
        if self._has_symm:
            self._lib.ncclCommWindowRegister.restype = _ncclResult_t
            self._lib.ncclCommWindowRegister.argtypes = [
                _ncclComm_t,
                _buffer_t,
                ctypes.c_size_t,
                ctypes.POINTER(_ncclWindow_t),
                ctypes.c_int,
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

    def window_register(self, comm, ptr, nbytes) -> "Any":
        """Register ``[ptr, ptr+nbytes)`` as a SYMMETRIC window on ``comm`` so
        NCCL uses its low-latency symmetric kernels. Returns the opaque window
        handle (kept alive by the caller's cache)."""
        win = _ncclWindow_t()
        self._chk(
            self._lib.ncclCommWindowRegister(
                comm, _buffer_t(ptr), nbytes, ctypes.byref(win),
                _NCCL_WIN_COLL_SYMMETRIC,
            )
        )
        return win


_LIB: "_NCCLLib | None" = None
_COMMS: Dict[Any, Tuple["_NCCLLib", Any]] = {}


def pynccl_enabled() -> bool:
    return os.environ.get("DSV4_CP_PYNCCL", "0") == "1"


def _init_comm(process_group, device: torch.device):
    """Build the pynccl communicator for ``process_group`` during startup.

    Collective (ncclCommInitRank) — must be reached rank-uniformly. Hot CP gather
    paths intentionally do not create this communicator; they only use the
    startup-created instance.
    """
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


def _require_comm(process_group):
    cached = _COMMS.get(process_group)
    if cached is None:
        raise RuntimeError(
            "pynccl CP communicator is not initialized; call pynccl_cp.init() "
            "during distributed startup before using DSV4_CP_PYNCCL/DSV4_CP_SYMM"
        )
    return cached


def _comm_ready(process_group) -> bool:
    return process_group in _COMMS


def pynccl_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor,
                                  stream, process_group) -> None:
    """``output[world*n] = all_gather(input[n])`` via direct ncclAllGather on
    ``stream`` (no Work/event). Caller fences cross-stream via its own event."""
    lib, comm = _require_comm(process_group)
    lib.all_gather(
        input.data_ptr(),
        output.data_ptr(),
        input.numel(),
        input.dtype,
        comm,
        stream.cuda_stream,
    )


# ---- NCCL symmetric-memory all-gather (DSV4_CP_SYMM) ---------------------------
# Recipe (cp_comm_opt.md §10, microbench-proven 2-2.5x on small AG, bit-exact):
#   * gather OUTPUT buffer allocated from a MemPool(backend.mem_allocator) so it
#     is ncclMemAlloc-backed (identical VA across ranks) — torch tensor;
#   * one fixed-capacity buffer per (role, dtype) is registered SYMMETRIC on the
#     pynccl comm during pynccl init. The capacity matches torch symmetric-
#     memory's max bytes for this (SM major, world_size), so oversized gathers
#     fall back to ordinary pynccl instead of allocating/registering a new
#     symmetric window mid-forward;
#   * raw pynccl ncclAllGather (torch's own all_gather does NOT route to the
#     symmetric kernel — must use raw ncclAllGather). Input needs NO registration.
_SYMM_POOL: Any = None
_SYMM_POOL_FAILED = False
_SYMM_WINDOWS: Dict[int, Any] = {}  # data_ptr -> window handle (keeps it alive)
_SYMM_INIT_BUDGET_NBYTES = 1024 * 1024 * 1024
_SYMM_ALLOWED_ROLES = frozenset((
    "main",
    "indexer",
    "state_read",
    "full_sync",
    "varlen",
))
_SYMM_INIT_PAIRS = (
    "main:bfloat16",
    "main:float32",
    "indexer:bfloat16",
    "indexer:float32",
    "state_read:float32",
    "full_sync:bfloat16",
    "varlen:bfloat16",
)


def symm_enabled() -> bool:
    return os.environ.get("DSV4_CP_SYMM", "0") == "1"


def symm_role_allowed(role: str) -> bool:
    return role in _SYMM_ALLOWED_ROLES


def _dtype_from_name(name: str) -> "torch.dtype | None":
    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float": "float32",
        "u8": "uint8",
        "fp8": "float8_e4m3fn",
        "fp8_e4m3": "float8_e4m3fn",
    }
    attr = aliases.get(name, name)
    return getattr(torch, attr, None)


def _symm_init_pairs() -> Tuple[Tuple[str, torch.dtype], ...]:
    pairs = []
    seen = set()
    for pair_name in _SYMM_INIT_PAIRS:
        if ":" not in pair_name:
            continue
        role, dtype_name = (part.strip() for part in pair_name.split(":", 1))
        dtype = _dtype_from_name(dtype_name)
        key = (role, dtype)
        if not role or dtype is None or key in seen or not symm_role_allowed(role):
            continue
        seen.add(key)
        pairs.append(key)
    return tuple(pairs)


def _symm_pool(process_group, device: torch.device, *, create: bool = False):
    """Build/read a CUDA MemPool backed by NCCL's symmetric-capable
    allocator (ncclMemAlloc). Tensors allocated inside it can be registered as
    symmetric windows. Returns None if unavailable (caller falls back)."""
    global _SYMM_POOL, _SYMM_POOL_FAILED
    if _SYMM_POOL is not None:
        return _SYMM_POOL
    if _SYMM_POOL_FAILED:
        return None
    if not create:
        return None
    try:
        backend = process_group._get_backend(device)
        _SYMM_POOL = torch.cuda.MemPool(backend.mem_allocator)
    except Exception:
        _SYMM_POOL_FAILED = True
        _SYMM_POOL = None
    return _SYMM_POOL


def symm_available(process_group, device: torch.device) -> bool:
    """True iff symmetric all-gather can be used (gate on + pool + comm support)."""
    if not symm_enabled():
        return False
    if _symm_pool(process_group, device) is None:
        return False
    if not _comm_ready(process_group):
        return False
    lib, _ = _require_comm(process_group)
    return bool(getattr(lib, "_has_symm", False))


def symm_empty(rows: int, cols: int, dtype: torch.dtype, device: torch.device,
               process_group) -> "torch.Tensor | None":
    """Allocate a ``[rows, cols]`` gather-output tensor in the symmetric MemPool.
    Returns None if the pool is unavailable (caller falls back to torch.empty)."""
    pool = _symm_pool(process_group, device)
    if pool is None:
        return None
    with torch.cuda.use_mem_pool(pool):
        return torch.empty((rows, cols), dtype=dtype, device=device)


# Persistent symmetric gather buffers. CRITICAL: window registration
# (ncclCommWindowRegister) is a COLLECTIVE call, so we keep it out of the hot
# shape-changing path. The ring backend allocates one torch-sized capacity buffer
# per (role, dtype), registers that whole window during pynccl init, and returns
# prefix views for live rows. Reuse follows the same role-ordered contract
# as the per-forward workspace union (consumed before the next same-role gather
# via stream ordering).
_SYMM_PERSIST: Dict[Any, torch.Tensor] = {}
_SYMM_CAPACITY_NBYTES: Dict[int, int] = {}  # data_ptr -> registered capacity


def _dtype_element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _symm_required_nbytes(rows: int, cols: int, dtype: torch.dtype) -> int:
    return int(rows) * int(cols) * _dtype_element_size(dtype)


def _torch_symm_mem_max_nbytes(process_group, device: torch.device) -> "int | None":
    """Return torch symmetric-memory's byte cap for this topology.

    Keep pynccl ring-symm aligned with ``TorchSymmMemCommunicator``: the same
    capability/world-size pairs are eligible, and the same max bytes decide
    whether a gather can use symmetric memory.
    """
    try:
        from rtp_llm.models_py.distributed.symm_mem import (
            TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES,
        )

        cc_major = torch.cuda.get_device_capability(device)[0]
        world_size = dist.get_world_size(process_group)
        return TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES.get(cc_major, {}).get(world_size)
    except Exception:
        return None


def _fits_torch_symm_mem_limit(
    rows: int, cols: int, dtype: torch.dtype, max_nbytes: "int | None"
) -> bool:
    return max_nbytes is not None and _symm_required_nbytes(rows, cols, dtype) <= max_nbytes


def _round_symm_rows(rows: int) -> int:
    """Capacity rows for RING-symm persistent buffers.

    Prefix-cache warmup and measure can use nearby-but-different row counts
    (e.g. prefix-only warmup, then 32k measure with block-rounded reuse).  The
    NCCL symmetric-window path is only microbench-like when the output window is
    already allocated+registered, so round rows up by default and return a view
    for the live rows.
    """
    rows = int(rows)
    if rows <= 1:
        return rows
    return 1 << (rows - 1).bit_length()


def _ring_symm_base(
    role: str,
    dtype: torch.dtype,
    device: torch.device,
    process_group,
    max_nbytes: int,
    *,
    create: bool = False,
) -> "torch.Tensor | None":
    cap_numel = int(max_nbytes) // _dtype_element_size(dtype)
    cap_key = ("ring", role, dtype, device.index, int(max_nbytes))
    base_flat = _SYMM_PERSIST.get(cap_key)
    if base_flat is not None:
        return base_flat
    if not create:
        return None
    pool = _symm_pool(process_group, device)
    if pool is None:
        return None
    with torch.cuda.use_mem_pool(pool):
        base_flat = torch.empty(cap_numel, dtype=dtype, device=device)
    _SYMM_PERSIST[cap_key] = base_flat
    _SYMM_CAPACITY_NBYTES[base_flat.data_ptr()] = int(max_nbytes)
    lib, comm = _require_comm(process_group)
    _ensure_registered(lib, comm, base_flat)
    return base_flat


def _init_symm_windows(process_group, device: torch.device) -> None:
    """Allocate and register pynccl ring-symm windows during startup."""
    if not symm_enabled():
        return
    if _symm_pool(process_group, device, create=True) is None:
        return
    if not symm_available(process_group, device):
        return
    max_nbytes = _torch_symm_mem_max_nbytes(process_group, device)
    if max_nbytes is None:
        return
    used_nbytes = 0
    for role, dtype in _symm_init_pairs():
        if not symm_role_allowed(role):
            continue
        if used_nbytes + int(max_nbytes) > _SYMM_INIT_BUDGET_NBYTES:
            break
        buf = _ring_symm_base(
            role, dtype, device, process_group, int(max_nbytes), create=True
        )
        if buf is not None:
            used_nbytes += int(max_nbytes)


def symm_persistent(role: str, rows: int, cols: int, dtype: torch.dtype,
                    device: torch.device, process_group,
                    profile_name: str = "") -> "torch.Tensor | None":
    """Return a reused ``[rows, cols]`` symmetric gather-output buffer for
    ``role``. None if the symmetric pool is unavailable or the requested output
    is larger than torch symmetric-memory's max byte cap.

    The RING-symmetric path allocates one max-capacity MemPool buffer per
    (role, dtype), window-registers the full capacity during pynccl init, and
    returns a shaped prefix view for the live bytes."""
    del profile_name
    rows = int(rows)
    cols = int(cols)
    max_nbytes = _torch_symm_mem_max_nbytes(process_group, device)
    if not _fits_torch_symm_mem_limit(rows, cols, dtype, max_nbytes):
        return None

    assert max_nbytes is not None
    exact_key = (role, rows, cols, dtype, device.index, int(max_nbytes))
    buf = _SYMM_PERSIST.get(exact_key)
    if buf is not None:
        return buf

    base_flat = _ring_symm_base(role, dtype, device, process_group, int(max_nbytes))
    if base_flat is None:
        return None
    view = base_flat[: rows * cols].view(rows, cols)
    _SYMM_PERSIST[exact_key] = view
    return view


def _ensure_registered(lib, comm, output: torch.Tensor) -> None:
    ptr = output.data_ptr()
    if ptr in _SYMM_WINDOWS:
        return
    nbytes = _SYMM_CAPACITY_NBYTES.get(
        ptr, output.numel() * output.element_size()
    )
    win = lib.window_register(comm, ptr, nbytes)
    _SYMM_WINDOWS[ptr] = win


def pynccl_symm_all_gather(output: torch.Tensor, input: torch.Tensor,
                           stream, process_group) -> None:
    """Symmetric all-gather into ``output`` (from :func:`symm_persistent`),
    ``input`` may be any tensor. Window-register ``output`` (cached) and issue
    raw ncclAllGather on the pynccl comm."""
    lib, comm = _require_comm(process_group)
    _ensure_registered(lib, comm, output)
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
# (symm > pynccl > torch.distributed) lives in ONE place instead of a copy-pasted
# use_symm/use_pynccl/if-elif-else block at each call site. The env-var gates and
# the symm buffer/comm machinery stay encapsulated here; callers only supply the
# input, a plain-output allocator, and the stream/process_group/role. The symm
# branch is the seam a future symmetric-memory rework plugs into without touching
# any call site.


def _select_cp_backend(role, rows, cols, dtype, device, process_group,
                       symm_variable, profile_name):
    """Pick the CP all-gather backend and (for the symm path) its persistent,
    window-registered output. Returns ``(symm_out_or_None, symm_ok, use_pynccl)``
    and never creates the pynccl comm/window on the hot path. ``symm_variable``
    is retained as a callsite annotation only; under DSV4_CP_SYMM only
    ``symm_role_allowed(role)`` attempts symm, and all other roles fall back to
    ordinary pynccl."""
    del symm_variable
    pynccl_requested = pynccl_enabled() or symm_enabled()
    if pynccl_requested and not _comm_ready(process_group):
        raise RuntimeError(
            "pynccl CP backend was requested but not initialized at startup"
        )

    symm_requested = symm_enabled() and symm_role_allowed(role)
    symm_ok = symm_requested and symm_available(process_group, device)
    symm_out = None
    if symm_ok:
        symm_out = symm_persistent(
            role, rows, cols, dtype, device, process_group, profile_name
        )
        symm_ok = symm_out is not None
    use_pynccl = pynccl_requested
    return symm_out, symm_ok, use_pynccl


def _launch_cp_all_gather(gathered, input, symm_ok, use_pynccl, stream,
                          process_group, profile_name, async_op):
    """Issue the gather on ``stream`` under a ``{profile_name}.launch[.backend]``
    range. Returns the torch Work (async torch path only) or None (pynccl/symm
    stream-ordered paths, or sync torch)."""
    from rtp_llm.models_py.modules.dsv4._profiler import record_function_range

    if symm_ok:
        with record_function_range(f"{profile_name}.launch.symm"):
            pynccl_symm_all_gather(gathered, input, stream, process_group)
        return None
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
    """Async CP all-gather. Picks symm/pynccl/torch, allocates the output (the
    persistent symm buffer, else ``alloc_plain(rows, cols, dtype)``), launches on
    ``gather_stream`` and records a completion event. Returns
    ``(gathered, work, completion_event)``; ``work`` is None on the pynccl/symm
    paths (stream-ordered — the caller fences via ``completion_event``, exactly
    as the torch path's event). The caller owns the wait_stream/record_stream
    edges around the input."""
    device = input.device
    dtype = input.dtype
    symm_out, symm_ok, use_pynccl = _select_cp_backend(
        role, rows, cols, dtype, device, process_group, symm_variable,
        profile_name,
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
    Work/event). When symm/pynccl is active the gather writes into the symm
    buffer / ``alloc_plain(rows, cols, dtype)`` on ``stream``; otherwise it
    defers to ``torch_fallback()`` — which keeps collective_torch.all_gather's own
    fast paths (torch symm_mem / rocm capture). Returns the gathered tensor."""
    device = input.device
    dtype = input.dtype
    symm_out, symm_ok, use_pynccl = _select_cp_backend(
        role, rows, cols, dtype, device, process_group, symm_variable,
        profile_name,
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


def init(process_group, device: torch.device) -> None:
    """Build all optional pynccl CP resources during distributed startup.

    This is intentionally not lazy: hot CP gather paths only consume the
    communicator and symmetric windows created here. Collective init must be
    reached rank-uniformly on every rank of ``process_group``. No-op unless
    DSV4_CP_PYNCCL / DSV4_CP_SYMM is set.
    """
    if not (pynccl_enabled() or symm_enabled()):
        return
    _init_comm(process_group, device)
    _init_symm_windows(process_group, device)

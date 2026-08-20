"""Small adapter around the two unstable NCCL/PyTorch integration seams.

The sleep/wake policy lives in :mod:`rtp_llm.utils.nccl_memory`. This module
keeps the implementation details that we will eventually be able to delete:

* loading the optional NCCL 2.29.7+ symbols and declaring their ABI; and
* extracting ``ncclComm_t`` pointers from PyTorch's private ProcessGroup API.

:func:`api` is the only entry point the policy uses, so the raw ``ctypes``
handle, the NCCL symbol names and the version pin never leave this file -- which
is the whole point: when PyTorch exposes public ``ProcessGroupNCCL``
suspend/resume methods, this module is what gets rewritten and the policy is
what stays. In particular, do not put collective voting or suspended-state
bookkeeping here: those are correctness policy, not an ABI concern.

TODO(nccl-memory): when that public API lands, implement :class:`NcclApi` on top
of it and delete the ctypes path; the policy layer should not need to change.
"""

import ctypes
import logging
import threading
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch

# The three symbols the feature needs, all added in NCCL 2.29.7. Probing for the
# symbols rather than parsing the version is deliberate: it is the property that
# actually matters, it stays correct if a vendor backports, and it makes this
# module inert -- not broken -- on an older runtime.
_REQUIRED_SYMBOLS = (
    "ncclCommSuspend",
    "ncclCommResume",
    "ncclCommMemStats",
)
_MIN_VERSION = "2.29.7"

_lock = threading.RLock()
_api: Optional["NcclApi"] = None


def _configure_abi(lib: ctypes.CDLL) -> None:
    """Declare the C ABI once for symbols present in this runtime.

    Verified against ``nccl.h`` 2.30.4 (lines 318/326/345). Nothing is passed by
    value, so a future signature change surfaces as a ``ctypes.ArgumentError``
    rather than as silent stack corruption -- provided the declarations here stay
    in step with the header.

    The per-symbol guard is intentional: old NCCL runtimes do not export the
    offload API and must remain a clean, runtime-detectable no-op.
    """

    for name, argtypes in (
        ("ncclCommSuspend", [ctypes.c_void_p, ctypes.c_int]),
        ("ncclCommResume", [ctypes.c_void_p]),
        (
            "ncclCommMemStats",
            [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)],
        ),
    ):
        fn = getattr(lib, name, None)
        if fn is not None:
            fn.argtypes = argtypes
            fn.restype = ctypes.c_int


def _load_nccl() -> Tuple[Optional[ctypes.CDLL], str]:
    """Return the process' NCCL handle and a human-readable version string.

    ``dlopen`` by SONAME rather than by path, and that is what makes it correct:
    glibc records ``DT_SONAME`` as an alias for the already-mapped object, so in
    a torch process (which maps ``libnccl.so.2`` eagerly via
    ``libtorch_cuda.so``'s ``DT_NEEDED``) this returns *torch's own* handle even
    when ``LD_LIBRARY_PATH`` points at a different copy. Loading a second copy
    would be worse than useless: it would keep its own communicator registry and
    we would suspend comms this process is not using.

    On failure the reason is returned in the version slot; the caller reports it
    verbatim, so a bad image is diagnosable from the sleep log alone.
    """

    try:
        # CDLL (rather than PyDLL) releases the GIL around suspend/resume. Those
        # calls can spend seconds in NCCL bootstrap barriers; holding the GIL
        # would stall the HTTP/control-plane thread for no reason.
        lib = ctypes.CDLL("libnccl.so.2")
    except OSError as exc:  # pragma: no cover - runtime-image dependent
        return None, f"libnccl.so.2 not loadable: {exc}"

    try:
        ver = ctypes.c_int()
        lib.ncclGetVersion(ctypes.byref(ver))
        value = ver.value
        version = f"{value // 10000}.{(value % 10000) // 100}.{value % 100}"
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        version = f"version query failed: {exc}"

    _configure_abi(lib)
    return lib, version


def api() -> "NcclApi":
    """The process-wide NCCL memory-offload façade, loading the library once."""

    global _api
    with _lock:
        if _api is None:
            lib, version = _load_nccl()
            _api = NcclApi(lib, version)
        return _api


class NcclApi:
    """Typed-ish façade for the optional NCCL memory-offload ABI.

    ``library`` is deliberately an object rather than a strict ``ctypes.CDLL`` so
    CPU-only tests can provide a tiny fake and so a future native Torch op can
    implement the same façade without changing the sleep policy.

    :meth:`stat` swallows its own errors because it is a probe; :meth:`suspend`
    and :meth:`resume` deliberately do not, because their caller has to treat a
    failure as a communicator-level fault rather than as a missing number.
    """

    def __init__(self, library: Optional[object], version: str = "unknown") -> None:
        self.library = library
        self.version = version

    @property
    def missing_symbols(self) -> List[str]:
        if self.library is None:
            return list(_REQUIRED_SYMBOLS)
        return [name for name in _REQUIRED_SYMBOLS if not hasattr(self.library, name)]

    @property
    def usable(self) -> bool:
        return self.library is not None and not self.missing_symbols

    @property
    def unavailable_reason(self) -> str:
        """Why this runtime cannot release communicator memory, or ``""``."""

        if self.library is None:
            # _load_nccl puts the dlopen failure in the version slot.
            return self.version
        missing = self.missing_symbols
        if missing:
            return (
                f"runtime NCCL {self.version} lacks {','.join(missing)} "
                f"(needs >= {_MIN_VERSION}); collective memory release unavailable"
            )
        return ""

    def stat(self, comm: int, stat: int) -> int:
        """One ``ncclCommMemStats`` field, or -1. Local, non-collective, no raise.

        -1 covers "symbol absent", "call failed" and "non-zero rc" alike: every
        caller treats all three the same way (abstain before the vote, raise on
        the wake path), so distinguishing them would add a branch nobody wants.
        """

        fn = getattr(self.library, "ncclCommMemStats", None)
        if fn is None:
            return -1
        value = ctypes.c_uint64()
        try:
            rc = int(fn(ctypes.c_void_p(comm), ctypes.c_int(stat), ctypes.byref(value)))
        except Exception:  # noqa: BLE001 - a diagnostic probe must not raise
            return -1
        return value.value if rc == 0 else -1

    def suspend(self, comm: int, flags: int) -> int:
        fn = getattr(self.library, "ncclCommSuspend", None)
        if fn is None:
            return -1
        return int(fn(ctypes.c_void_p(comm), ctypes.c_int(flags)))

    def resume(self, comm: int) -> int:
        fn = getattr(self.library, "ncclCommResume", None)
        if fn is None:
            return -1
        return int(fn(ctypes.c_void_p(comm)))


def _is_nccl_backend(backend: object) -> bool:
    """Whether this ``Backend`` is the NCCL one -- i.e. whether it should own a comm.

    Asked by class name rather than ``isinstance(_, ProcessGroupNCCL)`` so the
    question stays answerable in a CPU-only unit test with a fake backend, and so
    a build without NCCL answers False instead of tripping over a missing symbol.
    Only used to decide whether a *missing* accessor is worth an ERROR, so a wrong
    answer costs log volume, never correctness.
    """

    return "nccl" in type(backend).__name__.lower()


def comm_ptr_for_group(pg: object, device: object = None) -> int:
    """The ``ncclComm_t`` this ProcessGroup currently holds, or 0.

    Split out of :func:`enumerate_process_group_comms` so the sleep policy can
    re-read *one specific* group's pointer on the wake path without re-running the
    whole scan. That difference matters: the scan can come back short for reasons
    that have nothing to do with the group being asked about (a gloo sibling, a
    renamed ``_group_map``), and the wake path must not read "the scan found less
    than I remember" as "my communicator was destroyed".

    0 means "no entry for the current device" and is not an error here -- see the
    device note on :func:`enumerate_process_group_comms`. Exceptions are left to
    propagate: the caller decides what an unanswerable question is worth, and on
    the wake path the answer is "log it and trust the recorded pointer", because
    the caller holds a reference that keeps the communicator alive.
    """

    ctx = torch.cuda.device(device) if device is not None else nullcontext()
    with ctx:
        return int(pg._get_backend(torch.device("cuda"))._comm_ptr() or 0)


def enumerate_process_group_comms(
    device: object = None,
) -> List[Tuple[str, int, object]]:
    """Enumerate distinct NCCL communicators registered by rtp-llm.

    This is intentionally the only place that knows about PyTorch's private
    ``_group_map``, ``_get_backend`` and ``_comm_ptr``. Pointer de-duplication
    prevents one ProcessGroup registered under several rtp-llm keys from being
    suspended twice, which is ``ncclInvalidUsage``.

    The owning ProcessGroup is returned alongside its pointer so a caller that
    intends to hold the communicator across a sleep can hold the object that owns
    it, rather than holding a bare integer and later trying to work out whether it
    still means anything.

    ``device`` is not cosmetic. ``ProcessGroupNCCL::getCommPtr()`` looks the
    communicator up in ``devNCCLCommMap_`` keyed by
    ``c10::cuda::current_device()`` -- not by the group's own device -- and
    returns 0, silently and without throwing, when the key is absent. A hook
    thread whose current device was never set would therefore enumerate zero
    communicators on every rank but rank 0: not a crash, a silent asymmetry that
    vetoes the feature for the life of the process. Pinning the device makes the
    lookup match the rank that owns the comm. It cannot lazily create one -- the
    call is a mutex-guarded map lookup with no NCCL bootstrap in it -- so this is
    safe to call on any rank at any time.

    Every *unexpected* way of coming back short is logged, because all of them
    degrade to the same outcome -- a fingerprint that no peer matches, so the vote
    fails closed and the feature silently saves nothing -- and a private-API rename
    after a torch upgrade must not look like "there was nothing to do". The one
    expected way, a group with no NCCL backend, is silent and is recognised by the
    backend it hands back rather than by it raising: gloo groups do not raise here.

    Order is ``_group_map`` insertion order, identical on every rank because the
    groups are created in the same sequence during init. Rule (3) in
    :mod:`rtp_llm.utils.nccl_memory` depends on that, and its ``_fingerprint``
    verifies it rather than trusting it.

    TODO(nccl-memory): replace this scan with explicit engine-owned communicator
    registration once ``collective_torch`` provides a stable hook. That would
    also cover hidden ProcessGroupNCCL P2P communicators.
    """

    from rtp_llm.models_py.distributed import collective_torch

    out: List[Tuple[str, int, object]] = []
    seen = set()
    group_map = getattr(collective_torch, "_group_map", None)
    if group_map is None:
        logging.error(
            "[NcclMemory] collective_torch has no _group_map (private API rename); "
            "enumerating ZERO communicators and disabling collective release"
        )
        group_map = {}

    ctx = torch.cuda.device(device) if device is not None else nullcontext()
    with ctx:
        for key, pg in list(group_map.items()):
            try:
                backend = pg._get_backend(torch.device("cuda"))
            except Exception:  # noqa: BLE001 - no backend registered for CUDA at all
                continue
            reader = getattr(backend, "_comm_ptr", None)
            if reader is None:
                # The expected, uninteresting case. Group.SLEEP_QUIESCE is gloo,
                # and `_get_backend(cuda)` hands back its ProcessGroupGloo instead
                # of raising -- so "has a CUDA backend" is NOT the same question as
                # "owns an ncclComm_t". Gating the complaint on the backend actually
                # being the NCCL one is what keeps this branch meaning what it says:
                # measured on DSV4 PD, treating gloo as a rename logged an ERROR on
                # every rank of every sleep while the release was in fact working.
                if _is_nccl_backend(backend):
                    logging.error(
                        "[NcclMemory] %s is an NCCL backend with no _comm_ptr "
                        "(private API rename?) -- enumeration is incomplete, so "
                        "collective release will fail closed and save nothing",
                        key,
                    )
                continue
            try:
                comm = reader()
            except Exception as e:  # noqa: BLE001
                # Has the accessor but cannot answer: a real fault, never a gloo
                # group. Sharing the silent `continue` above is how a torch upgrade
                # turns this feature into a permanent no-op, so this branch is loud.
                logging.error(
                    "[NcclMemory] %s._comm_ptr() failed (%s) -- enumeration is "
                    "incomplete, so collective release will fail closed and save "
                    "nothing",
                    key,
                    e,
                )
                continue
            if not comm:
                # getCommPtr() returns 0 when devNCCLCommMap_ has no entry for
                # the current device: either the comm is not bootstrapped yet or
                # this thread's device is wrong. Both make the list incomplete.
                logging.warning(
                    "[NcclMemory] %s reports a null communicator (not yet "
                    "bootstrapped, or this thread's CUDA device does not own it) "
                    "-- excluded from the collective release",
                    key,
                )
                continue
            if comm in seen:
                continue
            seen.add(comm)
            out.append((str(key), int(comm), pg))
    return out


def reset_for_testing() -> None:
    """Drop the cached handle. Tests only; production loads once and keeps it."""

    global _api
    with _lock:
        _api = None

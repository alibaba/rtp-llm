"""Release and restore NCCL communicator GPU memory across a sleep/wake cycle.

A sleeping instance still holds a few GiB per rank that neither the weight VMM
pause nor ``empty_cache()`` can touch, because it does not belong to torch's
allocator at all. One measured slice of that residual is the NCCL communicator:
transport buffers, channel ``devPeers``, and peer imports. NCCL >= 2.29.7 can
hand them back on request:

  * ``ncclCommSuspend(comm, NCCL_SUSPEND_MEM)`` -- ``cuMemUnmap`` + ``cuMemRelease``
    while explicitly KEEPING the ``cuMemAddressReserve`` virtual range.
  * ``ncclCommResume(comm)`` -- remaps physical pages at the same virtual
    addresses and re-exchanges peer handles.

Preserving the VA is what makes this usable here at all: the decode role's
``output_proj`` all-reduce is inside a captured CUDA graph, so any mechanism that
moved the communicator's addresses would force a graph recapture -- tens of
seconds on the wake path, and the reason communicator destroy/rebuild was
rejected. Suspend/resume keeps every baked pointer valid, so wake costs a remap
instead of a recapture.

The bytes are not free: everything NCCL suspends is classified ``ncclMemOffload``
(not ``ncclMemScratch``), i.e. copied to *pinned host* memory and copied back on
resume. GPU memory is traded for host memory one-for-one, which is an accepted
trade here but is the reason this is opt-in rather than always-on. NVLS and
NET/RDMA buffers are ``ncclMemPersist`` and are never released.

Measured on DSV4-Flash PD (decode tp1/dp2/ep2, prefill tp2/ep2, single
``DP_AND_TP`` comm, 632 MiB tracked): 576 MiB/rank released, resume 0.6-3.3 s
over 5 cycles with no upward trend, zero graph recaptures, all post-wake
inference correct.

Correctness rules, all read off ``nccl/src/mem_manager.cc``. Each is a way to
brick a communicator and none of them is intuitive, so this docstring is the ONE
place they are argued; every other site on the sleep/wake path points here rather
than restating them. The long source-level derivations behind (2) and (5) are in
the ``--sleep_release_collective_memory`` commit message.

1. **Both calls are collective, unbounded, and abort-only.** No timeout anywhere
   and only the NCCL abort flag breaks out, so a rank that skips while its peers
   call is a permanent cluster hang rather than an error. Hence both directions
   are gated on the unanimous decision in :func:`_decide`, and nothing here ever
   calls into NCCL on a hunch.

2. **A failed call is never retried, in either direction.** The return code
   cannot say which side of the barrier the failure happened on -- the codes that
   prove pre-barrier are exactly the ones a retry cannot fix -- and two of
   resume's pre-barrier sites (``cuMemSetAccess``, the host-backup ``cudaMemcpy``)
   are not idempotent, so a retry would re-map an already mapped range and leak a
   handle. A retry buys nothing and can turn "restart this instance" into a hang.

3. **The sequence of communicators must be identical on every rank**, since each
   call is a per-communicator collective. So the comm list is fingerprinted into
   the vote and a divergent enumeration fails closed (see :func:`_canonical_key`
   -- raw group keys are *not* rank-invariant, which is the trap), and a failure
   part-way through the list does NOT abandon the rest: ``bootstrapSplit`` gives
   every communicator its own ``bootstrapState``, so comm[0]'s barrier cannot
   cross-match comm[1]'s and continuing rescues the peers waiting on comm[1]
   instead of stranding them.

4. **Double suspend / resume-without-suspend are hard errors**
   (``ncclInvalidUsage``), and on a non-blocking communicator they leave a sticky
   async error that poisons every later NCCL call. So the suspended set is
   tracked here rather than inferred.

5. **"Unsupported" must be established BEFORE the vote, never mid-loop.** The
   genuinely benign refusals (memory manager disabled, ``split_share`` with
   ``refCount > 1``) return ``ncclInvalidUsage`` -- but so does *rank-local sticky*
   ``comm->asyncResult`` via ``ncclCommEnsureReady``, so reading that code as
   "refused, nothing happened, skip it" would let one rank skip while its peers
   block in the barrier forever. There is no safe skip once the vote says GO.
   Every such refusal is also reachable via ``ncclCommMemStats``, which shares
   that exact preamble and is local and non-collective, so BOTH directions probe
   every communicator before voting. Suspend then *abstains* (safe: the peers
   count ``world_size - 1`` and the vote fails closed); resume has no safe skip
   and raises without voting instead.

6. **A ``ncclSuccess`` resume is not proof of a clean resume.** The peer
   re-import loop only ``WARN``s and continues, and it runs *after*
   ``manager->released`` is cleared, so ``ncclStatGpuMemSuspended`` can read 0
   even when individual peer buffers were left unmapped. The wake hook therefore
   performs a fail-closed, rank-symmetric evidence vote over the communicators it
   actually resumed. A failed or unknown stat poisons the instance before weight
   reload; for partial peer import, operators must still inspect NCCL's
   ``restoredPeerCount`` INFO line and its "Could not find matching handle info"
   warnings after a suspicious wake.

7. **Resume must complete before ANY collective on the wake path.** Tighter than
   it looks: the level-2 weight reload broadcasts every checkpoint tensor over the
   world communicator, so resume cannot be deferred to the generic "restore python
   caches" step, which runs *after* the reload. It therefore gets its own wake
   hook, ordered first, ahead of everything else the wake does.

8. **On a non-blocking communicator both calls return ``ncclInProgress``, which
   is a SUCCESS.** They return having only started, and the caller must poll
   ``ncclCommGetAsyncError`` to completion. This module refuses that configuration
   in :func:`capability` instead of polling, which is a deliberate trade rather
   than an omission: a polling loop here would need its own timeout and abort
   semantics next to a call that has neither, and -- since nothing in this repo
   selects non-blocking communicators -- it would never execute in production, so
   it could never be verified. Untested recovery code beside an unbounded
   collective is worse than a refusal. What is kept is the diagnosis: if rc=7 ever
   does appear, it names the cause instead of reading as a bare failure.

Enabled by ``--sleep_release_collective_memory``; inert (and loud about it) when
the runtime NCCL predates the API, so the version pin can be bumped
independently.
"""

import hashlib
import logging
import os
import re
import threading
import time
from datetime import timedelta
from typing import List, Optional, Sequence, Tuple

import torch

from rtp_llm.utils import nccl_memory_backend as _backend

# TODO(nccl-memory): when PyTorch exposes a public ProcessGroupNCCL memory-offload
# API, add that implementation to nccl_memory_backend and keep this policy layer
# unchanged. Until then the backend is intentionally the only raw-ABI/private-API
# glue in this feature.
# TODO(nccl-memory): add an explicitly strict deployment policy that turns a
# pre-vote capability/topology skip into a failed sleep. The default remains
# fail-closed/no-op so enabling the optional optimization cannot make an older or
# partially initialized rank wedge its peers; strict mode needs a collective
# preflight vote before it can safely change that contract.
# TODO(nccl-memory): add a native post-GO watchdog that trips NCCL's abort flag
# for a communicator stuck inside its unbounded suspend/resume barrier. The
# current vote timeout only bounds rendezvous; once GO is unanimous, a process
# death or an internal NCCL remap hang still requires the service supervisor to
# restart the rank group.
# TODO(nccl-memory): namespace rendezvous keys with a launcher-provided job/epoch
# nonce before enabling this on elastic workers. A TCPStore that outlives a rank
# restart can retain an old arrival counter and fabricate a quorum at seq=0.
# TODO(nccl-memory): make every individual Store RPC obey the shared vote deadline
# (and fail closed when the Store cannot install a bounded timeout); today the
# default TCPStore is bounded, but a custom PrefixStore can expose a weaker limit.
# TODO(nccl-memory): add a host-pinned-memory budget preflight. NCCL's offload is
# one-for-one GPU->pinned-host, so memlock/MemAvailable exhaustion can still occur
# after the unanimous vote and before the NCCL barrier returns.

_MiB = 1024.0 * 1024.0

# nccl.h: ncclCommMemStat_t
_STAT_SUSPEND = 0  # bytes that suspend COULD release
_STAT_SUSPENDED = 1  # 0 = active, 1 = currently suspended
_STAT_PERSIST = 2  # bytes suspend will never release (NVLS, net devmem)
_STAT_TOTAL = 3
# nccl.h: the only defined flag. NCCL validates nothing -- flags=0 silently
# succeeds having done nothing -- so this constant must be right.
_SUSPEND_MEM = 0x01

# Deliberately no "benign return code" list here: see rule (5). Every non-zero
# code from suspend/resume is treated as a hard failure, and the refusals that
# really are benign are caught by the pre-vote ncclCommMemStats probe instead.
#
# ncclInProgress is the one code that is not a failure at all: on a NON-blocking
# communicator both calls return it having only *started*, and the caller is
# expected to poll ncclCommGetAsyncError to completion (mem_manager.cc:1036,1088).
# We do not poll -- see rule (8) -- so this constant exists only to turn an
# otherwise baffling "rc=7" into a diagnosis.
_NCCL_IN_PROGRESS = 7

# The two switches that select non-blocking communicators. Read, never set.
# NCCL_COMM_BLOCKING is NCCL's own and only counts when explicitly 0; torch's is
# opt-in at 1. Neither is set anywhere in this repo, which is why rule (8) can
# afford to refuse rather than implement polling.
_NONBLOCKING_ENV = (
    ("TORCH_NCCL_USE_COMM_NONBLOCKING", "1"),
    ("NCCL_COMM_BLOCKING", "0"),
)

# Bounded deadline for the go/no-go decision. Not derived from the /sleep
# request's timeout_ms: the commit request carries timeout_ms=0 (the drain
# already happened during prepare), so there is no caller-supplied budget to
# inherit. Every rank that is going to arrive has already finished its drain and
# is microseconds away, so this is a liveness bound on a peer that died, not a
# work estimate -- generous enough to absorb GIL and scheduling jitter, short
# enough that a dead peer costs one sleep instead of a hung instance.
_VOTE_TIMEOUT_S = 30.0
_VOTE_POLL_S = 0.05
_VOTE_KEY_PREFIX = "rtp_llm/sleep_nccl_release"
_DECISION_GO = "go"
_DECISION_VETO = "veto"

_lock = threading.RLock()
# Communicators this process suspended, in suspend order, as
# (group key, comm ptr, owning ProcessGroup).
#
# The ProcessGroup reference is load-bearing, not bookkeeping. A raw pointer
# recorded across a sleep is only meaningful while the object that owns it lives,
# and resuming a pointer whose communicator was destroyed and reallocated is a
# use-after-free inside NCCL. Holding the group makes that unrepresentable for the
# case that can actually happen -- the group being dropped from
# ``collective_torch._group_map`` and collected while we sleep -- because a
# communicator with a live owner is not finalized.
#
# Honest limit: it does NOT cover ncclCommAbort from torch's watchdog, which
# destroys the communicator with the group still alive. That case is caught after
# the fact by the pointer cross-check on the wake path and by resume's own rc, and
# by then the process is unserviceable anyway.
_suspended: List[Tuple[str, int, object]] = []
# Set when a communicator is left in a state we cannot reconcile. Wake refuses to
# proceed rather than resuming on top of it, and sleep refuses to suspend again.
_poisoned: Optional[str] = None
# Sequence number namespacing the decision keys. Advanced ONLY after a unanimous
# GO, which is what makes it provably identical on every rank: a GO means all
# world_size ranks incremented this sequence's arrival key, so all of them
# advance together. See :func:`_decide`.
_vote_seq = 0
# Latched by the first VETO. A veto means the ranks disagreed or one went missing
# -- an anomaly we cannot reason about, so the feature stays off for the life of
# the process rather than risking a second, worse-informed attempt. This is also
# what makes stale decision keys unexploitable: after a veto nobody votes again.
_vote_disabled: Optional[str] = None


class NcclMemoryError(RuntimeError):
    """A communicator was left in a state the caller must not paper over."""


# Why NCCL failed the current sleep or wake transition, or None when it did not.
# Recorded at the sites that raise, which is what keeps the diagnostic from
# drifting away from the failure it describes, and read back over pybind by the
# C++ sleep hook (WeightManager.nccl_memory_status) so that SleepStatus.last_error
# names the NCCL stage instead of only the hook that ran it. Deliberately just the
# failure reason: everything else worth knowing -- bytes, latency, comm count,
# runtime version -- is already in this module's INFO lines, and duplicating it
# into a second state machine only creates something that can disagree with
# _suspended/_poisoned.
_last_failure: Optional[str] = None


def _record_failure(detail: str) -> str:
    """Remember why NCCL failed this transition. Returns ``detail`` unchanged."""

    global _last_failure
    _last_failure = detail
    return detail


def status_text() -> str:
    """One line naming the NCCL failure, or ``""`` when NCCL is not the cause.

    The empty string is the load-bearing case, not a degenerate one. The C++ hook
    appends this to ``SleepStatus.last_error`` whenever the hook it is attached to
    fails, and those hooks fail for several reasons that have nothing to do with
    NCCL -- the ``cuda_graph``/``weights`` VMM pause, the level-2 weight reload,
    the python cache restore. Returning a cheerful "state=idle" blob on those
    paths would point the operator at a healthy subsystem; ``hookFailureMessage``
    drops an empty detail and keeps its own accurate message instead.

    Deliberately lock-free. ``_lock`` is held across the untimed NCCL
    suspend/resume barrier, so a diagnostic that took it could block forever on
    precisely the hang it exists to explain. A ``str``/``None`` global is
    published atomically under the GIL, which is all this needs.
    """

    detail = _last_failure
    if detail is None:
        return ""
    # One line, because it lands in a single-line gRPC status field.
    return "[NcclMemory] " + " ".join(detail.split())


def _nonblocking_comm_reason() -> str:
    """Why non-blocking communicators are configured, or ``""``. See rule (8).

    Environment-only on purpose, and that is a real limit rather than an oversight:
    ``pg_options.config.blocking`` can select the same mode per group and is not
    visible from here. It is also the safe limit to have -- the env is identical on
    every rank, so refusing on it is symmetric and cannot make one rank skip while
    its peers enter the barrier, which is exactly what rule (5) forbids. A group
    that slips past this gate is caught after the fact by :func:`_rc_detail`.
    """

    for var, nonblocking_value in _NONBLOCKING_ENV:
        value = os.environ.get(var)
        if value is not None and value.strip() == nonblocking_value:
            return (
                f"{var}={value} selects non-blocking communicators, where "
                "ncclCommSuspend/Resume return ncclInProgress and must be polled "
                "to completion; polling is not implemented (rule 8), so the "
                "release is disabled rather than read as a failure"
            )
    return ""


def capability() -> Tuple[bool, str]:
    """``(usable, explanation)`` for the runtime NCCL.

    Which symbols are needed, and from which NCCL version, is the backend's
    business; this only decides what to do about the answer. ``ok=False`` is
    always a clean no-op, never an error, so an old runtime makes the feature
    inert rather than broken.
    """
    api = _backend.api()
    if not api.usable:
        return False, api.unavailable_reason
    if os.environ.get("NCCL_DISABLE_MEM_MANAGER", "0") == "1":
        # NCCL's own knob, read (never set) here: with the memory manager off,
        # every suspend would return ncclInvalidUsage, and rule (5) says that
        # refusal has to be recognised before the vote rather than mid-loop.
        return False, f"runtime NCCL {api.version} has NCCL_DISABLE_MEM_MANAGER=1"
    nonblocking = _nonblocking_comm_reason()
    if nonblocking:
        return False, f"runtime NCCL {api.version}: {nonblocking}"
    return True, f"runtime NCCL {api.version}"


def _rc_detail(key: str, rc: int) -> str:
    """Label a non-zero suspend/resume rc for the poison message.

    Only rc=7 gets special treatment, and only because it is the one code that
    means the opposite of what this module does with it -- see rule (8). Naming it
    here is what stops a future operator from debugging a bare "rc=7" against a
    header that calls it a success.
    """

    if rc == _NCCL_IN_PROGRESS:
        return (
            f"{key}(rc={rc} ncclInProgress -- this communicator is non-blocking, "
            "so the call SUCCEEDED and merely needs polling, which is not "
            "implemented. capability() only screens the environment, so this one "
            "was most likely configured via pg_options.config.blocking)"
        )
    return f"{key}(rc={rc})"


def comms(device: object = None) -> List[Tuple[str, int]]:
    """``(group key, raw ncclComm_t)`` for every distinct NCCL communicator.

    rtp-llm registers the same ProcessGroup under several keys in
    ``collective_torch._group_map``, and ``Group.SLEEP_QUIESCE`` is gloo (no
    communicator at all), so both duplicates and non-NCCL groups are dropped.
    Deduplication is by raw pointer, which is what makes the double-suspend rule
    (4) enforceable: suspending the same comm twice via two keys is an error.

    ``device`` is not cosmetic. ``ProcessGroupNCCL::getCommPtr()`` looks the
    communicator up in ``devNCCLCommMap_`` keyed by
    ``c10::cuda::current_device()`` -- not by the group's own device -- and
    returns 0 (silently, no throw) when the key is absent. A hook thread whose
    current device was never set would therefore enumerate zero communicators on
    every rank but rank 0, which is not a crash but a silent asymmetry: sleep
    would log "nothing to suspend" on some ranks and the vote would veto the
    feature for the life of the process. Pinning the device makes the lookup
    match the rank that owns the comm. It cannot lazily create one -- the call is
    a mutex-guarded map lookup with no NCCL bootstrap in it -- so this is safe to
    call on any rank at any time.

    Order is ``_group_map`` insertion order, identical on every rank because the
    groups are created in the same sequence during init. Rule (3) depends on
    that, and :func:`_fingerprint` verifies it rather than trusting it.

    The owning ProcessGroups are dropped here because most callers only report on
    what they find; :func:`suspend_for_sleep` uses the backend directly, since it
    is the one caller that must keep the communicators alive across the sleep.
    """
    return [
        (key, comm) for key, comm, _ in _backend.enumerate_process_group_comms(device)
    ]


# Trailing run of digits, e.g. the ``0`` in ``DP0``. A run, not one character:
# tp_size >= 10 exists.
_GROUP_INDEX_RE = re.compile(r"\d+$")


def _canonical_key(key: str) -> str:
    """A group key with its rank-dependent subgroup index stripped.

    The trap: ``collective_torch`` registers each rank's *own* subgroup under a key
    naming which one it is (``Group.DP.name + str(tp_rank)``), so the families and
    their positions match across ranks -- which is all rule (3) requires -- but the
    raw strings do not. Fingerprinting them raw would veto, and so permanently
    disable the feature, on the first sleep of any tp>1 *and* dp>1 deployment.
    Stripping cannot merge two families: no family name ends in a digit, and a rank
    belongs to exactly one DP and one TP subgroup.
    """
    return _GROUP_INDEX_RE.sub("", key)


def _fingerprint(found: Sequence[Tuple]) -> str:
    """A stable digest of the ordered comm-key list, shared across ranks.

    Takes anything whose first element is the group key, so the same digest covers
    the two-element rows :func:`comms` returns and the three-element rows carrying
    the owning ProcessGroup.

    Rule (3): ranks must walk the same communicators in the same order. Comm
    *pointers* differ per rank so only the keys go in, canonicalised by
    :func:`_canonical_key` -- on a world=4 tp=2 dp=2 job rank 0 holds
    ``[DP_AND_TP, DP0, TP0]`` and rank 3 holds ``[DP_AND_TP, DP1, TP1]``, and those
    two must digest identically. ``hashlib`` rather than ``hash()`` because
    ``hash()`` on a str is salted per process and would differ on every rank.
    """
    keys = "|".join(_canonical_key(row[0]) for row in found)
    return hashlib.sha1(keys.encode("utf-8")).hexdigest()[:12]


def _decide(reason: str, fingerprint: str) -> bool:
    """Bounded, unanimous, fail-closed go/no-go for entering the collective.

    This is the one thing standing between an unlucky sleep and an unrecoverable
    instance. Both suspend and resume block in an untimed ``bootstrapBarrier``
    that only the NCCL abort flag can break, and the sleep commit sequence has no
    cross-rank barrier of its own: a rank whose earlier hook failed goes to ERROR
    and never reaches the hook at all, while its peers would sit in that barrier
    forever holding ``transition_mutex_`` -- so not even ``/wake_up`` could be
    serviced and the only exit would be a restart.

    The hard part is not detecting a missing peer, it is guaranteeing that no two
    ranks reach *different* conclusions. Counting arrivals is not enough: each
    rank runs its own clock from its own arrival, so a rank that arrives 31 s late
    would see a full count and call into NCCL alone, after its peers had already
    timed out and skipped. So the arrival count only forms a *proposal*, and the
    decision itself is a single atomic ``compare_set`` on one key: whoever writes
    first wins, and every rank -- winner and loser alike -- acts on the value the
    CAS hands back. One key, one value, no reader/writer race.

    Both proposals are safe to follow if they lose:

      * proposed VETO, decided GO -- some rank saw all ``world_size`` arrivals, so
        every rank is present and calling is correct;
      * proposed GO, decided VETO -- skipping is always a safe no-op (the
        instance just keeps the NCCL bytes, i.e. today's behaviour).

    A VETO additionally disables the feature for the life of the process. That is
    not timidity, it is what makes the key namespacing sound: ``_vote_seq``
    advances only after a GO, and a GO proves all ranks incremented that
    sequence's key, so the sequence number is provably identical everywhere.
    Were ranks allowed to vote again after a veto, they could drift apart and a
    later vote could assemble a bogus quorum out of a previous cycle's
    half-incremented counter -- failing OPEN, the one outcome worse than failing
    closed.

    Implemented on the rendezvous TCPStore rather than a process group, on
    purpose: it is bounded by construction (unlike ``dist.barrier`` on the
    ~infinite-timeout sleep-quiesce group), it needs no rank-symmetric
    ``new_group`` threaded through init, and it cannot interleave with any
    collective, so it cannot perturb the quiesce consensus that shares this path.
    """
    global _vote_seq, _vote_disabled
    if _vote_disabled is not None:
        logging.warning(
            "[NcclMemory][%s] collective memory release stays disabled: %s",
            reason,
            _vote_disabled,
        )
        return False

    # --- Setup. Nothing is published yet, so bailing out here is a safe skip. ---
    try:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            # Single process, no rendezvous: NCCL's barrier short-circuits at
            # nRanks==1, so there is nobody to stay in step with.
            return True
        world_size = dist.get_world_size()
        if world_size <= 1:
            return True
        rank = dist.get_rank()
        from torch.distributed import distributed_c10d as c10d

        store = c10d._get_default_store()
        seq = _vote_seq
        # The fingerprint is part of the arrival key, not of the decision value:
        # ranks whose comm enumeration differs increment *different* counters, so
        # neither reaches world_size and both end up vetoing the same decision key
        # -- unanimous, fail-closed. Folding it into the decision value instead
        # would let one rank say GO while the mismatching peers skipped.
        arrive_key = f"{_VOTE_KEY_PREFIX}/{seq}/{reason}/arrive/{fingerprint}"
        decision_key = f"{_VOTE_KEY_PREFIX}/{seq}/{reason}/decision"

        # The deadline only bounds the gaps *between* store calls; each call has
        # its own timeout, and the rendezvous store is created with an effectively
        # infinite one. Without this a dead peer would hang inside store.add.
        # Only install the short timeout if we can also put the original back:
        # this store is the shared rendezvous store, and leaving a 30 s timeout on
        # it would silently shorten every unrelated future user.
        prev_timeout = getattr(store, "timeout", None)
        if prev_timeout is not None:
            try:
                store.set_timeout(timedelta(seconds=_VOTE_TIMEOUT_S))
            except Exception:  # noqa: BLE001 - not all Store subclasses expose it
                prev_timeout = None
    except Exception as e:  # noqa: BLE001 - a broken vote must never call NCCL
        _vote_disabled = f"vote mechanism unavailable: {e}"
        logging.warning(
            "[NcclMemory][%s] %s -- skipping NCCL release", reason, _vote_disabled
        )
        return False

    try:
        try:
            arrived = store.add(arrive_key, 1)
        except Exception as e:  # noqa: BLE001
            # Our arrival never became visible, so no peer can be counting us.
            # Still the safe branch.
            _vote_disabled = f"vote mechanism unavailable: {e}"
            logging.warning(
                "[NcclMemory][%s] %s -- skipping NCCL release", reason, _vote_disabled
            )
            return False

        # Past the point of no return: the arrival is now visible, so a peer may
        # count us toward its quorum and head for the untimed NCCL barrier. From
        # here this rank MUST come away with a decision -- "give up and skip" is
        # no longer one of the outcomes, it is the asymmetry the vote exists to
        # prevent. Hence the polling failures below only degrade the proposal, and
        # _publish_decision raises rather than returning.
        deadline = time.perf_counter() + _VOTE_TIMEOUT_S
        while arrived < world_size and time.perf_counter() < deadline:
            time.sleep(_VOTE_POLL_S)
            try:
                arrived = store.add(arrive_key, 0)
            except Exception as e:  # noqa: BLE001
                logging.warning(
                    "[NcclMemory][%s] arrival poll failed (%s) -- proposing VETO "
                    "with %d/%d counted",
                    reason,
                    e,
                    arrived,
                    world_size,
                )
                break
        proposal = _DECISION_GO if arrived >= world_size else _DECISION_VETO
        try:
            decided = _publish_decision(store, decision_key, proposal, reason)
        except NcclMemoryError as e:
            _vote_disabled = f"vote undecided at seq {seq} ({reason})"
            logging.error("[NcclMemory][%s] %s", reason, e)
            raise
    finally:
        if prev_timeout is not None:
            try:
                store.set_timeout(prev_timeout)
            except Exception as e:  # noqa: BLE001
                # The one place the guarantee above can break. Not silently: the
                # shared rendezvous store is now stuck on the vote's short timeout,
                # so unrelated future users of it may start timing out early.
                logging.error(
                    "[NcclMemory][%s] could not restore the rendezvous store's "
                    "timeout (%s) -- it is left at the vote's %.0fs instead of %s, "
                    "so unrelated later users of the shared store may time out early",
                    reason,
                    e,
                    _VOTE_TIMEOUT_S,
                    prev_timeout,
                )

    # Deliberately NOT wrapped in a catch-all: from here on the decision is
    # already published and shared, so swallowing an exception into "return
    # False" would turn a settled GO into a unilateral skip -- exactly the
    # asymmetry everything above is built to prevent. This block is pure local
    # bookkeeping and logging; if it somehow raises, the sleep must fail.
    if decided == _DECISION_GO:
        _vote_seq = seq + 1
        if proposal != _DECISION_GO:
            logging.warning(
                "[NcclMemory][%s] rank %d saw only %d/%d arrivals but the "
                "decision is GO (a peer counted all of them) -- proceeding, "
                "since every rank is provably present",
                reason,
                rank,
                arrived,
                world_size,
            )
        return True

    _vote_disabled = (
        f"vote VETOed at seq {seq} ({reason}): rank {rank} counted "
        f"{arrived}/{world_size} arrivals, decision={decided!r}"
    )
    logging.error(
        "[NcclMemory][%s] decision is VETO at seq %d (rank %d counted %d/%d "
        "arrivals within %.0fs) -- NCCL memory release is now disabled for the "
        "life of this process. Sleep/wake still work, just without the "
        "communicator saving; a peer was absent or the ranks disagreed.",
        reason,
        seq,
        rank,
        arrived,
        world_size,
        _VOTE_TIMEOUT_S,
    )
    return False


def _publish_decision(
    store: object, decision_key: str, proposal: str, reason: str
) -> str:
    """Settle the decision atomically, or raise. Never returns without a verdict.

    Reached only once this rank's arrival is visible to its peers, which is why
    there is no "give up and skip" branch: a peer that counted us may already be
    heading into an untimed NCCL barrier, and skipping would strand it there
    forever holding the sleep transition lock -- an instance that cannot even be
    woken. Failing the sleep instead is recoverable and loud.

    The retry loop is safe precisely because ``compare_set`` is idempotent for our
    purposes: the first writer's value is what every later call reads back, so
    re-issuing it cannot change an already-settled decision.
    """
    deadline = time.perf_counter() + _VOTE_TIMEOUT_S
    last: Optional[Exception] = None
    while True:
        try:
            decided = store.compare_set(decision_key, "", proposal)
            if isinstance(decided, bytes):
                decided = decided.decode("utf-8", "replace")
            return decided
        except Exception as e:  # noqa: BLE001
            last = e
            if time.perf_counter() >= deadline:
                raise NcclMemoryError(
                    f"[{reason}] this rank's arrival is already visible to its "
                    f"peers but the rendezvous store will not settle the decision "
                    f"({last}). Skipping is not safe -- a peer that counted us may "
                    "be entering an untimed NCCL barrier -- so this transition "
                    "fails instead."
                ) from e
            time.sleep(_VOTE_POLL_S)


def is_suspended() -> bool:
    """Whether this process is currently holding communicators suspended."""
    with _lock:
        return bool(_suspended)


def suspend_for_sleep(device: object, reason: str = "sleep") -> None:
    """Release NCCL communicator GPU memory. Collective across all ranks.

    Must be called only once the engine is quiesced: nothing in NCCL enforces
    that, and the ``cudaDeviceSynchronize()`` inside suspend would block on --
    or deadlock against -- a still-spinning collective kernel rather than
    returning an error.

    Raises :class:`NcclMemoryError` in the two cases that must not be papered
    over: a communicator left in a state that cannot be reconciled, and a vote
    that cannot be settled once this rank's arrival is already visible (see
    :func:`_publish_decision`). Both must fail the sleep so the operator learns
    about it before traffic returns. Every other outcome -- feature off, old
    NCCL, no communicator, nothing suspendable, this rank abstaining, a settled
    VETO -- is a logged no-op.
    """
    global _last_failure, _poisoned
    with _lock:
        # Scoped to this transition: whatever failed last time is either fixed or
        # about to be re-recorded below, and a stale reason attached to a fresh
        # hook failure is worse than none.
        _last_failure = None
        if _poisoned is not None:
            _record_failure(f"[{reason}] refusing to suspend: {_poisoned}")
            raise NcclMemoryError(
                f"refusing to suspend: communicator previously poisoned ({_poisoned})"
            )
        if _suspended:
            # Rule (4): a second suspend is ncclInvalidUsage and, on a
            # non-blocking comm, leaves a sticky async error. Treat the bad
            # bookkeeping as the bug it is rather than issuing the call.
            detail = f"{len(_suspended)} communicator(s) already suspended"
            _record_failure(f"[{reason}] refusing to suspend: {detail}")
            raise NcclMemoryError(f"refusing to suspend: {detail}")

        usable, why = capability()
        if not usable:
            # No failure recorded: a skip is a no-op and the sleep still succeeds,
            # so status_text() must stay empty and let the hook keep whatever
            # message actually describes its failure.
            #
            # WARNING, not INFO: the switch is on, so someone is expecting the
            # saving. Silence here would let a deployment run for weeks believing
            # it releases memory that it never touches.
            logging.warning("[NcclMemory][%s] skipped: %s", reason, why)
            return
        api = _backend.api()
        # The backend directly rather than comms(): this is the one caller that must
        # keep the communicators alive until resume, so it needs the owning
        # ProcessGroups, not just their pointers. See _suspended.
        found = _backend.enumerate_process_group_comms(device)
        if not found:
            # WARNING for the same reason capability() uses it: the switch is on, so
            # someone expects the saving, and on any multi-rank deployment "no
            # communicator" means the enumeration broke rather than that there is
            # nothing to do.
            logging.warning("[NcclMemory][%s] no NCCL communicator to suspend", reason)
            return

        # Rule (5): probe every communicator BEFORE voting and abstain on any
        # problem, because after the vote the peers are committed to the barrier and
        # there is no safe skip left.
        suspendable = 0
        unusable: List[str] = []
        for key, comm, _pg in found:
            val = api.stat(comm, _STAT_SUSPEND)
            already = api.stat(comm, _STAT_SUSPENDED)
            if val < 0 or already != 0:
                unusable.append(f"{key}(suspendable={val},suspended={already})")
            else:
                suspendable += val
        if unusable:
            logging.warning(
                "[NcclMemory][%s] abstaining: %d/%d communicator(s) cannot be "
                "suspended on this rank (%s). This rank will not call NCCL; its "
                "peers will count a missing arrival and the release is skipped "
                "everywhere, which is the intended fail-closed behaviour.",
                reason,
                len(unusable),
                len(found),
                ",".join(unusable),
            )
            return
        if suspendable <= 0:
            # Nothing to gain, and every avoided collective is one less way to
            # hang. Checked before the vote so it costs nothing.
            logging.info(
                "[NcclMemory][%s] nothing suspendable across %d comm(s) -- skipped",
                reason,
                len(found),
            )
            return
        try:
            decided = _decide(reason, _fingerprint(found))
        except NcclMemoryError as e:
            # The one vote failure that cannot be made safe: this rank's arrival is
            # visible, the decision is unsettled, and a peer may already be in the
            # untimed barrier. It fails the sleep either way; record it so the
            # operator gets the stage rather than a bare hook name.
            _record_failure(f"[{reason}] {e}")
            raise
        if not decided:
            # A settled VETO is a clean no-op -- nothing was suspended -- so no
            # failure is recorded and the sleep succeeds without the saving.
            return

        free_pre = _driver_free(device)
        t0 = time.perf_counter()
        failures: List[str] = []
        for key, comm, pg in found:
            # Rule (3): every rank walks this whole list even after a failure. The
            # loop must not exit early -- peers that succeeded on this comm move on
            # to the next one and would block in its barrier alone. That applies to
            # a raised exception (a ctypes.ArgumentError from an ABI drift, say)
            # every bit as much as to a non-zero rc, because the peers cannot tell
            # the two apart. Rules (2)+(5): past a GO there is no benign rc and no
            # retry, so both cases just get recorded and the walk continues.
            try:
                rc = api.suspend(comm, _SUSPEND_MEM)
                failure = None if rc == 0 else _rc_detail(key, rc)
            except Exception as e:  # noqa: BLE001
                failure = f"{key}(exc={e})"
            if failure is None:
                # The ProcessGroup, not just the pointer: see _suspended.
                _suspended.append((key, comm, pg))
            else:
                failures.append(failure)
        elapsed = time.perf_counter() - t0
        free_post = _driver_free(device)
        released = max(0, free_post - free_pre)
        logging.info(
            "[NcclMemory][%s] suspended %d/%d comm(s) in %.3fs: suspendable=%.1fMiB "
            "driver_free %.0f -> %.0fMiB (released %.1fMiB)%s",
            reason,
            len(_suspended),
            len(found),
            elapsed,
            suspendable / _MiB,
            free_pre / _MiB,
            free_post / _MiB,
            released / _MiB,
            f" FAILED={failures}" if failures else "",
        )
        if failures:
            _poisoned = f"ncclCommSuspend failed for {','.join(failures)}"
            _record_failure(f"[{reason}] {_poisoned}; restart the instance")
            raise NcclMemoryError(
                f"{_poisoned}; those communicators are half-released and cannot be "
                "retried (their peers have left the barrier), so this instance must "
                "be restarted rather than woken"
            )


def resume_after_wake(device: object, reason: str = "wake") -> None:
    """Remap NCCL communicator memory at its original addresses. Runs first on wake.

    Must complete before ANY collective on the wake path -- see rule (7).

    Gated on the same unanimous decision as suspend, and not on the grounds that
    the suspend vote already proved every rank participates: a rank can be poisoned
    *by* the suspend and go to ERROR, and at sleep level 1 there is no other world
    collective before ``restartEngine``, so this hook would wedge every healthy
    peer. Unlike suspend, nothing here is allowed to skip -- a failed probe or a
    VETO raises, because the memory is unmapped and there is no correct way to
    continue.

    Raises :class:`NcclMemoryError` if a communicator cannot be resumed. Failing
    the wake loudly is the only honest option: continuing would run collectives
    against unmapped virtual addresses.
    """
    global _last_failure, _poisoned
    with _lock:
        _last_failure = None
        # The poison check comes FIRST, before the empty-set early return, and the
        # order is load-bearing: suspend only records a communicator on rc==0, so
        # when *every* comm failed hard we are poisoned with an EMPTY suspended
        # set. Checking emptiness first would return silently here and let the
        # caller wake an instance whose communicators are half-released past their
        # barrier.
        if _poisoned is not None:
            _record_failure(f"[{reason}] refusing to wake: {_poisoned}")
            raise NcclMemoryError(
                f"refusing to wake: communicator poisoned at sleep ({_poisoned}); "
                "restart the instance"
            )
        if not _suspended:
            # Nothing was ever suspended -- the feature was off, vetoed, or found
            # nothing to release. A clean no-op, so no failure is recorded.
            return
        api = _backend.api()
        # Only reachable on a torn runtime: suspend went through capability(), which
        # requires all three symbols, so they were present minutes ago. Poison
        # anyway rather than merely raising -- the communicators are unmapped and
        # the one call that could remap them is gone, so this process can never
        # serve traffic again and a later /wake_up must not look retryable.
        if not api.usable:  # pragma: no cover
            _poisoned = (
                "communicators are suspended but the NCCL memory API is no longer "
                f"available: {api.unavailable_reason}"
            )
            _record_failure(f"[{reason}] {_poisoned}")
            raise NcclMemoryError(_poisoned)

        # Cross-check each recorded pointer against the group that owns it. Note
        # what this is NOT: it is not a liveness oracle. Holding the ProcessGroup
        # (see _suspended) is what keeps the communicator alive; this only catches
        # the residual case where it was destroyed anyway -- torch's watchdog
        # aborting it -- in which the pointer would be dangling and resuming it a
        # use-after-free.
        #
        # A missing/throwing lookup is not safe to ignore: a ProcessGroup reference
        # prevents Python GC, but cannot prevent the NCCL watchdog from aborting the
        # communicator underneath it. Passing the recorded raw pointer after an
        # unanswerable lookup could therefore be a use-after-free. Fail closed and
        # keep the recorded entry so ``is_suspended()`` remains truthful.
        pointer_failures: List[str] = []
        for key, comm, pg in _suspended:
            try:
                current = _backend.comm_ptr_for_group(pg, device)
            except Exception as e:  # noqa: BLE001
                pointer_failures.append(f"{key}(lookup_exc={e})")
                continue
            if not current:
                pointer_failures.append(f"{key}(recorded=0x{comm:x},lookup=0)")
            elif current != comm:
                pointer_failures.append(f"{key}(recorded=0x{comm:x},now=0x{current:x})")
        if pointer_failures:
            _poisoned = (
                f"communicator pointer validation failed for {','.join(pointer_failures)} "
                "while suspended; the recorded pointer cannot be trusted and "
                "resuming it could be a use-after-free"
            )
            _record_failure(f"[{reason}] {_poisoned}")
            raise NcclMemoryError(_poisoned)

        # Reverse suspend order, so the last communicator suspended is the first
        # resumed: peer handle exchange is pairwise, and unwinding in reverse is
        # the ordering least likely to surprise a future multi-comm topology.
        todo = list(reversed(_suspended))

        # Rule (5) applies in this direction too: ncclCommResume runs the same
        # CommCheck/ncclCommEnsureReady preamble, so it can also return BEFORE the
        # barrier and leave the peers that did enter blocked there forever. Probe
        # first, then vote.
        #
        # But unlike suspend a failure here cannot abstain -- the memory is already
        # unmapped and /wake_up cannot be re-driven -- so it raises WITHOUT voting.
        # The peers then count a missing arrival, veto, and raise too, so every rank
        # lands in a clean error state instead of one raising while N-1 hang.
        #
        # Honest limit: this closes the CommCheck/EnsureReady class only. A
        # cuMemCreate OOM inside resume's remap loop is also pre-barrier and is not
        # predictable from memstats, so a residual hang remains by construction; the
        # complete fix is a post-GO watchdog tripping comm->abortFlag, out of scope.
        unusable: List[str] = []
        for key, comm, _pg in todo:
            total = api.stat(comm, _STAT_TOTAL)
            already = api.stat(comm, _STAT_SUSPENDED)
            if total < 0 or already != 1:
                unusable.append(f"{key}(total={total},suspended={already})")
        if unusable:
            _poisoned = (
                f"pre-resume probe failed for {','.join(unusable)}: "
                "ncclCommResume would refuse before its barrier"
            )
            _record_failure(f"[{reason}] {_poisoned}")
            logging.error(
                "[NcclMemory][%s] refusing to resume: %d/%d communicator(s) fail the "
                "pre-vote probe on this rank (%s). Raising WITHOUT voting, so the "
                "peers count a missing arrival and raise as well rather than "
                "blocking in a barrier this rank never enters.",
                reason,
                len(unusable),
                len(todo),
                ",".join(unusable),
            )
            raise NcclMemoryError(
                f"{_poisoned}; those communicators' virtual addresses are still "
                "unmapped and every later collective would fault -- restart the "
                "instance"
            )

        try:
            decided = _decide(reason, _fingerprint(todo))
        except NcclMemoryError as e:
            # Unlike suspend, an unsettled vote here is terminal rather than merely
            # loud: the addresses stay unmapped, so poison as well as record.
            _poisoned = f"resume vote could not be settled: {e}"
            _record_failure(f"[{reason}] {_poisoned}")
            raise
        if not decided:
            _poisoned = "resume vote did not reach GO"
            _record_failure(f"[{reason}] {_poisoned}")
            raise NcclMemoryError(
                f"cannot resume {len(todo)} suspended communicator(s): {_poisoned}. "
                "Their virtual addresses are unmapped, so continuing would fault on "
                "the first collective -- restart the instance."
            )

        free_pre = _driver_free(device)
        t0 = time.perf_counter()
        failures: List[str] = []
        still_suspended: List[Tuple[str, int, object]] = []
        for key, comm, pg in todo:
            # Rules (3) and (2): walk the whole list even while failing, once each,
            # and treat a raised exception exactly like a non-zero rc -- the peers
            # blocked on the next comm's barrier cannot tell them apart.
            try:
                rc = api.resume(comm)
                failure = None if rc == 0 else _rc_detail(key, rc)
            except Exception as e:  # noqa: BLE001
                failure = f"{key}(exc={e})"
            if failure is not None:
                failures.append(failure)
                still_suspended.append((key, comm, pg))
        elapsed = time.perf_counter() - t0
        free_post = _driver_free(device)
        restored = max(0, free_pre - free_post)
        resumed = len(todo) - len(failures)
        # Keep the ones that failed. Clearing unconditionally would make
        # is_suspended() report False while communicators are in fact still
        # unmapped -- a lie told at exactly the moment someone is debugging why
        # collectives are faulting.
        logging.info(
            "[NcclMemory][%s] resumed %d/%d comm(s) in %.3fs: driver_free "
            "%.0f -> %.0fMiB (reclaimed %.1fMiB)%s",
            reason,
            resumed,
            len(todo),
            elapsed,
            free_pre / _MiB,
            free_post / _MiB,
            restored / _MiB,
            f" FAILED={failures}" if failures else "",
        )
        if failures:
            _suspended[:] = still_suspended
            _poisoned = f"ncclCommResume failed for {','.join(failures)}"
            _record_failure(f"[{reason}] {_poisoned}; restart the instance")
            raise NcclMemoryError(
                f"{_poisoned}; those communicators' virtual addresses are still "
                "unmapped and every later collective would fault -- restart the "
                "instance"
            )
        # Keep ``_suspended`` intact until the post-resume evidence is collectively
        # verified. A local stat failure must not let this rank report wake success
        # while peers proceed to weight reload and traffic.
        evidence_failures = _resume_evidence_failures(api, todo, reason)
        evidence_fingerprint = _fingerprint(todo)
        if evidence_failures:
            evidence_fingerprint += ":evidence-failed"
        try:
            evidence_decided = _decide(f"{reason}-verify", evidence_fingerprint)
        except NcclMemoryError as e:
            _poisoned = f"resume evidence vote could not be settled: {e}"
            _record_failure(f"[{reason}] {_poisoned}")
            raise
        if not evidence_decided or evidence_failures:
            detail = (
                ", ".join(evidence_failures)
                if evidence_failures
                else "peer evidence vote was vetoed"
            )
            _poisoned = f"resume evidence validation failed ({detail})"
            _record_failure(f"[{reason}] {_poisoned}; restart the instance")
            # The NCCL call returned success, but the communicator is not proven
            # usable. Retain the entries for diagnostics and prevent a retry.
            _suspended[:] = todo
            raise NcclMemoryError(
                f"{_poisoned}; communicator state is not safe for traffic -- "
                "restart the instance"
            )
        _suspended.clear()


def _resume_evidence_failures(
    api: "_backend.NcclApi",
    todo: Sequence[Tuple[str, int, object]],
    reason: str,
) -> List[str]:
    """Return communicators whose post-resume state cannot be proven active.

    Rule (6): resume returns ``ncclSuccess`` even when individual peer imports
    failed -- those paths only ``WARN`` and continue -- so the return code alone
    does not prove the communicator is whole. ``ncclStatGpuMemSuspended`` is the
    machine-readable cross-check available from here; a stuck 1, an error, or an
    unavailable entry is treated as failure. The caller turns the result into a
    rank-symmetric vote before allowing weight reload or traffic.
    """
    failures: List[str] = []
    for key, comm, _pg in todo:
        try:
            suspended = api.stat(comm, _STAT_SUSPENDED)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{key}(stat_exc={e})")
            continue
        if suspended != 0:
            failures.append(f"{key}(suspended={suspended})")
    if failures:
        logging.error(
            "[NcclMemory][%s] post-resume evidence failed for %d/%d comm(s): %s; "
            "the wake will fail closed",
            reason,
            len(failures),
            len(todo),
            ",".join(failures),
        )
    return failures


def _driver_free(device: object) -> int:
    """Physical free bytes per the driver, or 0 if unavailable.

    The driver's own number, not torch's: the whole point is memory torch never
    knew about, so ``memory_reserved()`` cannot see this change.
    """
    try:
        return int(torch.cuda.mem_get_info(device)[0])
    except Exception:  # noqa: BLE001 - pragma: no cover
        return 0


def _reset_for_testing() -> None:
    """Drop all cached state. Tests only."""
    global _last_failure, _poisoned, _vote_seq, _vote_disabled
    with _lock:
        _backend.reset_for_testing()
        _last_failure = None
        _suspended.clear()
        _poisoned = None
        _vote_seq = 0
        _vote_disabled = None

"""CPU-only tests for rtp_llm/utils/nccl_memory.py.

Nothing here touches a GPU or real NCCL. Two things are faked:

  * ``nccl_memory_backend.api`` returns a real :class:`NcclApi` wrapping a
    ``SimpleNamespace`` that carries only the symbols a given test wants to exist,
    because ``NcclApi.missing_symbols`` probes with ``hasattr`` and
    ``NcclApi.stat`` calls
    ``lib.ncclCommMemStats(c_void_p, c_int, byref(c_uint64))``. The fake writes
    back through the pointer (``out._obj.value = ...``), so the real ctypes
    marshalling is exercised rather than monkeypatched away. Faking at the
    ``api()`` seam rather than reaching inside ``nccl_memory`` is deliberate: it
    is the same boundary production uses, so a test that needed to reach past it
    would be evidence the boundary had leaked.
  * ``torch.distributed``'s four query functions plus
    ``distributed_c10d._get_default_store`` are patched, but the store handed
    back is a *real* ``TCPStore``. That is deliberate: the entire safety argument
    of :func:`nccl_memory._decide` rests on ``compare_set`` being an atomic CAS,
    and a dict stand-in would not exercise it.

Multiple ranks are simulated by calling ``_decide`` repeatedly from this one
process with ``get_rank``/``get_world_size`` patched, and by pre-poking the store
to stand in for a peer's arrival or a peer's already-published decision.
"""

import ctypes
import hashlib
import logging
import socket
import sys
import time
import types
import unittest
from datetime import timedelta
from typing import Dict, List, Optional, Tuple
from unittest import mock

import torch.distributed as dist
from torch.distributed import TCPStore
from torch.distributed import distributed_c10d as c10d

from rtp_llm.utils import nccl_memory
from rtp_llm.utils import nccl_memory_backend as backend

_ALL_SYMS = ("ncclCommSuspend", "ncclCommResume", "ncclCommMemStats")

# Arbitrary non-null "comm pointers". Only their identity matters.
_COMM_A = 0xAAAA0000
_COMM_B = 0xBBBB0000


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_store(attempts: int = 20) -> TCPStore:
    """A real single-rank TCPStore on a free port.

    Retried because an ephemeral port can be taken between the probe bind and
    the store's own bind (including by a previous test's store in TIME_WAIT);
    that race is in this helper, not in the code under test.
    """
    last: Optional[Exception] = None
    for _ in range(attempts):
        try:
            return TCPStore(
                "127.0.0.1", _free_port(), 1, True, timeout=timedelta(seconds=5)
            )
        except Exception as e:  # noqa: BLE001 - EADDRINUSE / DistNetworkError
            last = e
            time.sleep(0.05)
    raise AssertionError(f"could not bind a TCPStore after {attempts} tries: {last}")


class _FakeNccl:
    """Records every call and returns caller-chosen return codes."""

    def __init__(
        self,
        suspend_rc: object = 0,
        resume_rc: object = 0,
        suspendable: object = 64 * 1024 * 1024,
        suspended_flag: object = None,
        memstats_rc: object = 0,
    ) -> None:
        # (comm ptr, flags) in call order.
        self.suspend_calls: List[Tuple[int, int]] = []
        self.resume_calls: List[int] = []
        self.memstats_calls: List[Tuple[int, int]] = []
        # ncclStatGpuMemSuspended per comm, maintained by the calls themselves.
        # Real state, not a constant, because BOTH directions now read the flag in
        # their pre-vote probe: suspend requires 0, resume requires 1. A stateless
        # fake could only ever satisfy one of them. ``suspended_flag`` stays as an
        # explicit override for the tests that need NCCL to disagree with reality.
        self._suspended_state: dict = {}
        # One ordered log across all three entry points. Needed because the
        # pre-vote probe's guarantee is an ORDERING one -- every memstats read
        # before any suspend -- which per-method lists cannot express.
        self.calls: List[Tuple[str, int]] = []
        self._suspend_rc = suspend_rc
        self._resume_rc = resume_rc
        # Every "rc-like" and every stat value accepts either a scalar (same for
        # all communicators) or a {comm ptr: value} dict. The per-comm form is
        # what makes rule (5) testable at all: the whole danger is one rank, or
        # one comm, behaving differently from the rest.
        self.suspendable = suspendable
        # None = report the tracked state; anything else overrides it.
        self.suspended_flag = suspended_flag
        self.memstats_rc = memstats_rc

    @staticmethod
    def _rc(spec: object, comm: int) -> int:
        value = spec.get(comm, 0) if isinstance(spec, dict) else spec
        if isinstance(value, BaseException):
            # An rc spec may also be an exception instance. Not a convenience: an
            # ABI drift surfaces from ctypes as a raise rather than as a return
            # code, and the peers blocked in the next comm's barrier cannot tell
            # the two apart, so the loop must not either.
            raise value
        return int(value)  # type: ignore[arg-type]

    def ncclCommSuspend(self, comm: ctypes.c_void_p, flags: ctypes.c_int) -> int:
        ptr = comm.value or 0
        self.suspend_calls.append((ptr, flags.value))
        self.calls.append(("suspend", ptr))
        rc = self._rc(self._suspend_rc, ptr)
        if rc == 0:
            self._suspended_state[ptr] = 1
        return rc

    def ncclCommResume(self, comm: ctypes.c_void_p) -> int:
        ptr = comm.value or 0
        self.resume_calls.append(ptr)
        self.calls.append(("resume", ptr))
        rc = self._rc(self._resume_rc, ptr)
        if rc == 0:
            self._suspended_state[ptr] = 0
        return rc

    def ncclCommMemStats(
        self, comm: ctypes.c_void_p, stat: ctypes.c_int, out: object
    ) -> int:
        ptr = comm.value or 0
        self.memstats_calls.append((ptr, stat.value))
        self.calls.append(("memstats", ptr))
        if stat.value == nccl_memory._STAT_SUSPENDED:
            val = (
                self._suspended_state.get(ptr, 0)
                if self.suspended_flag is None
                else self._rc(self.suspended_flag, ptr)
            )
        elif stat.value == nccl_memory._STAT_PERSIST:
            val = 0
        else:  # _STAT_SUSPEND / _STAT_TOTAL
            val = self._rc(self.suspendable, ptr)
        out._obj.value = val  # type: ignore[attr-defined]
        return self._rc(self.memstats_rc, ptr)

    def as_lib(self, missing: Tuple[str, ...] = ()) -> types.SimpleNamespace:
        """A ctypes.CDLL stand-in exposing only the symbols not in ``missing``."""
        ns = types.SimpleNamespace()
        for name in _ALL_SYMS:
            if name not in missing:
                setattr(ns, name, getattr(self, name))
        return ns


class _StoreSpy:
    """Delegates to a real TCPStore while counting mutating calls."""

    def __init__(self, inner: TCPStore) -> None:
        self.inner = inner
        self.add_calls: List[Tuple[str, int]] = []
        self.compare_set_calls: List[Tuple[str, str, str]] = []

    @property
    def timeout(self) -> timedelta:
        return self.inner.timeout

    def set_timeout(self, t: timedelta) -> None:
        self.inner.set_timeout(t)

    def add(self, key: str, amount: int) -> int:
        self.add_calls.append((key, amount))
        return self.inner.add(key, amount)

    def compare_set(self, key: str, expected: str, desired: str) -> bytes:
        self.compare_set_calls.append((key, expected, desired))
        return self.inner.compare_set(key, expected, desired)

    @property
    def touches(self) -> int:
        return len(self.add_calls) + len(self.compare_set_calls)


class _ExplodingStore:
    """A store whose ``add`` fails, i.e. a broken rendezvous."""

    def __init__(self) -> None:
        self.add_calls = 0

    def add(self, key: str, amount: int) -> int:
        self.add_calls += 1
        raise RuntimeError("simulated store failure")

    def compare_set(self, key: str, expected: str, desired: str) -> bytes:
        raise AssertionError("compare_set must not be reached when add() failed")


class _HalfBrokenStore:
    """``add`` works, ``compare_set`` never does.

    The precise shape of the F2 bug: the arrival is published, so peers can count
    this rank and enter the untimed barrier, but the decision cannot be settled.
    There is no safe skip from here.
    """

    def __init__(self, inner: TCPStore) -> None:
        self.inner = inner
        self.compare_set_calls = 0

    @property
    def timeout(self) -> timedelta:
        return self.inner.timeout

    def set_timeout(self, t: timedelta) -> None:
        self.inner.set_timeout(t)

    def add(self, key: str, amount: int) -> int:
        return self.inner.add(key, amount)

    def compare_set(self, key: str, expected: str, desired: str) -> bytes:
        self.compare_set_calls += 1
        raise RuntimeError("simulated CAS failure")


class NcclMemoryTestBase(unittest.TestCase):
    def setUp(self) -> None:
        nccl_memory._reset_for_testing()
        self._world = 1
        self._rank = 0
        self._tcp = _make_store()
        self.store = _StoreSpy(self._tcp)
        self._store_lookups = 0

        def _get_store():
            self._store_lookups += 1
            return self.store

        patches = [
            mock.patch.object(dist, "is_available", lambda: True),
            mock.patch.object(dist, "is_initialized", lambda: True),
            mock.patch.object(dist, "get_world_size", lambda *a, **k: self._world),
            mock.patch.object(dist, "get_rank", lambda *a, **k: self._rank),
            mock.patch.object(c10d, "_get_default_store", _get_store),
            # Keep the vote fast: a timeout case must not cost 30s.
            mock.patch.object(nccl_memory, "_VOTE_TIMEOUT_S", 0.3),
            mock.patch.object(nccl_memory, "_VOTE_POLL_S", 0.01),
            # The module reads the driver's free bytes for logging only; stub it
            # so no test ever initialises CUDA.
            mock.patch.object(nccl_memory, "_driver_free", lambda device: 0),
            mock.patch.dict("os.environ", {"NCCL_DISABLE_MEM_MANAGER": "0"}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(nccl_memory._reset_for_testing)

    # --- helpers ---------------------------------------------------------
    def arrive_key(self, fingerprint: str, seq: int = 0, reason: str = "sleep") -> str:
        return f"{nccl_memory._VOTE_KEY_PREFIX}/{seq}/{reason}/arrive/{fingerprint}"

    def decision_key(self, seq: int = 0, reason: str = "sleep") -> str:
        return f"{nccl_memory._VOTE_KEY_PREFIX}/{seq}/{reason}/decision"

    def read_counter(self, key: str) -> int:
        # add(k, 0) reads without mutating; go straight to the inner store so it
        # does not pollute the spy's counts.
        return self._tcp.add(key, 0)

    def install_lib(self, fake: _FakeNccl, missing: Tuple[str, ...] = ()) -> _FakeNccl:
        # A real NcclApi over the fake handle, patched in at the one seam the
        # policy layer uses. Everything inside NcclApi -- the hasattr probing and
        # the ctypes marshalling -- therefore runs for real.
        api = backend.NcclApi(fake.as_lib(missing), "2.31.2")
        p = mock.patch.object(backend, "api", lambda: api)
        p.start()
        self.addCleanup(p.stop)
        return fake

    def install_comms(
        self,
        found: List[Tuple[str, int]],
        ptr_now: Optional[Dict[int, int]] = None,
        ptr_error: Optional[Dict[int, BaseException]] = None,
    ) -> Dict[int, object]:
        """Patch the two BACKEND seams, leaving the policy layer's own code live.

        Patching ``nccl_memory.comms`` instead would now skip the code under test:
        ``comms()`` drops the group column, and ``suspend_for_sleep`` deliberately
        bypasses it to keep the owning ProcessGroups alive across the sleep. Both
        seams take the device, because the whole reason it is threaded through is
        that ``getCommPtr()`` is keyed on ``current_device()``.

        ``ptr_now`` / ``ptr_error`` drive the wake-path cross-check: what a group's
        pointer reads back as later, and which lookup refuses to answer at all.
        """

        groups: Dict[int, object] = {
            comm: types.SimpleNamespace(key=key) for key, comm in found
        }
        rows = [(key, comm, groups[comm]) for key, comm in found]
        owner = {id(pg): comm for comm, pg in groups.items()}
        self.enumerate_devices: List[object] = []
        self.ptr_devices: List[object] = []

        def _enumerate(device: object = None) -> List[Tuple[str, int, object]]:
            self.enumerate_devices.append(device)
            return list(rows)

        def _ptr_for(pg: object, device: object = None) -> int:
            self.ptr_devices.append(device)
            comm = owner[id(pg)]
            if ptr_error and comm in ptr_error:
                raise ptr_error[comm]
            return (ptr_now or {}).get(comm, comm)

        for attr, value in (
            ("enumerate_process_group_comms", _enumerate),
            ("comm_ptr_for_group", _ptr_for),
        ):
            p = mock.patch.object(backend, attr, value)
            p.start()
            self.addCleanup(p.stop)
        return groups

    def suspended_keys(self) -> List[Tuple[str, int]]:
        """``_suspended`` without the ProcessGroup column."""
        return [(key, comm) for key, comm, _ in nccl_memory._suspended]

    def simulate_fresh_peer_process(self) -> None:
        """Reset only the per-process vote latch, keeping the shared store.

        A second rank is a second *process*: it has its own ``_vote_disabled`` /
        ``_vote_seq`` but the same rendezvous store. Resetting just those two
        globals is what "another rank calls _decide" looks like from here.
        """
        nccl_memory._vote_disabled = None
        nccl_memory._vote_seq = 0


class DecideTest(NcclMemoryTestBase):
    """The go/no-go vote. A wrong answer here hangs a cluster permanently."""

    # Case 1
    def test_single_rank_returns_true_without_touching_store(self) -> None:
        self._world = 1
        self.assertTrue(nccl_memory._decide("sleep", "fp"))
        self.assertEqual(self.store.touches, 0)
        self.assertEqual(self._store_lookups, 0)
        self.assertEqual(nccl_memory._vote_seq, 0)
        self.assertIsNone(nccl_memory._vote_disabled)

    # Case 1 (variant): no rendezvous at all.
    def test_dist_not_initialised_returns_true_without_touching_store(self) -> None:
        self._world = 8  # would matter if it were ever read
        with mock.patch.object(dist, "is_initialized", lambda: False):
            self.assertTrue(nccl_memory._decide("sleep", "fp"))
        self.assertEqual(self.store.touches, 0)
        with mock.patch.object(dist, "is_available", lambda: False):
            self.assertTrue(nccl_memory._decide("sleep", "fp"))
        self.assertEqual(self.store.touches, 0)
        self.assertIsNone(nccl_memory._vote_disabled)

    # Case 2
    def test_all_ranks_arrive_decides_go_and_advances_seq(self) -> None:
        self._world = 2
        fp = "abc123"
        # The peer rank arrived first.
        self._tcp.add(self.arrive_key(fp), 1)

        self.assertTrue(nccl_memory._decide("sleep", fp))

        self.assertEqual(nccl_memory._vote_seq, 1)
        self.assertIsNone(nccl_memory._vote_disabled)
        self.assertEqual(self.read_counter(self.arrive_key(fp)), 2)
        self.assertEqual(self._tcp.get(self.decision_key()), b"go")
        # Full count on the first poll: no busy-wait iterations.
        self.assertEqual(self.store.add_calls, [(self.arrive_key(fp), 1)])

    # Case 3
    def test_missing_peer_vetoes_latches_and_second_call_is_free(self) -> None:
        self._world = 2
        fp = "abc123"

        self.assertFalse(nccl_memory._decide("sleep", fp))
        self.assertIsNotNone(nccl_memory._vote_disabled)
        self.assertEqual(nccl_memory._vote_seq, 0)  # a VETO never advances the seq
        self.assertEqual(self._tcp.get(self.decision_key()), b"veto")
        counter_after_first = self.read_counter(self.arrive_key(fp))
        self.assertEqual(counter_after_first, 1)
        touches_after_first = self.store.touches

        # The latch is what makes the stale seq-0 keys unexploitable: a second
        # attempt must not re-enter the vote at all.
        self.assertFalse(nccl_memory._decide("sleep", fp))
        self.assertEqual(self.read_counter(self.arrive_key(fp)), counter_after_first)
        self.assertEqual(self.store.touches, touches_after_first)

    # Case 4 -- the late arriver. The scenario that would otherwise call an
    # uninterruptible collective alone.
    def test_late_arriver_with_full_count_still_follows_published_veto(self) -> None:
        self._world = 2
        fp = "abc123"
        # The peer waited, timed out, published VETO and skipped the collective.
        self.assertEqual(
            self._tcp.compare_set(self.decision_key(), "", "veto"), b"veto"
        )
        # ...and only then does this rank show up, seeing a FULL arrival count.
        self._tcp.add(self.arrive_key(fp), 1)

        self.assertFalse(nccl_memory._decide("sleep", fp))

        self.assertEqual(self.read_counter(self.arrive_key(fp)), 2)  # proposal was GO
        self.assertEqual(nccl_memory._vote_seq, 0)
        self.assertIsNotNone(nccl_memory._vote_disabled)
        self.assertEqual(self._tcp.get(self.decision_key()), b"veto")

    # Case 5 -- the mirror: proposed VETO, decided GO. Following GO is safe
    # because a GO proves some rank counted every arrival.
    def test_timed_out_rank_follows_published_go(self) -> None:
        self._world = 2
        fp = "abc123"
        # A peer counted all arrivals (in a window this rank missed) and
        # published GO before this rank's own deadline expired.
        self.assertEqual(self._tcp.compare_set(self.decision_key(), "", "go"), b"go")

        self.assertTrue(nccl_memory._decide("sleep", fp))

        self.assertEqual(nccl_memory._vote_seq, 1)
        self.assertIsNone(nccl_memory._vote_disabled)
        # This rank really did propose VETO: it never saw world_size arrivals.
        self.assertLess(self.read_counter(self.arrive_key(fp)), self._world)
        self.assertIn((self.decision_key(), "", "veto"), self.store.compare_set_calls)

    # Case 6 -- divergent comm enumeration must fail closed.
    def test_diverging_fingerprints_make_both_ranks_veto(self) -> None:
        self._world = 2
        fp_a, fp_b = "fingerprint_a", "fingerprint_b"

        self._rank = 0
        self.assertFalse(nccl_memory._decide("sleep", fp_a))

        # Rank 1 is a different process: same store, its own latch.
        self.simulate_fresh_peer_process()
        self._rank = 1
        self.assertFalse(nccl_memory._decide("sleep", fp_b))

        # Different arrival counters, so neither could ever reach quorum...
        self.assertEqual(self.read_counter(self.arrive_key(fp_a)), 1)
        self.assertEqual(self.read_counter(self.arrive_key(fp_b)), 1)
        # ...and both landed on the SAME decision key, so the answer is unanimous.
        self.assertEqual(self._tcp.get(self.decision_key()), b"veto")
        self.assertIsNotNone(nccl_memory._vote_disabled)

    # Case 8
    def test_store_failure_vetoes_and_latches(self) -> None:
        self._world = 2
        broken = _ExplodingStore()
        with mock.patch.object(c10d, "_get_default_store", lambda: broken):
            self.assertFalse(nccl_memory._decide("sleep", "fp"))
        self.assertEqual(broken.add_calls, 1)  # no retry
        self.assertIsNotNone(nccl_memory._vote_disabled)
        self.assertIn("vote mechanism unavailable", nccl_memory._vote_disabled)
        self.assertEqual(nccl_memory._vote_seq, 0)

    def test_store_timeout_is_restored(self) -> None:
        """A dead peer must not leave the rendezvous store on a 30s timeout."""
        self._world = 2
        before = self._tcp.timeout
        self.assertFalse(nccl_memory._decide("sleep", "fp"))
        self.assertEqual(self._tcp.timeout, before)

    # Case 8b -- the F2 bug. `add` succeeded, so this rank is COMMITTED.
    def test_cas_failure_after_a_visible_arrival_raises(self) -> None:
        """Once the arrival is visible, "give up and skip" is off the table.

        This used to be swallowed into ``return False``, which is the worst
        available outcome: a peer that counted this rank's arrival heads into an
        untimed ``bootstrapBarrier`` while this rank quietly walks away, and it
        does so holding ``transition_mutex_`` -- so the instance cannot even be
        woken, only restarted. Raising fails one sleep instead.
        """
        self._world = 2
        broken = _HalfBrokenStore(self._tcp)
        with mock.patch.object(c10d, "_get_default_store", lambda: broken):
            with self.assertRaises(nccl_memory.NcclMemoryError) as cm:
                nccl_memory._decide("sleep", "fp")

        self.assertIn("already visible to its peers", str(cm.exception))
        # It retried rather than giving up on the first error...
        self.assertGreater(broken.compare_set_calls, 1)
        # ...and the arrival really was published, which is what made the raise
        # mandatory.
        self.assertEqual(self.read_counter(self.arrive_key("fp")), 1)
        self.assertIsNotNone(nccl_memory._vote_disabled)
        self.assertEqual(nccl_memory._vote_seq, 0)
        # The shared rendezvous store must not be left on the vote's short
        # timeout even on the raising path.
        self.assertEqual(self._tcp.timeout, timedelta(seconds=5))

    def test_suspend_propagates_an_undecided_vote(self) -> None:
        """The raise must reach the sleep, not be absorbed by suspend_for_sleep."""
        fake = self.install_lib(_FakeNccl())
        self.install_comms([("Group.DP_AND_TP", _COMM_A)])
        self._world = 2
        broken = _HalfBrokenStore(self._tcp)
        with mock.patch.object(c10d, "_get_default_store", lambda: broken):
            with self.assertRaises(nccl_memory.NcclMemoryError):
                nccl_memory.suspend_for_sleep(device=None)
        self.assertEqual(fake.suspend_calls, [])
        self.assertEqual(self.suspended_keys(), [])
        # An unsettled vote is the single most confusing way for a sleep to fail,
        # so it is the last one that should reach the operator as a bare hook
        # name. It fails the sleep either way; the diagnostic is the point.
        self.assertIn("will not settle the decision", nccl_memory.status_text())

    def test_resume_poisons_when_the_vote_cannot_be_settled(self) -> None:
        """An unsettled vote on the WAKE path is terminal, not just loud.

        Suspend can leave the instance healthy after this (nothing was released),
        but here the addresses are already unmapped and the resume never happened,
        so a retried ``/wake_up`` must not look viable. Hence poison, not merely a
        raise.
        """
        fake = _FakeNccl()
        self.install_lib(fake)
        self.install_comms([("Group.DP_AND_TP", _COMM_A)])
        nccl_memory.suspend_for_sleep(device=None)
        self.assertEqual(len(fake.resume_calls), 0)

        self._world = 2
        broken = _HalfBrokenStore(self._tcp)
        with mock.patch.object(c10d, "_get_default_store", lambda: broken):
            with self.assertRaises(nccl_memory.NcclMemoryError):
                nccl_memory.resume_after_wake(device=None)

        self.assertEqual(fake.resume_calls, [])  # never entered the barrier
        self.assertIsNotNone(nccl_memory._poisoned)
        self.assertIn("could not be settled", nccl_memory._poisoned)
        self.assertIn("could not be settled", nccl_memory.status_text())
        # Still suspended, so a later /sleep is refused rather than doubling up.
        self.assertTrue(nccl_memory.is_suspended())

    # Case 16 -- more than two ranks, mixed proposals.
    def test_three_ranks_with_one_missing_all_follow_one_veto(self) -> None:
        self._world = 3
        fp = "abc123"
        # Rank 1 arrived; rank 2 is dead and never will.
        self._tcp.add(self.arrive_key(fp), 1)

        self._rank = 0
        self.assertFalse(nccl_memory._decide("sleep", fp))
        self.simulate_fresh_peer_process()
        self._rank = 1
        self.assertFalse(nccl_memory._decide("sleep", fp))

        self.assertEqual(self._tcp.get(self.decision_key()), b"veto")
        self.assertEqual(self.read_counter(self.arrive_key(fp)), 3)

    def test_three_ranks_reach_one_unanimous_decision(self) -> None:
        """Mixed proposals across three ranks still collapse to one answer.

        Simulating ranks sequentially means the first two necessarily time out at
        1/3 and 2/3 and propose VETO, while the third sees a full count and
        proposes GO. Only ONE value can ever be stored (that is what the CAS
        buys), and every rank acts on what it reads back. Unanimity, not which
        way it lands, is the property under test -- a split here is the cluster
        hang everything else in this module is arranged to prevent.
        """
        self._world = 3
        fp = "abc123"
        decisions = []
        for rank in range(3):
            self.simulate_fresh_peer_process()
            self._rank = rank
            decisions.append(nccl_memory._decide("sleep", fp))

        self.assertEqual(self.read_counter(self.arrive_key(fp)), 3)
        self.assertEqual(len(set(decisions)), 1, f"ranks disagreed: {decisions}")
        stored = self._tcp.get(self.decision_key())
        self.assertEqual(decisions[0], stored == b"go")

    def test_stale_arrival_counter_would_fabricate_a_quorum(self) -> None:
        """Pins the one precondition the vote does NOT defend itself against.

        A leftover arrival counter at this sequence number is indistinguishable
        from live peers, so a rank would propose GO alone. That is safe here only
        because of an architectural fact, not because of a check: ``_vote_seq``
        advances solely after a GO (which proves every rank advanced with it), so
        keys are never reused within a process lifetime, and a restart brings a
        brand-new rendezvous store. This test exists so that anyone who makes the
        store outlive the ranks -- or lets a veto re-vote -- sees the consequence
        spelled out rather than discovering it as a cluster hang.
        """
        self._world = 2
        fp = "abc123"
        self._tcp.add(self.arrive_key(fp), 2)  # stale: nobody is actually there

        self.assertTrue(nccl_memory._decide("sleep", fp))
        self.assertEqual(self._tcp.get(self.decision_key()), b"go")

    def test_sleep_and_wake_votes_use_separate_decision_keys(self) -> None:
        """Suspend and resume vote independently within one sequence number."""
        self._world = 2
        fp = "abc123"
        self._tcp.add(self.arrive_key(fp, reason="sleep"), 1)
        self.assertTrue(nccl_memory._decide("sleep", fp))
        self.assertEqual(nccl_memory._vote_seq, 1)

        self._tcp.add(self.arrive_key(fp, seq=1, reason="wake"), 1)
        self.assertTrue(nccl_memory._decide("wake", fp))
        self.assertEqual(nccl_memory._vote_seq, 2)


class FingerprintTest(NcclMemoryTestBase):
    # Case 7
    def test_fingerprint_is_a_stable_cross_process_digest(self) -> None:
        found = [("Group.DP_AND_TP", _COMM_A), ("Group.TP", _COMM_B)]
        # Hardcoded on purpose. sha1("Group.DP_AND_TP|Group.TP")[:12]. If someone
        # ever swaps hashlib for hash(), this fails -- hash() on a str is salted
        # by PYTHONHASHSEED and would differ on every rank, so ranks would
        # increment different arrival keys and the feature would silently never
        # reach quorum.
        self.assertEqual(nccl_memory._fingerprint(found), "dd4f550d71f0")
        self.assertEqual(
            nccl_memory._fingerprint(found),
            hashlib.sha1(b"Group.DP_AND_TP|Group.TP").hexdigest()[:12],
        )
        self.assertEqual(nccl_memory._fingerprint([]), "da39a3ee5e6b")

    # The F3 bug: without canonicalisation the feature is silently DEAD on any
    # deployment with tp>1 AND dp>1 -- every rank fingerprints a different string,
    # so the arrival counters never reach quorum, the first sleep vetoes, and the
    # latch keeps it off for the life of the process. No crash, no error: just
    # memory that is never released and a log line nobody reads.
    def test_per_rank_subgroup_indices_do_not_change_the_fingerprint(self) -> None:
        # world=4, tp=2, dp=2. collective_torch registers each rank's OWN
        # subgroup under a key naming which one it is: Group.DP.name + str(tp_rank)
        # and Group.TP.name + str(dp_rank).
        rank0 = [("Group.DP_AND_TP", _COMM_A), ("DP0", _COMM_B), ("TP0", 0x1111)]
        rank3 = [("Group.DP_AND_TP", 0x2222), ("DP1", 0x3333), ("TP1", 0x4444)]
        self.assertEqual(
            nccl_memory._fingerprint(rank0), nccl_memory._fingerprint(rank3)
        )

    def test_canonical_key_strips_only_the_trailing_digit_run(self) -> None:
        self.assertEqual(nccl_memory._canonical_key("DP0"), "DP")
        # A run, not one character: tp_size >= 10 exists.
        self.assertEqual(nccl_memory._canonical_key("TP12"), "TP")
        self.assertEqual(
            nccl_memory._canonical_key("Group.DP_AND_TP"), "Group.DP_AND_TP"
        )
        # Digits that are not at the end are part of the name, not an index.
        self.assertEqual(nccl_memory._canonical_key("Group.V2.TP"), "Group.V2.TP")

    def test_canonicalisation_cannot_merge_two_families(self) -> None:
        """Stripping must not make two distinct comms look like one.

        It cannot: no family name ends in a digit, and a rank belongs to exactly
        one DP and one TP subgroup, so no rank's list can hold two keys that
        collapse together. If a future group naming scheme breaks that, this fails
        instead of silently shrinking the fingerprint's discriminating power.
        """
        found = [("Group.DP_AND_TP", _COMM_A), ("DP0", _COMM_B), ("TP0", 0x1111)]
        canon = [nccl_memory._canonical_key(k) for k, _ in found]
        self.assertEqual(len(set(canon)), len(canon), canon)

    def test_fingerprint_ignores_pointers_but_not_keys_or_order(self) -> None:
        a = [("Group.DP_AND_TP", _COMM_A), ("Group.TP", _COMM_B)]
        # Comm pointers are rank-dependent, so they must not be in the digest.
        b = [("Group.DP_AND_TP", 0x1234), ("Group.TP", 0x5678)]
        self.assertEqual(nccl_memory._fingerprint(a), nccl_memory._fingerprint(b))
        # Order and membership are exactly what rule (3) is about.
        self.assertNotEqual(
            nccl_memory._fingerprint(a), nccl_memory._fingerprint(list(reversed(a)))
        )
        self.assertNotEqual(
            nccl_memory._fingerprint(a), nccl_memory._fingerprint(a[:1])
        )


class SuspendTest(NcclMemoryTestBase):
    # Case 9
    def test_unusable_capability_is_a_silent_noop(self) -> None:
        fake = self.install_lib(_FakeNccl(), missing=("ncclCommSuspend",))
        self.install_comms([("Group.DP_AND_TP", _COMM_A)])
        self._world = 2

        nccl_memory.suspend_for_sleep(device=None, reason="sleep")  # must not raise

        self.assertEqual(self.suspended_keys(), [])
        self.assertIsNone(nccl_memory._poisoned)
        self.assertFalse(nccl_memory.is_suspended())
        self.assertEqual(fake.suspend_calls, [])
        self.assertEqual(self.store.touches, 0)

    def test_no_communicator_is_a_noop(self) -> None:
        self.install_lib(_FakeNccl())
        self.install_comms([])
        self._world = 2

        nccl_memory.suspend_for_sleep(device=None)

        self.assertEqual(self.suspended_keys(), [])
        self.assertIsNone(nccl_memory._poisoned)
        self.assertEqual(self.store.touches, 0)

    # Case 15
    def test_zero_suspendable_skips_the_vote_entirely(self) -> None:
        fake = self.install_lib(_FakeNccl(suspendable=0))
        self.install_comms([("Group.DP_AND_TP", _COMM_A)])
        self._world = 2  # a vote WOULD veto (no peer) -- prove none happened

        nccl_memory.suspend_for_sleep(device=None)

        self.assertEqual(self.store.touches, 0)
        self.assertEqual(self._store_lookups, 0)
        self.assertIsNone(nccl_memory._vote_disabled)
        self.assertEqual(fake.suspend_calls, [])
        self.assertEqual(self.suspended_keys(), [])
        self.assertIsNone(nccl_memory._poisoned)

    def test_veto_skips_the_collective(self) -> None:
        fake = self.install_lib(_FakeNccl())
        self.install_comms([("Group.DP_AND_TP", _COMM_A)])
        self._world = 2  # peer never arrives -> VETO

        nccl_memory.suspend_for_sleep(device=None)  # must not raise

        self.assertEqual(fake.suspend_calls, [])
        self.assertEqual(self.suspended_keys(), [])
        self.assertIsNone(nccl_memory._poisoned)
        self.assertIsNotNone(nccl_memory._vote_disabled)

    def test_every_no_op_skip_leaves_the_diagnostic_empty(self) -> None:
        """A skip must not claim the hook's failure message.

        The C++ hook appends ``status_text()`` to ``last_error`` whenever the hook
        it is attached to fails, and that hook also covers the cuda_graph/weights
        VMM pause and the level-2 weight reload. Every one of the skips below
        leaves the sleep *succeeding* as far as NCCL is concerned, so a non-empty
        string here would attach an NCCL blob to somebody else's failure and send
        the operator to a healthy subsystem.
        """
        cases = {
            "old runtime": lambda: self.install_lib(
                _FakeNccl(), missing=("ncclCommSuspend",)
            ),
            "nothing suspendable": lambda: self.install_lib(_FakeNccl(suspendable=0)),
            "unhealthy comm, abstain": lambda: self.install_lib(
                _FakeNccl(memstats_rc=4)
            ),
        }
        for name, install in cases.items():
            with self.subTest(skip=name):
                nccl_memory._reset_for_testing()
                install()
                self.install_comms([("Group.DP_AND_TP", _COMM_A)])
                self._world = 2
                nccl_memory.suspend_for_sleep(device=None)  # must not raise
                self.assertEqual(nccl_memory.status_text(), "")

        # And a settled VETO, which is the one skip that does reach the store.
        nccl_memory._reset_for_testing()
        self.install_lib(_FakeNccl())
        self.install_comms([("Group.DP_AND_TP", _COMM_A)])
        self._world = 2
        nccl_memory.suspend_for_sleep(device=None)
        self.assertIsNotNone(nccl_memory._vote_disabled)
        self.assertEqual(nccl_memory.status_text(), "")

    def test_a_fresh_failure_replaces_the_previous_one(self) -> None:
        """The diagnostic is scoped to the transition being reported.

        Otherwise a sleep that failed on NCCL would keep explaining every later
        hook failure, long after the real cause had moved elsewhere.
        """
        self.install_lib(_FakeNccl(suspend_rc=2))
        self.install_comms([("Group.DP_AND_TP", _COMM_A)])
        with self.assertRaises(nccl_memory.NcclMemoryError):
            nccl_memory.suspend_for_sleep(device=None)
        self.assertIn("ncclCommSuspend failed", nccl_memory.status_text())

        # The poison now dominates: it is why THIS transition failed.
        with self.assertRaises(nccl_memory.NcclMemoryError):
            nccl_memory.suspend_for_sleep(device=None)
        self.assertIn("refusing to suspend", nccl_memory.status_text())

    def test_happy_path_suspends_every_comm_with_the_mem_flag(self) -> None:
        fake = self.install_lib(_FakeNccl())
        self.install_comms([("Group.DP_AND_TP", _COMM_A), ("Group.TP", _COMM_B)])

        nccl_memory.suspend_for_sleep(device=None)

        self.assertEqual(
            fake.suspend_calls,
            [
                (_COMM_A, nccl_memory._SUSPEND_MEM),
                (_COMM_B, nccl_memory._SUSPEND_MEM),
            ],
        )
        self.assertEqual(
            self.suspended_keys(),
            [("Group.DP_AND_TP", _COMM_A), ("Group.TP", _COMM_B)],
        )
        self.assertTrue(nccl_memory.is_suspended())
        self.assertIsNone(nccl_memory._poisoned)
        # A successful suspend must leave the diagnostic EMPTY. The C++ hook
        # appends status_text() to every failure of the hook it is attached to,
        # and that hook fails for several non-NCCL reasons (the cuda_graph and
        # weights VMM pause, the level-2 reload), so anything non-empty here
        # would point the operator at a subsystem that is working.
        self.assertEqual(nccl_memory.status_text(), "")

    # Case 10 -- rule (5). The pre-vote probe, and the reason there is no longer
    # any such thing as a benign return code after the vote.
    def test_failing_memstats_probe_abstains_before_voting(self) -> None:
        """A comm that cannot report its stats must not be suspended at all.

        This is the whole point of rule (5). ``ncclCommSuspend`` and
        ``ncclCommMemStats`` share their preamble (``CommCheck`` then
        ``ncclCommEnsureReady``, which folds in *rank-local sticky*
        ``asyncResult``), so a rank whose comm is unhealthy is exactly a rank
        whose memstats fail. Abstaining here is safe -- the peers count
        ``world_size - 1`` arrivals and everyone vetoes -- whereas discovering
        the same condition after the vote is not safe at all, because by then
        the peers are committed to an untimed barrier.
        """
        fake = self.install_lib(_FakeNccl(memstats_rc=4))
        self.install_comms([("Group.DP_AND_TP", _COMM_A)])
        self._world = 2

        nccl_memory.suspend_for_sleep(device=None)  # must not raise

        self.assertEqual(fake.suspend_calls, [])
        # Abstention means NOT voting: this rank's arrival must never become
        # visible, or a peer could count it and head into the barrier.
        self.assertEqual(self.store.touches, 0)
        self.assertEqual(self._store_lookups, 0)
        self.assertEqual(self.suspended_keys(), [])
        self.assertIsNone(nccl_memory._poisoned)
        self.assertIsNone(nccl_memory._vote_disabled)

    def test_already_suspended_comm_abstains_before_voting(self) -> None:
        """Rule (4): a comm NCCL already considers suspended is not ours to touch.

        Somebody else suspended it (or our bookkeeping is wrong). Either way a
        second suspend is ``ncclInvalidUsage`` and, on a non-blocking comm,
        leaves a sticky async error poisoning every later NCCL call -- so this
        has to be caught by the probe, before the vote.
        """
        fake = self.install_lib(_FakeNccl(suspended_flag=1))
        self.install_comms([("Group.DP_AND_TP", _COMM_A)])
        self._world = 2

        nccl_memory.suspend_for_sleep(device=None)

        self.assertEqual(fake.suspend_calls, [])
        self.assertEqual(self.store.touches, 0)
        self.assertEqual(self.suspended_keys(), [])
        self.assertIsNone(nccl_memory._poisoned)

    def test_one_bad_comm_abstains_for_all_of_them(self) -> None:
        """Partial health is not a licence to suspend the healthy subset.

        Suspending only comm B would walk a different communicator list than the
        peers -- rule (3) -- so the abstention has to be all-or-nothing.
        """
        fake = self.install_lib(_FakeNccl(memstats_rc={_COMM_A: 0, _COMM_B: 2}))
        self.install_comms([("Group.DP_AND_TP", _COMM_A), ("Group.TP", _COMM_B)])
        self._world = 2

        nccl_memory.suspend_for_sleep(device=None)

        self.assertEqual(fake.suspend_calls, [])
        self.assertEqual(self.store.touches, 0)

    def test_invalid_usage_after_a_go_vote_is_a_hard_failure(self) -> None:
        """rc 4/5 used to be classified "refused, nothing happened, skip it".

        That was the F1 bug, and it is unsound in the one direction that matters:
        ``ncclCommEnsureReady`` returns ``ncclInvalidArgument`` from *rank-local*
        sticky state, so one rank could "refuse" and skip while its peers got
        ``ncclSuccess`` and blocked in the barrier forever. Once the vote says GO
        the only safe reading of any non-zero code is "this comm is broken":
        poison, raise, and let the operator restart -- a recoverable outcome,
        unlike a wedged instance that cannot even be woken.
        """
        for rc in (4, 5):
            with self.subTest(rc=rc):
                nccl_memory._reset_for_testing()
                fake = self.install_lib(_FakeNccl(suspend_rc=rc))
                self.install_comms([("Group.DP_AND_TP", _COMM_A)])

                with self.assertRaises(nccl_memory.NcclMemoryError) as cm:
                    nccl_memory.suspend_for_sleep(device=None)

                self.assertEqual(len(fake.suspend_calls), 1)  # rule (2): no retry
                self.assertIn(f"rc={rc}", str(cm.exception))
                self.assertIsNotNone(nccl_memory._poisoned)
                self.assertEqual(self.suspended_keys(), [])

    def test_probe_reads_every_comm_before_any_suspend_call(self) -> None:
        """Ordering: the whole probe must precede the whole collective.

        Interleaving probe and suspend per-comm would reintroduce the bug --
        comm B's refusal would be discovered only after comm A had already been
        suspended, i.e. after this rank was committed.
        """
        fake = self.install_lib(_FakeNccl())
        self.install_comms([("Group.DP_AND_TP", _COMM_A), ("Group.TP", _COMM_B)])

        nccl_memory.suspend_for_sleep(device=None)

        kinds = [kind for kind, _ in fake.calls]
        first_suspend = kinds.index("suspend")
        self.assertNotIn("suspend", kinds[:first_suspend])
        probed_before = {
            c for kind, c in fake.calls[:first_suspend] if kind == "memstats"
        }
        self.assertEqual(probed_before, {_COMM_A, _COMM_B})
        self.assertEqual(len(fake.suspend_calls), 2)

    # Cases 11 + 12 -- rules (2) and (3).
    def test_hard_failure_walks_the_whole_list_once_then_raises(self) -> None:
        fake = self.install_lib(_FakeNccl(suspend_rc={_COMM_A: 2, _COMM_B: 0}))
        self.install_comms([("Group.DP_AND_TP", _COMM_A), ("Group.TP", _COMM_B)])

        with self.assertRaises(nccl_memory.NcclMemoryError) as cm:
            nccl_memory.suspend_for_sleep(device=None)

        # Rule (3): the failure on comm A must NOT abandon comm B -- peers that
        # succeeded on A are already inside B's barrier.
        self.assertEqual([c for c, _ in fake.suspend_calls], [_COMM_A, _COMM_B])
        # Rule (2): exactly one call per comm. A retry would re-enter a barrier
        # its peers have left and block forever under the transition lock.
        self.assertEqual(len(fake.suspend_calls), 2)
        self.assertIsNotNone(nccl_memory._poisoned)
        self.assertIn("Group.DP_AND_TP(rc=2)", nccl_memory._poisoned)
        self.assertIn("Group.DP_AND_TP(rc=2)", str(cm.exception))
        # The diagnostic the C++ hook reports must name the NCCL stage and the
        # offending comm, on one line, so the operator does not have to correlate
        # a bare "releaseRestorableGpuMemory failed" against the rank logs.
        status = nccl_memory.status_text()
        self.assertIn("ncclCommSuspend failed", status)
        self.assertIn("Group.DP_AND_TP(rc=2)", status)
        self.assertNotIn("\n", status)

    def test_a_raising_suspend_still_walks_the_whole_list_and_poisons(self) -> None:
        """Rule (3) has to hold for an exception, not just for a non-zero rc.

        This is the case a plain ``for``-loop gets wrong. ``ctypes`` raises
        ``ArgumentError`` on a signature mismatch, so an NCCL header change turns
        comm A's call into an exception; without a per-comm guard it would unwind
        the loop, never call comm B, and never set ``_poisoned``. Comm B's peers --
        which succeeded on A and moved on -- would then sit in B's untimed barrier
        forever, and the next ``/sleep`` would happily suspend again on top of it.
        """
        boom = ctypes.ArgumentError("simulated ABI drift")
        fake = self.install_lib(_FakeNccl(suspend_rc={_COMM_A: boom, _COMM_B: 0}))
        self.install_comms([("Group.DP_AND_TP", _COMM_A), ("Group.TP", _COMM_B)])

        with self.assertRaises(nccl_memory.NcclMemoryError) as cm:
            nccl_memory.suspend_for_sleep(device=None)

        # Comm B was still attempted, exactly once, and the raise did not escape
        # as itself: it arrives as the NcclMemoryError that fails the sleep.
        self.assertEqual([c for c, _ in fake.suspend_calls], [_COMM_A, _COMM_B])
        self.assertIn("Group.DP_AND_TP(exc=", str(cm.exception))
        self.assertIn("simulated ABI drift", str(cm.exception))
        self.assertIsNotNone(nccl_memory._poisoned)
        # B succeeded, so it is tracked as suspended and must be resumed later.
        self.assertEqual(self.suspended_keys(), [("Group.TP", _COMM_B)])
        self.assertIn("Group.DP_AND_TP(exc=", nccl_memory.status_text())

    def test_a_raising_resume_still_walks_the_whole_list_and_poisons(self) -> None:
        """Same guarantee in the wake direction, where there is no safe skip."""
        boom = ctypes.ArgumentError("simulated ABI drift")
        fake = _FakeNccl(resume_rc={_COMM_B: boom, _COMM_A: 0})
        found = [("Group.DP_AND_TP", _COMM_A), ("Group.TP", _COMM_B)]
        self.install_lib(fake)
        self.install_comms(found)
        nccl_memory.suspend_for_sleep(device=None)

        with self.assertRaises(nccl_memory.NcclMemoryError) as cm:
            nccl_memory.resume_after_wake(device=None)

        # Reverse order: B first (it raises), then A must still be attempted.
        self.assertEqual(fake.resume_calls, [_COMM_B, _COMM_A])
        self.assertIn("Group.TP(exc=", str(cm.exception))
        self.assertIsNotNone(nccl_memory._poisoned)
        # Only the comm that failed stays on the suspended list: is_suspended()
        # must not claim False while a communicator is genuinely unmapped.
        self.assertEqual(self.suspended_keys(), [("Group.TP", _COMM_B)])

    def test_poisoned_refuses_a_later_suspend(self) -> None:
        fake = self.install_lib(_FakeNccl(suspend_rc=2))
        self.install_comms([("Group.DP_AND_TP", _COMM_A)])
        with self.assertRaises(nccl_memory.NcclMemoryError):
            nccl_memory.suspend_for_sleep(device=None)
        calls_after_first = len(fake.suspend_calls)

        with self.assertRaises(nccl_memory.NcclMemoryError) as cm:
            nccl_memory.suspend_for_sleep(device=None)
        self.assertIn("previously poisoned", str(cm.exception))
        self.assertEqual(len(fake.suspend_calls), calls_after_first)

    # Case 13 -- rule (4).
    def test_double_suspend_raises_without_calling_nccl_again(self) -> None:
        fake = self.install_lib(_FakeNccl())
        self.install_comms([("Group.DP_AND_TP", _COMM_A)])
        nccl_memory.suspend_for_sleep(device=None)
        self.assertEqual(len(fake.suspend_calls), 1)

        with self.assertRaises(nccl_memory.NcclMemoryError) as cm:
            nccl_memory.suspend_for_sleep(device=None)
        self.assertIn("already suspended", str(cm.exception))
        self.assertEqual(len(fake.suspend_calls), 1)


class ResumeTest(NcclMemoryTestBase):
    def _suspend_ok(
        self,
        fake: _FakeNccl,
        found: List[Tuple[str, int]],
        ptr_now: Optional[Dict[int, int]] = None,
        ptr_error: Optional[Dict[int, BaseException]] = None,
    ) -> Dict[int, object]:
        """Reach a suspended state, optionally scripting the wake-path re-read.

        ``ptr_now`` / ``ptr_error`` are installed up front rather than by a second
        ``install_comms`` call, because a second call would mint new ProcessGroup
        objects: the recorded ones would then be unknown to the lookup and every
        such test would collapse into the same "unanswerable" branch instead of
        the one it means to exercise.
        """
        self.install_lib(fake)
        groups = self.install_comms(found, ptr_now=ptr_now, ptr_error=ptr_error)
        nccl_memory.suspend_for_sleep(device=None)
        self.assertEqual(len(nccl_memory._suspended), len(found))
        return groups

    # Case 14a
    def test_resume_without_suspend_is_a_noop(self) -> None:
        fake = self.install_lib(_FakeNccl())
        self.install_comms([("Group.DP_AND_TP", _COMM_A)])
        self._world = 2  # a vote would VETO; prove none is attempted

        nccl_memory.resume_after_wake(device=None)  # must not raise

        self.assertEqual(fake.resume_calls, [])
        self.assertEqual(self.store.touches, 0)
        self.assertIsNone(nccl_memory._poisoned)

    # Case 14b
    def test_resume_clears_suspended_set(self) -> None:
        fake = _FakeNccl()
        found = [("Group.DP_AND_TP", _COMM_A), ("Group.TP", _COMM_B)]
        self._suspend_ok(fake, found)

        nccl_memory.resume_after_wake(device=None)

        # Reverse suspend order, per the module's unwind rule.
        self.assertEqual(fake.resume_calls, [_COMM_B, _COMM_A])
        self.assertEqual(self.suspended_keys(), [])
        self.assertFalse(nccl_memory.is_suspended())
        self.assertIsNone(nccl_memory._poisoned)
        self.assertEqual(nccl_memory.status_text(), "")

    # Case 14c
    def test_resume_failure_raises_and_poisons(self) -> None:
        fake = _FakeNccl(resume_rc={_COMM_A: 3, _COMM_B: 0})
        found = [("Group.DP_AND_TP", _COMM_A), ("Group.TP", _COMM_B)]
        self._suspend_ok(fake, found)

        with self.assertRaises(nccl_memory.NcclMemoryError) as cm:
            nccl_memory.resume_after_wake(device=None)

        # Rules (2)/(3) again: whole list, once each, no retry.
        self.assertEqual(fake.resume_calls, [_COMM_B, _COMM_A])
        self.assertIsNotNone(nccl_memory._poisoned)
        self.assertIn("ncclCommResume failed", str(cm.exception))

    # The F6 bug.
    def test_failed_resume_keeps_reporting_itself_suspended(self) -> None:
        """``is_suspended()`` must not lie about a comm that is still unmapped.

        Clearing the set unconditionally made ``is_suspended()`` return False
        while comm A's virtual addresses were in fact still unmapped -- a lie told
        at exactly the moment someone is debugging why collectives fault. The
        comms that DID resume drop out; the one that failed stays.
        """
        fake = _FakeNccl(resume_rc={_COMM_A: 3, _COMM_B: 0})
        found = [("Group.DP_AND_TP", _COMM_A), ("Group.TP", _COMM_B)]
        self._suspend_ok(fake, found)

        with self.assertRaises(nccl_memory.NcclMemoryError):
            nccl_memory.resume_after_wake(device=None)

        self.assertTrue(nccl_memory.is_suspended())
        self.assertEqual(self.suspended_keys(), [("Group.DP_AND_TP", _COMM_A)])

    def test_no_retry_after_a_failed_resume(self) -> None:
        """Rule (2): the second wake must refuse, not re-issue ncclCommResume.

        The poison check fires first. A retry cannot be shown to be pre-barrier
        from the return code, and two of the pre-barrier failure sites leave the
        VA mapped with the entry still ``Released``, so a retry would double-map
        and leak a handle.
        """
        fake = _FakeNccl(resume_rc=3)
        self._suspend_ok(fake, [("Group.DP_AND_TP", _COMM_A)])
        with self.assertRaises(nccl_memory.NcclMemoryError):
            nccl_memory.resume_after_wake(device=None)
        calls_after_first = len(fake.resume_calls)

        with self.assertRaises(nccl_memory.NcclMemoryError) as cm:
            nccl_memory.resume_after_wake(device=None)
        self.assertIn("poisoned at sleep", str(cm.exception))
        self.assertEqual(len(fake.resume_calls), calls_after_first)

    def test_resume_flags_a_comm_that_still_reports_suspended(self) -> None:
        """Rule (6): rc==0 is not proof. The cross-check must still look.

        NCCL's peer re-import loop only ``WARN``s and continues, so a resume can
        return ``ncclSuccess`` with buffers left unmapped. This asserts the
        diagnostic reads the flag back and, per its contract, does not turn a
        committed wake into a failure.
        """
        fake = _FakeNccl()
        self._suspend_ok(fake, [("Group.DP_AND_TP", _COMM_A)])
        fake.suspended_flag = 1  # NCCL still thinks it is suspended

        with self.assertLogs(level="ERROR") as logs:
            nccl_memory.resume_after_wake(device=None)  # must NOT raise

        self.assertTrue(
            any("still reports suspended" in m for m in logs.output), logs.output
        )

    # Rule (5) on the resume side. Suspend has always probed before voting; resume
    # used to go straight from _decide into ncclCommResume, even though resume runs
    # the SAME CommCheck/ncclCommEnsureReady preamble and so can return before its
    # bootstrapBarrier -- one rank bailing while its N-1 peers block there forever,
    # and on this side that is strictly worse because the memory is already unmapped
    # and /wake_up cannot be re-driven.
    def test_failing_memstats_probe_before_resume_raises_without_voting(self) -> None:
        """A comm whose preamble now fails must not be handed to ncclCommResume.

        Abstaining is not available here (there is no safe skip on the wake path),
        so the honest move is to raise before publishing an arrival: the peers see a
        missing vote, veto, and raise too, which is N clean errors instead of one
        error and N-1 permanent hangs.
        """
        fake = _FakeNccl()
        self._suspend_ok(fake, [("Group.DP_AND_TP", _COMM_A)])
        # Exactly what a sticky rank-local comm->asyncResult looks like from here:
        # memstats and resume share the preamble that fails.
        fake.memstats_rc = 7
        self._world = 2
        touches_before = self.store.touches

        with self.assertRaises(nccl_memory.NcclMemoryError) as cm:
            nccl_memory.resume_after_wake(device=None)

        self.assertEqual(fake.resume_calls, [])
        # WITHOUT voting: no arrival may become visible, or a peer could count it
        # and head into a barrier this rank never enters.
        self.assertEqual(self.store.touches, touches_before)
        self.assertEqual(self._store_lookups, 0)
        self.assertIn("pre-resume probe failed", str(cm.exception))
        self.assertIsNotNone(nccl_memory._poisoned)
        # Still unmapped, so is_suspended() must keep saying so.
        self.assertTrue(nccl_memory.is_suspended())

    def test_comm_nccl_no_longer_reports_suspended_raises_without_voting(self) -> None:
        """Rule (4) mirrored: resume-without-suspend is ncclInvalidUsage.

        If NCCL disagrees with our bookkeeping the resume is a hard error that
        leaves a sticky async result poisoning every later call, so it has to be
        caught by the probe rather than issued and inspected.
        """
        fake = _FakeNccl()
        self._suspend_ok(fake, [("Group.DP_AND_TP", _COMM_A)])
        fake.suspended_flag = 0  # NCCL: "that communicator is active"

        with self.assertRaises(nccl_memory.NcclMemoryError) as cm:
            nccl_memory.resume_after_wake(device=None)

        self.assertEqual(fake.resume_calls, [])
        self.assertEqual(self.store.touches, 0)
        self.assertIn("suspended=0", str(cm.exception))
        self.assertIsNotNone(nccl_memory._poisoned)

    def test_resume_probe_reads_every_comm_before_any_resume_call(self) -> None:
        """The whole probe must precede the whole collective, as on the sleep side.

        Interleaving per-comm would rediscover comm A's refusal only after comm B
        had been resumed, i.e. after this rank was already committed.
        """
        fake = _FakeNccl()
        found = [("Group.DP_AND_TP", _COMM_A), ("Group.TP", _COMM_B)]
        self._suspend_ok(fake, found)
        fake.calls.clear()

        nccl_memory.resume_after_wake(device=None)

        kinds = [kind for kind, _ in fake.calls]
        first_resume = kinds.index("resume")
        probed_before = {
            c for kind, c in fake.calls[:first_resume] if kind == "memstats"
        }
        self.assertEqual(probed_before, {_COMM_A, _COMM_B})
        self.assertEqual(fake.resume_calls, [_COMM_B, _COMM_A])

    def test_resume_vote_veto_raises_because_memory_is_unmapped(self) -> None:
        fake = _FakeNccl()
        self._suspend_ok(fake, [("Group.DP_AND_TP", _COMM_A)])
        # Now a peer dies before the wake vote.
        self._world = 2

        with self.assertRaises(nccl_memory.NcclMemoryError) as cm:
            nccl_memory.resume_after_wake(device=None)

        self.assertEqual(fake.resume_calls, [])
        self.assertIn("did not reach GO", str(cm.exception))
        self.assertIsNotNone(nccl_memory._poisoned)

    def test_partial_suspend_failure_makes_wake_refuse(self) -> None:
        """Comm B suspended, comm A failed: the wake must refuse to build on it."""
        fake = _FakeNccl(suspend_rc={_COMM_A: 2, _COMM_B: 0})
        self.install_lib(fake)
        self.install_comms([("Group.DP_AND_TP", _COMM_A), ("Group.TP", _COMM_B)])
        with self.assertRaises(nccl_memory.NcclMemoryError):
            nccl_memory.suspend_for_sleep(device=None)
        self.assertEqual(self.suspended_keys(), [("Group.TP", _COMM_B)])

        with self.assertRaises(nccl_memory.NcclMemoryError) as cm:
            nccl_memory.resume_after_wake(device=None)
        self.assertIn("refusing to wake", str(cm.exception))
        self.assertEqual(fake.resume_calls, [])

    def test_wake_after_total_suspend_failure_refuses(self) -> None:
        """The poisoned guard must fire even with an EMPTY suspended set.

        Regression test for a real ordering bug: ``resume_after_wake`` used to
        test ``if not _suspended: return`` BEFORE ``_poisoned``. Suspend only
        records a comm on rc==0, so when *every* communicator failed hard nothing
        was appended and the guard -- with its "restart the instance" message --
        was unreachable, letting a caller wake a process whose communicators are
        half-released past their barrier. Contrast
        :meth:`test_partial_suspend_failure_makes_wake_refuse`, where one comm did
        land in ``_suspended`` and the guard fired even before the fix; that
        asymmetry is what isolated the ordering as the cause.
        """
        fake = _FakeNccl(suspend_rc=2)
        self.install_lib(fake)
        self.install_comms([("Group.DP_AND_TP", _COMM_A), ("Group.TP", _COMM_B)])
        with self.assertRaises(nccl_memory.NcclMemoryError):
            nccl_memory.suspend_for_sleep(device=None)
        self.assertIsNotNone(nccl_memory._poisoned)
        self.assertEqual(self.suspended_keys(), [])

        with self.assertRaises(nccl_memory.NcclMemoryError):
            nccl_memory.resume_after_wake(device=None)
        # Refused before touching NCCL: resuming a comm whose suspend failed
        # mid-release would be a collective its peers never enter.
        self.assertEqual(fake.resume_calls, [])

    # Case 14g -- the one residual dangling case the retained ProcessGroup
    # reference cannot rule out: the group survived but its communicator was
    # aborted and rebuilt underneath it, so the recorded pointer is dangling and
    # resuming it would be a use-after-free.
    def test_replaced_communicator_is_not_resumed(self) -> None:
        fake = _FakeNccl()
        self._suspend_ok(
            fake,
            [("Group.DP_AND_TP", _COMM_A)],
            ptr_now={_COMM_A: 0xDEAD0000},
        )

        with self.assertRaises(nccl_memory.NcclMemoryError) as cm:
            nccl_memory.resume_after_wake(device=None)

        self.assertEqual(fake.resume_calls, [])
        message = str(cm.exception)
        self.assertIn("were replaced while suspended", message)
        # Both pointers, so the log says which comm and what it became.
        self.assertIn(f"recorded=0x{_COMM_A:x}", message)
        self.assertIn("now=0xdead0000", message)
        self.assertIsNotNone(nccl_memory._poisoned)

    # Case 14h -- and the two ways the cross-check can fail to *answer*. Neither is
    # evidence of anything: only a pointer that reads back DIFFERENT is. Treating
    # silence as poison here is what used to turn a torch rename, or a hook thread
    # on the wrong device, into a mandatory instance restart on every wake.
    def test_throwing_pointer_reread_resumes_on_the_recorded_pointer(self) -> None:
        fake = _FakeNccl()
        self._suspend_ok(
            fake,
            [("Group.DP_AND_TP", _COMM_A)],
            ptr_error={_COMM_A: RuntimeError("private API renamed")},
        )

        nccl_memory.resume_after_wake(device=None)  # must not raise

        self.assertEqual(fake.resume_calls, [_COMM_A])
        self.assertEqual(self.suspended_keys(), [])
        self.assertIsNone(nccl_memory._poisoned)

    def test_null_pointer_reread_resumes_on_the_recorded_pointer(self) -> None:
        # 0 is "no entry for this thread's current device", not "destroyed".
        fake = _FakeNccl()
        self._suspend_ok(fake, [("Group.DP_AND_TP", _COMM_A)], ptr_now={_COMM_A: 0})

        nccl_memory.resume_after_wake(device=None)  # must not raise

        self.assertEqual(fake.resume_calls, [_COMM_A])
        self.assertEqual(self.suspended_keys(), [])
        self.assertIsNone(nccl_memory._poisoned)

    # Case 14i -- what makes the above safe. _suspended holds the owning
    # ProcessGroup, not just an integer, so the communicator cannot be collected
    # out from under the recorded pointer while this process sleeps.
    def test_suspended_state_retains_the_owning_process_group(self) -> None:
        groups = self._suspend_ok(_FakeNccl(), [("Group.DP_AND_TP", _COMM_A)])

        self.assertEqual(
            nccl_memory._suspended, [("Group.DP_AND_TP", _COMM_A, groups[_COMM_A])]
        )


class _FakeBackend:
    def __init__(self, comm: int) -> None:
        self._comm = comm

    def _comm_ptr(self) -> int:
        return self._comm


class ProcessGroupGloo:
    """What ``_get_backend(cuda)`` really returns for a gloo group.

    Named after the real class because :func:`backend._is_nccl_backend` reads the
    class name. The point of this fake is that it does NOT raise and does NOT have
    ``_comm_ptr``: the earlier fake modelled gloo as a raise, which is what let a
    false ERROR on every rank of every sleep ship past the whole test file.
    """


class ProcessGroupNCCL:
    """An NCCL backend whose ``_comm_ptr`` accessor is gone -- a real rename."""


class _FakePg:
    """A ProcessGroup stand-in.

    ``comm=None`` models a group with no CUDA backend registered at all (a raise);
    pass ``backend=`` directly for the cases where the backend object itself is
    what matters.
    """

    def __init__(self, comm: Optional[int], backend: object = None) -> None:
        self._comm = comm
        self._backend = backend

    def _get_backend(self, device: object) -> object:
        if self._backend is not None:
            return self._backend
        if self._comm is None:
            raise RuntimeError("no backend registered for cuda")
        return _FakeBackend(self._comm)


class CommsTest(NcclMemoryTestBase):
    """Enumeration. Getting this wrong is silent, not loud."""

    def install_group_map(self, mapping: dict) -> None:
        """Install a stub ``collective_torch`` exposing exactly ``mapping``.

        A stub rather than the real module, and not for speed: ``comms()`` reads
        one attribute off it, so importing the real one would add zero coverage
        while pulling ``torch.distributed``'s process-group machinery -- and a
        bazel dep on ``//rtp_llm/models_py:models`` -- into a CPU-only unit test
        of dictionary iteration. Patching ``sys.modules`` at the *package* name is
        what the lazy ``from ... import collective_torch`` inside ``comms()``
        actually resolves against.
        """
        pkg = types.ModuleType("rtp_llm.models_py.distributed")
        stub = types.ModuleType("rtp_llm.models_py.distributed.collective_torch")
        stub._group_map = mapping  # type: ignore[attr-defined]
        pkg.collective_torch = stub  # type: ignore[attr-defined]
        p = mock.patch.dict(
            sys.modules,
            {
                "rtp_llm.models_py.distributed": pkg,
                "rtp_llm.models_py.distributed.collective_torch": stub,
            },
        )
        p.start()
        self.addCleanup(p.stop)

    def test_dedupes_by_pointer_and_drops_non_nccl_groups(self) -> None:
        """One ProcessGroup registered under several keys is ONE communicator.

        Deduplication by raw pointer is what makes rule (4) enforceable:
        suspending the same comm twice via two aliases is ``ncclInvalidUsage``.
        The gloo SLEEP_QUIESCE group has no communicator at all and must vanish
        rather than raise.
        """
        self.install_group_map(
            {
                "Group.DP_AND_TP": _FakePg(_COMM_A),
                "DP0": _FakePg(_COMM_A),  # same PG under a second key
                "TP0": _FakePg(_COMM_B),
                # gloo, as it really presents itself: a backend object, no raise
                "Group.SLEEP_QUIESCE": _FakePg(None, backend=ProcessGroupGloo()),
                "Group.NO_CUDA_BACKEND": _FakePg(None),  # _get_backend() raises
                "Group.BROKEN": _FakePg(0),  # getCommPtr() returned nullptr
            }
        )
        self.assertEqual(
            nccl_memory.comms(),
            [("Group.DP_AND_TP", _COMM_A), ("TP0", _COMM_B)],
        )

    def test_gloo_group_is_dropped_without_crying_rename(self) -> None:
        """A gloo sibling is routine, so it must not log at ERROR.

        Regression test for a shipped-and-measured defect: ``_get_backend(cuda)``
        hands back ``ProcessGroupGloo`` rather than raising, which put the routine
        gloo group down the "private API rename" branch. Every sleep and every wake
        then logged an ERROR claiming the release would "save nothing" -- on a run
        where all 24 suspend/resume calls in fact succeeded. False ERRORs are worse
        than no logging: they train whoever is on call to ignore the real one.
        """
        self.install_group_map(
            {"Group.SLEEP_QUIESCE": _FakePg(None, backend=ProcessGroupGloo())}
        )

        with self.assertLogs(level="ERROR") as captured:
            logging.error("[probe] assertLogs needs at least one record")
            self.assertEqual(nccl_memory.comms(), [])
        self.assertEqual(len(captured.records), 1, captured.output)

    def test_throwing_accessor_is_loud_regardless_of_backend_name(self) -> None:
        """Having the accessor but not being able to answer is never routine.

        Unlike a missing accessor, this needs no class-name guess: only an NCCL
        backend has ``_comm_ptr`` to begin with, so the complaint is unconditional.
        """

        class _Throwing:
            def _comm_ptr(self) -> int:
                raise RuntimeError("comm map is locked")

        self.install_group_map({"Group.DP_AND_TP": _FakePg(None, backend=_Throwing())})

        with self.assertLogs(level="ERROR") as captured:
            self.assertEqual(nccl_memory.comms(), [])
        self.assertIn("comm map is locked", "\n".join(captured.output))

    def test_missing_accessor_on_an_nccl_backend_is_still_loud(self) -> None:
        """The flip side: dropping the ERROR entirely would hide a real rename.

        An NCCL backend that cannot yield a pointer silently becomes a permanent
        no-op after a torch upgrade, so this one has to stay at ERROR.
        """
        self.install_group_map(
            {"Group.DP_AND_TP": _FakePg(None, backend=ProcessGroupNCCL())}
        )

        with self.assertLogs(level="ERROR") as captured:
            self.assertEqual(nccl_memory.comms(), [])
        self.assertIn("private API rename", "\n".join(captured.output))

    def test_enumeration_hands_back_the_owning_process_group(self) -> None:
        """The third column is what keeps a suspended communicator alive.

        ``comms()`` drops it, so nothing else in this file would notice if the scan
        stopped returning it -- and the loss would be invisible until a
        ProcessGroup was collected during a real sleep.
        """
        pg = _FakePg(_COMM_A)
        self.install_group_map({"Group.DP_AND_TP": pg})

        self.assertEqual(
            backend.enumerate_process_group_comms(),
            [("Group.DP_AND_TP", _COMM_A, pg)],
        )
        # And the same pointer is readable one group at a time, which is how the
        # wake path re-checks without re-running the whole scan.
        self.assertEqual(backend.comm_ptr_for_group(pg), _COMM_A)

    def test_comm_ptr_for_group_reports_zero_rather_than_guessing(self) -> None:
        # 0 is "no entry for the current device"; the caller decides what that is
        # worth, so this must not be smoothed over into a raise or a stale value.
        self.assertEqual(backend.comm_ptr_for_group(_FakePg(0)), 0)

    def test_device_is_pinned_around_the_lookup(self) -> None:
        """``getCommPtr()`` is keyed on ``current_device()``, not the PG's device.

        So a hook thread whose current device was never set enumerates ZERO
        communicators -- silently, no throw -- on every rank but rank 0. That is
        not a crash, it is an asymmetry: sleep logs "nothing to suspend" on some
        ranks, the vote never reaches quorum, and the feature is off for the life
        of the process. Hence the device context.
        """
        entered = []
        requested = []

        class _Ctx:
            def __enter__(self_inner):
                entered.append("enter")

            def __exit__(self_inner, *a):
                entered.append("exit")
                return False

        def _device(d):
            # Capture the ordinal, not just the fact that a context was entered:
            # pinning the WRONG device is the failure mode this guards, and it
            # looks identical to pinning the right one from the enter/exit log.
            requested.append(d)
            return _Ctx()

        self.install_group_map({"Group.DP_AND_TP": _FakePg(_COMM_A)})
        with mock.patch("torch.cuda.device", _device):
            self.assertEqual(
                nccl_memory.comms(device=3), [("Group.DP_AND_TP", _COMM_A)]
            )
        self.assertEqual(entered, ["enter", "exit"])
        self.assertEqual(requested, [3])

        # device=None must NOT touch torch.cuda.device: comms() is reachable
        # from paths that must never initialise CUDA as a side effect.
        def _boom(d):
            raise AssertionError("torch.cuda.device must not be entered for None")

        with mock.patch("torch.cuda.device", _boom):
            self.assertEqual(nccl_memory.comms(), [("Group.DP_AND_TP", _COMM_A)])

    def test_both_directions_forward_the_device_to_the_lookup(self) -> None:
        """The device only helps if it actually reaches the enumeration.

        Everything above proves ``comms(device=N)`` pins N. This proves the two
        production entry points pass their device through rather than defaulting
        to None -- which would silently reinstate the wrong-device asymmetry on
        every rank whose hook thread never called ``set_device``, and would look
        exactly like "this deployment has no communicators".
        """
        fake = self.install_lib(_FakeNccl())
        self.install_comms([("Group.DP_AND_TP", _COMM_A)])

        nccl_memory.suspend_for_sleep(device=5)
        self.assertEqual(fake.suspend_calls, [(_COMM_A, nccl_memory._SUSPEND_MEM)])
        nccl_memory.resume_after_wake(device=5)
        self.assertEqual(fake.resume_calls, [_COMM_A])

        # suspend enumerates once; resume enumerates once more for the post-resume
        # evidence check and does one per-group pointer cross-check. Every one of
        # them must be device-5, not None.
        self.assertEqual(set(self.enumerate_devices), {5})
        self.assertGreaterEqual(len(self.enumerate_devices), 2)
        self.assertEqual(self.ptr_devices, [5])


class MemStatsTest(NcclMemoryTestBase):
    def test_stat_returns_minus_one_on_error(self) -> None:
        """A non-zero rc must never leak the value NCCL left in the out-param.

        -1 is what the pre-vote probe reads as "unusable, abstain". Returning 999
        here would look like a healthy 999-byte communicator instead.
        """

        class Failing:
            def ncclCommMemStats(self, comm, stat, out):
                out._obj.value = 999
                return 7

        self.assertEqual(backend.NcclApi(Failing()).stat(_COMM_A, 0), -1)

    def test_stat_returns_minus_one_when_the_call_raises(self) -> None:
        """The probe is a diagnostic and must degrade, not propagate.

        A ctypes.ArgumentError from an ABI drift would otherwise escape the
        pre-vote probe as an exception, and the pre-vote path is exactly where a
        clean abstain is available.
        """

        class Exploding:
            def ncclCommMemStats(self, comm, stat, out):
                raise ctypes.ArgumentError("simulated ABI drift")

        self.assertEqual(backend.NcclApi(Exploding()).stat(_COMM_A, 0), -1)

    def test_stat_returns_minus_one_when_the_symbol_is_absent(self) -> None:
        self.assertEqual(backend.NcclApi(types.SimpleNamespace()).stat(_COMM_A, 0), -1)
        self.assertEqual(backend.NcclApi(None).stat(_COMM_A, 0), -1)

    def test_capability_respects_disable_env(self) -> None:
        self.install_lib(_FakeNccl())
        with mock.patch.dict("os.environ", {"NCCL_DISABLE_MEM_MANAGER": "1"}):
            usable, why = nccl_memory.capability()
        self.assertFalse(usable)
        self.assertIn("NCCL_DISABLE_MEM_MANAGER=1", why)
        usable, why = nccl_memory.capability()
        self.assertTrue(usable)

    # Rule (8): on a non-blocking communicator both calls return ncclInProgress,
    # which is a SUCCESS needing polling. This module refuses that configuration
    # instead, and the refusal has to be recognised HERE -- before the vote -- or
    # rule (5) is violated: an rc=7 discovered mid-loop makes this rank raise while
    # its peers sit in an untimed barrier.
    def test_capability_refuses_non_blocking_communicators(self) -> None:
        for env in (
            {"TORCH_NCCL_USE_COMM_NONBLOCKING": "1"},
            {"NCCL_COMM_BLOCKING": "0"},
        ):
            with self.subTest(**env):
                self.install_lib(_FakeNccl())
                with mock.patch.dict("os.environ", env):
                    usable, why = nccl_memory.capability()
                self.assertFalse(usable)
                self.assertIn(next(iter(env)), why)
                self.assertIn("ncclInProgress", why)

    def test_capability_ignores_the_blocking_settings_of_those_switches(self) -> None:
        """Only the non-blocking *value* counts, and only when set.

        The switches are asymmetric -- torch's is opt-in at 1, NCCL's is opt-out at
        0 -- so a gate that merely tested for presence would disable the feature on
        a config that explicitly asked for blocking communicators.
        """
        self.install_lib(_FakeNccl())
        for env in (
            {"TORCH_NCCL_USE_COMM_NONBLOCKING": "0"},
            {"NCCL_COMM_BLOCKING": "1"},
        ):
            with self.subTest(**env):
                with mock.patch.dict("os.environ", env):
                    usable, _ = nccl_memory.capability()
                self.assertTrue(usable)

    def test_in_progress_rc_is_named_rather_than_reported_bare(self) -> None:
        """rc=7 means the opposite of what this module does with it.

        ``nccl.h`` calls ncclInProgress a success, so a bare "rc=7" in the sleep
        log would send an operator looking for a failure that the header says did
        not happen. The poison message has to carry the diagnosis.
        """
        fake = _FakeNccl(suspend_rc={_COMM_A: nccl_memory._NCCL_IN_PROGRESS})
        self.install_lib(fake)
        self.install_comms([("Group.DP_AND_TP", _COMM_A)])

        with self.assertRaises(nccl_memory.NcclMemoryError) as cm:
            nccl_memory.suspend_for_sleep(device=None)

        message = str(cm.exception)
        self.assertIn("rc=7 ncclInProgress", message)
        self.assertIn("needs polling", message)
        self.assertIn("pg_options.config.blocking", message)


class BackendLoaderTest(unittest.TestCase):
    """The loader itself, which every other test stubs out.

    Without this the ``dlopen``, the version parse and the ``argtypes``
    declarations never execute at all -- so the one thing that could catch a real
    ABI or SONAME drift would be the production sleep path.
    """

    def tearDown(self) -> None:
        backend.reset_for_testing()

    def test_api_is_cached_and_resettable(self) -> None:
        backend.reset_for_testing()
        first = backend.api()
        self.assertIs(backend.api(), first)  # loaded once, not per call
        backend.reset_for_testing()
        self.assertIsNot(backend.api(), first)

    def test_unavailable_reason_is_empty_exactly_when_usable(self) -> None:
        for library, usable in (
            (None, False),
            (types.SimpleNamespace(), False),
            (_FakeNccl().as_lib(), True),
        ):
            with self.subTest(library=type(library).__name__):
                api = backend.NcclApi(library, "2.31.2")
                self.assertEqual(api.usable, usable)
                self.assertEqual(api.unavailable_reason == "", usable)

    def test_missing_symbols_names_every_absent_symbol(self) -> None:
        api = backend.NcclApi(_FakeNccl().as_lib(missing=("ncclCommResume",)))
        self.assertEqual(api.missing_symbols, ["ncclCommResume"])
        self.assertFalse(api.usable)
        self.assertCountEqual(backend.NcclApi(None).missing_symbols, _ALL_SYMS)

    def test_real_library_loads_and_reports_a_parseable_version(self) -> None:
        """Against the runtime's actual libnccl, when there is one.

        Skipped rather than failed when NCCL is absent, because this module is
        meant to be inert on such a runtime. What it does catch when NCCL *is*
        present: a SONAME that no longer resolves, and a version that no longer
        parses -- both of which would otherwise surface as a mystery "collective
        memory release unavailable" in production.
        """
        backend.reset_for_testing()
        api = backend.api()
        if api.library is None:
            self.skipTest(f"no libnccl.so.2 in this image: {api.version}")
        # major.minor.patch, all numeric. A failed query lands here as prose.
        self.assertRegex(api.version, r"^\d+\.\d+\.\d+$")
        major, minor, _ = (int(p) for p in api.version.split("."))
        if (major, minor) < (2, 29):
            self.assertFalse(api.usable)
            self.assertIn("lacks", api.unavailable_reason)
            self.skipTest(f"NCCL {api.version} predates the suspend API")
        # 2.29.7+ must export all three, or the version pin is wrong.
        self.assertEqual(api.missing_symbols, [])
        self.assertTrue(api.usable)
        # argtypes really were declared, so a bad call is a Python error rather
        # than stack corruption. NULL comm: NCCL rejects it, we just want an int.
        self.assertIsInstance(api.stat(0, nccl_memory._STAT_TOTAL), int)


if __name__ == "__main__":
    unittest.main()

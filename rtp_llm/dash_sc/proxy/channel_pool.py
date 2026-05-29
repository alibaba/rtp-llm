"""Round-robin grpc.aio channel pool with age-based recycling for the dash_sc
forward proxy (predict_v2.proto).

Why a bespoke pool instead of reusing ``rtp_llm.utils.grpc_host_channel_pool``:
the forward target is a single **L4 (TCP) PVL VIP**. One
``grpc.aio.insecure_channel(vip)`` is one keepalive-pinned HTTP/2 connection
that sticks to whichever real machine the VIP picked at TCP-connect time and
never rebalances — so backend machines added behind the VIP later get zero
traffic, and ``get_state()`` dead-channel detection can't see it (a connection
to an old-but-alive machine stays ``READY`` forever). ``GrpcHostChannelPool``
keys channels by *address*, so it perceives peer churn only because the caller
hands it new addresses; here there is exactly one address, so its whole model
does not apply. This pool instead recycles healthy-but-stale channels *within*
the single address on a timer, forcing periodic reconnect so the VIP eventually
hands us the current machine set.

Hard invariant: an in-use connection is NEVER interrupted. A channel is closed
*only* when its in-flight RPC count reaches zero — no grace-timeout close. The
cost is that an immortal stream pins its (old) channel open until it ends; the
retiring list is bounded only by stream lifetime, which is acceptable here.

Loop affinity: ``grpc.aio.Channel`` objects and the background recycler task are
event-loop affine, so both are created lazily inside :meth:`ensure_started`
(called from the running loop), never in ``__init__``. ``threading.Lock`` guards
all state; ``channel.close()`` is always awaited *outside* the lock (awaiting
under a sync lock would stall the single loop thread).
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from typing import Any, Callable, List, Optional, Tuple


class _PooledChannel:
    __slots__ = (
        "addr",
        "addr_idx",
        "channel",
        "stub",
        "deadline",
        "inflight",
        "retiring",
        "closing",
    )

    def __init__(
        self,
        addr: str,
        addr_idx: int,
        channel: Any,
        stub: Any,
        deadline: float,
    ):
        self.addr = addr
        self.addr_idx = addr_idx
        self.channel = channel
        self.stub = stub
        # monotonic time at which the recycler should retire this channel.
        self.deadline = deadline
        self.inflight = 0
        self.retiring = False
        # Set exactly once, under the lock, by whichever party (release path or
        # recycler sweep or close) first observes inflight==0 on a retiring
        # channel — guarantees a single owner closes it (no double close).
        self.closing = False


class ForwardChannelPool:
    """A pool of ``channels_per_addr`` grpc.aio channels per forward address.

    Selection is round-robin across every channel of every address. The caller
    must :meth:`acquire` before issuing an RPC and :meth:`release` in a
    ``finally`` exactly once — that pairing is what tracks in-flight count so a
    retiring channel is closed only when truly idle.
    """

    def __init__(
        self,
        forward_addrs: List[str],
        channels_per_addr: int,
        *,
        stub_factory: Callable[[Any], Any],
        channel_factory: Callable[[str], Any],
        max_age_ms: int = 0,
        recycle_interval_ms: Optional[int] = None,
        max_recycle_per_tick: int = 1,
        jitter_frac: float = 0.2,
    ):
        self._forward_addrs = list(forward_addrs)
        self._channels_per_addr = max(1, channels_per_addr)
        self._stub_factory = stub_factory
        self._channel_factory = channel_factory

        # max_age <= 0 disables recycling entirely: the pool is built once and
        # never reconnected (backward-compatible with the static pool).
        self._max_age = max(0.0, max_age_ms / 1000.0)
        if recycle_interval_ms is not None and recycle_interval_ms > 0:
            self._interval = recycle_interval_ms / 1000.0
        else:
            # Scan ~4x per max_age window so an aged channel is retired within
            # roughly a quarter of its lifetime overshoot.
            self._interval = max(1.0, self._max_age / 4.0) if self._max_age else 0.0
        self._max_recycle_per_tick = max(1, max_recycle_per_tick)
        self._jitter_frac = min(0.9, max(0.0, jitter_frac))

        self._lock = threading.Lock()
        self._active: List[_PooledChannel] = []
        self._retiring: List[_PooledChannel] = []
        self._rr_idx = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._recycler_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        # Keep references to in-flight close() tasks so they aren't GC'd before
        # completing; discarded via done-callback.
        self._closing_tasks: set = set()
        self._closed = False

    # ---------- lifecycle ----------

    def ensure_started(self) -> None:
        """Build the channel pool (and start the recycler) on the running loop.

        Idempotent and cheap to call on every RPC. Must be invoked from within
        the asyncio loop that owns the gRPC server (``open()`` / a servicer
        handler); ``grpc.aio`` channels and the recycler task bind to it.
        """
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._closed:
                return
            if self._active:
                if self._loop is not loop:
                    raise RuntimeError(
                        "ForwardChannelPool belongs to a different asyncio event "
                        "loop; open and use one pool on one loop"
                    )
                return

            for addr_i, addr in enumerate(self._forward_addrs):
                for _ in range(self._channels_per_addr):
                    self._active.append(self._build_locked(addr, addr_i))
            self._loop = loop

            if self._max_age > 0 and self._recycler_task is None:
                self._recycler_task = loop.create_task(self._recycle_loop())

        logging.info(
            "[DashScGrpc] ForwardChannelPool started: %d addrs x %d ch/addr = %d, "
            "max_age=%.1fs interval=%.1fs per_tick=%d",
            len(self._forward_addrs),
            self._channels_per_addr,
            len(self._active),
            self._max_age,
            self._interval,
            self._max_recycle_per_tick,
        )

    def _build_locked(self, addr: str, addr_idx: int) -> _PooledChannel:
        # insecure_channel is synchronous (connect is lazy), so building under
        # the lock adds no await and keeps the critical section trivial.
        channel = self._channel_factory(addr)
        stub = self._stub_factory(channel)
        return _PooledChannel(addr, addr_idx, channel, stub, self._next_deadline())

    def _next_deadline(self) -> float:
        if self._max_age <= 0:
            return float("inf")
        # Independent jitter per channel so the N connections never expire in
        # lockstep, and recycled channels don't re-converge into a future herd.
        jitter = self._max_age * random.uniform(-self._jitter_frac, self._jitter_frac)
        return time.monotonic() + self._max_age + jitter

    # ---------- acquire / release ----------

    def acquire(self) -> Optional[_PooledChannel]:
        """Pick the next channel (round-robin) and mark one in-flight RPC.

        Returns ``None`` only when the pool is empty — during the
        ``server.stop`` -> ``close`` shutdown window a stray dispatched RPC may
        still reach the handler; the caller turns ``None`` into ``UNAVAILABLE``.
        The increment and the selection happen in one critical section so the
        recycler cannot retire-and-close a channel between pick and increment.
        """
        with self._lock:
            if not self._active:
                return None
            i = self._rr_idx
            self._rr_idx = (self._rr_idx + 1) % len(self._active)
            pc = self._active[i]
            pc.inflight += 1
            return pc

    def release(self, pc: _PooledChannel) -> None:
        """Mark one in-flight RPC done; close the channel if it is retiring+idle.

        Synchronous and uninterruptible by design — called from the handler's
        ``finally`` where an ``await`` could be skipped by ``CancelledError``.
        The actual ``channel.close()`` is *scheduled* (not awaited) so this stays
        sync; scheduling is safe because the handler runs on the owning loop.
        """
        to_close: Optional[_PooledChannel] = None
        with self._lock:
            pc.inflight -= 1
            if pc.retiring and pc.inflight <= 0 and not pc.closing:
                pc.closing = True
                to_close = pc
        if to_close is not None:
            self._schedule_close(to_close)

    def _schedule_close(self, pc: _PooledChannel) -> None:
        try:
            task = asyncio.ensure_future(self._close_one(pc))
        except Exception as e:  # no running loop (shouldn't happen on hot path)
            logging.warning("[DashScGrpc] failed to schedule channel close: %s", e)
            return
        self._closing_tasks.add(task)
        task.add_done_callback(self._closing_tasks.discard)

    # ---------- background recycle ----------

    async def _recycle_loop(self) -> None:
        logging.info(
            "[DashScGrpc] channel recycle loop started (interval=%.1fs)",
            self._interval,
        )
        try:
            while not self._closed:
                await asyncio.sleep(self._interval)
                try:
                    await self._recycle_once()
                except Exception as e:
                    # One bad tick must never kill recycling.
                    logging.error(
                        "[DashScGrpc] channel recycle tick failed: %s", e, exc_info=True
                    )
        except asyncio.CancelledError:
            logging.info("[DashScGrpc] channel recycle loop cancelled")
        finally:
            logging.info("[DashScGrpc] channel recycle loop stopped")

    async def _recycle_once(self) -> None:
        now = time.monotonic()
        ready: List[_PooledChannel] = []
        with self._lock:
            if self._closed:
                return
            # (1) Retire aged-out active channels: build a replacement and swap
            #     it into the slot in place (preserves list length so _rr_idx %
            #     len stays valid), then move the old one to _retiring. We do
            #     NOT close here — only when its in-flight RPCs drain.
            recycled = 0
            for idx, pc in enumerate(self._active):
                if recycled >= self._max_recycle_per_tick:
                    break
                if now >= pc.deadline:
                    self._active[idx] = self._build_locked(pc.addr, pc.addr_idx)
                    pc.retiring = True
                    self._retiring.append(pc)
                    recycled += 1
            # (2) Sweep retiring channels whose in-flight count has drained.
            for pc in self._retiring:
                if pc.inflight <= 0 and not pc.closing:
                    pc.closing = True
                    ready.append(pc)
            if recycled:
                logging.info(
                    "[DashScGrpc] recycled %d channel(s); %d retiring, %d active",
                    recycled,
                    len(self._retiring),
                    len(self._active),
                )
        for pc in ready:
            await self._close_one(pc)

    async def _close_one(self, pc: _PooledChannel) -> None:
        try:
            await pc.channel.close()
            logging.info("[DashScGrpc] closed retired channel for %s", pc.addr)
        except Exception as e:
            logging.warning(
                "[DashScGrpc] failed to close channel for %s: %s", pc.addr, e
            )
        finally:
            with self._lock:
                try:
                    self._retiring.remove(pc)
                except ValueError:
                    pass

    # ---------- shutdown ----------

    async def close(self) -> None:
        """Graceful shutdown. Call AFTER ``server.stop(grace)`` has drained
        in-flight RPCs, so by here every channel is idle and force-closing the
        residue cannot interrupt a live RPC.
        """
        task = None
        to_close: List[_PooledChannel] = []
        with self._lock:
            if self._closed:
                return
            self._closed = True
            task = self._recycler_task
            self._recycler_task = None
            to_close = list(self._active) + list(self._retiring)
            self._active.clear()
            self._retiring.clear()

        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logging.warning("[DashScGrpc] recycle task stop failed: %s", e)

        for pc in to_close:
            try:
                await pc.channel.close()
            except Exception as e:
                logging.warning("[DashScGrpc] forward channel close failed: %s", e)

    # ---------- introspection (tests / logging) ----------

    def stats(self) -> Tuple[int, int]:
        with self._lock:
            return len(self._active), len(self._retiring)

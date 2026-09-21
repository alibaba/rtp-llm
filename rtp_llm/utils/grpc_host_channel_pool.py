import asyncio
import logging
import math
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional, Tuple, Union

import grpc
from grpc import aio

GrpcChannelOption = Tuple[str, Union[str, int]]


@dataclass
class GrpcHostChannel:
    host: str
    channel: aio.Channel
    active_calls: int = 0
    idle_since: float = field(default_factory=time.monotonic)


class GrpcHostChannelPool:
    """Event-loop-local channel cache; leases must cover the entire RPC.

    Ordinary eviction only closes entries with zero leases. Explicit close()
    force-closes even active RPCs for process shutdown. max_channels bounds
    cached entries; retired channels close asynchronously.
    """

    def __init__(
        self,
        options: Optional[List[GrpcChannelOption]] = None,
        cleanup_interval: float = 60,
        *,
        idle_ttl: float = 600,
        max_channels: int = 1024,
        acquire_timeout: float = 5,
    ):
        for name, value in (
            ("cleanup_interval", cleanup_interval),
            ("idle_ttl", idle_ttl),
            ("acquire_timeout", acquire_timeout),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value}")
        if (
            isinstance(max_channels, bool)
            or not isinstance(max_channels, int)
            or max_channels <= 0
        ):
            raise ValueError(
                f"max_channels must be a positive integer, got {max_channels}"
            )
        self._options = options or []
        self._channels: Dict[str, GrpcHostChannel] = {}
        self._condition = asyncio.Condition()
        self._cleanup_interval = cleanup_interval
        self._idle_ttl = idle_ttl
        self._max_channels = max_channels
        self._acquire_timeout = acquire_timeout
        self._cleanup_task: Optional[asyncio.Task] = None
        self._close_tasks: set[asyncio.Task] = set()
        self._stopped = False

    def _retire(self, entry: GrpcHostChannel) -> None:
        # Keep ownership even if the acquiring caller is cancelled.
        task = asyncio.create_task(self._close_channel(entry))
        self._close_tasks.add(task)
        task.add_done_callback(self._close_tasks.discard)

    async def _close_channel(self, entry: GrpcHostChannel) -> None:
        try:
            await asyncio.wait_for(entry.channel.close(), timeout=2)
        except Exception:
            logging.warning(
                "Failed to close grpc channel for %s", entry.host, exc_info=True
            )

    @staticmethod
    def _is_channel_closed(entry: GrpcHostChannel) -> bool:
        try:
            # Observation must not wake idle channels.
            return entry.channel.get_state() == grpc.ChannelConnectivity.SHUTDOWN
        except Exception:
            logging.warning(
                "Failed to inspect grpc channel for %s", entry.host, exc_info=True
            )
            return False

    async def _take(self, target: str) -> GrpcHostChannel:
        async with self._condition:
            while True:
                if self._stopped:
                    raise RuntimeError(
                        f"GrpcHostChannelPool is closed, target={target}"
                    )
                if self._cleanup_task is None:
                    self._cleanup_task = asyncio.create_task(self._cleanup_loop())
                entry = self._channels.get(target)
                if entry is not None and self._is_channel_closed(entry):
                    if entry.active_calls:
                        await self._condition.wait()
                        continue
                    del self._channels[target]
                    self._retire(entry)
                    entry = None
                if entry is None:
                    if len(self._channels) >= self._max_channels:
                        idle = [
                            e for e in self._channels.values() if e.active_calls == 0
                        ]
                        if not idle:
                            await self._condition.wait()
                            continue
                        oldest = min(idle, key=lambda e: e.idle_since)
                        del self._channels[oldest.host]
                        self._retire(oldest)
                    entry = GrpcHostChannel(
                        target, aio.insecure_channel(target, options=self._options)
                    )
                    self._channels[target] = entry
                    self._condition.notify_all()
                entry.active_calls += 1
                return entry

    @asynccontextmanager
    async def acquire(
        self, target: str, *, timeout: Optional[float] = None
    ) -> AsyncIterator[aio.Channel]:
        """Timeout limits capacity waiting, not RPC duration.

        Pass a remaining request deadline to shorten the default wait.
        """
        wait = (
            self._acquire_timeout
            if timeout is None
            else min(timeout, self._acquire_timeout)
        )
        if not math.isfinite(wait) or wait <= 0:
            raise asyncio.TimeoutError(
                f"No time left to acquire grpc channel for {target}"
            )
        # Cancellation may race _take completing after it increments the count.
        task = asyncio.create_task(self._take(target))
        try:
            entry = await asyncio.wait_for(asyncio.shield(task), timeout=wait)
        except BaseException:
            task.cancel()
            try:
                entry = await task
            except (asyncio.CancelledError, Exception):
                pass
            else:
                await self._release(entry)
            raise
        try:
            yield entry.channel
        finally:
            await self._release(entry)

    async def _release(self, entry: GrpcHostChannel) -> None:
        async with self._condition:
            entry.active_calls -= 1
            if entry.active_calls == 0:
                entry.idle_since = time.monotonic()
                self._condition.notify_all()

    async def _cleanup_closed(self) -> None:
        async with self._condition:
            now = time.monotonic()
            for target, entry in list(self._channels.items()):
                if entry.active_calls == 0 and (
                    now - entry.idle_since >= self._idle_ttl
                    or self._is_channel_closed(entry)
                ):
                    del self._channels[target]
                    self._retire(entry)
            self._condition.notify_all()

    async def _cleanup_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_closed()
        except asyncio.CancelledError:
            pass

    async def close(self) -> None:
        async with self._condition:
            self._stopped = True
            cleanup = self._cleanup_task
            self._cleanup_task = None
            if cleanup is not None:
                cleanup.cancel()
            for entry in self._channels.values():
                self._retire(entry)
            self._channels.clear()
            self._condition.notify_all()
            pending = list(self._close_tasks)
        if cleanup is not None:
            try:
                await cleanup
            except asyncio.CancelledError:
                if not cleanup.cancelled():
                    raise
        if pending:
            await asyncio.gather(*(asyncio.shield(task) for task in pending))

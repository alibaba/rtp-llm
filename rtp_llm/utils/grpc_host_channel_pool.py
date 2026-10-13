import asyncio
import logging
import time
from typing import Dict, Iterable, List, Optional, Tuple, Union

import grpc
from grpc import aio

GrpcChannelOption = Tuple[str, Union[str, int]]


class GrpcHostChannel:
    __slots__ = ("host", "channel", "last_used_at", "transient_failure_since")

    def __init__(self, host: str, channel: aio.Channel):
        self.host = host
        self.channel = channel
        self.last_used_at = time.monotonic()
        self.transient_failure_since: Optional[float] = None


class GrpcHostChannelPool:
    """
    A pool of grpc channels keyed by host address.
    """

    def __init__(
        self,
        options: Optional[List[GrpcChannelOption]] = None,
        cleanup_interval: int = 60,
        transient_failure_grace_period: float = 300,
        idle_channel_ttl: float = 300,
    ):
        """
        :param options: aio.insecure_channel 的 gRPC options
        :param cleanup_interval: 后台清理任务的执行间隔（秒）
        :param transient_failure_grace_period: TRANSIENT_FAILURE 持续多久后允许后台淘汰（秒）
        :param idle_channel_ttl: IDLE channel 未被使用多久后允许后台淘汰（秒）
        """
        self._options = options or []
        self._channels: Dict[str, GrpcHostChannel] = {}
        self._closed_channels: List[GrpcHostChannel] = []
        self._pending_close_channels: Dict[int, GrpcHostChannel] = {}
        self._lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._cleanup_interval = cleanup_interval
        self._transient_failure_grace_period = transient_failure_grace_period
        self._idle_channel_ttl = idle_channel_ttl
        self._cleanup_task: Optional[asyncio.Task] = None  # type: ignore
        self._stopped = False

    def __del__(self):
        try:
            if not self._stopped:
                self._stopped = True
                if self._cleanup_task:
                    self._cleanup_task.cancel()
                self._channels.clear()
                self._closed_channels.clear()
                self._pending_close_channels.clear()
        except Exception as e:
            logging.warning("Failed to cleanup GrpcHostChannelPool in __del__: %s", e)

    async def close(self):
        async with self._close_lock:
            cleanup_task = None
            try:
                async with self._lock:
                    if not self._stopped:
                        self._stopped = True
                        cleanup_task = self._cleanup_task
                        self._cleanup_task = None
                    self._queue_entries_for_close(self._channels.values())
                    self._queue_entries_for_close(self._closed_channels)
                    self._channels.clear()
                    self._closed_channels.clear()

                if cleanup_task:
                    cleanup_task.cancel()
                    try:
                        await cleanup_task
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logging.warning(
                            "Failed to stop grpc channel cleanup task: %s", e
                        )

                async with self._lock:
                    to_close = list(self._pending_close_channels.values())
                closed_entries = await self._close_entries(to_close)
                await self._remove_closed_entries(closed_entries)
            except Exception as e:
                logging.warning("Failed to close GrpcHostChannelPool: %s", e)

    async def get(self, target: str) -> aio.Channel:
        """
        Get or create a channel for `target`.
        """
        # Ensure cleanup task is started (with lock to prevent race condition)
        async with self._lock:
            if self._stopped:
                raise RuntimeError(f"GrpcHostChannelPool is closed, target={target}")
            if self._cleanup_task is None and not self._stopped:
                loop = asyncio.get_running_loop()
                self._cleanup_task = loop.create_task(self._cleanup_loop())
                logging.info(
                    f"Channel cleanup task started in get() cleanup_interval={self._cleanup_interval}s)"
                )

            entry = self._channels.get(target)

            # check and recreate if needed
            if entry and self._is_channel_closed(entry):
                self._closed_channels.append(entry)
                entry = None
                logging.info(f"Channel for {target} is closed, recreating new channel")
            if not entry:
                # Just create new channel, let cleanup loop handle the old one
                ch = aio.insecure_channel(target, options=self._options)
                entry = GrpcHostChannel(target, ch)
                self._channels[target] = entry
            entry.last_used_at = time.monotonic()

        return entry.channel

    def _queue_entries_for_close(self, entries: Iterable[GrpcHostChannel]) -> None:
        """Register entries for closing while the caller holds ``_lock``."""
        for entry in entries:
            self._pending_close_channels[id(entry)] = entry

    async def _remove_closed_entries(
        self, entries: Iterable[GrpcHostChannel]
    ) -> None:
        async with self._lock:
            for entry in entries:
                self._pending_close_channels.pop(id(entry), None)

    # ---------- background cleanup ----------

    async def _cleanup_loop(self):
        logging.info(
            f"Channel cleanup loop started, will run every {self._cleanup_interval}s"
        )
        try:
            while not self._stopped:
                await asyncio.sleep(self._cleanup_interval)
                try:
                    await self._cleanup_closed()
                except Exception as e:
                    # Catch all exceptions to prevent loop from stopping
                    logging.error(f"Error in channel cleanup: {e}", exc_info=True)
        except asyncio.CancelledError:
            logging.info("Channel cleanup loop cancelled")
        finally:
            logging.info("Channel cleanup loop stopped")

    async def _cleanup_closed(self):
        """
        Find shutdown or persistently unavailable channels, remove them from the
        pool, and close them.

        A recent TRANSIENT_FAILURE is retained because gRPC channels recover from
        that state automatically. A channel that stays unavailable beyond the
        grace period is evicted so dynamic, permanently offline peers do not
        accumulate forever. Long-unused IDLE channels are also evicted without
        forcing a connection attempt.
        """
        async with self._cleanup_lock:
            async with self._lock:
                newly_closed = list(self._closed_channels)
                self._closed_channels.clear()
                total_channels = len(self._channels)
                for target, entry in list(self._channels.items()):
                    try:
                        # Check if channel is closed
                        if self._is_channel_closed(
                            entry,
                            evict_stale_transient_failure=True,
                            evict_stale_idle=True,
                        ):
                            logging.info(
                                f"Channel {entry.host} is shutdown or persistently "
                                "unavailable, marking for cleanup"
                            )
                            newly_closed.append(entry)
                            del self._channels[
                                target
                            ]  # remove reference to prevent memory leak
                    except Exception as e:
                        # Log error but continue checking other channels
                        logging.warning(f"Error checking channel {entry.host}: {e}")

                self._queue_entries_for_close(newly_closed)
                to_close = list(self._pending_close_channels.values())
                remaining_channels = len(self._channels)
                if to_close:
                    logging.info(
                        f"Channel cleanup: closing {len(to_close)} closed/offline channels, {remaining_channels} channels remaining (was {total_channels})"
                    )
                elif total_channels > 0:
                    logging.debug(
                        f"Channel cleanup: no closed channels found, {total_channels} active channels"
                    )

            closed_entries = await self._close_entries(to_close)
            await self._remove_closed_entries(closed_entries)

    async def _close_entries(
        self, entries: List[GrpcHostChannel]
    ) -> List[GrpcHostChannel]:
        # Close outside lock
        closed_count = 0
        failed_count = 0
        closed_entries: List[GrpcHostChannel] = []
        seen = set()
        for entry in entries:
            entry_id = id(entry)
            if entry_id in seen:
                continue
            seen.add(entry_id)
            try:
                await asyncio.wait_for(entry.channel.close(), timeout=2.0)
                closed_count += 1
                closed_entries.append(entry)
                logging.info(f"Successfully closed channel for {entry.host}")
            except asyncio.TimeoutError:
                failed_count += 1
                logging.warning(f"Timeout while closing channel for {entry.host}")
            except Exception as e:
                failed_count += 1
                logging.warning(f"Error closing channel for {entry.host}: {e}")

        if entries:
            logging.info(
                f"Channel cleanup completed: {closed_count} channels closed successfully, {failed_count} failed"
            )
        return closed_entries

    def _is_channel_closed(
        self,
        entry: GrpcHostChannel,
        evict_stale_transient_failure: bool = False,
        evict_stale_idle: bool = False,
    ) -> bool:
        """
        check if the gRPC channel is closed
        """
        try:
            state = entry.channel.get_state()
            now = time.monotonic()
            if state == grpc.ChannelConnectivity.SHUTDOWN:
                logging.info(f"channel for [{entry.host}] is shutdown")
                return True
            elif state == grpc.ChannelConnectivity.TRANSIENT_FAILURE:
                if entry.transient_failure_since is None:
                    entry.transient_failure_since = now
                failure_duration = now - entry.transient_failure_since
                if (
                    evict_stale_transient_failure
                    and failure_duration >= self._transient_failure_grace_period
                ):
                    logging.info(
                        f"channel for [{entry.host}] has remained in "
                        f"TRANSIENT_FAILURE for {failure_duration:.1f}s"
                    )
                    return True
                logging.info(
                    f"channel for [{entry.host}] is in TRANSIENT_FAILURE state; "
                    "keeping it for automatic reconnection"
                )
            elif state == grpc.ChannelConnectivity.IDLE:
                entry.transient_failure_since = None
                idle_duration = now - entry.last_used_at
                if evict_stale_idle and idle_duration >= self._idle_channel_ttl:
                    logging.info(
                        f"channel for [{entry.host}] has remained unused in "
                        f"IDLE for {idle_duration:.1f}s"
                    )
                    return True
            elif state == grpc.ChannelConnectivity.READY:
                entry.transient_failure_since = None
            return False
        except Exception as e:
            logging.error(f"check channel for [{entry.host}] closed failed:{str(e)}")
            return True

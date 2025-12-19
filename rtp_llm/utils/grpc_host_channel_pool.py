import asyncio
import logging
from typing import Dict, List, Optional, Tuple

import grpc
from grpc import aio


class GrpcHostChannel:
    __slots__ = ("host", "channel")

    def __init__(self, host: str, channel: aio.Channel):
        self.host = host
        self.channel = channel


class GrpcHostChannelPool:
    """
    A pool of grpc channels keyed by host address.
    """

    def __init__(
        self,
        options: Optional[List[Tuple[str, str]]] = None,
        cleanup_interval: int = 60,
    ):
        """
        :param options: aio.insecure_channel 的 gRPC options
        """
        self._options = options or []
        self._channels: Dict[str, GrpcHostChannel] = {}
        self._closed_channels: List[GrpcHostChannel] = []
        self._lock = asyncio.Lock()
        self._cleanup_interval = cleanup_interval
        self._cleanup_task: Optional[asyncio.Task] = None  # type: ignore
        self._stopped = False

    def __del__(self):
        try:
            if not self._stopped:
                self._stopped = True
                if self._cleanup_task:
                    self._cleanup_task.cancel()
                self._channels.clear()
        except Exception as e:
            logging.warning("Failed to cleanup GrpcHostChannelPool in __del__: %s", e)

    async def get(self, target: str) -> aio.Channel:
        """
        Get or create a channel for `target`.
        """
        # Ensure cleanup task is started (with lock to prevent race condition)
        async with self._lock:
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

        return entry.channel

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
        Find closed channels (including offline peers), remove them from the pool, and close them.
        This prevents memory leak when peers go offline and are no longer accessed via get().
        """
        to_close: List[GrpcHostChannel] = []
        try:
            async with self._lock:
                to_close = [_ for _ in self._closed_channels]
                self._closed_channels.clear()
                total_channels = len(self._channels)
                for target, entry in list(self._channels.items()):
                    try:
                        # Check if channel is closed
                        if self._is_channel_closed(entry, try_to_connect=True):
                            logging.info(
                                f"Channel {entry.host} is closed/offline, marking for cleanup"
                            )
                            to_close.append(entry)
                            del self._channels[
                                target
                            ]  # remove reference to prevent memory leak
                    except Exception as e:
                        # Log error but continue checking other channels
                        logging.warning(f"Error checking channel {entry.host}: {e}")

                remaining_channels = len(self._channels)
                if to_close:
                    logging.info(
                        f"Channel cleanup: closing {len(to_close)} closed/offline channels, {remaining_channels} channels remaining (was {total_channels})"
                    )
                elif total_channels > 0:
                    logging.debug(
                        f"Channel cleanup: no closed channels found, {total_channels} active channels"
                    )
        except Exception as e:
            # Log error but don't re-raise to prevent cleanup loop from stopping
            logging.error(f"Error in _cleanup_closed: {e}", exc_info=True)

        # Close outside lock
        closed_count = 0
        failed_count = 0
        for entry in to_close:
            try:
                await asyncio.wait_for(entry.channel.close(), timeout=2.0)
                closed_count += 1
                logging.info(f"Successfully closed channel for {entry.host}")
            except asyncio.TimeoutError:
                failed_count += 1
                logging.warning(f"Timeout while closing channel for {entry.host}")
            except Exception as e:
                failed_count += 1
                logging.warning(f"Error closing channel for {entry.host}: {e}")

        if to_close:
            logging.info(
                f"Channel cleanup completed: {closed_count} channels closed successfully, {failed_count} failed"
            )

    def _is_channel_closed(
        self, entry: GrpcHostChannel, try_to_connect: bool = False
    ) -> bool:
        """
        check if the gRPC channel is closed
        """
        try:
            state = entry.channel.get_state(try_to_connect=try_to_connect)
            if state == grpc.ChannelConnectivity.SHUTDOWN:
                logging.info(f"channel for [{entry.host}] is shutdown")
                return True
            elif state == grpc.ChannelConnectivity.TRANSIENT_FAILURE:
                logging.info(
                    f"channel for [{entry.host}] is in TRANSIENT_FAILURE state (peer is offline/closed)"
                )
                return True
            return False
        except Exception as e:
            logging.error(f"check channel for [{entry.host}] closed failed:{str(e)}")
            return True

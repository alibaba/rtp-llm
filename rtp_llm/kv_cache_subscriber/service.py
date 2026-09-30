from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from rtp_llm.kv_cache_subscriber.config import SubscriberConfig
from rtp_llm.kv_cache_subscriber.models import CacheDiff, CacheDiffTracker, CacheSnapshot
from rtp_llm.kv_cache_subscriber.reporter import KvcmReporter
from rtp_llm.kv_cache_subscriber.source import CacheStatusSource

logger = logging.getLogger(__name__)


class SubscriberService:
    """Poll RTP snapshots and converge KVCM to the acknowledged local state."""

    def __init__(
        self,
        config: SubscriberConfig,
        source: CacheStatusSource,
        reporter: KvcmReporter,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._source = source
        self._reporter = reporter
        self._clock = clock
        self._tracker = CacheDiffTracker(config.deletion_confirmations)
        self._reporter_started = False
        self._node_registered = False
        self._cold_reset_pending = config.reset_on_start
        self._force_full_add = True
        self._block_size: int | None = None
        self._source_failures = 0
        self._next_full_refresh_at = 0.0
        self._last_heartbeat_at = 0.0

    @property
    def tracker(self) -> CacheDiffTracker:
        return self._tracker

    @property
    def node_registered(self) -> bool:
        return self._node_registered

    async def _ensure_reporter(self, snapshot: CacheSnapshot) -> None:
        if self._reporter_started:
            if snapshot.block_size != self._block_size:
                raise RuntimeError(
                    "RTP cache block size changed while subscriber was running: "
                    f"{self._block_size} -> {snapshot.block_size}"
                )
            return
        await self._reporter.start(snapshot.block_size)
        self._reporter_started = True
        self._block_size = snapshot.block_size

    async def _ensure_node_registered(self) -> None:
        if self._node_registered:
            return
        if self._cold_reset_pending:
            # A successful full RTP snapshot makes this reset authoritative: it
            # removes stale locations left by an older subscriber process.
            await self._reporter.report_host_down()
            self._tracker.reset()
        await self._reporter.register_node()
        self._node_registered = True
        self._cold_reset_pending = False
        self._force_full_add = True

    async def process_snapshot(self, snapshot: CacheSnapshot) -> CacheDiff:
        """Process one complete snapshot; commit only after KVCM acknowledges it."""

        self._source_failures = 0
        await self._ensure_reporter(snapshot)
        await self._ensure_node_registered()

        now = self._clock()
        force_full_add = self._force_full_add or now >= self._next_full_refresh_at
        diff = self._tracker.plan(
            snapshot.keys,
            force_full_add=force_full_add,
        )
        try:
            if not diff.empty:
                await self._reporter.report_diff(diff, snapshot.block_size)
        except Exception as exc:
            self._tracker.mark_uncertain(diff)
            if getattr(exc, "code", None) == "NODE_NOT_REGISTERED":
                self._node_registered = False
                self._force_full_add = True
            raise

        self._tracker.commit(diff)
        if force_full_add:
            self._force_full_add = False
            self._next_full_refresh_at = now + self._config.full_refresh_interval_s
        return diff

    async def handle_source_failure(self) -> bool:
        """Record a failed full pull and report HOST_DOWN at the threshold.

        Returns True only when KVCM acknowledged the host-down transition.
        """

        self._source_failures += 1
        if (
            not self._reporter_started
            or not self._node_registered
            or self._source_failures < self._config.engine_failure_threshold
        ):
            return False
        await self._reporter.report_host_down()
        self._node_registered = False
        self._tracker.reset()
        self._force_full_add = True
        return True

    async def _maybe_heartbeat(self) -> None:
        if not self._node_registered:
            return
        now = self._clock()
        if now - self._last_heartbeat_at < self._config.kvcm_heartbeat_interval_s:
            return
        try:
            await self._reporter.report_heartbeat()
            self._last_heartbeat_at = now
        except Exception as exc:
            if getattr(exc, "code", None) == "NODE_NOT_REGISTERED":
                self._node_registered = False
                self._force_full_add = True
            logger.warning("failed to report KVCM heartbeat", exc_info=True)

    async def run(self) -> None:
        logger.info(
            "RTP KV cache subscriber started: endpoints=%s, poll_interval_s=%s",
            self._config.rtp_endpoints,
            self._config.poll_interval_s,
        )
        try:
            while True:
                try:
                    snapshot = await self._source.fetch_snapshot()
                except Exception:
                    logger.warning("GetCacheStatus full pull failed", exc_info=True)
                    try:
                        await self.handle_source_failure()
                    except Exception:
                        logger.warning("failed to report RTP host down", exc_info=True)
                else:
                    try:
                        diff = await self.process_snapshot(snapshot)
                        if not diff.empty:
                            logger.info(
                                "KVCM cache diff acknowledged: added=%d removed=%d "
                                "snapshot_keys=%d version=%d",
                                len(diff.added),
                                len(diff.removed),
                                len(snapshot.keys),
                                snapshot.version,
                            )
                    except Exception:
                        logger.warning(
                            "failed to synchronize RTP cache snapshot to KVCM; "
                            "acknowledged baseline was not advanced",
                            exc_info=True,
                        )
                await self._maybe_heartbeat()
                await asyncio.sleep(self._config.poll_interval_s)
        finally:
            await self._source.close()
            await self._reporter.close()

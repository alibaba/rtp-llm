from __future__ import annotations

import asyncio
import base64
import functools
import hashlib
import logging
import threading
import zlib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from .VLLMBlockHash import calculate_vllm_block_hashes

logger = logging.getLogger("vllm.navi_sched.client")
_MAX_REPLAY_BYTES = 2 * 1024 * 1024
_MAX_COMPRESSED_REPLAY_BYTES = 512 * 1024
_REPLAY_LOG_CHUNK_BYTES = 8 * 1024
_MAX_REPLAY_SAMPLES = 100


class NaviSchedEndpointRouter:
    def __init__(
        self,
        client,
        target_biz_name: str,
        timeout_ms: int = 1000,
        block_size: int = 16,
        route_worker_count: int = 64,
        replay_sample_interval: int = 0,
    ):
        if not target_biz_name:
            raise ValueError("target_biz_name is required")
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if route_worker_count <= 0:
            raise ValueError("route_worker_count must be positive")
        if replay_sample_interval < 0:
            raise ValueError("replay_sample_interval must be nonnegative")
        self._client = client
        self._target_biz_name = str(target_biz_name)
        self._timeout_ms = int(timeout_ms)
        self._block_size = int(block_size)
        self._route_slots = threading.BoundedSemaphore(route_worker_count)
        self._replay_log_slots = threading.BoundedSemaphore(2)
        self._replay_sample_interval = int(replay_sample_interval)
        self._replay_sequence = 0
        self._replay_samples = 0
        self._replay_lock = threading.Lock()
        self._executor_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=route_worker_count,
            thread_name_prefix="NaviSchedRoute",
        )
        self._replay_log_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="NaviSchedReplayLog",
        )
        self._closed = False

    def route(
        self,
        prompt_token_ids: Sequence[int],
        request_id: int | None = None,
        block_hashes=None,
        timeout_ms: int | None = None,
        forward_payload: bytes = b"",
        block_hash_request=None,
    ):
        effective_timeout_ms = self._timeout_ms if timeout_ms is None else int(timeout_ms)
        if effective_timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        options = {
            "target_biz_name": self._target_biz_name,
            "timeout_ms": effective_timeout_ms,
            "request_id": request_id,
        }
        token_ids = [int(token_id) for token_id in prompt_token_ids]
        hashes = (
            calculate_vllm_block_hashes(
                token_ids,
                self._block_size,
                request=block_hash_request,
            )
            if block_hashes is None
            else [int(value) for value in block_hashes]
        )
        collect_replay = False
        if self._replay_sample_interval and logger.isEnabledFor(logging.INFO):
            with self._replay_lock:
                eligible = (
                    self._replay_samples < _MAX_REPLAY_SAMPLES
                    and self._replay_sequence % self._replay_sample_interval == 0
                )
                self._replay_sequence += 1
                if eligible:
                    collect_replay = self._replay_log_slots.acquire(blocking=False)
                if collect_replay:
                    self._replay_samples += 1
        if collect_replay:
            options["collect_replay"] = True
        try:
            decision = self._client.schedule(
                len(token_ids),
                block_hashes=hashes,
                block_size=self._block_size,
                forward_payload=forward_payload,
                **options,
            )
        except Exception:
            if collect_replay:
                self._replay_log_slots.release()
            raise
        if collect_replay:
            if decision.replay is None:
                self._replay_log_slots.release()
            else:
                try:
                    future = self._replay_log_executor.submit(self._log_replay, decision)
                except Exception:
                    self._replay_log_slots.release()
                    logger.exception("navi-replay-sample submit failed request_id=%s",
                                     decision.request_id)
                else:
                    future.add_done_callback(
                        lambda unused: self._replay_log_slots.release()
                    )
        return decision

    @staticmethod
    def _log_replay(decision) -> None:
        try:
            raw = decision.replay.to_json().encode("utf-8")
            if len(raw) > _MAX_REPLAY_BYTES:
                logger.warning("navi-replay-sample skipped request_id=%s reason=oversized size=%s",
                               decision.request_id, len(raw))
                return
            compressed = zlib.compress(raw, level=1)
            if len(compressed) > _MAX_COMPRESSED_REPLAY_BYTES:
                logger.warning("navi-replay-sample skipped request_id=%s reason=compressed_oversized size=%s",
                               decision.request_id, len(compressed))
                return
            encoded = base64.b64encode(compressed).decode("ascii")
            digest = hashlib.sha256(raw).hexdigest()
            parts = [encoded[index:index + _REPLAY_LOG_CHUNK_BYTES]
                     for index in range(0, len(encoded), _REPLAY_LOG_CHUNK_BYTES)]
            for index, part in enumerate(parts, start=1):
                logger.info(
                    "navi-replay-sample request_id=%s schedule_id=%s part=%s total=%s sha256=%s data=%s",
                    decision.request_id, decision.schedule_id, index,
                    len(parts), digest, part,
                )
        except Exception:
            logger.exception("navi-replay-sample failed request_id=%s", decision.request_id)

    async def route_async(self, prompt_token_ids: Sequence[int], **options):
        return await asyncio.wrap_future(
            self._submit(
                functools.partial(
                    self.route,
                    prompt_token_ids,
                    **options,
                )
            )
        )

    def route_generate_request(
        self,
        request,
        request_id: int | None = None,
        block_hashes=None,
        timeout_ms: int | None = None,
    ):
        return self.route(
            request.token_ids,
            request_id=request_id or getattr(request, "request_id", None),
            block_hashes=block_hashes,
            timeout_ms=timeout_ms,
        )

    async def route_generate_request_async(self, request, **options):
        return await asyncio.wrap_future(
            self._submit(
                functools.partial(
                    self.route_generate_request,
                    request,
                    **options,
                )
            )
        )

    def _submit(self, callback):
        with self._executor_lock:
            if self._closed:
                raise RuntimeError("NaviSched endpoint router is closed")
            if not self._route_slots.acquire(blocking=False):
                raise RuntimeError("NaviSched route worker capacity exhausted")
            try:
                future = self._executor.submit(callback)
            except Exception:
                self._route_slots.release()
                raise
        future.add_done_callback(lambda unused: self._route_slots.release())
        return future

    def close(self) -> None:
        with self._executor_lock:
            if self._closed:
                return
            self._closed = True
            executor = self._executor
        executor.shutdown(wait=True, cancel_futures=True)
        self._replay_log_executor.shutdown(wait=True, cancel_futures=False)

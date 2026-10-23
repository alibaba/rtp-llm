"""Content-inspection coordination, isolated from GPU computation.

The CPU-source path never rewrites model URLs. Legacy adapters retain their
media rewrite contract until they opt into the CPU preprocessing sidecar.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

from rtp_llm.multimodal.greennet_hook import GreenNetVerdict
from rtp_llm.multimodal.mm_embedding_cache import MMEmbeddingCacheEntry
from rtp_llm.multimodal.mm_work_item import _input_has_url
from rtp_llm.multimodal.source_download_executor import SourceDownloadExecutor
from rtp_llm.ops import MultimodalInput


class InspectionTimeoutError(concurrent.futures.TimeoutError):
    """A threaded timeout that survives Python 3.10's asyncio Future bridge.

    asyncio converts the exact concurrent.futures.TimeoutError type back into
    asyncio.TimeoutError when chaining futures; a subtype preserves the public
    engine's existing isinstance(..., concurrent.futures.TimeoutError) contract.
    """


class SourceInspection:
    """One-shot ownership handoff from CPU workers to the background loop."""

    def __init__(self):
        self.sources = concurrent.futures.Future()
        self.verdict = None
        self._lock = threading.Lock()

    def supply(self, mm_inputs, sources, cached_entries):
        # Only the owning request calls supply; cancellation may race with it.
        with self._lock:
            future = self.sources
            if future is None or future.cancelled():
                raise concurrent.futures.CancelledError("Inspection already ended")
            future.set_result((mm_inputs, sources, cached_entries))

    def supply_work_items(self, work_items):
        """Bind the audit to the bytes that produced each embedding."""
        source_inputs, sources, cached_entries = [], [], []
        producers = {
            item.cache_entry: item
            for item in work_items
            if item.inspection_sources is not None and item.cache_entry is not None
        }
        for item in work_items:
            producer = producers.get(item.cache_entry, item)
            if producer.inspection_sources is not None:
                source_inputs.extend(item.mm_inputs)
                sources.extend(producer.inspection_sources)
            elif item.cache_entry is not None:
                cached_entries.append(item.cache_entry)
            else:
                raise RuntimeError("Missing CPU inspection sources")
        self.supply(source_inputs, sources, cached_entries)
        for item in work_items:
            item.inspection_sources = None

    def cancel(self):
        with self._lock:
            if self.sources is not None:
                self.sources.cancel()
        if self.verdict is not None:
            self.verdict.cancel()


class MMInspectionCoordinator:
    def __init__(self, mm_part, vit_config, hash_key_cache, provider, report_error):
        self.mm_part = mm_part
        self.vit_config = vit_config
        self._hash_key_cache = hash_key_cache
        self._greennet_provider = provider
        self.report_vit_error = report_error
        self._greennet_loop = None
        self._greennet_loop_closed = False
        self._greennet_loop_thread = None
        self._greennet_loop_lock = threading.Lock()
        self._greennet_timeout_s = float(vit_config.mm_timeout_ms or 120000) / 1000.0
        self._source_download_executor = SourceDownloadExecutor(
            vit_config.vit_source_download_concurrency
        )

    def _greennet_enabled(self) -> bool:
        # Effective enablement (internal source present AND runtime flag on).
        # When off, every greennet path is skipped so behavior is identical to
        # the pre-greennet engine — no asyncio loop, no rewrite, no inspect.
        return self._greennet_provider.is_enabled()

    def _request_scoped_inspection(self) -> bool:
        return self._greennet_enabled() and getattr(
            self.mm_part, "inspection_verdict_is_request_scoped", False
        )

    def _ensure_greennet_loop(self) -> asyncio.AbstractEventLoop:
        """Lazily start the dedicated asyncio loop that hosts greennet inspect
        tasks. The loop runs in a daemon thread so background uploads / POSTs
        survive across worker-thread calls."""
        with self._greennet_loop_lock:
            if self._greennet_loop_closed:
                raise RuntimeError("GreenNet loop is closed")
            if self._greennet_loop is None:
                loop = asyncio.new_event_loop()

                def run_loop():
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_forever()
                    finally:
                        try:
                            pending = asyncio.all_tasks(loop)
                            for task in pending:
                                task.cancel()
                            if pending:
                                loop.run_until_complete(
                                    asyncio.gather(*pending, return_exceptions=True)
                                )
                            loop.run_until_complete(loop.shutdown_asyncgens())
                            loop.run_until_complete(self._greennet_provider.close())
                        finally:
                            loop.close()
                            asyncio.set_event_loop(None)

                thread = threading.Thread(
                    target=run_loop, daemon=True, name="greennet-loop"
                )
                thread.start()
                self._greennet_loop = loop
                self._greennet_loop_thread = thread
        return self._greennet_loop

    def _begin_greennet(
        self,
        mm_inputs: List[MultimodalInput],
        entry: Optional[MMEmbeddingCacheEntry] = None,
        request_id: int = 0,
        user_id: str = "",
        service_name: str = "",
        deadline: Optional[float] = None,
        worker_sources: bool = False,
        external_request_id: str = "",
    ) -> Tuple[
        List[MultimodalInput], Optional["concurrent.futures.Future"], Optional[Any]
    ]:
        """Return inference inputs, the verdict future and cancellation handle.

        Supporting CPU workers supply original encoded bytes after preprocessing;
        this path does not rewrite or download inference URLs here. Legacy media
        adapters retain their existing download/frame-extraction rewrite contract.
        Inspection runs independently on the GreenNet loop in both cases.

        If ``entry`` is given, a done-callback stamps the verdict onto it the
        moment inspection finishes — so ``WaitGreenNetVerdict`` unblocks without
        waiting for ViT. Caller is responsible for cancelling ``handle``.

        On a no-op provider, returns the inputs unchanged with no future/handle
        (and stamps a passing verdict on ``entry`` if provided)."""
        if not self._greennet_enabled():
            if entry is not None:
                entry.set_greennet_verdict(GreenNetVerdict(passed=True), checked=False)
            return mm_inputs, None, None

        loop = self._ensure_greennet_loop()
        req = SimpleNamespace(
            # Never replace the numeric ID used by scheduling/cancellation.
            id=external_request_id or str(request_id),
            model_name="",
            user_id=user_id,
            service_name=service_name,
            greennet_passed=(
                not self._request_scoped_inspection()
                and len(mm_inputs) == 1
                and _input_has_url(mm_inputs[0])
                and self._hash_key_cache.greennet_passed(mm_inputs[0].cache_key())
            ),
        )
        if deadline is None:
            deadline = time.monotonic() + self._greennet_timeout_s
        if (
            worker_sources
            and self.mm_part.supports_inspection_sources
            and not self._request_scoped_inspection()
        ):
            if req.greennet_passed:
                if entry is not None:
                    entry.set_greennet_verdict(GreenNetVerdict(passed=True))
                return mm_inputs, None, None
            return self._begin_sources(mm_inputs, req, entry, deadline, loop)
        ownership_lock = threading.Lock()
        abandoned = threading.Event()
        unclaimed = {}

        async def prepare():
            source_inputs = None
            prepare_sources = getattr(self.mm_part, "prepare_inspection_inputs", None)
            if callable(prepare_sources):
                source_inputs = await self._source_download_executor.run(
                    prepare_sources,
                    mm_inputs,
                    self.vit_config,
                    deadline=deadline,
                    cancellation_event=abandoned,
                )
                if source_inputs is not None and len(source_inputs) != len(mm_inputs):
                    raise ValueError(
                        "inspection source preparation changed input count"
                    )
            if abandoned.is_set():
                raise asyncio.CancelledError()
            handle = await self._greennet_provider.preprocess_and_submit(
                req, source_inputs if source_inputs is not None else mm_inputs
            )
            # A provider may finish concurrently with timeout, or even suppress
            # cancellation. Keep ownership until the waiting thread takes it.
            with ownership_lock:
                if not abandoned.is_set():
                    unclaimed["handle"] = handle
                    return handle, source_inputs
            handle.cancel()
            raise asyncio.CancelledError()

        preprocess_future = asyncio.run_coroutine_threadsafe(prepare(), loop)
        try:
            handle, source_inputs = preprocess_future.result(
                timeout=max(0.0, deadline - time.monotonic())
            )
            with ownership_lock:
                unclaimed.pop("handle", None)
        except BaseException:
            with ownership_lock:
                abandoned.set()
                late_handle = unclaimed.pop("handle", None)
            preprocess_future.cancel()
            self._cancel_greennet(late_handle)
            raise

        try:
            # A model can pin lossless source bytes while the safety provider
            # builds its own preview. Its verdict still gates cache publication
            # and every result; only the inference input is preserved here.
            rewritten = list(
                source_inputs if source_inputs is not None else handle.rewritten_inputs
            )

            async def wait_verdict():
                try:
                    return await asyncio.wait_for(
                        handle.wait_result(),
                        timeout=max(0.0, deadline - time.monotonic()),
                    )
                except asyncio.TimeoutError as error:
                    # Python 3.10 has distinct asyncio/concurrent timeout types.
                    # Keep the public threaded engine's existing exception contract.
                    raise InspectionTimeoutError(
                        "GreenNet deadline exceeded"
                    ) from error

            verdict_future = asyncio.run_coroutine_threadsafe(wait_verdict(), loop)
        except BaseException:
            self._cancel_greennet(handle)
            raise
        if entry is not None:

            def _stamp(fut: "concurrent.futures.Future") -> None:
                try:
                    verdict = fut.result()
                except Exception as error:  # noqa: BLE001 - convert to process error
                    inspect_error = RuntimeError(f"greennet inspect failed: {error}")
                    self.report_vit_error(inspect_error, entry)
                    verdict = GreenNetVerdict(
                        passed=False, code=11, message=str(inspect_error)
                    )
                    try:
                        setattr(verdict, "_vit_error_qps_reported", True)
                    except Exception:
                        pass
                try:
                    entry.set_greennet_verdict(verdict)
                except Exception as error:
                    self.report_vit_error(error, entry)

            verdict_future.add_done_callback(_stamp)
        return rewritten, verdict_future, handle

    def _begin_sources(self, mm_inputs, request, entry, deadline, loop):
        handoff = SourceInspection()

        async def inspect():
            handle = None
            try:

                async def run():
                    nonlocal handle
                    source_inputs, sources, cached_entries = await asyncio.wrap_future(
                        handoff.sources
                    )
                    with handoff._lock:
                        handoff.sources = None
                    # Never approve a cached tensor using a fresh download of
                    # a mutable URL. Its owner must supply the matching verdict.
                    for cached in cached_entries:
                        verdict = await asyncio.to_thread(
                            cached.wait_greennet,
                            timeout=max(0.0, deadline - time.monotonic()),
                        )
                        if verdict is None:
                            raise RuntimeError(
                                "Cached embedding has no inspection verdict"
                            )
                        if not verdict.passed:
                            return verdict
                        if not cached.greennet_passed:
                            raise RuntimeError("Cached embedding was not inspected")
                    if not source_inputs:
                        return GreenNetVerdict(passed=True)
                    handle = await self._greennet_provider.submit_sources(
                        request, source_inputs, sources
                    )
                    # The provider owns the source bytes until its task finishes.
                    del sources
                    return await handle.wait_result()

                return await asyncio.wait_for(
                    run(), timeout=max(0.0, deadline - time.monotonic())
                )
            except asyncio.TimeoutError as error:
                raise InspectionTimeoutError("GreenNet deadline exceeded") from error
            finally:
                if handle is not None:
                    handle.cancel()
                # Future results hold strong references even after consumption.
                with handoff._lock:
                    handoff.sources = None

        future = asyncio.run_coroutine_threadsafe(inspect(), loop)
        handoff.verdict = future
        if entry is not None:

            def stamp(completed):
                try:
                    verdict = completed.result()
                    if verdict is None:
                        raise RuntimeError("GreenNet returned no verdict")
                except BaseException as error:
                    verdict = GreenNetVerdict(False, code=11, message=str(error))
                entry.set_greennet_verdict(verdict)

            future.add_done_callback(stamp)
        return mm_inputs, future, handoff

    def _cancel_greennet(self, handle: Optional[Any]) -> None:
        if handle is None or self._greennet_loop is None:
            return
        try:
            self._greennet_loop.call_soon_threadsafe(handle.cancel)
        except Exception as e:  # noqa: BLE001
            logging.warning(f"greennet handle cancel failed: {e}")

    def _shutdown_greennet_loop(self) -> None:
        with self._greennet_loop_lock:
            self._greennet_loop_closed = True
            loop = self._greennet_loop
            thread = self._greennet_loop_thread
            if loop is None:
                return
            if not loop.is_closed():
                loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2.0)
        if thread.is_alive():
            logging.warning("GreenNet loop is still draining cancelled tasks")
            return
        self._greennet_loop = None
        self._greennet_loop_thread = None

    def close(self):
        self._shutdown_greennet_loop()
        self._source_download_executor.shutdown()

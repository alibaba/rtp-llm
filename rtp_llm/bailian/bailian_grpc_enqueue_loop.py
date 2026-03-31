"""Async event loop used by Bailian gRPC ``enqueue`` (Uvicorn main loop or dedicated fallback)."""

from __future__ import annotations

import asyncio
import logging
import threading
import time

# Set from FrontendApp startup so async ``enqueue`` shares the Uvicorn loop.
_enqueue_loop: asyncio.AbstractEventLoop | None = None

# Fallback when ``_enqueue_loop`` is unset (e.g. tests / standalone fake without FrontendApp).
_async_loop: asyncio.AbstractEventLoop | None = None
_async_loop_thread: threading.Thread | None = None
_async_loop_lock = threading.Lock()


def set_bailian_grpc_enqueue_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _enqueue_loop
    _enqueue_loop = loop
    logging.info("[BailianGrpc] enqueue_event_loop set (uvicorn main loop)")


def get_bailian_grpc_enqueue_event_loop() -> asyncio.AbstractEventLoop | None:
    return _enqueue_loop


def _get_async_loop() -> asyncio.AbstractEventLoop:
    """Dedicated event loop in its own thread when the uvicorn loop is not wired."""
    global _async_loop, _async_loop_thread
    with _async_loop_lock:
        if _async_loop is not None and _async_loop.is_running():
            return _async_loop
        _async_loop = asyncio.new_event_loop()

        def _run_loop():
            asyncio.set_event_loop(_async_loop)
            _async_loop.run_forever()

        _async_loop_thread = threading.Thread(target=_run_loop, daemon=True)
        _async_loop_thread.start()
        deadline = time.monotonic() + 5.0
        while not _async_loop.is_running() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not _async_loop.is_running():
            raise RuntimeError("Async loop failed to start for enqueue")
    return _async_loop


def resolve_loop_for_enqueue() -> asyncio.AbstractEventLoop:
    """Loop on which ``visitor.enqueue`` must run: uvicorn main loop if set, else fallback."""
    if _enqueue_loop is not None and _enqueue_loop.is_running():
        return _enqueue_loop
    return _get_async_loop()

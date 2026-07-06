import logging
import threading
import time
from typing import Optional


class FrontendShutdownManager:
    """Tracks frontend draining state and accepted in-flight requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._unavailable = False
        self._draining = False
        self._drain_reason = ""
        self._drain_started_at: Optional[float] = None
        self._active_requests = 0

    def start_unavailable(self, reason: str) -> None:
        with self._lock:
            if self._unavailable:
                return
            self._unavailable = True
            self._drain_reason = reason
            self._drain_started_at = time.time()
            active_requests = self._active_requests
        logging.info(
            "Frontend entering pre-stop unavailable state, reason=%s, active_requests=%s",
            reason,
            active_requests,
        )

    def start_draining(self, reason: str) -> None:
        with self._lock:
            if self._draining:
                return
            self._unavailable = True
            self._draining = True
            self._drain_reason = reason
            if self._drain_started_at is None:
                self._drain_started_at = time.time()
            active_requests = self._active_requests
        logging.info(
            "Frontend entering graceful shutdown drain, reason=%s, active_requests=%s",
            reason,
            active_requests,
        )

    def is_draining(self) -> bool:
        with self._lock:
            return self._draining

    def is_unavailable(self) -> bool:
        with self._lock:
            return self._unavailable

    def drain_reason(self) -> str:
        with self._lock:
            return self._drain_reason

    def drain_elapsed_seconds(self) -> float:
        with self._lock:
            if self._drain_started_at is None:
                return 0.0
            return time.time() - self._drain_started_at

    def try_begin_request(self) -> bool:
        with self._lock:
            if self._unavailable:
                return False
            self._active_requests += 1
            return True

    def finish_request(self) -> int:
        with self._lock:
            if self._active_requests <= 0:
                logging.warning(
                    "Frontend active request counter underflow during shutdown drain"
                )
                self._active_requests = 0
                return 0
            self._active_requests -= 1
            active_requests = self._active_requests
        if self.is_draining():
            logging.info(
                "Frontend request finished during drain, active_requests=%s",
                active_requests,
            )
        return active_requests

    def active_request_count(self) -> int:
        with self._lock:
            return self._active_requests

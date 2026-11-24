import logging
import time
from multiprocessing import Lock, Value
from typing import Optional

from rtp_llm.frontend.request_id_generator import (
    EPOCH_2020_MS,
    REQUEST_ID_TIMESTAMP_LIMIT_MS,
)

class ConcurrencyException(Exception):
    pass


class ConcurrencyController:
    def __init__(self, max_concurrency: int = 1) -> None:
        now_ms = time.time_ns() // 1_000_000
        if not EPOCH_2020_MS <= now_ms < REQUEST_ID_TIMESTAMP_LIMIT_MS:
            raise RuntimeError(
                "Frontend clock is outside the request-ID timestamp range; synchronize the host clock before startup"
            )
        self.max_concurrency = max_concurrency
        self.lock = Lock()
        self.current_concurrency = Value("i", 0)
        self.request_counter = Value("Q", 0)

    def get_available_concurrency(self) -> int:
        with self.lock:
            return self.max_concurrency - self.current_concurrency.value

    def increment(self, request_count: int = 1) -> int:
        if request_count < 1:
            raise ValueError("request_count must be positive")
        with self.lock:
            if self.current_concurrency.value >= self.max_concurrency:
                raise ConcurrencyException(
                    f"Concurrency limit {self.max_concurrency} reached"
                )
            # A logical slot contains the wall-clock millisecond and a 12-bit sequence.
            next_slot = self.request_counter.value + 1
            # Preserve the rotating sequence used by request-ID-based DP routing
            # when advancing to a new wall-clock millisecond.
            now_ms = time.time_ns() // 1_000_000
            sequence = next_slot
            if EPOCH_2020_MS <= now_ms < REQUEST_ID_TIMESTAMP_LIMIT_MS:
                sequence = max(next_slot, (now_ms << 12) | (next_slot & 0xFFF))
            last_slot = sequence + request_count - 1
            if (sequence >> 12) < EPOCH_2020_MS or (
                last_slot >> 12
            ) >= REQUEST_ID_TIMESTAMP_LIMIT_MS:
                raise RuntimeError("Cannot allocate request IDs within the signed 64-bit range")
            self.request_counter.value = last_slot
            self.current_concurrency.value += 1
            return sequence

    def decrement(self) -> None:
        with self.lock:
            self.current_concurrency.value -= 1

    def get_request_counter(self) -> int:
        with self.lock:
            return self.request_counter.value

    def __enter__(self) -> None:
        self.increment()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.decrement()


global_controller: Optional[ConcurrencyController] = None


def init_controller(concurrency_config, dp_size=1):
    """Initialize concurrency controller.

    Args:
        concurrency_config: ConcurrencyConfig object.
        dp_size: Data parallel size. If None,
    """

    concurrency_limit = concurrency_config.concurrency_limit
    global_concurrency_limit = concurrency_limit * dp_size
    logging.info(
        f"concurrency_limit : {concurrency_limit}, global_concurrency_limit : {global_concurrency_limit}"
    )
    controller = ConcurrencyController(global_concurrency_limit)
    return controller


def set_global_controller(_global_controller: ConcurrencyController):
    global global_controller
    global_controller = _global_controller


def get_global_controller() -> Optional[ConcurrencyController]:
    return global_controller

import logging
import time
from multiprocessing import Lock, Value
from typing import Optional


class ConcurrencyException(Exception):
    pass


class ConcurrencyController:
    def __init__(self, max_concurrency: int = 1) -> None:
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
            sequence = max(self.request_counter.value + 1, (time.time_ns() // 1_000_000) << 12)
            self.request_counter.value = sequence + request_count - 1
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

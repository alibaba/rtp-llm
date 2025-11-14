import logging
import os
from multiprocessing import Lock, Value
from typing import Optional

from rtp_llm.distribute.worker_info import g_parallel_info


class ConcurrencyException(Exception):
    pass


class ConcurrencyController:
    def __init__(self, max_concurrency: int = 1) -> None:
        self.max_concurrency = max_concurrency
        self.lock = Lock()
        self.current_concurrency = Value("i", 0)
        self.request_counter = Value("i", 0)

    def get_available_concurrency(self) -> int:
        with self.lock:
            return self.max_concurrency - self.current_concurrency.value

    def increment(self) -> None:
        while True:
            with self.lock:
                if self.current_concurrency.value < self.max_concurrency:
                    self.current_concurrency.value += 1
                    self.request_counter.value += 1
                    return self.request_counter.value

                raise ConcurrencyException(
                    f"Concurrency limit {self.max_concurrency} reached"
                )

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


def init_controller(concurrency_config, dp_size=None):
    """Initialize concurrency controller.
    
    Args:
        concurrency_config: ConcurrencyConfig object.
        dp_size: Data parallel size. If None, uses g_parallel_info.dp_size.
    """
    
    if dp_size is None:
        dp_size = g_parallel_info.dp_size
    
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

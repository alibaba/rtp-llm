import os
import time
import logging
from multiprocessing import Process, Lock, Value

class ConcurrencyException(Exception):
    pass

class ConcurrencyController:
    def __init__(self, max_concurrency: int = 1, block: bool = False) -> None:
        self.max_concurrency = max_concurrency
        self.block = block
        self.lock = Lock()
        self.current_concurrency = Value('i', 0)
        self.request_counter = Value('i', 0)

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

            if not self.block:
                raise ConcurrencyException(f"Concurrency limit {self.max_concurrency} reached")
            else:
                time.sleep(0.1)

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
        
global_controller = None

def init_controller():
    concurrency_with_block = bool(int(os.environ.get('CONCURRENCY_WITH_BLOCK', 0)))
    concurrency_limit = int(os.environ.get('CONCURRENCY_LIMIT', 32))
    logging.info(f"concurrency_limit : {concurrency_limit}, concurrency_with_block : {concurrency_with_block}")
    controller = ConcurrencyController(concurrency_limit, block=concurrency_with_block)
    return controller

def set_global_controller(_global_controller):
    global global_controller
    global_controller = _global_controller
    
def get_global_controller():
    if global_controller:
        return global_controller

    if int(os.environ.get('FT_SERVER_TEST', 0)) == 1:
        set_global_controller(init_controller())

    return global_controller
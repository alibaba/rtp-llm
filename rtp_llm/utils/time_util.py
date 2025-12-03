import logging
import time
from functools import wraps
from typing import Optional


def current_time_ms() -> float:
    return time.time() * 1000


class Timer(object):
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self):
        self.start_time = current_time_ms()

    def stop(self):
        self.end_time = current_time_ms()

    def cost_ms(self):
        if self.end_time is None or self.start_time is None:
            raise Exception(
                f"timer not work properly, start_time: {self.start_time}, end_time: {self.end_time}"
            )
        return self.end_time - self.start_time

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self.stop()


def timer_wrapper(description=""):
    def decorator(func):
        # 使用 wraps 保留原函数的元数据
        @wraps(func)
        def wrapped(*args, **kwargs):
            with Timer() as t:
                result = func(*args, **kwargs)
            desc = description if description else func.__name__
            logging.info(f"{desc} took: {t.cost_ms()/1000:.2f}s")
            return result

        return wrapped

    return decorator

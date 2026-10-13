"""Coordinate runtime CUDA Graph capture with host-side CUDA operations.

The gate is process-wide because CUDA's global capture mode checks other
threads too. Ordinary operations remain concurrent; only capture is exclusive.
Never hold an operation scope while waiting for a request result: its producer
may need to capture. Pool methods take the gate before their mutex so capture
(and Python GC during capture preparation) cannot deadlock on a pool lock.
"""

import threading
from contextlib import contextmanager


class CudaGraphGate:
    def __init__(self):
        self._condition = threading.Condition()
        self._readers = 0
        self._waiting_captures = 0
        self._capturing_thread = None
        self._local = threading.local()

    @contextmanager
    def operation(self):
        depth = getattr(self._local, "depth", 0)
        if depth or self._capturing_thread == threading.get_ident():
            self._local.depth = depth + 1
            try:
                yield
            finally:
                self._local.depth = depth
            return
        with self._condition:
            while self._capturing_thread is not None or self._waiting_captures:
                self._condition.wait()
            self._readers += 1
            self._local.depth = 1
        try:
            yield
        finally:
            self._local.depth = 0
            with self._condition:
                self._readers -= 1
                if self._readers == 0:
                    self._condition.notify_all()

    @contextmanager
    def capture(self):
        if getattr(self._local, "depth", 0):
            raise RuntimeError("CUDA capture cannot upgrade an operation scope")
        if self._capturing_thread == threading.get_ident():
            raise RuntimeError("Nested CUDA capture is not supported")
        with self._condition:
            self._waiting_captures += 1
            try:
                while self._capturing_thread is not None or self._readers:
                    self._condition.wait()
                self._capturing_thread = threading.get_ident()
            finally:
                self._waiting_captures -= 1
                self._condition.notify_all()
        try:
            yield
        finally:
            with self._condition:
                self._capturing_thread = None
                self._condition.notify_all()


cuda_graph_gate = CudaGraphGate()

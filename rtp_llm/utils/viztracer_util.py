import asyncio
import functools
import logging
import os
import threading
import time
from contextlib import nullcontext

import viztracer

from rtp_llm.config.py_config_modules import StaticConfig

# Configuration
DEFAULT_OUTPUT_DIR = StaticConfig.profiling_debug_config.log_path
DEFAULT_MIN_DURATION_MS = float(
    StaticConfig.profiling_debug_config.viztracer_min_duration_ms
)
DEFAULT_VIZTRACER_ENABLE = bool(StaticConfig.profiling_debug_config.viztracer_enable)

# Global state
_global_tracer = None
_tracer_lock = threading.RLock()
_pid = os.getpid()


def is_viztracer_enabled(force_trace=False):
    """Check if viztracer is enabled"""
    return DEFAULT_VIZTRACER_ENABLE or force_trace


def _get_global_tracer():
    """Get or create global tracer"""
    global _global_tracer
    if _global_tracer is None:
        with _tracer_lock:
            if _global_tracer is None:
                _global_tracer = viztracer.VizTracer(
                    tracer_entries=2000000,
                    log_gc=True,
                    verbose=0,
                    ignore_frozen=True,
                    log_async=True,
                )
    return _global_tracer


class SmartTraceScope:
    """Efficient tracing scope"""

    def __init__(
        self, name=None, min_duration_ms=None, output_dir=None, force_trace=False
    ):
        self._enabled = is_viztracer_enabled(force_trace)
        if not self._enabled:
            return

        self.name = name
        self.min_duration_ms = min_duration_ms or DEFAULT_MIN_DURATION_MS
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR
        self.force_trace = force_trace
        self.start_time = None
        self.tracer = None

    def _generate_filename(self):
        """Generate trace filename"""
        timestamp = int(time.time() * 1000)
        if self.name:
            safe_name = self.name.replace("/", "_").replace("\\", "_")
            return f"{_pid}_{timestamp}_{safe_name}.json"
        return f"{_pid}_{timestamp}.json"

    def _save_trace(self, duration_ms):
        """Save trace if conditions are met"""
        if self.force_trace or duration_ms >= self.min_duration_ms:
            filename = self._generate_filename()
            filepath = os.path.join(self.output_dir, filename)
            os.makedirs(self.output_dir, exist_ok=True)

            try:
                self.tracer.save(filepath)
                logging.info("Trace saved: %s (%.2fms)", filepath, duration_ms)
                return filepath
            except Exception as e:
                logging.error("Failed to save trace: %s", e)
        return None

    def __enter__(self):
        if not self._enabled:
            return None

        self.tracer = _get_global_tracer()
        self.start_time = time.perf_counter()

        try:
            self.tracer.start()
        except Exception as e:
            logging.error("Failed to start tracer: %s", e)
            return None

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._enabled or not self.start_time:
            return False

        duration_ms = (time.perf_counter() - self.start_time) * 1000

        try:
            self.tracer.stop()
            self._save_trace(duration_ms)
            self.tracer.clear()
        except Exception as e:
            logging.error("Failed to stop tracer: %s", e)

        return False

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)

    @classmethod
    def decorate(
        cls, name=None, min_duration_ms=None, output_dir=None, force_trace=False
    ):
        """Decorator for tracing functions"""
        if not is_viztracer_enabled(force_trace):
            return lambda func: func

        def decorator(func):
            trace_name = name if isinstance(name, str) else None
            name_gen = name if callable(name) else None

            if asyncio.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    final_name = (
                        trace_name
                        or (name_gen and name_gen(func, *args, **kwargs))
                        or func.__name__
                    )
                    async with cls(
                        final_name, min_duration_ms, output_dir, force_trace=force_trace
                    ):
                        return await func(*args, **kwargs)

                return async_wrapper
            else:

                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    final_name = (
                        trace_name
                        or (name_gen and name_gen(func, *args, **kwargs))
                        or func.__name__
                    )
                    with cls(
                        final_name, min_duration_ms, output_dir, force_trace=force_trace
                    ):
                        return func(*args, **kwargs)

                return sync_wrapper

        return decorator


def trace_scope(name=None, min_duration_ms=None, output_dir=None, force_trace=False):
    """Create tracing scope"""
    if not is_viztracer_enabled(force_trace):
        return nullcontext()
    return SmartTraceScope(name, min_duration_ms, output_dir, force_trace=force_trace)


# Aliases
trace_func = SmartTraceScope.decorate

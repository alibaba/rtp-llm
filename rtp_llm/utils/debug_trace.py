import functools
from contextlib import contextmanager

import viztracer


def get_global_tracer():
    tracer = viztracer.get_tracer()
    if tracer is None:
        tracer = viztracer.VizTracer(tracer_entries=2000000, log_gc=True)
    return tracer


@contextmanager
def trace_scope(name):
    tracer = get_global_tracer()
    # tracer = viztracer.VizTracer(tracer_entries=2000000, log_gc=True, register_global=False)
    tracer.start()
    start_time = tracer.getts()
    try:
        yield
    finally:
        end_time = tracer.getts()
        tracer.stop()
        if end_time - start_time >= 100000:  # 100ms
            tracer.save(
                f"/home/silu.zsl/RTP-LLM/github-opensource/{name}.json"
                if name is not None
                else None
            )
        tracer.clear()


def trace_func(name_gen=lambda f, *args, **kwargs: f.__name__):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            name = name_gen(f, *args, **kwargs) if name_gen is not None else None
            with trace_scope(name):
                return f(*args, **kwargs)

        return wrapper

    return decorator

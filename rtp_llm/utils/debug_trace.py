import functools
import time
from contextlib import contextmanager
import viztracer


@contextmanager
def trace_scope(name, time_threshold_ms=200):
    """追踪代码块"""
    tracer = viztracer.VizTracer(
        tracer_entries=2000000, 
        log_gc=True, 
        register_global=False
    )
    tracer.start()
    start_time = tracer.getts()
    try:
        yield
    finally:
        end_time = tracer.getts()
        tracer.stop()
        
        # getts() 返回纳秒，转换为毫秒
        elapsed_ms = (end_time - start_time) / 1_000_000
        
        # if elapsed_ms >= time_threshold_ms:
        if name:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"/home/caihaowen.chw/work/RTP-LLM/github-opensource/{name}_{elapsed_ms:.2f}ms_{timestamp}.json"
            tracer.save(filename)
            print(f"⚠️ 检测到慢调用! 耗时: {elapsed_ms:.2f}ms, 已保存: {filename}")
        # else:
        #     print(f"✓ 正常: {elapsed_ms:.2f}ms (阈值: {time_threshold_ms}ms)")
        
        tracer.clear()


def trace_func(name_gen=lambda f, *args, **kwargs: f.__name__, time_threshold_ms=200):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            name = name_gen(f, *args, **kwargs) if name_gen is not None else None
            with trace_scope(name, time_threshold_ms):
                return f(*args, **kwargs)
        return wrapper
    return decorator

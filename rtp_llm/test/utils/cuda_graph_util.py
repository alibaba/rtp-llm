from contextlib import contextmanager

import torch


@contextmanager
def record_cuda_graph(
    pool=None, stream=None, capture_error_mode: str = "global", dump_path=None
):
    """Record CUDA operations into a graph without replaying it."""
    g = torch.cuda.CUDAGraph()
    if dump_path is not None:
        g.enable_debug_mode()
    with torch.cuda.graph(
        cuda_graph=g, pool=pool, stream=stream, capture_error_mode=capture_error_mode
    ):
        yield g
    if dump_path is not None:
        g.debug_dump(dump_path)


def capture_graph(fn, num_warmups: int = 50):
    """Warm up ``fn``, capture it into a CUDA graph, and replay it once."""
    # Warmup before capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(num_warmups):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    # Capture graph
    with record_cuda_graph() as g:
        fn()
    # Replay graph
    g.replay()
    torch.cuda.synchronize()
    return g

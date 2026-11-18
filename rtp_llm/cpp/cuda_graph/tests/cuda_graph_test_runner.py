"""Lazily import the compiled CUDA Graph test binding staged by setup.py."""

from rtp_llm.test.utils.extension_loader import load_test_extension


def load_cuda_graph_test_binding():
    return load_test_extension(
        "libtest_cuda_graph_runner",
        "rtp_llm/libs/test/libtest_cuda_graph_runner.so",
        "rtp_llm/cpp/cuda_graph/tests/libtest_cuda_graph_runner.so",
    )


class CudaGraphRunner:
    """Instantiate the native runner only after pytest marker selection.

    The CUDA Graph driver modules must remain importable on non-CUDA workers so
    pytest can deselect their H20-marked tests before execution.
    """

    def __new__(cls, *args, **kwargs):
        return load_cuda_graph_test_binding().CudaGraphRunner(*args, **kwargs)

    @staticmethod
    def topologyWidths(*args, **kwargs):
        return load_cuda_graph_test_binding().CudaGraphRunner.topologyWidths(
            *args, **kwargs
        )


__all__ = ["CudaGraphRunner", "load_cuda_graph_test_binding"]

from enum import Enum


class CudaGraphSelectionMode(str, Enum):
    EAGER = "eager"
    DECODE_GRAPH = "decode_graph"
    GENERATION_PREFILL_GRAPH = "generation_prefill_graph"


class GenerationPrefillCudaGraphUnsupportedBackend(RuntimeError):
    """The semantic attention backend cannot run in a generation-prefill CUDA Graph."""

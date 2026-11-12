"""Out-of-process resident weights: keep a model's weights in CPU memfds and CUDA VMM memory across restarts."""

from rtp_llm.resident_weights.store import GLOBAL_SCOPE
from rtp_llm.resident_weights.weights import (
    ResidentWeights,
    RestoredTensor,
    WeightDict,
    checkpoint_digest,
    prepare_host_for_cuda,
    resident_init_scalar,
    resident_init_tensors,
    resident_weights_of,
    retain_host_file,
    should_consume_loaded,
    writable_view,
)

__all__ = [
    "GLOBAL_SCOPE",
    "ResidentWeights",
    "RestoredTensor",
    "WeightDict",
    "checkpoint_digest",
    "prepare_host_for_cuda",
    "resident_init_scalar",
    "resident_init_tensors",
    "resident_weights_of",
    "retain_host_file",
    "should_consume_loaded",
    "writable_view",
]

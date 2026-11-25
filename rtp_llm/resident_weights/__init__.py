"""Out-of-process resident weights: keep a model's weights in CPU memfds and CUDA VMM memory across restarts.

Scope: inference and engine restarts only. Resident weights are mapped read-only,
and RL or other online base-weight updates are not supported; ``WeightManager.update``
rejects them in any process that holds resident target or draft weights.

Lifecycle (``--resident_weights_mode``): the store is a detached process that outlives
every engine. In ``prod`` mode start_server owns an instance store and stops it on a clean
exit; a failure exit keeps it for the next start. In ``dev`` mode nothing stops it. Records
otherwise stay, holding GPU and host memory, until the user runs
``python -m rtp_llm.resident_weights.store drop/stop`` or the container goes away, including
after the feature is turned off.
"""

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

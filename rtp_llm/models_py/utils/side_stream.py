"""One cached side stream per device for intra-layer overlap.

Both the MoE shared expert and the DSA indexer only depend on data that is
already live when they start, so they can run concurrently with the routed
experts / q_b_proj instead of serializing behind them on the main stream. They
never overlap each other (attention finishes before the MoE starts), so a single
stream per device is enough and keeps the caching allocator from reserving a
separate block pool per user.
"""

from typing import Dict

import torch

_side_streams: Dict[int, torch.cuda.Stream] = {}


def is_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


def side_stream(device: torch.device) -> torch.cuda.Stream:
    """The device's shared overlap stream, created on first use outside capture."""
    index = torch.cuda.current_device() if device.index is None else int(device.index)
    stream = _side_streams.get(index)
    if stream is None:
        stream = torch.cuda.Stream(device=torch.device("cuda", index))
        _side_streams[index] = stream
    return stream

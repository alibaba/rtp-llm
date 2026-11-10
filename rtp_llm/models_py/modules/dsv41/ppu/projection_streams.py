"""One owned side stream for independent PPU compute.

The default stream runs query work or routed experts. The side stream runs
KV projections or shared experts. Both results join on the caller stream
before their consumers run.
"""

import threading

import torch


def is_supported(state, x):
    """Decide eligibility before any branch or cache side effect."""
    if (
        state is None
        or x.device.type != "cuda"
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or not x.shape[0]
        or bool(getattr(state, "prefill", True))
        or bool(getattr(state, "idle", True))
        or bool(getattr(state, "is_graph", False))
        or bool(getattr(state, "verify", False))
        or bool(getattr(state, "compacted", False))
        or bool(getattr(state, "restoring_aux", False))
        or getattr(state, "context", None) is not None
        or getattr(state, "active_token_mask", None) is not None
    ):
        return False
    runtime = getattr(state, "runtime", None)
    policy = getattr(runtime, "policy", None)
    model = getattr(runtime, "model", None)
    cache_policy = getattr(getattr(state, "cache", None), "policy", None)
    attn = getattr(state, "attn", None)
    lengths = tuple(getattr(state, "lengths", ()))
    if (
        getattr(policy, "mode", None) != "full"
        or int(getattr(policy, "max_rollback_tokens", -1)) != 0
        or model is None
        or bool(getattr(model, "_is_speculative", True))
        or bool(getattr(attn, "is_target_verify", False))
        or bool(getattr(attn, "is_packed_target_verify", False))
        or bool(getattr(cache_policy, "ced", False))
        or bool(getattr(cache_policy, "bounded_replay", False))
        or bool(getattr(cache_policy, "encoder_replay", False))
        or not lengths
        or any(int(length) != 1 for length in lengths)
        or len(lengths) != x.shape[0]
    ):
        return False
    return (
        not torch.cuda.is_current_stream_capturing()
        and torch.cuda.current_stream(x.device) == torch.cuda.default_stream(x.device)
    )


def _record_stream(value, stream):
    """Retain nested callback results until their consuming stream finishes."""
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            value.record_stream(stream)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _record_stream(item, stream)
    elif isinstance(value, dict):
        for item in value.values():
            _record_stream(item, stream)


class QKVStreams:
    """One side stream and two reusable events, owned for the model lifetime.

    ``query_fn(x)`` may return ``(qr, q)`` and ``kv_fn(x)`` returns normalized
    BF16 KV. The side callback must be rank-local compute with no collectives
    or cache mutation. The shared runtime is nonreentrant; the lock also rejects
    overlapping calls through this resource instead of re-recording events.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("QKVStreams requires a PPU torch.cuda device")
        self.side_stream = torch.cuda.Stream(device=self.device)
        self.producer = torch.cuda.Event(enable_timing=False)
        self.done = torch.cuda.Event(enable_timing=False)
        self._lock = threading.Lock()
        self.calls = 0

    def run(self, x, query_fn, kv_fn):
        """Enqueue independent branches and join before returning to consumers.

        Event reuse is ordered: the caller's previous ``wait_event(done)``
        precedes the next producer record, while the side stream preserves its
        own previous done record before waiting on the next producer. No event
        or stream is allocated per call. Failure propagates after enqueueing a
        join for any submitted side work; there is no serial retry.
        """
        current = torch.cuda.current_stream(self.device)
        if (
            x.device != self.device
            or current != torch.cuda.default_stream(self.device)
            or torch.cuda.is_current_stream_capturing()
        ):
            raise ValueError("QKVStreams requires its device's default eager stream")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("QKVStreams does not permit overlapping event reuse")
        done_recorded = False
        try:
            self.producer.record(current)
            with torch.cuda.stream(self.side_stream):
                self.side_stream.wait_event(self.producer)
                x.record_stream(self.side_stream)
                try:
                    kv_result = kv_fn(x)
                finally:
                    # Also join work submitted by a callback that raises.
                    self.done.record(self.side_stream)
                    done_recorded = True
            query_result = query_fn(x)
            _record_stream(kv_result, current)
            _record_stream(query_result, current)
            self.calls += 1
            return query_result, kv_result
        finally:
            if done_recorded:
                current.wait_event(self.done)
            self._lock.release()

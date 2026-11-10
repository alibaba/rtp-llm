"""Private opt-in scoped chunking for MCore nonsharded gradient clipping."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import math

_active = ContextVar('q36_bounded_clip_active', default=False)
_CHUNK_PARAMS = 128


def install(optimizer_module, clip_module):
    """Replace only MCore's imported function alias; inactive calls delegate exactly."""
    current = optimizer_module.clip_grad_by_total_norm_fp32
    if getattr(current, '_q36_bounded_clip', False):
        if current._q36_original is not clip_module.clip_grad_by_total_norm_fp32:
            raise RuntimeError('bounded clip source changed')
        return
    if current is not clip_module.clip_grad_by_total_norm_fp32:
        raise RuntimeError('unexpected upstream clip alias')
    original = current

    @wraps(original)
    def scoped(parameters, max_norm, total_norm, use_decoupled_grad=False):
        if not _active.get():
            return original(parameters, max_norm, total_norm, use_decoupled_grad)
        if not isinstance(parameters, (list, tuple)):
            raise TypeError('qualified MCore clip expects a materialized parameter list')
        if not isinstance(total_norm, (float, int)) or not math.isfinite(total_norm):
            raise ValueError('qualified global norm must be a finite scalar')
        if not isinstance(max_norm, (float, int)) or not math.isfinite(max_norm):
            raise ValueError('qualified clip threshold must be a finite scalar')
        attr = 'decoupled_grad' if use_decoupled_grad else 'grad'
        spans = {}
        for parameter in parameters:
            grad = getattr(parameter, attr, None)
            if grad is None or not grad.numel():
                continue
            if not grad.is_contiguous():
                raise ValueError('qualified clip requires contiguous gradient views')
            storage = grad.untyped_storage()
            key = (str(grad.device), storage.data_ptr())
            start = grad.storage_offset() * grad.element_size()
            spans.setdefault(key, []).append((start, start + grad.numel() * grad.element_size()))
        for items in spans.values():
            items.sort()
            if any(b[0] < a[1] for a, b in zip(items, items[1:])):
                raise ValueError('overlapping gradient views cannot use bounded clip')
        for start in range(0, len(parameters), _CHUNK_PARAMS):
            chunk = parameters[start:start + _CHUNK_PARAMS]
            if any(getattr(parameter, attr, None) is not None for parameter in chunk):
                original(chunk, max_norm, total_norm, use_decoupled_grad)

    scoped._q36_bounded_clip = True
    scoped._q36_original = original
    optimizer_module.clip_grad_by_total_norm_fp32 = scoped


@contextmanager
def scope():
    token = _active.set(True)
    try:
        yield
    finally:
        _active.reset(token)

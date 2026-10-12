"""Opt-in forward metadata logging; never reads tensor values."""

import itertools
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

import torch

_ENABLED = os.environ.get("RTP_LLM_TRACE_FORWARD") == "1"
_LAYER = os.environ.get("RTP_LLM_TRACE_FORWARD_LAYER", "0")
_CONTEXT = ContextVar("rtp_forward_trace", default=None)
_SEQUENCE = itertools.count(1)
_LOG = logging.getLogger(__name__)


def metadata(value):
    if isinstance(value, torch.Tensor):
        return (f"shape={tuple(value.shape)} stride={value.stride()} "
                f"dtype={value.dtype} device={value.device}")
    return type(value).__name__


def traced_forward(fn):
    @wraps(fn)
    def wrapped(self, inputs, *args, **kwargs):
        if not _ENABLED:
            return fn(self, inputs, *args, **kwargs)
        forward_id = next(_SEQUENCE)
        token = _CONTEXT.set((forward_id, None))
        _LOG.info("[forward_trace] pid=%s forward=%s python.enter model=%s input_ids=(%s)",
                  os.getpid(), forward_id, type(self).__name__, metadata(inputs.input_ids))
        try:
            with torch.profiler.record_function(f"qwen3.forward.{forward_id}"):
                output = fn(self, inputs, *args, **kwargs)
            _LOG.info("[forward_trace] pid=%s forward=%s python.return hidden=(%s)",
                      os.getpid(), forward_id, metadata(output.hidden_states))
            return output
        finally:
            _CONTEXT.reset(token)
    return wrapped


@contextmanager
def layer_scope(layer):
    context = _CONTEXT.get()
    if context is None:
        yield
        return
    token = _CONTEXT.set((context[0], layer))
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def trace_call(stage, fn, *args, **kwargs):
    context = _CONTEXT.get()
    if context is None or (context[1] is not None and _LAYER != "all"
                           and str(context[1]) != _LAYER):
        return fn(*args, **kwargs)
    forward_id, layer = context
    name = f"qwen3.forward.{forward_id}.layer.{layer}.{stage}"
    owner = getattr(fn, "__self__", fn)
    _LOG.info("[forward_trace] pid=%s forward=%s layer=%s stage=%s begin impl=%s inputs=%s",
              os.getpid(), forward_id, layer, stage, type(owner).__name__,
              "; ".join(metadata(arg) for arg in args))
    with torch.profiler.record_function(name):
        output = fn(*args, **kwargs)
    _LOG.info("[forward_trace] pid=%s forward=%s layer=%s stage=%s end output=(%s)",
              os.getpid(), forward_id, layer, stage, metadata(output))
    return output

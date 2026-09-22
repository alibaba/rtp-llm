"""Collect independent W4A16 weights before online FP8 quantization."""

import logging
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from rtp_llm.utils.model_weight import W

_EXTRA_WEIGHTS = ContextVar("w4a16_weights", default=None)


def w4a16_key(name: str, component: str) -> str:
    return f"{name}.w4a16.{component}"


@contextmanager
def collect_w4a16_weights():
    weights = {}
    token = _EXTRA_WEIGHTS.set(weights)
    try:
        yield weights
    finally:
        _EXTRA_WEIGHTS.reset(token)
        weights.clear()


def capture_w4a16_weight(descriptor, tensor, load_config):
    weights = _EXTRA_WEIGHTS.get()
    if weights is None:
        return
    source = getattr(descriptor, "_unquantized_weight", descriptor)
    if not getattr(getattr(source, "config", None), "enable_w4a16_sm120", False):
        return
    if source.name == W.ffn_w2:
        seed = 1
    elif source.name in (W.ffn_w1, W.ffn_w3, W.ffn_w13):
        # w1/w3 must share one seed: the gated w13 merge keeps one signs tensor.
        seed = 0
    else:
        return  # e.g. FFN biases
    if tensor.dtype != torch.bfloat16:
        raise ValueError("W4A16 quantization requires BF16 source weights")
    # FP8 descriptors use different TP layouts; split the source as BF16.
    local = source._split({source.name: tensor}, load_config)[source.name].T
    n, k = local.shape
    from rtp_llm.models_py.kernels.cuda.w4a16_sm120 import quantize_weight, support

    if not support(n, k):
        logging.warning(
            "W4A16 weight %s uses the default path for shape %s", source.name, (n, k)
        )
        return
    packed, scales, signs = quantize_weight(
        local.to(load_config.w4a16_device), seed=seed
    )
    for key, value in (("packed", packed), ("scales", scales), ("signs", signs)):
        weights[w4a16_key(source.name, key)] = value

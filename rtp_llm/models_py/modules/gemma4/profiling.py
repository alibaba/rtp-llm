"""Optional host annotations; they never change tensors or CUDA streams."""

import os
import threading
from collections.abc import Mapping
from contextlib import ExitStack
from functools import wraps

import torch

from rtp_llm.models_py.model_desc.block_map import get_attention_inputs_value

ENABLED = os.environ.get("GEMMA4_PROFILE_RANGES", "0") == "1"


def _range(name):
    stack = ExitStack()
    stack.enter_context(torch.profiler.record_function(name))
    if torch.version.cuda is not None:
        stack.enter_context(torch.cuda.nvtx.range(name))
    return stack


def model_forward_range(forward):
    if not ENABLED:
        return forward

    @wraps(forward)
    def annotated(self, inputs, *args, **kwargs):
        attention = get_attention_inputs_value(inputs)
        if isinstance(attention, Mapping):
            attention = next(iter(attention.values()))
        phase = "prefill" if attention.is_prefill else "decode"
        with _range(f"model/{phase}"):
            return forward(self, inputs, *args, **kwargs)

    return annotated


def register_module_ranges(model):
    if not ENABLED:
        return

    def annotate(module, name):
        state = threading.local()

        def before(_module, _inputs):
            if not hasattr(state, "stack"):
                state.stack = []
            state.stack.append(_range(name))

        def after(_module, _inputs, _output):
            state.stack.pop().close()

        module.register_forward_pre_hook(before)
        module.register_forward_hook(after, always_call=True)

    annotate(model.embed_tokens, "embedding")
    annotate(model.norm, "norm")
    for idx, layer in enumerate(model.layers):
        annotate(layer, f"layer/{idx}")
        annotate(layer.self_attn, "attention/main")
        annotate(layer.mlp, "mlp")
        annotate(layer.router, "moe/router")
        annotate(layer.experts, "moe/experts")
        for child in layer.modules():
            if child.__class__.__name__ == "Gemma4RMSNorm":
                annotate(child, "norm")

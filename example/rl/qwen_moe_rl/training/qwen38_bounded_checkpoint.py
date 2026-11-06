"""Bound FSDP all-gather lifetimes without whole-decoder recomputation.

The decoder's attention region, router, shared expert, and individual already
sharded expert chunks are separate non-reentrant checkpoints. A normalized
checkpoint input is recreated from its residual input by a saved-tensor hook;
the original normalization autograd node still receives the combined gradient.
No parameter, source name, route, loss, gradient, or optimizer is replaced.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
import hashlib
import inspect
import types

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class _NormalizedInput:
    source: torch.Tensor
    normalizer: object
    shape: tuple
    dtype: torch.dtype
    device: torch.device


class _RecreateNormalizedInputs:
    """Save the shared residual storage, never the normalized BF16 activation."""
    def __init__(self, normalized, source, normalizer, audit):
        self.pointer = normalized.untyped_storage().data_ptr()
        self.dtype = normalized.dtype
        self.device = normalized.device
        self.numel = normalized.numel()
        self.source = source
        self.normalizer = normalizer
        self.audit = audit

    def pack(self, tensor):
        if (tensor.device == self.device and tensor.dtype == self.dtype
                and tensor.numel() == self.numel and tensor.is_contiguous()
                and tensor.untyped_storage().data_ptr() == self.pointer):
            self.audit['normalized_inputs_deferred'] += 1
            return _NormalizedInput(self.source, self.normalizer, tuple(tensor.shape),
                                    tensor.dtype, tensor.device)
        return tensor.detach()

    def unpack(self, packed):
        if isinstance(packed, _NormalizedInput):
            # The enclosing decoder's real pre-backward hook already restores
            # its dense norm parameters. Norm itself has no FSDP child group.
            with torch.no_grad():
                value = packed.normalizer(packed.source).reshape(packed.shape)
            if value.dtype != packed.dtype or value.device != packed.device:
                raise RuntimeError('Recreated normalization input changed dtype/device')
            self.audit['normalized_inputs_recreated'] += 1
            return value
        return packed


def _checkpoint(function, *args, **kwargs):
    return checkpoint(function, *args, use_reentrant=False, early_stop=False, **kwargs)


def _bounded_forward(layer, hidden_states, position_embeddings, attention_mask=None,
                     position_ids=None, past_key_values=None, **kwargs):
    if (not layer.training or not torch.is_grad_enabled()
            or not layer._q38_bounded_checkpoint_enabled):
        return layer._q38_original_decoder_forward(hidden_states, position_embeddings,
            attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, **kwargs)
    if past_key_values is not None:
        raise ValueError('Bounded full-policy training must not retain a generation cache')

    def attention_region(value):
        value = layer.input_layernorm(value)
        if layer.block_type == 'linear_attention':
            return layer.linear_attn(hidden_states=value, cache_params=None,
                                     attention_mask=attention_mask, **kwargs)
        value, _ = layer.self_attn(hidden_states=value, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=None,
            position_embeddings=position_embeddings, **kwargs)
        return value

    hidden_states = hidden_states + _checkpoint(attention_region, hidden_states)
    residual = hidden_states
    normalized = _checkpoint(layer.post_attention_layernorm, residual)
    flat = normalized.view(-1, normalized.shape[-1])
    recreator = _RecreateNormalizedInputs(normalized, residual,
        layer.post_attention_layernorm, layer._q38_bounded_checkpoint_audit)

    def shared_region(value):
        return F.sigmoid(layer.mlp.shared_expert_gate(value)) * layer.mlp.shared_expert(value)

    # The checkpoint input saver is outside each checkpoint's internal hooks.
    # It is intercepted here, so all chunk inputs share only the residual
    # activation storage. Recreated norm values participate in forward
    # recomputation, while their gradients flow to the original norm node.
    with torch.autograd.graph.saved_tensors_hooks(recreator.pack, recreator.unpack):
        shared = _checkpoint(shared_region, flat)
        _, routing_weights, selected_experts = _checkpoint(layer.mlp.gate, flat)
        output = torch.zeros_like(flat)
        for chunk in layer.mlp.experts.chunks:
            contributions = _checkpoint(chunk, flat, selected_experts, routing_weights)
            for token_idx, contribution in contributions:
                output.index_add_(0, token_idx, contribution)
        output = output + shared
    return residual + output.reshape_as(residual)


def install_bounded_checkpoints(model, *, expected_layers=92):
    """Preserve all registered modules/parameter names; change AC boundaries."""
    from qwen38_chunked_experts import ChunkedExperts
    layers = list(model.model.layers)
    if len(layers) != expected_layers or any(type(layer).__name__ != 'Qwen3_5MoeDecoderLayer' for layer in layers):
        raise ValueError('Review the exact HF Qwen3.5-MoE decoder before installing bounded checkpoints')
    if any(not isinstance(layer.mlp.experts, ChunkedExperts) for layer in layers):
        raise ValueError('Explicit bounded expert FSDP chunks must already be installed')
    if any(hasattr(layer, '_q38_original_decoder_forward') for layer in layers):
        raise ValueError('Bounded checkpoint installation must happen exactly once')
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
    reviewed_decoder_sha = 'ad3dee6510a0d84b56534e6bfa5aec988c5c557d46867cda41ce893bd92e4499'
    actual_decoder_sha = hashlib.sha256(inspect.getsource(Qwen3_5MoeDecoderLayer.forward).encode()).hexdigest()
    if actual_decoder_sha != reviewed_decoder_sha:
        raise ValueError('The actual HF decoder arithmetic changed; review this source adaptation')
    if model.config.output_router_logits:
        raise ValueError('The bounded initial actor path does not capture duplicate recomputed router outputs')
    signature = list(inspect.signature(Qwen3_5MoeDecoderLayer.forward).parameters)
    if signature != ['self','hidden_states','position_embeddings','attention_mask','position_ids','past_key_values','kwargs']:
        raise ValueError('The actual HF decoder signature changed; review this source adaptation')
    original_names = tuple(name for name, _ in model.named_parameters())
    for layer in layers:
        layer._q38_original_decoder_forward = layer.forward
        layer._q38_bounded_checkpoint_enabled = False
        layer._q38_bounded_checkpoint_audit = dict(normalized_inputs_deferred=0,
            normalized_inputs_recreated=0, whole_decoder_checkpointing=False,
            checkpoint_unit='attention+norm/router/shared/one_expert_chunk')
        layer.forward = types.MethodType(_bounded_forward, layer)
    original_enable = model.gradient_checkpointing_enable
    original_disable = model.gradient_checkpointing_disable

    @functools.wraps(original_enable)
    def enable(gradient_checkpointing_kwargs=None):
        options = dict(gradient_checkpointing_kwargs or {})
        if options.get('use_reentrant',False):
            raise ValueError('Bounded expert checkpoints require non-reentrant autograd')
        options.update(use_reentrant=False,early_stop=False)
        result = original_enable(gradient_checkpointing_kwargs=options)
        for layer in layers:
            # HF GradientCheckpointingLayer.__call__ must not wrap the whole
            # decoder around the independently checkpointed FSDP chunks.
            layer.gradient_checkpointing = False
            layer._q38_bounded_checkpoint_enabled = True
        model.gradient_checkpointing = True
        return result

    @functools.wraps(original_disable)
    def disable():
        result = original_disable()
        for layer in layers:layer._q38_bounded_checkpoint_enabled = False
        model.gradient_checkpointing = False
        return result

    model.gradient_checkpointing_enable = enable
    model.gradient_checkpointing_disable = disable
    if tuple(name for name, _ in model.named_parameters()) != original_names:
        raise RuntimeError('Checkpoint installation changed canonical parameter names')
    model._q38_bounded_checkpoint_source_sha256 = actual_decoder_sha
    return dict(layers=len(layers), canonical_parameter_names_preserved=True,
        checkpoint_unit='attention+norm/router/shared/one_expert_chunk',
        normalized_inputs_saved_as_residual_recomputation=True,
        actual_decoder_source_sha256=model._q38_bounded_checkpoint_source_sha256)

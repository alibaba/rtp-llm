"""Qwen2 draft model, loading directly into the inference module tree."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from rtp_llm.models_py.module_base import _mark_loaded
from rtp_llm.models_py.new_models.model_base import select_fmha_impl_for_layer
from rtp_llm.models_py.new_models.mtp import MTPBlock
from rtp_llm.models_py.new_models.qwen2.language import Qwen2ForCausalLM
from rtp_llm.models_py.new_models.qwen2.trt_checkpoint import Qwen2CheckpointMapper
from rtp_llm.models_py.read_spec import unwrap_checkpoint_tensor
if TYPE_CHECKING:
    from rtp_llm.ops.compute_ops import PyModelInputs


class Qwen2MTPForCausalLM(Qwen2ForCausalLM):
    """Match Qwen2MtpModel's hidden-first projection and decoder layout.

    This is not Qwen3-Next's embedding-first MTP convention. The checkpoint
    keeps the input fusion and final norm under decoder layer zero; all
    decoder layers otherwise retain their checkpoint indices.
    """

    WEIGHTS_MAPPER = Qwen2CheckpointMapper(
        exact_mapping={
            "model.layers.0.e_norm.weight": "mtp_block.e_norm.weight",
            "model.layers.0.h_norm.weight": "mtp_block.h_norm.weight",
            "model.layers.0.eh_proj.weight": "mtp_block.fc.weight",
            "model.layers.0.final_head.norm.weight": "norm.weight",
        },
    )

    def __init__(self, model_config: Any, load_config: Any):
        super().__init__(model_config, load_config)
        cfg = self._dense_config(model_config, load_config)
        self.mtp_block = MTPBlock(
            cfg["hidden_size"],
            rms_norm_eps=cfg["rms_norm_eps"],
            reverse_concat=True,
            params_dtype=cfg["params_dtype"],
        )
        # Main consumes layer zero's fusion only, but the checkpoint format
        # declares these tensors for all draft layers. Validate without retaining
        # unused device storage.
        hidden = cfg["hidden_size"]
        self._extra_draft_shapes = {
            f"layers.{i}.{name}": shape
            for i in range(1, len(self.layers))
            for name, shape in (
                ("e_norm.weight", (hidden,)),
                ("h_norm.weight", (hidden,)),
                ("eh_proj.weight", (hidden, 2 * hidden)),
                ("final_head.norm.weight", (hidden,)),
            )
        }
        self._extra_draft_seen = set()
        # These optional integer maps belong to the sampler, not the decoder.
        self.register_buffer("d2t", None)
        self.register_buffer("t2d", None)

    def load_weights(self, weights):
        weights = weights.items() if isinstance(weights, dict) else weights

        def decoder_weights():
            for name, tensor in weights:
                mapped = self.WEIGHTS_MAPPER.map_name(name)
                if mapped in ("d2t", "t2d"):
                    tensor, _ = unwrap_checkpoint_tensor(tensor)
                    if (
                        tensor.dtype
                        not in (
                            torch.bool,
                            torch.uint8,
                            torch.int8,
                            torch.int16,
                            torch.int32,
                            torch.int64,
                        )
                        or tensor.ndim != 1
                    ):
                        raise ValueError(
                            f"{name} must be a one-dimensional integer vocabulary map"
                        )
                    if getattr(self, mapped) is not None:
                        raise RuntimeError(f"Duplicate checkpoint weight: {name}")
                    self._buffers[mapped] = tensor.to(dtype=torch.int64, copy=True)
                    _mark_loaded(self, mapped)
                    continue
                # OldLoader declares these tensors for every layer, but the
                # actual Qwen2MtpModel only consumes layer zero's input fusion
                # and final norm. They are not additional decoder operations.
                if mapped in self._extra_draft_shapes:
                    value, _ = unwrap_checkpoint_tensor(tensor)
                    expected = self._extra_draft_shapes[mapped]
                    # As for layer zero, both on-disk EH orientations are valid.
                    shapes = (expected, expected[::-1]) if mapped.endswith(".eh_proj.weight") else (expected,)
                    if (
                        mapped in self._extra_draft_seen
                        or not value.is_floating_point()
                        or tuple(value.shape) not in shapes
                    ):
                        raise ValueError(f"Invalid or duplicate draft tensor {name}")
                    self._extra_draft_seen.add(mapped)
                    continue
                yield name, tensor

        super().load_weights(decoder_weights())

    def validate_weights_loaded(self, loaded_tensor_ids=None):
        super().validate_weights_loaded(loaded_tensor_ids)
        missing = self._extra_draft_shapes.keys() - self._extra_draft_seen
        if missing:
            raise RuntimeError(
                f"Missing declared draft fusion tensors: {sorted(missing)}"
            )
        if (self.d2t is None) != (self.t2d is None):
            raise RuntimeError("Qwen2 MTP vocabulary maps require both d2t and t2d")

    def runtime_weight_view(self):
        weights = super().runtime_weight_view()
        if self.d2t is not None:
            weights["multi_tokens_predict_d2t_map"] = self.d2t
            weights["multi_tokens_predict_t2d_map"] = self.t2d
        return weights

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None):
        if inputs.input_hiddens is None:
            raise ValueError("Qwen2 MTP requires input_hiddens")
        hidden_states = self.mtp_block(
            self.embed_tokens(inputs.input_ids), inputs.input_hiddens
        )
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                select_fmha_impl_for_layer(fmha_impl, self.kv_cache, i),
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        from rtp_llm.ops.compute_ops import PyModelOutputs

        return PyModelOutputs(self.norm(hidden_states))

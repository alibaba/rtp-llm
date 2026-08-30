"""GLM-4.7 NextN (MTP) draft model.

GLM-4.7 carries one next-token-prediction layer inside the main checkpoint
(``num_nextn_predict_layers=1``, sitting at ``model.layers.<num_hidden_layers>``).
Structurally it is a normal GLM-4.7 MoE decoder layer plus the DeepSeek-style
MTP scaffold: ``enorm``/``hnorm`` on the two inputs, ``eh_proj`` fusing them, and
its own ``embed_tokens`` and ``shared_head`` (norm + head).

The layer itself is built from :class:`GenericMoeDecoderLayer`, the same class the
target model uses, so the draft issues exactly the kernels the target does. Only
the fusion step and the MTP-private output norm live here.

Two GLM-specific details that differ from the reference MTP models:

* the concat order is ``[embed; hidden]``, not qwen2_mtp's ``[hidden; embed]``.
  GLM's ``eh_proj`` is trained on ``[e; h]``; swapping it produces plausible but
  wrong draft tokens, which shows up only as a low acceptance rate.
* ``GenericMoeDecoderLayer`` carries the residual out of band, so the output norm
  is :class:`RMSResNorm` over ``(hidden_states, residual)``, not a plain
  :class:`RMSNorm` over the hidden alone.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import select_fmha_impl_for_layer
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeDecoderLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import Embedding, LinearFactory, RMSNorm, RMSResNorm
from rtp_llm.ops import HWKernelConfig, MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

# Token slice for the eh_proj fusion. The concat it feeds is
# (num_tokens, 2 * hidden_size); at 190K prompt tokens that is ~3.7 GiB in bf16,
# the single largest block prefill asks for. Slicing keeps the transient at
# (slice, 2 * hidden_size) and is exact rather than an approximation: RMSNorm
# normalises per row and a linear is row-independent, so each slice produces the
# same values it would have as part of the whole. A decode step is far below the
# threshold and takes the unsliced path.
_EH_PROJ_CHUNK_TOKENS_DEFAULT = 8192
_EH_PROJ_CHUNK_ENV = "GLM4_MOE_MTP_EH_PROJ_CHUNK_TOKENS"


def _eh_proj_chunk_tokens() -> int:
    raw = os.environ.get(_EH_PROJ_CHUNK_ENV)
    if not raw:
        return _EH_PROJ_CHUNK_TOKENS_DEFAULT
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_EH_PROJ_CHUNK_ENV} must be an integer, got {raw!r}"
        ) from exc
    if value <= 0:
        raise ValueError(f"{_EH_PROJ_CHUNK_ENV} must be positive, got {value}")
    return value


class Glm4MoeMtpModel(GptModelBase):
    """Single-layer GLM-4.7 NextN draft over the generic MoE decoder layer."""

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config: MoeConfig,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config: Optional["HWKernelConfig"] = None,
        device_resource_config=None,
    ) -> None:
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        if self.layer_num != 1:
            raise RuntimeError(
                "GLM-4.7 NextN draft has exactly one layer, got "
                f"{self.layer_num}; check Glm4MoeNextN._create_config"
            )

        self._hidden_size = model_config.hidden_size
        self._eh_proj_chunk = _eh_proj_chunk_tokens()

        layer_weights = weights.weights[0]
        self.embed_tokens = Embedding(
            model_config,
            parallelism_config,
            weights.get_global_weight(W.embedding),
        )
        # enorm/hnorm are standalone (no residual): they normalise the embedding
        # and the target's last hidden separately, before the eh_proj fusion.
        self.enorm = RMSNorm(
            layer_weights[W.multi_tokens_predict_enorm],
            eps=model_config.layernorm_eps,
        )
        self.hnorm = RMSNorm(
            layer_weights[W.multi_tokens_predict_hnorm],
            eps=model_config.layernorm_eps,
        )
        self.eh_proj = LinearFactory.create_linear_from_weights(
            layer_weights,
            W.multi_tokens_predict_eh_proj,
            quant_config=model_config.quant_config,
            hw_kernel_config=py_hw_kernel_config,
        )

        enable_cuda_graph = bool(
            py_hw_kernel_config is not None and py_hw_kernel_config.enable_cuda_graph
        )
        self.layers = nn.ModuleList(
            [
                GenericMoeDecoderLayer(
                    model_config,
                    parallelism_config,
                    layer_weights,
                    weights.global_weights,
                    0,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph=enable_cuda_graph,
                    hw_kernel_config=py_hw_kernel_config,
                )
            ]
        )
        # The draft has its own output norm (shared_head.norm), not the target's
        # model.norm, and it arrives under the MTP final-layernorm key.
        self.norm = RMSResNorm(
            layer_weights[W.multi_tokens_predict_final_ln_gamma],
            eps=model_config.layernorm_eps,
        )

    def _fuse_embed_and_hidden(
        self, input_ids: torch.Tensor, last_hidden: torch.Tensor
    ) -> torch.Tensor:
        """``eh_proj([enorm(embed); hnorm(hidden)])``, in token slices."""
        num_tokens = input_ids.shape[0]
        step = self._eh_proj_chunk
        if num_tokens <= step:
            e_norm = self.enorm(self.embed_tokens(input_ids))
            h_norm = self.hnorm(last_hidden)
            # GLM order: embedding first. See the module docstring.
            eh_concat = torch.cat([e_norm, h_norm], dim=-1)
            del e_norm, h_norm
            return self.eh_proj(eh_concat)

        out = None
        for start in range(0, num_tokens, step):
            end = min(start + step, num_tokens)
            e_norm = self.enorm(self.embed_tokens(input_ids[start:end]))
            h_norm = self.hnorm(last_hidden[start:end])
            eh_concat = torch.cat([e_norm, h_norm], dim=-1)
            del e_norm, h_norm
            piece = self.eh_proj(eh_concat)
            del eh_concat
            if out is None:
                # Allocated from the first slice so dtype and width come from
                # eh_proj itself rather than being assumed here.
                out = torch.empty(
                    (num_tokens, piece.shape[-1]),
                    device=piece.device,
                    dtype=piece.dtype,
                )
            out[start:end].copy_(piece)
            del piece
        return out

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        hidden_states = self._fuse_embed_and_hidden(input_ids, inputs.input_hiddens)
        if hidden_states.shape[-1] != self._hidden_size:
            # A missing transpose on eh_proj lands here instead of producing
            # silently wrong drafts further down.
            raise RuntimeError(
                "GLM-4.7 NextN eh_proj must map the [embed; hidden] concat back "
                f"to hidden_size={self._hidden_size}, got {hidden_states.shape[-1]}"
            )

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        residual = torch.zeros_like(hidden_states)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            layer_fmha_impl = select_fmha_impl_for_layer(fmha_impl, self.kv_cache, i)
            output = decoder_layer(
                hidden_states,
                residual,
                layer_fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
            hidden_states = output.hidden_states
            residual = output.residual
        hidden_states, _ = self.norm(hidden_states, residual)
        return PyModelOutputs(hidden_states)


__all__ = ["Glm4MoeMtpModel"]

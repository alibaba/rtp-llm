"""Python model for Qwen2.5-Omni talker running inside rtp-llm's C++ engine.

The talker's custom embedding differs from standard QWenV2:
  embed_tokens(codec_token) + thinker_hidden_state → proj(3584→896) → transformer → norm

The C++ engine handles the autoregressive loop, KV cache, and FMHA attention.
This Python model is called by the engine at each step via forward().
"""

import threading
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3 import Qwen3DecoderLayer
from rtp_llm.models_py.modules import Embedding, RMSNorm
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class Qwen2_5OmniTalkerModel(GptModelBase):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        quant_config: Optional[object] = None,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )

        self.embed_tokens = Embedding(
            config, parallelism_config, weights.get_global_weight(W.embedding)
        )

        self.proj_weight = weights.get_global_weight("thinker_to_talker_proj.weight")
        self.proj_bias = weights.get_global_weight("thinker_to_talker_proj.bias")

        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    config,
                    parallelism_config,
                    idx,
                    weights.weights[idx],
                    quant_config,
                    py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )

        self._thinker_hidden_states: Optional[torch.Tensor] = None
        self._step: int = 0
        self._thinker_state_lock = threading.Lock()

    def set_thinker_hidden_states(self, hidden_states: torch.Tensor) -> None:
        """Store thinker hidden states before starting generation.

        Args:
            hidden_states: [num_thinker_tokens, embedding_size] tensor
        """
        with self._thinker_state_lock:
            self._thinker_hidden_states = hidden_states.to(
                device=self.proj_weight.device, dtype=self.proj_weight.dtype
            )
            self._step = 0

    def clear_thinker_hidden_states(self) -> None:
        with self._thinker_state_lock:
            self._thinker_hidden_states = None
            self._step = 0

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        num_tokens = input_ids.shape[0]

        inputs_embeds = self.embed_tokens(input_ids)

        with self._thinker_state_lock:
            if self._thinker_hidden_states is not None:
                max_idx = self._thinker_hidden_states.shape[0]
                start = min(self._step, max_idx - 1)
                end = min(self._step + num_tokens, max_idx)
                thinker_hs = self._thinker_hidden_states[start:end]

                if thinker_hs.shape[0] < num_tokens:
                    pad_count = num_tokens - thinker_hs.shape[0]
                    last_hs = self._thinker_hidden_states[-1:].expand(pad_count, -1)
                    thinker_hs = torch.cat([thinker_hs, last_hs], dim=0)

                inputs_embeds = inputs_embeds + thinker_hs
                self._step += num_tokens

        hidden_states = F.linear(inputs_embeds, self.proj_weight, self.proj_bias)

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            select_block_map_for_layer(inputs.attention_inputs, i)
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)

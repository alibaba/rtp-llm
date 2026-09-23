"""MiniMax-M3.1 runtime model specializations."""

import logging
from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.model_desc.minimax_m3 import (
    MiniMaxM3DecoderLayer,
    MiniMaxM3Model,
)
from rtp_llm.models_py.modules import DenseMLP
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs


class _MiniMaxM31MSAQueryContext:
    """No-op FMHA context for the all-sparse MSA model.

    MiniMax-M3.1 has an MSA attention layer in every transformer block.  The
    regular ``GenericMoeModel`` still asks the model for an FMHA context, but
    MSA performs its own paged-cache/indexer path and must not instantiate a
    CP FlashInfer/MLA implementation.  Returning this context keeps the
    generic model output contract (``fmha_params is None``) without entering
    ``fill_mla_params``.
    """

    fmha_params = None

    def prepare_cuda_graph(self, _attn_inputs) -> None:
        return None

    def support_cuda_graph(self) -> bool:
        return True


class _MockNVFP4Moe(nn.Module):
    """Shared expert only; skips unsupported M3.1 routed NVFP4 experts."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
    ) -> None:
        super().__init__()
        self.shared_expert = DenseMLP(
            config.activation_type,
            parallelism_config,
            weights,
            config.quant_config,
            swiglu_oai_params=(config.swiglu_alpha, config.swiglu_limit),
        )

    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.shared_expert(hidden_states)


class MiniMaxM31DecoderLayer(MiniMaxM3DecoderLayer):
    def _create_mlp(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        moe_config,
        max_generate_batch_size: int,
        enable_cuda_graph: bool,
        hw_kernel_config: Optional[HWKernelConfig],
        layer_idx: int,
    ) -> nn.Module:
        if bool(getattr(config, "mock_nvfp4_moe", False)) and (
            layer_idx in config.moe_layer_index
        ):
            if layer_idx == config.moe_layer_index[0]:
                logging.warning(
                    "M3_M31_MOCK_NVFP4_MOE is active: routed NVFP4 experts are "
                    "skipped and only the MXFP8 shared expert runs; this mode "
                    "validates structure only"
                )
            return _MockNVFP4Moe(config, parallelism_config, weights)
        return super()._create_mlp(
            config,
            parallelism_config,
            weights,
            moe_config,
            max_generate_batch_size,
            enable_cuda_graph,
            hw_kernel_config,
            layer_idx,
        )


class MiniMaxM31Model(MiniMaxM3Model):
    decoder_layer_cls = MiniMaxM31DecoderLayer

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> Any:
        # Every M3.1 layer is MSAAttention, including target verification.  MSA
        # owns its paged cache, indexer and speculative metadata and does not
        # consume GenericMoeModel's FMHA implementation.  Constructing the
        # inherited M3 dense/FlashInfer target-verify context is both redundant
        # and incorrect for the all-sparse checkpoint (it can enter
        # fill_mla_params despite there being no full-attention layer).
        del inputs, is_cuda_graph
        return _MiniMaxM31MSAQueryContext()


__all__ = ["MiniMaxM31DecoderLayer", "MiniMaxM31Model"]

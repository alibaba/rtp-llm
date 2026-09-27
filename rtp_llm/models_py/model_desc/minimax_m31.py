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

    def _prepare_prefill_moe_chunk_plan(self, inputs, hidden_states, layers):
        from rtp_llm.models_py.model_desc.generic_moe import (
            cuda_graph_capture_forward_enabled,
            cuda_graph_warmup_forward_enabled,
        )

        attn = inputs.attention_inputs
        if (
            not attn.is_prefill
            or getattr(attn, "is_target_verify", False)
            or cuda_graph_capture_forward_enabled()
            or cuda_graph_warmup_forward_enabled()
            or (hidden_states.is_cuda and torch.cuda.is_current_stream_capturing())
        ):
            return None

        import torch.distributed as dist

        from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_nvfp4_wrapper import (
            MegaMoeNvfp4Wrapper,
        )
        from rtp_llm.models_py.modules.glm5_mega_moe.prefill_chunk_plan import (
            PrefillChunkPlan,
            local_chunk_count,
        )

        routed = [
            getattr(layer.mlp, "fused_moe", None) for layer in layers[: self.layer_num]
        ]
        nvfp4 = [m.mega_moe for m in routed if isinstance(m, MegaMoeNvfp4Wrapper)]
        if not nvfp4:
            return None
        first = nvfp4[0]
        group = first._mega_group
        if dist.get_world_size(group) == 1:
            return None
        capacity = int(first._mega_buf.num_max_tokens_per_rank)
        if any(
            m._mega_group is not group
            or int(m._mega_buf.num_max_tokens_per_rank) != capacity
            for m in nvfp4
        ):
            raise ValueError(
                "NVFP4 prefill layers must share EP group and chunk capacity"
            )
        if any(
            m is not None and not isinstance(m, MegaMoeNvfp4Wrapper) for m in routed
        ):
            raise ValueError(
                "mixed routed MoE strategies cannot use NVFP4 prefill plan"
            )
        # One forward-local collective, including fake/short ranks. Use actual
        # local embedding rows (already CP-local), never divide by CP again.
        # +/- capacity also detects inconsistent buffer limits across ranks.
        agreed = torch.tensor(
            [local_chunk_count(hidden_states.shape[0], capacity), capacity, -capacity],
            dtype=torch.int64,
            device=hidden_states.device,
        )
        dist.all_reduce(agreed, op=dist.ReduceOp.MAX, group=group)
        chunks, max_capacity, neg_min_capacity = agreed.tolist()
        if max_capacity != -neg_min_capacity:
            raise ValueError("NVFP4 prefill chunk capacity differs across EP ranks")
        return PrefillChunkPlan(capacity, chunks)

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

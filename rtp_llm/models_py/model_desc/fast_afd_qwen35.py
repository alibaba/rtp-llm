"""Qwen3.5 MoE Attention/Expert disaggregation.

The attention rank owns the decoder, routing, and shared expert. The expert
rank owns only routed expert weights and answers per-layer requests. The
transport protocol lives in ``distributed.fast_afd`` so model construction
does not depend on a particular point-to-point implementation.
"""

import logging
from typing import List

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.distributed.fast_afd import FastAFDClient, FastAFDExpertService
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3_next import Qwen35Model
from rtp_llm.models_py.modules import FusedMoeFactory
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class Qwen35AFDAttentionModel(Qwen35Model):
    """Full Qwen3.5 decoder with routed MoE forwarded to the expert rank."""

    # The protocol entry is required independently of whether C++ splits inputs.
    requires_micro_batch_forward = True

    # Each self.forward below already applies Qwen35Model's final RMSResNorm.
    micro_batch_outputs_are_normalized = True

    # PyWrappedModel must omit its synthetic second microbatch when no split
    # is planned. Replaying it would update GDN and KV state twice.
    use_real_micro_batches_only = True

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        ffn_config = parallelism_config.ffn_disaggregate_config
        service_rank = ffn_config.attention_dp_size * ffn_config.attention_tp_size
        client = FastAFDClient(
            service_rank=service_rank,
            hidden_size=model_config.hidden_size,
            top_k=model_config.moe_k,
            device=torch.device(weights.device),
            activation_dtype=weights.dtype,
            expert_count=model_config.expert_num,
        )
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            moe_config,
            max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
            remote_expert_client=client,
        )
        self.fast_afd_client = client
        self.fast_afd_global_idle = False

    def forward_micro_batch(self, inputs: List[PyModelInputs]) -> List[PyModelOutputs]:
        # The service accepts routed-expert requests in the order each AG
        # produces them, so the ordinary decoder forward can be reused for
        # each real microbatch, including Qwen3.5's recurrent GDN state path.
        self.fast_afd_client.begin_step()
        try:
            outputs = [self.forward(micro_input) for micro_input in inputs]
        except BaseException:
            try:
                self.fast_afd_client.abort()
            except Exception:
                logging.exception("FastAFD failed to abort the expert step")
            raise
        self.fast_afd_client.finish()
        self.fast_afd_global_idle = self.fast_afd_client.global_idle
        return outputs

    def stop_fast_afd(self) -> None:
        """Notify the expert rank after the attention engine loop exits."""
        self.fast_afd_client.stop()


class Qwen35AFDExpertModel(GptModelBase):
    """One-rank routed expert service for all Qwen3.5 MoE layers."""

    requires_micro_batch_forward = True

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        # The transport uses the union world (N attention ranks + this rank),
        # but this one expert rank owns the complete routed expert set. MoE
        # strategy selection must see a rank-local execution topology.
        local_expert_parallelism = ParallelismConfig()
        local_expert_parallelism.tp_size = 1
        local_expert_parallelism.tp_rank = 0
        local_expert_parallelism.dp_size = 1
        local_expert_parallelism.dp_rank = 0
        local_expert_parallelism.ep_size = 1
        local_expert_parallelism.ep_rank = 0
        local_expert_parallelism.ffn_tp_size = 1
        local_expert_parallelism.ffn_tp_rank = 0
        local_expert_parallelism.world_size = 1
        local_expert_parallelism.world_rank = 0
        local_expert_parallelism.local_rank = parallelism_config.local_rank
        local_expert_parallelism.local_world_size = parallelism_config.local_world_size
        if len(weights.weights) != model_config.num_layers:
            raise ValueError(
                "FastAFD expert rank needs one routed weight set per "
                f"Qwen3.5 layer: expected {model_config.num_layers}, "
                f"got {len(weights.weights)}"
            )
        self.layers = nn.ModuleList()
        for layer_idx, layer_weights in enumerate(weights.weights):
            if W.moe_w1 not in layer_weights or W.moe_w2 not in layer_weights:
                raise ValueError(
                    f"FastAFD expert rank missing routed expert weights "
                    f"for Qwen3.5 layer {layer_idx}"
                )
            adapter = MoEConfigAdapter(
                model_config=model_config,
                parallelism_config=local_expert_parallelism,
                moe_config=moe_config,
                quant_config=model_config.quant_config,
                enable_cuda_graph=False,
            )
            # Qwen3.5's independently sized shared expert stays on AG, even
            # when a local MoE strategy could fuse it into routed execution.
            adapter.n_shared_experts = 0
            adapter.has_shared_expert_gate = False
            fused_moe = FusedMoeFactory().create_fused_moe(adapter, layer_weights)
            if fused_moe.includes_shared_expert:
                raise RuntimeError(
                    f"FastAFD expert rank selected shared-expert backend "
                    f"for Qwen3.5 layer {layer_idx}"
                )
            self.layers.append(fused_moe)

        ffn_config = parallelism_config.ffn_disaggregate_config
        attention_rank_count = (
            ffn_config.attention_dp_size * ffn_config.attention_tp_size
        )
        self.fast_afd_service = FastAFDExpertService(
            attention_ranks=list(range(attention_rank_count)),
            fused_moe_by_layer=dict(enumerate(self.layers)),
            hidden_size=model_config.hidden_size,
            top_k=model_config.moe_k,
            device=torch.device(weights.device),
            activation_dtype=weights.dtype,
            expert_count=model_config.expert_num,
        )
        self.fast_afd_service_finished = False
        self.fast_afd_global_idle = False

    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        raise RuntimeError("FastAFD expert rank must run forward_micro_batch")

    def forward_micro_batch(self, inputs: List[PyModelInputs]) -> List[PyModelOutputs]:
        if inputs:
            raise ValueError("FastAFD expert rank does not accept model inputs")
        self.fast_afd_service_finished = self.fast_afd_service.serve_until_done()
        self.fast_afd_global_idle = self.fast_afd_service.global_idle
        return []

"""K3 MTP contract from vLLM v0.28.0 (2cf0a6915ce5), independent of EAGLE3."""

from typing import Optional

import torch
from torch import nn

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather_trim,
    get_process_group,
)
from rtp_llm.models_py.distributed.sequence_parallel import (
    SequenceParallelLayout,
    local_physical_token_view,
    sequence_parallel_layout_from_attention_inputs,
)
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.kimi_k3 import resolve_kimi_k3_moe_strategy
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import Embedding, LinearFactory, RMSNorm
from rtp_llm.models_py.modules.kimi_k3.all_gather_gemm import (
    all_gather_gemm,
    configure_all_gather_gemm,
)
from rtp_llm.models_py.modules.kimi_k3.gemm_reduce_scatter import (
    configure_gemm_reduce_scatter,
    gemm_reduce_scatter,
)
from rtp_llm.models_py.modules.kimi_k3.mla import KimiK3MLA
from rtp_llm.models_py.modules.kimi_k3.moe import KimiK3LatentMoE
from rtp_llm.models_py.modules.kimi_k3.moe_se import KimiK3LatentMoESE
from rtp_llm.models_py.modules.kimi_k3.utils import (
    collective_gemm_workspace_global_tokens,
    prefill_chunk_tokens,
    sequence_offsets,
)
from rtp_llm.ops.compute_ops import PyModelOutputs
from rtp_llm.utils.model_weight import W


def mtp_positions(inputs):
    """NoPE batches need not carry explicit positions; reconstruct from cache lengths."""
    positions = inputs.combo_position_ids
    total = inputs.input_ids.numel()
    if positions is not None and positions.numel():
        if positions.ndim != 1 or positions.numel() != total:
            raise ValueError("K3 MTP requires one absolute position per input token")
        return positions
    attn = inputs.attention_inputs
    device = inputs.input_ids.device
    lengths = attn.input_lengths.to(device=device, dtype=torch.long)
    # Graph prefill keeps a sequence_lengths scratch buffer even though every
    # request is context. Plain decode, conversely, leaves prefix_lengths unset.
    # Use the execution mode before interpreting these optional buffers.
    decode = (
        lengths.new_empty(0)
        if getattr(attn, "is_prefill", False) or attn.sequence_lengths is None
        else attn.sequence_lengths.to(device=device, dtype=torch.long)
    )
    prefixes = (
        lengths.new_zeros(lengths.numel() - decode.numel())
        if attn.prefix_lengths is None
        else attn.prefix_lengths.to(device=device, dtype=torch.long)
    )
    query_lengths = torch.cat((torch.ones_like(decode), lengths[decode.numel() :]))
    offsets = torch.cat((decode, prefixes))
    if query_lengths.numel() != offsets.numel():
        raise ValueError("K3 MTP positions require matching query/cache lengths")
    if total == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if query_lengths.numel() == 0:
        raise ValueError("K3 MTP tokens require nonempty request metadata")
    ends = query_lengths.cumsum(0)
    starts = torch.cat((ends.new_zeros(1), ends[:-1]))
    tokens = torch.arange(total, device=device)
    requests = torch.searchsorted(ends, tokens, right=True).clamp(max=ends.numel() - 1)
    return tokens - starts[requests] + offsets[requests]


class KimiK3MtpLayer(nn.Module):
    def __init__(self, config, parallelism, weights, moe_strategy):
        super().__init__()
        self.attn_tp_size = int(parallelism.get_attn_tp_size())
        self.enorm = RMSNorm(weights["kimi_k3.mtp.enorm"], config.layernorm_eps)
        self.hnorm = RMSNorm(weights["kimi_k3.mtp.hnorm"], config.layernorm_eps)
        self.eh_proj = LinearFactory.create_linear_from_weights(
            weights, "kimi_k3.mtp.eh_proj"
        )
        self.input_norm = RMSNorm(weights[W.pre_ln_gamma], config.layernorm_eps)
        self.post_norm = RMSNorm(weights[W.post_ln_gamma], config.layernorm_eps)
        # vLLM MLA uses config.rms_norm_eps, unlike the legacy RTP target default.
        self.attention = KimiK3MLA(
            config, parallelism, weights, 0, latent_norm_eps=config.layernorm_eps
        )
        moe_cls = (
            KimiK3LatentMoESE if moe_strategy == "mega_moe_se" else KimiK3LatentMoE
        )
        self.moe = moe_cls(config, parallelism, weights, 0)

    @staticmethod
    def _local_projection(x, weight):
        return (
            weight(x)
            if not isinstance(weight, torch.Tensor)
            else torch.matmul(x, weight)
        )

    def forward(
        self,
        embedding,
        previous_h,
        positions,
        fmha_impl,
        kv_cache,
        attention_inputs,
        *,
        sp_layout: SequenceParallelLayout,
    ):
        embedding = torch.where(positions.reshape(-1, 1) == 0, 0, embedding)
        x = self.eh_proj(
            torch.cat(
                (self.enorm(embedding), self.hnorm(previous_h.contiguous())), dim=-1
            )
        )
        normalized = self.input_norm(x)
        projection_weights = self.attention.tp_input_projection_weights()
        projected_qkv_a = (
            [
                self._local_projection(normalized, weight)
                for weight in projection_weights
            ]
            if self.attn_tp_size == 1
            else all_gather_gemm(
                normalized,
                projection_weights,
                logical_m=sp_layout.tokens.physical_tokens,
            )
        )
        attention_projection_input = self.attention(
            normalized,
            fmha_impl,
            kv_cache,
            attention_inputs=attention_inputs,
            sp_layout=sp_layout,
            projected_qkv_a=projected_qkv_a,
        )
        output_weight = self.attention.output_projection_weight()
        attention_output = (
            self._local_projection(attention_projection_input, output_weight)
            if self.attn_tp_size == 1
            else gemm_reduce_scatter(
                attention_projection_input,
                output_weight,
                get_process_group(Group.TP),
                pad_rows=False,
            )
        )
        a = x + attention_output
        local_valid_tokens = (
            sp_layout.tokens.local_valid_tokens
            if sp_layout.tokens.local_valid_tokens < sp_layout.tokens.local_tokens
            else None
        )
        return a + self.moe(self.post_norm(a), valid_token_count=local_valid_tokens)


class KimiK3MtpModel(GptModelBase):
    requires_sequence_parallel_padding = True

    def __init__(
        self,
        model_config,
        parallelism_config,
        weights,
        max_generate_batch_size,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
        moe_config=None,
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
        self.embedding = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.media_token_id = model_config.mm_related_params.special_token_ids.get(
            "image_token_index"
        )
        self.layer = KimiK3MtpLayer(
            model_config,
            parallelism_config,
            weights.weights[0],
            resolve_kimi_k3_moe_strategy(moe_config),
        )
        self.final_norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), model_config.layernorm_eps
        )
        self.hidden_size = model_config.hidden_size
        self._max_batch = max_generate_batch_size
        self._proposal_steps = model_config.gen_num_per_cycle
        self._decode_role = False
        self._recurrent: Optional[torch.Tensor] = None
        self._recurrent_valid_tokens = 0
        self._all_gather_gemm_configured = False
        self._gemm_reduce_scatter_configured = False

    def initialize(self, init_resource):
        super().initialize(init_resource)
        self._decode_role = bool(init_resource.is_decode_role)
        if self._decode_role:
            capacity = max(
                self._max_batch,
                int(getattr(init_resource, "max_decode_graph_batch_size", 1)),
            )
            capacity *= max(1, self._proposal_steps + 1)
            self._recurrent = self.embedding.weight.new_empty(
                capacity, self.hidden_size
            )
        tp_size = int(self.parallelism_config.get_attn_tp_size())
        if self._decode_role:
            max_global_tokens = capacity
        else:
            max_global_tokens = collective_gemm_workspace_global_tokens(
                int(self.config.max_seq_len),
                int(init_resource.max_context_batch_size),
                prefill_chunk_tokens(),
            )
        max_local_tokens = (max_global_tokens + tp_size - 1) // tp_size
        max_physical_tokens = max_local_tokens * tp_size
        if tp_size > 1 and not self._all_gather_gemm_configured:
            configure_all_gather_gemm(
                get_process_group(Group.TP),
                self.embedding.weight.device,
                max_m=max_physical_tokens,
                k=self.hidden_size,
                dtype=self.embedding.weight.dtype,
            )
            self._all_gather_gemm_configured = True
        if tp_size > 1 and not self._gemm_reduce_scatter_configured:
            configure_gemm_reduce_scatter(
                get_process_group(Group.TP),
                self.embedding.weight.device,
                max_m=max_physical_tokens,
                n=self.hidden_size,
            )
            self._gemm_reduce_scatter_configured = True
        return True

    def _embed_shifted_tokens(self, inputs):
        # vLLM 0.28.0 K3MTP does not support external MM embeddings. RTP
        # uses feature hashes as cache-key tokens: restore the media token
        # before embedding, preserving the already shifted request layout.
        features = inputs.multimodal_inputs.multimodal_features
        if not features:
            return self.embedding(inputs.input_ids)
        locs = inputs.multimodal_inputs.mm_features_locs_host
        if locs is None or locs.numel() != len(features):
            raise ValueError(
                "K3 MTP multimodal features require matching host locations"
            )
        ranges = sequence_offsets(
            inputs.attention_inputs.cu_seqlens,
            inputs.input_ids.numel(),
            cu_seqlens_host=inputs.attention_inputs.cu_seqlens_host,
        )
        if self.media_token_id is None:
            raise ValueError("K3 MTP media tokens require media_placeholder_token_id")
        ids = inputs.input_ids.clone()
        for feature, loc in zip(features, locs.tolist()):
            start = max(s for s, _ in ranges if s <= loc + feature.size(0) - 1)
            dropped = max(0, start - loc + 1)
            offset = max(loc - 1, start)
            count = feature.size(0) - dropped
            if count > 0:
                ids.narrow(0, offset, count).fill_(self.media_token_id)
        return self.embedding(ids)

    def get_mtp_target_hidden_states(self, num_tokens):
        if self._recurrent is None:
            return None
        rows = self._recurrent_valid_tokens if num_tokens < 0 else int(num_tokens)
        if rows > self._recurrent.size(0):
            raise ValueError("K3 MTP recurrent rows exceed buffer capacity")
        return self._recurrent[:rows]

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        # Planner metadata is materialized before forward, including capture.
        select_block_map_for_layer(inputs.attention_inputs, 0)
        return super().prepare_fmha_impl(inputs, is_cuda_graph)

    def forward(self, inputs, fmha_impl=None):
        previous_h = inputs.input_hiddens
        if (
            previous_h is None
            or previous_h.ndim != 2
            or previous_h.shape[-1] != self.hidden_size
            or previous_h.shape[0] != inputs.input_ids.numel()
        ):
            raise ValueError(
                "K3 MTP requires one pre-norm hidden state of width H per token"
            )
        attention_inputs = inputs.attention_inputs
        tp_size = int(self.parallelism_config.get_attn_tp_size())
        tp_rank = int(self.parallelism_config.get_attn_tp_rank())
        sp_layout = sequence_parallel_layout_from_attention_inputs(
            attention_inputs,
            physical_tokens=int(inputs.input_ids.numel()),
            world_size=tp_size,
            rank=tp_rank,
        )
        positions = local_physical_token_view(mtp_positions(inputs), sp_layout)
        embedding = local_physical_token_view(
            self._embed_shifted_tokens(inputs), sp_layout
        )
        previous_h = local_physical_token_view(previous_h, sp_layout)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        select_block_map_for_layer(inputs.attention_inputs, 0)
        h = self.layer(
            embedding,
            previous_h,
            positions,
            fmha_impl,
            self.kv_cache.get_layer_cache(0) if self.kv_cache else None,
            attention_inputs,
            sp_layout=sp_layout,
        )
        h = all_gather_trim(h, sp_layout.tokens.logical_tokens, group=Group.TP)
        if self._decode_role:
            if self._recurrent is None or h.size(0) > self._recurrent.size(0):
                raise ValueError(
                    "K3 MTP recurrent Decode buffer is not initialized or too small"
                )
            self._recurrent[: h.size(0)].copy_(h)
        else:
            self._recurrent = h
        self._recurrent_valid_tokens = h.size(0)
        return PyModelOutputs(self.final_norm(h), fmha_impl.fmha_params)

"""Preview2 real-weight DSpARK runner with an explicit math contract.

The runner uses real projections, paged KV, and the shared commit/propose
protocol; there is no mock execution. The wrapper derives norm, mask, and
window semantics from the validated released checkpoint configuration.
"""

from dataclasses import dataclass

import torch
from torch import nn

from rtp_llm.models.minimax_m31_dspark import DSPARK_FINAL_NORM, DSPARK_HIDDEN_NORM
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.kernels.cuda.mxfp8_ops import mxfp8_quant_act_packed
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.dsv4.fp8._cp_slot_mapping import cp_kv_slot_mapping
from rtp_llm.models_py.modules.factory.attention.common import (
    apply_write_cache_store,
    create_write_cache_store_impl,
    reshape_paged_kv_cache,
)
from rtp_llm.models_py.modules.hybrid.minimax_m31_dspark_layer import (
    MiniMaxM31DSparkLayer,
)
from rtp_llm.models_py.speculative.attention_inputs import primary_attention_inputs
from rtp_llm.models_py.speculative.dspark_proposer_mixin import DSparkProposerMixin
from rtp_llm.models_py.speculative.minimax_m31_dspark_context import map_cp_context_rows
from rtp_llm.models_py.triton_kernels.dspark_swa import (
    DSparkGemmaRMSNorm,
    commit_paged_gqa_kv,
)
from rtp_llm.utils.model_weight import W


@dataclass(frozen=True)
class MiniMaxM31DSparkMath:
    # No implicit defaults: a checkpoint/model reference must settle these.
    hidden_norm_gemma: bool
    final_norm_gemma: bool
    causal_query: bool
    window_left: int

    def __post_init__(self):
        if (
            any(
                type(x) is not bool
                for x in (
                    self.hidden_norm_gemma,
                    self.final_norm_gemma,
                    self.causal_query,
                )
            )
            or type(self.window_left) is not int
            or self.window_left < 0
        ):
            raise ValueError(
                "DSpARK math requires explicit boolean norms/mask and a nonnegative window"
            )


class _DSparkAttentionContext:
    fmha_params = None

    def prepare_cuda_graph(self, _inputs):
        # All dynamic metadata is read from graph-owned input tensors.
        return None


class MiniMaxM31DSparkModel(DSparkProposerMixin, GptModelBase):
    def __init__(
        self,
        config,
        parallelism_config,
        weight,
        moe_config=None,
        max_generate_batch_size=0,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
        *,
        math_contract: MiniMaxM31DSparkMath,
    ):
        super().__init__(
            config,
            parallelism_config,
            weight,
            max_generate_batch_size,
            fmha_config,
            py_hw_kernel_config,
            device_resource_config,
        )
        self.math_contract = math_contract
        gamma = int(config.gen_num_per_cycle)
        self.init_dspark_proposer(
            width=gamma,
            query_width=gamma + int(not config.dspark_sample_from_anchor),
            noise_token_id=config.dspark_noise_token_id,
            aux_feature_dim=len(config.dspark_target_layer_ids) * config.hidden_size,
            hidden_dim=config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [
                MiniMaxM31DSparkLayer(
                    config, parallelism_config, layer, py_hw_kernel_config
                )
                for layer in weight.weights
            ]
        )
        self.embedding = weight.global_weights[W.embedding]
        self.feature_projection = weight.global_weights[W.dspark_fc_w]
        hidden_norm_cls = (
            DSparkGemmaRMSNorm if math_contract.hidden_norm_gemma else RMSNorm
        )
        final_norm_cls = (
            DSparkGemmaRMSNorm if math_contract.final_norm_gemma else RMSNorm
        )
        self.hidden_norm = hidden_norm_cls(
            weight.global_weights[DSPARK_HIDDEN_NORM],
            config.layernorm_eps,
        )
        self.final_norm = final_norm_cls(
            weight.global_weights[DSPARK_FINAL_NORM],
            config.layernorm_eps,
        )
        cp = parallelism_config.prefill_cp_config
        # Decode may carry the remote Prefill CP configuration while its own
        # TP size is one. That is not a CP-local feature layout.
        self.cp_enabled = cp.method.value != 0 and int(parallelism_config.tp_size) > 1
        self.cp_size = int(parallelism_config.tp_size) if self.cp_enabled else 1
        self.cp_rank = int(parallelism_config.tp_rank) if self.cp_enabled else 0
        self.cache_cp_size = self.cp_size if cp.kv_cache_sharded else 1
        self.cache_cp_rank = self.cp_rank if cp.kv_cache_sharded else 0
        self.page_size = int(config.attn_config.tokens_per_block)
        kernel_page = int(config.attn_config.kernel_tokens_per_block) or self.page_size
        if kernel_page != self.page_size:
            raise ValueError(
                "DSpARK GQA currently requires matching physical and kernel page sizes"
            )

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        return _DSparkAttentionContext()

    def combine_hidden_states(self, features):
        return self.hidden_norm(features @ self.feature_projection)

    def map_commit_rows(self, starts, lengths, committed_ends, row_count, inputs):
        if not self.cp_enabled:
            return super().map_commit_rows(
                starts, lengths, committed_ends, row_count, inputs
            )
        attn = primary_attention_inputs(inputs.attention_inputs)
        info = attn.context_parallel_info
        shuffle = info.prefill_shuffle_indices
        if shuffle.numel() != row_count:
            raise ValueError(
                "DSpARK CP-local hidden rows must match runtime shuffle rows"
            )
        device = starts.device
        prefixes = attn.prefix_lengths.to(device=device)
        requests, positions = map_cp_context_rows(
            info.prefill_cp_chunk_lengths.to(device=device),
            shuffle.to(device=device),
            prefixes,
            committed_ends - prefixes,
        )
        return requests, positions, info

    def _layer_cache(self, index, attn):
        if self.kv_cache is None:
            raise RuntimeError("DSpARK requires initialized paged KV cache")
        select_block_map_for_layer(attn, index)
        layer_cache = self.kv_cache.get_layer_cache(index)
        layer = self.layers[index]
        cache = reshape_paged_kv_cache(
            layer_cache.kv_cache_base,
            layer.kv_heads,
            self.page_size,
            layer.dim,
        )
        if cache.dtype != torch.bfloat16:
            raise ValueError(
                "preview2 GQA draft cache must be BF16, independent of target KV4"
            )
        return layer_cache, cache, attn.kv_cache_kernel_block_id_device

    def _commit_slots(self, requests, positions, table):
        valid = (requests >= 0) & (positions >= 0)
        safe_req = requests.clamp(min=0).to(torch.int64)
        safe_pos = positions.clamp(min=0).to(torch.int64)
        if self.cache_cp_size > 1:
            slots = cp_kv_slot_mapping(
                safe_pos,
                table,
                safe_req,
                self.page_size,
                self.page_size,
                1,
                self.cache_cp_size,
                self.cache_cp_rank,
                owner_tokens_per_block=self.page_size,
            )
        else:
            blocks = safe_pos // self.page_size
            valid &= (safe_req < table.shape[0]) & (blocks < table.shape[1])
            if table.numel() == 0:
                return torch.full_like(safe_pos, -1), torch.zeros_like(valid)
            physical = table[
                safe_req.clamp(max=table.shape[0] - 1),
                blocks.clamp(max=table.shape[1] - 1),
            ]
            slots = (
                physical.to(torch.int64) * self.page_size + safe_pos % self.page_size
            )
            valid &= physical >= 0
        return torch.where(valid, slots, -1), valid

    def commit_feature_rows(
        self,
        main_x,
        context_req_ids,
        context_positions,
        committed_ends,
        inputs,
        commit_ctx=None,
    ):
        attn = primary_attention_inputs(inputs.attention_inputs)
        # Never gather the wide target feature tensor. Gather small projected
        # per-layer K/V plus row identity, then write only owned page-RR slots.
        metadata = torch.stack((context_req_ids, context_positions), dim=1)
        if self.cp_size > 1:
            metadata = all_gather(metadata, group=Group.TP)
        requests, positions = metadata.unbind(1)
        quantized, scales = mxfp8_quant_act_packed(main_x)
        writer = create_write_cache_store_impl(attn, self.kv_cache)
        rope_positions = context_positions.clamp(min=0)
        # Tables/row identities are read-only within this commit. Reuse only
        # matching views, not merely matching shapes or layer group labels.
        # This is forward-local: graph replay still executes the mapping with
        # fresh table contents, and no mapping survives into the next round.
        slot_maps = {}
        for index, layer in enumerate(self.layers):
            k, v = layer.project_context_kv(
                quantized,
                rope_positions,
                input_scales=scales,
            )
            if self.cp_size > 1:
                packed = torch.stack((k, v), dim=1)
                # stack owns independent storage; release the projection views
                # before allocating the larger all-gather destination.
                del k, v
                packed = all_gather(packed, group=Group.TP)
                k, v = packed.unbind(1)
            layer_cache, cache, table = self._layer_cache(index, attn)
            table_key = (
                table.device,
                table.data_ptr(),
                tuple(table.shape),
                tuple(table.stride()),
                table.dtype,
            )
            if table_key not in slot_maps:
                slot_maps[table_key] = self._commit_slots(requests, positions, table)
            slots, valid = slot_maps[table_key]
            commit_paged_gqa_kv(
                k,
                v,
                cache,
                slots,
                valid,
            )
            apply_write_cache_store(writer, attn, layer_cache)
            del k, v
            if self.cp_size > 1:
                # Do not keep this layer's gathered payload alive while the
                # next layer projects its local K/V. All uses are stream-ordered.
                del packed

    def forward_query_block(
        self,
        query_ids,
        query_positions,
        prefix_lengths,
        active_requests,
        inputs,
        fmha_impl,
    ):
        if int(self.parallelism_config.tp_size) > 1:
            raise RuntimeError("DSpARK proposal currently requires PD decode TP1")
        attn = primary_attention_inputs(inputs.attention_inputs)
        hidden = torch.nn.functional.embedding(query_ids.long(), self.embedding)
        query_lens = active_requests.to(torch.int32) * self._dspark_query_width
        for index, layer in enumerate(self.layers):
            _, cache, table = self._layer_cache(index, attn)
            hidden = layer(
                hidden,
                query_positions,
                cache,
                table,
                prefix_lengths,
                query_lens,
                causal=self.math_contract.causal_query,
                window_left=self.math_contract.window_left,
            )
        return hidden.reshape(-1, self._dspark_hidden_dim)

    def compute_draft_hidden_states(self, hidden):
        return self.final_norm(hidden).contiguous()

    def forward_commit(self, inputs, fmha_impl=None):
        return self.run_commit_step(inputs, self.feature_projection.device)

    def forward_propose(self, inputs, fmha_impl=None):
        return self.run_propose_step(inputs, fmha_impl, self.feature_projection.device)

    def forward(self, inputs, fmha_impl=None):
        raise RuntimeError(
            "DSpARK must explicitly select forward_commit or forward_propose"
        )

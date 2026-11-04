"""PageRR execution policy over the unified absorbed sparse MLA implementation."""

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from rtp_llm.models_py.utils.module_profiler import profile_scope

from .dsa_page_rr_params import DsaPageRRParams
from .flashmla_sparse_impl import SparseMlaImpl, SparseMlaOp, flash_mla_sparse_fwd
from rtp_llm.models_py.distributed.collective_torch import get_mla_dcp


class QueryCpLayout:
    """Complete heads and already-local queries require no layout collective."""

    def __init__(self, heads):
        self.heads = heads

    def scatter_queries(self, query):
        if query.shape[1] != self.heads:
            raise ValueError("query CP must contain every attention head")
        return query

    def restore_queries(self, output):
        return output


class PageRRSparseMlaOp(SparseMlaOp):
    def __init__(self, *args, parallelism_config, workspace_owner=None, **kwargs):
        super().__init__(*args, parallelism_config=parallelism_config, **kwargs)
        self.workspace_owner = workspace_owner
        geometry = SimpleNamespace(
            head_num=self.num_heads,
            kv_lora_rank=self.kv_lora_rank,
            rope_head_dim=self.qk_rope_head_dim,
        )
        self.communicator = (
            QueryCpLayout(self.num_heads)
            if workspace_owner is not None and workspace_owner.prefill_query_cp
            else get_mla_dcp(geometry, torch.device("cuda", torch.cuda.current_device()), torch.bfloat16)
        )
        # The pinned FlashMLA BF16 sparse kernel accepts 64/128 heads.
        self.kernel_heads = 64 if self.communicator.heads <= 64 else 128
        if self.communicator.heads > self.kernel_heads:
            raise ValueError(
                "BF16 sparse FlashMLA currently supports at most 128 heads"
            )
        # Retain every capture bucket's scheduler tensors across preparations.
        self._fp8_schedules = {}

    def plan(self, params, block_table, attn_inputs=None):
        self.params = params
        self.execute = self._prefill if params.is_prefill else self._decode
        if self.workspace_owner.sfa_use_fp8 and not params.is_prefill:
            from rtp_llm.models_py.utils.flash_mla_legacy import get_mla_metadata

            rows = params.positions_d.numel()
            key = (rows, self.kernel_heads, params.kernel_page_size, self.top_k)
            if key not in self._fp8_schedules:
                if params.forbid_realloc:
                    raise ValueError("FP8 SFA scheduler was not prepared before graph replay")
                self._fp8_schedules[key] = get_mla_metadata(
                    cache_seqlens=None,
                    num_q_tokens_per_head_k=rows * self.kernel_heads,
                    num_heads_q=self.kernel_heads,
                    num_heads_k=1,
                    is_fp8_kvcache=True,
                    topk=self.top_k,
                )
            self._fp8_metadata, self._fp8_num_splits = self._fp8_schedules[key]

    projects_values = True

    def _attention(self, query, kv, slots):
        # Query CP/head gather and padding happen once, outside the KV loop.
        if not query.shape[0]:
            return query.new_empty(
                (0, self.kernel_heads, self.kv_lora_rank)
            ), query.new_empty((0, self.kernel_heads), dtype=torch.float32)
        if self.workspace_owner.sfa_use_fp8 and not self.params.is_prefill:
            from rtp_llm.models_py.utils.flash_mla_legacy import flash_mla_with_kvcache

            output, lse = flash_mla_with_kvcache(
                q=query.unsqueeze(1),
                k_cache=kv.view(torch.uint8).view(-1, self.params.kernel_page_size, 1, 656),
                block_table=self.params.block_table,
                cache_seqlens=None,
                head_dim_v=self.kv_lora_rank,
                tile_scheduler_metadata=self._fp8_metadata,
                num_splits=self._fp8_num_splits,
                is_fp8_kvcache=True,
                indices=slots.unsqueeze(1),
                softmax_scale=self.scale,
            )
            # The public decode result is natural-log LSE [rows, heads, 1].
            # Keep the existing BF16 V-up and projected-state DCP merge.
            return output.squeeze(1), lse.squeeze(-1)
        output, _, lse = flash_mla_sparse_fwd(
            query, kv, slots.unsqueeze(1), self.scale, d_v=self.kv_lora_rank
        )
        return output, lse

    def _prefill(self, query):
        return self.communicator.scatter_queries(query)

    def _decode(self, query):
        if query.shape[1] == self.communicator.heads:
            return query.contiguous()
        return self.communicator.query_gather(
            query.transpose(0, 1).contiguous()
        ).transpose(0, 1)

    def _prefill_kv_chunks(self, query, kv, topk_indices, layer_id, workspace):
        from rtp_llm.models_py.kernels.cuda.dsa_kv_chunk import (
            check_sparse_lse,
            map_dsa_chunk_slots,
            zero_sparse_empty_rows,
        )

        from rtp_llm.models_py.triton_kernels.mla.lse_merge import merge_attention_states_in_place as merge_mla_states_in_place

        params, execution = self.params, self.params.sfa_execution
        rows, heads = query.shape[0], self.communicator.heads
        latent_dim, value_dim = self.kv_lora_rank, self.workspace_owner.value_dim
        chunked = execution.plan.chunked
        slots = workspace("slots", (rows, self.top_k), torch.int32)
        if chunked:
            running = workspace("running", (rows, heads, latent_dim), torch.float32)
            running_lse = workspace("running_lse", (rows, heads), torch.float32)
            running.zero_()
            running_lse.fill_(-float("inf"))
        raw_cache = kv.view(-1, params.kernel_page_size, kv.shape[-1])
        if self.workspace_owner.sfa_use_fp8:
            raw_cache = raw_cache.view(torch.uint8)
        for block in execution.blocks:
            history = params.cache_adapter.read_prefix_chunk(
                raw_cache, params.block_table, block.descriptor, workspace=workspace,
                allow_reserved_block_zero=params.is_warmup,
            )
            if self.workspace_owner.sfa_use_fp8:
                from rtp_llm.ops.compute_ops import rtp_llm_ops

                dense = workspace(
                    "cache.dequant",
                    (history.shape[0], self.kv_lora_rank + self.qk_rope_head_dim),
                    torch.bfloat16,
                )
                table, lengths, starts = block.fp8_dequant_metadata
                # Reuse the existing 656-byte codec on a canonical chunk;
                # only this bounded BF16 chunk is materialized after transport.
                with profile_scope("RTP::attention.dsa.sfa.kv_dequant"):
                    rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache(
                        history.view(1, history.shape[0], 656),
                        dense[:, :self.kv_lora_rank], dense[:, self.kv_lora_rank:],
                        table, lengths, starts, 1,
                    )
                history = dense
            with profile_scope("RTP::attention.dsa.sfa.chunk_slot_map"):
                map_dsa_chunk_slots(topk_indices, params, block, slots)
            with profile_scope("RTP::attention.dsa.sfa.chunk_qk_pv"):
                partial, lse = self._attention(query, history.unsqueeze(1), slots)
            if chunked:
                with profile_scope("RTP::attention.dsa.sfa.chunk_latent_lse_merge"):
                    merge_mla_states_in_place(
                        running, running_lse, partial[:, :heads], lse[:, :heads],
                        raw_partial_lse=True,
                    )
                del partial, lse
            else:
                zero_sparse_empty_rows(partial, lse, heads)
                # Keep the kernel's real-head view: no FP32 state or merge for
                # a single block. Only empty rows were written above.
                latent = partial[:, :heads]
            del history
        with profile_scope("RTP::attention.dsa.sfa.lse_check"):
            check_sparse_lse(running_lse if chunked else lse[:, :heads])
        if chunked:
            # Reuses the now-dead KV arena; only one cast and V-up per layer.
            latent = workspace("latent", (rows, heads, latent_dim), torch.bfloat16)
            with profile_scope("RTP::attention.dsa.sfa.final_latent_cast"):
                latent.copy_(running)
        output = workspace("output", (rows, heads, value_dim), torch.bfloat16)
        with profile_scope("RTP::attention.dsa.sfa.final_v_up"):
            torch.bmm(
                latent.transpose(0, 1),
                self.workspace_owner.value_weights[layer_id],
                out=output.transpose(0, 1),
            )
        from rtp_llm.models_py.utils.smoke_evidence import completed_dsa_chunks
        completed_dsa_chunks("sfa", params, len(execution.blocks), rows)
        output = self.communicator.restore_queries(output)
        execution.finish()
        return output

    def forward(self, q, kv, topk_indices, kv_scale=None, layer_id=0):
        from rtp_llm.models_py.kernels.cuda.dsa_kv_chunk import (
            map_dsa_chunk_slots,
            normalize_sparse_partial,
        )

        from rtp_llm.models_py.triton_kernels.mla.lse_merge import merge_attention_states_in_place as merge_mla_states_in_place

        params = self.params
        execution = params.sfa_execution
        workspace = execution.begin()
        heads = self.communicator.heads
        if params.is_prefill and params.query_cp and q.shape[1] == self.kernel_heads:
            # The Prefill producer has already built the kernel's final layout.
            query = q
        else:
            query = self.execute(q)
            query = F.pad(query, (0, 0, 0, self.kernel_heads - heads)).contiguous()
        rows, value_dim = query.shape[0], self.workspace_owner.value_dim
        if not execution.blocks:
            output = query.new_zeros((rows, heads, value_dim))
            execution.finish()
            return (
                self.communicator.restore_queries(output)
                if params.is_prefill
                else output[
                    :,
                    params.shard_rank
                    * self.num_heads : (params.shard_rank + 1)
                    * self.num_heads,
                ]
            )
        if params.is_prefill:
            return self._prefill_kv_chunks(query, kv, topk_indices, layer_id, workspace)
        running = workspace("running", (rows, heads, value_dim), torch.float32)
        running_lse = workspace("running_lse", (rows, heads), torch.float32)
        projected = workspace("projected", (rows, heads, value_dim), torch.bfloat16)
        partial_lse = workspace("partial_lse", (rows, heads), torch.float32)
        slots = workspace("slots", (rows, self.top_k), torch.int32)
        running.zero_()
        running_lse.fill_(-float("inf"))
        weights = self.workspace_owner.value_weights[layer_id]
        for block in execution.blocks:
            # Decode retains owner-local paged reads and projected-state DCP
            # combine. The Prefill latent merge policy is selected above.
            history = kv.view(-1, kv.shape[-1])
            with profile_scope("RTP::attention.dsa.sfa.chunk_slot_map"):
                map_dsa_chunk_slots(
                    topk_indices, params, block, slots, block_table=execution.block_ids
                )
            with profile_scope("RTP::attention.dsa.sfa.chunk_qk_pv"):
                partial, lse = self._attention(query, history.unsqueeze(1), slots)
                normalize_sparse_partial(partial, lse, heads, partial_lse)
            # Linear V-up is per head and preserves the original attention LSE.
            # Keep only the 128-dimensional FP32 running state between blocks.
            with profile_scope("RTP::attention.dsa.sfa.chunk_v_up"):
                torch.bmm(
                    partial[:, :heads].transpose(0, 1),
                    weights,
                    out=projected.transpose(0, 1),
                )
            with profile_scope("RTP::attention.dsa.sfa.chunk_lse_merge"):
                merge_mla_states_in_place(running, running_lse, projected, partial_lse)
            del partial, lse, history
        output = workspace("output", (rows, heads, value_dim), torch.bfloat16)
        output.copy_(running)
        lengths = (topk_indices >= 0).sum(-1).to(torch.int32).contiguous()
        # Only the DCP boundary converts to the established base-2 contract.
        lse_base2 = (running_lse / math.log(2)).contiguous()
        output = self.communicator.combine(
            output, lse_base2, lengths, output_dim=value_dim
        ).transpose(0, 1)
        execution.finish()
        return output


class SparseMlaPageRRImpl(SparseMlaImpl):
    def __init__(
        self,
        config,
        parallelism,
        weights,
        inputs,
        fmha_config,
        is_cuda_graph,
        *,
        workspace_owner=None
    ):
        if workspace_owner is None:
            from .dsa_kv_workspace import DsaWorkspaceOwner

            workspace_owner = DsaWorkspaceOwner.from_model(config, parallelism)
        self.workspace_owner = workspace_owner
        attention = config.getAttentionConfigs(parallelism.get_attn_tp_size())
        if attention.mla_fp8_compute:
            raise ValueError(
                "DSA SFA uses BF16 computation independently of cache dtype"
            )
        self.page_rr_params = DsaPageRRParams(
            attention,
            parallelism,
            max_seq_len=int(config.max_seq_len),
            workspace_owner=workspace_owner,
            is_cuda_graph=is_cuda_graph,
        )
        super().__init__(
            attention,
            inputs,
            weights.weights,
            None,
            fmha_config,
            max_seq_len=config.max_seq_len,
            is_cuda_graph=is_cuda_graph,
            parallelism_config=parallelism,
            fmha_impl=PageRRSparseMlaOp,
            apply_rope=False,
            fmha_impl_kwargs={"workspace_owner": workspace_owner},
        )
        # SM100 sparse Decode loads FP32 scales through E8M0 conversion.
        # Prefill and Decode must therefore write the same power-of-two scales.
        self.kv_cache_write_op.use_ue8m0 = workspace_owner.sfa_use_fp8

    def create_params(self, inputs):
        self.fmha_params = self.page_rr_params
        self.rope_params = self.page_rr_params
        self.cp_params = self.page_rr_params
        self.prepare(inputs)

    def _write_kv_cache(self, compressed_kv, k_pe, kv_cache):
        params = self.page_rr_params
        if not (params.is_prefill and params.query_cp):
            return super()._write_kv_cache(compressed_kv, k_pe, kv_cache)
        # Global positions are carried by the shared CP plan. Address mapping
        # is applied only after the projected values reach every cache owner.
        width = self.kv_lora_rank + self.rope_head_dim
        if (compressed_kv.stride() == (width, 1)
                and k_pe.stride() == (width, 1)
                and k_pe.untyped_storage().data_ptr() == compressed_kv.untyped_storage().data_ptr()
                and k_pe.storage_offset() == compressed_kv.storage_offset() + self.kv_lora_rank):
            values = compressed_kv.as_strided((compressed_kv.shape[0], width), (width, 1))
        else:
            values = torch.cat((compressed_kv, k_pe), dim=-1)

        def write(value, slots):
            self.kv_cache_write_op.forward(
                value[:, :self.kv_lora_rank], value[:, self.kv_lora_rank:],
                kv_cache, SimpleNamespace(slot_mapping=slots),
            )

        params.ingress_plans["sfa"].run(values, write)

    def _apply_input_bmm(self, q, layer_id):
        params = self.page_rr_params
        query_cp = params.is_prefill and params.query_cp
        if not query_cp and q.shape[1] == self.num_heads:
            return super()._apply_input_bmm(q, layer_id)
        from rtp_llm.models_py.kernels.cuda.dsa_buffer_ops import absorbed_query
        from rtp_llm.utils.model_weight import W

        if q.shape[1] != self.workspace_owner.heads:
            raise ValueError("Sparse replicated Q must contain every attention head")
        return absorbed_query(q, self.weights[layer_id][W.mla_kc],
                              self.kv_lora_rank,
                              self.fmha_impl.kernel_heads if query_cp else q.shape[1])

    def prepare(self, inputs, forbid_realloc=False):
        self.page_rr_params.prepare(inputs, forbid_realloc=forbid_realloc)
        # Reserve the small Decode SFA arena before freezing the Graph
        # Indexer's history capacity. Actual execution still runs Indexer first.
        phases = ("sfa", "indexer") if (
            self.page_rr_params.is_cuda_graph and not self.page_rr_params.is_prefill
        ) else ("indexer", "sfa")
        for phase in phases:
            self.page_rr_params.prepare_execution(phase)
        self.attn_inputs = inputs
        self.fmha_impl.plan(
            self.page_rr_params, self.page_rr_params.block_table, inputs
        )

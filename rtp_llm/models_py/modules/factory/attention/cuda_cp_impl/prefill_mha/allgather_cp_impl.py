import logging
from dataclasses import dataclass
from typing import Optional

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    ParamsBase,
    PyAttentionInputs,
    fill_mla_params,
)

logger = logging.getLogger(__name__)

from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from flashinfer.cascade import merge_state
from flashinfer.page import append_paged_kv_cache

from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
    generate_full_causal_kv_indices,
    generate_q_indices,
    plan_prefix_paged_attention,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    get_py_flashinfer_workspace_buffer,
)


@dataclass(frozen=True)
class _ShardedCachePlan:
    append_rows: torch.Tensor
    append_page_ids: torch.Tensor
    append_offsets: torch.Tensor
    local_prefix_page_ids: torch.Tensor


class PCPAllGatherAttnOp:
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
        backend: str = "auto",  # "auto", "fa2", or "fa3"
        causal: bool = True,
        kv_layout: str = "NHD",  # "NHD" or "HND"
    ):
        """
        Args:
            config: Model configuration
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (for GQA/MQA)
            head_dim: Dimension of each head
            backend: FlashInfer backend ("auto", "fa2", or "fa3")
            causal: Whether to use causal masking
            kv_layout: KV cache layout ("NHD" or "HND")
        """
        super().__init__()
        self.attn_inputs = attn_inputs
        self.attn_configs = attn_configs
        self.num_qo_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.backend = backend
        self.kv_layout = kv_layout

        assert causal == True
        self.device = torch.cuda.current_device()
        self.workspace_buffer = get_py_flashinfer_workspace_buffer()

        self.cp_info = attn_inputs.context_parallel_info

        self.prefill_cp_rank = parallelism_config.tp_rank
        self.prefill_cp_size = parallelism_config.tp_size

        self.kv_cache_sharded = parallelism_config.prefill_cp_config.kv_cache_sharded
        self.q_data_type = torch.bfloat16
        if self.kv_cache_sharded:
            if attn_configs.kv_cache_dtype != KvCacheDataType.BASE:
                raise ValueError("Sharded ALL_GATHER requires an unquantized KV cache")
            self.q_data_type = attn_configs.dtype
            if self.q_data_type not in (torch.float16, torch.bfloat16):
                raise ValueError("Sharded ALL_GATHER requires FP16 or BF16 attention")
        self.physical_tokens_per_block = attn_configs.tokens_per_block
        self.seq_size_per_block = (
            attn_configs.kernel_tokens_per_block
            if self.kv_cache_sharded
            else self.physical_tokens_per_block
        )
        if self.kv_cache_sharded and (
            self.seq_size_per_block <= 0
            or self.physical_tokens_per_block <= 0
            or self.physical_tokens_per_block % self.seq_size_per_block
        ):
            raise ValueError("CP cache blocks must contain whole kernel pages")

        self.q0_idx = self.q1_idx = None
        self.kv0_idx = self.kv1_idx = None
        self.kv_restore_unpad_indices = None

        self.prefill_wrappers = {
            "ragged": {
                name: BatchPrefillWithRaggedKVCacheWrapper(
                    self.workspace_buffer,
                    kv_layout=kv_layout,
                    backend=backend,
                )
                for name in ["part0", "part1"]
            },
            "paged": {
                "prefix": BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    kv_layout="HND",
                    backend=backend,
                ),
            },
        }

    def support(self, attention_inputs: PyAttentionInputs) -> bool:
        return attention_inputs.is_prefill

    def prepare(
        self, attention_inputs: PyAttentionInputs
    ) -> ParamsBase | _ShardedCachePlan:
        if self.kv_cache_sharded and attention_inputs.input_lengths.numel() == 0:
            raise ValueError("Sharded ALL_GATHER requires a non-empty prefill batch")
        cu_seqlens = attention_inputs.cu_seqlens_device[
            : attention_inputs.input_lengths.size(0) + 1
        ]
        padding_mask = self.cp_info.prefill_qkv_padding_mask
        kv_restore_indices = self.cp_info.prefill_qkv_restore_indice
        self.kv_restore_unpad_indices = kv_restore_indices[padding_mask == 1]

        qo_indptr = cu_seqlens // 2

        q0_idx, q1_idx = generate_q_indices(self.cp_info.prefill_cp_chunk_lengths)
        kv0_idx, kv1_idx = generate_full_causal_kv_indices(
            self.cp_info.prefill_cp_chunk_lengths,
            self.prefill_cp_rank,
            self.prefill_cp_size,
        )

        self.kv0_idx = kv_restore_indices[kv0_idx]
        self.kv1_idx = kv_restore_indices[kv1_idx]
        self.q0_idx = torch.tensor(q0_idx, device=self.device)
        self.q1_idx = torch.tensor(q1_idx, device=self.device)

        self._plan_ragged(qo_indptr)
        self.has_prefix = bool(attention_inputs.prefix_lengths.any().item())
        if self.kv_cache_sharded:
            return self._prepare_sharded_cache(attention_inputs, cu_seqlens)

        params = fill_mla_params(
            self.attn_inputs.prefix_lengths,
            self.attn_inputs.sequence_lengths,
            self.cp_info.prefill_actual_input_lengths_cpu,
            self.attn_inputs.kv_cache_kernel_block_id,
            self.attn_configs.kernel_tokens_per_block,
        )

        if self.has_prefix:
            plan_prefix_paged_attention(
                self.prefill_wrappers["paged"]["prefix"],
                cu_seqlens,
                attention_inputs.prefix_lengths,
                params,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=self.seq_size_per_block,
                device=self.device,
            )
        return params

    def _prepare_sharded_cache(
        self, inputs: PyAttentionInputs, qo_indptr: torch.Tensor
    ) -> _ShardedCachePlan:
        block_size = self.physical_tokens_per_block
        page_size = self.seq_size_per_block
        cp_size = self.prefill_cp_size
        cp_rank = self.prefill_cp_rank
        pages_per_block = block_size // page_size
        cp_block_size = block_size * cp_size

        lengths = self.cp_info.prefill_actual_input_lengths_cpu.tolist()
        prefixes = inputs.prefix_lengths.tolist() or [0] * len(lengths)
        block_table = inputs.kv_cache_kernel_block_id.cpu()
        append_rows, append_page_ids, append_offsets = [], [], []
        local_prefix_page_ids, prefix_page_owners, prefix_page_offsets = [], [], []
        prefix_indptr = [0]
        row_start = 0
        local_prefix_count = 0

        for batch, (prefix, length) in enumerate(zip(prefixes, lengths)):
            # CP ownership follows physical blocks. FlashInfer addresses kernel pages.
            positions = torch.arange(prefix, prefix + length)
            logical_blocks = positions // block_size
            owned_positions = (logical_blocks % cp_size == cp_rank).nonzero().flatten()
            local_positions = (
                (logical_blocks[owned_positions] // cp_size) * block_size
                + positions[owned_positions] % block_size
            )
            append_rows.append(row_start + owned_positions)
            append_page_ids.append(block_table[batch, local_positions // page_size])
            append_offsets.append(local_positions % page_size)
            row_start += length

            # Gather equally sized batches from each rank, then index the rank-major
            # buffer in logical prefix order. Padding pages never enter the plan.
            prefix_pages = (prefix + page_size - 1) // page_size
            local_pages = (
                (prefix + cp_block_size - 1) // cp_block_size * pages_per_block
            )
            local_prefix_page_ids.append(block_table[batch, :local_pages])
            prefix_indptr.append(prefix_indptr[-1] + prefix_pages)
            logical_pages = torch.arange(prefix_pages)
            prefix_blocks = logical_pages // pages_per_block
            prefix_page_owners.append(prefix_blocks % cp_size)
            prefix_page_offsets.append(
                local_prefix_count
                + (prefix_blocks // cp_size) * pages_per_block
                + logical_pages % pages_per_block
            )
            local_prefix_count += local_pages

        if self.has_prefix:
            prefix_indices = (
                torch.cat(prefix_page_owners) * local_prefix_count
                + torch.cat(prefix_page_offsets)
            )
            last_page_len = [(p - 1) % page_size + 1 if p else 0 for p in prefixes]
            self.prefill_wrappers["paged"]["prefix"].plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=torch.tensor(prefix_indptr, dtype=torch.int32),
                paged_kv_indices=prefix_indices.to(self.device, dtype=torch.int32),
                paged_kv_last_page_len=torch.tensor(last_page_len, dtype=torch.int32),
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim,
                page_size=page_size,
                causal=False,
                q_data_type=self.q_data_type,
            )

        return _ShardedCachePlan(
            append_rows=torch.cat(append_rows).to(self.device),
            append_page_ids=torch.cat(append_page_ids).to(
                self.device, dtype=torch.long
            ),
            append_offsets=torch.cat(append_offsets).to(self.device),
            local_prefix_page_ids=torch.cat(local_prefix_page_ids).to(
                self.device, dtype=torch.long
            ),
        )

    def _plan_ragged(self, qo_indptr: torch.Tensor) -> None:
        kv_indptr_part0 = qo_indptr * (self.prefill_cp_rank + 1)
        kv_indptr_part1 = qo_indptr * (2 * self.prefill_cp_size - self.prefill_cp_rank)
        common_params = {
            "num_qo_heads": self.num_qo_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim_qk": self.head_dim,
            "causal": True,
            "q_data_type": self.q_data_type,
        }
        self.prefill_wrappers["ragged"]["part0"].plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr_part0,
            **common_params,
        )
        self.prefill_wrappers["ragged"]["part1"].plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr_part1,
            **common_params,
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        params: Optional[ParamsBase | _ShardedCachePlan] = None,
    ) -> torch.Tensor:
        qkv = qkv.reshape(qkv.shape[0], -1)
        q, k, v = torch.split(
            qkv,
            [
                self.head_dim * self.num_qo_heads,
                self.head_dim * self.num_kv_heads,
                self.head_dim * self.num_kv_heads,
            ],
            dim=-1,
        )
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        all_keys = all_gather(k, group=Group.TP).reshape(
            k.shape[0] * self.prefill_cp_size, self.num_kv_heads, self.head_dim
        )
        all_values = all_gather(v, group=Group.TP).reshape(
            v.shape[0] * self.prefill_cp_size, self.num_kv_heads, self.head_dim
        )
        q_reshaped = q.reshape(-1, self.num_qo_heads, self.head_dim)

        # TODO: make write local kvcache async
        restore_k = all_keys[self.kv_restore_unpad_indices]
        restore_v = all_values[self.kv_restore_unpad_indices]
        kv_cache_tensor = kv_cache.kv_cache_base.view(
            -1, 2, self.num_kv_heads, self.seq_size_per_block, self.head_dim
        )
        if isinstance(params, _ShardedCachePlan):
            if params.append_rows.numel():
                kv_cache_tensor[params.append_page_ids, 0, :, params.append_offsets] = (
                    restore_k.index_select(0, params.append_rows)
                )
                kv_cache_tensor[params.append_page_ids, 1, :, params.append_offsets] = (
                    restore_v.index_select(0, params.append_rows)
                )
            if self.has_prefix:
                kv_cache_tensor = all_gather(
                    kv_cache_tensor.index_select(0, params.local_prefix_page_ids),
                    group=Group.TP,
                )
        else:
            append_paged_kv_cache(
                append_key=restore_k,
                append_value=restore_v,
                batch_indices=params.batch_indice_d,
                positions=params.positions_d,
                paged_kv_cache=kv_cache_tensor,
                kv_indices=params.page_indice_d,
                kv_indptr=params.decode_page_indptr_d,
                kv_last_page_len=params.paged_kv_last_page_len_d,
                kv_layout="HND",
            )

        q0 = torch.index_select(q_reshaped, 0, self.q0_idx).contiguous()
        q1 = torch.index_select(q_reshaped, 0, self.q1_idx).contiguous()

        k0 = torch.index_select(all_keys, 0, self.kv0_idx).contiguous()
        k1 = torch.index_select(all_keys, 0, self.kv1_idx).contiguous()
        v0 = torch.index_select(all_values, 0, self.kv0_idx).contiguous()
        v1 = torch.index_select(all_values, 0, self.kv1_idx).contiguous()
        if self.has_prefix:
            prefix_out, prefix_lse = self.prefill_wrappers["paged"]["prefix"].run(
                q_reshaped, kv_cache_tensor, return_lse=True
            )

            out0, lse0 = self.prefill_wrappers["ragged"]["part0"].run(
                q0, k0, v0, return_lse=True
            )
            out1, lse1 = self.prefill_wrappers["ragged"]["part1"].run(
                q1, k1, v1, return_lse=True
            )
            out0, _ = merge_state(
                v_a=prefix_out[self.q0_idx],
                s_a=prefix_lse[self.q0_idx],
                v_b=out0,
                s_b=lse0,
            )
            out1, _ = merge_state(
                v_a=prefix_out[self.q1_idx],
                s_a=prefix_lse[self.q1_idx],
                v_b=out1,
                s_b=lse1,
            )
            output = torch.empty_like(q_reshaped)
            output[self.q0_idx] = out0
            output[self.q1_idx] = out1
            return output
        else:
            output = torch.empty_like(q_reshaped)
            output[self.q0_idx] = self.prefill_wrappers["ragged"]["part0"].run(
                q0, k0, v0
            )
            output[self.q1_idx] = self.prefill_wrappers["ragged"]["part1"].run(
                q1, k1, v1
            )
            return output

"""Fused all-gather context-parallel prefill attention.

The operator keeps RTP-LLM's zigzag CP layout and KV-cache side effect, but
reduces the per-layer data path to one packed K/V all-gather, one paged-cache
append, and one paged attention call.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper
from flashinfer.page import append_paged_kv_cache

from rtp_llm.models_py.distributed.collective_torch import Group, _get_group
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    get_py_flashinfer_workspace_buffer,
    release_py_flashinfer_workspace_buffer,
)
from rtp_llm.ops import AttentionConfigs, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    ParamsBase,
    PyAttentionInputs,
    fill_mla_params,
    get_scalar_type,
)


class _CPGeometry:
    """Prepare-time mapping from gathered zigzag slots to paged requests."""

    def __init__(
        self,
        attn_inputs: PyAttentionInputs,
        cp_size: int,
        cp_rank: int,
        page_size: int,
        device: torch.device,
    ):
        cp_info = attn_inputs.context_parallel_info
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.page_size = page_size
        self.device = device

        self.chunk_lengths: List[int] = cp_info.prefill_cp_chunk_lengths.tolist()
        self.prefix_lengths: List[int] = attn_inputs.prefix_lengths.tolist()
        self.batch_size = len(self.chunk_lengths)
        self.half_lengths = [chunk_length // 2 for chunk_length in self.chunk_lengths]
        self.tokens_local = sum(self.chunk_lengths)
        self.gathered_slots = self.cp_size * self.tokens_local
        self.padded_lengths = [
            chunk_length * self.cp_size for chunk_length in self.chunk_lengths
        ]

        assert all(
            chunk_length % 2 == 0 for chunk_length in self.chunk_lengths
        ), "fused CP prefill requires an even local chunk length"
        assert self.batch_size > 0, "fused CP prefill requires a non-empty batch"

        # restore[sequence_position] gives its rank-major all-gather slot. Scatter
        # the cache metadata through that permutation so append performs the
        # zigzag restore without materializing sequence-ordered K and V tensors.
        restore = cp_info.prefill_qkv_restore_indice.to(
            device=device, dtype=torch.int64
        )
        assert restore.numel() == sum(self.padded_lengths), (
            f"restore indices ({restore.numel()}) must cover the padded sequence "
            f"({sum(self.padded_lengths)})"
        )

        batch_parts = [
            torch.full((length,), batch, dtype=torch.int32, device=device)
            for batch, length in enumerate(self.padded_lengths)
        ]
        position_parts = [
            torch.arange(length, dtype=torch.int32, device=device).add(
                self.prefix_lengths[batch]
            )
            for batch, length in enumerate(self.padded_lengths)
        ]
        batch_by_position = (
            batch_parts[0] if self.batch_size == 1 else torch.cat(batch_parts)
        )
        position_by_position = (
            position_parts[0] if self.batch_size == 1 else torch.cat(position_parts)
        )

        self.gathered_batch = torch.empty(
            self.gathered_slots, dtype=torch.int32, device=device
        )
        self.gathered_position = torch.empty(
            self.gathered_slots, dtype=torch.int32, device=device
        )
        self.gathered_batch[restore] = batch_by_position
        self.gathered_position[restore] = position_by_position

    def half_kv_lengths(self, batch: int) -> tuple[int, int]:
        """Return the low/high zigzag requests' KV lengths in padded space."""
        half_length = self.half_lengths[batch]
        prefix_length = self.prefix_lengths[batch]
        padded_length = self.padded_lengths[batch]
        return (
            prefix_length + (self.cp_rank + 1) * half_length,
            prefix_length + padded_length - self.cp_rank * half_length,
        )

    def paged_plan_tensors(
        self, block_ids_host: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build two overlapping paged requests per non-empty batch item."""
        qo_indptr = [0]
        kv_indptr = [0]
        last_page_lengths: List[int] = []
        page_indices: List[torch.Tensor] = []

        for batch in range(self.batch_size):
            half_length = self.half_lengths[batch]
            if half_length == 0:
                continue
            block_ids = block_ids_host[batch]
            for kv_length in self.half_kv_lengths(batch):
                page_count = (kv_length + self.page_size - 1) // self.page_size
                assert page_count > 0, "non-empty CP request has no KV page"
                page_indices.append(block_ids[:page_count])
                kv_indptr.append(kv_indptr[-1] + page_count)
                last_page_lengths.append(kv_length - (page_count - 1) * self.page_size)
                qo_indptr.append(qo_indptr[-1] + half_length)

        assert page_indices, "fused CP prefill produced no paged requests"
        return (
            torch.tensor(qo_indptr, dtype=torch.int32),
            torch.tensor(kv_indptr, dtype=torch.int32),
            torch.cat(page_indices).to(dtype=torch.int32, device=self.device),
            torch.tensor(last_page_lengths, dtype=torch.int32),
        )


class PCPFusedPagedAttnOp:
    """All-gather CP prefill using a packed collective and paged attention."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
        backend: str = "auto",
        causal: bool = True,
        kv_layout: str = "NHD",
    ):
        assert causal, "CP prefill only supports causal attention"
        assert parallelism_config is not None

        self.attn_inputs = attn_inputs
        self.attn_configs = attn_configs
        self.num_qo_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.seq_size_per_block = (
            attn_configs.kernel_tokens_per_block or attn_configs.tokens_per_block
        )
        assert self.seq_size_per_block == attn_configs.tokens_per_block, (
            "fused CP prefill requires kernel_tokens_per_block "
            f"({attn_configs.kernel_tokens_per_block}) to equal tokens_per_block "
            f"({attn_configs.tokens_per_block})"
        )

        self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = get_scalar_type(attn_inputs.dtype)
        self.cp_info = attn_inputs.context_parallel_info
        self.prefill_cp_rank = parallelism_config.tp_rank
        self.prefill_cp_size = parallelism_config.tp_size
        self.workspace_buffer = get_py_flashinfer_workspace_buffer()
        self.paged_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, kv_layout="HND", backend=backend
        )

        self.geometry: Optional[_CPGeometry] = None
        self.kv_gather: Optional[torch.Tensor] = None
        self.kv_local: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.process_group = None

    def __del__(self):
        if hasattr(self, "workspace_buffer"):
            release_py_flashinfer_workspace_buffer(self.workspace_buffer)

    def support(self, attention_inputs: PyAttentionInputs) -> bool:
        return attention_inputs.is_prefill

    def prepare(self, attention_inputs: PyAttentionInputs) -> ParamsBase:
        geometry = _CPGeometry(
            attention_inputs,
            self.prefill_cp_size,
            self.prefill_cp_rank,
            self.seq_size_per_block,
            self.device,
        )
        self.geometry = geometry

        # CPFlashInferImpl is shared by every transformer layer in a prefill. The
        # request-scoped scratch below is therefore allocated once and reused
        # by all layers, without process-global buffers that could alias requests.
        self.kv_gather = torch.empty(
            geometry.gathered_slots,
            2,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.kv_local = torch.empty(
            geometry.tokens_local,
            2,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.output = torch.empty(
            geometry.tokens_local,
            self.num_qo_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.process_group = _get_group(Group.TP)

        params = fill_mla_params(
            attention_inputs.prefix_lengths,
            attention_inputs.sequence_lengths,
            self.cp_info.prefill_actual_input_lengths_cpu,
            attention_inputs.kv_cache_kernel_block_id,
            self.seq_size_per_block,
        )
        qo_indptr, kv_indptr, page_indices, last_page_lengths = (
            geometry.paged_plan_tensors(attention_inputs.kv_cache_kernel_block_id)
        )
        self.paged_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=page_indices,
            paged_kv_last_page_len=last_page_lengths,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            page_size=self.seq_size_per_block,
            causal=True,
            q_data_type=self.dtype,
        )
        return params

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        params: Optional[ParamsBase] = None,
    ) -> torch.Tensor:
        assert self.geometry is not None
        assert self.kv_gather is not None and self.kv_local is not None
        assert self.output is not None and kv_cache is not None and params is not None

        qkv = qkv.view(qkv.shape[0], -1)
        q_width = self.num_qo_heads * self.head_dim
        q = qkv[:, :q_width].view(-1, self.num_qo_heads, self.head_dim)

        # Pack K and V into one local tensor. NCCL gathers both into the persistent
        # destination without relying on overlapping collective input/output.
        self.kv_local.copy_(
            qkv[:, q_width:].view(-1, 2, self.num_kv_heads, self.head_dim)
        )
        torch.distributed.all_gather_into_tensor(
            self.kv_gather, self.kv_local, group=self.process_group
        )

        kv_cache_tensor = kv_cache.kv_cache_base.view(
            -1,
            2,
            self.num_kv_heads,
            self.seq_size_per_block,
            self.head_dim,
        )
        append_paged_kv_cache(
            append_key=self.kv_gather[:, 0],
            append_value=self.kv_gather[:, 1],
            batch_indices=self.geometry.gathered_batch,
            positions=self.geometry.gathered_position,
            paged_kv_cache=kv_cache_tensor,
            kv_indices=params.page_indice_d,
            kv_indptr=params.decode_page_indptr_d,
            kv_last_page_len=params.paged_kv_last_page_len_d,
            kv_layout="HND",
        )

        return self.paged_wrapper.run(q, kv_cache_tensor, out=self.output)

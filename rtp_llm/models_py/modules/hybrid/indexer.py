from typing import Dict, Optional, Tuple

import deep_gemm
import flashinfer.rope as rope
import torch
from torch import nn

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules import LayerNorm
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops import AttentionConfigs, HWKernelConfig
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs, rtp_llm_ops
from rtp_llm.utils.model_weight import W

from .test.indexer_ref import act_quant


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Hadamard transform for activation rotation.
    Note: This is a simplified version. For production use, you may need
    to use optimized CUDA kernels like sgl_kernel.hadamard_transform.
    """
    assert x.dtype == torch.bfloat16
    from fast_hadamard_transform import hadamard_transform

    hidden_size = x.size(-1)
    assert (
        hidden_size & (hidden_size - 1)
    ) == 0, "Hidden size must be a power of 2 for Hadamard transform."

    return hadamard_transform(x, scale=hidden_size**-0.5)


# 正确的 UE8M0 unpacking
def unpack_ue8m0_scale(sf_packed: torch.Tensor) -> torch.Tensor:
    # sf_packed: (..., num_scales), dtype=int32
    # UE8M0 格式：scale 存储在 int32 的最低字节

    # 直接使用位运算提取最低字节，避免 view 操作
    sf_u8 = (sf_packed & 0xFF).to(torch.int32)  # 提取最低字节
    # 左移到 float32 的指数位置（bit 23-30）
    sf_i32 = sf_u8 << 23
    # 重新解释为 float32
    sf_fp32 = sf_i32.view(torch.float32)
    return sf_fp32


class Indexer(nn.Module):
    """
    Indexer for DeepSeek-V3.2 DSA (DeepSeek Sparse Attention) mechanism.
    Adapted from sglang's Indexer implementation.
    """

    def __init__(
        self,
        attn_config: AttentionConfigs,
        weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        layer_idx: int,
        layernorm_eps: float,
        quant_config: object,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        self.index_n_heads = attn_config.indexer_head_num
        self.index_head_dim = attn_config.indexer_head_dim
        self.index_topk = attn_config.indexer_topk

        self.rope_head_dim = attn_config.rope_head_dim
        self.block_size = 128
        self.scale_fmt = "ue8m0"  # FP8 quantization format
        self.softmax_scale = self.index_head_dim**-0.5
        self.weights_scale = self.index_n_heads**-0.5
        self.blocksize = attn_config.tokens_per_block
        self.indexer_size = self.index_head_dim / 2 + self.index_head_dim / 128 * 2

        self.wq_b = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_qb_w,
            W.mla_indexer_qb_s,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        self.wk = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_k_w,
            W.mla_indexer_k_s,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        self.k_norm = LayerNorm(
            weights[W.mla_indexer_k_norm_w],
            weights[W.mla_indexer_k_norm_b],
            eps=layernorm_eps,
        )

        self.weights_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_weights_proj_w,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        self.cos_sin_cache = global_weights[W.rope_cos_sin_cache]

    def prepare(self, attention_inputs: PyAttentionInputs):
        """Prepare indexer parameters from attention inputs"""
        self.params = self._prepare_params(attention_inputs)
        self.is_prefill = attention_inputs.is_prefill

    def _prepare_params(self, attention_inputs: PyAttentionInputs):

        from types import SimpleNamespace

        # Extract basic information
        is_prefill = attention_inputs.is_prefill
        input_lengths = attention_inputs.input_lengths  # [batch_size]
        sequence_lengths = attention_inputs.sequence_lengths  # [decode_batch_size]
        kv_cache_block_id = (
            attention_inputs.kv_cache_block_id_device
        )  # [batch_size, max_blocks]
        seq_size_per_block = 64  # Page size, typically 64

        if is_prefill:
            # Prefill mode
            batch_size = input_lengths.size(0)
            seq_lens = input_lengths.to(torch.int32)
            extend_seq_lens = input_lengths.cpu().tolist()

            # cu_seqlens_q: cumulative sequence lengths [0, len1, len1+len2, ...]
            cu_seqlens_q = attention_inputs.cu_seqlens.to(torch.int32).to("cuda")

            # expanded_seq_lens: repeat each seq_len for each token in that sequence
            # For example: if seq_lens = [3, 2], expanded = [3, 3, 3, 2, 2]
            expanded_seq_lens = torch.repeat_interleave(seq_lens, seq_lens)

            # topk_indices_offset: cumulative offset for ragged KV layout
            # cu_seqlens_q[:-1] gives the starting position of each sequence
            topk_indices_offset = torch.repeat_interleave(
                cu_seqlens_q[:-1], seq_lens.to("cuda")
            ).to(torch.int32)

            # Calculate total tokens and allocate tensors
            total_tokens = seq_lens.sum().item()
            positions_d = torch.empty(total_tokens, dtype=torch.int32, device="cuda")
            batch_indice_d = torch.empty(total_tokens, dtype=torch.int32, device="cuda")

            # Reference: FlashInferMlaParams.cc fillParamsInternal (prefill branch)
            offset = 0
            for i in range(batch_size):
                input_length = extend_seq_lens[i]
                # prefix_length: length already in KV cache = sequence_length - input_length
                # If sequence_lengths is defined, use it; otherwise assume prefix_length = 0
                if sequence_lengths is not None and sequence_lengths.size(0) > i:
                    prefix_length = sequence_lengths[i].item() - input_length
                else:
                    prefix_length = 0

                # Fill batch_indice_d and positions_d for this batch
                # Reference: FlashInferMlaParams.cc line 202-203
                for j in range(input_length):
                    batch_indice_d[offset + j] = i
                    positions_d[offset + j] = j + prefix_length

                offset += input_length

            # Calculate kv_indices, kv_indptr, kv_last_page_len
            # Reference: FlashInferMlaParams.cc lines 252-268
            kv_indptr = torch.empty(batch_size + 1, dtype=torch.int32, device="cuda")
            kv_last_page_len = torch.empty(batch_size, dtype=torch.int32, device="cuda")
            kv_indptr[0] = 0

            total_pages = 0
            kv_indices_list = []

            for i in range(batch_size):
                # seq_len is the total KV length including both prefix and new tokens
                input_length = extend_seq_lens[i]
                if sequence_lengths is not None and sequence_lengths.size(0) > i:
                    seq_len = sequence_lengths[i].item()
                else:
                    seq_len = input_length

                # Calculate number of pages needed for this sequence
                # Reference: FlashInferMlaParams.cc line 256
                current_page_num = (
                    seq_len + seq_size_per_block - 1
                ) // seq_size_per_block

                # Extract page indices from kv_cache_block_id
                # Reference: FlashInferMlaParams.cc lines 262-265
                if kv_cache_block_id is not None and kv_cache_block_id.size(0) > 0:
                    page_indices = kv_cache_block_id[i, :current_page_num]
                    kv_indices_list.append(page_indices)
                else:
                    # Fallback: sequential page allocation
                    page_indices = torch.arange(
                        total_pages,
                        total_pages + current_page_num,
                        dtype=torch.int32,
                        device="cuda",
                    )
                    kv_indices_list.append(page_indices)

                total_pages += current_page_num

                # Fill kv_indptr
                # Reference: FlashInferMlaParams.cc line 267
                kv_indptr[i + 1] = total_pages

                # Calculate last page length
                # Reference: FlashInferMlaParams.cc line 252
                kv_last_page_len[i] = (seq_len - 1) % seq_size_per_block + 1

            kv_indices = torch.cat(kv_indices_list, dim=0)

            # page_table_1: use kv_cache_block_id_device if available
            max_seq_len = seq_lens.max().item()
            page_table_1 = (
                torch.arange(
                    max_seq_len, dtype=torch.int32, device=input_lengths.device
                )
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        else:
            # Decode mode
            batch_size = sequence_lengths.size(0)
            seq_lens = sequence_lengths.to(torch.int32)
            extend_seq_lens = [1] * batch_size

            # In decode mode, each request generates 1 token
            cu_seqlens_q = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=sequence_lengths.device
            )

            # expanded_seq_lens: in decode mode, it's same as seq_lens
            expanded_seq_lens = seq_lens

            # topk_indices_offset: not used in decode mode, but set for compatibility
            topk_indices_offset = cu_seqlens_q[:-1]

            # Calculate batch_indice_d and positions_d for decode mode
            # Reference: FlashInferMlaParams.cc line 246-247
            batch_indice_d = torch.arange(batch_size, dtype=torch.int32, device="cuda")
            positions_d = sequence_lengths.to(torch.int32).to("cuda")

            # Calculate kv_indices, kv_indptr, kv_last_page_len for decode
            kv_indptr = torch.empty(batch_size + 1, dtype=torch.int32, device="cuda")
            kv_last_page_len = torch.empty(batch_size, dtype=torch.int32, device="cuda")
            kv_indptr[0] = 0

            total_pages = 0
            kv_indices_list = []

            for i in range(batch_size):
                # In decode mode, seq_len is sequence_lengths[i] + 1 (new token)
                # Reference: FlashInferMlaParams.cc line 248
                seq_len = sequence_lengths[i].item() + 1

                # Calculate number of pages needed
                current_page_num = (
                    seq_len + seq_size_per_block - 1
                ) // seq_size_per_block

                # Extract page indices from kv_cache_block_id
                if kv_cache_block_id is not None and kv_cache_block_id.size(0) > 0:
                    page_indices = kv_cache_block_id[i, :current_page_num]
                    kv_indices_list.append(page_indices)
                else:
                    # Fallback: sequential page allocation
                    page_indices = torch.arange(
                        total_pages,
                        total_pages + current_page_num,
                        dtype=torch.int32,
                        device="cuda",
                    )
                    kv_indices_list.append(page_indices)

                total_pages += current_page_num
                kv_indptr[i + 1] = total_pages

                # Calculate last page length
                kv_last_page_len[i] = (seq_len - 1) % seq_size_per_block + 1

            kv_indices = torch.cat(kv_indices_list, dim=0)

            # page_table_1: kv cache block indices
            max_seq_len = seq_lens.max().item()
            page_table_1 = (
                torch.arange(
                    max_seq_len, dtype=torch.int32, device=sequence_lengths.device
                )
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        # Calculate slot_mapping for indexer KV cache
        # slot_mapping maps logical token positions to physical KV cache locations
        # Formula: slot_mapping = block_numbers * block_size + block_offsets
        #
        # Example:
        # - positions_d = [0, 1, 2, ..., 65, 66]  (token positions in sequence)
        # - block_size = 64
        # - block_indices = [0, 0, 0, ..., 1, 1]  (logical block index)
        # - block_table[batch_id] = [100, 200, ...]  (physical block IDs)
        # - block_numbers = [100, 100, ..., 200, 200]
        # - block_offsets = [0, 1, 2, ..., 1, 2]
        # - slot_mapping = [6400, 6401, ..., 12801, 12802]

        total_tokens = positions_d.shape[0]

        # Step 1: Calculate block indices (which logical block each token belongs to)
        block_indices = positions_d // seq_size_per_block  # [total_tokens]

        # Step 2: Calculate block offsets (position within each block)
        block_offsets = positions_d % seq_size_per_block  # [total_tokens]

        # Step 3: Get physical block numbers from block_table using gather
        # Create linear indices for gathering: batch_id * max_blocks + block_idx
        if kv_cache_block_id is not None and kv_cache_block_id.numel() > 0:
            max_blocks = kv_cache_block_id.size(1)
            # Flatten block_table for gather operation
            flat_block_table = kv_cache_block_id.view(-1)  # [batch_size * max_blocks]

            # Calculate flat indices: batch_id * max_blocks + block_idx
            flat_indices = batch_indice_d.long() * max_blocks + block_indices.long()

            # Clamp indices to valid range
            flat_indices = torch.clamp(flat_indices, 0, flat_block_table.size(0) - 1)

            # Gather physical block numbers
            block_numbers = flat_block_table[flat_indices]  # [total_tokens]

            # Step 4: Calculate final slot mapping
            slot_mapping = (
                block_numbers.long() * seq_size_per_block + block_offsets.long()
            )
        else:
            # Fallback: use kv_indices if block_table is not available
            # This assumes kv_indices contains sequential physical block IDs
            slot_mapping = torch.empty(
                total_tokens, dtype=torch.int64, device=positions_d.device
            )
            for i in range(total_tokens):
                batch_id = batch_indice_d[i].item()
                block_idx = block_indices[i].item()

                # Get physical block from kv_indices
                page_start = kv_indptr[batch_id].item()
                page_end = kv_indptr[batch_id + 1].item()

                if page_start + block_idx < page_end:
                    block_number = kv_indices[page_start + block_idx].item()
                    slot_mapping[i] = (
                        block_number * seq_size_per_block + block_offsets[i]
                    )
                else:
                    slot_mapping[i] = -1  # Invalid slot

        return SimpleNamespace(
            expanded_seq_lens=expanded_seq_lens,
            page_table_1=page_table_1,
            cu_seqlens_q=cu_seqlens_q,
            topk_indices_offset=topk_indices_offset,
            batch_size=batch_size,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
            positions_d=positions_d,
            batch_indice_d=batch_indice_d,
            kv_indices=kv_indices,
            kv_indptr=kv_indptr,
            kv_last_page_len=kv_last_page_len,
            slot_mapping=slot_mapping,
            block_table=attention_inputs.kv_cache_block_id_device,
            cu_seq_lens=attention_inputs.cu_seqlens,
        )

    # @torch.compile(dynamic=True)
    def _get_logits_head_gate(
        self, x: torch.Tensor, q_scale: torch.Tensor
    ) -> torch.Tensor:
        weights = self.weights_proj(x.float())
        weights = weights * self.weights_scale
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        return weights

    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        ks: Optional[torch.Tensor] = None,
        is_paged: bool = False,
        is_ragged: bool = False,
        use_nas_fuse_topk: bool = True,
    ) -> torch.Tensor:
        from rtp_llm.models_py.kernels.cuda.fast_topk import (
            fast_topk_transform_fused,
            fast_topk_transform_ragged_fused,
            fast_topk_v2,
        )

        if not use_nas_fuse_topk:
            return fast_topk_v2(logits, self.params.expanded_seq_lens, topk)
        elif is_paged:
            # NOTE(dark): if fused, we return a transformed page table directly
            return fast_topk_transform_fused(
                score=logits,
                lengths=self.params.expanded_seq_lens.to("cuda"),  # expanded_seq_lens
                page_table_size_1=self.params.page_table_1.to("cuda"),  # page_indices
                cu_seqlens_q=self.params.cu_seqlens_q.to("cuda"),  # bs + 1
                topk=topk,
                row_starts=None,
            )
        elif is_ragged:
            return fast_topk_transform_ragged_fused(
                score=logits,
                lengths=self.params.expanded_seq_lens.to("cuda"),
                topk_indices_offset=self.params.topk_indices_offset.to("cuda"),
                topk=topk,
                row_starts=ks,
            )
        else:
            assert False, f"Unsupported {self.topk_transform_method = }"

    def _get_topk_paged(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:

        weights = weights.view(-1, self.index_n_heads)
        kv_cache_fp8 = kv_cache.kv_scale_base

        block_kv = self.blocksize
        num_heads_kv = 1
        head_dim_with_sf = 132
        kv_cache_fp8 = kv_cache_fp8.view(
            kv_cache_fp8.shape[0], block_kv, num_heads_kv, head_dim_with_sf
        ).view(dtype=torch.uint8)
        # len of k
        seqlens_32 = self.params.seq_lens.to(torch.int32).to("cuda")
        max_block_len = (
            (seqlens_32.max().item() + self.blocksize - 1)
            // self.blocksize
            * self.blocksize
        )
        block_tables = torch.zeros(
            (self.params.batch_size, max_block_len),
            device=q_fp8.device,
            dtype=torch.int32,
        )
        schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
            seqlens_32, self.blocksize, deep_gemm.get_num_sms()
        )
        max_seq_len = block_tables.shape[1] * self.blocksize

        logits = deep_gemm.fp8_paged_mqa_logits(
            q_fp8.unsqueeze(1),
            kv_cache_fp8.view(dtype=torch.uint8),
            weights,
            seqlens_32,
            block_tables,
            schedule_metadata,
            max_seq_len,
            clean_logits=False,
        )
        # NOTE(dark): logits should be cleaned in topk_transform（adapter from sglang）
        topk_result = self.topk_transform(logits, self.index_topk, is_paged=True)
        return topk_result

    def _get_topk_ragged(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
    ) -> torch.Tensor:

        weights = weights.squeeze(-1)
        kv_fp8 = (k_fp8, k_scale.view(torch.float32))
        # 收集所有 batch 的 ks 和 ke
        ks_list = []
        ke_list = []

        q_offset = 0  # query 的累积偏移
        k_offset = 0  # KV cache 的累积偏移

        for i in range(self.params.batch_size):
            seq_len = self.params.seq_lens[i]  # KV cache 总长度
            extend_seq_len = self.params.extend_seq_lens[i]  # 新增 token 数

            # 当前 batch 的所有新 token 起始位置相同
            ks = torch.full(
                (extend_seq_len,), k_offset, dtype=torch.int32, device=q_fp8.device
            )

            # 每个新 token 能看到的 KV 长度递增
            seq_lens_expanded = torch.arange(
                seq_len - extend_seq_len + 1,
                seq_len + 1,
                dtype=torch.int32,
                device=q_fp8.device,
            )

            # 结束位置 = 起始位置 + 可见长度
            ke = ks + seq_lens_expanded

            ks_list.append(ks)
            ke_list.append(ke)

            # 更新偏移量
            q_offset += extend_seq_len
            k_offset += seq_len

        # 拼接所有 batch
        ks = torch.cat(ks_list, dim=0)
        ke = torch.cat(ke_list, dim=0)

        logits = deep_gemm.fp8_mqa_logits(
            q_fp8,
            kv_fp8,
            weights,
            ks,
            ke,
            clean_logits=False,
        )
        # NOTE(dark): logits should be cleaned in topk_transform（adapter from sglang）
        topk_result = self.topk_transform(
            logits, self.index_topk, ks=ks, is_ragged=True
        )

        return topk_result

    def _get_q_k_bf16(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
    ):
        q = self.wq_b(q_lora)
        q = q.view(-1, self.index_n_heads, self.index_head_dim)
        q_pe = q[:, :, : self.index_head_dim - self.rope_head_dim]

        k = self.wk(x)
        k = self.k_norm(k)
        k_pe = k[:, : self.index_head_dim - self.rope_head_dim]

        # same as vllm indexer rope
        rope._apply_rope_pos_ids_cos_sin_cache(
            q=q_pe,
            k=k_pe.unsqueeze(1),
            q_rope=q_pe,
            k_rope=k_pe.unsqueeze(1),
            cos_sin_cache=self.cos_sin_cache,
            pos_ids=self.params.positions_d,
            interleave=False,
        )

        query = rotate_activation(q)
        key = rotate_activation(k)

        return query, key

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        query, key = self._get_q_k_bf16(q_lora, hidden_states)
        query = query.view(-1, self.index_head_dim)
        q_fp8, q_scale = sgl_per_token_group_quant_fp8(
            query,
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        q_fp8 = q_fp8.view(-1, self.index_n_heads, self.index_head_dim)
        q_scale_unpacked = unpack_ue8m0_scale(q_scale)
        q_scale = q_scale_unpacked.view(-1, self.index_n_heads, 1)
        weights = self._get_logits_head_gate(hidden_states, q_scale)

        assert kv_cache is not None, "kv_cache is required"

        slot_mapping = self.params.slot_mapping
        rtp_llm_ops.indexer_k_quant_and_cache(
            key,  # Original key in BF16/FP16 [num_tokens, index_head_dim]
            kv_cache.kv_scale_base,  # [num_blocks, block_size, cache_stride]
            slot_mapping,  # [num_tokens] physical slot indices
            self.block_size,  # quantization block size (128)
            self.scale_fmt,  # "ue8m0" for power-of-2 scaling
        )
        if self.is_prefill:
            num_tokens = key.shape[0]
            k_fp8 = torch.empty(
                (num_tokens, self.index_head_dim),
                dtype=torch.float8_e4m3fn,
                device=key.device,
            )
            k_scale = torch.empty(
                (num_tokens, self.index_head_dim // self.block_size * 4),
                dtype=torch.uint8,
                device=key.device,
            )

            rtp_llm_ops.cp_gather_indexer_k_quant_cache(
                kv_cache.kv_scale_base,  # [num_blocks, block_size, cache_stride]
                k_fp8,  # output [num_tokens, index_head_dim]
                k_scale,  # output [num_tokens, scale_size]
                self.params.block_table,  # [batch_size, num_blocks]
                self.params.cu_seq_lens,  # [batch_size + 1]
            )

        if not self.is_prefill:
            topk_result = self._get_topk_paged(q_fp8, weights, kv_cache)
        else:
            topk_result = self._get_topk_ragged(q_fp8, weights, k_fp8, k_scale)

        return topk_result

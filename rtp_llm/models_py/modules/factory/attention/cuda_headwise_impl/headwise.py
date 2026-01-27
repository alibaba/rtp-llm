from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import torch
from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
from flashinfer.cascade import merge_state

from rtp_llm.models_py.modules.factory.attention.attn_factory import ConfigManager
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHAPrefillImplBase,
    FMHAType,
)
from rtp_llm.ops import AttentionConfigs, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCachePrefillOp,
    KVCache,
    PyAttentionInputs,
)
from rtp_kernel.sparse_attention import BatchPrefillWithSparseAttention

# ----------------------------
# Data Models
# ----------------------------
@dataclass(frozen=True)
class HeadWiseRuntimeConfig:
    sink_token_num: int = 4
    swa_token_num: int = 8192
    seqlen_threshold: int = 16384


class HeadWisePrefillAttnOp:
    """
    HeadWise Prefill Attention:
      - retrieval heads: full attention
      - non-retrieval heads: sink + sliding window attention, 通过 merge_state 合并
    """

    def __init__(
        self, attn_configs: AttentionConfigs, parallelism_config: ParallelismConfig
    ) -> None:

        self.rank = parallelism_config.tp_rank

        self.head_num = attn_configs.head_num
        self.head_num_kv = attn_configs.kv_head_num
        self.size_per_head = attn_configs.size_per_head
        self.paged_size = attn_configs.tokens_per_block

        self.dtype = torch.bfloat16

        if ConfigManager.get_headwise_config() is not None:
            self.headwise_all_config = ConfigManager.get_headwise_config()
        logging.info(f"self.headwise_all_config = {self.headwise_all_config}")
        self.hw_cfg = HeadWiseRuntimeConfig(
            sink_token_num=self.headwise_all_config.get("sink_token_num", 4),
            swa_token_num=self.headwise_all_config.get("swa_token_num", 8192),
            seqlen_threshold=self.headwise_all_config.get("seqlen_threshold", 16384),
        )

        self.workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        self.retrieval_heads: Optional[torch.Tensor] = None
        self.non_retrieval_heads: Optional[torch.Tensor] = None
        self.input_lengths: Optional[torch.Tensor] = None
        self.kv_lengths: Optional[torch.Tensor] = None


        self.full_attention_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, "HND", backend="fa3"
        )
        self.sparse_attention_wrapper = BatchPrefillWithSparseAttention(
            self.workspace_buffer, "HND", backend="fa3"
        )

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        # rtp_kernel only support cuda 12.8+
        major, minor = map(int, torch.version.cuda.split(".")[:2])
        return (major, minor) >= (12, 8)

    def _get_paged_metadata(
        self, q_len: int, kv_len: int, kv_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_indptr = torch.tensor([0, q_len], dtype=torch.int32, device="cuda")
        num_pages = (kv_len + self.paged_size - 1) // self.paged_size
        kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device="cuda")
        kv_last_page_len = torch.tensor(
            [(kv_len - 1) % self.paged_size + 1], dtype=torch.int32, device="cuda"
        )
        return q_indptr, kv_indptr, kv_indices, kv_last_page_len

    # ----------------------------
    # Headwise config
    # ----------------------------
    def _get_headwise_config(self, layer_idx: int):
        """根据层索引提取并分类当前 Rank 负责的头"""
        start = self.head_num * self.rank
        end = start + self.head_num

        layer_config = torch.tensor(
            self.headwise_all_config[str(layer_idx)], device="cuda"
        )
        current_rank_weights = layer_config[start:end]

        self.non_retrieval_heads = current_rank_weights == 0
        self.retrieval_heads = current_rank_weights == 1

    def prepare(self, attn_inputs: PyAttentionInputs) -> None:
        self.input_lengths = attn_inputs.input_lengths
        self.kv_lengths = attn_inputs.input_lengths  if attn_inputs.prefix_lengths.item()==0 else attn_inputs.input_lengths + attn_inputs.prefix_lengths
        self.kv_indices = attn_inputs.kv_cache_block_id_device
        self.batch_wrappers = []
        self.use_headwise_flags = []

        for i, length_tensor in enumerate(self.input_lengths):
            q_len = int(length_tensor.item())
            kv_len = int(self.kv_lengths[i].item()) if self.kv_lengths[i] > 0 else q_len

            item = self._plan_one_sequence(
                q_len=q_len, kv_len=kv_len, kv_indices=self.kv_indices[i]
            )
            self.batch_wrappers.append(item[0])
            self.use_headwise_flags.append(item[1])

    def _plan_one_sequence(
        self, q_len: int, kv_len: int, kv_indices: torch.Tensor
    ):
        """为单条序列构建 wrappers 并 plan。"""
        self.meta = self._get_paged_metadata(q_len, kv_len, kv_indices)
        
       
        # 小于16384：直接 full attention
        if kv_len < self.hw_cfg.seqlen_threshold or q_len < self.hw_cfg.sink_token_num:
            self.full_attention_wrapper.plan(
                *self.meta,
                num_qo_heads=self.head_num,
                num_kv_heads=self.head_num_kv,
                head_dim_qk=self.size_per_head,
                page_size=self.paged_size,
                causal=True,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
            )
            return (self.full_attention_wrapper, False)
        # 大于16384：
        else:
            #prefix cache hit && only need combine ( rectangle party + sliding window attention)
            # if q_len < kv_len:
            #     pass
            # # no prefix or prefix cache not hit
            # else:
            qo_head_split_indptr = torch.tensor([0, self.hw_cfg.sink_token_num], 
                                    dtype=torch.int32, device="cuda")
            qo_indptr = torch.tensor([0, q_len - self.hw_cfg.sink_token_num],
                                    dtype=torch.int32, device="cuda")
            kv_indptr = torch.tensor([0, kv_len],
                                    dtype=torch.int32, device="cuda") 
            
            seq_lens_tensor = torch.tensor([kv_len], dtype=torch.int32, device="cuda")
            
            self.sparse_attention_wrapper.plan(
                qo_head_split_indptr, qo_indptr, kv_indptr,
                num_qo_heads=self.head_num, num_kv_heads=self.head_num_kv, head_dim=self.size_per_head,
                head_split=self.hw_cfg.sink_token_num,
                seq_lens=seq_lens_tensor, block_tables=kv_indices.unsqueeze(0),
                window_left=self.hw_cfg.swa_token_num,
                q_data_type=self.dtype, kv_data_type=self.dtype,
            )
            return (self.sparse_attention_wrapper, True)

    # ----------------------------
    # Forward logic
    # ----------------------------
    def forward(
        self, fmha_input: torch.Tensor, kv_cache: Any, fmha_params: Any
    ) -> torch.Tensor:
        total_tokens = fmha_input.shape[0]
        output = torch.empty(
            (total_tokens, self.head_num, self.size_per_head),
            dtype=fmha_input.dtype,
            device=fmha_input.device,
        )
       
        k_cache = kv_cache.kv_cache_base[:, 0, ...]
        v_cache = kv_cache.kv_cache_base[:, 1, ...]

        offset = 0
        for i, wrapper in enumerate(self.batch_wrappers):
            
            q_len = int(self.input_lengths[i].item())
            kv_len = int(self.kv_lengths[i].item()) if self.kv_lengths[i] > 0 else q_len

            q, self.k, self.v = self._slice_q(fmha_input, offset, q_len)
            if self.use_headwise_flags[i]:
                res = self._apply_headwise(
                    q, k_cache, v_cache, wrapper, self.kv_indices[i], q_len=q_len, kv_len=kv_len
                )
            else:
                res = self.full_attention_wrapper.run(q, (k_cache, v_cache))

            output[offset : offset + q_len] = res
            offset += q_len

        return output.view(total_tokens, -1)

    def _slice_q(
        self, fmha_input: torch.Tensor, offset: int, q_len: int
    ) -> torch.Tensor:
        qkv = fmha_input[offset : offset + q_len].view(q_len, -1, self.size_per_head)
        q, k, v = torch.split(
            qkv, [self.head_num, self.head_num_kv, self.head_num_kv], dim=1
        )
        return q, k, v

    def _apply_headwise(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        wrapper,
        kv_indices: torch.Tensor,
        q_len: int,
        kv_len: int,
    ) -> torch.Tensor:
        out = torch.empty(
            (q_len, self.head_num, self.size_per_head), dtype=q.dtype, device=q.device
        )

        # 1) retrieval heads: full attention
        if self.retrieval_heads is not None and self.retrieval_heads.any():
            # Plan full_attention_wrapper before using it for retrieval heads
            num_retrieval_heads = self.retrieval_heads.sum().item()
            self.full_attention_wrapper.plan(
                *self.meta,
                num_qo_heads=num_retrieval_heads,
                num_kv_heads=self.head_num_kv,
                head_dim_qk=self.size_per_head,
                page_size=self.paged_size,
                causal=True,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
            )
            out[:, self.retrieval_heads, :] = self.full_attention_wrapper.run(
                q[:, self.retrieval_heads, :], (k_cache, v_cache)
            )

        # 2) non-retrieval heads: sink + swa
        if self.non_retrieval_heads is not None and self.non_retrieval_heads.any():
            h = self.non_retrieval_heads
            out[:, h, :] = self._run_non_retrieval(
                q[:, h, :], k_cache, v_cache, q_len=q_len, kv_len=kv_len
            )

        return out

    def _run_non_retrieval(
        self,
        q_h: torch.Tensor,  # [q_len, Hn, D]
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        q_len: int,
        kv_len: int,
    ) -> torch.Tensor:
        k_cache_contiguous = k_cache.contiguous()
        v_cache_contiguous = v_cache.contiguous()
        
        if q_len == kv_len:
            qf1 = q_h[:self.hw_cfg.sink_token_num]
            qf2 = q_h[self.hw_cfg.sink_token_num:]
            return self.sparse_attention_wrapper.run(qf1, qf2, k_cache=k_cache_contiguous, v_cache=v_cache_contiguous)
        else:
            return self.sparse_attention_wrapper.run(None, q_h, k_cache=k_cache_contiguous, v_cache=v_cache_contiguous)


class HeadWisePrefillImpl(FMHAPrefillImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        parallelism_config: ParallelismConfig,
        attn_inputs: PyAttentionInputs,
    ) -> None:
        super().__init__(
            HeadWisePrefillAttnOp(attn_configs, parallelism_config),
            FusedRopeKVCachePrefillOp(attn_configs),
            attn_inputs,
        )

    def support(self) -> bool:
        return True

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.HEADWISE

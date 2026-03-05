from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import torch

_HAS_FLASHINFER = False
_HAS_RTP_KERNEL = False

try:
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    from flashinfer.cascade import merge_state
    _HAS_FLASHINFER = True
except ImportError as e:
    logging.warning(f"FlashInfer not found: {e}")

try:
    from rtp_kernel.sparse_attention import BatchPrefillWithSparseAttention
    _HAS_RTP_KERNEL = True
except ImportError as e:
    logging.warning(f"rtp_kernel.sparse_attention not found: {e}")

try:
    from rtp_llm.models_py.modules.factory.attention import common
    from rtp_llm.ops.compute_ops import (
        FusedRopeKVCachePrefillOpQKVOut,
        PyAttentionInputs,
    )
except ImportError as e:
    logging.warning(f"rtp_llm attention common/compute_ops not found: {e}")

from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, ParallelismConfig


class ConfigManager:
    _headwise_config = None

    @classmethod
    def set_headwise_config(cls, config):
        if cls._headwise_config is not None:
            logging.info("Warning: headwise_config already set. Ignoring new config.")
            return

        if not hasattr(config, "headwise_config"):
            logging.info("Model Not Support Headwise")
            return

        cls._headwise_config = config.headwise_config

    @classmethod
    def get_headwise_config(cls):
        return cls._headwise_config

    @classmethod
    def is_config_set(cls):
        return cls._headwise_config is not None


# ----------------------------
# Data Models
# ----------------------------
@dataclass(frozen=True)
class HeadWiseRuntimeConfig:
    sink_token_num: int = 4
    swa_token_num: int = 8192
    seqlen_threshold: int = 16384


@dataclass
class BatchWrapperItem:
    use_headwise: bool
    full_wrappers: Optional[Any] = None
    swa_wrappers: Optional[Any] = None
    # 存储 meta 信息，用于 forward 阶段的 retrieval heads
    meta: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None
    q_len: int = 0
    kv_len: int = 0
    kv_indices: Optional[torch.Tensor] = None


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

        if ConfigManager.is_config_set() is not None:
            self.headwise_all_config = ConfigManager.get_headwise_config()

            self.hw_cfg = HeadWiseRuntimeConfig(
                sink_token_num=self.headwise_all_config.get("sink_token_num", 4),
                swa_token_num=self.headwise_all_config.get("swa_token_num", 8192),
                seqlen_threshold=self.headwise_all_config.get(
                    "seqlen_threshold", 16384
                ),
            )

        self.workspace_buffer = torch.empty(
            256 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        )
        self.workspace_sparse_buffer = torch.empty(
            256 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        )

        # runtime states
        self.retrieval_heads: Optional[torch.Tensor] = None
        self.non_retrieval_heads: Optional[torch.Tensor] = None
        self.batch_wrappers: List[BatchWrapperItem] = []
        self.input_lengths: Optional[torch.Tensor] = None
        self.kv_lengths: Optional[torch.Tensor] = None

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        if not (_HAS_FLASHINFER and _HAS_RTP_KERNEL):
            return False
        major, minor = map(int, torch.version.cuda.split(".")[:2])
        return (major, minor) >= (12, 8) and ConfigManager.is_config_set()

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
        self.kv_lengths = attn_inputs.input_lengths + attn_inputs.prefix_lengths
        self.kv_indices = attn_inputs.kv_cache_block_id_device

        self.batch_wrappers = []

        for i, length_tensor in enumerate(self.input_lengths):
            q_len = int(length_tensor.item())
            kv_len = int(self.kv_lengths[i].item()) if self.kv_lengths[i] > 0 else q_len

            wrapper_item = self._plan_one_sequence(
                q_len=q_len, kv_len=kv_len, kv_indices=self.kv_indices[i]
            )
            self.batch_wrappers.append(wrapper_item)

    def _plan_one_sequence(
        self, q_len: int, kv_len: int, kv_indices: torch.Tensor
    ) -> BatchWrapperItem:
        meta = self._get_paged_metadata(q_len, kv_len, kv_indices)

        # small than 16384
        if kv_len < self.hw_cfg.seqlen_threshold or q_len < self.hw_cfg.sink_token_num:
            full_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer, "HND", backend="fa3"
            )
            full_wrapper.plan(
                *meta,
                num_qo_heads=self.head_num,
                num_kv_heads=self.head_num_kv,
                head_dim_qk=self.size_per_head,
                page_size=self.paged_size,
                causal=True,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
            )
            return BatchWrapperItem(
                use_headwise=False,
                full_wrappers=full_wrapper,
                meta=meta,
                q_len=q_len,
                kv_len=kv_len,
                kv_indices=kv_indices,
            )

        else:
            sparse_wrapper = BatchPrefillWithSparseAttention(
                self.workspace_sparse_buffer, "HND", backend="fa3"
            )

            qo_head_split_indptr = torch.tensor(
                [0, self.hw_cfg.sink_token_num], dtype=torch.int32, device="cuda"
            )
            qo_indptr = torch.tensor(
                [0, q_len - self.hw_cfg.sink_token_num],
                dtype=torch.int32,
                device="cuda",
            )
            kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device="cuda")

            seq_lens_tensor = torch.tensor([kv_len], dtype=torch.int32, device="cuda")

            sparse_wrapper.plan(
                qo_head_split_indptr,
                qo_indptr,
                kv_indptr,
                num_qo_heads=self.head_num,
                num_kv_heads=self.head_num_kv,
                head_dim=self.size_per_head,
                head_split=self.hw_cfg.sink_token_num,
                seq_lens=seq_lens_tensor,
                block_tables=kv_indices.unsqueeze(0),
                window_left=self.hw_cfg.swa_token_num,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
            )
            return BatchWrapperItem(
                use_headwise=True,
                swa_wrappers=sparse_wrapper,
                meta=meta,
                q_len=q_len,
                kv_len=kv_len,
                kv_indices=kv_indices,
            )

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
        kv_base = kv_cache.kv_cache_base  # [block_num, 2*kv_head_num*page_size*head_dim] (packed 2D)
        block_num = kv_base.shape[0]
        kv_expanded = kv_base.view(block_num, 2, self.head_num_kv, self.paged_size, self.size_per_head)
        k_cache = kv_expanded[:, 0, ...].contiguous()     # [block_num, kv_head_num, page_size, head_dim]
        v_cache = kv_expanded[:, 1, ...].contiguous()     # [block_num, kv_head_num, page_size, head_dim]

        offset = 0
        for i, wrapper_item in enumerate(self.batch_wrappers):

            q_len = int(self.input_lengths[i].item())
            kv_len = int(self.kv_lengths[i].item()) if self.kv_lengths[i] > 0 else q_len

            q = self._slice_q(fmha_input, offset, q_len)

            if wrapper_item.use_headwise:
                res = self._apply_headwise(
                    q, k_cache, v_cache, wrapper_item, q_len=q_len, kv_len=kv_len
                )
            else:
                res = wrapper_item.full_wrappers.run(q, (k_cache, v_cache))

            output[offset : offset + q_len] = res
            offset += q_len

        return output.view(total_tokens, -1)

    def _slice_q(
        self, fmha_input: torch.Tensor, offset: int, q_len: int
    ) -> torch.Tensor:
        qkv = fmha_input[offset : offset + q_len].view(q_len, -1, self.size_per_head)
        q, _, _ = torch.split(
            qkv, [self.head_num, self.head_num_kv, self.head_num_kv], dim=1
        )
        return q

    def _apply_headwise(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        wrapper_item: BatchWrapperItem,
        q_len: int,
        kv_len: int,
    ) -> torch.Tensor:
        out = torch.empty(
            (q_len, self.head_num, self.size_per_head), dtype=q.dtype, device=q.device
        )

        if self.retrieval_heads is not None and self.retrieval_heads.any():
            num_retrieval_heads = self.retrieval_heads.sum().item()
            retrieval_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer, "HND", backend="fa3"
            )
            retrieval_wrapper.plan(
                *wrapper_item.meta,
                num_qo_heads=num_retrieval_heads,
                num_kv_heads=self.head_num_kv,
                head_dim_qk=self.size_per_head,
                page_size=self.paged_size,
                causal=True,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
            )
            out[:, self.retrieval_heads, :] = retrieval_wrapper.run(
                q[:, self.retrieval_heads, :], (k_cache, v_cache)
            )

        if self.non_retrieval_heads is not None and self.non_retrieval_heads.any():
            h = self.non_retrieval_heads
            out[:, h, :] = self._run_non_retrieval(
                q[:, h, :], k_cache, v_cache, wrapper_item, q_len=q_len, kv_len=kv_len
            )

        return out

    def _run_non_retrieval(
        self,
        q_h: torch.Tensor,  # [q_len, Hn, D]
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        wrapper_item: BatchWrapperItem,
        q_len: int,
        kv_len: int,
    ) -> torch.Tensor:
        k_cache_contiguous = k_cache.contiguous()
        v_cache_contiguous = v_cache.contiguous()

        if q_len == kv_len:
            qf1 = q_h[: self.hw_cfg.sink_token_num]
            qf2 = q_h[self.hw_cfg.sink_token_num :]
            return wrapper_item.swa_wrappers.run(
                qf1, qf2, k_cache=k_cache_contiguous, v_cache=v_cache_contiguous
            )
        else:
            return wrapper_item.swa_wrappers.run(
                None, q_h, k_cache=k_cache_contiguous, v_cache=v_cache_contiguous
            )


class HeadWisePrefillImpl(FMHAImplBase):

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = HeadWisePrefillAttnOp(attn_configs, parallelism_config)
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQKVOut(attn_configs)
        self.attn_configs = attn_configs

        # Store input info
        self.attn_inputs = attn_inputs

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:

        return not attn_configs.use_mla

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        self.fmha_impl._get_headwise_config(layer_idx)

       
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

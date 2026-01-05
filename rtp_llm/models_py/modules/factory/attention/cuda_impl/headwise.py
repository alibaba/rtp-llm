import json
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from flashinfer.cascade import merge_state

from rtp_llm.distribute.worker_info import g_parallel_info
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHADecodeImplBase,
    FMHAPrefillImplBase,
    FMHAType,
)
from rtp_llm.ops import AttentionConfigs
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCachePrefillOp,
    KVCache,
    PyAttentionInputs,
)


@dataclass
class BatchWrapperItem:
    """包装单条序列的注意力算子状态"""

    use_headwise: bool
    # 基础 Full Attention Wrapper
    full_wrapper: BatchPrefillWithPagedKVCacheWrapper = None
    # Head-wise 模式下的额外 Wrappers
    swa_wrapper: Optional[BatchPrefillWithPagedKVCacheWrapper] = None
    sink_prefix_wrapper: Optional[BatchPrefillWithPagedKVCacheWrapper] = None  # 原 top
    sink_rest_wrapper: Optional[BatchPrefillWithPagedKVCacheWrapper] = None  # 原 sink


class HeadWisePrefillAttnOp:
    def __init__(self, attn_configs: AttentionConfigs) -> None:
        self.rank = g_parallel_info.tp_rank
        self.head_num = attn_configs.head_num
        self.head_num_kv = attn_configs.kv_head_num
        self.size_per_head = attn_configs.size_per_head
        self.dtype = torch.bfloat16
        self.paged_size = attn_configs.tokens_per_block

        # 配置常量
        self.sink_token_num = 4
        self.swa_token_num = 8192
        self.seqlen_threshold = 16384
        self.headwise_all_config = attn_configs.headwise_config

        # 预分配全局工作空间
        global_workspace_size = 256 * 1024 * 1024
        self.workspace_buffer = torch.empty(
            global_workspace_size, dtype=torch.uint8, device="cuda"
        )

        # 运行时状态
        self.retrieval_heads: Optional[torch.Tensor] = None
        self.non_retrieval_heads: Optional[torch.Tensor] = None
        self.batch_wrappers: List[BatchWrapperItem] = []
        self.input_lengths: Optional[torch.Tensor] = None

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        major, minor = map(int, torch.version.cuda.split(".")[:2])
        return (major, minor) < (12, 8)

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

    def _make_wrapper(self) -> BatchPrefillWithPagedKVCacheWrapper:
        return BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, "HND", backend="fa2"
        )

    def _get_paged_metadata(
        self, q_len: int, kv_len: int, kv_indices: torch.Tensor
    ) -> Tuple:
        """计算 Paged KV Cache 所需的元数据指针"""
        q_indptr = torch.tensor([0, q_len], dtype=torch.int32, device="cuda")
        num_pages = (kv_len + self.paged_size - 1) // self.paged_size
        kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device="cuda")
        kv_last_page_len = torch.tensor(
            [(kv_len - 1) % self.paged_size + 1], dtype=torch.int32, device="cuda"
        )
        return q_indptr, kv_indptr, kv_indices, kv_last_page_len

    def prepare(self, attn_inputs: PyAttentionInputs) -> None:
        """预处理 Batch 中每条序列的算子计划"""
        self.input_lengths = attn_inputs.input_lengths
        self.kv_lengths = attn_inputs.prefix_lengths
        self.batch_wrappers = []

        for i, length_tensor in enumerate(self.input_lengths):
            q_len = int(length_tensor.item())
            # 是否命中 KV Cache
            kv_len = self.kv_lengths[i].item() if self.kv_lengths[i] > 0 else q_len
            kv_indices = attn_inputs.kv_cache_block_id_device[i]

            # 基础 Full Attention 规划
            full_wrapper = self._make_wrapper()
            meta = self._get_paged_metadata(q_len, kv_len, kv_indices)

            full_wrapper.plan(
                *meta,
                self.head_num,
                self.head_num_kv,
                self.size_per_head,
                self.paged_size,
                causal=True,
                q_data_type=self.dtype
            )

            if kv_len >= self.seqlen_threshold:
                """
                这里分两种情况：
                1. q_len < self.swa_token_num - self.sink_token
                2. q_len > self.swa_token_num - self.sink_token
                """
                if q_len != kv_len and q_len < self.swa_token_num - self.sink_token_num:
                    item = BatchWrapperItem(
                        use_headwise=True,
                        full_wrapper=full_wrapper,
                        swa_wrapper=self._make_wrapper(),
                        sink_rest_wrapper=self._make_wrapper(),
                    )
                    sink_meta = self._get_paged_metadata(
                        q_len, self.sink_token_num, kv_indices[0:1]
                    )
                    item.sink_rest_wrapper.plan(
                        *sink_meta,
                        self.head_num,
                        self.head_num_kv,
                        self.size_per_head,
                        self.paged_size,
                        causal=False,
                        q_data_type=self.dtype
                    )
                    # 3. SWA (Sliding Window Attention)
                    item.swa_wrapper.plan(
                        *meta,
                        self.head_num,
                        self.head_num_kv,
                        self.size_per_head,
                        self.paged_size,
                        causal=True,
                        window_left=self.swa_token_num,
                        q_data_type=self.dtype
                    )
                else:
                    item = BatchWrapperItem(
                        use_headwise=True,
                        full_wrapper=full_wrapper,
                        swa_wrapper=self._make_wrapper(),
                        sink_prefix_wrapper=self._make_wrapper(),
                        sink_rest_wrapper=self._make_wrapper(),
                    )
                    # 1. Sink Prefix (前 4 个 token)
                    sink_meta = self._get_paged_metadata(
                        self.sink_token_num, self.sink_token_num, kv_indices[0:1]
                    )
                    item.sink_prefix_wrapper.plan(
                        *sink_meta,
                        self.head_num,
                        self.head_num_kv,
                        self.size_per_head,
                        self.paged_size,
                        causal=True,
                        q_data_type=self.dtype
                    )

                    # 2. Sink Rest (针对滑动窗口内非 prefix 部分的补丁)
                    rest_len = self.swa_token_num - self.sink_token_num
                    rest_q_meta = (
                        torch.tensor([0, rest_len], dtype=torch.int32, device="cuda"),
                    ) + sink_meta[1:]
                    item.sink_rest_wrapper.plan(
                        *rest_q_meta,
                        self.head_num,
                        self.head_num_kv,
                        self.size_per_head,
                        self.paged_size,
                        causal=False,
                        q_data_type=self.dtype
                    )

                    # 3. SWA (Sliding Window Attention)
                    item.swa_wrapper.plan(
                        *meta,
                        self.head_num,
                        self.head_num_kv,
                        self.size_per_head,
                        self.paged_size,
                        causal=True,
                        window_left=self.swa_token_num,
                        q_data_type=self.dtype
                    )
            else:
                item = BatchWrapperItem(use_headwise=False, full_wrapper=full_wrapper)

            self.batch_wrappers.append(item)

    def _apply_headwise_logic(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        wrapper: BatchWrapperItem,
    ) -> torch.Tensor:
        """执行具体的注意力计算逻辑"""
        seq_len = q.size(0)
        out = torch.empty(
            (seq_len, self.head_num, self.size_per_head), dtype=q.dtype, device=q.device
        )

        # 1. 处理全量注意力头 (Retrieval Heads)
        if self.retrieval_heads.any():
            out[:, self.retrieval_heads, :] = wrapper.full_wrapper.forward(
                q[:, self.retrieval_heads, :], (k_cache, v_cache), causal=True
            )

        # 2. 处理压缩注意力头 (Non-Retrieval Heads: Sink + SWA)
        if self.non_retrieval_heads.any():
            if wrapper.sink_prefix_wrapper is None:
                h_idx = self.non_retrieval_heads
                sink_n = self.sink_token_num
                swa_n = self.swa_token_num

                # A. 计算 SWA 部分
                o_swa, lse_swa = wrapper.swa_wrapper.forward_return_lse(
                    q[:, h_idx, :], (k_cache, v_cache), causal=True
                )

                o_sink, lse_sink = wrapper.sink_rest_wrapper.forward_return_lse(
                    q[:, h_idx, :], (k_cache, v_cache), causal=False
                )

                o_swa, lse_swa = merge_state(o_sink, lse_sink, o_swa, lse_swa)

                out[:, h_idx, :] = o_swa
            else:
                h_idx = self.non_retrieval_heads
                sink_n = self.sink_token_num
                swa_n = self.swa_token_num

                # A. 计算 SWA 部分
                o_swa, lse_swa = wrapper.swa_wrapper.forward_return_lse(
                    q[:, h_idx, :],
                    (k_cache, v_cache),
                    causal=True,
                    window_left=self.swa_token_num,
                )

                # B. 计算 Sink (Prefix) 部分 - 对应 SWA 窗口起始位置
                # 注：这里逻辑需根据实际模型需求确认 q 的切片位置
                q_sink_prefix = q[seq_len - swa_n : seq_len - swa_n + sink_n, h_idx, :]
                o_prefix, lse_prefix = wrapper.sink_prefix_wrapper.forward_return_lse(
                    q_sink_prefix, (k_cache, v_cache), causal=True
                )

                # C. 计算 Sink (Rest) 部分
                q_sink_rest = q[seq_len - swa_n + sink_n :, h_idx, :]
                o_rest, lse_rest = wrapper.sink_rest_wrapper.forward_return_lse(
                    q_sink_rest, (k_cache, v_cache), causal=False
                )

                # 合并 Sink 结果
                o_sink_total = torch.cat([o_prefix, o_rest], dim=0)
                lse_sink_total = torch.cat([lse_prefix, lse_rest], dim=0)

                # 将 Sink 结果合并进 SWA 的最后窗口部分
                o_swa[-swa_n:], lse_swa[-swa_n:] = merge_state(
                    o_sink_total, lse_sink_total, o_swa[-swa_n:], lse_swa[-swa_n:]
                )

                out[:, h_idx, :] = o_swa

        return out

    def forward(
        self, fmha_input: torch.Tensor, kv_cache: Any, fmha_params: Any
    ) -> torch.Tensor:

        total_tokens = fmha_input.shape[0]
        output = torch.empty(
            (total_tokens, self.head_num, self.size_per_head),
            dtype=fmha_input.dtype,
            device=fmha_input.device,
        )

        # 提取底层的 Cache Tensor
        k_cache = kv_cache.k_cache_base[:, 0, ...]
        v_cache = kv_cache.k_cache_base[:, 1, ...]

        offset = 0
        for i, wrapper in enumerate(self.batch_wrappers):
            length = int(self.input_lengths[i])
            # 切片并 View 为 [L, H, D]
            qkv = fmha_input[offset : offset + length].view(
                length, -1, self.size_per_head
            )
            q, _, _ = torch.split(
                qkv, [self.head_num, self.head_num_kv, self.head_num_kv], dim=1
            )

            if wrapper.use_headwise:
                res = self._apply_headwise_logic(q, k_cache, v_cache, wrapper)
            else:
                res = wrapper.full_wrapper.forward(q, (k_cache, v_cache), causal=True)

            output[offset : offset + length] = res
            offset += length

        return output.view(total_tokens, -1)


class HeadWisePrefillImpl(FMHAPrefillImplBase):
    def __init__(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            HeadWisePrefillAttnOp(attn_configs),
            FusedRopeKVCachePrefillOp(attn_configs),
            attn_inputs,
        )

    def support(self) -> bool:
        return True

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.HEADWISE

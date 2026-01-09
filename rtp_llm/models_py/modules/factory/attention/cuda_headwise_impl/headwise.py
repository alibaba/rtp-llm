from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper
from flashinfer.cascade import merge_state

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
    """每条序列对应的 wrapper 组合"""

    use_headwise: bool
    full_wrapper: BatchPrefillWithPagedKVCacheWrapper

    swa_wrapper: Optional[BatchPrefillWithPagedKVCacheWrapper] = None
    sink_prefix_wrapper: Optional[BatchPrefillWithPagedKVCacheWrapper] = None
    sink_rest_wrapper: Optional[BatchPrefillWithPagedKVCacheWrapper] = None

    @property
    def has_sink_prefix(self) -> bool:
        return self.sink_prefix_wrapper is not None


# ----------------------------
# Main Operator
# ----------------------------
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

        # 这里保持你原来固定 dtype
        self.dtype = torch.bfloat16

        self.headwise_all_config = attn_configs.headwise_config
        self.hw_cfg = HeadWiseRuntimeConfig(
            sink_token_num=self.headwise_all_config.get("sink_token_num", 4),
            swa_token_num=self.headwise_all_config.get("swa_token_num", 8192),
            seqlen_threshold=self.headwise_all_config.get("seqlen_threshold", 16384),
        )

        self.workspace_buffer = self._alloc_workspace(256 * 1024 * 1024)

        # runtime states（prepare 时生成）
        self.retrieval_heads: Optional[torch.Tensor] = None
        self.non_retrieval_heads: Optional[torch.Tensor] = None
        self.batch_wrappers: List[BatchWrapperItem] = []
        self.input_lengths: Optional[torch.Tensor] = None
        self.kv_lengths: Optional[torch.Tensor] = None

    # ----------------------------
    # Utilities
    # ----------------------------
    @staticmethod
    def _alloc_workspace(nbytes: int) -> torch.Tensor:
        return torch.empty(nbytes, dtype=torch.uint8, device="cuda")

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        major, minor = map(int, torch.version.cuda.split(".")[:2])
        return (major, minor) < (12, 8)

    def _make_wrapper(self) -> BatchPrefillWithPagedKVCacheWrapper:
        return BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, "HND", backend="fa2"
        )

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

    # ----------------------------
    # Planning
    # ----------------------------
    def prepare(self, attn_inputs: PyAttentionInputs) -> None:
        self.input_lengths = attn_inputs.input_lengths
        self.kv_lengths = attn_inputs.prefix_lengths
        self.batch_wrappers = []

        for i, length_tensor in enumerate(self.input_lengths):
            q_len = int(length_tensor.item())
            kv_len = max(q_len, int(self.kv_lengths[i].item()))
            kv_indices = attn_inputs.kv_cache_block_id_device[i]

            item = self._plan_one_sequence(
                q_len=q_len, kv_len=kv_len, kv_indices=kv_indices
            )
            self.batch_wrappers.append(item)

    def _plan_one_sequence(
        self, q_len: int, kv_len: int, kv_indices: torch.Tensor
    ) -> BatchWrapperItem:
        """为单条序列构建 wrappers 并 plan。"""
        meta = self._get_paged_metadata(q_len, kv_len, kv_indices)

        full = self._make_wrapper()
        full.plan(
            *meta,
            self.head_num,
            self.head_num_kv,
            self.size_per_head,
            self.paged_size,
            causal=True,
            q_data_type=self.dtype,
        )

        # 不启用 headwise：直接 full attention
        if kv_len < self.hw_cfg.seqlen_threshold:
            return BatchWrapperItem(use_headwise=False, full_wrapper=full)

        # 启用 headwise
        # 情况 A：q_len != kv_len 且 q_len < kv_len - swa - sink（你原代码的分支）
        if (
            q_len != kv_len
            and q_len < kv_len - self.hw_cfg.swa_token_num - self.hw_cfg.sink_token_num
        ):
            return self._plan_headwise_case_a(full, q_len, kv_len, kv_indices, meta)

        # 情况 B：默认分支（带 sink_prefix + sink_rest + swa）
        return self._plan_headwise_case_b(full, q_len, kv_len, kv_indices, meta)

    def _plan_headwise_case_a(
        self,
        full: BatchPrefillWithPagedKVCacheWrapper,
        q_len: int,
        kv_len: int,
        kv_indices: torch.Tensor,
        meta: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> BatchWrapperItem:
        item = BatchWrapperItem(
            use_headwise=True,
            full_wrapper=full,
            swa_wrapper=self._make_wrapper(),
            sink_rest_wrapper=self._make_wrapper(),
        )

        # sink_rest：只对 sink_token_num 做补充（注意你原代码取 kv_indices[0:1]）
        sink_meta = self._get_paged_metadata(
            q_len, self.hw_cfg.sink_token_num, kv_indices[0:1]
        )
        item.sink_rest_wrapper.plan(
            *sink_meta,
            self.head_num,
            self.head_num_kv,
            self.size_per_head,
            self.paged_size,
            causal=False,
            q_data_type=self.dtype,
        )

        item.swa_wrapper.plan(
            *meta,
            self.head_num,
            self.head_num_kv,
            self.size_per_head,
            self.paged_size,
            causal=True,
            window_left=self.hw_cfg.swa_token_num,
            q_data_type=self.dtype,
        )
        return item

    def _plan_headwise_case_b(
        self,
        full: BatchPrefillWithPagedKVCacheWrapper,
        q_len: int,
        kv_len: int,
        kv_indices: torch.Tensor,
        meta: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> BatchWrapperItem:
        item = BatchWrapperItem(
            use_headwise=True,
            full_wrapper=full,
            swa_wrapper=self._make_wrapper(),
            sink_prefix_wrapper=self._make_wrapper(),
            sink_rest_wrapper=self._make_wrapper(),
        )

        # sink_prefix：前 sink_token_num token
        sink_meta = self._get_paged_metadata(
            self.hw_cfg.sink_token_num, self.hw_cfg.sink_token_num, kv_indices[0:1]
        )
        item.sink_prefix_wrapper.plan(
            *sink_meta,
            self.head_num,
            self.head_num_kv,
            self.size_per_head,
            self.paged_size,
            causal=True,
            q_data_type=self.dtype,
        )

        # sink_rest：剩余需要补的 q 段（你原代码的 rest_q_meta 构造方式）
        rest_len = kv_len - self.hw_cfg.swa_token_num - self.hw_cfg.sink_token_num
        rest_q_indptr = torch.tensor([0, rest_len], dtype=torch.int32, device="cuda")
        rest_q_meta = (rest_q_indptr,) + sink_meta[1:]

        item.sink_rest_wrapper.plan(
            *rest_q_meta,
            self.head_num,
            self.head_num_kv,
            self.size_per_head,
            self.paged_size,
            causal=False,
            q_data_type=self.dtype,
        )

        item.swa_wrapper.plan(
            *meta,
            self.head_num,
            self.head_num_kv,
            self.size_per_head,
            self.paged_size,
            causal=True,
            window_left=self.hw_cfg.swa_token_num,
            q_data_type=self.dtype,
        )
        return item

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

        # cache: [pages, 2, ...]
        k_cache = kv_cache.k_cache_base[:, 0, ...]
        v_cache = kv_cache.k_cache_base[:, 1, ...]

        offset = 0
        for i, wrapper in enumerate(self.batch_wrappers):
            q_len = int(self.input_lengths[i].item())
            kv_len = max(q_len, int(self.kv_lengths[i].item()))

            q = self._slice_q(fmha_input, offset, q_len)

            if wrapper.use_headwise:
                res = self._apply_headwise(
                    q, k_cache, v_cache, wrapper, q_len=q_len, kv_len=kv_len
                )
            else:
                res = wrapper.full_wrapper.forward(q, (k_cache, v_cache), causal=True)

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
        wrapper: BatchWrapperItem,
        q_len: int,
        kv_len: int,
    ) -> torch.Tensor:
        out = torch.empty(
            (q_len, self.head_num, self.size_per_head), dtype=q.dtype, device=q.device
        )

        # 1) retrieval heads: full attention
        if self.retrieval_heads is not None and self.retrieval_heads.any():
            out[:, self.retrieval_heads, :] = wrapper.full_wrapper.forward(
                q[:, self.retrieval_heads, :], (k_cache, v_cache), causal=True
            )

        # 2) non-retrieval heads: sink + swa
        if self.non_retrieval_heads is not None and self.non_retrieval_heads.any():
            h = self.non_retrieval_heads
            out[:, h, :] = self._run_non_retrieval(
                q[:, h, :], k_cache, v_cache, wrapper, q_len=q_len, kv_len=kv_len
            )

        return out

    def _run_non_retrieval(
        self,
        q_h: torch.Tensor,  # [q_len, Hn, D]
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        wrapper: BatchWrapperItem,
        q_len: int,
        kv_len: int,
    ) -> torch.Tensor:
        # SWA 主体
        o_swa, lse_swa = wrapper.swa_wrapper.forward_return_lse(
            q_h, (k_cache, v_cache), causal=True, window_left=self.hw_cfg.swa_token_num
        )

        # case A：没有 sink_prefix_wrapper，直接把 sink_rest 的结果 merge 到 swa
        if not wrapper.has_sink_prefix:
            o_sink, lse_sink = wrapper.sink_rest_wrapper.forward_return_lse(
                q_h, (k_cache, v_cache), causal=False
            )
            o, _ = merge_state(o_sink, lse_sink, o_swa, lse_swa)
            return o

        # case B：有 sink_prefix + sink_rest：只对最后窗口部分做 patch merge（沿用你原切片公式）
        sink_n = self.hw_cfg.sink_token_num
        swa_n = self.hw_cfg.swa_token_num
        start = q_len - kv_len + swa_n

        q_prefix = q_h[start : start + sink_n]
        q_rest = q_h[start + sink_n :]

        o_prefix, lse_prefix = wrapper.sink_prefix_wrapper.forward_return_lse(
            q_prefix, (k_cache, v_cache), causal=True
        )
        o_rest, lse_rest = wrapper.sink_rest_wrapper.forward_return_lse(
            q_rest, (k_cache, v_cache), causal=False
        )

        o_sink_total = torch.cat([o_prefix, o_rest], dim=0)
        lse_sink_total = torch.cat([lse_prefix, lse_rest], dim=0)

        patched_o, _ = merge_state(
            o_sink_total, lse_sink_total, o_swa[start:], lse_swa[start:]
        )
        o_swa[start:] = patched_o
        return o_swa


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

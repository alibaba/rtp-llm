from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper
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
    """每条序列对应的 wrapper 组合，支持动态选择不同数量的 Head"""

    use_headwise: bool

    # 列表索引 i 对应 num_qo_heads = i + 1
    full_wrappers: List[Any] = field(default_factory=list)
    swa_wrappers: List[Any] = field(default_factory=list)
    sink_prefix_wrappers: List[Any] = field(default_factory=list)
    sink_rest_wrappers: List[Any] = field(default_factory=list)

    def get_full_wrapper(self, num_heads: int):
        return self.full_wrappers[num_heads - 1]

    def get_swa_wrapper(self, num_heads: int):
        return self.swa_wrappers[num_heads - 1]

    def get_sink_prefix_wrapper(self, num_heads: int):
        return self.sink_prefix_wrappers[num_heads - 1]

    def get_sink_rest_wrapper(self, num_heads: int):
        return self.sink_rest_wrappers[num_heads - 1]

    @property
    def has_sink_prefix(self) -> bool:
        return len(self.sink_prefix_wrappers) > 0


# 模块级全局变量，初始设置为 None
global_workspace_buffer = None


class HeadWisePrefillAttnOp:
    """
    HeadWise Prefill Attention:
      - retrieval heads: full attention
      - non-retrieval heads: sink + sliding window attention, 通过 merge_state 合并
    """

    def __init__(
        self, attn_configs: AttentionConfigs, parallelism_config: ParallelismConfig
    ) -> None:
        global global_workspace_buffer  # 声明使用全局变量

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

        # 检查并初始化全局 workspace buffer
        if global_workspace_buffer is None:
            logging.info("Initializing global workspace buffer")
            global_workspace_buffer = self._alloc_workspace(512 * 1024 * 1024)
        else:
            logging.info("Using existing global workspace buffer")

        # 使用全局的 workspace buffer
        self.workspace_buffer = global_workspace_buffer

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
            self.workspace_buffer, "HND", backend="auto"
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

    def _plan_wrappers_for_range(
        self,
        meta: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        causal: bool,
        window_left: int = -1,
    ) -> List[Any]:
        """为 1 到 self.head_num 个 head 分别执行 plan"""
        wrappers = []
        # 计算 GQA 组大小
        group_size = self.head_num // self.head_num_kv

        for h in range(1, self.head_num + 1):
            wrapper = self._make_wrapper()
            # 动态计算对应的 KV head 数量
            # 保证至少 1 个 KV head，且随 Q head 数量按比例增加
            h_kv = max(1, (h + group_size - 1) // group_size)

            wrapper.plan(
                *meta,
                h,  # num_qo_heads
                h_kv,  # num_kv_heads
                self.size_per_head,
                self.paged_size,
                causal=causal,
                window_left=window_left,
                q_data_type=self.dtype,
            )
            wrappers.append(wrapper)
        return wrappers

    # ----------------------------
    # Planning
    # ----------------------------
    def prepare(self, attn_inputs: PyAttentionInputs) -> None:
        self.input_lengths = attn_inputs.input_lengths
        self.kv_lengths = attn_inputs.prefix_lengths
        self.batch_wrappers = []

        for i, length_tensor in enumerate(self.input_lengths):
            q_len = int(length_tensor.item())
            kv_len = int(self.kv_lengths[i].item()) if self.kv_lengths[i] > 0 else q_len
            kv_indices = attn_inputs.kv_cache_block_id_device[i]

            item = self._plan_one_sequence(
                q_len=q_len, kv_len=kv_len, kv_indices=kv_indices
            )
            self.batch_wrappers.append(item)

    def _plan_one_sequence(
        self, q_len: int, kv_len: int, kv_indices: torch.Tensor
    ) -> BatchWrapperItem:
        """为单条序列构建一系列 wrappers 并 plan。"""
        meta = self._get_paged_metadata(q_len, kv_len, kv_indices)

        # 无论如何，都需要准备全量 Attention 的 wrappers (1 ~ head_num)
        full_wrappers = self._plan_wrappers_for_range(meta, causal=True)

        # 长度未达阈值：不启用 headwise 分离
        if kv_len < self.hw_cfg.seqlen_threshold:
            return BatchWrapperItem(use_headwise=False, full_wrappers=full_wrappers)

        # 长度超过阈值：根据 Q/KV 关系选择 A 方案或 B 方案
        if (
            q_len != kv_len
            and q_len < kv_len - self.hw_cfg.swa_token_num - self.hw_cfg.sink_token_num
        ):
            return self._plan_headwise_case_a(
                full_wrappers, q_len, kv_len, kv_indices, meta
            )

        return self._plan_headwise_case_b(
            full_wrappers, q_len, kv_len, kv_indices, meta
        )

    def _plan_headwise_case_a(
        self,
        full_wrappers: List[Any],
        q_len: int,
        kv_len: int,
        kv_indices: torch.Tensor,
        meta: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> BatchWrapperItem:
        """Case A: Q 长度较短，且处于序列末尾（非对角线块）"""
        # sink_rest: 针对 Q(all) 关注 KV(0:sink)
        sink_meta = self._get_paged_metadata(
            q_len, self.hw_cfg.sink_token_num, kv_indices[0:1]
        )

        return BatchWrapperItem(
            use_headwise=True,
            full_wrappers=full_wrappers,
            swa_wrappers=self._plan_wrappers_for_range(
                meta, causal=True, window_left=self.hw_cfg.swa_token_num
            ),
            sink_rest_wrappers=self._plan_wrappers_for_range(sink_meta, causal=False),
        )

    def _plan_headwise_case_b(
        self,
        full_wrappers: List[Any],
        q_len: int,
        kv_len: int,
        kv_indices: torch.Tensor,
        meta: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> BatchWrapperItem:
        """Case B: 默认分支（带 sink_prefix + sink_rest + swa）"""
        # sink_prefix：前 sink_token_num 个 token
        sink_prefix_meta = self._get_paged_metadata(
            self.hw_cfg.sink_token_num, self.hw_cfg.sink_token_num, kv_indices[0:1]
        )

        # sink_rest：剩余需要补的 q 段关注 sink 区域
        rest_len = kv_len - self.hw_cfg.swa_token_num - self.hw_cfg.sink_token_num
        rest_q_indptr = torch.tensor([0, rest_len], dtype=torch.int32, device="cuda")
        # 组装 meta: q_indptr 不同，kv 对应 sink 区域
        sink_rest_meta = (
            rest_q_indptr,
            sink_prefix_meta[1],
            sink_prefix_meta[2],
            sink_prefix_meta[3],
        )

        return BatchWrapperItem(
            use_headwise=True,
            full_wrappers=full_wrappers,
            swa_wrappers=self._plan_wrappers_for_range(
                meta, causal=True, window_left=self.hw_cfg.swa_token_num
            ),
            sink_prefix_wrappers=self._plan_wrappers_for_range(
                sink_prefix_meta, causal=True
            ),
            sink_rest_wrappers=self._plan_wrappers_for_range(
                sink_rest_meta, causal=False
            ),
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

        # cache: [pages, 2, ...]
        k_cache = kv_cache.kv_cache_base[:, 0, ...]
        v_cache = kv_cache.kv_cache_base[:, 1, ...]

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
                # 不启用 headwise 时，使用对应全量 head_num 的 wrapper
                full_wrapper = wrapper_item.get_full_wrapper(self.head_num)
                res = full_wrapper.forward(q, (k_cache, v_cache), causal=True)

            output[offset : offset + q_len] = res
            offset += q_len

        return output.view(total_tokens, -1)

    def _slice_q(
        self, fmha_input: torch.Tensor, offset: int, q_len: int
    ) -> torch.Tensor:
        qkv = fmha_input[offset : offset + q_len].view(q_len, -1, self.size_per_head)
        # 假设 fmha_input 的布局是 [q, k, v]
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

        # 1) retrieval heads: 使用全量 attention
        if self.retrieval_heads is not None:
            # 计算 retrieval head 的数量
            n_ret = int(self.retrieval_heads.sum().item())
            if n_ret > 0:
                # 获取 plan 好的对应 n_ret 个 head 的 wrapper
                full_wrapper = wrapper_item.get_full_wrapper(n_ret)
                out[:, self.retrieval_heads, :] = full_wrapper.forward(
                    q[:, self.retrieval_heads, :], (k_cache, v_cache), causal=True
                )

        # 2) non-retrieval heads: 使用 sink + swa
        if self.non_retrieval_heads is not None:
            # 计算 non-retrieval head 的数量
            n_non_ret = int(self.non_retrieval_heads.sum().item())
            if n_non_ret > 0:
                h_mask = self.non_retrieval_heads
                out[:, h_mask, :] = self._run_non_retrieval(
                    q[:, h_mask, :],
                    k_cache,
                    v_cache,
                    wrapper_item,
                    q_len=q_len,
                    kv_len=kv_len,
                    num_heads=n_non_ret,
                )

        return out

    def _run_non_retrieval(
        self,
        q_h: torch.Tensor,  # [q_len, num_heads, D]
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        wrapper_item: BatchWrapperItem,
        q_len: int,
        kv_len: int,
        num_heads: int,
    ) -> torch.Tensor:
        from flashinfer.cascade import merge_state

        # 获取对应 head 数量的 SWA wrapper
        swa_wrapper = wrapper_item.get_swa_wrapper(num_heads)

        # SWA 主体计算
        o_swa, lse_swa = swa_wrapper.forward_return_lse(
            q_h, (k_cache, v_cache), causal=True
        )

        # Case A：只有 sink_rest (Q 比较短，直接补 sink)
        if not wrapper_item.has_sink_prefix:
            sink_rest_wrapper = wrapper_item.get_sink_rest_wrapper(num_heads)
            o_sink, lse_sink = sink_rest_wrapper.forward_return_lse(
                q_h, (k_cache, v_cache), causal=False
            )
            o, _ = merge_state(o_sink, lse_sink, o_swa, lse_swa)
            return o

        # Case B：有 sink_prefix + sink_rest (Q 覆盖了对角线块和非对角线块)
        sink_n = self.hw_cfg.sink_token_num
        swa_n = self.hw_cfg.swa_token_num
        # 这里的 start 指的是 Q 中需要进行 Patch Merge 的起始位置
        # 即 Q 中那些其对应的 KV 范围已经超出了 SWA 窗口的 token
        start = q_len - kv_len + swa_n

        # 此时 q_h 的 head 维度已经是过滤后的 num_heads
        q_prefix = q_h[start : start + sink_n]
        q_rest = q_h[start + sink_n :]

        # 获取对应的 wrappers
        prefix_wrapper = wrapper_item.get_sink_prefix_wrapper(num_heads)
        rest_wrapper = wrapper_item.get_sink_rest_wrapper(num_heads)

        # 计算 Sink 部分
        o_prefix, lse_prefix = prefix_wrapper.forward_return_lse(
            q_prefix, (k_cache, v_cache), causal=True
        )
        o_rest, lse_rest = rest_wrapper.forward_return_lse(
            q_rest, (k_cache, v_cache), causal=False
        )

        o_sink_total = torch.cat([o_prefix, o_rest], dim=0)
        lse_sink_total = torch.cat([lse_prefix, lse_rest], dim=0)

        # 合并 SWA 和 Sink 的结果
        # 注意：只有从 start 开始的 Q 才会丢失旧的 Sink 信息，需要合并

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

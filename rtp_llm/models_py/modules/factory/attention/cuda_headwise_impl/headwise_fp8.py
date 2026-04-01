from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

_HAS_FLASH_ATTN_3 = False
_HAS_RTP_KERNEL_FP8 = False

try:
    from flash_attn_interface import flash_attn_varlen_func
    _HAS_FLASH_ATTN_3 = True
except ImportError as e:
    logging.warning(f"flash_attn_interface not found: {e}")

try:
    from rtp_kernel.sparse_attention_fp8 import BatchPrefillWithSparseAttentionFP8
    from rtp_kernel.kvcache import kvcache_extract_prefill
    _HAS_RTP_KERNEL_FP8 = True
except ImportError as e:
    logging.warning(f"rtp_kernel FP8 modules not found: {e}")

try:
    from rtp_llm.models_py.modules.factory.attention import common
    from rtp_llm.ops.compute_ops import (
        FusedRopeKVCachePrefillOpQKVOut,
        PyAttentionInputs,
    )
except ImportError as e:
    logging.warning(f"rtp_llm attention common/compute_ops not found: {e}")

from rtp_llm.models_py.modules.factory.attention.cuda_headwise_impl.headwise import (
    HeadWiseRuntimeConfig,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, ParallelismConfig


def _headwise_prefill_fp8_runtime_ready(attn_inputs) -> bool:
    """Check flash_attn_3 + rtp_kernel_fp8 + CUDA >= 12.8 + headwise_config on attn_inputs."""
    if not (_HAS_FLASH_ATTN_3 and _HAS_RTP_KERNEL_FP8):
        return False
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = map(int, torch.version.cuda.split(".")[:2])
    except (AttributeError, TypeError, ValueError):
        return False
    return (major, minor) >= (12, 8) and getattr(attn_inputs, 'headwise_config', None) is not None


@dataclass
class FP8BatchItem:
    use_headwise: bool
    q_len: int = 0
    kv_len: int = 0
    kv_indices: Optional[torch.Tensor] = None


class HeadWiseFP8PrefillAttnOp:
    """
    HeadWise FP8 Prefill Attention (all-FP8 inputs):
      - retrieval heads: full attention via flash_attn_varlen_func (Q/K/V all FP8)
      - non-retrieval heads: sparse attention via BatchPrefillWithSparseAttentionFP8 (Q/K/V all FP8)
      - short sequences: full attention via flash_attn_varlen_func for all heads (Q/K/V all FP8)
    """

    def __init__(
        self, attn_configs: AttentionConfigs, parallelism_config: ParallelismConfig,
        headwise_config: Optional[dict] = None,
    ) -> None:
        self.rank = parallelism_config.tp_rank

        self.head_num = attn_configs.head_num
        self.head_num_kv = attn_configs.kv_head_num
        self.size_per_head = attn_configs.size_per_head
        self.paged_size = attn_configs.tokens_per_block

        if headwise_config is not None:
            self.headwise_all_config = headwise_config

            self.hw_cfg = HeadWiseRuntimeConfig(
                sink_token_num=self.headwise_all_config.get("sink_token_num", 4),
                swa_token_num=self.headwise_all_config.get("swa_token_num", 8192),
                seqlen_threshold=self.headwise_all_config.get(
                    "seqlen_threshold", 16384
                ),
            )

        self.sparse_fp8 = BatchPrefillWithSparseAttentionFP8()

        self.retrieval_heads: Optional[torch.Tensor] = None
        self.non_retrieval_heads: Optional[torch.Tensor] = None
        self.batch_items: List[FP8BatchItem] = []
        self.input_lengths: Optional[torch.Tensor] = None
        self.kv_lengths: Optional[torch.Tensor] = None
        self._dtype_logged = False

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        if not (_HAS_FLASH_ATTN_3 and _HAS_RTP_KERNEL_FP8):
            return False
        major, minor = map(int, torch.version.cuda.split(".")[:2])
        return (major, minor) >= (12, 8) and getattr(attn_inputs, 'headwise_config', None) is not None

    def _get_headwise_config(self, layer_idx: int):
        layer_key = str(layer_idx)
        if layer_key not in self.headwise_all_config:
            logging.warning(
                f"[HeadWiseFP8] layer_idx={layer_idx} not found in headwise_config, "
                f"falling back to all-retrieval heads"
            )
            self.retrieval_heads = torch.ones(self.head_num, dtype=torch.bool, device="cuda")
            self.non_retrieval_heads = torch.zeros(self.head_num, dtype=torch.bool, device="cuda")
            return

        start = self.head_num * self.rank
        end = start + self.head_num

        layer_config = torch.tensor(
            self.headwise_all_config[layer_key], device="cuda"
        )
        current_rank_weights = layer_config[start:end]

        self.non_retrieval_heads = current_rank_weights == 0
        self.retrieval_heads = current_rank_weights == 1

    def prepare(self, attn_inputs: PyAttentionInputs) -> None:
        self.input_lengths = attn_inputs.input_lengths
        self.kv_lengths = attn_inputs.input_lengths + attn_inputs.prefix_lengths
        self.kv_indices = attn_inputs.kv_cache_block_id_device

        self.batch_items = []

        input_lens_list = self.input_lengths.cpu().tolist()
        kv_lens_list = self.kv_lengths.cpu().tolist()

        for i, q_len in enumerate(input_lens_list):
            kv_len = kv_lens_list[i] if kv_lens_list[i] > 0 else q_len

            use_headwise = (
                kv_len >= self.hw_cfg.seqlen_threshold
                and q_len >= self.hw_cfg.sink_token_num
                and q_len == kv_len
            )
            self.batch_items.append(
                FP8BatchItem(
                    use_headwise=use_headwise,
                    q_len=q_len,
                    kv_len=kv_len,
                    kv_indices=self.kv_indices[i],
                )
            )

    def forward(
        self, fmha_input: torch.Tensor, kv_cache: Any, fmha_params: Any
    ) -> torch.Tensor:
        total_tokens = fmha_input.shape[0]
        output = torch.empty(
            (total_tokens, self.head_num, self.size_per_head),
            dtype=fmha_input.dtype,
            device=fmha_input.device,
        )

        kv_base = kv_cache.kv_cache_base
        block_num = kv_base.shape[0]
        kv_expanded = kv_base.view(
            block_num, 2, self.head_num_kv, self.paged_size, self.size_per_head
        )
        # HND: [block_num, kv_head_num, page_size, head_dim] — FP8
        k_slice = kv_expanded[:, 0, ...]
        v_slice = kv_expanded[:, 1, ...]
        k_cache_hnd = k_slice if k_slice.is_contiguous() else k_slice.contiguous()
        v_cache_hnd = v_slice if v_slice.is_contiguous() else v_slice.contiguous()

        offset = 0
        for i, item in enumerate(self.batch_items):
            q_len = item.q_len
            q = self._slice_q(fmha_input, offset, q_len)

            if item.use_headwise:
                res = self._apply_headwise(
                    q, k_cache_hnd, v_cache_hnd,
                    item, q_len=q_len, kv_len=item.kv_len,
                )
            else:
                res = self._run_full_attention(
                    q, k_cache_hnd, v_cache_hnd, item,
                )

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

    def _extract_kv_fp8(
        self,
        k_cache_hnd: torch.Tensor,
        v_cache_hnd: torch.Tensor,
        item: FP8BatchItem,
    ):
        """Extract KV from paged FP8 cache, keeping FP8 dtype."""
        device = k_cache_hnd.device
        fp8_dtype = k_cache_hnd.dtype
        seq_lens = torch.tensor([item.kv_len], dtype=torch.int32, device=device)
        block_tables = item.kv_indices.unsqueeze(0).to(torch.int32)

        k_ext = torch.zeros(
            item.kv_len, self.head_num_kv, self.size_per_head,
            dtype=fp8_dtype, device=device,
        )
        v_ext = torch.zeros(
            item.kv_len, self.head_num_kv, self.size_per_head,
            dtype=fp8_dtype, device=device,
        )
        kvcache_extract_prefill(
            k_cache_hnd, v_cache_hnd, seq_lens, block_tables,
            k_ext, v_ext, self.paged_size,
        )
        return k_ext, v_ext

    # ----------------------------
    # Non-headwise: full attention for all heads (all FP8)
    # ----------------------------
    def _run_full_attention(
        self,
        q: torch.Tensor,
        k_cache_hnd: torch.Tensor,
        v_cache_hnd: torch.Tensor,
        item: FP8BatchItem,
    ) -> torch.Tensor:
        k_ext, v_ext = self._extract_kv_fp8(k_cache_hnd, v_cache_hnd, item)
        q_fp8 = q.to(k_ext.dtype)

        cu_q = torch.tensor([0, item.q_len], dtype=torch.int32, device=q.device)
        cu_k = torch.tensor([0, item.kv_len], dtype=torch.int32, device=q.device)

        out = flash_attn_varlen_func(
            q_fp8, k_ext, v_ext,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=item.q_len,
            max_seqlen_k=item.kv_len,
            causal=True,
        )
        if not self._dtype_logged:
            logging.info(
                f"[HeadWiseFP8] full_attn path: kv_cache={k_cache_hnd.dtype}, "
                f"kv_ext={k_ext.dtype}, q_orig={q.dtype}, q_fp8={q_fp8.dtype}, "
                f"out={out.dtype}, q_len={item.q_len}, kv_len={item.kv_len}"
            )
            self._dtype_logged = True
        return out

    # ----------------------------
    # Headwise: retrieval=full, non-retrieval=sparse (all FP8)
    # ----------------------------
    def _apply_headwise(
        self,
        q: torch.Tensor,
        k_cache_hnd: torch.Tensor,
        v_cache_hnd: torch.Tensor,
        item: FP8BatchItem,
        q_len: int,
        kv_len: int,
    ) -> torch.Tensor:
        out = torch.empty(
            (q_len, self.head_num, self.size_per_head), dtype=q.dtype, device=q.device
        )

        # Extract KV once (FP8), reuse for both retrieval and non-retrieval
        k_ext, v_ext = self._extract_kv_fp8(k_cache_hnd, v_cache_hnd, item)
        ret_out = None
        sparse_out = None

        if self.retrieval_heads is not None and self.retrieval_heads.any():
            retrieval_q = q[:, self.retrieval_heads, :].to(k_ext.dtype)

            cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=q.device)
            cu_k = torch.tensor([0, kv_len], dtype=torch.int32, device=q.device)

            ret_out = flash_attn_varlen_func(
                retrieval_q, k_ext, v_ext,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=q_len,
                max_seqlen_k=kv_len,
                causal=True,
            )
            out[:, self.retrieval_heads, :] = ret_out

        if self.non_retrieval_heads is not None and self.non_retrieval_heads.any():
            h = self.non_retrieval_heads
            non_ret_q = q[:, h, :].to(k_ext.dtype)
            seq_lens = torch.tensor([kv_len], dtype=torch.int32, device=q.device)

            sparse_out = self.sparse_fp8.forward(
                q=non_ret_q,
                k=k_ext,
                v=v_ext,
                seq_lens=seq_lens,
                num_qo_heads=non_ret_q.shape[1],
                num_kv_heads=self.head_num_kv,
                head_dim=self.size_per_head,
                head_split=self.hw_cfg.sink_token_num,
                window_left=self.hw_cfg.swa_token_num,
            )
            out[:, h, :] = sparse_out

        if not self._dtype_logged:
            _ret_dtype = ret_out.dtype if ret_out is not None else "N/A"
            _sp_dtype = sparse_out.dtype if sparse_out is not None else "N/A"
            logging.info(
                f"[HeadWiseFP8] headwise path: kv_cache={k_cache_hnd.dtype}, "
                f"kv_ext={k_ext.dtype}, q_orig={q.dtype}, "
                f"q_to_cache={k_ext.dtype}, "
                f"ret_out={_ret_dtype}, sparse_out={_sp_dtype}, "
                f"out={out.dtype}, q_len={q_len}, kv_len={kv_len}"
            )
            self._dtype_logged = True

        return out


class HeadWiseFP8PrefillImpl(FMHAImplBase):

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        headwise_config = getattr(attn_inputs, 'headwise_config', None)
        self.fmha_impl = HeadWiseFP8PrefillAttnOp(attn_configs, parallelism_config, headwise_config)
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQKVOut(attn_configs)
        self.attn_configs = attn_configs

        self.attn_inputs = attn_inputs

        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        if attn_configs.use_mla or attn_configs.kv_cache_dtype != KvCacheDataType.FP8:
            return False
        return _headwise_prefill_fp8_runtime_ready(attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        self.fmha_impl._get_headwise_config(layer_idx)

        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

import functools
from typing import Optional

import flashinfer
import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.common import (
    apply_rope_and_kv_cache,
    copy_kv_cache_offset,
    create_write_cache_store_op,
    write_to_cache_store,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    KVCache,
    PyAttentionInputs,
)

# Constants
DEFAULT_WORKSPACE_SIZE_MB = (
    512  # Memory workspace size in MB, todo(Yingyi): read from config
)

# Reuse this workspace buffer across all TRTLLM MHA wrappers
g_zero_workspace_buffer = None


@functools.cache
def is_sm_100() -> bool:
    return torch.cuda.get_device_capability()[0] in [10]


class FlashInferTRTLLMParams(object):
    def __init__(
        self,
        batch_size: int,
        max_q_len: int = 0,
        max_kv_len: int = 0,
        max_seq_len: int = 0,
        seq_lens: Optional[torch.Tensor] = None,
        input_lens: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_kv_seqlens: Optional[torch.Tensor] = None,
    ):

        self.batch_size = batch_size
        self.max_q_len = max_q_len  # for prefill
        self.max_kv_len = max_kv_len  # for prefill
        self.max_seq_len = max_seq_len  # for decode
        self.seq_lens = seq_lens
        self.input_lens = input_lens
        self.block_tables = block_tables
        self.cu_seqlens = cu_seqlens
        self.cu_kv_seqlens = cu_kv_seqlens


def create_g_workspace_buffer(device: str = "cuda:0"):
    global g_zero_workspace_buffer
    if g_zero_workspace_buffer is None:
        g_zero_workspace_buffer = torch.zeros(
            DEFAULT_WORKSPACE_SIZE_MB * 1024 * 1024,
            dtype=torch.uint8,
            device=device,
        )
    return g_zero_workspace_buffer


class FlashInferTRTLLMDecodeImpl(FMHAImplBase):
    """FlashInfer TRTLLM Decode implementation."""

    def __init__(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        # Initialize TRTLLM decode operation directly
        self.attn_configs = attn_configs
        self.head_dim = attn_configs.size_per_head
        self.head_num = attn_configs.head_num
        self.scaling = self.head_dim**-0.5
        self.local_head_num = attn_configs.head_num
        self.workspace_buffer = create_g_workspace_buffer()

        # Initialize rope and cache operations
        self.rope_kvcache_impl = FusedRopeKVCacheDecodeOp(attn_configs)
        self.input_lengths = attn_inputs.input_lengths
        self.cu_seq_lens = attn_inputs.cu_seqlens
        self.fmha_params = None
        self.rope_params = None

        self.attn_inputs = attn_inputs
        self.write_cache_store_impl = create_write_cache_store_op(attn_inputs)
        
        # Prepare parameters
        self.fmha_params = self._prepare_fmha_params(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)

    def _support_impl(self, attention_inputs: PyAttentionInputs):
        """Check if TRTLLM decode is supported."""
        if not is_sm_100():
            return False
        if (
            attention_inputs.is_prefill
            and attention_inputs.input_lengths[0] < 6
            and (attention_inputs.input_lengths == attention_inputs.input_lengths[0])
            .all()
            .item()
        ):
            return True
        return not attention_inputs.is_prefill

    def _prepare_fmha_params(self, attention_inputs: PyAttentionInputs) -> FlashInferTRTLLMParams:
        """Prepare FMHA parameters for decode."""
        if not attention_inputs.is_prefill:
            sequence_lengths = torch.ones_like(
                attention_inputs.sequence_lengths,
                device="cuda",
                dtype=attention_inputs.sequence_lengths.dtype,
            )
            sequence_lengths.copy_(
                attention_inputs.sequence_lengths, non_blocking=True
            ).add_(1)
            return FlashInferTRTLLMParams(
                batch_size=attention_inputs.sequence_lengths.size(0),
                max_seq_len=attention_inputs.sequence_lengths.max().item() + 1,
                seq_lens=sequence_lengths,
                block_tables=attention_inputs.kv_cache_block_id_device,
            )
        else:
            q_len = attention_inputs.input_lengths[0].item()
            sequence_lengths = torch.zeros_like(
                attention_inputs.prefix_lengths,
                device="cuda",
                dtype=attention_inputs.prefix_lengths.dtype,
            )
            sequence_lengths.copy_(
                attention_inputs.prefix_lengths, non_blocking=True
            ).add_(q_len)
            return FlashInferTRTLLMParams(
                batch_size=attention_inputs.prefix_lengths.size(0),
                max_seq_len=attention_inputs.prefix_lengths.max().item() + q_len,
                seq_lens=sequence_lengths,
                block_tables=attention_inputs.kv_cache_block_id_device,
            )

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        """Check if current implementation supports given inputs."""
        if not is_sm_100():
            return False
        if (
            attn_inputs.is_prefill
            and attn_inputs.input_lengths[0] < 6
            and (attn_inputs.input_lengths == attn_inputs.input_lengths[0])
            .all()
            .item()
        ):
            return True
        return (not attn_inputs.is_prefill) and (not attn_configs.use_mla)
    
    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        need_rope_kv_cache: bool = True,
    ) -> torch.Tensor:
        """执行前向传播计算。"""
        # Apply RoPE and KV cache operations
        fmha_input = apply_rope_and_kv_cache(
            qkv, kv_cache, self.rope_kvcache_impl, self.rope_params, need_rope_kv_cache
        )
        
        # Write to cache store if needed
        write_to_cache_store(kv_cache, self.attn_inputs, self.write_cache_store_impl)
        
        # Run TRTLLM decode attention directly
        q = fmha_input
        dtype = kv_cache.kv_cache_base.dtype
        q_type = q.dtype
        q = q.to(dtype)
        o_type = q_type
        q = q.contiguous().view(-1, self.local_head_num, self.head_dim)
        q_scale = 1.0
        k_scale = 1.0
        bmm1_scale = q_scale * k_scale * self.scaling
        bmm2_scale = 1.0
        o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=kv_cache.kv_cache_base,
            workspace_buffer=self.workspace_buffer,
            block_tables=self.fmha_params.block_tables,
            seq_lens=self.fmha_params.seq_lens,
            max_seq_len=self.fmha_params.max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            window_left=-1,
            sinks=None,
            out_dtype=o_type,
            q_len_per_req=q.shape[0] // self.fmha_params.seq_lens.shape[0],
        )
        return o.view(-1, self.local_head_num * self.head_dim).to(q_type)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        new_fmha_params = self._prepare_fmha_params(attn_inputs)
        self.fmha_params.seq_lens.copy_(new_fmha_params.seq_lens, non_blocking=True)

        new_rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        copy_kv_cache_offset(self.rope_params.kv_cache_offset, new_rope_params.kv_cache_offset)

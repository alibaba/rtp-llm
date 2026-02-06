import functools
from typing import Optional

import flashinfer
import torch

from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, FMHAType
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOpQOut,
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
    global g_zero_workspace_buffer, g_empty_workspace_buffer
    if g_zero_workspace_buffer is None:
        g_zero_workspace_buffer = torch.zeros(
            DEFAULT_WORKSPACE_SIZE_MB * 1024 * 1024,
            dtype=torch.uint8,
            device=device,
        )
    return g_zero_workspace_buffer


class FlashInferTRTLLMPrefillOp(object):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
    ):
        self.attn_configs = attn_configs
        self.head_dim = attn_configs.size_per_head
        self.head_num = attn_configs.head_num
        self.scaling = self.head_dim**-0.5
        self.local_head_num = attn_configs.head_num
        self.seq_size_per_block = attn_configs.tokens_per_block
        self.workspace_buffer = create_g_workspace_buffer()

    def support(self, attention_inputs: PyAttentionInputs):
        return (
            is_sm_100()
            and attention_inputs.is_prefill
            and attention_inputs.kv_cache_block_id_device is not None
        )

    def prepare(self, attention_inputs: PyAttentionInputs) -> FlashInferTRTLLMParams:
        prefix_lengths = torch.zeros_like(
            attention_inputs.input_lengths,
            device="cuda",
            dtype=attention_inputs.input_lengths.dtype,
        )
        input_lengths = torch.zeros_like(
            attention_inputs.input_lengths,
            device="cuda",
            dtype=attention_inputs.input_lengths.dtype,
        )
        prefix_lengths.copy_(attention_inputs.prefix_lengths, non_blocking=True)
        input_lengths.copy_(attention_inputs.input_lengths, non_blocking=True)
        sequence_lengths = input_lengths + prefix_lengths
        page_size = self.seq_size_per_block
        page_per_seq = (sequence_lengths + page_size - 1) // page_size
        cu_kv_seqlens = torch.zeros(
            attention_inputs.input_lengths.shape[0] + 1,
            device="cuda",
            dtype=attention_inputs.input_lengths.dtype,
        )
        cu_kv_seqlens[1:] = torch.cumsum(page_per_seq, dim=0, dtype=torch.int32)
        return FlashInferTRTLLMParams(
            batch_size=attention_inputs.input_lengths.size(0),
            max_q_len=attention_inputs.input_lengths.max().item(),
            max_kv_len=(
                attention_inputs.prefix_lengths + attention_inputs.input_lengths
            )
            .max()
            .item(),
            seq_lens=sequence_lengths,
            input_lens=attention_inputs.input_lengths,
            block_tables=attention_inputs.kv_cache_block_id_device,
            cu_seqlens=attention_inputs.cu_seqlens,
            cu_kv_seqlens=cu_kv_seqlens,
        )

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: Optional[KVCache],
        fmha_params: FlashInferTRTLLMParams,
    ) -> torch.Tensor:
        dtype = kv_cache.kv_cache_base.dtype
        q_type = q.dtype
        q = q.to(dtype)
        o_type = q_type
        q = q.contiguous().view(-1, self.local_head_num, self.head_dim)
        q_scale = 1.0
        k_scale = 1.0
        bmm1_scale = q_scale * k_scale * self.scaling
        bmm2_scale = 1.0
        o = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
            query=q,
            kv_cache=kv_cache.kv_cache_base,
            workspace_buffer=self.workspace_buffer,
            block_tables=fmha_params.block_tables,
            seq_lens=fmha_params.seq_lens,
            max_q_len=fmha_params.max_q_len,
            max_kv_len=fmha_params.max_kv_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            batch_size=fmha_params.batch_size,
            cum_seq_lens_q=fmha_params.cu_seqlens,
            cum_seq_lens_kv=fmha_params.cu_kv_seqlens,
            window_left=-1,
            # TODO: add attention_sink operation or nvfp4 scale factor if needed
            sinks=None,
            out_dtype=o_type,  # model_runner.dtype
        )

        return o.view(-1, self.local_head_num * self.head_dim).to(q_type)


class FlashInferTRTLLMDecodeOp(object):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
    ):
        self.attn_configs = attn_configs
        self.head_dim = attn_configs.size_per_head
        self.head_num = attn_configs.head_num
        self.scaling = self.head_dim**-0.5
        self.local_head_num = attn_configs.head_num
        self.workspace_buffer = create_g_workspace_buffer()

    def support(self, attention_inputs: PyAttentionInputs):
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

    def prepare(self, attention_inputs: PyAttentionInputs) -> FlashInferTRTLLMParams:
        if not attention_inputs.is_prefill:
            # need transfer to cuda, cuda graph can capture the add
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

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: Optional[KVCache],
        fmha_params: FlashInferTRTLLMParams,
    ) -> torch.Tensor:
        dtype = kv_cache.kv_cache_base.dtype
        q_type = q.dtype
        q = q.to(dtype)
        o_type = q_type

        q = q.contiguous().view(-1, self.local_head_num, self.head_dim)
        q_scale = 1.0
        k_scale = 1.0
        bmm1_scale = q_scale * k_scale * self.scaling
        bmm2_scale = 1.0
        # sink: additional value per head in the denominator of the softmax.

        # Call TRT-LLM kernel
        # raw_out: like q, [bs, acc_q_len, num_q_heads, head_dim] but with output dtype
        o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=kv_cache.kv_cache_base,
            workspace_buffer=self.workspace_buffer,
            block_tables=fmha_params.block_tables,
            seq_lens=fmha_params.seq_lens,
            max_seq_len=fmha_params.max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            window_left=-1,
            # TODO: add attention_sink operation or nvfp4 scale factor if needed
            sinks=None,
            out_dtype=o_type,  # model_runner.dtype
            q_len_per_req=q.shape[0] // fmha_params.seq_lens.shape[0],
        )
        return o.view(-1, self.local_head_num * self.head_dim).to(q_type)


class FlashInferTRTLLMPrefillImpl(FMHAImplBase):

    def __init__(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            FlashInferTRTLLMPrefillOp(attn_configs),
            FusedRopeKVCachePrefillOpQOut(attn_configs),
            attn_inputs,
        )

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.FLASH_INFER

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        new_fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.fmha_params.seq_lens.copy_(new_fmha_params.seq_lens, non_blocking=True)
        self.fmha_params.cu_kv_seqlens.copy_(
            new_fmha_params.cu_kv_seqlens, non_blocking=True
        )

        new_rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        new_offset = new_rope_params.kv_cache_offset
        old_offset = self.rope_params.kv_cache_offset
        self.copy_kv_cache_offset(old_offset, new_offset)


class FlashInferTRTLLMSpecDecodeImpl(FMHAImplBase):

    def __init__(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            FlashInferTRTLLMDecodeOp(attn_configs),
            FusedRopeKVCachePrefillOpQOut(attn_configs),
            attn_inputs,
        )
        self.support_ = self.support_ and (not attn_configs.use_mla)

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.FLASH_INFER

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        new_fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.fmha_params.seq_lens.copy_(new_fmha_params.seq_lens, non_blocking=True)

        new_rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        new_offset = new_rope_params.kv_cache_offset
        old_offset = self.rope_params.kv_cache_offset
        self.copy_kv_cache_offset(old_offset, new_offset)


class FlashInferTRTLLMDecodeImpl(FMHAImplBase):

    def __init__(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            FlashInferTRTLLMDecodeOp(attn_configs),
            FusedRopeKVCacheDecodeOp(attn_configs),
            attn_inputs,
        )
        self.support_ = self.support_ and (not attn_configs.use_mla)

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.FLASH_INFER

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        new_fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.fmha_params.seq_lens.copy_(new_fmha_params.seq_lens, non_blocking=True)

        new_rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        new_offset = new_rope_params.kv_cache_offset
        old_offset = self.rope_params.kv_cache_offset
        self.copy_kv_cache_offset(old_offset, new_offset)

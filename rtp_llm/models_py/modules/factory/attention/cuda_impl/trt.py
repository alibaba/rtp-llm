from typing import NamedTuple, Optional

import torch

try:
    from flashinfer.prefill import trtllm_fmha_v2_prefill
except (ImportError, AttributeError):
    trtllm_fmha_v2_prefill = None

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_impl.utils import (
    is_cuda_12_9_or_later,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.models_py.utils.arch import is_sm12x, is_sm90
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, ParallelismConfig, RopeStyle
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCachePrefillOpQKVOut,
    FusedRopeKVCachePrefillOpQOut,
    LayerKVCache,
    PyAttentionInputs,
)

# flashinfer JIT only generates flash_attention=True kernels;
# fmha_v2_run.cu determine_launch_params requires s >= 16 for flash_attention.
# Pad max_q_len / max_kv_len to this minimum so the dispatch selects flash kernels.
_TRTLLM_FMHA_V2_MIN_SEQ_LEN = 16

# This interface uses persistent CTAs, so partial outputs stay within each CTA
# instead of being stored in the workspace. Only a few runtime variables need
# workspace storage.
_TRTLLM_FMHA_V2_WORKSPACE_SIZE_BYTES = 1024
_g_trtllm_fmha_v2_workspace_pool: list[torch.Tensor] = []
_g_trtllm_fmha_v2_pool_lock = __import__("threading").Lock()


def _supports_trtllm_fmha_v2(attn_configs: AttentionConfigs) -> bool:
    if (
        not is_cuda_12_9_or_later()
        or trtllm_fmha_v2_prefill is None
        or attn_configs.head_num <= 0
        or attn_configs.kv_head_num <= 0
        or attn_configs.head_num % attn_configs.kv_head_num != 0
    ):
        return False

    if is_sm90():
        return attn_configs.dtype in {
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
        } and attn_configs.size_per_head in {
            32,
            40,
            48,
            64,
            80,
            96,
            104,
            128,
            160,
            192,
            256,
        }

    if is_sm12x():
        return (
            attn_configs.dtype in {torch.float16, torch.bfloat16}
            and attn_configs.kv_cache_dtype != KvCacheDataType.FP8
            and attn_configs.size_per_head in {64, 128, 256, 512}
        )

    return False


def _get_trtllm_fmha_v2_workspace(device: str = "cuda") -> torch.Tensor:
    with _g_trtllm_fmha_v2_pool_lock:
        if _g_trtllm_fmha_v2_workspace_pool:
            return _g_trtllm_fmha_v2_workspace_pool.pop()
        return torch.zeros(
            _TRTLLM_FMHA_V2_WORKSPACE_SIZE_BYTES,
            dtype=torch.uint8,
            device=device,
        )


def _release_trtllm_fmha_v2_workspace(buf: torch.Tensor) -> None:
    with _g_trtllm_fmha_v2_pool_lock:
        _g_trtllm_fmha_v2_workspace_pool.append(buf)


class TRTLLMFMHAv2Params(NamedTuple):
    batch_size: int
    max_q_len: int
    max_kv_len: int
    seq_lens: torch.Tensor
    cu_seqlens: torch.Tensor
    cu_kv_seqlens: Optional[torch.Tensor] = None  # paged only
    block_tables: Optional[torch.Tensor] = None  # paged only


class TRTLLMFMHAv2PagedPrefillOp:
    """Paged prefill op via trtllm_fmha_v2_prefill Q_PAGED_KV_HND layout."""

    def __init__(self, attn_configs: AttentionConfigs) -> None:
        self.attn_configs = attn_configs
        self.head_dim = attn_configs.size_per_head
        self.head_num = attn_configs.head_num
        self.kv_head_num = attn_configs.kv_head_num
        self.seq_size_per_block = attn_configs.kernel_tokens_per_block
        self.kv_cache_dtype = attn_configs.kv_cache_dtype
        self.bmm1_scale = (
            attn_configs.softmax_extra_scale
            / attn_configs.q_scaling
            * self.head_dim**-0.5
        )
        # RTP FlashInfer's "padding" mode masks only padded tokens, leaving valid
        # tokens bidirectional as required by non-causal attention.
        self.mask_mode = "causal" if attn_configs.is_causal else "padding"
        self.workspace_buffer = _get_trtllm_fmha_v2_workspace()

    def __del__(self) -> None:
        _release_trtllm_fmha_v2_workspace(self.workspace_buffer)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        page_size = attn_configs.kernel_tokens_per_block
        return _supports_trtllm_fmha_v2(attn_configs) and (
            page_size > 0 and page_size.bit_count() == 1
        )

    def prepare(self, attn_inputs: PyAttentionInputs) -> TRTLLMFMHAv2Params:
        cu_seqlens = attn_inputs.cu_seqlens_device
        cu_kv_seqlens = attn_inputs.cu_kv_seqlens_device
        seq_lens = cu_kv_seqlens[1:] - cu_kv_seqlens[:-1]

        block_tables = attn_inputs.kv_cache_kernel_block_id_device
        if block_tables is not None and not block_tables.is_cuda:
            block_tables = block_tables.to("cuda", non_blocking=True)

        max_kv = max(
            (attn_inputs.prefix_lengths + attn_inputs.input_lengths).max().item(),
            _TRTLLM_FMHA_V2_MIN_SEQ_LEN,
        )
        return TRTLLMFMHAv2Params(
            batch_size=attn_inputs.input_lengths.size(0),
            max_q_len=max(
                attn_inputs.input_lengths.max().item(), _TRTLLM_FMHA_V2_MIN_SEQ_LEN
            ),
            max_kv_len=max_kv,
            seq_lens=seq_lens,
            cu_seqlens=cu_seqlens,
            cu_kv_seqlens=cu_kv_seqlens,
            block_tables=block_tables,
        )

    def prepare_cuda_graph(
        self,
        params: TRTLLMFMHAv2Params,
    ) -> None:
        torch.sub(
            params.cu_kv_seqlens[1:],
            params.cu_kv_seqlens[:-1],
            out=params.seq_lens,
        )

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: LayerKVCache,
        params: TRTLLMFMHAv2Params,
    ) -> torch.Tensor:
        assert (
            params.block_tables is not None
        ), "kv_cache is required for paged TRT-LLM FMHA v2"

        q_type = q.dtype
        compute_dtype = (
            torch.float8_e4m3fn
            if self.kv_cache_dtype == KvCacheDataType.FP8
            else q_type
        )
        q = q.to(compute_dtype).contiguous().view(-1, self.head_num, self.head_dim)
        kv_cache_5d = common.reshape_paged_kv_cache(
            kv_cache.kv_cache_base,
            self.kv_head_num,
            self.seq_size_per_block,
            self.head_dim,
        )
        if kv_cache_5d.dtype != compute_dtype:
            raise TypeError(
                f"Q and paged KV cache must use the same dtype, got "
                f"Q={compute_dtype} and KV={kv_cache_5d.dtype}"
            )
        o = trtllm_fmha_v2_prefill(
            qkv=(q, kv_cache_5d),
            input_layout="Q_PAGED_KV_HND",
            workspace_buffer=self.workspace_buffer,
            seq_lens=params.seq_lens,
            max_q_len=params.max_q_len,
            max_kv_len=params.max_kv_len,
            bmm1_scale=self.bmm1_scale,
            bmm2_scale=1.0,
            batch_size=params.batch_size,
            cum_seq_lens_q=params.cu_seqlens,
            cum_seq_lens_kv=params.cu_kv_seqlens,
            block_tables=params.block_tables,
            out_dtype=q_type,
            mask_mode=self.mask_mode,
        )
        return o.view(-1, self.head_num * self.head_dim)


class TRTLLMFMHAv2PrefillOp:
    """Non-paged prefill op via TRT-LLM FMHA v2 packed or contiguous QKV."""

    def __init__(self, attn_configs: AttentionConfigs) -> None:
        self.attn_configs = attn_configs
        self.head_dim = attn_configs.size_per_head
        self.head_num = attn_configs.head_num
        self.kv_head_num = attn_configs.kv_head_num
        self.q_size = self.head_num * self.head_dim
        self.kv_size = self.kv_head_num * self.head_dim
        self.kv_cache_dtype = attn_configs.kv_cache_dtype
        self.bmm1_scale = (
            attn_configs.softmax_extra_scale
            / attn_configs.q_scaling
            * self.head_dim**-0.5
        )
        # RTP FlashInfer's "padding" mode masks only padded tokens, leaving valid
        # tokens bidirectional as required by non-causal attention.
        self.mask_mode = "causal" if attn_configs.is_causal else "padding"
        self.workspace_buffer = _get_trtllm_fmha_v2_workspace()

    def __del__(self) -> None:
        _release_trtllm_fmha_v2_workspace(self.workspace_buffer)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        has_prefix = (
            attn_inputs.prefix_lengths is not None
            and attn_inputs.prefix_lengths.numel() > 0
            and attn_inputs.prefix_lengths.any().item()
        )
        return _supports_trtllm_fmha_v2(attn_configs) and not has_prefix

    def prepare(self, attn_inputs: PyAttentionInputs) -> TRTLLMFMHAv2Params:
        self.attention_type = "mha" if self.head_num == self.kv_head_num else "gqa"
        cu_seqlens = attn_inputs.cu_seqlens_device
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_len = max(
            attn_inputs.input_lengths.max().item(), _TRTLLM_FMHA_V2_MIN_SEQ_LEN
        )
        return TRTLLMFMHAv2Params(
            batch_size=attn_inputs.input_lengths.size(0),
            max_q_len=max_len,
            max_kv_len=max_len,
            seq_lens=seq_lens,
            cu_seqlens=cu_seqlens,
        )

    def prepare_cuda_graph(
        self,
        params: TRTLLMFMHAv2Params,
    ) -> None:
        torch.sub(params.cu_seqlens[1:], params.cu_seqlens[:-1], out=params.seq_lens)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        params: TRTLLMFMHAv2Params,
    ) -> torch.Tensor:
        q_type = qkv.dtype
        compute_dtype = (
            torch.float8_e4m3fn
            if self.kv_cache_dtype == KvCacheDataType.FP8
            else q_type
        )
        if self.attention_type == "mha":
            fmha_input = (
                qkv.to(compute_dtype)
                .contiguous()
                .view(-1, 3, self.head_num, self.head_dim)
            )
            input_layout = "PACKED_QKV"
        else:
            q = (
                qkv[:, : self.q_size]
                .to(compute_dtype)
                .contiguous()
                .view(-1, self.head_num, self.head_dim)
            )
            kv = (
                qkv[:, self.q_size :]
                .to(compute_dtype)
                .contiguous()
                .view(-1, 2, self.kv_head_num, self.head_dim)
            )
            fmha_input = (q, kv)
            input_layout = "CONTIGUOUS_Q_KV"
        o = trtllm_fmha_v2_prefill(
            qkv=fmha_input,
            input_layout=input_layout,
            workspace_buffer=self.workspace_buffer,
            seq_lens=params.seq_lens,
            max_q_len=params.max_q_len,
            max_kv_len=params.max_kv_len,
            bmm1_scale=self.bmm1_scale,
            bmm2_scale=1.0,
            batch_size=params.batch_size,
            cum_seq_lens_q=params.cu_seqlens,
            cum_seq_lens_kv=params.cu_seqlens,
            out_dtype=q_type,
            mask_mode=self.mask_mode,
        )
        return o.view(-1, self.head_num * self.head_dim)


class FlashInferTRTLLMFMHAv2PagedPrefillImpl(FMHAImplBase):

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.fmha_impl = TRTLLMFMHAv2PagedPrefillOp(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQOut(attn_configs)
        self.attn_inputs = attn_inputs
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return TRTLLMFMHAv2PagedPrefillOp.support(attn_configs, attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int,
    ) -> torch.Tensor:
        fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs) -> None:
        self.fmha_impl.prepare_cuda_graph(self.fmha_params)
        new_kv_cache_offset = self.rope_kvcache_impl.prepare(
            attn_inputs
        ).kv_cache_offset
        if new_kv_cache_offset is not None:
            common.copy_kv_cache_offset(
                self.rope_params.kv_cache_offset, new_kv_cache_offset
            )


class FlashInferTRTLLMFMHAv2PrefillImpl(FMHAImplBase):
    """Non-paged prefill selecting packed MHA or contiguous GQA input."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.need_rope = attn_configs.rope_config.style != RopeStyle.No
        self.fmha_impl = TRTLLMFMHAv2PrefillOp(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQKVOut(attn_configs)
        self.attn_inputs = attn_inputs
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return TRTLLMFMHAv2PrefillOp.support(attn_configs, attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: Optional[int] = 0,
    ) -> torch.Tensor:
        fmha_input = (
            self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
            if self.need_rope or kv_cache is not None
            else qkv
        )
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs) -> None:
        self.fmha_impl.prepare_cuda_graph(self.fmha_params)
        new_kv_cache_offset = self.rope_kvcache_impl.prepare(
            attn_inputs
        ).kv_cache_offset
        assert (self.rope_params.kv_cache_offset is None) == (
            new_kv_cache_offset is None
        )
        if new_kv_cache_offset is not None:
            common.copy_kv_cache_offset(
                self.rope_params.kv_cache_offset, new_kv_cache_offset
            )

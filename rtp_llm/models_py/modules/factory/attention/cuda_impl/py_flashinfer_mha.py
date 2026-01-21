from typing import Optional

import torch
from flashinfer.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)

from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHADecodeImplBase,
    FMHAPrefillImplBase,
    FMHAType,
)
from rtp_llm.ops import AttentionConfigs
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOp,
    KVCache,
    ParamsBase,
    PyAttentionInputs,
    fill_mla_params,
)


class PyFlashinferPrefillPagedAttnOp(object):
    """FlashInfer Prefill Attention Op with Paged KV Cache support"""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        backend: str = "auto",
    ) -> None:
        self.g_workspace_buffer = torch.empty(
            512 * 1024 * 1024,
            dtype=torch.int8,
            device="cuda",
        )
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head
        self.head_dim_vo = attn_configs.size_per_head
        self.page_size = attn_configs.tokens_per_block
        self.datatype = torch.float16

        # Use Paged KV Cache wrapper
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.g_workspace_buffer,
            "NHD",
            backend=backend,
        )

    def prepare(
        self,
        attn_inputs: PyAttentionInputs,
    ) -> ParamsBase:
        """
        Prepare the prefill wrapper with paged KV cache parameters

        Args:
            attn_inputs: Attention inputs containing sequence information
            paged_kv_indptr: Page count boundaries [batch_size + 1]
            paged_kv_indices: Actual page IDs [total_pages]
            paged_kv_last_page_len: Valid length of last page [batch_size]
        """
        qo_indptr = attn_inputs.cu_seqlens[: attn_inputs.input_lengths.size(0) + 1]

        # from rtp_llm.models_py.utils.debug import set_trace_on_tty
        # set_trace_on_tty()
        flashinfer_prefill_params = fill_mla_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id_host,
            self.page_size,
        )
        # Get torch.dtype from attention configs
        self.prefill_wrapper.plan(
            qo_indptr,
            flashinfer_prefill_params.decode_page_indptr_d,
            flashinfer_prefill_params.page_indice_d,
            flashinfer_prefill_params.paged_kv_last_page_len_d,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.page_size,
            causal=True,
            q_data_type=self.datatype,
        )
        return ParamsBase()

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return attn_inputs.prefix_lengths.numel() > 0

    def forward(
        self, q: torch.Tensor, kv_cache: Optional[KVCache], params: ParamsBase
    ) -> torch.Tensor:
        """
        Forward pass with paged KV cache

        Args:
            q: Query tensor [total_tokens, num_heads, head_dim]
            kv_cache: Paged KV cache [num_pages, 2, page_size, kv_heads, head_dim]
            params: Parameters (not used currently)

        Returns:
            output: [total_tokens, num_heads, head_dim]
        """
        assert kv_cache is not None, "kv_cache is required for paged attention"
        assert (
            q.dim() == 3
        ), f"Expected q to be 3D tensor [total_tokens, num_heads, head_dim], got {q.dim()}D"
        return self.prefill_wrapper.run(q, kv_cache.kv_cache_base)


class PyFlashinferPrefillAttnOp(object):
    def __init__(self, attn_configs: AttentionConfigs, backend: str = "auto") -> None:
        self.g_workspace_buffer = torch.empty(
            512 * 1024 * 1024,
            dtype=torch.int8,
            device="cuda",
        )
        # attn_configs.head_num and kv_head_num are already divided by tp_size in ModelConfig::getAttentionConfigs
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head
        # TODO: maybe use v_head_dim
        self.head_dim_vo = attn_configs.size_per_head
        self.prefill_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self.g_workspace_buffer,
            "NHD",
            backend=backend,
        )
        # Default to fp16, can be overridden if needed
        self.datatype = torch.float16

    def prepare(self, attn_inputs: PyAttentionInputs) -> ParamsBase:
        """
        Prepare the prefill wrapper

        Args:
            attn_inputs: Attention inputs containing sequence information
        """
        batch_size = attn_inputs.input_lengths.size(0)
        cu_seqlens = attn_inputs.cu_seqlens[: batch_size + 1]

        self.prefill_wrapper.plan(
            cu_seqlens,
            cu_seqlens,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.head_dim_vo,
            causal=True,
            q_data_type=self.datatype,
        )
        return ParamsBase()

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    ## 1. pure prefill attn: qkv contains q and k,v
    ## 2. paged attn: qkv is only q, and kv is in kv_cache
    def forward(
        self, qkv: torch.Tensor, kv_cache: Optional[KVCache], params: ParamsBase
    ) -> torch.Tensor:
        qkv = qkv.reshape(qkv.shape[0], -1)
        q, k, v = torch.split(
            qkv,
            [
                self.head_dim_qk * self.local_head_num,
                self.head_dim_qk * self.local_kv_head_num,
                self.head_dim_vo * self.local_kv_head_num,
            ],
            dim=-1,
        )
        q = q.reshape(q.shape[0], self.local_head_num, self.head_dim_qk)
        k = k.reshape(k.shape[0], self.local_kv_head_num, self.head_dim_qk)
        v = v.reshape(v.shape[0], self.local_kv_head_num, self.head_dim_vo)
        return self.prefill_wrapper.run(q, k, v)


class PyFlashinferPrefillImpl(FMHAPrefillImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
    ) -> None:
        super().__init__(
            PyFlashinferPrefillAttnOp(attn_configs),
            FusedRopeKVCachePrefillOp(attn_configs),
            attn_inputs,
        )

    def support(self):
        return True

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.PY_FLASHINFER_PREFILL


class PyFlashinferPagedPrefillImpl(FMHAPrefillImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
    ) -> None:
        super().__init__(
            PyFlashinferPrefillPagedAttnOp(attn_configs),
            FusedRopeKVCachePrefillOp(attn_configs),
            attn_inputs,
        )

    def support(self):
        return True

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.PY_FLASHINFER_PREFILL


from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper


def determine_use_tensor_core_from_configs(attn_configs: AttentionConfigs) -> bool:
    """Determine whether to use tensor cores based on attention configs."""
    # Use tensor cores for larger head dimensions and when kv_head_num matches requirements
    return attn_configs.head_num // attn_configs.kv_head_num >= 4


class PyFlashinferDecodeAttnOp(object):
    def __init__(self, attn_configs: AttentionConfigs) -> None:
        # Get dtype from attn_configs (ScalarType is automatically converted to torch.dtype by pybind11)
        self.dtype = attn_configs.dtype
        self.g_workspace_buffer = torch.empty(
            512 * 1024 * 1024,
            dtype=torch.int8,
            device="cuda",
        )
        # attn_configs already has head_num and kv_head_num divided by tp_size
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head
        self.head_dim_vo = attn_configs.size_per_head
        self.seq_size_per_block = attn_configs.tokens_per_block
        self.use_tensor_core = determine_use_tensor_core_from_configs(attn_configs)
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.g_workspace_buffer,
            "HND",
            use_tensor_cores=self.use_tensor_core,
        )
        # Default to fp16, can be overridden if needed

    def prepare(self, attn_inputs: PyAttentionInputs):
        # from rtp_llm.models_py.utils.debug import set_trace_on_tty
        # set_trace_on_tty()
        flashinfer_decode_params = fill_mla_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id_host,
            self.seq_size_per_block,
        )
        # Get torch.dtype from attention configs
        self.decode_wrapper.plan(
            flashinfer_decode_params.decode_page_indptr_d,
            flashinfer_decode_params.page_indice_d,
            flashinfer_decode_params.paged_kv_last_page_len_d,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.seq_size_per_block,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
        )
        return flashinfer_decode_params

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def forward(
        self, q: torch.Tensor, kv_cache: Optional[KVCache], params: ParamsBase
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required"
        q = q.reshape(q.shape[0], self.local_head_num, self.head_dim_qk)
        return self.decode_wrapper.run(q, kv_cache.kv_cache_base)


class PyFlashinferDecodeImpl(FMHADecodeImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        cos_sin_cache: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(
            PyFlashinferDecodeAttnOp(attn_configs),
            FusedRopeKVCacheDecodeOp(attn_configs),
            attn_inputs,
        )
        self.support_ = attn_configs.use_mla == False

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.PY_FLASHINFER_DECODE

    def support(self):
        return self.support_

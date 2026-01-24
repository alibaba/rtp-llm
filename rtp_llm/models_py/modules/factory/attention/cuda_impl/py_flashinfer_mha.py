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
        self.g_workspace_buffer = torch.zeros(
            512 * 1024 * 1024,
            dtype=torch.uint8,
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
            "HND",
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
        # print(f"\n[PyFlashinferPrefillPagedAttnOp.prepare] ========== REUSE CACHE DEBUG ==========")
        # print(f"  Batch size: {attn_inputs.input_lengths.size(0)}")
        # print(f"  🔑 prefix_lengths (reused tokens): {attn_inputs.prefix_lengths}")
        # print(f"  📏 sequence_lengths (total): {attn_inputs.sequence_lengths}")
        # print(f"  ➕ input_lengths (new tokens): {attn_inputs.input_lengths}")
        # print(f"  📊 Reuse ratio per sample: {[f'{p.item()}/{s.item()} ({p.item()/s.item()*100:.1f}%)' for p, s in zip(attn_inputs.prefix_lengths, attn_inputs.sequence_lengths)]}")
        # print(f"  💾 kv_cache_block_id_host shape: {attn_inputs.kv_cache_block_id_host.shape}")
        # print(f"  💾 kv_cache_block_id_host (all): {attn_inputs.kv_cache_block_id_host}")
        # import debugpy
        # debugpy.listen(("localhost", 44553))
        # print("Waiting for debugger attach...")
        # debugpy.wait_for_client()
        # debugpy.breakpoint()
        # Validation checks for cache reuse
        # for i in range(attn_inputs.input_lengths.size(0)):
        # prefix_len = attn_inputs.prefix_lengths[i].item()
        # seq_len = attn_inputs.sequence_lengths[i].item()
        # input_len = attn_inputs.input_lengths[i].item()

        # if prefix_len + input_len != seq_len:
        #     print(f"  ⚠️  WARNING Sample {i}: prefix_len({prefix_len}) + input_len({input_len}) != seq_len({seq_len})")

        # if prefix_len > 0:
        #     print(f"  ✅ Sample {i} IS reusing cache: {prefix_len} tokens reused, {input_len} new tokens")
        # else:
        #     print(f"  ℹ️  Sample {i} NOT reusing cache (fresh prefill): {seq_len} tokens")

        flashinfer_prefill_params = fill_mla_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id_host,
            self.page_size,
        )

        # print(f"\n[PyFlashinferPrefillPagedAttnOp.prepare] FlashInfer parameters:", flush=True)
        # print(f"  qo_indptr shape: {qo_indptr.shape}, device: {qo_indptr.device}", flush=True)
        # print(f"  qo_indptr: {qo_indptr}", flush=True)
        # print(f"  decode_page_indptr_d shape: {flashinfer_prefill_params.decode_page_indptr_d.shape}", flush=True)
        # print(f"  decode_page_indptr_d: {flashinfer_prefill_params.decode_page_indptr_d}", flush=True)
        # print(f"  page_indice_d shape: {flashinfer_prefill_params.page_indice_d.shape}", flush=True)
        # print(f"  page_indice_d (all): {flashinfer_prefill_params.page_indice_d}", flush=True)
        # print(f"  paged_kv_last_page_len_d shape: {flashinfer_prefill_params.paged_kv_last_page_len_d.shape}", flush=True)
        # print(f"  paged_kv_last_page_len_d: {flashinfer_prefill_params.paged_kv_last_page_len_d}", flush=True)
        # print(f"  local_head_num (Q): {self.local_head_num}", flush=True)
        # print(f"  local_kv_head_num (KV): {self.local_kv_head_num}", flush=True)
        # print(f"  head_dim_qk: {self.head_dim_qk}", flush=True)
        # print(f"  page_size: {self.page_size}", flush=True)
        # print(f"  q_data_type: {self.datatype}", flush=True)

        # print(f"  kv_data_type: {self.datatype}", flush=True)

        # # Validate paged_kv_last_page_len_d
        # last_page_lens = flashinfer_prefill_params.paged_kv_last_page_len_d
        # invalid_lens = (last_page_lens <= 0) | (last_page_lens > self.page_size)
        # if invalid_lens.any():
        #     print(f"\n  ⚠️  WARNING: Invalid paged_kv_last_page_len_d values detected!", flush=True)
        #     print(f"    page_size: {self.page_size}", flush=True)
        #     print(f"    Invalid values: {last_page_lens[invalid_lens]}", flush=True)
        #     print(f"    Valid range: (0, {self.page_size}]", flush=True)

        # Save parameters for forward statistics and debugging
        self.page_indice_d = flashinfer_prefill_params.page_indice_d
        self.qo_indptr_d = qo_indptr
        self.paged_kv_last_page_len_d = (
            flashinfer_prefill_params.paged_kv_last_page_len_d
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
            kv_data_type=self.datatype,  # Critical fix: must specify KV cache data type!
        )

        print(f"  ✅ Prefill wrapper.plan() completed successfully", flush=True)
        return ParamsBase()

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

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

        # print(f"\n[PyFlashinferPrefillPagedAttnOp.forward] Running attention:", flush=True)
        # print(f"  q shape: {q.shape}, dtype: {q.dtype}, device: {q.device}", flush=True)
        # print(f"  kv_cache shape: {kv_cache.kv_cache_base.shape}, dtype: {kv_cache.kv_cache_base.dtype}", flush=True)
        # print(f"  Expected KV cache format: [num_pages, 2, page_size={self.page_size}, kv_heads={self.local_kv_head_num}, head_dim={self.head_dim_qk}]", flush=True)

        # # Print Q statistics
        # print(f"\n  📊 Q Statistics:", flush=True)
        # print(f"    Q max: {q.max().item():.6f}", flush=True)
        # print(f"    Q min: {q.min().item():.6f}", flush=True)
        # print(f"    Q mean: {q.mean().item():.6f}", flush=True)
        # print(f"    Q std: {q.std().item():.6f}", flush=True)

        # # Check Q for NaN/Inf BEFORE FlashInfer
        # q_has_nan = torch.isnan(q).any().item()
        # q_has_inf = torch.isinf(q).any().item()
        # if q_has_nan or q_has_inf:
        #     print(f"\n  ⚠️  WARNING: Q contains NaN: {q_has_nan}, Inf: {q_has_inf} BEFORE FlashInfer!", flush=True)
        #     if q_has_nan:
        #         print(f"    Q NaN count: {torch.isnan(q).sum().item()} / {q.numel()}", flush=True)
        #     if q_has_inf:
        #         print(f"    Q Inf count: {torch.isinf(q).sum().item()} / {q.numel()}", flush=True)
        # else:
        #     print(f"  ✅ Q has no NaN/Inf before FlashInfer", flush=True)

        # # Print KV Cache statistics for used blocks
        # if hasattr(self, 'page_indice_d') and self.page_indice_d is not None:
        #     # Index the KV cache blocks that will be used
        #     used_kv_blocks = kv_cache.kv_cache_base[self.page_indice_d]
        #     print(f"\n  📦 KV Cache Statistics (used {self.page_indice_d.shape[0]} blocks):", flush=True)
        #     print(f"    Used page indices (all): {self.page_indice_d}", flush=True)
        #     print(f"    KV max: {used_kv_blocks.max().item():.6f}", flush=True)
        #     print(f"    KV min: {used_kv_blocks.min().item():.6f}", flush=True)
        #     print(f"    KV mean: {used_kv_blocks.mean().item():.6f}", flush=True)
        #     print(f"    KV std: {used_kv_blocks.std().item():.6f}", flush=True)

        #     # Check KV cache for NaN/Inf BEFORE FlashInfer
        #     kv_has_nan = torch.isnan(used_kv_blocks).any().item()
        #     kv_has_inf = torch.isinf(used_kv_blocks).any().item()
        #     if kv_has_nan or kv_has_inf:
        #         print(f"\n  ⚠️  WARNING: Used KV blocks contain NaN: {kv_has_nan}, Inf: {kv_has_inf} BEFORE FlashInfer!", flush=True)
        #         if kv_has_nan:
        #             print(f"    KV NaN count: {torch.isnan(used_kv_blocks).sum().item()} / {used_kv_blocks.numel()}", flush=True)
        #         if kv_has_inf:
        #             print(f"    KV Inf count: {torch.isinf(used_kv_blocks).sum().item()} / {used_kv_blocks.numel()}", flush=True)
        #     else:
        #         print(f"  ✅ Used KV blocks have no NaN/Inf before FlashInfer", flush=True)
        # else:
        #     print(f"\n  ⚠️  page_indice_d not available, skipping KV cache statistics", flush=True)

        # # Print FlashInfer wrapper state
        # print(f"\n  🔧 FlashInfer Wrapper Parameters:", flush=True)
        # if hasattr(self, 'qo_indptr_d') and self.qo_indptr_d is not None:
        #     print(f"    qo_indptr_d: {self.qo_indptr_d.tolist()}", flush=True)
        # if hasattr(self, 'paged_kv_last_page_len_d') and self.paged_kv_last_page_len_d is not None:
        #     print(f"    paged_kv_last_page_len_d: {self.paged_kv_last_page_len_d.tolist()}", flush=True)

        # Validate page indices
        # num_pages = kv_cache.kv_cache_base.shape[0]
        # if hasattr(self, 'page_indice_d') and self.page_indice_d is not None:
        #     max_page_idx = self.page_indice_d.max().item()
        #     min_page_idx = self.page_indice_d.min().item()
        #     if max_page_idx >= num_pages or min_page_idx < 0:
        #         print(f"\n  ⚠️  WARNING: Invalid page indices detected!", flush=True)
        #         print(f"    KV cache num_pages: {num_pages}", flush=True)
        #         print(f"    page_indice_d range: [{min_page_idx}, {max_page_idx}]", flush=True)
        #         print(f"    Out-of-bounds indices will cause memory errors!", flush=True)

        # print(f"\n  🚀 Calling FlashInfer prefill_wrapper.run()...", flush=True)
        if self.page_indice_d.numel() > 0:
            _ = kv_cache.kv_cache_base[self.page_indice_d].clone()

        out = self.prefill_wrapper.run(q, kv_cache.kv_cache_base)
        # print(f"  ✅ FlashInfer completed", flush=True)

        # print(f"  ✅ Attention forward completed, output shape: {out.shape}", flush=True)

        # # Print last 10 tokens, first 10 dimensions each
        # print(f"\n  🔍 Output sample (last 10 tokens, first 10 dims each):", flush=True)
        # num_tokens_to_print = min(10, out.shape[0])
        # start_idx = out.shape[0] - num_tokens_to_print
        # for i in range(start_idx, out.shape[0]):
        #     token_data = out[i].flatten()[:10]  # Flatten and take first 10 values
        #     print(f"    Token {i}: {token_data.tolist()}", flush=True)

        # # Check for NaN and Inf
        # has_nan = torch.isnan(out).any().item()
        # has_inf = torch.isinf(out).any().item()
        # if has_nan or has_inf:
        #     print(f"\n  ⚠️  WARNING: Output contains NaN: {has_nan}, Inf: {has_inf}", flush=True)
        #     if has_nan:
        #         nan_count = torch.isnan(out).sum().item()
        #         print(f"    NaN count: {nan_count} / {out.numel()}", flush=True)
        #     if has_inf:
        #         inf_count = torch.isinf(out).sum().item()
        #         print(f"    Inf count: {inf_count} / {out.numel()}", flush=True)
        # else:
        #     print(f"\n  ✅ No NaN or Inf in output", flush=True)

        return out


class PyFlashinferPrefillAttnOp(object):
    def __init__(self, attn_configs: AttentionConfigs, backend: str = "auto") -> None:

        self.g_workspace_buffer = torch.zeros(
            512 * 1024 * 1024,
            dtype=torch.uint8,
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
            backend=backend,
        )
        self.datatype = attn_configs.dtype

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
        return (
            attn_inputs.prefix_lengths.numel() <= 0
            or attn_inputs.prefix_lengths.sum().item() == 0
        )

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
        return self.support_

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.PY_FLASHINFER_PREFILL_RAGGED


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
        return self.support_

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.PY_FLASHINFER_PREFILL_PAGED


from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper


def determine_use_tensor_core_from_configs(attn_configs: AttentionConfigs) -> bool:
    """Determine whether to use tensor cores based on attention configs."""
    # Use tensor cores for larger head dimensions and when kv_head_num matches requirements
    return attn_configs.head_num // attn_configs.kv_head_num >= 4


class PyFlashinferDecodeAttnOp(object):
    def __init__(self, attn_configs: AttentionConfigs) -> None:
        # Get dtype from attn_configs (ScalarType is automatically converted to torch.dtype by pybind11)
        self.dtype = attn_configs.dtype

        self.g_workspace_buffer = torch.zeros(
            512 * 1024 * 1024,
            dtype=torch.uint8,
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
        self.datatype = attn_configs.dtype

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

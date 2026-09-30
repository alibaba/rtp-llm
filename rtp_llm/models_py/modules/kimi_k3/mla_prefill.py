"""K3 expanded prefill MLA, including bounded historical cache reads."""

import os
import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import MlaFlashInferPrefillOp
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_fp8_kernels import (
    gather_fp8_prefix, gather_fp8_prefix_slice, quantize_fp8,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_qkv_fp8_quant import quantize_qkv_fp8
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_fused_fp8_epilogue import fused_mla_fp8_epilogue
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import mla_fp8_kernels
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_prefix_chunk_plan import plan_prefix_chunks
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_state_merge import merge_mla_states_in_place
from rtp_llm.models_py.modules.kimi_k3.linear import KimiK3Bf16Linear
from rtp_llm.utils.model_weight import W
from rtp_llm.models_py.modules.kimi_k3.native_mla_prefill import KimiK3TokenspeedPrefill
from rtp_llm.ops import KvCacheDataType

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    MlaFlashInferPrefillImpl,
)


class KimiK3MlaPrefillOp(MlaFlashInferPrefillOp):
    def _attention_dtype(self):
        return torch.float8_e4m3fn if self.kv_cache_type == KvCacheDataType.FP8 else torch.bfloat16

    def _create_prefill_wrapper(self):
        return KimiK3TokenspeedPrefill(fp8_compute=self.kv_cache_type == KvCacheDataType.FP8)

    def plan(self, mla_params):
        self.prefill_wrapper.plan(
            mla_params.qo_indptr_d,
            mla_params.prefill_ragged_kv_len_indptr_d,
            self.num_heads,
            self.num_heads,
            self.qk_rope_head_dim + self.qk_nope_head_dim,
            self.v_head_dim,
            sm_scale=(1.0 / (self.qk_rope_head_dim + self.qk_nope_head_dim) ** 0.5)
            * self.softmax_extra_scale,
            causal=True,
            q_data_type=self._attention_dtype(),
            kv_data_type=self._attention_dtype(),
            qo_indptr_host=mla_params.qo_indptr_h,
            kv_indptr_host=mla_params.prefill_ragged_kv_len_indptr_h,
        )
        self.reuse_cache_page_indice = mla_params.reuse_cache_page_indice_d
        self.qo_indptr = mla_params.qo_indptr_d
        self.batch_reuse_info_vec = mla_params.batch_reuse_info_vec_d
        self.total_kv_lens = int(mla_params.prefill_ragged_kv_len_indptr_h[-1])
        if self.kv_cache_type == KvCacheDataType.FP8:
            q_offsets = tuple(int(v) for v in mla_params.qo_indptr_h.tolist())
            kv_offsets = tuple(
                int(v) for v in mla_params.prefill_ragged_kv_len_indptr_h.tolist()
            )
            self._prefix_q_offsets = q_offsets
            self._prefix_lens = tuple(
                kv_offsets[i + 1] - kv_offsets[i] - (q_offsets[i + 1] - q_offsets[i])
                for i in range(len(q_offsets) - 1)
            )
            self._prefix_plan = plan_prefix_chunks(
                tuple(q_offsets[i + 1] - q_offsets[i]
                      for i in range(len(q_offsets) - 1)),
                self._prefix_lens,
                page_tokens=self.token_per_block,
                heads=self.num_heads,
                qk_dim=self.qk_rope_head_dim + self.qk_nope_head_dim,
                v_dim=self.v_head_dim,
                operand_bytes=1,
                budget_gib=float(os.environ.get(
                    "RTP_MLA_PREFILL_EXPANDED_KV_BUDGET_GIB", "6.0"
                )),
            )
        self.block_table = mla_params.page_indice_d.unsqueeze(0)
        self.workspace_starts = torch.zeros(
            1, dtype=torch.int32, device=self.block_table.device
        )
        self.seq_lens = mla_params.prefill_ragged_kv_len_indptr_d[-1:]

    def _reuse_kv_cache_indexed_batched(self, compressed_kv, k_pe, kv_cache):
        if self.kv_cache_type == KvCacheDataType.FP8:
            if self.reuse_cache_page_indice is None or self.reuse_cache_page_indice.numel() == 0:
                return compressed_kv, k_pe
            if kv_cache is None:
                raise ValueError("K3 FP8 prefix reuse requires a target KV cache")
            latent = torch.empty((self.total_kv_lens, self.kv_lora_rank),
                                 dtype=torch.bfloat16, device=compressed_kv.device)
            suffix = torch.empty((self.total_kv_lens, self.qk_rope_head_dim),
                                 dtype=torch.bfloat16, device=compressed_kv.device)
            gather_fp8_prefix(
                latent, suffix, compressed_kv, k_pe,
                kv_cache.kv_cache_base.view(-1, self.token_per_block,
                                            self.kv_lora_rank + self.qk_rope_head_dim),
                self.reuse_cache_page_indice, self.batch_reuse_info_vec,
                self.qo_indptr, self.token_per_block, scale=1.0,
            )
            return latent, suffix
        latent, suffix = super()._reuse_kv_cache_indexed_batched(compressed_kv, k_pe, kv_cache)
        # The shared gather reserves full cache pages, but packs only each
        # request's prefix and query rows. TokenSpeed expects that valid extent.
        # Narrow before kv_b_proj so unused capacity is neither projected nor read.
        return latent[: self.total_kv_lens], suffix[: self.total_kv_lens]

    def _make_kv_b_proj(self, layer_id):
        weight = self.weights[layer_id][W.mla_kv_b_w]
        if weight.is_cuda and weight.dtype == torch.bfloat16:
            return KimiK3Bf16Linear(weight)
        return super()._make_kv_b_proj(layer_id)

    def _project_kv(self, projection, compressed_kv, k_pe):
        """Project K/V into the optional packed layout around the RoPE gap."""
        left, middle, right = (
            self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim
        )
        head_splits = (left, middle, right)
        if projection.supports_skip_head_mid(compressed_kv, head_splits):
            packed = projection.forward_skip_head_mid(
                compressed_kv, head_splits
            ).view(-1, self.num_heads, left + middle + right)
            packed[..., left : left + middle].copy_(
                k_pe.view(-1, 1, middle)
            )
            return packed[..., : left + middle], packed[..., -right:]

        projected = projection(compressed_kv).view(
            -1, self.num_heads, left + right
        )
        key = self._concat_and_cast_mha_k(
            projected[..., :left], k_pe.view(-1, 1, middle)
        )
        return key, projected[..., left:]

    def forward_with_cache_insert(
        self, q, compressed_kv, k_pe, kv_cache, layer_id,
        slot_mapping, cache_scale, cache_scale_value,
    ):
        """Fuse ordinary FP8 MLA operand conversion with current KV insertion.

        Prefix reuse and a nonunit cache scale keep their existing paths.
        The caller performs the PD cache-store handoff after this method.
        """
        if (self.kv_cache_type != KvCacheDataType.FP8 or kv_cache is None
                or cache_scale_value != 1.0 or self._prefix_plan.chunked
                or any(self._prefix_lens) or mla_fp8_kernels._FP8_DIAGNOSTICS
                or compressed_kv.shape[0] != q.shape[0]
                or slot_mapping.dtype != torch.int64
                or kv_cache.kv_cache_base.dtype != torch.float8_e4m3fn
                or not kv_cache.kv_cache_base.is_contiguous()):
            return None

        left, middle, right = (
            self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim
        )
        projection = self._make_kv_b_proj(layer_id)
        head_splits = (left, middle, right)
        if projection.supports_skip_head_mid(compressed_kv, head_splits):
            projected = projection.forward_skip_head_mid(
                compressed_kv, head_splits
            ).view(-1, self.num_heads, left + middle + right)
            k_nope, value = projected[..., :left], projected[..., -right:]
        else:
            projected = projection(compressed_kv).view(
                -1, self.num_heads, left + right
            )
            k_nope, value = projected[..., :left], projected[..., left:]

        cache = kv_cache.kv_cache_base.view(
            -1, self.token_per_block, self.kv_lora_rank + middle
        )
        with torch.profiler.record_function("RTP::attention.mla.fused_fp8_epilogue"):
            q_fp8, k_fp8, v_fp8 = fused_mla_fp8_epilogue(
                q, k_nope, k_pe.view(-1, middle), compressed_kv, value,
                cache, slot_mapping, cache_scale, cache_scale,
                cache_scale, cache_scale, assume_unit_scales=True,
            )
        return self.prefill_wrapper.run(q_fp8, k_fp8, v_fp8).view(
            -1, self.num_heads, self.v_head_dim
        )

    def _forward_fp8_chunked(self, q, compressed_kv, k_pe, kv_cache, layer_id):
        if kv_cache is None or self.reuse_cache_page_indice is None:
            raise ValueError("chunked FP8 MLA Prefill requires a paged target KV cache")
        cache = kv_cache.kv_cache_base.view(
            -1, self.token_per_block, self.kv_lora_rank + self.qk_rope_head_dim
        )
        projection = self._make_kv_b_proj(layer_id)

        # Current Q attends causally to current K/V. Historical K/V is merged
        # from separate noncausal calls, so none of it is expanded here.
        current_k, current_v = self._project_kv(
            projection, compressed_kv, k_pe
        )
        q_fp8, current_k_fp8, current_v_fp8 = quantize_qkv_fp8(
            q, current_k, current_v
        )
        output, output_lse = self.prefill_wrapper.run_partial(
            q_fp8, current_k_fp8, current_v_fp8,
            qo_indptr=self.qo_indptr,
            kv_indptr=self.qo_indptr,
            max_q=self.prefill_wrapper.max_q,
            max_k=self.prefill_wrapper.max_q,
            causal=True,
        )
        del current_k, current_v, current_k_fp8, current_v_fp8

        for segment in self._prefix_plan.slices:
            owner = segment.owner
            q_start = self._prefix_q_offsets[owner]
            q_len = self._prefix_q_offsets[owner + 1] - q_start
            if q_len == 0:
                continue
            latent = torch.empty(
                (segment.length, self.kv_lora_rank),
                dtype=torch.bfloat16, device=q.device,
            )
            suffix = torch.empty(
                (segment.length, self.qk_rope_head_dim),
                dtype=torch.bfloat16, device=q.device,
            )
            gather_fp8_prefix_slice(
                latent, suffix, cache, self.reuse_cache_page_indice,
                self.batch_reuse_info_vec, self.token_per_block,
                owner=owner, start=segment.start,
                prefix_len=self._prefix_lens[owner], scale=1.0,
            )
            key, value = self._project_kv(projection, latent, suffix)
            key_fp8 = quantize_fp8(key)
            value_fp8 = quantize_fp8(value)
            q_indptr = torch.tensor((0, q_len), dtype=torch.int32, device=q.device)
            kv_indptr = torch.tensor(
                (0, segment.length), dtype=torch.int32, device=q.device
            )
            partial, partial_lse = self.prefill_wrapper.run_partial(
                q_fp8.narrow(0, q_start, q_len), key_fp8, value_fp8,
                qo_indptr=q_indptr, kv_indptr=kv_indptr,
                max_q=q_len, max_k=segment.length, causal=False,
            )
            merge_mla_states_in_place(
                output.narrow(0, q_start, q_len),
                output_lse.narrow(0, q_start, q_len),
                partial, partial_lse,
            )
            del latent, suffix, key, value, key_fp8, value_fp8
            del q_indptr, kv_indptr, partial, partial_lse
        return output.view(-1, self.num_heads, self.v_head_dim)

    def forward(self, q, compressed_kv, k_pe, kv_cache, layer_id):
        if self.kv_cache_type != KvCacheDataType.FP8:
            return super().forward(q, compressed_kv, k_pe, kv_cache, layer_id)
        if self._prefix_plan.chunked:
            return self._forward_fp8_chunked(
                q, compressed_kv, k_pe, kv_cache, layer_id
            )
        compressed_kv, k_pe = self._reuse_kv_cache_indexed_batched(
            compressed_kv, k_pe, kv_cache
        )
        k, v = self._project_kv(
            self._make_kv_b_proj(layer_id), compressed_kv, k_pe
        )
        q_fp8, k_fp8, v_fp8 = quantize_qkv_fp8(
            q, k, v
        )
        return self.prefill_wrapper.run(q_fp8, k_fp8, v_fp8).view(
            -1, self.num_heads, self.v_head_dim
        )


class KimiK3MlaPrefillImpl(MlaFlashInferPrefillImpl):
    prefill_op_type = KimiK3MlaPrefillOp

    def __init__(
        self, config, parallelism, weights, inputs, fmha_config, is_cuda_graph
    ):
        if is_cuda_graph:
            raise ValueError(
                "K3 ordinary prefill requires eager planning; verify and draft "
                "updates use the separate paged MLA graph implementation"
            )
        attention = config.getAttentionConfigs(parallelism.get_attn_tp_size())
        inputs.headwise_config = getattr(config, "headwise_config", None)
        # Keep expanded attention for both full and reused prefixes. Native
        # TokenSpeed arithmetic runs under RTP's cache/state planning.
        super().__init__(
            attention,
            inputs,
            weights.weights,
            None,  # K3 uses NoPE.
            fmha_config,
            quant_config=config.attention_projection_quant_config,
            max_seq_len=config.max_seq_len,
            is_cuda_graph=False,
            parallelism_config=parallelism,
            allow_absorb=False,
        )

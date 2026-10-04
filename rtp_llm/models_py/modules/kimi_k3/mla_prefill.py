"""K3 expanded prefill MLA, including bounded historical cache reads."""

import os
import json
import logging
import time
import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import MlaFlashInferPrefillOp
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_fp8_kernels import (
    gather_bf16_prefix_slice, gather_fp8_prefix, gather_fp8_prefix_slice,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_forward_plan import (
    FlashMLAForwardRoute, plan_flashmla_forward,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_qkv_fp8_quant import quantize_kv_fp8, quantize_qkv_fp8
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_fused_fp8_epilogue import fused_mla_fp8_epilogue
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import mla_fp8_kernels
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_prefix_chunk_plan import plan_prefix_chunks
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_state_merge import merge_mla_states_in_place
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_page_rr_cache import MlaPageRRCacheAdapter
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_page_rr_prefill_params import (
    build_mla_page_rr_prefill_params,
)
from rtp_llm.models_py.modules.kimi_k3.linear import KimiK3Bf16Linear
from rtp_llm.utils.model_weight import W
from rtp_llm.models_py.modules.kimi_k3.native_mla_prefill import KimiK3TokenspeedPrefill
from rtp_llm.ops import KvCacheDataType

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    MlaFlashInferPrefillImpl,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import check_attention_inputs


_BF16_FLASH_MLA_WORKSPACES = {}
_LOGGED_FP8_PREFIX_PLANS = set()


def _bf16_flash_mla_workspace(device):
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _BF16_FLASH_MLA_WORKSPACES:
        _BF16_FLASH_MLA_WORKSPACES[index] = torch.empty(
            32 * 1024 * 1024, dtype=torch.uint8, device=device
        )
    return _BF16_FLASH_MLA_WORKSPACES[index]


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
        self.reuse_cache_page_indice = getattr(mla_params, "reuse_cache_page_indice_d", None)
        self.page_rr_adapter = getattr(self, "page_rr_adapter", None)
        self._page_rr_attn_inputs = getattr(mla_params, "attn_inputs", None)
        self.qo_indptr = mla_params.qo_indptr_d
        self._bf16_kv_indptr = mla_params.prefill_ragged_kv_len_indptr_d
        self.batch_reuse_info_vec = mla_params.batch_reuse_info_vec_d
        self.total_kv_lens = int(mla_params.prefill_ragged_kv_len_indptr_h[-1])
        q_offsets = tuple(int(v) for v in mla_params.qo_indptr_h.tolist())
        kv_offsets = tuple(
            int(v) for v in mla_params.prefill_ragged_kv_len_indptr_h.tolist()
        )
        self._prefix_q_offsets = q_offsets
        self._prefix_lens = tuple(
            kv_offsets[i + 1] - kv_offsets[i] - (q_offsets[i + 1] - q_offsets[i])
            for i in range(len(q_offsets) - 1)
        )
        q_lens = tuple(q_offsets[i + 1] - q_offsets[i]
                       for i in range(len(q_offsets) - 1))
        budget_gib = float(os.environ.get(
            "RTP_MLA_PREFILL_EXPANDED_KV_BUDGET_GIB", "6.0"
        ))
        self._bf16_forward_plan = None
        if self.kv_cache_type == KvCacheDataType.FP8:
            self._prefix_plan = plan_prefix_chunks(
                q_lens,
                self._prefix_lens,
                page_tokens=(self.page_rr_adapter.page_tokens
                             if self.page_rr_adapter is not None
                             else self.token_per_block),
                heads=self.num_heads,
                qk_dim=self.qk_rope_head_dim + self.qk_nope_head_dim,
                v_dim=self.v_head_dim,
                operand_bytes=1,
                budget_gib=budget_gib,
            )
            if (os.environ.get("KIMI_K3_SMOKE_EVIDENCE") == "1"
                    and self.page_rr_adapter is not None
                    and self.page_rr_adapter.shard_rank == 0
                    and self._prefix_plan.chunked):
                launch_tokens = tuple(
                    segment.length for segment in self._prefix_plan.slices
                )
                evidence_key = (q_lens, self._prefix_lens,
                                self._prefix_plan.capacity_tokens, launch_tokens)
                if evidence_key not in _LOGGED_FP8_PREFIX_PLANS:
                    _LOGGED_FP8_PREFIX_PLANS.add(evidence_key)
                    logging.info("[K3_SMOKE_EVENT] %s", json.dumps({
                        "kind": "mla_prefix", "backend": "fp8_tokenspeed",
                        "route": "hybrid", "query_tokens": sum(q_lens),
                        "prefix_tokens": sum(self._prefix_lens),
                        "capacity_tokens": self._prefix_plan.capacity_tokens,
                        "launch_tokens": launch_tokens,
                    }, sort_keys=True))
        else:
            self._bf16_forward_plan = plan_flashmla_forward(
                q_lens, self._prefix_lens,
                prefix_chunk_alignment_tokens=(
                    self.page_rr_adapter.page_tokens
                    if self.page_rr_adapter is not None
                    else self.token_per_block
                ),
                expanded_kv_budget_gib=budget_gib,
                expanded_kv_bytes_per_token=(
                    self.num_heads
                    * (self.qk_nope_head_dim + self.qk_rope_head_dim
                       + self.v_head_dim) * 2
                ),
            )
            if (os.environ.get("KIMI_K3_SMOKE_EVIDENCE") == "1"
                    and self._bf16_forward_plan.route is FlashMLAForwardRoute.HYBRID):
                logging.info("[K3_SMOKE_EVENT] %s", json.dumps({
                    "kind": "mla_prefix", "backend": "bf16_flashmla",
                    "route": "hybrid", "query_tokens": sum(q_lens),
                    "prefix_tokens": sum(self._prefix_lens),
                    "capacity_tokens": self._bf16_forward_plan.capacity_tokens,
                    "launch_tokens": [launch.expanded_kv_tokens for launch
                                      in self._bf16_forward_plan.prefix_launches],
                }, sort_keys=True))
        self.block_table = (
            self._page_rr_attn_inputs.kv_cache_kernel_block_id_device
            if self.page_rr_adapter is not None
            else mla_params.page_indice_d.unsqueeze(0)
        )
        self.workspace_starts = torch.zeros(
            1, dtype=torch.int32, device=self.block_table.device
        )
        self.seq_lens = mla_params.prefill_ragged_kv_len_indptr_d[-1:]

    def _reuse_kv_cache_indexed_batched(self, compressed_kv, k_pe, kv_cache):
        if self.page_rr_adapter is not None and any(self._prefix_lens):
            if kv_cache is None:
                raise ValueError("MLA Page-RR prefix reuse requires a paged KV cache")
            cache = kv_cache.kv_cache_base.view(
                -1, self.page_rr_adapter.kernel_page_tokens,
                self.kv_lora_rank + self.qk_rope_head_dim,
            )
            restored = self.page_rr_adapter.read_prefix(
                cache, self.block_table, self._prefix_lens
            )
            if self.kv_cache_type == KvCacheDataType.FP8:
                restored = restored.to(torch.bfloat16)
            q_offsets = self._prefix_q_offsets
            latent = torch.cat([
                torch.cat((
                    restored.narrow(0, sum(self._prefix_lens[:i]), prefix)[:, :self.kv_lora_rank],
                    compressed_kv.narrow(0, q_offsets[i], q_offsets[i + 1] - q_offsets[i]),
                ), dim=0)
                for i, prefix in enumerate(self._prefix_lens)
            ], dim=0)
            suffix = torch.cat([
                torch.cat((
                    restored.narrow(0, sum(self._prefix_lens[:i]), prefix)[:, self.kv_lora_rank:],
                    k_pe.narrow(0, q_offsets[i], q_offsets[i + 1] - q_offsets[i]),
                ), dim=0)
                for i, prefix in enumerate(self._prefix_lens)
            ], dim=0)
            return latent, suffix
        if self.page_rr_adapter is not None:
            return compressed_kv, k_pe
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
        slot_mapping, cache_scale, cache_scale_value, cache_written,
    ):
        """Fuse ordinary FP8 MLA operand conversion with current KV insertion.

        Prefix reuse and a nonunit cache scale keep their existing paths.
        The caller performs the PD cache-store handoff after this method.
        """
        if (self.page_rr_adapter is not None
                or self.kv_cache_type != KvCacheDataType.FP8 or kv_cache is None
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
        cache_written()
        # The fused epilogue owns its FP8 Q/K/V outputs. Drop the BF16
        # projection and its views before TokenSpeed allocates workspace.
        del projected, k_nope, value
        return self.prefill_wrapper.run(q_fp8, k_fp8, v_fp8).view(
            -1, self.num_heads, self.v_head_dim
        )

    def _forward_fp8_chunked(self, q, compressed_kv, k_pe, kv_cache, layer_id):
        if kv_cache is None or (self.page_rr_adapter is None
                                and self.reuse_cache_page_indice is None):
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
        # The FP8 operands own separate storage. Release the expanded BF16
        # K/V before TokenSpeed allocates its attention workspace.
        del current_k, current_v
        output, output_lse = self.prefill_wrapper.run_partial(
            q_fp8, current_k_fp8, current_v_fp8,
            qo_indptr=self.qo_indptr,
            kv_indptr=self.qo_indptr,
            max_q=self.prefill_wrapper.max_q,
            max_k=self.prefill_wrapper.max_q,
            causal=True,
        )
        del current_k_fp8, current_v_fp8

        for segment in self._prefix_plan.slices:
            owner = segment.owner
            q_start = self._prefix_q_offsets[owner]
            q_len = self._prefix_q_offsets[owner + 1] - q_start
            if q_len == 0:
                continue
            if self.page_rr_adapter is not None:
                descriptor = self.page_rr_adapter.build_prefix_chunk_descriptor(
                    (owner,), (segment.start,), (segment.length,),
                    feature_width=self.kv_lora_rank + self.qk_rope_head_dim,
                )
                restored = self.page_rr_adapter.read_prefix_chunk(
                    cache, self.block_table, descriptor
                ).to(torch.bfloat16)
                # As in the feat Page-RR path, project from a contiguous
                # compressed-KV buffer. The sliced cache record has row stride
                # kv_lora_rank + qk_rope_head_dim and cannot enter FP8 quant.
                latent = torch.empty(
                    (segment.length, self.kv_lora_rank),
                    dtype=torch.bfloat16, device=q.device,
                )
                latent.copy_(restored[:, :self.kv_lora_rank])
                suffix = restored[:, self.kv_lora_rank:]
            else:
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
            del latent, suffix
            if self.page_rr_adapter is not None:
                del restored
            key_fp8, value_fp8 = quantize_kv_fp8(key, value)
            del key, value
            q_indptr = torch.tensor((0, q_len), dtype=torch.int32, device=q.device)
            kv_indptr = torch.tensor(
                (0, segment.length), dtype=torch.int32, device=q.device
            )
            partial, partial_lse = self.prefill_wrapper.run_partial(
                q_fp8.narrow(0, q_start, q_len), key_fp8, value_fp8,
                qo_indptr=q_indptr, kv_indptr=kv_indptr,
                max_q=q_len, max_k=segment.length, causal=False,
            )
            if os.environ.get("KIMI_K3_SMOKE_EVIDENCE") == "1":
                logging.info("[K3_SMOKE_EVENT] %s", json.dumps({
                    "kind": "mla_prefix_executed", "backend": "fp8_tokenspeed",
                    "layer_id": layer_id, "rank": (
                        self.page_rr_adapter.shard_rank
                        if self.page_rr_adapter is not None else 0
                    ), "owner": owner, "start": segment.start,
                    "length": segment.length, "query_tokens": q_len,
                    "time_ns": time.time_ns(),
                }, sort_keys=True))
            merge_mla_states_in_place(
                output.narrow(0, q_start, q_len),
                output_lse.narrow(0, q_start, q_len),
                partial, partial_lse,
            )
            del key_fp8, value_fp8
            del q_indptr, kv_indptr, partial, partial_lse
        return output.view(-1, self.num_heads, self.v_head_dim)

    def _run_bf16_partial(self, q, k, v, qo_indptr, kv_indptr, *,
                          max_q, max_k, causal):
        import flash_mla.cuda as flash_mla_cuda

        output = torch.empty(
            (q.shape[0], self.num_heads, self.v_head_dim),
            dtype=torch.bfloat16, device=q.device,
        )
        lse = torch.empty(
            (self.num_heads, q.shape[0]), dtype=torch.float32, device=q.device,
        ).transpose(0, 1)
        flash_mla_cuda.dense_prefill_fwd(
            _bf16_flash_mla_workspace(q.device),
            q, k, v, qo_indptr, kv_indptr, output, lse,
            int(causal), self.prefill_wrapper.scale,
            max_q, max_k, True,
        )
        return output, lse.contiguous()

    def _forward_bf16_chunked(self, q, compressed_kv, k_pe, kv_cache, layer_id):
        plan = self._bf16_forward_plan
        if kv_cache is None or (self.page_rr_adapter is None
                                and self.reuse_cache_page_indice is None):
            raise ValueError("chunked BF16 MLA requires a paged draft KV cache")
        projection = self._make_kv_b_proj(layer_id)
        splits = (self.qk_nope_head_dim, self.qk_rope_head_dim,
                  self.v_head_dim)
        if not projection.supports_skip_head_mid(compressed_kv, splits):
            raise RuntimeError(
                "chunked BF16 MLA requires packed KV-up projection: "
                f"input_shape={tuple(compressed_kv.shape)} "
                f"input_stride={tuple(compressed_kv.stride())} "
                f"input_dtype={compressed_kv.dtype} "
                f"weight_shape={tuple(projection.weight.shape)} "
                f"weight_stride={tuple(projection.weight.stride())} "
                f"weight_dtype={projection.weight.dtype} "
                f"weight_transpose_contiguous={projection.weight.T.is_contiguous()}"
            )
        cache = kv_cache.kv_cache_base.view(
            -1, self.token_per_block,
            self.kv_lora_rank + self.qk_rope_head_dim,
        )

        # Reuse one packed KV workspace for current-query and historical-prefix
        # projections, matching feat/k3_dev's FlashMLA forward workspace.
        packed_capacity = max(q.shape[0], plan.max_expanded_kv_tokens)
        packed_buffer = torch.empty(
            (packed_capacity, self.num_heads * sum(splits)),
            dtype=torch.bfloat16, device=q.device,
        )
        current_packed = packed_buffer.narrow(0, 0, compressed_kv.shape[0])
        projection.forward_skip_head_mid(
            compressed_kv, splits, output=current_packed
        )
        current_shaped = current_packed.view(
            compressed_kv.shape[0], self.num_heads, sum(splits)
        )
        current_shaped[..., splits[0]:splits[0] + splits[1]].copy_(
            k_pe.view(-1, 1, splits[1])
        )
        current_k = current_shaped[..., :-self.v_head_dim]
        current_v = current_shaped[..., -self.v_head_dim:]
        max_q = max(
            self._prefix_q_offsets[i + 1] - self._prefix_q_offsets[i]
            for i in range(len(self._prefix_lens))
        )
        output, output_lse = self._run_bf16_partial(
            q, current_k, current_v, self.qo_indptr, self.qo_indptr,
            max_q=max_q, max_k=max_q, causal=True,
        )
        del current_k, current_v, current_shaped, current_packed
        canonical = output.float() if plan.requires_fp32_accumulator else output

        capacity = plan.max_expanded_kv_tokens
        latent_buffer = torch.empty(
            (capacity, self.kv_lora_rank), dtype=torch.bfloat16,
            device=q.device,
        )
        rope_buffer = (
            torch.empty(
                (capacity, self.qk_rope_head_dim), dtype=torch.bfloat16,
                device=q.device,
            )
            if self.page_rr_adapter is None else None
        )
        for launch in plan.prefix_launches:
            for segment in launch.slices:
                owner = segment.request_idx
                q_start = self._prefix_q_offsets[owner]
                q_len = self._prefix_q_offsets[owner + 1] - q_start
                if not q_len:
                    continue
                length = segment.prefix_len
                latent = latent_buffer.narrow(0, 0, length)
                packed = packed_buffer.narrow(0, 0, length)
                shaped = packed.view(length, self.num_heads, sum(splits))
                if self.page_rr_adapter is not None:
                    descriptor = self.page_rr_adapter.build_prefix_chunk_descriptor(
                        (owner,), (segment.prefix_start,), (length,),
                        feature_width=self.kv_lora_rank + self.qk_rope_head_dim,
                    )
                    restored = self.page_rr_adapter.read_prefix_chunk(
                        cache, self.block_table, descriptor
                    )
                    latent.copy_(restored[:, :self.kv_lora_rank])
                    # skip-head-mid leaves the RoPE columns untouched. Fill
                    # them from the restored PageRR view before KV-up, so a
                    # separate full-prefix RoPE buffer never overlaps the
                    # restored pages and the packed projection workspace.
                    shaped[..., splits[0]:splits[0] + splits[1]].copy_(
                        restored[:, self.kv_lora_rank:].view(length, 1, splits[1])
                    )
                    del restored
                else:
                    rope = rope_buffer.narrow(0, 0, length)
                    gather_bf16_prefix_slice(
                        latent, rope, cache, self.reuse_cache_page_indice,
                        self.batch_reuse_info_vec, self.token_per_block,
                        owner=owner, start=segment.prefix_start,
                        prefix_len=self._prefix_lens[owner], scale=1.0,
                    )
                projection.forward_skip_head_mid(
                    latent, splits, output=packed
                )
                if rope_buffer is not None:
                    shaped[..., splits[0]:splits[0] + splits[1]].copy_(
                        rope[:, None, :]
                    )
                q_part = q.narrow(0, q_start, q_len)
                q_indptr = torch.tensor((0, q_len), dtype=torch.int32,
                                        device=q.device)
                kv_indptr = torch.tensor((0, length), dtype=torch.int32,
                                         device=q.device)
                partial, partial_lse = self._run_bf16_partial(
                    q_part, shaped[..., :-self.v_head_dim],
                    shaped[..., -self.v_head_dim:],
                    q_indptr, kv_indptr, max_q=q_len, max_k=length,
                    causal=False,
                )
                merge_mla_states_in_place(
                    canonical.narrow(0, q_start, q_len),
                    output_lse.narrow(0, q_start, q_len),
                    partial, partial_lse,
                )
        return canonical.to(torch.bfloat16)

    def _forward_bf16_full(self, q, compressed_kv, k_pe, kv_cache, layer_id):
        # feat/k3_dev uses FlashMLA for the BF16 draft Prefill path. Keep the
        # FP8 target on TokenSpeed and use the same BF16 backend for the draft.
        compressed_kv, k_pe = self._reuse_kv_cache_indexed_batched(
            compressed_kv, k_pe, kv_cache
        )
        key, value = self._project_kv(
            self._make_kv_b_proj(layer_id), compressed_kv, k_pe
        )
        del compressed_kv, k_pe
        output, _ = self._run_bf16_partial(
            q, key, value,
            self.qo_indptr,
            self._bf16_kv_indptr,
            max_q=self.prefill_wrapper.max_q,
            max_k=self.prefill_wrapper.max_k,
            causal=True,
        )
        return output

    def forward(self, q, compressed_kv, k_pe, kv_cache, layer_id):
        if self.kv_cache_type != KvCacheDataType.FP8:
            if self._bf16_forward_plan.route is FlashMLAForwardRoute.HYBRID:
                return self._forward_bf16_chunked(
                    q, compressed_kv, k_pe, kv_cache, layer_id
                )
            return self._forward_bf16_full(
                q, compressed_kv, k_pe, kv_cache, layer_id
            )
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
        del k, v
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
        self._page_rr_adapter = None
        if (parallelism.prefill_cp_config.kv_cache_sharded
                and parallelism.tp_size > 1):
            if parallelism.prefill_cp_config.is_enabled():
                raise ValueError("MLA Page-RR requires Prefill Query CP disabled")
            self._page_rr_adapter = MlaPageRRCacheAdapter(
                page_tokens=int(attention.tokens_per_block),
                kernel_page_tokens=int(attention.kernel_tokens_per_block),
                shard_size=int(parallelism.tp_size),
                shard_rank=int(parallelism.tp_rank),
            )
        # Keep expanded attention for both full and reused prefixes. RTP owns
        # cache/state planning; FP8 uses TokenSpeed and BF16 draft uses the
        # same FlashMLA dense path as feat/k3_dev.
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

    def create_params(self, attn_inputs):
        if self._page_rr_adapter is None:
            return super().create_params(attn_inputs)
        self.prepare(attn_inputs)

    def prepare(self, attn_inputs, forbid_realloc=False):
        if self._page_rr_adapter is None:
            return super().prepare(attn_inputs, forbid_realloc)
        if forbid_realloc:
            raise ValueError("MLA Page-RR Prefill uses eager metadata only")
        check_attention_inputs(attn_inputs)
        params = build_mla_page_rr_prefill_params(
            attn_inputs, self._page_rr_adapter.kernel_page_tokens
        )
        table = attn_inputs.kv_cache_kernel_block_id_device
        self._page_rr_adapter.validate_block_table_capacity(
            table, params.kv_lens_host
        )
        logical = int(getattr(attn_inputs, "logical_token_count", 0))
        physical = int(getattr(attn_inputs, "physical_token_count", 0))
        params.slot_mapping = self._page_rr_adapter.slot_mapping(
            params.positions_d,
            params.batch_indice_d,
            table,
            valid_token_count=logical if physical > logical else None,
        )
        if os.environ.get("KIMI_K3_SMOKE_EVIDENCE") == "1":
            owned = int((params.slot_mapping[:logical] >= 0).sum().item())
            padding_slots = int((params.slot_mapping[logical:] >= 0).sum().item())
            logging.info("[K3_SMOKE_EVENT] %s", json.dumps({
                "kind": "mla_page_rr_prefill", "rank": self._page_rr_adapter.shard_rank,
                "shards": self._page_rr_adapter.shard_size,
                "page_tokens": self._page_rr_adapter.page_tokens,
                "logical_tokens": logical, "physical_tokens": physical,
                "padding_rows": max(0, physical - logical),
                "owned_token_rows": owned, "padding_owned_slots": padding_slots,
                "time_ns": time.time_ns(),
            }, sort_keys=True))
        params.slot_mapping.record_stream(
            torch.cuda.current_stream(params.slot_mapping.device)
        )
        self.attn_inputs = attn_inputs
        self.fmha_params = params
        self.rope_params = params
        self.fmha_impl.page_rr_adapter = self._page_rr_adapter
        self.fmha_impl.plan(params)

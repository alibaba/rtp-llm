"""K3 MLA with expanded prefill and absorbed paged attention."""

import os

import torch

from rtp_llm.config.cuda_graph import CudaGraphSelectionMode
from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import MlaImplBase
from rtp_llm.models_py.triton_kernels.mla.lse_merge import merge_attention_states_in_place
from rtp_llm.models_py.triton_kernels.common.activation import (
    concat_and_cast_mha_k,
    quantize_fp8,
    quantize_kv_fp8,
    quantize_qkv_fp8,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
    MlaKVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.mla_prefill_plan import plan_prefix_chunks
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_page_rr_cache_gather import MlaPageRRCacheGather
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.utils.model_weight import W
from rtp_llm.ops import KvCacheDataType, RoleType, compute_ops
from rtp_llm.models_py.distributed.collective_torch import get_mla_dcp
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mla_metadata import (
    PageRRMlaMetadata,
)


_BF16_FLASH_MLA_WORKSPACES = {}
_PAGED_MLA_WORKSPACES = {}


def _bf16_flash_mla_workspace(device):
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _BF16_FLASH_MLA_WORKSPACES:
        _BF16_FLASH_MLA_WORKSPACES[index] = torch.empty(
            32 * 1024 * 1024, dtype=torch.uint8, device=device
        )
    return _BF16_FLASH_MLA_WORKSPACES[index]


class KimiK3MlaImpl(MlaImplBase):
    """Expanded prompt prefill; paged Decode, Target Verify and MTP updates."""

    cuda_graph_selection_modes = frozenset({
        CudaGraphSelectionMode.EAGER, CudaGraphSelectionMode.DECODE_GRAPH,
    })

    @staticmethod
    def _uses_paged_attention(inputs):
        return (inputs.is_target_verify or inputs.is_mtp_draft_update
                or not inputs.is_prefill)

    @staticmethod
    def support(attn_configs, inputs):
        return attn_configs.use_mla and not attn_configs.is_sparse

    @classmethod
    def support_parallelism_config(cls, parallelism):
        return (parallelism is not None and parallelism.tp_size > 1
                and parallelism.prefill_cp_config.kv_cache_sharded
                and not parallelism.prefill_cp_config.is_enabled())

    def __init__(
        self, attn_configs, inputs, weights, cos_sin_cache=None,
        fmha_config=None, quant_config=None, max_seq_len=0,
        is_cuda_graph=False, parallelism_config=None,
    ):
        self._paged = self._uses_paged_attention(inputs)
        if is_cuda_graph and not self._paged:
            raise ValueError("K3 ordinary prefill requires eager planning")
        parallelism = parallelism_config
        if not (parallelism.tp_size > 1
                and parallelism.prefill_cp_config.kv_cache_sharded):
            raise ValueError("K3 MLA requires Page-RR: TP > 1 and sharded KV cache")
        if parallelism.prefill_cp_config.is_enabled():
            raise ValueError("MLA Page-RR requires Query CP disabled")
        super().__init__(
            attn_configs, inputs, weights, None, fmha_config,
            quant_config=quant_config,
            max_seq_len=max_seq_len,
            is_cuda_graph=is_cuda_graph,
            parallelism_config=parallelism,
        )
        self.num_heads = attn_configs.head_num
        self.kv_lora_rank = attn_configs.kv_lora_rank
        self.qk_rope_head_dim = attn_configs.rope_head_dim
        self.qk_nope_head_dim = attn_configs.nope_head_dim
        self.v_head_dim = attn_configs.v_head_dim
        self.token_per_block = attn_configs.kernel_tokens_per_block
        self.softmax_extra_scale = attn_configs.softmax_extra_scale
        self.kv_cache_type = attn_configs.kv_cache_dtype
        self.fp8_compute = attn_configs.mla_fp8_compute
        self.q_replicated = parallelism.role_type == RoleType.DECODE
        self.page_rr_metadata = PageRRMlaMetadata(
            int(self.attn_configs.tokens_per_block), int(self.token_per_block),
            int(parallelism.tp_size), int(parallelism.tp_rank),
        )
        if self._paged:
            self._init_paged(inputs)
        else:
            self._init_prefill()
        self.write_cache_store_impl = common.create_write_cache_store_impl(inputs)
        self.prepare(inputs)

    def _init_prefill(self):
        if (self.num_heads <= 0
                or (self.qk_rope_head_dim + self.qk_nope_head_dim, self.v_head_dim)
                != (192, 128)):
            raise ValueError("K3 expanded MLA requires positive heads and QK/V dimensions 192/128")
        self.scale = (
            1.0 / (self.qk_rope_head_dim + self.qk_nope_head_dim) ** 0.5
        ) * self.softmax_extra_scale
        parallelism = self.parallelism_config
        self._replicated_query_head_start = (
            parallelism.tp_rank * self.num_heads if self.q_replicated else None
        )
        self._page_rr_adapter = MlaPageRRCacheGather(
            page_tokens=int(self.attn_configs.tokens_per_block),
            kernel_page_tokens=int(self.token_per_block),
            shard_size=int(parallelism.tp_size),
            shard_rank=int(parallelism.tp_rank),
        )
        self.prefill_wrapper = None
        if self.kv_cache_type == KvCacheDataType.FP8:
            from rtp_llm.models_py.modules.kimi_k3.tokenspeed_prefill import TokenspeedMlaPrefill

            self.prefill_wrapper = TokenspeedMlaPrefill(self.num_heads, self.scale)
        self.kv_cache_write_op = MlaKVCacheWriteOp(
            kv_cache_dtype=self.kv_cache_type, fp8_compute=self.fp8_compute,
        )

    def _init_paged(self, inputs):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.tokenspeed_mla_page_rr import (
            tokenspeed_mla_page_rr_decode,
        )

        self._run_paged_attention = tokenspeed_mla_page_rr_decode
        parallelism = self.parallelism_config
        self.query_heads = (
            self.num_heads * parallelism.tp_size
            if self.q_replicated else self.num_heads
        )
        self.batch = inputs.input_lengths.numel()
        self.tokens = inputs.physical_token_count
        if self.batch <= 0 or self.tokens <= 0:
            raise ValueError("K3 MLA requires nonempty physical rows")
        if inputs.is_target_verify and self.tokens % self.batch:
            raise ValueError("K3 target verify requires rectangular physical rows")
        device = next(w[W.mla_vc].device for w in self.weights if W.mla_vc in w)
        projection = next(w[W.mla_kc] for w in self.weights if W.mla_kc in w)
        if projection.shape[0] != self.query_heads:
            raise ValueError("K3 MLA KC heads do not match the configured query layout")
        if ((self.kv_lora_rank, self.qk_nope_head_dim, self.qk_rope_head_dim)
                != (512, 128, 64) or self.token_per_block not in (64, 128)):
            raise ValueError("Native MLA requires latent512/NoPE128/PE64 and page64 or page128")
        if self.fp8_compute and self.token_per_block != 128:
            raise ValueError("K3 FP8 MLA requires 128-token kernel pages")
        operand_dtype = torch.float8_e4m3fn if self.fp8_compute else torch.bfloat16
        workspace_key = (device, operand_dtype)
        if workspace_key not in _PAGED_MLA_WORKSPACES:
            _PAGED_MLA_WORKSPACES[workspace_key] = torch.empty(
                512 * 1024 * 1024, dtype=torch.uint8, device=device
            )
        self._paged_workspace = _PAGED_MLA_WORKSPACES[workspace_key]
        if self._paged_workspace.dtype != torch.uint8 or not self._paged_workspace.is_cuda:
            raise ValueError("K3 MLA workspace must be a CUDA uint8 tensor")
        self.scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5 * self.softmax_extra_scale
        # Prefill and paged attention share the unit-scale FP8 cache contract.
        self._q_scale = 1.0
        self._kv_scale_value = 1.0
        self._kv_scale = torch.full((), self._kv_scale_value, dtype=torch.float32, device=device)
        self._query_buffer = None
        self._absorbed_query_buffer = None
        if self.fp8_compute:
            shape = (self.tokens, self.query_heads, self.kv_lora_rank + self.qk_rope_head_dim)
            self._absorbed_query_buffer = torch.empty(shape, dtype=torch.bfloat16, device=device)
            self._query_buffer = torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)
        self.page_rr_communicator = get_mla_dcp(self.attn_configs, device, operand_dtype)

    def support_cuda_graph(self):
        return self._paged

    def prepare(self, inputs, forbid_realloc=False):
        if self._uses_paged_attention(inputs) != self._paged:
            raise ValueError("K3 MLA instance cannot switch between prefill and paged attention")
        if self._paged:
            self._prepare_paged(inputs, forbid_realloc)
        else:
            if forbid_realloc:
                raise ValueError("MLA Page-RR Prefill uses eager metadata only")
            self._prepare_prefill(inputs)

    def prepare_cuda_graph(self, inputs):
        if not self.support_cuda_graph():
            raise ValueError("MLA Page-RR Prefill uses eager metadata only")
        self.prepare(inputs, forbid_realloc=True)

    def forward(self, q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices=None):
        if topk_indices is not None:
            raise ValueError("K3 Page-RR MLA requires dense attention")
        if self._paged:
            if kv_cache is None:
                raise ValueError("K3 MLA requires a dense paged KV cache")
            query = self._absorb_query_and_write_cache(
                q, compressed_kv, k_pe, kv_cache.kv_cache_base,
                self.page_rr_metadata.slot_mapping, self.weights[layer_id][W.mla_kc],
            )
        else:
            query = q
            if self._replicated_query_head_start is not None:
                query = q.narrow(1, self._replicated_query_head_start, self.num_heads).contiguous()
            self.kv_cache_write_op.forward(
                compressed_kv, k_pe, kv_cache, self.fmha_params
            )
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        if self._paged:
            return self._forward_paged(query, kv_cache, layer_id)
        return self._forward_prefill(query, compressed_kv, k_pe, kv_cache, layer_id)

    def _prepare_prefill(self, attn_inputs):
        self.page_rr_metadata.prepare_prefill(attn_inputs)
        self.attn_inputs = attn_inputs
        self.fmha_params = self.page_rr_metadata
        self._plan_prefill(self.fmha_params)

    def _prepare_paged(self, inputs, forbid_realloc=False):
        if (
            inputs.input_lengths.numel() != self.batch
            or inputs.physical_token_count != self.tokens
        ):
            raise ValueError("K3 MLA replay requires the captured physical shape")
        self.attn_inputs = inputs
        self.page_rr_metadata.prepare_decode(inputs, forbid_realloc)

    def _plan_prefill(self, mla_params):
        self.qo_indptr = mla_params.qo_indptr_d
        self._bf16_kv_indptr = mla_params.prefill_ragged_kv_len_indptr_d
        self._prefix_q_offsets = mla_params.q_offsets_host
        self._prefix_lens = tuple(mla_params.prefix_lens_host)
        q_lens = tuple(mla_params.q_lens_host)
        budget_gib = float(os.environ.get(
            "RTP_MLA_PREFILL_EXPANDED_KV_BUDGET_GIB", "6.0"
        ))
        fp8 = self.kv_cache_type == KvCacheDataType.FP8
        self._prefix_plan = plan_prefix_chunks(
            q_lens,
            self._prefix_lens,
            page_tokens=self._page_rr_adapter.page_tokens,
            # FP8 keeps both its BF16 producer and the FP8 attention operands.
            bytes_per_token=(
                self.num_heads
                * (self.qk_nope_head_dim + self.qk_rope_head_dim + self.v_head_dim)
                * (3 if fp8 else 2)
            ),
            budget_gib=budget_gib,
            pack_requests=not fp8,
        )
        self.block_table = mla_params.attn_inputs.kv_cache_kernel_block_id_device

    def _reuse_kv_cache_indexed_batched(self, compressed_kv, k_pe, kv_cache):
        if any(self._prefix_lens):
            if kv_cache is None:
                raise ValueError("MLA Page-RR prefix reuse requires a paged KV cache")
            cache = kv_cache.kv_cache_base.view(
                -1, self._page_rr_adapter.kernel_page_tokens,
                self.kv_lora_rank + self.qk_rope_head_dim,
            )
            restored = self._page_rr_adapter.read_prefix(
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
        return compressed_kv, k_pe

    def _make_kv_b_proj(self, layer_id):
        weight = self.weights[layer_id][W.mla_kv_b_w]
        if weight.is_cuda and weight.dtype == torch.bfloat16:
            return CudaF16Linear(weight)
        from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory

        return LinearFactory.create_linear_from_weights(
            self.weights[layer_id], W.mla_kv_b_w, W.mla_kv_b_s,
            None, self.quant_config,
        )

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
        key = concat_and_cast_mha_k(
            projected[..., :left], k_pe.view(-1, 1, middle),
            self.num_heads, left, middle,
        )
        return key, projected[..., left:]

    def _forward_fp8_chunked(self, q, compressed_kv, k_pe, kv_cache, layer_id):
        if kv_cache is None:
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
            max_q=self.fmha_params.max_q,
            max_k=self.fmha_params.max_q,
            causal=True,
        )
        del current_k_fp8, current_v_fp8

        for segment in self._prefix_plan.slices:
            owner = segment.owner
            q_start = self._prefix_q_offsets[owner]
            q_len = self._prefix_q_offsets[owner + 1] - q_start
            if q_len == 0:
                continue
            descriptor = self._page_rr_adapter.build_prefix_chunk_descriptor(
                (owner,), (segment.start,), (segment.length,),
                feature_width=self.kv_lora_rank + self.qk_rope_head_dim,
            )
            restored = self._page_rr_adapter.read_prefix_chunk(
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
            key, value = self._project_kv(projection, latent, suffix)
            del latent, suffix
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
            merge_attention_states_in_place(
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
            int(causal), self.scale,
            max_q, max_k, True,
        )
        return output, lse.contiguous()

    def _forward_bf16_chunked(self, q, compressed_kv, k_pe, kv_cache, layer_id):
        plan = self._prefix_plan
        if kv_cache is None:
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
        max_q = self.fmha_params.max_q
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
        for segment in plan.slices:
            owner = segment.owner
            q_start = self._prefix_q_offsets[owner]
            q_len = self._prefix_q_offsets[owner + 1] - q_start
            if not q_len:
                continue
            length = segment.length
            latent = latent_buffer.narrow(0, 0, length)
            packed = packed_buffer.narrow(0, 0, length)
            shaped = packed.view(length, self.num_heads, sum(splits))
            descriptor = self._page_rr_adapter.build_prefix_chunk_descriptor(
                (owner,), (segment.start,), (length,),
                feature_width=self.kv_lora_rank + self.qk_rope_head_dim,
            )
            restored = self._page_rr_adapter.read_prefix_chunk(
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
            projection.forward_skip_head_mid(
                latent, splits, output=packed
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
            merge_attention_states_in_place(
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
            max_q=self.fmha_params.max_q,
            max_k=self.fmha_params.max_k,
            causal=True,
        )
        return output

    def _forward_prefill(self, q, compressed_kv, k_pe, kv_cache, layer_id):
        if self.kv_cache_type != KvCacheDataType.FP8:
            if self._prefix_plan.chunked:
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
        return self.prefill_wrapper.run(q_fp8, k_fp8, v_fp8, self.fmha_params).view(
            -1, self.num_heads, self.v_head_dim
        )

    def _absorb_query_and_write_cache(self, q, kv, k_pe, cache, slot_mapping, k_weight):
        """Project Q into latent space and insert this step's compressed K/V."""
        if any(t.dtype != torch.bfloat16 for t in (q, kv, k_pe, k_weight)):
            raise ValueError("MLA projection and absorption inputs require BF16")
        cache_dtype = torch.float8_e4m3fn if self.fp8_compute else torch.bfloat16
        if cache.dtype != cache_dtype:
            raise ValueError(f"MLA cache requires {cache_dtype} storage")
        if q.shape[1:] != (self.query_heads, self.qk_nope_head_dim + self.qk_rope_head_dim):
            raise ValueError("Unexpected K3 MLA query layout")
        if slot_mapping.numel() != q.shape[0] or slot_mapping.dtype != torch.int64:
            raise ValueError("K3 MLA requires one int64 cache slot per physical query")
        if self.fp8_compute:
            if (self._query_buffer is None or self._absorbed_query_buffer is None
                    or q.shape[0] > self._query_buffer.shape[0]):
                raise ValueError("FP8 MLA query exceeds reserved graph buffer")
            absorbed_q = self._absorbed_query_buffer[:q.shape[0]]
            # As in feat/k3_dev's TokenSpeed path, write both parts into the
            # graph-stable BF16 query before quantization. This avoids a cat
            # allocation and its full-query copy on every Verify replay.
            absorbed_q[..., self.kv_lora_rank:].copy_(q[..., self.qk_nope_head_dim:])
            torch.bmm(
                q[..., :self.qk_nope_head_dim].transpose(0, 1), k_weight,
                out=absorbed_q[..., :self.kv_lora_rank].transpose(0, 1),
            )
            compute_ops.concat_and_cache_mla(
                kv, k_pe, cache.view(-1, self.token_per_block, self.kv_lora_rank + self.qk_rope_head_dim),
                slot_mapping, "fp8", self._kv_scale,
            )
            return quantize_fp8(
                absorbed_q, self._q_scale, self._query_buffer[:q.shape[0]],
            )
        latent_q = torch.bmm(q[..., :self.qk_nope_head_dim].transpose(0, 1), k_weight)
        absorbed_q = torch.empty(
            (q.shape[0], self.query_heads, self.kv_lora_rank + self.qk_rope_head_dim),
            dtype=q.dtype, device=q.device,
        )
        compute_ops.mla_decode_q_concat_cache_insert(
            latent_q.transpose(0, 1), q[..., self.qk_nope_head_dim:], kv, k_pe,
            absorbed_q, cache.view(-1, self.token_per_block, self.kv_lora_rank + self.qk_rope_head_dim),
            slot_mapping,
        )
        return absorbed_q

    def _forward_paged(self, query, kv_cache, layer_id):
        weights = self.weights[layer_id]
        local_query = query.transpose(0, 1).contiguous()
        gathered = (
            local_query
            if self.q_replicated
            else self.page_rr_communicator.query_gather(local_query)
        )
        batch, queries = self.page_rr_metadata.local_causal_lens.shape
        dimension = self.kv_lora_rank + self.qk_rope_head_dim
        partial, lse = self._run_paged_attention(
            # TokenSpeed accepts the gathered head-major strides, as in
            # feat/k3_dev. Avoid copying the entire FP8 query after AG.
            gathered.transpose(0, 1).view(
                batch, queries, self.page_rr_communicator.heads, dimension
            ),
            kv_cache.kv_cache_base.view(-1, self.token_per_block, dimension),
            self._paged_workspace.view(torch.int8),
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.page_rr_metadata.query_block_tables,
            self.page_rr_metadata.local_causal_lens,
            self.page_rr_metadata.query_block_tables.shape[1]
            * self.token_per_block,
            self.scale
            * (
                self._q_scale * self._kv_scale_value
                if self.fp8_compute
                else 1.0
            ),
            output_scale=self._kv_scale_value if self.fp8_compute else 1.0,
            normalize_empty=False,
        )
        merged = self.page_rr_communicator.combine(
            partial.view(
                self.tokens,
                self.page_rr_communicator.heads,
                self.kv_lora_rank,
            ),
            lse.view(self.tokens, self.page_rr_communicator.heads),
            self.page_rr_metadata.local_causal_lens,
        )
        output = torch.empty(
            (self.tokens, self.num_heads, weights[W.mla_vc].shape[-1]),
            dtype=torch.bfloat16, device=query.device,
        )
        torch.bmm(merged, weights[W.mla_vc], out=output.transpose(0, 1))
        return output

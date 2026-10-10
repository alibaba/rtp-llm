"""Gemma4 text-stack python model descriptor (milestone-1 correctness path).

Reference math: HF transformers ``modeling_gemma4.py`` (Gemma4-26B-A4B-it):

- 30 decoder layers: 25 sliding-attention layers (16Q x 8KV x 256, RoPE
  theta=1e4 over the full head dim, sliding window 1024) and 5 full-attention
  layers (16Q x 2KV x 512, K==V, RoPE theta=1e6 restricted to the first 25%
  of the frequency slots — HF "proportional" partial rope).
- Sandwich residual structure with a parallel dense MLP (gelu_tanh) and a
  routed MoE block (128 experts, top-8); the router consumes the raw residual
  stream. Every layer output is scaled by a per-layer scalar.
- Attention softmax scale is 1.0 (the effective scale is baked into the
  k_norm weight). QK-norm (weighted) and a weightless V-norm are applied
  *before* RoPE. All RMSNorms multiply the weight directly (no +1).

Weight-name contract (frozen with the weight loader writer): the new Gemma4
names may not exist in ``utils/model_weight.py`` yet while the loader work
lands in parallel, so they are resolved through :func:`_wattr` with the agreed
fallback strings.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.model_desc.block_map import get_attention_inputs_value
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    Embedding,
    FMHAImplBase,
    FusedMoeFactory,
    LinearFactory,
    MultimodalEmbeddingInjector,
)
from rtp_llm.models_py.modules.factory.attention import common as attn_common
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.gemma4.core import (
    _W_LAYER_SCALAR,
    _W_MOE_ROUTER_EXPERT_SCALE,
    _W_MOE_ROUTER_SCALE,
    _W_POST_FFN1_LN_GAMMA,
    _W_POST_FFN2_LN_GAMMA,
    _W_PRE_FFN2_LN_GAMMA,
    _W_PRE_FFN_LN_GAMMA,
    GEMMA4_TAG_FULL,
    GEMMA4_TAG_SWA,
    Gemma4DenseMLP,
    Gemma4Experts,
    Gemma4LayerGeometry,
    Gemma4RMSNorm,
    Gemma4RopeTable,
    Gemma4Router,
    _layer_type_at,
    _proportional_inv_freq,
    _rotate_half,
    _wattr,
    apply_gemma4_rope,
    build_gemma4_layer_geometry,
    gemma4_add_bf16,
    gemma4_add_norm,
    gemma4_geglu_tanh,
    gemma4_norm_add,
    gemma4_residual_scale,
    gemma4_rms_norm,
)
from rtp_llm.ops import HybridAttentionType, MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyModelInputs,
    PyModelOutputs,
    rtp_llm_ops,
)
from rtp_llm.utils.model_weight import W
from torch import nn
from torch.nn import functional as F


def _host_i32(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None or t.numel() == 0:
        return t
    return t.cpu() if t.is_cuda else t


def _device_or(
    device_tensor: Optional[torch.Tensor], host_tensor: Optional[torch.Tensor]
):
    if device_tensor is not None and device_tensor.numel() > 0:
        return device_tensor
    return host_tensor


class Gemma4TorchFMHAImpl(FMHAImplBase):
    """Torch SDPA reference implementation with paged KV cache support.

    Custom contract (differs from the stock ``FMHAImplBase.forward(qkv, ...)``):
    the caller (Gemma4Attention) applies QK/V norms and RoPE itself and calls
    ``forward(q, k, v, kv_cache)`` with post-RoPE tensors. RoPE tables and
    token positions are served from this impl (``positions``/``rope_cos_sin``).

    KV write mirrors ``KVCacheWriteOp`` via ``flashinfer.page
    .append_paged_kv_cache`` (HND layout). Attention is computed per request
    with scale=1.0, fp32 softmax, and a position-based causal (+ optional
    sliding-window) mask. Not CUDA-graph capturable; not performance tuned.
    """

    accepts_fmha_config = False

    def __init__(
        self,
        model_config: ModelConfig,
        geometry: Gemma4LayerGeometry,
        attn_inputs: Any,
        page_size: int,
        parallelism_config: Optional[ParallelismConfig] = None,
    ):
        self.geometry = geometry
        self.attn_inputs = attn_inputs
        self.page_size = page_size
        self.sdpa_causal_limit = (
            int(model_config.attn_config.sliding_window)
            if model_config is not None
            else 1024
        )
        self.max_attention_matrix_elements = 8 * 1024 * 1024
        self.max_cached_attention_mask_elements = 64 * 1024 * 1024
        self.attention_query_chunk_size = 512
        self.rope_table = Gemma4RopeTable(
            geometry.head_dim,
            geometry.rope_theta,
            geometry.rope_partial_rotary_factor,
        )
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self._dense_allowed_masks: Dict[tuple, torch.Tensor] = {}
        self._dense_allowed_mask_elements = 0
        self._fill_params(attn_inputs)
        self.write_cache_store_impl = attn_common.create_write_cache_store_impl(
            attn_inputs
        )
        # full-layer graph path: capture-time row width cache (fixed during
        # the eager warmup forward that precedes capture)
        self._full_row_pages = None
        # Capture inputs are separate objects; retain their mode for replay.
        self._graph_mode = bool(getattr(attn_inputs, "is_cuda_graph", False))
        self.vision_group_ids: Optional[torch.Tensor] = None

    # -- parameter planning (mirrors PyFlashinferPrefillPagedAttnOp.prepare) --

    def _fill_params(self, attn_inputs: Any) -> None:
        self._dense_allowed_masks.clear()
        self._dense_allowed_mask_elements = 0
        block_id_host = attn_inputs.kv_cache_kernel_block_id
        if block_id_host is None or block_id_host.numel() == 0:
            block_id_host = attn_inputs.kv_cache_kernel_block_id_device
        if block_id_host is None:
            # Cacheless/warmup inputs carry no block table; the params object
            # accepts an empty int32 tensor for a zero-block layout.
            block_id_host = torch.zeros(0, dtype=torch.int32)
        if attn_inputs.input_lengths.is_cuda:
            self.fmha_params.fill_params_mha_device(
                _device_or(
                    attn_inputs.prefix_lengths_device, attn_inputs.prefix_lengths
                ),
                attn_inputs.sequence_lengths,
                _device_or(attn_inputs.input_lengths_device, attn_inputs.input_lengths),
                _device_or(
                    attn_inputs.kv_cache_kernel_block_id_device,
                    attn_inputs.kv_cache_kernel_block_id,
                ),
                self.page_size,
            )
        else:
            self.fmha_params.fill_params(
                _host_i32(attn_inputs.prefix_lengths),
                _host_i32(attn_inputs.sequence_lengths),
                _host_i32(attn_inputs.input_lengths),
                _host_i32(block_id_host),
                self.page_size,
            )

    def prepare_cuda_graph(self, attn_inputs: Any) -> None:
        """CUDA-graph replay prepare: refresh the params buffers in place.

        The decode graph runner calls this with the current step's attention
        inputs before each replay (cuda_graph_runner.cc callPrepareCudaGraph).
        ``fill_params*`` refreshes the pre-allocated device buffers in place;
        a forbidden reallocation raises inside the C++ helper. Attention
        planning stays lazy in forward() (the SWA impl plans per forward on
        the refreshed params; the reference impl reads them directly).
        """
        self._fill_params(attn_inputs)

    @staticmethod
    def support(attn_configs, attn_inputs) -> bool:
        return True

    # -- rope service for Gemma4Attention --

    def positions(self, num_tokens: int) -> torch.Tensor:
        return self.fmha_params.positions_d.narrow(0, 0, num_tokens)

    def rope_cos_sin(self, positions: torch.Tensor):
        return self.rope_table.cos_sin(positions)

    def rope_cos_sin_bf16(self, positions: torch.Tensor):
        return self.rope_table.cos_sin_bf16(positions)

    # -- forward --

    def _request_layout(self) -> tuple[list[int], list[int]]:
        # Prefill: cu_seqlens_device carries the per-request NEW-token prefix
        # sums (input_lengths holds the same new-token counts). Decode:
        # input_lengths is cumulative per request while each request only
        # contributes one q token, so derive the layout from the batch size
        # instead (positions/1-token-per-request).
        if self.attn_inputs.is_prefill:
            lengths = self.attn_inputs.input_lengths
        else:
            lengths = torch.ones(
                self.attn_inputs.sequence_lengths.numel(),
                dtype=torch.int32,
                device=(
                    self.attn_inputs.sequence_lengths.device
                    if self.attn_inputs.sequence_lengths.is_cuda
                    else "cpu"
                ),
            )
        lengths_host = lengths.cpu() if lengths.is_cuda else lengths
        lengths_list = lengths_host.tolist()
        starts = [0]
        for length in lengths_list:
            starts.append(starts[-1] + int(length))
        return lengths_list, starts

    def _paged_views(self, kv_cache: LayerKVCache):
        paged = kv_cache.kv_cache_base
        if paged.dim() == 2:
            paged = attn_common.reshape_paged_kv_cache(
                paged,
                self.geometry.kv_head_num,
                self.page_size,
                self.geometry.head_dim,
            )
        if paged.dim() != 5:
            raise ValueError(f"unexpected kv cache dim: {paged.dim()}")
        if paged.shape[2] != self.geometry.kv_head_num:
            raise ValueError(
                "kv cache kv_head_num mismatch: got "
                f"{paged.shape[2]}, expected {self.geometry.kv_head_num}"
            )
        if paged.shape[4] != self.geometry.head_dim:
            raise ValueError(
                "kv cache head_dim mismatch: got "
                f"{paged.shape[4]}, expected {self.geometry.head_dim}"
            )
        if paged.shape[3] != self.page_size:
            raise ValueError(
                f"kv cache page size mismatch: got {paged.shape[3]}, "
                f"expected {self.page_size}"
            )
        return paged[:, 0], paged[:, 1]

    def _write_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: LayerKVCache,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> None:
        import flashinfer.page as page

        if key.dtype != k_cache.dtype:
            raise ValueError(
                f"key dtype {key.dtype} must match K cache dtype {k_cache.dtype}"
            )
        if value.dtype != v_cache.dtype:
            raise ValueError(
                f"value dtype {value.dtype} must match V cache dtype {v_cache.dtype}"
            )
        nnz = key.size(0)
        batch_indices = self.fmha_params.batch_indice_d.narrow(0, 0, nnz)
        positions = self.fmha_params.positions_d.narrow(0, 0, nnz)

        if (
            self.geometry.tag == GEMMA4_TAG_SWA
            and key.dtype == torch.bfloat16
            and key.is_contiguous()
            and value.is_contiguous()
            and batch_indices.dtype == torch.int32
            and positions.dtype == torch.int32
            and self.fmha_params.page_indice_d.dtype == torch.int32
            and self.fmha_params.decode_page_indptr_d.dtype == torch.int32
            and torch.version.hip is None
        ):
            rtp_llm_ops.gemma4_append_swa_kv_cache_bf16(
                key,
                value,
                batch_indices,
                positions,
                k_cache,
                v_cache,
                self.fmha_params.page_indice_d,
                self.fmha_params.decode_page_indptr_d,
                self.page_size,
            )
            return

        def _append(k_t, v_t, bi_t, pos_t):
            page.append_paged_kv_cache(
                k_t,
                v_t,
                bi_t,
                pos_t,
                (k_cache, v_cache),
                self.fmha_params.page_indice_d,
                self.fmha_params.decode_page_indptr_d,
                self.fmha_params.paged_kv_last_page_len_d,
                "HND",
            )

        # Sparse (SWA) block tables: positions mapping to NULL pages cannot be
        # stored (the append kernel would compute an out-of-range cache
        # address). Append only the tokens whose position lands on a resident
        # block; skipped tokens are outside the sliding window by construction.
        # Validity is computed device-side (one tiny reduction sync on the
        # all-valid fast path): the previous host implementation transferred 4
        # tensors and ran a per-token Python loop per layer per step (25
        # sliding layers), which dominated the attention-side step time.
        # CUDA-graph capture forbids the host read in bool(valid.all()), so
        # under capture (or when explicitly forced) the dense table this
        # run's tail policy guarantees is trusted directly: append all
        # tokens, no validity branch.
        if torch.cuda.is_current_stream_capturing():
            _append(key, value, batch_indices, positions)
            return
        if self.geometry.tag == GEMMA4_TAG_FULL:
            _append(key, value, batch_indices, positions)
            attn_common.apply_write_cache_store(
                self.write_cache_store_impl, self.attn_inputs, kv_cache
            )
            return
        pages_all = self.fmha_params.page_indice_d
        indptr_all = self.fmha_params.decode_page_indptr_d
        num_pages = k_cache.shape[0]
        n_indptr = indptr_all.numel()
        n_pages_all = pages_all.numel()
        if n_indptr == 0 or n_pages_all == 0:
            return
        bi_long = batch_indices.long()
        b_ok = bi_long <= (n_indptr - 2)
        b_clamped = bi_long.clamp(0, max(n_indptr - 1, 0))
        slot = torch.gather(indptr_all, 0, b_clamped) + torch.div(
            positions.long(), self.page_size, rounding_mode="floor"
        )
        s_ok = (slot >= 0) & (slot < n_pages_all)
        page_id = pages_all[slot.clamp(0, n_pages_all - 1)]
        valid = b_ok & s_ok & (page_id >= 0) & (page_id < num_pages)
        if bool(valid.all()):
            _append(key, value, batch_indices, positions)
        else:
            idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
            if idx.numel() > 0:
                _append(key[idx], value[idx], batch_indices[idx], positions[idx])
        attn_common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

    def _gather_request(
        self,
        index: int,
        page_indptr_host: list[int],
        last_page_len_host: list[int],
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        start_pos: int = 0,
        end_pos: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pages = self.fmha_params.page_indice_d[
            page_indptr_host[index] : page_indptr_host[index + 1]
        ]
        kv_heads = k_cache.shape[1]
        head_dim = k_cache.shape[3]
        kvlen = (page_indptr_host[index + 1] - page_indptr_host[index] - 1) * (
            self.page_size
        ) + last_page_len_host[index]
        range_start = max(0, min(start_pos, kvlen))
        range_end = kvlen if end_pos is None else max(range_start, min(end_pos, kvlen))
        first_page = range_start // self.page_size
        page_end = (range_end + self.page_size - 1) // self.page_size
        pages = pages[first_page:page_end]
        first_offset = range_start - first_page * self.page_size
        token_count = range_end - range_start
        if (
            k_cache.is_cuda
            and k_cache.dtype == torch.bfloat16
            and v_cache.dtype == torch.bfloat16
            and pages.dtype == torch.int32
            and pages.is_contiguous()
            and head_dim % 8 == 0
            and torch.version.hip is None
        ):
            return rtp_llm_ops.gemma4_gather_paged_kv_bf16(
                k_cache,
                v_cache,
                pages,
                first_offset,
                token_count,
                self.page_size,
            )
        safe_pages = pages.clamp(0, k_cache.shape[0] - 1)
        keys = (
            k_cache[safe_pages]
            .permute(0, 2, 1, 3)
            .reshape(-1, kv_heads, head_dim)[first_offset : first_offset + token_count]
        )
        values = (
            v_cache[safe_pages]
            .permute(0, 2, 1, 3)
            .reshape(-1, kv_heads, head_dim)[first_offset : first_offset + token_count]
        )
        valid = (
            ((pages >= 0) & (pages < k_cache.shape[0]))[:, None]
            .expand(-1, self.page_size)
            .reshape(-1)[first_offset : first_offset + token_count]
        )
        if not bool(valid.all()):
            keys = torch.where(valid[:, None, None], keys, torch.zeros_like(keys))
            values = torch.where(valid[:, None, None], values, torch.zeros_like(values))
        return keys, values, valid

    def _forward_prefill_current_kv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: LayerKVCache,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        lengths, starts = self._request_layout()
        prefix_lengths = self.attn_inputs.prefix_lengths
        if prefix_lengths is None or prefix_lengths.numel() == 0:
            prefix_lengths_list = [0] * len(lengths)
        else:
            prefix_lengths_host = (
                prefix_lengths.cpu() if prefix_lengths.is_cuda else prefix_lengths
            )
            prefix_lengths_list = [int(value) for value in prefix_lengths_host.tolist()]
        if len(prefix_lengths_list) != len(lengths):
            raise RuntimeError(
                f"Gemma4 prefill prefix count {len(prefix_lengths_list)} "
                f"does not match batch {len(lengths)}"
            )

        positions = self.fmha_params.positions_d.narrow(0, 0, q.size(0))
        needs_prefix = any(prefix_length > 0 for prefix_length in prefix_lengths_list)
        page_indptr_host = last_page_len_host = None
        if needs_prefix:
            page_indptr_host = self.fmha_params.decode_page_indptr_d.cpu().tolist()
            last_page_len_host = (
                self.fmha_params.paged_kv_last_page_len_d.cpu().tolist()
            )

        outputs = []
        for index, prefix_length in enumerate(prefix_lengths_list):
            start, end = starts[index], starts[index + 1]
            current_keys = k[start:end]
            current_values = v[start:end]
            kv_offset = int(positions[start].item())
            if prefix_length < 0 or kv_offset != prefix_length:
                raise RuntimeError(
                    f"Gemma4 prefill request {index} has prefix_length "
                    f"{prefix_length} but first position {kv_offset}"
                )

            if prefix_length > 0:
                prefix_start = (
                    max(0, prefix_length - self.geometry.sliding_window + 1)
                    if self.geometry.sliding_window > 0
                    else 0
                )
                cached_keys, cached_values, cached_valid = self._gather_request(
                    index,
                    page_indptr_host,
                    last_page_len_host,
                    k_cache,
                    v_cache,
                    start_pos=prefix_start,
                    end_pos=prefix_length,
                )
                expected_prefix_tokens = prefix_length - prefix_start
                if cached_keys.size(0) != expected_prefix_tokens:
                    raise RuntimeError(
                        f"Gemma4 prefill request {index} cache length "
                        f"{cached_keys.size(0)} does not match required prefix "
                        f"length {expected_prefix_tokens}"
                    )
                if not bool(cached_valid.all()):
                    raise RuntimeError(
                        f"Gemma4 prefill request {index} is missing required "
                        f"prefix KV in [{prefix_start}, {prefix_length})"
                    )
                current_keys = torch.cat((cached_keys, current_keys), dim=0)
                current_values = torch.cat((cached_values, current_values), dim=0)
                kv_offset = prefix_start

            query_group_ids = (
                self.vision_group_ids[start:end]
                if self.vision_group_ids is not None
                else None
            )
            outputs.append(
                self._attend_request(
                    q[start:end],
                    current_keys,
                    current_values,
                    positions[start:end],
                    kv_offset,
                    query_group_ids=query_group_ids,
                )
            )

        self._write_cache(k, v, kv_cache, k_cache, v_cache)
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)

    def _attend_request_chunked(
        self,
        q: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        q_positions: torch.Tensor,
        kv_offset: Any,
        kv_valid: Optional[torch.Tensor],
        query_group_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        window = self.geometry.sliding_window
        if window > 0:
            query_chunk_size = self.attention_query_chunk_size
        else:
            query_chunk_size = max(
                1, self.max_attention_matrix_elements // max(keys.size(0), 1)
            )
        use_exact_causal_softmax = (
            self.geometry.tag == GEMMA4_TAG_FULL
            and q.dtype == torch.bfloat16
            and q.is_cuda
            and torch.version.hip is None
            and q.size(0) == 8192
            and keys.size(0) == 8192
            and query_chunk_size == 1024
            and kv_valid is None
            and query_group_ids is None
        )
        use_exact_window_softmax = (
            self.geometry.tag == GEMMA4_TAG_SWA
            and window == 1024
            and q.dtype == torch.bfloat16
            and q.is_cuda
            and torch.version.hip is None
            and q.size(0) == 8192
            and keys.size(0) == 8192
            and query_chunk_size == 512
            and kv_valid is None
            and query_group_ids is None
        )
        use_exact_masked_softmax = use_exact_causal_softmax or use_exact_window_softmax
        kv_offset_value = int(
            kv_offset.item() if isinstance(kv_offset, torch.Tensor) else kv_offset
        )
        allowed = None
        if kv_valid is None and not use_exact_masked_softmax:
            mask_elements = q.size(0) * keys.size(0)
            cached_mask_elements = getattr(self, "_dense_allowed_mask_elements", 0)
            cache_key = (
                q.size(0),
                keys.size(0),
                kv_offset_value,
                window,
            )
            mask_cache = getattr(self, "_dense_allowed_masks", None)
            if mask_cache is None:
                mask_cache = {}
                self._dense_allowed_masks = mask_cache
            allowed = mask_cache.get(cache_key)
            if (
                allowed is None
                and mask_elements
                <= getattr(self, "max_cached_attention_mask_elements", 0)
                - cached_mask_elements
            ):
                kv_positions = (
                    torch.arange(keys.size(0), device=q.device) + kv_offset_value
                )
                allowed = kv_positions[None, :] <= q_positions[:, None]
                if window:
                    allowed = allowed & (
                        kv_positions[None, :] > q_positions[:, None] - window
                    )
                mask_cache[cache_key] = allowed
                self._dense_allowed_mask_elements = cached_mask_elements + mask_elements
        precomputed_key_group_ids = None
        if query_group_ids is not None:
            query_group_ids = query_group_ids.to(device=q.device, dtype=torch.long)
            precomputed_key_group_ids = torch.full(
                (keys.size(0),), -1, dtype=torch.long, device=q.device
            )
            key_indices = q_positions.to(torch.long) - kv_offset_value
            valid_indices = (key_indices >= 0) & (key_indices < keys.size(0))
            precomputed_key_group_ids[key_indices[valid_indices]] = query_group_ids[
                valid_indices
            ]
        kv_heads_expanded = False
        group = self.geometry.head_num // self.geometry.kv_head_num
        if kv_valid is None and group > 1:
            if use_exact_causal_softmax:
                keys, values = rtp_llm_ops.gemma4_expand_kv_heads_8_bf16(
                    keys.contiguous(), values.contiguous()
                )
            elif use_exact_window_softmax:
                keys, values = rtp_llm_ops.gemma4_expand_kv_heads_2_bf16(
                    keys.contiguous(), values.contiguous()
                )
            else:
                keys = keys.repeat_interleave(group, dim=1)
                values = values.repeat_interleave(group, dim=1)
            kv_heads_expanded = True
        direct_output = torch.empty_like(q) if use_exact_masked_softmax else None
        outputs = []
        # Trimming masked key columns changes BF16 GEMM selection and breaks HF parity.
        for query_start in range(0, q.size(0), query_chunk_size):
            query_end = min(query_start + query_chunk_size, q.size(0))
            chunk_positions = q_positions[query_start:query_end]
            chunk_output = self._attend_request(
                q[query_start:query_end],
                keys,
                values,
                chunk_positions,
                kv_offset_value,
                kv_valid=kv_valid,
                query_group_ids=(
                    None
                    if query_group_ids is None
                    else query_group_ids[query_start:query_end]
                ),
                _allow_chunking=False,
                _expand_kv_heads=False,
                _precomputed_allowed=(
                    None if allowed is None else allowed[query_start:query_end]
                ),
                _kv_heads_expanded=kv_heads_expanded,
                _precomputed_key_group_ids=precomputed_key_group_ids,
                _causal_query_start=(query_start if use_exact_masked_softmax else None),
                _window_left=(window if use_exact_window_softmax else -1),
                _output=(
                    None
                    if direct_output is None
                    else direct_output[query_start:query_end]
                ),
            )
            if direct_output is None:
                outputs.append(chunk_output)
        return direct_output if direct_output is not None else torch.cat(outputs, dim=0)

    def _attend_request(
        self,
        q: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        q_positions: torch.Tensor,
        kv_offset: Any = 0,
        kv_valid: Optional[torch.Tensor] = None,
        query_group_ids: Optional[torch.Tensor] = None,
        _allow_chunking: bool = True,
        _expand_kv_heads: bool = True,
        _precomputed_allowed: Optional[torch.Tensor] = None,
        _kv_heads_expanded: bool = False,
        _precomputed_key_group_ids: Optional[torch.Tensor] = None,
        _causal_query_start: Optional[int] = None,
        _window_left: int = -1,
        _output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """q [T, H, D]; keys/values [S, kv_heads, D]; q_positions [T].

        ``kv_offset`` is the absolute position of ``keys[0]`` (0 for the paged
        path where the gathered keys span positions 0..kvlen-1; the first q
        position for the cacheless path). ``kv_valid`` [S] marks positions
        whose KV is resident; evicted (sparse-table) positions are excluded
        from the softmax entirely.
        """
        group = self.geometry.head_num // self.geometry.kv_head_num
        window = self.geometry.sliding_window
        has_visual_group = (
            self.geometry.tag == GEMMA4_TAG_SWA
            and query_group_ids is not None
            and bool((query_group_ids >= 0).any())
        )
        if (
            q.size(0) == 1
            and keys.size(0) == 8192
            and q.dtype == torch.bfloat16
            and keys.dtype == torch.bfloat16
            and values.dtype == torch.bfloat16
            and q.is_cuda
            and q.is_contiguous()
            and keys.is_contiguous()
            and values.is_contiguous()
            and kv_valid is not None
            and query_group_ids is None
            and self.geometry.head_num == 16
            and self.geometry.kv_head_num in (2, 8)
            and self.geometry.head_dim in (256, 512)
            and torch.version.hip is None
        ):
            if self.geometry.kv_head_num == 2:
                keys, values = rtp_llm_ops.gemma4_expand_kv_heads_8_bf16(keys, values)
            else:
                keys, values = rtp_llm_ops.gemma4_expand_kv_heads_2_bf16(keys, values)
            query_states = q.transpose(0, 1).unsqueeze(0)
            key_states = keys.transpose(0, 1).unsqueeze(0)
            value_states = values.transpose(0, 1).unsqueeze(0)
            attention_weights = torch.matmul(query_states, key_states.transpose(2, 3))
            probabilities = rtp_llm_ops.gemma4_softmax_8192_bf16(
                attention_weights,
                keys.size(0) - 1,
                self.geometry.sliding_window,
            )
            output = torch.matmul(probabilities, value_states)
            return output.squeeze(0).transpose(0, 1).contiguous()
        if (
            _allow_chunking
            and q.size(0) > 1
            and q.size(0) * keys.size(0) > self.max_attention_matrix_elements
        ):
            return self._attend_request_chunked(
                q,
                keys,
                values,
                q_positions,
                kv_offset,
                kv_valid,
                query_group_ids,
            )
        dense_cache = kv_valid is None or bool(kv_valid.all())
        initial_prefill = (
            not has_visual_group
            and dense_cache
            and q.size(0) == keys.size(0)
            and keys.size(0) <= self.sdpa_causal_limit
            and (window <= 0 or keys.size(0) <= window)
            and int(q_positions[0].item()) == int(kv_offset)
        )
        if initial_prefill:
            output = F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                keys.transpose(0, 1).unsqueeze(0),
                values.transpose(0, 1).unsqueeze(0),
                attn_mask=None,
                dropout_p=0.0,
                scale=1.0,
                is_causal=True,
                enable_gqa=group > 1,
            )
            return output.squeeze(0).transpose(0, 1).contiguous()
        if group > 1 and _expand_kv_heads and not _kv_heads_expanded:
            if (
                keys.is_cuda
                and keys.dtype == torch.bfloat16
                and values.dtype == torch.bfloat16
                and keys.is_contiguous()
                and values.is_contiguous()
                and self.geometry.head_num == 16
                and self.geometry.kv_head_num == 2
                and self.geometry.head_dim == 512
                and torch.version.hip is None
            ):
                keys, values = rtp_llm_ops.gemma4_expand_kv_heads_8_bf16(keys, values)
            elif (
                keys.is_cuda
                and keys.dtype == torch.bfloat16
                and values.dtype == torch.bfloat16
                and keys.is_contiguous()
                and values.is_contiguous()
                and self.geometry.head_num == 16
                and self.geometry.kv_head_num == 8
                and self.geometry.head_dim == 256
                and torch.version.hip is None
            ):
                keys, values = rtp_llm_ops.gemma4_expand_kv_heads_2_bf16(keys, values)
            else:
                keys = keys.repeat_interleave(group, dim=1)
                values = values.repeat_interleave(group, dim=1)
        if _causal_query_start is not None:
            allowed = None
            kv_positions = None
        elif _precomputed_allowed is None:
            kv_positions = torch.arange(keys.size(0), device=q.device) + kv_offset
            allowed = kv_positions[None, :] <= q_positions[:, None]
            if window:
                allowed = allowed & (
                    kv_positions[None, :] > q_positions[:, None] - window
                )
        else:
            allowed = _precomputed_allowed
            kv_positions = None
        if has_visual_group:
            query_group_ids = query_group_ids.to(device=q.device, dtype=torch.long)
            key_group_ids = _precomputed_key_group_ids
            if key_group_ids is None:
                key_group_ids = torch.full(
                    (keys.size(0),), -1, dtype=torch.long, device=q.device
                )
                key_indices = q_positions.to(torch.long) - torch.as_tensor(
                    kv_offset, dtype=torch.long, device=q.device
                )
                valid_indices = (key_indices >= 0) & (key_indices < keys.size(0))
                key_group_ids[key_indices[valid_indices]] = query_group_ids[
                    valid_indices
                ]
            same_visual_group = (query_group_ids[:, None] == key_group_ids[None, :]) & (
                query_group_ids[:, None] >= 0
            )
            allowed = allowed | same_visual_group
        if kv_valid is not None:
            allowed = allowed & kv_valid[None, :]
        if group > 1 and not _expand_kv_heads and not _kv_heads_expanded:
            query_states = q.reshape(
                q.size(0), self.geometry.kv_head_num, group, q.size(-1)
            ).permute(1, 2, 0, 3)
            key_states = keys.transpose(0, 1).unsqueeze(1)
            value_states = values.transpose(0, 1).unsqueeze(1)
            attention_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
            if kv_valid is None and not has_visual_group:
                attention_weights.masked_fill_(
                    ~allowed[None, None, :, :], torch.finfo(q.dtype).min
                )
            else:
                attention_mask = torch.zeros(
                    1,
                    1,
                    q.size(0),
                    keys.size(0),
                    dtype=q.dtype,
                    device=q.device,
                ).masked_fill(~allowed[None, None, :, :], torch.finfo(q.dtype).min)
                attention_weights = attention_weights + attention_mask
            probabilities = F.softmax(
                attention_weights,
                dim=-1,
                dtype=torch.float32,
            ).to(q.dtype)
            if kv_valid is not None:
                row_has_key = allowed.any(dim=-1)
                probabilities = torch.where(
                    row_has_key[None, None, :, None],
                    probabilities,
                    torch.zeros_like(probabilities),
                )
            output = torch.matmul(probabilities, value_states)
            return output.permute(2, 0, 1, 3).reshape(
                q.size(0), self.geometry.head_num, q.size(-1)
            )
        if _causal_query_start is not None and _window_left < 0:
            if _output is not None:
                key_len = _causal_query_start + q.size(0)
                attention_weights = rtp_llm_ops.gemma4_qk_bmm_8192_bf16_key_len(
                    q.contiguous(), keys, key_len
                )
            else:
                attention_weights = rtp_llm_ops.gemma4_qk_bmm_8192_bf16(
                    q.contiguous(), keys
                )
            probabilities = rtp_llm_ops.gemma4_softmax_8192_bf16(
                attention_weights, _causal_query_start, _window_left
            )
            if _output is not None:
                rtp_llm_ops.gemma4_pv_bmm_8192_bf16_out_key_len(
                    probabilities, values, _output, key_len
                )
                return _output
            return rtp_llm_ops.gemma4_pv_bmm_8192_bf16(probabilities, values)

        query_states = q.transpose(0, 1).unsqueeze(0)
        key_states = keys.transpose(0, 1).unsqueeze(0)
        value_states = values.transpose(0, 1).unsqueeze(0)
        attention_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        if _causal_query_start is not None:
            probabilities = rtp_llm_ops.gemma4_softmax_8192_bf16(
                attention_weights, _causal_query_start, _window_left
            )
        else:
            if kv_valid is None and not has_visual_group:
                attention_weights.masked_fill_(
                    ~allowed[None, None, :, :], torch.finfo(q.dtype).min
                )
            else:
                attention_mask = torch.zeros(
                    1,
                    1,
                    q.size(0),
                    keys.size(0),
                    dtype=q.dtype,
                    device=q.device,
                ).masked_fill(~allowed[None, None, :, :], torch.finfo(q.dtype).min)
                attention_weights = attention_weights + attention_mask
            if (
                self.geometry.tag == GEMMA4_TAG_SWA
                and q.size(0) == 512
                and attention_weights.size(-1) == 8192
                and attention_weights.dtype == torch.bfloat16
                and attention_weights.is_cuda
                and torch.version.hip is None
                and kv_valid is None
                and not has_visual_group
            ):
                probabilities = rtp_llm_ops.gemma4_softmax_8192_bf16(attention_weights)
            else:
                probabilities = F.softmax(
                    attention_weights, dim=-1, dtype=torch.float32
                ).to(q.dtype)
            if kv_valid is not None:
                row_has_key = allowed.any(dim=-1)
                probabilities = torch.where(
                    row_has_key[None, None, :, None],
                    probabilities,
                    torch.zeros_like(probabilities),
                )
        if (
            _causal_query_start is not None
            and _window_left >= 0
            and probabilities.shape == (1, 16, 512, 8192)
            and values.shape == (8192, 16, 256)
            and probabilities.dtype == torch.bfloat16
            and values.dtype == torch.bfloat16
            and probabilities.is_contiguous()
            and values.is_contiguous()
            and torch.version.hip is None
        ):
            if _output is not None:
                rtp_llm_ops.gemma4_swa_pv_bmm_8192_bf16_out(
                    probabilities, values, _output
                )
                return _output
            return rtp_llm_ops.gemma4_swa_pv_bmm_8192_bf16(probabilities, values)
        output = torch.matmul(probabilities, value_states)
        return output.squeeze(0).transpose(0, 1).contiguous()

    def _forward_swa_graph(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: LayerKVCache,
    ) -> torch.Tensor:
        """Graph-capture path for the SLIDING-WINDOW (swa) layers.

        Replaces the flashinfer paged-prefill wrapper under graph mode with
        page-chunked SDPA + LSE-merge restricted to the WINDOW pages (all
        shapes capture-time constants; per-replay variation flows through
        the params' device buffers). For q_len contiguous queries, the union
        of their windows spans window + q_len - 1 positions, plus at most one
        page of alignment slack.

        Note: the original motivation (stale captured launch) was retracted
        - the probe33 isolation had a clone-binding defect; the actual
        graph-mode flip root cause was the colliding-index index_add_ in
        the MoE accumulation (see numerical-issues report). This torch path
        is kept as the deterministic, capture-safe SWA computation.
        """
        window = self.geometry.sliding_window
        batch = (
            self.attn_inputs.input_lengths.numel()
            if self.attn_inputs.is_prefill
            else self.attn_inputs.sequence_lengths.numel()
        )
        total_tokens = q.size(0)
        if batch == 0 or total_tokens == 0:
            return q.reshape(total_tokens, -1).clone()
        paged = kv_cache.kv_cache_base
        if paged.dim() == 2:
            paged = attn_common.reshape_paged_kv_cache(
                paged,
                self.geometry.kv_head_num,
                self.page_size,
                self.geometry.head_dim,
            )
        k_cache, v_cache = paged[:, 0], paged[:, 1]
        self._write_cache(k, v, kv_cache, k_cache, v_cache)
        pages_all = self.fmha_params.page_indice_d
        indptr = self.fmha_params.decode_page_indptr_d
        last_page_len = self.fmha_params.paged_kv_last_page_len_d
        num_pages = k_cache.shape[0]
        kv_heads = self.geometry.kv_head_num
        head_dim = self.geometry.head_dim
        head_num = self.geometry.head_num
        group = head_num // kv_heads
        row_pages = (indptr[1 : batch + 1] - indptr[:batch]).clamp(min=0)
        kvlen = (row_pages - 1).clamp(min=0) * self.page_size + last_page_len[
            :batch
        ]  # [batch]
        # fixed chunk count: trailing pages that can hold the union of every
        # query's window. Multi-query target verification extends that union by
        # q_len - 1 positions before the final query's window.
        max_row_pages = self._full_row_pages
        if max_row_pages is None:
            max_row_pages = int(row_pages.max().item()) if batch > 0 else 1
            self._full_row_pages = max(max_row_pages, 1)
        q_len = max(total_tokens // batch, 1)
        window_span = window + q_len - 1
        win_pages = (window_span + self.page_size - 1) // self.page_size + 1
        max_chunks = min(win_pages, max_row_pages)
        max_chunks = max(max_chunks, 1)
        kvlen_f = kvlen.float()
        q_pos = (
            self.fmha_params.positions_d.narrow(0, 0, total_tokens)
            .reshape(batch, q_len)
            .float()
        )
        # q -> [batch, H, q_len, D]
        qb = q.reshape(batch, q_len, head_num, head_dim).transpose(1, 2)
        out_acc = None
        lse_acc = None
        for chunk in range(max_chunks):
            # trailing page index for this chunk: pages_all[end - 1 - chunk]
            # CHUNK EXISTENCE per row: a row with row_pages pages only has
            # chunks 0..row_pages-1. For chunk >= row_pages the (clamped)
            # page_slot reads ANOTHER request's page (or page 0) and the
            # clamped base_pos would place it at positions 0..page_size-1 -
            # which the pos<kvlen/window checks do NOT reliably exclude for
            # long rows (kvlen > page_size). Mask those chunks out per row.
            chunk_exists = (row_pages - 1 - chunk) >= 0  # [B]
            end_idx = indptr[1 : batch + 1]
            page_slot = (end_idx - 1 - chunk).clamp(min=0)
            page_id = pages_all[page_slot.clamp(0, pages_all.numel() - 1)]
            page_ok = (page_id >= 0) & (page_id < num_pages) & chunk_exists
            # number of FULL pages below this chunk's page within the request
            # chunk 0 = last page; its base position = (row_pages-1)*page_size
            base_pos = (row_pages - 1 - chunk).clamp(min=0) * self.page_size
            pos = (
                torch.arange(self.page_size, device=q.device)
                .unsqueeze(0)
                .expand(batch, self.page_size)
                + base_pos.unsqueeze(1)
            ).float()  # [B, page]
            # logical validity per SLOT, derived ONLY from metadata (chunk
            # existence, page validity, kvlen, causality, window) - never
            # from the numeric content of the gathered KV
            slot_valid = (
                chunk_exists.unsqueeze(1)
                & page_ok.unsqueeze(1)
                & (pos < kvlen_f.unsqueeze(1))
            )
            allowed = (
                slot_valid.unsqueeze(1)
                & (pos.unsqueeze(1) <= q_pos.unsqueeze(2))
                & (pos.unsqueeze(1) > (q_pos - float(window)).unsqueeze(2))
            )  # [B, q_len, page]
            # gather with a SAFE page id for logically-invalid rows/slots
            # (their values are fully masked below); the selection is a true
            # zero-choose, not multiply (0 * NaN = NaN). NaN at LOGICALLY
            # VALID slots is NOT cleaned: it propagates into the output and
            # must surface as a failure to be root-caused (write/layout bug)
            safe_id = torch.where(
                page_ok,
                page_id.clamp(0, num_pages - 1),
                torch.zeros_like(page_id),
            )
            zero_page = torch.zeros_like(k_cache[0])
            kc_raw = k_cache[safe_id]  # [B, KVH, page, D]
            vc_raw = v_cache[safe_id]
            # slot-level choose: valid slots keep the gathered values (even
            # NaN), invalid slots get exact zeros
            slot_keep = slot_valid[:, None, :].to(kc_raw.dtype)  # [B,1,page]
            kc = torch.where(
                slot_keep[:, :, :, None].expand_as(kc_raw) > 0,
                kc_raw,
                zero_page.expand_as(kc_raw),
            )
            vc = torch.where(
                slot_keep[:, :, :, None].expand_as(vc_raw) > 0,
                vc_raw,
                zero_page.expand_as(vc_raw),
            )
            mask = torch.where(allowed, 0.0, float("-inf")).to(q.dtype)
            mask = mask.unsqueeze(1)  # [B,1,q_len,page]
            co = torch.nn.functional.scaled_dot_product_attention(
                qb,
                kc,
                vc,
                attn_mask=mask,
                scale=1.0,
                enable_gqa=(group > 1),
            )
            if group > 1:
                kc_e = kc.repeat_interleave(group, dim=1)
            else:
                kc_e = kc
            scores = (
                torch.einsum("bhqd,bhkd->bhqk", qb.float(), kc_e.float()) + mask.float()
            )
            lse = torch.logsumexp(scores, dim=-1)
            any_a = allowed.any(dim=2)
            lse = torch.where(
                any_a.unsqueeze(1), lse, torch.full_like(lse, float("-inf"))
            )
            co = torch.where(any_a.unsqueeze(1).unsqueeze(-1), co, torch.zeros_like(co))
            if out_acc is None:
                out_acc = co.float()
                lse_acc = lse
            else:
                m = torch.maximum(lse_acc, lse)
                m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
                w_old = torch.exp(lse_acc - m_safe)
                w_new = torch.exp(lse - m_safe)
                denom = w_old + w_new
                denom_safe = torch.where(denom > 0, denom, torch.ones_like(denom))
                out_acc = (
                    out_acc * w_old.unsqueeze(-1) + co.float() * w_new.unsqueeze(-1)
                ) / denom_safe.unsqueeze(-1)
                lse_acc = m + torch.log(denom_safe)
                lse_acc = torch.where(
                    denom > 0, lse_acc, torch.full_like(lse_acc, float("-inf"))
                )
        if out_acc is None:
            out_acc = torch.zeros(
                batch,
                head_num,
                q_len,
                head_dim,
                dtype=q.dtype,
                device=q.device,
            ).float()
        out = out_acc.to(q.dtype)
        out = out.transpose(1, 2).reshape(total_tokens, head_num, head_dim)
        return out

    def _forward_full_graph(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: LayerKVCache,
    ) -> torch.Tensor:
        """Graph-capture path for the FULL-attention layers (EXPERIMENTAL).

        The eager full-layer path is host-driven (per-request python loop with
        .cpu() metadata) and cannot be captured. This path keeps every shape a
        capture-time constant and all data on device:

        - batch: the graph bucket's fixed batch size (one captured graph per
          bucket; smaller live batches pad with masked rows)
        - kv access: PAGE-CHUNKED loop with a fixed number of chunks (the
          capture max pages); each chunk gathers one page per request
          ([batch, page_size, kv_heads, head_dim] - small), runs SDPA on the
          chunk with an additive mask (chunk-local positions vs the device
          kvlen), and merges via log-sum-exp. The dense gather of the whole
          KV history OOMs at max_seq (5 full layers x k/v x copies ~ 10GB);
          the chunk loop keeps peak memory at one page chunk
        - all shapes are capture-time constants; per-replay variation flows
          through the params' device buffers (page ids, indptr, last_page_len)

        Numerics differ from eager (chunked SDPA + LSE merge vs einsum);
        replay-vs-eager equivalence is asserted by gemma4_cuda_graph_probe_test
        at the token level, per the acceptance requirements.
        """
        batch = (
            self.attn_inputs.input_lengths.numel()
            if self.attn_inputs.is_prefill
            else self.attn_inputs.sequence_lengths.numel()
        )
        total_tokens = q.size(0)
        if batch == 0 or total_tokens == 0:
            return q.reshape(total_tokens, -1).clone()
        paged = kv_cache.kv_cache_base
        if paged.dim() == 2:
            paged = attn_common.reshape_paged_kv_cache(
                paged,
                self.geometry.kv_head_num,
                self.page_size,
                self.geometry.head_dim,
            )
        k_cache, v_cache = paged[:, 0], paged[:, 1]
        self._write_cache(k, v, kv_cache, k_cache, v_cache)
        pages_all = self.fmha_params.page_indice_d
        indptr = self.fmha_params.decode_page_indptr_d
        last_page_len = self.fmha_params.paged_kv_last_page_len_d
        num_pages = k_cache.shape[0]
        # fixed chunk count: the capture-time max row width (eager warmup runs
        # before capture, so .item() here is safe and fixes the loop length)
        max_row_pages = self._full_row_pages
        if max_row_pages is None:
            row_pages_t = (indptr[1 : batch + 1] - indptr[:batch]).clamp(min=0)
            max_row_pages = int(row_pages_t.max().item()) if batch > 0 else 1
            max_row_pages = max(max_row_pages, 1)
            self._full_row_pages = max_row_pages
        kv_heads = self.geometry.kv_head_num
        head_dim = self.geometry.head_dim
        head_num = self.geometry.head_num
        group = head_num // kv_heads
        # per-request kvlen on device: (row pages - 1) * page_size + lpl
        row_pages = (indptr[1 : batch + 1] - indptr[:batch]).clamp(min=0)
        kvlen = (row_pages - 1).clamp(min=0) * self.page_size + last_page_len[
            :batch
        ]  # [batch]
        q_len = total_tokens // batch
        q_pos = self.fmha_params.positions_d.narrow(0, 0, total_tokens).reshape(
            batch, q_len
        )
        # q -> [batch, H, q_len, D]
        qb = q.reshape(batch, q_len, head_num, head_dim).transpose(1, 2).contiguous()
        # running LSE merge state
        out_acc = None
        lse_acc = None
        for chunk in range(max_row_pages):
            # page id per request for this chunk: pages_all[indptr[b] + chunk]
            slot = (indptr[:batch] + chunk).clamp_(0, pages_all.numel() - 1)
            page_id = pages_all[slot]  # [batch]
            page_ok = (page_id >= 0) & (page_id < num_pages)  # [batch]
            safe_id = page_id.clamp(0, num_pages - 1)
            # k_cache[safe_id]: [batch, kv_heads, page_size, head_dim];
            # out-of-table slots read arbitrary (possibly uninitialized/NaN)
            # pages - NaN would survive `scores + (-inf mask)` (NaN+x=NaN), so
            # sanitize the gathered chunk; masked positions contribute nothing
            kc = torch.nan_to_num(k_cache[safe_id])
            vc = torch.nan_to_num(v_cache[safe_id])
            # chunk-local flat positions: chunk * page_size + [0, page_size)
            pos_lo = chunk * self.page_size
            pos = (
                torch.arange(self.page_size, device=q.device).unsqueeze(0) + pos_lo
            )  # [batch, page_size]
            slot_valid = (pos < kvlen.unsqueeze(1)) & page_ok.unsqueeze(1)
            allowed = slot_valid.unsqueeze(1) & (pos.unsqueeze(1) <= q_pos.unsqueeze(2))
            mask = torch.where(allowed, 0.0, float("-inf")).to(q.dtype)
            mask = mask.unsqueeze(1)  # [batch,1,q_len,page_size]
            chunk_out = torch.nn.functional.scaled_dot_product_attention(
                qb,  # [batch, H, q_len, D]
                kc,  # [batch, kv_heads, page_size, D]
                vc,
                attn_mask=mask,
                scale=1.0,
                enable_gqa=(group > 1),
            )  # [batch, H, q_len, D]
            # per-head max over kv positions of the masked scores == -inf row
            # detection: rows fully masked produce NaN from softmax; compute
            # LSE as logsumexp of scores via SDPA's mathematical equivalent:
            # use the mask row-sum trick - rows with no allowed key have
            # softmax denom 0. Instead of re-deriving scores, detect full-mask
            # rows from the mask itself.
            any_allowed = allowed.any(dim=2)  # [batch,q_len]
            # logsumexp of this chunk's scores per q head: scores = q@k^T
            # (scale 1.0); recompute cheaply on the chunk only for LSE.
            # GQA: expand kv heads to q heads to match the SDPA layout
            if group > 1:
                kc_e = kc.repeat_interleave(group, dim=1)
            else:
                kc_e = kc
            # scores [batch, H, q_len, page_size]
            scores = torch.einsum("bhqd,bhkd->bhqk", qb.float(), kc_e.float())
            scores = scores + mask.float()
            lse = torch.logsumexp(scores, dim=-1)  # [batch, H, q_len]
            # rows fully masked: lse = -inf, chunk_out = NaN -> treat as no
            # contribution
            lse = torch.where(
                any_allowed.unsqueeze(1), lse, torch.full_like(lse, float("-inf"))
            )
            chunk_out = torch.where(
                any_allowed.unsqueeze(1).unsqueeze(-1),
                chunk_out,
                torch.zeros_like(chunk_out),
            )
            # Running-merge invariant (flash-style): maintain the UNNORMALIZED
            # weighted sum `num_acc = sum_c exp(lse_c) * out_c` conceptually,
            # materialized as (out_acc, lse_acc) where out_acc is the softmax-
            # normalized running mean and lse_acc = log(sum_c exp(lse_c)).
            # Merge: m = max(lse_acc, lse); s = exp(lse_acc - m) +
            # exp(lse - m); out = (out_acc * exp(lse_acc-m) + out_c *
            # exp(lse-m)) / s; lse_acc = m + log(s). The previous code stored
            # lse_acc = m WITHOUT log(s), so the third chunk onward weighed
            # already-merged chunks incorrectly (e.g. three equal-LSE chunks
            # got 1/4, 1/4, 1/2 instead of 1/3 each) - caught by supervision.
            if out_acc is None:
                out_acc = chunk_out.float()
                lse_acc = lse
            else:
                m = torch.maximum(lse_acc, lse)
                m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
                w_old = torch.exp(lse_acc - m_safe)
                w_new = torch.exp(lse - m_safe)
                denom = w_old + w_new
                # both -inf (nothing seen yet): keep zeros, lse stays -inf
                denom_safe = torch.where(denom > 0, denom, torch.ones_like(denom))
                out_acc = (
                    out_acc * w_old.unsqueeze(-1)
                    + chunk_out.float() * w_new.unsqueeze(-1)
                ) / denom_safe.unsqueeze(-1)
                lse_acc = m + torch.log(denom_safe)
                lse_acc = torch.where(
                    denom > 0, lse_acc, torch.full_like(lse_acc, float("-inf"))
                )
        if out_acc is None:
            out_acc = torch.zeros(
                batch,
                head_num,
                q_len,
                head_dim,
                dtype=q.dtype,
                device=q.device,
            ).float()
        out = out_acc.to(q.dtype)
        # [batch, H, q_len, D] -> [total, H, D]
        out = out.transpose(1, 2).reshape(total_tokens, head_num, head_dim)
        return out

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        if self._graph_mode:
            batch = (
                self.attn_inputs.input_lengths.numel()
                if self.attn_inputs.is_prefill
                else self.attn_inputs.sequence_lengths.numel()
            )
            if batch <= 0 or q.size(0) % batch != 0:
                raise RuntimeError(
                    "Gemma4 CUDA graph attention requires a fixed query width per request"
                )
        if (
            kv_cache is not None
            and self._graph_mode
            and self.geometry.tag == GEMMA4_TAG_SWA
        ):
            return self._forward_swa_graph(q, k, v, kv_cache)
        if (
            kv_cache is not None
            and self._graph_mode
            and self.geometry.tag == GEMMA4_TAG_FULL
        ):
            return self._forward_full_graph(q, k, v, kv_cache)
        total_tokens = q.size(0)
        lengths, starts = self._request_layout()
        if starts[-1] != total_tokens:
            raise RuntimeError(
                f"packed tokens {total_tokens} != sum(input_lengths) {starts[-1]}"
            )
        k_cache = v_cache = None
        page_indptr_host = last_page_len_host = None
        if kv_cache is not None:
            k_cache, v_cache = self._paged_views(kv_cache)
            if self.attn_inputs.is_prefill:
                return self._forward_prefill_current_kv(
                    q, k, v, kv_cache, k_cache, v_cache
                )
            self._write_cache(k, v, kv_cache, k_cache, v_cache)
            page_indptr_host = self.fmha_params.decode_page_indptr_d.cpu().tolist()
            last_page_len_host = (
                self.fmha_params.paged_kv_last_page_len_d.cpu().tolist()
            )
        positions = self.fmha_params.positions_d.narrow(0, 0, total_tokens)
        outputs = []
        for i, _length in enumerate(lengths):
            start, end = starts[i], starts[i + 1]
            if kv_cache is not None:
                keys, values, kv_valid = self._gather_request(
                    i, page_indptr_host, last_page_len_host, k_cache, v_cache
                )
                kv_offset = 0
            else:
                keys = k[start:end]
                values = v[start:end]
                kv_offset = positions[start]
                kv_valid = None
            query_group_ids = (
                self.vision_group_ids[start:end]
                if self.vision_group_ids is not None
                else None
            )
            outputs.append(
                self._attend_request(
                    q[start:end],
                    keys,
                    values,
                    positions[start:end],
                    kv_offset,
                    kv_valid=kv_valid,
                    query_group_ids=query_group_ids,
                )
            )
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


class Gemma4SwaFlashinferImpl(Gemma4TorchFMHAImpl):
    """Production attention backend for the sliding-window (SWA) layers.

    Same contract as ``Gemma4TorchFMHAImpl`` — ``forward(q, k, v, kv_cache)``
    receives post-QK/V-norm post-RoPE tensors — but the attention core is
    flashinfer's ``BatchPrefillWithPagedKVCacheWrapper`` running with causal
    masking, the layer group's sliding window (``window_left``) and Gemma4's
    attention scale (``sm_scale=1.0``, no ``1/sqrt(d)``). The KV cache write
    path is shared with the reference implementation (flashinfer append with
    NULL-block skip), so write/read behavior is identical.

    FlashInfer derives each query's position as ``(kv_len - q_len + j)``: the
    query tokens are the tail of the paged KV sequence, which matches the
    engine's append-after-prefix layout (prefill: ``prefix + j``; decode: the
    single new token at ``seq_len - 1``).

    Prefill uses the torch current-KV path so sparse retention cannot erase
    early query history. Eager decode uses FlashInfer only for dense tables and
    falls back to the sparse-aware torch path when NULL pages are present.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        geometry: Gemma4LayerGeometry,
        attn_inputs: Any,
        page_size: int,
        parallelism_config: Optional[ParallelismConfig] = None,
    ):
        if geometry.tag != GEMMA4_TAG_SWA or geometry.sliding_window <= 0:
            raise ValueError(
                "Gemma4SwaFlashinferImpl requires a sliding-window (swa) layer"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("gemma4 SWA flashinfer backend requires CUDA")
        super().__init__(
            model_config, geometry, attn_inputs, page_size, parallelism_config
        )
        from flashinfer import (
            BatchPrefillWithPagedKVCacheWrapper,
            BatchPrefillWithRaggedKVCacheWrapper,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
            get_py_flashinfer_workspace_buffer,
        )

        # Gemma4 attends the last `sliding_window` positions including the
        # current one: kv in [q - window + 1, q]. FlashInfer's window_left is
        # the inclusive left bound (kv in [q - window_left, q]).
        self._window_left = geometry.sliding_window - 1
        self._workspace = get_py_flashinfer_workspace_buffer()
        self._wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self._workspace, "HND", backend="auto"
        )
        self._ragged_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self._workspace, "NHD", backend="auto", use_cuda_graph=False
        )
        self._ragged_planned = False
        # grow-only decode qo_indptr buffer (fixed address across graph replays)
        self._decode_qo_buf = None
        self._wrapper_graph_bound = False
        self._plan_dtype = torch.bfloat16
        # full-layer graph path: capture-time row width cache (the eager
        # warmup forward runs before capture and fixes this shape constant)
        self._full_row_pages = None
        if getattr(attn_inputs, "is_cuda_graph", False):
            # capture path: the decode graph runner creates this impl from
            # capture inputs (is_cuda_graph=True) and warms up + captures
            # immediately; prepare_cuda_graph only runs at REPLAY time, so the
            # graph buffers must be bound and the wrapper planned HERE, before
            # any captured forward (flashinfer's plan is host-side and cannot
            # run inside a capture region)
            self._bind_graph_and_plan()

    def _bind_graph_and_plan(self) -> None:
        batch = self.fmha_params.decode_page_indptr_d.numel() - 1
        if batch < 0:
            return
        qo_buf = self._decode_qo_buf_or_new(batch + 1)
        vals = torch.arange(batch + 1, dtype=torch.int32, device=qo_buf.device)
        qo_buf[: batch + 1].copy_(vals)
        self._wrapper._use_cuda_graph = True
        self._wrapper._qo_indptr_buf = qo_buf
        self._wrapper._paged_kv_indptr_buf = self.fmha_params.decode_page_indptr_d.view(
            -1
        )
        self._wrapper._paged_kv_last_page_len_buf = (
            self.fmha_params.paged_kv_last_page_len_d.view(-1)
        )
        self._wrapper._paged_kv_indices_buf = self.fmha_params.page_indice_d
        self._wrapper._fixed_batch_size = batch
        # plan must receive a SOURCE distinct from the bound destination
        # buffers (flashinfer's graph-mode plan copies source -> buf; the
        # same tensor on both sides is an aliasing error)
        qo_src = torch.arange(batch + 1, dtype=torch.int32, device=qo_buf.device)
        self._plan_wrapper(qo_src)
        self._wrapper_graph_bound = True

    @staticmethod
    def support(geometry: Gemma4LayerGeometry) -> bool:
        # FULL layers (global_head_dim=512, k_equals_v) are excluded:
        # flashinfer's ragged prefill rejects head_dim 512 on SM89
        # ("Error in BatchPrefillWithRaggedKVCacheDispatched", verified
        # 2026-10-10 with a minimal repro: D=128 OK, D=512 fails). Their
        # eager chunked attention stays on the reference path; revisit on
        # platforms whose flashinfer supports 512-dim heads (SM100).
        if geometry.tag != GEMMA4_TAG_SWA or geometry.sliding_window <= 0:
            return False
        if not torch.cuda.is_available():
            return False
        try:
            import flashinfer  # noqa: F401
        except ImportError:
            return False
        return True

    def __del__(self):
        workspace = getattr(self, "_workspace", None)
        if workspace is not None:
            from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
                release_py_flashinfer_workspace_buffer,
            )

            release_py_flashinfer_workspace_buffer(workspace)
            self._workspace = None

    def prepare_cuda_graph(self, attn_inputs: Any) -> None:
        """CUDA-graph replay prepare (SWA impl).

        flashinfer's ``plan`` is host-side and cannot run inside a capture
        region, so the wrapper is planned at __init__ (capture path, before
        any captured forward) AND re-planned here before every replay -
        matching the stock PyFlashinferPagedPrefillImpl pattern whose
        prepare_cuda_graph calls the full prepare on every replay. (The
        probe33 'stale plan causes the flips' attribution was RETRACTED -
        that isolation had a clone-binding defect; the flip root cause was
        the colliding-index index_add_ in the MoE accumulation. The
        re-planning is retained as correct wrapper hygiene.)
        """
        self._fill_params(attn_inputs)
        if not self._wrapper_graph_bound:
            self._bind_graph_and_plan()
            return
        # replay path: re-plan with the refreshed params so the derived
        # scheduling tables match this step's metadata (plan copies fresh
        # sources into the bound fixed-address buffers; runs outside the
        # capture region so host-side work is legal here)
        batch = self.fmha_params.decode_page_indptr_d.numel() - 1
        if batch < 0:
            return
        qo_src = torch.arange(
            batch + 1, dtype=torch.int32, device=self._decode_qo_buf.device
        )
        self._plan_wrapper(qo_src)

    def _plan_wrapper(self, qo_indptr: torch.Tensor) -> None:
        kv_indptr = self.fmha_params.decode_page_indptr_d
        last_page_len = self.fmha_params.paged_kv_last_page_len_d
        batch = qo_indptr.numel() - 1
        if kv_indptr.numel() != batch + 1 or last_page_len.numel() != batch:
            raise RuntimeError(
                f"paged KV metadata batch mismatch: kv_indptr {kv_indptr.numel()} "
                f"last_page_len {last_page_len.numel()} batch {batch}"
            )
        # In graph mode flashinfer's plan copies each source into the bound
        # destination buffers; passing the destination tensors themselves as
        # sources is an aliasing error, so hand plan fresh clones (this runs
        # once outside capture; replays refresh the bound buffers instead)
        if getattr(self._wrapper, "_use_cuda_graph", False):
            kv_indptr = kv_indptr.clone()
            last_page_len = last_page_len.clone()
            page_indices = self.fmha_params.page_indice_d.clone()
        else:
            page_indices = self.fmha_params.page_indice_d
        self._wrapper.plan(
            qo_indptr,
            kv_indptr,
            page_indices,
            last_page_len,
            self.geometry.head_num,
            self.geometry.kv_head_num,
            self.geometry.head_dim,
            self.page_size,
            causal=True,
            sm_scale=1.0,
            window_left=self._window_left,
            q_data_type=self._plan_dtype,
            kv_data_type=self._plan_dtype,
            o_data_type=self._plan_dtype,
        )

    def _decode_qo_buf_or_new(self, n: int) -> torch.Tensor:
        buf = self._decode_qo_buf
        # exact size: flashinfer's graph-mode plan copy_ requires the bound
        # buffer to match the source length exactly (no oversized buffers)
        if buf is None or buf.numel() != n:
            buf = torch.empty(n, dtype=torch.int32, device="cuda")
            self._decode_qo_buf = buf
        return buf

    def _qo_indptr(self, device) -> tuple[int, torch.Tensor]:
        """Device-side qo_indptr without a host round-trip per layer.

        Prefill: the engine's cu_seqlens_device carries the per-request
        NEW-token prefix sums (the same contract the stock PyFlashinfer prefill
        op relies on); fall back to the host layout when the field is absent
        (unit-test fixtures). Decode: one q token per request -> arange.
        """
        if self.attn_inputs.is_prefill:
            batch = self.attn_inputs.input_lengths.size(0)
            cu = getattr(self.attn_inputs, "cu_seqlens_device", None)
            if cu is not None and cu.numel() >= batch + 1:
                return batch, cu[: batch + 1]
        else:
            batch = self.attn_inputs.sequence_lengths.numel()
            # graph-safe: the wrapper's bound qo buffer must have EXACTLY
            # batch+1 elements (flashinfer graph plan copies exact sizes);
            # in graph mode the buffer was pre-bound at construction and is
            # only refreshed here
            buf = self._decode_qo_buf
            if buf is None or buf.device != device or buf.numel() != batch + 1:
                buf = torch.empty(batch + 1, dtype=torch.int32, device=device)
                self._decode_qo_buf = buf
            view = buf
            vals = torch.arange(batch + 1, dtype=torch.int32, device=device)
            view.copy_(vals)
            return batch, view
        lengths, starts = self._request_layout()
        return len(lengths), torch.tensor(starts, dtype=torch.int32, device=device)

    def _run_ragged_prefill(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        _, qo_indptr = self._qo_indptr(q.device)
        if not self._ragged_planned:
            self._ragged_wrapper.plan(
                qo_indptr,
                qo_indptr,
                self.geometry.head_num,
                self.geometry.kv_head_num,
                self.geometry.head_dim,
                self.geometry.head_dim,
                causal=True,
                sm_scale=1.0,
                window_left=self._window_left,
                q_data_type=q.dtype,
                kv_data_type=k.dtype,
            )
            self._ragged_planned = True
        return self._ragged_wrapper.run(q, k, v)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        if self.attn_inputs.is_prefill:
            prefix_lengths = getattr(self.attn_inputs, "prefix_lengths", None)
            has_prefix = bool(
                prefix_lengths is not None
                and torch.count_nonzero(prefix_lengths).item() > 0
            )
            if (
                self._graph_mode
                or self.vision_group_ids is not None
                or kv_cache is None
                or has_prefix
            ):
                return super().forward(q, k, v, kv_cache)
            paged = kv_cache.kv_cache_base
            if paged.dim() == 2:
                paged = attn_common.reshape_paged_kv_cache(
                    paged,
                    self.geometry.kv_head_num,
                    self.page_size,
                    self.geometry.head_dim,
                )
            self._write_cache(k, v, kv_cache, paged[:, 0], paged[:, 1])
            return self._run_ragged_prefill(q, k, v)
        if self.vision_group_ids is not None:
            if self._graph_mode:
                raise RuntimeError(
                    "Gemma4 vision-group attention is not supported in CUDA graph mode"
                )
            return super().forward(q, k, v, kv_cache)
        if kv_cache is None:
            # cacheless forward (tests/warmup): reference path keeps the
            # position-based window semantics without a paged layout
            return super().forward(q, k, v, kv_cache)
        if self._graph_mode:
            # graph mode: use the deterministic capture-safe torch chunked
            # window path (the flashinfer wrapper under capture was replaced
            # while isolating the flips; the final root cause was the MoE
            # index_add_, but this path is kept as unit-verified)
            return self._forward_swa_graph(q, k, v, kv_cache)
        total_tokens = q.size(0)
        batch, qo_indptr = self._qo_indptr(q.device)
        if batch == 0:
            return q.reshape(total_tokens, -1).clone()
        paged = kv_cache.kv_cache_base
        if paged.dim() == 2:
            paged = attn_common.reshape_paged_kv_cache(
                paged,
                self.geometry.kv_head_num,
                self.page_size,
                self.geometry.head_dim,
            )
        page_indices = self.fmha_params.page_indice_d
        sparse_table = bool(
            ((page_indices < 0) | (page_indices >= paged.size(0))).any()
        )
        if sparse_table:
            return super().forward(q, k, v, kv_cache)
        k_cache, v_cache = paged[:, 0], paged[:, 1]
        self._write_cache(k, v, kv_cache, k_cache, v_cache)
        if self._wrapper_graph_bound:
            # graph mode: the wrapper was planned in prepare_cuda_graph with
            # fixed-address buffers; flashinfer's plan is host-side and may
            # not run inside a capture region
            return self._wrapper.run(q, paged)
        batch = qo_indptr.numel() - 1
        kv_indptr = self.fmha_params.decode_page_indptr_d
        last_page_len = self.fmha_params.paged_kv_last_page_len_d
        if kv_indptr.numel() != batch + 1 or last_page_len.numel() != batch:
            raise RuntimeError(
                f"paged KV metadata batch mismatch: kv_indptr {kv_indptr.numel()} "
                f"last_page_len {last_page_len.numel()} batch {batch}"
            )
        self._plan_wrapper(qo_indptr)
        # the whole [pages, 2, kv_heads, page_size, head_dim] tensor, matching
        # the stock PyFlashinfer call convention; flashinfer's run() unbinds
        # it into k/v views and only requires consistent K/V strides
        return self._wrapper.run(q, paged)


# ---------------------------------------------------------------------------
# Attention module
# ---------------------------------------------------------------------------


class Gemma4Attention(nn.Module):
    """HF Gemma4TextAttention math (modeling_gemma4.py:1126-1240).

    Order: qkv projection -> per-head QK-norm -> RoPE -> weightless V-norm ->
    attention (scale=1.0) -> output projection. The full-attention layers
    have no separate v projection (K == V at checkpoint); the loader
    duplicates the k segment into the qkv weight, so the v segment read here
    is numerically the k projection output, matching HF
    ``v_norm(k_proj(x))``.
    """

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        geometry: Gemma4LayerGeometry,
        layer_idx: int = 0,
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional[Any] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.geometry = geometry
        self.parallelism_config = parallelism_config
        self.tp_size = parallelism_config.get_attn_tp_size()
        self.variance_epsilon = config.layernorm_eps
        self.qkv_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.attn_qkv_w,
            W.attn_qkv_s,
            W.attn_qkv_b,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
            weight_scale_2_key=W.attn_qkv_s2,
            input_scale_key=W.attn_qkv_i_s,
        )
        self.o_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.attn_o_w,
            W.attn_o_s,
            W.attn_o_b,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
            weight_scale_2_key=W.attn_o_s2,
            input_scale_key=W.attn_o_i_s,
        )
        self.o_proj.maybe_cache_quant_scale(1024)
        self.q_norm = Gemma4RMSNorm(weights[W.q_ln_gamma], config.layernorm_eps)
        self.k_norm = Gemma4RMSNorm(weights[W.k_ln_gamma], config.layernorm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Gemma4TorchFMHAImpl,
        kv_cache: Optional[LayerKVCache] = None,
        rope_cache: Optional[Dict[str, tuple]] = None,
    ) -> torch.Tensor:
        total_tokens = hidden_states.size(0)
        geometry = self.geometry
        q_size = geometry.head_num * geometry.head_dim
        kv_size = geometry.kv_head_num * geometry.head_dim
        qkv_weight = getattr(self.qkv_proj, "weight", None)
        if (
            geometry.k_equals_v
            and isinstance(qkv_weight, torch.Tensor)
            and qkv_weight.is_floating_point()
        ):
            qkv_bias = getattr(self.qkv_proj, "bias", None)
            q_bias = qkv_bias[:q_size] if qkv_bias is not None else None
            k_bias = (
                qkv_bias[q_size : q_size + kv_size] if qkv_bias is not None else None
            )
            q = F.linear(hidden_states, qkv_weight[:q_size], q_bias)
            k = F.linear(
                hidden_states,
                qkv_weight[q_size : q_size + kv_size],
                k_bias,
            )
            v = k
        else:
            qkv = self.qkv_proj(hidden_states)
            q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
            if geometry.k_equals_v:
                v = k
        q = q.reshape(total_tokens, geometry.head_num, geometry.head_dim)
        k = k.reshape(total_tokens, geometry.kv_head_num, geometry.head_dim)
        v = v.reshape(total_tokens, geometry.kv_head_num, geometry.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = gemma4_rms_norm(v, None, self.variance_epsilon)
        positions = fmha_impl.positions(total_tokens)
        # G fused rope: skip the cos/sin table and apply rotation directly
        # from positions + inv_freq in one Triton kernel.
        if (
            os.environ.get("GEMMA4_FUSED_RESIDUAL", "0") == "1"
            and q.is_cuda
            and q.dtype == torch.bfloat16
            and q.dim() == 3
            and q.shape[-1] % 2 == 0
        ):
            from rtp_llm.models_py.modules.gemma4.rope import rope as _g_rope

            inv_freq = fmha_impl.rope_table._inv_freq_on(positions.device)
            q_rotated = _g_rope(q, positions, inv_freq)
            k_rotated = _g_rope(k, positions, inv_freq)
            if q_rotated is not None and k_rotated is not None:
                q, k = q_rotated, k_rotated
                attn_output = fmha_impl.forward(q, k, v, kv_cache)
                attn_output = attn_output.reshape(total_tokens, q_size).contiguous()
                output = self.o_proj(attn_output)
                if self.tp_size > 1:
                    output = all_reduce(output, group=Group.TP)
                return output
        if rope_cache is not None and geometry.tag in rope_cache:
            # per-forward cache: all layers of one tag share the same impl
            # (hence the same positions) and the same rope table, so the
            # 25 sliding layers previously recomputed identical cos/sin
            cos, sin = rope_cache[geometry.tag]
        else:
            if q.dtype == torch.bfloat16:
                cos, sin = fmha_impl.rope_cos_sin_bf16(positions)
            else:
                cos, sin = fmha_impl.rope_cos_sin(positions)
                cos = cos.to(q.dtype)
                sin = sin.to(q.dtype)
            if rope_cache is not None:
                rope_cache[geometry.tag] = (cos, sin)
        if (
            q.is_cuda
            and q.dtype == torch.bfloat16
            and k.dtype == torch.bfloat16
            and q.is_contiguous()
            and k.is_contiguous()
            and cos.is_contiguous()
            and sin.is_contiguous()
            and torch.version.hip is None
        ):
            q, k = rtp_llm_ops.gemma4_qk_rope_bf16(q, k, cos, sin)
        else:
            q = apply_gemma4_rope(q, cos, sin)
            k = apply_gemma4_rope(k, cos, sin)
        attn_output = fmha_impl.forward(q, k, v, kv_cache)
        attn_output = attn_output.reshape(total_tokens, q_size).contiguous()
        output = self.o_proj(attn_output)
        if self.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class Gemma4DecoderLayer(nn.Module):
    """HF Gemma4TextDecoderLayer sandwich structure (1355-1403)."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        geometry: Optional[Gemma4LayerGeometry] = None,
        quant_config: Optional[object] = None,
        moe_config: Optional[MoeConfig] = None,
        enable_cuda_graph: bool = False,
        hw_kernel_config: Optional[Any] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        if geometry is None:
            geometry = build_gemma4_layer_geometry(
                config, parallelism_config, layer_idx
            )
        self.geometry = geometry
        eps = config.layernorm_eps
        self.self_attn = Gemma4Attention(
            config,
            parallelism_config,
            weights,
            geometry,
            layer_idx=layer_idx,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        self.mlp = Gemma4DenseMLP(
            weights,
            parallelism_config,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        self.router = Gemma4Router(weights, config.hidden_size, config.moe_k, eps)
        self.experts = Gemma4Experts(
            weights,
            parallelism_config,
            model_config=config,
            moe_config=moe_config,
            enable_cuda_graph=enable_cuda_graph,
        )
        self.input_layernorm = Gemma4RMSNorm(weights[W.pre_ln_gamma], eps)
        self.post_attention_layernorm = Gemma4RMSNorm(weights[W.post_ln_gamma], eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            weights[_W_PRE_FFN_LN_GAMMA], eps
        )
        self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(
            weights[_W_PRE_FFN2_LN_GAMMA], eps
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            weights[W.post_ffn_ln_gamma], eps
        )
        self.post_feedforward_layernorm_1 = Gemma4RMSNorm(
            weights[_W_POST_FFN1_LN_GAMMA], eps
        )
        self.post_feedforward_layernorm_2 = Gemma4RMSNorm(
            weights[_W_POST_FFN2_LN_GAMMA], eps
        )
        self.layer_scalar = weights[_W_LAYER_SCALAR]  # [1]

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Gemma4TorchFMHAImpl,
        kv_cache: Optional[LayerKVCache] = None,
        rope_cache: Optional[Dict[str, tuple]] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            hidden_states, fmha_impl, kv_cache, rope_cache=rope_cache
        )
        hidden_states = gemma4_norm_add(
            attn_out,
            self.post_attention_layernorm.weight,
            residual,
            self.post_attention_layernorm.variance_epsilon,
        )

        residual = hidden_states
        dense_out = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        dense_out = self.post_feedforward_layernorm_1(dense_out)

        top_w, top_idx = self.router(residual)
        expert_input = self.pre_feedforward_layernorm_2(residual)
        expert_out = self.experts(expert_input, top_idx, top_w)
        expert_out = self.post_feedforward_layernorm_2(expert_out)

        hidden_states = gemma4_add_norm(
            dense_out,
            expert_out,
            self.post_feedforward_layernorm.weight,
            self.post_feedforward_layernorm.variance_epsilon,
        )
        return gemma4_residual_scale(
            hidden_states, residual, self.layer_scalar
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Gemma4Model(GptModelBase):
    """Gemma4 python descriptor (Gemma4ForConditionalGeneration text stack)."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        quant_config: Optional[object] = None,
        moe_config: Optional[MoeConfig] = None,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        if config.moe_k <= 0:
            raise ValueError("Gemma4Model requires config.moe_k > 0")
        if quant_config is None:
            quant_config = config.quant_config
        enable_cuda_graph = bool(
            getattr(py_hw_kernel_config, "enable_cuda_graph", False)
        )
        embed_weight = weights.get_global_weight(W.embedding)
        self.embed_tokens = Embedding(config, parallelism_config, embed_weight)
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        # HF scales the embedding by sqrt(hidden_size) cast to the embedding
        # dtype (53.066... -> 53.0 under bf16).
        self.embed_scale = float(
            torch.tensor(config.hidden_size**0.5, dtype=embed_weight.dtype)
        )
        self.geometries = [
            build_gemma4_layer_geometry(config, parallelism_config, idx)
            for idx in range(self.layer_num)
        ]
        self.layers = nn.ModuleList(
            [
                Gemma4DecoderLayer(
                    config,
                    parallelism_config,
                    weights.weights[idx],
                    idx,
                    geometry=self.geometries[idx],
                    quant_config=quant_config,
                    moe_config=moe_config,
                    enable_cuda_graph=enable_cuda_graph,
                    hw_kernel_config=py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = Gemma4RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), config.layernorm_eps
        )
        # last graph-mode mirrored onto the experts (transition log state);
        # see _mirror_experts_graph_mode for the lifecycle contract
        self._experts_graph_mode_last = False

    def _mirror_experts_graph_mode(
        self, graph_mode: bool, decode_mode: bool = False
    ) -> None:
        for decoder_layer in self.layers[: self.layer_num]:
            decoder_layer.experts._graph_mode = graph_mode
            decoder_layer.experts._decode_mode = decode_mode
        self._experts_graph_mode_last = graph_mode

    def _layer_tag(self, layer_idx: int) -> str:
        return self.geometries[layer_idx].tag

    def _layer_kv_cache(self, layer_idx: int, tag: str) -> Optional[LayerKVCache]:
        if self.kv_cache is None:
            return None
        groups = self.kv_cache.get_layer_cache_groups(layer_idx)
        if len(groups) == 1:
            return groups[0]
        for cache in groups:
            if str(cache.tag) == tag:
                return cache
        available = [str(cache.tag) for cache in groups]
        raise RuntimeError(
            f"layer {layer_idx} has no KV cache group {tag!r}; available={available}"
        )

    def _page_size_for_tag(self, tag: str) -> int:
        if self.kv_cache is not None:
            try:
                return int(self.kv_cache.get_kernel_seq_size_per_block(tag))
            except Exception as error:  # pragma: no cover - config mismatch
                raise RuntimeError(
                    f"KV cache has no page size for tag {tag!r}"
                ) from error
        attn_configs = self.config.getAttentionConfigs(
            self.parallelism_config.get_attn_tp_size()
        )
        page_size = int(attn_configs.kernel_tokens_per_block)
        if page_size <= 0:
            page_size = int(attn_configs.tokens_per_block)
        return page_size

    def _geometry_for_tag(self, tag: str) -> Gemma4LayerGeometry:
        for geometry in self.geometries:
            if geometry.tag == tag:
                return geometry
        raise RuntimeError(
            f"unknown Gemma4 attention tag {tag!r}; expected "
            f"{[GEMMA4_TAG_SWA, GEMMA4_TAG_FULL]}"
        )

    @staticmethod
    def _visual_group_ids(inputs: PyModelInputs) -> Optional[torch.Tensor]:
        multimodal_inputs = getattr(inputs, "multimodal_inputs", None)
        if multimodal_inputs is None or not multimodal_inputs.multimodal_features:
            return None
        features = multimodal_inputs.multimodal_features
        locations = multimodal_inputs.mm_features_locs
        if locations.numel() != len(features):
            raise ValueError(
                f"multimodal features ({len(features)}) and locations "
                f"({locations.numel()}) length mismatch"
            )
        group_ids = torch.full(
            (inputs.input_ids.numel(),),
            -1,
            dtype=torch.long,
            device=inputs.input_ids.device,
        )
        for group_id, (feature, location) in enumerate(
            zip(
                features, locations.to(device="cpu", dtype=torch.long).view(-1).tolist()
            )
        ):
            if feature is None or feature.numel() == 0:
                continue
            length = feature.size(0)
            if location < 0 or location + length > group_ids.numel():
                raise ValueError(
                    f"visual group {group_id} range [{location}, {location + length}) "
                    f"is outside {group_ids.numel()} input tokens"
                )
            segment = group_ids.narrow(0, location, length)
            if bool((segment >= 0).any()):
                raise ValueError(f"visual group {group_id} overlaps another group")
            segment.fill_(group_id)
        return group_ids

    def prepare_fmha_impl(
        self,
        inputs: PyModelInputs,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode: Optional[str] = None,
    ):
        attention_inputs = get_attention_inputs_value(inputs)
        if isinstance(attention_inputs, Mapping):
            groups = attention_inputs.items()
        else:
            groups = (
                (tag, attention_inputs) for tag in (GEMMA4_TAG_SWA, GEMMA4_TAG_FULL)
            )
        impls = {}
        for tag, group_inputs in groups:
            geometry = self._geometry_for_tag(tag)
            # stock pattern (qwen3_next.py:1546): the flag must be set on the
            # attention inputs BEFORE impl construction so graph-capable impls
            # bind their fixed-address buffers at __init__ (capture happens
            # after this; prepare_cuda_graph only runs at replay time)
            group_inputs.is_cuda_graph = bool(is_cuda_graph)
            impls[tag] = self._create_fmha_impl(geometry, group_inputs)
        return impls

    def _create_fmha_impl(self, geometry: Gemma4LayerGeometry, group_inputs: Any):
        page_size = self._page_size_for_tag(geometry.tag)
        if group_inputs.is_prefill and group_inputs.context_parallel_info is not None:
            from rtp_llm.models_py.modules.gemma4.context_parallel import (
                Gemma4ContextParallelFMHAImpl,
            )

            return Gemma4ContextParallelFMHAImpl(
                self.config, geometry, group_inputs, page_size, self.parallelism_config
            )
        cp_config = self.parallelism_config.prefill_cp_config
        if (
            not group_inputs.is_prefill
            and cp_config.is_enabled()
            and cp_config.kv_cache_sharded
            and self.parallelism_config.tp_size > 1
        ):
            from rtp_llm.models_py.modules.gemma4.context_parallel import (
                Gemma4ContextParallelDecodeFMHAImpl,
            )

            return Gemma4ContextParallelDecodeFMHAImpl(
                self.config, geometry, group_inputs, page_size, self.parallelism_config
            )
        if (
            os.environ.get("RTP_LLM_GEMMA4_ATTN_BACKEND", "flashinfer") == "flashinfer"
            and Gemma4SwaFlashinferImpl.support(geometry)
        ):
            # Opt-in production SWA backend: flashinfer ragged prefill (no
            # prefix) and dense-table eager decode. Graph decode, sparse page
            # tables, prefixed prefill and vision groups fall back to the
            # reference path inside the implementation itself, so the switch
            # only changes the numerics of the phases it actually serves.
            return Gemma4SwaFlashinferImpl(
                self.config,
                geometry,
                group_inputs,
                page_size,
                parallelism_config=self.parallelism_config,
            )
        return Gemma4TorchFMHAImpl(
            self.config,
            geometry,
            group_inputs,
            page_size,
            parallelism_config=self.parallelism_config,
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        embedding_inputs = getattr(inputs, "embedding_inputs", None)
        text_tokens_mask = (
            embedding_inputs.text_tokens_mask if embedding_inputs is not None else None
        )
        hidden_states = self.embed_tokens(input_ids, None, None, text_tokens_mask)
        if (
            hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and hidden_states.numel() % 8 == 0
            and torch.version.hip is None
        ):
            hidden_states = rtp_llm_ops.gemma4_scale_bf16(
                hidden_states, self.embed_scale
            )
        else:
            hidden_states = hidden_states * self.embed_scale
        multimodal_inputs = getattr(inputs, "multimodal_inputs", None)
        if multimodal_inputs is not None and multimodal_inputs.multimodal_features:
            hidden_states = self.multimodal_embedding_injector(
                hidden_states,
                multimodal_inputs.multimodal_features,
                multimodal_inputs.mm_features_locs,
            )
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        # per-forward rope cos/sin cache, keyed by attention tag: layers of
        # one tag share positions and rope table, so cos/sin are computed
        # once per forward instead of once per layer
        rope_cache: Dict[str, tuple] = {}
        # graph-capture mode: the impl dict was built with is_cuda_graph
        # (graph runner contract); mirror the flag onto the MoE experts so
        # they take the device-side grouped path during capture. Mirrored
        # UNCONDITIONALLY (True and False) so capture never leaks into the
        # post-capture non-graph forwards - see _mirror_experts_graph_mode
        graph_mode = isinstance(fmha_impl, Mapping) and any(
            getattr(impl, "_graph_mode", False) for impl in fmha_impl.values()
        )
        implementations = (
            tuple(fmha_impl.values())
            if isinstance(fmha_impl, Mapping)
            else (fmha_impl,)
        )
        visual_group_ids = self._visual_group_ids(inputs)
        for implementation in implementations:
            implementation.vision_group_ids = (
                visual_group_ids
                if implementation.geometry.tag == GEMMA4_TAG_SWA
                and implementation.attn_inputs.is_prefill
                else None
            )
        prefill_modes = {
            bool(implementation.attn_inputs.is_prefill)
            for implementation in implementations
        }
        if len(prefill_modes) != 1:
            raise RuntimeError(
                f"Gemma4 attention groups disagree on prefill mode: {prefill_modes}"
            )
        self._mirror_experts_graph_mode(graph_mode, decode_mode=not prefill_modes.pop())
        for idx, decoder_layer in enumerate(self.layers[: self.layer_num]):
            tag = self._layer_tag(idx)
            if isinstance(fmha_impl, Mapping):
                if tag not in fmha_impl:
                    raise RuntimeError(
                        f"FMHA tag {tag!r} is missing; available={list(fmha_impl)}"
                    )
                layer_fmha_impl = fmha_impl[tag]
            else:
                layer_fmha_impl = fmha_impl
            hidden_states = decoder_layer(
                hidden_states,
                layer_fmha_impl,
                kv_cache=self._layer_kv_cache(idx, tag),
                rope_cache=rope_cache,
            )
        return PyModelOutputs(self.norm(hidden_states))


__all__ = [
    "Gemma4Attention",
    "Gemma4DecoderLayer",
    "Gemma4DenseMLP",
    "Gemma4Experts",
    "Gemma4LayerGeometry",
    "Gemma4Model",
    "Gemma4RMSNorm",
    "Gemma4RopeTable",
    "Gemma4Router",
    "Gemma4SwaFlashinferImpl",
    "Gemma4TorchFMHAImpl",
    "apply_gemma4_rope",
    "build_gemma4_layer_geometry",
    "gemma4_rms_norm",
]

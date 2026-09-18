"""Sparse GQA FMHA impl for qwen4's full-attention layers.

The indexer picks which compressed-block tokens each query may read; this impl
consumes that selection and runs :func:`sparse_prefill_attn
<rtp_llm.models_py.modules.qwen4_exp.sparse_fmha.sparse_prefill_attn>` instead
of the dense attention.

Routing (see ``attn_factory._select_attention_impl_key``): selected only when
``attn_configs.is_sparse`` (indexer enabled), ``not use_mla`` and the explicit
QSA FMHA opt-in is enabled. Production qwen4 serving still rejects QSA before
model loading until the side-cache scorer is end-to-end.

Scope of this restricted production bridge:

* the model-side indexer hands over packed ``[tokens, selected]`` indices;
* main Q/K RoPE and K/V cache writes use the same fused production op as TRT
  prefill, including the common cache-store hook;
* ordinary prefill sparse attention reads this invocation's rotated K/V;
* explicitly opted-in MTP draft incremental prefill reads the full visible
  prefix and candidate window from the main paged K/V cache;
* ordinary decode can read selected tokens from the main paged K/V cache.

Target verification is restricted to a contiguous text-only window no wider
than one QSA compression group. Under that bound, writing the uncommitted
physical tail cannot alias committed indexer state. Prefix-cache reuse, PD, CP
and CUDA Graph remain fail-fast.

The top-level Qwen4 serving gate remains closed until those missing modes and
the indexer's two side pools are integrated.
"""

from typing import Optional

import torch

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs
from rtp_llm.ops.fused_rope_kvcache_op import (
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOpQKVOut,
)


class SparseGqaFmhaImpl(FMHAImplBase):
    """Block-sparse GQA attention for qwen4 prefill and paged decode."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.parallelism_config = parallelism_config
        self.head_num = attn_configs.head_num
        self.kv_head_num = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        if self.head_num % self.kv_head_num != 0:
            raise ValueError(
                f"head_num {self.head_num} not divisible by kv_head_num "
                f"{self.kv_head_num}"
            )
        self.q_width = self.head_num * self.head_dim
        self.kv_width = self.kv_head_num * self.head_dim
        self.is_prefill = bool(attn_inputs.is_prefill)
        self.is_target_verify = bool(attn_inputs.is_target_verify)
        self._is_mtp_draft = False
        self._selected_indices: Optional[torch.Tensor] = None
        self._qsa_main_cache_mutation_started = False
        self._validate_runtime_mode()
        rope_writer_cls = (
            FusedRopeKVCachePrefillOpQKVOut
            if self.is_prefill
            else FusedRopeKVCacheDecodeOp
        )
        self.rope_kvcache_impl = rope_writer_cls(attn_configs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    def _validate_runtime_mode(self) -> None:
        """Keep the correctness bridge inside its deliberately narrow scope."""
        inputs = self.attn_inputs
        if inputs.is_target_verify and not self.is_prefill:
            raise RuntimeError(
                "qwen4_exp sparse GQA target verification must use context-style prefill"
            )
        if inputs.is_cuda_graph:
            raise RuntimeError("qwen4_exp sparse GQA does not support CUDA Graph")
        if inputs.is_s_padded:
            raise RuntimeError("qwen4_exp sparse GQA does not support padded execution")
        if inputs.context_parallel_info is not None:
            raise RuntimeError(
                "qwen4_exp sparse GQA does not support context parallelism"
            )
        if self.is_prefill and inputs.cache_store_inputs is not None:
            raise RuntimeError("qwen4_exp sparse GQA indexer state does not support PD")

    def set_mtp_draft_mode(self, enabled: bool) -> None:
        """Explicitly select the MTP draft incremental-prefill bridge.

        The flag lives on this model-local implementation rather than on the
        pybind ``PyAttentionInputs`` object.  It must be set by the Qwen4 MTP
        model before side-cache validation/writes; a non-zero prefix alone is
        deliberately not treated as sufficient evidence that this mode is
        safe. Prefix-free draft prefill deliberately keeps the ordinary local
        sparse-prefill path; the paged bridge is selected only once a prefix
        exists.
        """
        if not isinstance(enabled, bool):
            raise TypeError("qwen4_exp MTP draft mode must be a bool")
        if enabled and (not self.is_prefill or self.is_target_verify):
            raise RuntimeError(
                "qwen4_exp MTP draft mode only supports ordinary prefill"
            )
        if self._selected_indices is not None:
            raise RuntimeError(
                "qwen4_exp MTP draft mode cannot change with an unconsumed selection"
            )
        self._is_mtp_draft = enabled

    def begin_qsa_cache_transaction(self) -> None:
        """Reset the main-cache phase before the paired side-cache write."""
        self._qsa_main_cache_mutation_started = False

    def qsa_main_cache_mutation_started(self) -> bool:
        """Whether the fused main KV writer may have mutated its cache."""
        return self._qsa_main_cache_mutation_started

    def _is_mtp_incremental_prefill(self) -> bool:
        return (
            self.is_prefill
            and not self.is_target_verify
            and self._is_mtp_draft
            and self._has_nonzero_prefill_prefix()
        )

    def _has_nonzero_prefill_prefix(self) -> bool:
        prefixes = self.attn_inputs.prefix_lengths
        return bool(prefixes.numel()) and bool(torch.any(prefixes != 0).item())

    def _validate_main_paged_geometry(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        sequence_bases: torch.Tensor,
        query_lengths: torch.Tensor,
        device: torch.device,
    ) -> int:
        """Validate all pages that a paged prefill/decode may observe."""
        batch_size = int(sequence_bases.numel())
        if (
            sequence_bases.dim() != 1
            or query_lengths.dim() != 1
            or int(query_lengths.numel()) != batch_size
            or batch_size <= 0
        ):
            raise RuntimeError(
                "qwen4_exp QSA paged sequence metadata has invalid geometry"
            )
        sequence_bases = sequence_bases.to(
            device=device, dtype=torch.int32, non_blocking=True
        )
        query_lengths = query_lengths.to(
            device=device, dtype=torch.int32, non_blocking=True
        )
        if bool(torch.any(sequence_bases < 0).item()) or bool(
            torch.any(query_lengths <= 0).item()
        ):
            raise RuntimeError(
                "qwen4_exp QSA paged sequence bases must be non-negative and "
                "query lengths positive"
            )

        page_size = int(self.attn_configs.kernel_tokens_per_block)
        if page_size <= 0:
            raise RuntimeError(
                f"qwen4_exp sparse GQA has invalid kernel page size {page_size}"
            )
        if page_size & (page_size - 1) or self.head_dim & (self.head_dim - 1):
            raise RuntimeError(
                "qwen4_exp QSA paged attention requires power-of-two page and head sizes"
            )
        if (
            block_table is None
            or block_table.dim() != 2
            or int(block_table.shape[0]) != batch_size
            or block_table.dtype != torch.int32
            or block_table.device != device
        ):
            raise RuntimeError(
                "qwen4_exp QSA paged attention requires a device int32 main block table"
            )

        if cache is None or cache.dtype != torch.bfloat16:
            raise RuntimeError(
                "qwen4_exp QSA paged attention requires a BF16 main KV cache"
            )
        if cache.device != device:
            raise RuntimeError(
                "qwen4_exp QSA paged main cache and activations must share a device"
            )
        if cache.dim() == 2:
            required_width = 2 * self.kv_head_num * page_size * self.head_dim
            if int(cache.shape[1]) < required_width or cache.stride(1) != 1:
                raise RuntimeError(
                    "qwen4_exp QSA paged packed main-cache geometry is invalid"
                )
            cache_blocks = int(cache.shape[0])
        elif cache.dim() == 5:
            if tuple(cache.shape[1:]) != (
                2,
                self.kv_head_num,
                page_size,
                self.head_dim,
            ):
                raise RuntimeError("qwen4_exp QSA paged main-cache geometry is invalid")
            expected_inner_strides = (
                self.kv_head_num * page_size * self.head_dim,
                page_size * self.head_dim,
                self.head_dim,
                1,
            )
            if tuple(cache.stride()[1:]) != expected_inner_strides:
                raise RuntimeError(
                    "qwen4_exp QSA paged main cache must use HND inner layout"
                )
            cache_blocks = int(cache.shape[0])
        else:
            raise RuntimeError(
                "qwen4_exp QSA paged main cache must be packed 2-D or HND 5-D"
            )

        if block_table.numel() and int(block_table.max().item()) >= cache_blocks:
            raise RuntimeError(
                "qwen4_exp QSA paged main block table contains an "
                "out-of-range physical block"
            )
        visible_ends = sequence_bases + query_lengths
        required_columns = (visible_ends + page_size - 1) // page_size
        if bool(torch.any(required_columns > int(block_table.shape[1])).item()):
            raise RuntimeError(
                "qwen4_exp QSA paged main block table does not cover the visible KV"
            )
        columns = torch.arange(
            int(block_table.shape[1]), device=block_table.device
        ).unsqueeze(0)
        required = columns < required_columns.unsqueeze(1)
        invalid = required & ((block_table <= 0) | (block_table >= cache_blocks))
        if bool(torch.any(invalid).item()):
            raise RuntimeError(
                "qwen4_exp QSA paged visible KV resolves to an unallocated "
                "or out-of-range physical block"
            )
        return page_size

    def validate_qsa_before_side_write(
        self,
        qsa_runtime,
        indexer,
        hidden_states: torch.Tensor,
    ) -> None:
        """Validate side and main paged geometry before either cache is written."""
        if self._selected_indices is not None:
            raise RuntimeError("qwen4_exp sparse GQA has an unconsumed selection")
        qsa_runtime.validate_before_projection(
            indexer=indexer,
            token_count=int(hidden_states.shape[0]),
            device=hidden_states.device,
        )
        is_incremental_prefill = self._is_mtp_incremental_prefill()
        if (
            self.is_prefill
            and not self.is_target_verify
            and self._has_nonzero_prefill_prefix()
            and not is_incremental_prefill
        ):
            raise RuntimeError(
                "qwen4_exp sparse GQA non-zero-prefix prefill requires explicit "
                "MTP draft mode"
            )
        if self.is_prefill and not self.is_target_verify and not is_incremental_prefill:
            return
        if qsa_runtime.main_cache is None:
            raise RuntimeError(
                "qwen4_exp QSA paged attention requires the main KV cache"
            )
        if qsa_runtime.main_inputs is not self.attn_inputs:
            def _inputs_brief(inputs) -> str:
                try:
                    prefixes = inputs.prefix_lengths
                    prefix_head = (
                        prefixes[:4].tolist() if prefixes is not None else None
                    )
                except Exception as error:  # pragma: no cover - diagnostics only
                    prefix_head = f"unavailable({error!r})"
                return (
                    f"id={id(inputs)}"
                    f" is_prefill={getattr(inputs, 'is_prefill', None)}"
                    f" is_target_verify={getattr(inputs, 'is_target_verify', None)}"
                    f" prefix_head={prefix_head}"
                )

            raise RuntimeError(
                "qwen4_exp QSA paged main attention inputs are inconsistent"
                f" [runtime {_inputs_brief(qsa_runtime.main_inputs)} |"
                f" impl {_inputs_brief(self.attn_inputs)} |"
                f" impl_flags prefill={self.is_prefill}"
                f" verify={self.is_target_verify}"
                f" mtp_draft={self._is_mtp_draft}"
                f" incremental={is_incremental_prefill}]"
            )
        if hidden_states.dtype != torch.bfloat16:
            raise RuntimeError(
                "qwen4_exp QSA paged attention currently requires BF16 hidden states"
            )
        if not hidden_states.is_cuda:
            raise RuntimeError("qwen4_exp QSA paged attention requires CUDA tensors")

        if self.is_target_verify or is_incremental_prefill:
            sequence_bases = self.attn_inputs.prefix_lengths
            query_lengths = self.attn_inputs.input_lengths
        else:
            sequence_bases = self.attn_inputs.sequence_lengths
            query_lengths = torch.ones_like(sequence_bases)
        if int(hidden_states.shape[0]) != int(query_lengths.sum().item()):
            raise RuntimeError(
                "qwen4_exp QSA paged token count does not match batch/query geometry"
            )
        self._validate_main_paged_geometry(
            qsa_runtime.main_cache.kv_cache_base,
            self.attn_inputs.kv_cache_kernel_block_id_device,
            sequence_bases,
            query_lengths,
            hidden_states.device,
        )

    def set_selected_indices(self, selected_indices: torch.Tensor) -> None:
        """Set one layer's packed selection without widening the global FMHA ABI."""
        if self._selected_indices is not None:
            raise RuntimeError("qwen4_exp sparse GQA has an unconsumed selection")
        self._selected_indices = selected_indices

    def _prepare_paged_inputs(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        selected_indices: torch.Tensor,
    ) -> dict:
        """Validate every paged-reader invariant before the main KV writer runs."""
        if qkv.dim() != 2:
            raise ValueError(
                f"qwen4_exp sparse GQA qkv must be rank 2, got {tuple(qkv.shape)}"
            )
        expected_qkv_width = self.q_width + 2 * self.kv_width
        if int(qkv.shape[1]) != expected_qkv_width:
            raise ValueError(
                "qwen4_exp sparse GQA qkv width must be "
                f"{expected_qkv_width}, got {qkv.shape[1]}"
            )
        if qkv.dtype != torch.bfloat16 or not qkv.is_cuda:
            raise ValueError("qwen4_exp sparse paged GQA requires BF16 CUDA qkv")
        if kv_cache is None or kv_cache.kv_cache_base is None:
            raise RuntimeError("qwen4_exp sparse GQA decode requires the main KV cache")

        if self.is_target_verify:
            lengths_source = self.attn_inputs.prefix_lengths
        else:
            lengths_source = self.attn_inputs.sequence_lengths
        if lengths_source.dim() != 1:
            raise ValueError("qwen4_exp sparse GQA sequence metadata must be rank 1")
        batch = int(lengths_source.numel())
        tokens = int(qkv.shape[0])
        if batch <= 0 or tokens % batch:
            raise ValueError(
                f"qwen4_exp sparse GQA decode tokens={tokens} do not divide batch={batch}"
            )
        query_len = tokens // batch

        if selected_indices.dim() == 2:
            if int(selected_indices.shape[0]) != tokens:
                raise ValueError(
                    "packed decode selected_indices must have one row per token"
                )
            selected_indices = selected_indices.view(
                batch, query_len, int(selected_indices.shape[1])
            )
        if selected_indices.dim() != 3 or tuple(selected_indices.shape[:2]) != (
            batch,
            query_len,
        ):
            raise ValueError(
                "decode selected_indices must be [B, query_len, K], got "
                f"{tuple(selected_indices.shape)}"
            )
        if selected_indices.dtype != torch.int32:
            raise ValueError("decode selected_indices must be int32")
        if selected_indices.device != qkv.device:
            raise ValueError(
                "decode selected_indices and qkv must share one CUDA device"
            )

        if self.is_target_verify:
            input_lengths = self.attn_inputs.input_lengths
            if input_lengths.numel() != batch or bool(
                torch.any(input_lengths != query_len).item()
            ):
                raise ValueError(
                    "qwen4_exp target-verify input lengths must match query_len"
                )
        sequence_lengths = lengths_source.to(
            device=qkv.device, dtype=torch.int32, non_blocking=True
        ).contiguous()
        if bool(torch.any(sequence_lengths < 0).item()):
            raise ValueError(
                "qwen4_exp sparse GQA sequence lengths must be non-negative"
            )
        query_offsets = torch.arange(
            1, query_len + 1, device=qkv.device, dtype=torch.int32
        )
        kv_lens = sequence_lengths.view(batch, 1) + query_offsets.view(1, query_len)

        block_table = self.attn_inputs.kv_cache_kernel_block_id_device
        if (
            block_table is None
            or block_table.dim() != 2
            or tuple(block_table.shape[:1]) != (batch,)
            or block_table.dtype != torch.int32
            or block_table.device != qkv.device
        ):
            raise RuntimeError(
                f"qwen4_exp sparse GQA requires a device int32 [{batch}, max_blocks] table"
            )
        if int(block_table.shape[1]) == 0:
            raise RuntimeError(
                "qwen4_exp sparse GQA block table must contain a logical block"
            )
        page_size = int(self.attn_configs.kernel_tokens_per_block)
        if page_size <= 0 or page_size & (page_size - 1):
            raise RuntimeError(
                "qwen4_exp sparse GQA kernel page size must be a positive power of two"
            )
        if self.head_dim <= 0 or self.head_dim & (self.head_dim - 1):
            raise RuntimeError(
                "qwen4_exp sparse GQA head size must be a positive power of two"
            )

        cache = kv_cache.kv_cache_base
        if cache.dtype != torch.bfloat16 or cache.device != qkv.device:
            raise RuntimeError(
                "qwen4_exp sparse GQA requires a device-local BF16 main KV cache"
            )
        if cache.dim() == 2:
            required_width = 2 * self.kv_head_num * page_size * self.head_dim
            if int(cache.shape[1]) < required_width or cache.stride(1) != 1:
                raise RuntimeError(
                    "qwen4_exp sparse GQA packed main-cache geometry is invalid"
                )
            cache_blocks = int(cache.shape[0])
        elif cache.dim() == 5:
            if tuple(cache.shape[1:]) != (
                2,
                self.kv_head_num,
                page_size,
                self.head_dim,
            ):
                raise RuntimeError(
                    "qwen4_exp sparse GQA main-cache geometry is invalid"
                )
            expected_inner_strides = (
                self.kv_head_num * page_size * self.head_dim,
                page_size * self.head_dim,
                self.head_dim,
                1,
            )
            if tuple(cache.stride()[1:]) != expected_inner_strides:
                raise RuntimeError(
                    "qwen4_exp sparse GQA main cache must use HND inner layout"
                )
            cache_blocks = int(cache.shape[0])
        else:
            raise RuntimeError(
                "qwen4_exp sparse GQA main cache must be packed 2-D or HND 5-D"
            )

        required_columns = (kv_lens[:, -1] + page_size - 1) // page_size
        if bool(torch.any(required_columns > int(block_table.shape[1])).item()):
            raise RuntimeError(
                "qwen4_exp sparse GQA main block table does not cover the visible KV"
            )
        columns = torch.arange(
            int(block_table.shape[1]), device=block_table.device
        ).unsqueeze(0)
        required = columns < required_columns.unsqueeze(1)
        invalid = required & ((block_table <= 0) | (block_table >= cache_blocks))
        if bool(torch.any(invalid).item()):
            raise RuntimeError(
                "qwen4_exp sparse GQA visible KV resolves to an unallocated or "
                "out-of-range physical block"
            )

        if selected_indices.numel():
            if int(selected_indices.min().item()) < -1:
                raise ValueError("selected indices may only use -1 as padding")
            invalid_visible = (selected_indices >= 0) & (
                selected_indices >= kv_lens.unsqueeze(-1)
            )
            if bool(torch.any(invalid_visible).item()):
                raise ValueError(
                    "selected indices contain a token outside the row's visible KV"
                )

        return {
            "batch": batch,
            "query_len": query_len,
            "tokens": tokens,
            "selected_indices": selected_indices,
            "kv_lens": kv_lens,
            "block_table": block_table,
            "page_size": page_size,
        }

    def _prepare_incremental_prefill_inputs(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        selected_indices: torch.Tensor,
    ) -> dict:
        """Build ragged per-request paged plans before the production writer."""
        if not self._is_mtp_incremental_prefill():
            raise RuntimeError(
                "qwen4_exp incremental prefill requires explicit MTP draft mode"
            )
        if qkv.dim() != 2:
            raise ValueError(
                f"qwen4_exp sparse GQA qkv must be rank 2, got {tuple(qkv.shape)}"
            )
        expected_qkv_width = self.q_width + 2 * self.kv_width
        if int(qkv.shape[1]) != expected_qkv_width:
            raise ValueError(
                "qwen4_exp sparse GQA qkv width must be "
                f"{expected_qkv_width}, got {qkv.shape[1]}"
            )
        if qkv.dtype != torch.bfloat16 or not qkv.is_cuda:
            raise ValueError(
                "qwen4_exp sparse incremental prefill requires BF16 CUDA qkv"
            )
        if kv_cache is None or kv_cache.kv_cache_base is None:
            raise RuntimeError(
                "qwen4_exp sparse incremental prefill requires the main KV cache"
            )

        input_lengths = self.attn_inputs.input_lengths
        prefixes = self.attn_inputs.prefix_lengths
        if input_lengths.dim() != 1 or prefixes.dim() != 1:
            raise ValueError(
                "qwen4_exp incremental prefill lengths and prefixes must be rank 1"
            )
        batch = int(input_lengths.numel())
        if batch <= 0 or int(prefixes.numel()) != batch:
            raise ValueError(
                "qwen4_exp incremental prefill lengths and prefixes disagree"
            )
        input_lengths_device = input_lengths.to(
            device=qkv.device, dtype=torch.int32, non_blocking=True
        )
        prefixes_device = prefixes.to(
            device=qkv.device, dtype=torch.int32, non_blocking=True
        )
        if bool(torch.any(input_lengths_device <= 0).item()):
            raise ValueError(
                "qwen4_exp incremental prefill input lengths must be positive"
            )
        if bool(torch.any(prefixes_device < 0).item()):
            raise ValueError(
                "qwen4_exp incremental prefill prefixes must be non-negative"
            )

        cu_seqlens = self.attn_inputs.cu_seqlens_device
        expected_cu = torch.zeros(batch + 1, dtype=torch.int32, device=qkv.device)
        expected_cu[1:] = input_lengths_device.cumsum(0)
        if (
            cu_seqlens is None
            or cu_seqlens.dim() != 1
            or tuple(cu_seqlens.shape) != (batch + 1,)
            or cu_seqlens.dtype != torch.int32
            or cu_seqlens.device != qkv.device
            or not bool(torch.equal(cu_seqlens, expected_cu))
        ):
            raise ValueError(
                "qwen4_exp incremental prefill cu_seqlens do not match input lengths"
            )
        tokens = int(expected_cu[-1].item())
        if int(qkv.shape[0]) != tokens:
            raise ValueError(
                "qwen4_exp incremental prefill cu_seqlens do not partition qkv: "
                f"tokens={qkv.shape[0]}, cu_end={tokens}"
            )
        if selected_indices.dim() != 2 or int(selected_indices.shape[0]) != tokens:
            raise ValueError(
                "qwen4_exp incremental prefill selected_indices must be packed "
                f"[{tokens}, K], got {tuple(selected_indices.shape)}"
            )
        if selected_indices.dtype != torch.int32:
            raise ValueError(
                "qwen4_exp incremental prefill selected_indices must be int32"
            )
        if selected_indices.device != qkv.device:
            raise ValueError(
                "qwen4_exp incremental prefill selection and qkv must share CUDA"
            )

        block_table = self.attn_inputs.kv_cache_kernel_block_id_device
        page_size = self._validate_main_paged_geometry(
            kv_cache.kv_cache_base,
            block_table,
            prefixes_device,
            input_lengths_device,
            qkv.device,
        )

        cu_host = expected_cu.cpu().tolist()
        requests = []
        for request_idx in range(batch):
            start, end = cu_host[request_idx : request_idx + 2]
            query_len = end - start
            kv_lens = prefixes_device[request_idx] + torch.arange(
                1, query_len + 1, dtype=torch.int32, device=qkv.device
            )
            request_selection = selected_indices[start:end]
            if request_selection.numel():
                if int(request_selection.min().item()) < -1:
                    raise ValueError("selected indices may only use -1 as padding")
                invalid_visible = (request_selection >= 0) & (
                    request_selection >= kv_lens.unsqueeze(-1)
                )
                if bool(torch.any(invalid_visible).item()):
                    raise ValueError(
                        "incremental prefill selected indices contain a token "
                        "outside the row's visible KV"
                    )
            requests.append(
                {
                    "start": start,
                    "end": end,
                    "kv_lens": kv_lens.unsqueeze(0).contiguous(),
                    "selected_indices": request_selection.unsqueeze(0).contiguous(),
                }
            )

        return {
            "batch": batch,
            "tokens": tokens,
            "block_table": block_table,
            "page_size": page_size,
            "requests": requests,
        }

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        """Sparse GQA owns explicitly enabled sparse, non-MLA attention."""
        return bool(
            attn_configs.is_sparse
            and not attn_configs.use_mla
            and attn_configs.use_sparse_gqa_fmha
        )

    def support_cuda_graph(self) -> bool:
        # The paged reader still uses dynamic metadata and is not graph-safe.
        return False

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache] = None,
        layer_idx: int = 0,
        selected_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Rotate/write packed QKV, then run per-request sparse prefill attention.

        Production calls :meth:`set_selected_indices` immediately before the
        standard three-argument FMHA call. Focused component tests may pass the
        tensor directly. Both forms use packed ``[total_tokens, K]`` indices;
        the legacy uniform ``[B, S, K]`` form is accepted after validation.
        """
        if selected_indices is not None and self._selected_indices is not None:
            raise RuntimeError("qwen4_exp sparse GQA received two selections")
        uses_stored_selection = selected_indices is None
        if selected_indices is None:
            selected_indices = self._selected_indices
        if selected_indices is None:
            raise ValueError(
                "SparseGqaFmhaImpl requires the indexer's selected_indices; "
                "the attention module must run the indexer and pass it in"
            )

        is_incremental_prefill = self._is_mtp_incremental_prefill()
        if (
            self.is_prefill
            and not self.is_target_verify
            and self._has_nonzero_prefill_prefix()
            and not is_incremental_prefill
        ):
            raise RuntimeError(
                "qwen4_exp sparse GQA non-zero-prefix prefill requires explicit "
                "MTP draft mode"
            )

        paged_plan = None
        if is_incremental_prefill:
            paged_plan = self._prepare_incremental_prefill_inputs(
                qkv, kv_cache, selected_indices
            )
        elif self.is_target_verify or not self.is_prefill:
            paged_plan = self._prepare_paged_inputs(qkv, kv_cache, selected_indices)

        # The fused op writes main K/V. Mark the phase before entering it:
        # even a launch-time exception may follow a partial mutation.
        self._qsa_main_cache_mutation_started = True
        pre_writer_shape = tuple(qkv.shape)
        qkv = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        # The production decode writer returns [tokens, q_heads, head_dim] while
        # the paged kernels consume the flat [tokens, width] view.
        if qkv.dim() == 3 and int(qkv.numel()) == int(qkv.shape[0]) * self.q_width:
            qkv = qkv.reshape(int(qkv.shape[0]), self.q_width)
        if self.is_target_verify:
            if int(qkv.shape[1]) < self.q_width:
                raise RuntimeError(
                    "qwen4_exp target-verify RoPE writer returned an undersized tensor"
                )
            common.apply_write_cache_store(
                self.write_cache_store_impl, self.attn_inputs, kv_cache
            )
            output = self._forward_paged(qkv[:, : self.q_width], kv_cache, paged_plan)
            if uses_stored_selection:
                self._selected_indices = None
            return output
        if is_incremental_prefill:
            if int(qkv.shape[1]) < self.q_width:
                raise RuntimeError(
                    "qwen4_exp incremental-prefill RoPE writer returned an "
                    "undersized tensor"
                )
            common.apply_write_cache_store(
                self.write_cache_store_impl, self.attn_inputs, kv_cache
            )
            output = self._forward_ragged_paged(
                qkv[:, : self.q_width], kv_cache, paged_plan
            )
            if uses_stored_selection:
                self._selected_indices = None
            return output
        if not self.is_prefill:
            if qkv.dim() != 2 or int(qkv.shape[1]) != self.q_width:
                prefixes = getattr(self.attn_inputs, "prefix_lengths", None)
                prefix_head = (
                    prefixes[:4].tolist() if prefixes is not None else None
                )
                raise RuntimeError(
                    "qwen4_exp decode RoPE writer returned invalid query geometry"
                    f" [qkv={tuple(qkv.shape)} in={pre_writer_shape}"
                    f" q_width={self.q_width} kv_width={self.kv_width}"
                    f" dim={qkv.dim()} writer={type(self.rope_kvcache_impl).__name__}"
                    f" verify={self.is_target_verify} draft={self._is_mtp_draft}"
                    f" prefix_head={prefix_head}]"
                )
            common.apply_write_cache_store(
                self.write_cache_store_impl, self.attn_inputs, kv_cache
            )
            output = self._forward_paged(qkv, kv_cache, paged_plan)
            if uses_stored_selection:
                self._selected_indices = None
            return output

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        from rtp_llm.models_py.modules.qwen4_exp.sparse_fmha import sparse_prefill_attn

        tokens = int(qkv.shape[0])
        q, k, v = torch.split(qkv, [self.q_width, self.kv_width, self.kv_width], dim=-1)

        lengths = [int(length) for length in self.attn_inputs.input_lengths.tolist()]
        if (
            not lengths
            or any(length <= 0 for length in lengths)
            or sum(lengths) != tokens
        ):
            raise ValueError(
                f"qwen4_exp sparse GQA input_lengths={lengths} do not partition "
                f"the {tokens} packed tokens"
            )
        if selected_indices.dim() == 3:
            batch, seq_len, width = selected_indices.shape
            if batch != len(lengths) or any(length != seq_len for length in lengths):
                raise ValueError(
                    "rank-3 selected_indices require uniform input lengths matching "
                    f"[{batch}, {seq_len}], got {lengths}"
                )
            selected_indices = selected_indices.reshape(tokens, width)
        if selected_indices.dim() != 2 or int(selected_indices.shape[0]) != tokens:
            raise ValueError(
                "selected_indices must be packed [total_tokens, K], got "
                f"{tuple(selected_indices.shape)} for total_tokens={tokens}"
            )

        outputs = []
        offset = 0
        for seq_len in lengths:
            end = offset + seq_len
            seq_q = q[offset:end].reshape(1, seq_len, self.head_num, self.head_dim)
            seq_k = k[offset:end].reshape(1, seq_len, self.kv_head_num, self.head_dim)
            seq_v = v[offset:end].reshape(1, seq_len, self.kv_head_num, self.head_dim)
            seq_out = sparse_prefill_attn(
                seq_q.transpose(1, 2).contiguous(),
                seq_k.transpose(1, 2).contiguous(),
                seq_v.transpose(1, 2).contiguous(),
                selected_indices[offset:end].unsqueeze(0).contiguous(),
            )
            outputs.append(seq_out.transpose(1, 2).reshape(seq_len, self.q_width))
            offset = end
        output = torch.cat(outputs, dim=0)
        if uses_stored_selection:
            self._selected_indices = None
        return output

    def _forward_paged(
        self,
        q: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        plan: dict,
    ) -> torch.Tensor:
        """Run paged sparse GQA after the production decode RoPE/KV writer."""
        assert kv_cache is not None
        batch = int(plan["batch"])
        query_len = int(plan["query_len"])
        tokens = int(q.shape[0])
        if tokens != int(plan["tokens"]) or int(q.shape[1]) != self.q_width:
            raise ValueError(
                "decode RoPE writer returned query geometry inconsistent with preflight"
            )
        selected_indices = plan["selected_indices"]
        kv_lens = plan["kv_lens"]
        block_table = plan["block_table"]
        page_size = int(plan["page_size"])

        from rtp_llm.models_py.modules.qwen4_exp.sparse_paged_fmha import (
            sparse_paged_gqa_attn,
        )

        q = q.view(batch, query_len, self.head_num, self.head_dim).transpose(1, 2)
        output = sparse_paged_gqa_attn(
            q.contiguous(),
            kv_cache.kv_cache_base,
            block_table,
            kv_lens,
            selected_indices.contiguous(),
            page_size=page_size,
            kv_head_num=self.kv_head_num,
        )
        return output.transpose(1, 2).reshape(tokens, self.q_width)

    def _forward_ragged_paged(
        self,
        q: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        plan: dict,
    ) -> torch.Tensor:
        """Run one B=1 paged call per packed incremental-prefill request."""
        assert kv_cache is not None
        tokens = int(plan["tokens"])
        if q.dim() != 2 or tuple(q.shape) != (tokens, self.q_width):
            raise ValueError(
                "incremental-prefill RoPE writer returned query geometry "
                "inconsistent with preflight"
            )

        from rtp_llm.models_py.modules.qwen4_exp.sparse_paged_fmha import (
            sparse_paged_gqa_attn,
        )

        block_table = plan["block_table"]
        page_size = int(plan["page_size"])
        outputs = []
        for request_idx, request in enumerate(plan["requests"]):
            start, end = int(request["start"]), int(request["end"])
            query_len = end - start
            request_q = q[start:end].view(1, query_len, self.head_num, self.head_dim)
            request_q = request_q.transpose(1, 2).contiguous()
            request_output = sparse_paged_gqa_attn(
                request_q,
                kv_cache.kv_cache_base,
                block_table[request_idx : request_idx + 1].contiguous(),
                request["kv_lens"],
                request["selected_indices"],
                page_size=page_size,
                kv_head_num=self.kv_head_num,
            )
            outputs.append(
                request_output.transpose(1, 2).reshape(query_len, self.q_width)
            )
        return torch.cat(outputs, dim=0)

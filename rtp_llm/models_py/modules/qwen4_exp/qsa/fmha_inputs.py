"""fmha_inputs; validation/layout planning does not own mutation state."""

import itertools
from typing import Optional

import torch

from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_cache_views import (
    qsa_assert_valid,
    qsa_strict_validation_enabled,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_step_plan import (
    host_input_mirror,
    prefill_mirror_facts,
    prefill_step_plan_hit,
)
from rtp_llm.ops.compute_ops import LayerKVCache

from .fmha_plans import IncrementalPrefillInputs, PagedInputs, PrefillRequest


def _prepare_paged_inputs(
    self,
    qkv: torch.Tensor,
    kv_cache: Optional[LayerKVCache],
    selected_indices: torch.Tensor,
) -> PagedInputs:
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
        raise ValueError("decode selected_indices and qkv must share one CUDA device")

    if self.is_target_verify:
        # Host-mirror comparison: the pinned engine source is free, and a
        # device source (RTP_LLM_DEVICE_INPUT) is read back once per model
        # step by the shared mirror cache instead of synchronizing the device
        # copy in every layer.
        input_lengths = host_input_mirror(self.attn_inputs, "input_lengths")
        if input_lengths is None:
            input_lengths = self.attn_inputs.input_lengths
        if input_lengths.numel() != batch or bool(
            torch.any(input_lengths != query_len).item()
        ):
            raise ValueError(
                "qwen4_exp target-verify input lengths must match query_len"
            )
    graph_paged = bool(
        self.attn_inputs.is_cuda_graph
        and (not self.is_prefill or self.is_target_verify)
    )
    if graph_paged and lengths_source.device.type != "cpu":
        raise RuntimeError("qwen4_exp sparse GQA Graph requires host lengths")
    if graph_paged and bool(torch.any(lengths_source < 0).item()):
        raise ValueError("qwen4_exp sparse GQA sequence lengths must be non-negative")
    # Host mirror of the length source shared by the sign check below and the
    # capacity preflight: the pinned engine source passes through for free,
    # and a device source (RTP_LLM_DEVICE_INPUT) is read back once per model
    # step by the mirror cache instead of synchronizing the device copy in
    # every layer.
    lengths_mirror = host_input_mirror(
        self.attn_inputs,
        "prefix_lengths" if self.is_target_verify else "sequence_lengths",
    )
    if lengths_mirror is None:
        lengths_mirror = lengths_source
    cached = getattr(self, "_qsa_target_visible_lengths", None)
    self._qsa_target_visible_lengths = None
    if graph_paged and self.is_target_verify and cached is not None:
        owner_inputs, owner_tokens, kv_lens = cached
        if not (
            owner_inputs is self.attn_inputs
            and owner_tokens == tokens
            and isinstance(kv_lens, torch.Tensor)
            and kv_lens.device == qkv.device
            and kv_lens.dtype == torch.int32
            and tuple(kv_lens.shape) == (batch, query_len)
            and kv_lens.is_contiguous()
        ):
            raise RuntimeError("qwen4_exp main target metadata owner/geometry changed")
    else:
        sequence_lengths = lengths_source.to(
            device=qkv.device, dtype=torch.int32, non_blocking=True
        ).contiguous()
        if not graph_paged:
            if lengths_mirror.device.type == "cpu":
                invalid_lengths = bool(torch.any(lengths_mirror < 0).item())
            else:
                invalid_lengths = bool(torch.any(sequence_lengths < 0).item())
            if invalid_lengths:
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
    if (
        cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
        or cache.device != qkv.device
    ):
        raise RuntimeError(
            "qwen4_exp sparse GQA requires a device-local BF16 or FP8 E4M3 main KV cache"
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
            raise RuntimeError("qwen4_exp sparse GQA main-cache geometry is invalid")
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

    if not graph_paged:
        required_columns = (kv_lens[:, -1] + page_size - 1) // page_size
    # Capacity preflight on the host mirror: the graph branch already computed
    # it from the pinned host lengths, and the eager branch now does the same
    # (the host formula equals the legacy device ``required_columns``
    # reduction) instead of a per-layer device readback.
    if lengths_mirror.device.type == "cpu":
        exceeds_capacity = bool(
            torch.any(
                (lengths_mirror + query_len + page_size - 1) // page_size
                > int(block_table.shape[1])
            ).item()
        )
    elif graph_paged:
        exceeds_capacity = bool(
            torch.any(
                (lengths_source + query_len + page_size - 1) // page_size
                > int(block_table.shape[1])
            ).item()
        )
    else:
        exceeds_capacity = bool(
            torch.any(required_columns > int(block_table.shape[1])).item()
        )
    if exceeds_capacity:
        raise RuntimeError(
            "qwen4_exp sparse GQA main block table does not cover the visible KV"
        )
    if graph_paged:
        return PagedInputs(
            **{
                "batch": batch,
                "query_len": query_len,
                "tokens": tokens,
                "selected_indices": selected_indices,
                "kv_lens": kv_lens,
                "block_table": block_table,
                "page_size": page_size,
            }
        )
    columns = torch.arange(
        int(block_table.shape[1]), device=block_table.device
    ).unsqueeze(0)
    required = columns < required_columns.unsqueeze(1)
    invalid = required & ((block_table <= 0) | (block_table >= cache_blocks))
    if qsa_strict_validation_enabled():
        if bool(torch.any(invalid).item()):
            raise RuntimeError(
                "qwen4_exp sparse GQA visible KV resolves to an unallocated or "
                "out-of-range physical block"
            )
    else:
        # Engine/allocator invariant: one enqueue-only device verdict replaces
        # the per-layer host-synchronizing scalar read; strict mode keeps the
        # exact legacy diagnostics.
        qsa_assert_valid(
            ~torch.any(invalid),
            "qwen4_exp sparse GQA visible KV resolves to an unallocated or "
            "out-of-range physical block",
        )

    if qsa_strict_validation_enabled() and selected_indices.numel():
        # The per-request selection re-check duplicates the FMHA entry
        # validation (_validate_sparse_paged_indices is a superset); strict
        # mode keeps it for its exact per-request diagnostics.
        if int(selected_indices.min().item()) < -1:
            raise ValueError("selected indices may only use -1 as padding")
        invalid_visible = (selected_indices >= 0) & (
            selected_indices >= kv_lens.unsqueeze(-1)
        )
        if bool(torch.any(invalid_visible).item()):
            raise ValueError(
                "selected indices contain a token outside the row's visible KV"
            )

    return PagedInputs(
        **{
            "batch": batch,
            "query_len": query_len,
            "tokens": tokens,
            "selected_indices": selected_indices,
            "kv_lens": kv_lens,
            "block_table": block_table,
            "page_size": page_size,
        }
    )


def _prepare_incremental_prefill_inputs(
    self,
    qkv: torch.Tensor,
    kv_cache: Optional[LayerKVCache],
    selected_indices: torch.Tensor,
) -> IncrementalPrefillInputs:
    """Build ragged per-request paged plans before the production writer.

    The layer-independent planning metadata (device length/prefix mirrors, cu
    boundaries, per-request visible-KV rows) is frozen once per model step in
    a ``Qwen4ExpPrefillStepPlan`` by the QSA geometry pass; on a plan hit
    this per-layer planner consumes those fields and only rebuilds what is
    actually layer-local (the selected-index views). Without a plan it falls
    back to the legacy inline build.
    """
    if not self._is_prefix_reuse_prefill():
        raise RuntimeError("qwen4_exp incremental prefill requires an explicit prefix")
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
        raise ValueError("qwen4_exp sparse incremental prefill requires BF16 CUDA qkv")
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
        raise ValueError("qwen4_exp incremental prefill lengths and prefixes disagree")
    # Request-data checks: one frozen facts pass per step (non-strict path)
    # replaces the per-layer mirror reads; a fingerprint-only plan hit then
    # serves the frozen device mirrors and cu boundaries.  Strict mode and
    # the no-plan fallback keep the legacy per-call reads below.
    step = None
    facts = (
        None
        if qsa_strict_validation_enabled()
        else prefill_mirror_facts(self.attn_inputs)
    )
    if facts is not None:
        if not facts.lengths_positive:
            raise ValueError("qwen4_exp incremental prefill input lengths must be positive")
        if not facts.prefixes_nonneg:
            raise ValueError("qwen4_exp incremental prefill prefixes must be non-negative")
        step = prefill_step_plan_hit(
            self.attn_inputs,
            facts.lengths_sum,
            device=qkv.device,
            batch_size=batch,
        )
        if (
            step is not None
            and step.kv_lens_rows is not None
            and step.cu_host is not None
            and step.input_lengths_i32 is not None
            and step.prefixes_i32 is not None
        ):
            input_lengths_device = step.input_lengths_i32
            prefixes_device = step.prefixes_i32
            cu_host = step.cu_host
        else:
            step = None
    if step is None:
        # Request-data checks run on the per-step CPU mirrors; a device source
        # (RTP_LLM_DEVICE_INPUT) was already read back once this step by the
        # gate/plan build, so these fail-fast checks stay host-side and free.
        lengths_mirror = host_input_mirror(self.attn_inputs, "input_lengths")
        prefix_mirror = host_input_mirror(self.attn_inputs, "prefix_lengths")
        if bool(torch.any(lengths_mirror <= 0)):
            raise ValueError("qwen4_exp incremental prefill input lengths must be positive")
        if bool(torch.any(prefix_mirror < 0)):
            raise ValueError("qwen4_exp incremental prefill prefixes must be non-negative")
        lengths_list = lengths_mirror.tolist()
        prefix_list = prefix_mirror.tolist()
        input_lengths_device = input_lengths.to(
            device=qkv.device, dtype=torch.int32, non_blocking=True
        )
        prefixes_device = prefixes.to(
            device=qkv.device, dtype=torch.int32, non_blocking=True
        )
        # Host cumsum on the pinned mirrors replaces the device readback.
        cu_host = list(itertools.accumulate(lengths_list, initial=0))

    cu_seqlens = self.attn_inputs.cu_seqlens_device
    if (
        cu_seqlens is None
        or cu_seqlens.dim() != 1
        or tuple(cu_seqlens.shape) != (batch + 1,)
        or cu_seqlens.dtype != torch.int32
        or cu_seqlens.device != qkv.device
    ):
        raise ValueError(
            "qwen4_exp incremental prefill cu_seqlens do not match input lengths"
        )
    if step is None:
        expected_cu = torch.zeros(batch + 1, dtype=torch.int32, device=qkv.device)
        expected_cu[1:] = input_lengths_device.cumsum(0)
        if qsa_strict_validation_enabled():
            if not bool(torch.equal(cu_seqlens, expected_cu)):
                raise ValueError(
                    "qwen4_exp incremental prefill cu_seqlens do not match input lengths"
                )
        else:
            qsa_assert_valid(
                (cu_seqlens == expected_cu).all(),
                "qwen4_exp incremental prefill cu_seqlens do not match input lengths",
            )
    tokens = cu_host[-1]
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
        raise ValueError("qwen4_exp incremental prefill selected_indices must be int32")
    if selected_indices.device != qkv.device:
        raise ValueError(
            "qwen4_exp incremental prefill selection and qkv must share CUDA"
        )

    if qsa_strict_validation_enabled():
        # The production path validated this same cache/block_table/geometry in
        # validate_qsa_before_side_write before the projection; the repeat
        # here is redundant. Strict mode keeps it for direct component calls.
        block_table = self.attn_inputs.kv_cache_kernel_block_id_device
        page_size = self._validate_main_paged_geometry(
            kv_cache.kv_cache_base,
            block_table,
            prefixes_device,
            input_lengths_device,
            qkv.device,
        )
    else:
        block_table = self.attn_inputs.kv_cache_kernel_block_id_device
        page_size = int(self.attn_configs.kernel_tokens_per_block)

    requests = []
    for request_idx in range(batch):
        start, end = cu_host[request_idx : request_idx + 2]
        query_len = end - start
        if step is not None:
            # The frozen [1, length] KV rows need no per-layer reshape; the
            # strict per-request diagnostics below never run on a plan hit.
            kv_lens_row = step.kv_lens_rows[request_idx]
        else:
            kv_lens_vec = prefixes_device[request_idx] + torch.arange(
                1, query_len + 1, dtype=torch.int32, device=qkv.device
            )
            kv_lens_row = kv_lens_vec.unsqueeze(0).contiguous()
        request_selection = selected_indices[start:end]
        if qsa_strict_validation_enabled() and request_selection.numel():
            # The per-request selection re-check duplicates the FMHA entry
            # validation (_validate_sparse_paged_indices is a superset);
            # strict mode keeps it for its exact per-request diagnostics.
            if int(request_selection.min().item()) < -1:
                raise ValueError("selected indices may only use -1 as padding")
            invalid_visible = (request_selection >= 0) & (
                request_selection >= kv_lens_vec.unsqueeze(-1)
            )
            if bool(torch.any(invalid_visible).item()):
                raise ValueError(
                    "incremental prefill selected indices contain a token "
                    "outside the row's visible KV"
                )
        requests.append(
            PrefillRequest(
                start,
                end,
                kv_lens_row,
                request_selection.unsqueeze(0).contiguous(),
            )
        )

    return IncrementalPrefillInputs(
        **{
            "batch": batch,
            "tokens": tokens,
            "block_table": block_table,
            "page_size": page_size,
            "requests": requests,
            "selected_indices": selected_indices,
            "prefixes_device": prefixes_device,
            "step_plan": step,
        }
    )

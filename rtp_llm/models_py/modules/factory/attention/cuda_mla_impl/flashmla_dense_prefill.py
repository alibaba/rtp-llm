"""Dense causal MLA Prefill backed by FlashMLA's SM100 kernel.

This adapter keeps RTP's existing projections, RoPE/no-RoPE handling, cache
write and output projection unchanged; only the dense causal attention core is
replaced.
"""

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, cast

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_forward_plan import (
    FlashMLAForwardPlan,
    FlashMLAForwardRoute,
    FlashMLAPrefixLaunch,
    plan_flashmla_forward,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_state_merge import (
    merge_attention_states_segmented_in_place,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_page_rr_cache import (
    MlaPageRRCacheAdapter,
    MlaPageRRChunkDescriptor,
)
from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory
from rtp_llm.models_py.modules.factory.linear.linear_base import LinearBase
from rtp_llm.models_py.modules.factory.linear.quantized_activation import retained_bf16
from rtp_llm.ops import KvCacheDataType
from rtp_llm.ops.compute_ops import LayerKVCache, rtp_llm_ops
from rtp_llm.utils.model_weight import W

_FLASHMLA_WORKSPACES: Dict[int, torch.Tensor] = {}
_K3_PACKED_KV_HEAD_SPLITS = (128, 64, 128)
logger = logging.getLogger(__name__)
_LOGGED_SMOKE_PREFIX_PLANS: set[tuple[object, ...]] = set()


def _log_page_rr_prefix_plan_once(
    plan: FlashMLAForwardPlan,
    adapter: Optional[MlaPageRRCacheAdapter],
    *,
    q_lens: Iterable[int],
    prefix_lens: Iterable[int],
    alignment_tokens: int,
) -> None:
    """Emit host-only evidence that Page-RR used the chunked prefix path."""
    if (
        os.environ.get("KIMI_K3_SMOKE_EVIDENCE") != "1"
        or adapter is None
        or adapter.shard_rank != 0
        or plan.route is not FlashMLAForwardRoute.HYBRID
    ):
        return
    launch_tokens = tuple(launch.expanded_kv_tokens for launch in plan.prefix_launches)
    q_lens = tuple(q_lens)
    prefix_lens = tuple(prefix_lens)
    key = (
        adapter.shard_size,
        q_lens,
        prefix_lens,
        plan.capacity_tokens,
        alignment_tokens,
        launch_tokens,
    )
    if key in _LOGGED_SMOKE_PREFIX_PLANS:
        return
    _LOGGED_SMOKE_PREFIX_PLANS.add(key)
    logger.info(
        "[K3_SMOKE_EVENT] %s",
        json.dumps(
            {
                "kind": "mla_prefix",
                "backend": "page_rr",
                "route": plan.route.value,
                "tp": adapter.shard_size,
                "query_tokens": sum(q_lens),
                "prefix_tokens": sum(prefix_lens),
                "capacity_tokens": plan.capacity_tokens,
                "alignment_tokens": alignment_tokens,
                "launch_tokens": list(launch_tokens),
            },
            sort_keys=True,
        ),
    )


def _workspace(device: torch.device) -> torch.Tensor:
    """Return FlashMLA's reusable 32 MiB inference workspace per device."""

    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    workspace = _FLASHMLA_WORKSPACES.get(device_index)
    if workspace is None:
        workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
        _FLASHMLA_WORKSPACES[device_index] = workspace
    return workspace


@dataclass(frozen=True, slots=True)
class _FlashMLAPrefixRuntimeLaunch:
    spec: FlashMLAPrefixLaunch
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    gather_qo_indptr: torch.Tensor
    batch_reuse_info: torch.Tensor
    destination_starts: torch.Tensor
    q_range: Optional[tuple[int, int]]
    max_q_len: int
    max_kv_len: int
    page_rr_descriptor: Optional[MlaPageRRChunkDescriptor]


@dataclass
class _FlashMLAForwardWorkspace:
    """Buffers reused on one stream across prefix launches and model layers.

    A new plan releases these buffers. The returned attention output aliases
    this workspace and must be consumed before the next layer's forward.
    """

    compressed_kv: torch.Tensor
    packed_kv: torch.Tensor
    packed_q: Optional[torch.Tensor]
    attention_output: torch.Tensor
    attention_lse_storage: torch.Tensor
    output_bf16: torch.Tensor
    fp32_output: Optional[torch.Tensor]
    canonical_lse: torch.Tensor
    num_heads: int

    @classmethod
    def allocate(
        cls,
        *,
        plan: FlashMLAForwardPlan,
        compressed_kv: torch.Tensor,
        q: torch.Tensor,
        kv_lora_rank: int,
        num_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        packed_q_tokens: int,
    ) -> "_FlashMLAForwardWorkspace":
        compressed_kv = retained_bf16(compressed_kv)
        q_tokens = q.shape[0]
        packed_features = num_heads * (qk_head_dim + v_head_dim)
        expanded_kv_tokens = max(q_tokens, plan.max_expanded_kv_tokens)
        return cls(
            compressed_kv=compressed_kv.new_empty(
                (plan.max_expanded_kv_tokens, kv_lora_rank)
            ),
            packed_kv=compressed_kv.new_empty((expanded_kv_tokens, packed_features)),
            packed_q=(
                q.new_empty((packed_q_tokens, *q.shape[1:]))
                if packed_q_tokens
                else None
            ),
            attention_output=q.new_empty(
                (
                    plan.max_partial_state_tokens,
                    num_heads,
                    v_head_dim,
                )
            ),
            attention_lse_storage=torch.empty(
                plan.max_partial_state_tokens * num_heads,
                dtype=torch.float32,
                device=q.device,
            ),
            output_bf16=q.new_empty((q_tokens, num_heads, v_head_dim)),
            fp32_output=(
                torch.empty(
                    (q_tokens, num_heads, v_head_dim),
                    dtype=torch.float32,
                    device=q.device,
                )
                if plan.requires_fp32_accumulator
                else None
            ),
            canonical_lse=torch.empty(
                (num_heads, q_tokens),
                dtype=torch.float32,
                device=q.device,
            ).transpose(0, 1),
            num_heads=num_heads,
        )

    def compressed_kv_buffer(self, tokens: int) -> torch.Tensor:
        return self.compressed_kv.narrow(0, 0, tokens)

    def packed_kv_buffer(self, tokens: int) -> torch.Tensor:
        return self.packed_kv.narrow(0, 0, tokens)

    def q_buffer(self, tokens: int) -> torch.Tensor:
        return cast(torch.Tensor, self.packed_q).narrow(0, 0, tokens)

    def attention_buffers(self, tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.attention_output.narrow(0, 0, tokens)
        # FlashMLA requires head-major LSE with the current launch's token stride.
        lse = torch.as_strided(
            self.attention_lse_storage,
            size=(tokens, self.num_heads),
            stride=(1, tokens),
        )
        return output, lse


def _prefix_sum(lengths: Iterable[int]) -> list[int]:
    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return values


class FlashMLADeviceParams:
    """Device-resident metadata consumed by dense FlashMLA Prefill.

    This deliberately is not a ``FlashInferMlaAttnParams``.  In particular it
    owns no pinned host staging buffer whose lifetime can end when the draft
    model's next decode forward replaces ``held_attn_pyobj_``.
    """

    def __init__(
        self,
        *,
        attn_inputs: Any,
        q_lens_host: List[int],
        kv_lens_host: List[int],
        prefix_lens_host: List[int],
        qo_indptr_d: torch.Tensor,
        kv_indptr_d: torch.Tensor,
        positions_d: torch.Tensor,
        batch_indice_d: torch.Tensor,
        batch_reuse_info_vec_d: torch.Tensor,
        batch_reuse_info_host: tuple[tuple[int, int, int, int], ...],
    ) -> None:
        self.attn_inputs = attn_inputs
        self.q_lens_host = q_lens_host
        self.kv_lens_host = kv_lens_host
        self.prefix_lens_host = prefix_lens_host
        self.qo_indptr_d = qo_indptr_d
        self.prefill_ragged_kv_len_indptr_d = kv_indptr_d
        self.positions_d = positions_d
        self.batch_indice_d = batch_indice_d
        self.batch_reuse_info_vec_d = batch_reuse_info_vec_d
        self.batch_reuse_info_host = batch_reuse_info_host
        self.has_reuse_cache = any(prefix_lens_host)
        # MlaKVCacheWriteOp asks the wrapper to derive this from live device
        # positions and the currently selected HybridCache group.
        self.slot_mapping = None


def _host_i32_values(tensor: torch.Tensor) -> List[int]:
    return [int(value) for value in tensor.tolist()]


def build_flashmla_device_params(
    attn_inputs: Any,
    page_size: int,
) -> FlashMLADeviceParams:
    """Build dense-prefill metadata without FlashInfer's pinned ``buf_h``.

    Host mirrors are used only for shape scalars required by FlashMLA's Python
    API.  Indptrs, positions, reuse metadata and page IDs stay on CUDA.  The
    fixed-width branch is the K3 MTP draft-prefill hot path (currently q_len=4).
    """

    input_lengths_d = attn_inputs.input_lengths
    prefix_lengths_d = attn_inputs.prefix_lengths
    q_lens = _host_i32_values(attn_inputs.input_lengths_host)
    prefix_lens = _host_i32_values(attn_inputs.prefix_lengths_host)
    batch_size = len(q_lens)

    total_q = sum(q_lens)
    kv_lens = [
        q_len + prefix_len
        for q_len, prefix_len in zip(q_lens, prefix_lens, strict=True)
    ]
    qo_indptr_d = attn_inputs.cu_seqlens
    kv_indptr_d = attn_inputs.cu_kv_seqlens

    device = input_lengths_d.device
    packed_indices = torch.arange(total_q, dtype=torch.int32, device=device)
    fixed_q_len = q_lens[0] if all(q_len == q_lens[0] for q_len in q_lens) else 0
    if fixed_q_len:
        # K3 draft-prefill: q_len == propose_step + 1, currently 4.
        batch_indice_d = torch.div(packed_indices, fixed_q_len, rounding_mode="floor")
        local_positions = torch.remainder(packed_indices, fixed_q_len)
    else:
        padding_offset = attn_inputs.padding_offset
        max_q_len = max(q_lens)
        padded_indices = packed_indices + padding_offset
        batch_indice_d = torch.div(padded_indices, max_q_len, rounding_mode="floor")
        local_positions = torch.remainder(padded_indices, max_q_len)
    positions_d = (
        prefix_lengths_d.index_select(0, batch_indice_d.to(torch.int64))
        + local_positions
    )

    block_table = getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)
    if block_table is None or block_table.numel() == 0:
        block_table = torch.empty((batch_size, 0), dtype=torch.int32, device=device)
    max_blocks = int(block_table.shape[1])

    batch_ids_d = torch.arange(batch_size, dtype=torch.int32, device=device)
    page_counts_d = torch.div(
        prefix_lengths_d + page_size - 1,
        page_size,
        rounding_mode="floor",
    )
    batch_reuse_info_vec_d = torch.stack(
        (
            batch_ids_d,
            prefix_lengths_d,
            batch_ids_d * max_blocks,
            page_counts_d,
        ),
        dim=1,
    )
    batch_reuse_info_host = tuple(
        (
            batch_idx,
            prefix_len,
            batch_idx * max_blocks,
            (prefix_len + page_size - 1) // page_size,
        )
        for batch_idx, prefix_len in enumerate(prefix_lens)
    )
    # Async MTP preparation can allocate these tensors on a producer stream.
    # Record their main-forward consumer stream before the producer-owned
    # PyAttentionInputs (or the transient plan) can be replaced next step.
    current_stream = torch.cuda.current_stream(device)
    for tensor in (
        input_lengths_d,
        prefix_lengths_d,
        qo_indptr_d,
        kv_indptr_d,
        block_table,
        positions_d,
        batch_indice_d,
        batch_reuse_info_vec_d,
    ):
        tensor.record_stream(current_stream)

    return FlashMLADeviceParams(
        attn_inputs=attn_inputs,
        q_lens_host=q_lens,
        kv_lens_host=kv_lens,
        prefix_lens_host=prefix_lens,
        qo_indptr_d=qo_indptr_d,
        kv_indptr_d=kv_indptr_d,
        positions_d=positions_d,
        batch_indice_d=batch_indice_d,
        batch_reuse_info_vec_d=batch_reuse_info_vec_d,
        batch_reuse_info_host=batch_reuse_info_host,
    )


class MlaFlashMLAPrefillOp:
    """FlashMLA dense-varlen Prefill core for the generic RTP MLA pipeline."""

    def __init__(
        self,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        v_head_dim: int,
        kernel_page_tokens: int,
        prefix_chunk_alignment_tokens: int,
        softmax_extra_scale: float,
        use_mla: bool,
        weights: List[Dict[str, torch.Tensor]] | None,
        quant_config: Optional[object] = None,
        kv_cache_dtype: KvCacheDataType = KvCacheDataType.BASE,
        expanded_kv_budget_gib: float = 0.0,
        fp8_compute: bool = False,
        q_scale: float = 1.0,
        kv_scale: float = 1.0,
        page_rr_cache_adapter: Optional[MlaPageRRCacheAdapter] = None,
        prepared_kv_b_projections=None,
    ) -> None:
        if weights is None:
            raise ValueError("FlashMLA Prefill requires MLA projection weights")
        kernel_page_tokens = int(kernel_page_tokens)
        prefix_chunk_alignment_tokens = int(prefix_chunk_alignment_tokens)
        if kernel_page_tokens <= 0 or prefix_chunk_alignment_tokens <= 0:
            raise ValueError(
                "kernel page tokens and prefix chunk alignment must be positive"
            )
        if prefix_chunk_alignment_tokens % kernel_page_tokens:
            raise ValueError(
                "prefix chunk alignment must be divisible by kernel page tokens"
            )
        expanded_kv_budget_gib = float(expanded_kv_budget_gib)
        if not math.isfinite(expanded_kv_budget_gib) or expanded_kv_budget_gib < 0:
            raise ValueError(
                "FlashMLA expanded KV GiB budget must be non-negative and finite"
            )
        self.fp8_compute = fp8_compute
        self.q_scale = q_scale
        self.kv_scale = kv_scale
        if fp8_compute:
            if kv_cache_dtype != KvCacheDataType.FP8:
                raise ValueError("FP8 MLA Prefill requires ordinary E4M3 cache")
            from .tokenspeed_mla_impl import (
                _ensure_tokenspeed_cutlass_compat,
                _is_tokenspeed_blackwell,
            )

            if not _is_tokenspeed_blackwell():
                raise RuntimeError("K3 FP8 MLA Prefill requires SM100 or SM103")
            _ensure_tokenspeed_cutlass_compat()
            from tokenspeed_mla.mla_prefill import (
                tokenspeed_mla_prefill,
                warmup_compile_prefill,
            )

            # Cached by the dependency; prepare causal and prefix variants
            # during initialization, before a real request can reach either.
            warmup_compile_prefill(
                q_dtype=torch.float8_e4m3fn,
                d_qk=qk_nope_head_dim + qk_rope_head_dim,
                d_v=v_head_dim,
                enable_pdl=False,
            )
            self.tokenspeed_prefill = tokenspeed_mla_prefill
            self.flash_mla_cuda = None
        else:
            if kv_cache_dtype != KvCacheDataType.BASE:
                raise ValueError(
                    "dense FlashMLA Prefill currently requires BF16 KV cache"
                )
            try:
                import flash_mla.cuda as flash_mla_cuda
            except ImportError as error:
                raise RuntimeError(
                    "dense FlashMLA Prefill requires a compatible flash-mla package"
                ) from error
            self.flash_mla_cuda = flash_mla_cuda
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.kernel_page_tokens = kernel_page_tokens
        self.prefix_chunk_alignment_tokens = prefix_chunk_alignment_tokens
        self.expanded_kv_budget_gib = expanded_kv_budget_gib
        self.page_rr_cache_adapter = page_rr_cache_adapter
        if (
            page_rr_cache_adapter is not None
            and page_rr_cache_adapter.page_tokens != prefix_chunk_alignment_tokens
        ):
            raise ValueError(
                "Page-RR physical page size must match prefix chunk alignment"
            )
        self.scale = (
            (qk_nope_head_dim + qk_rope_head_dim) ** -0.5
        ) * softmax_extra_scale
        self.weights = weights
        self.quant_config = quant_config
        # K3 supplies model-owned projections. Other MLA models retain their
        # existing lazy construction instead of preparing every layer here.
        self._kv_b_projections = prepared_kv_b_projections
        self._prefix_producer = None
        if quant_config is not None and quant_config.get_method() == "FP8_PER_BLOCK":
            from .mla_prefix_fp8_producer import Fp8MlaPrefixGather

            self._prefix_producer = Fp8MlaPrefixGather(kv_scale if fp8_compute else 1.0)
        self.use_mla = use_mla
        self.qo_indptr: Optional[torch.Tensor] = None
        self.kv_indptr: Optional[torch.Tensor] = None
        self.max_q_len = 0
        self.max_kv_len = 0
        self.has_reuse_cache = False
        self.batch_reuse_info_vec: Optional[torch.Tensor] = None
        self.total_kv_lens = 0
        self.q_lens: List[int] = []
        self.kv_lens: List[int] = []
        self.batch_reuse_info_host: tuple[tuple[int, int, int, int], ...] = ()
        self._direct_attn_inputs: Optional[Any] = None
        self._forward_plan: Optional[FlashMLAForwardPlan] = None
        self._q_offsets: tuple[int, ...] = ()
        self._prefix_runtime_launches: tuple[_FlashMLAPrefixRuntimeLaunch, ...] = ()
        self._page_rr_full_descriptor: Optional[MlaPageRRChunkDescriptor] = None
        self._forward_workspace: Optional[_FlashMLAForwardWorkspace] = None

    def release_forward_workspace(self) -> None:
        """Drop per-plan scratch once all target layers have consumed it.

        The plan and cache metadata remain usable. Subsequent forwards can
        allocate fresh scratch; stream-ordered allocator reuse needs no flush.
        """
        self._forward_workspace = None
        self._fp8_prefix_rope = None

    def plan(self, mla_params: FlashMLADeviceParams) -> None:
        self.q_lens = list(mla_params.q_lens_host)
        self.kv_lens = list(mla_params.kv_lens_host)
        prefix_lens = list(mla_params.prefix_lens_host)
        self.qo_indptr = mla_params.qo_indptr_d
        self.kv_indptr = mla_params.prefill_ragged_kv_len_indptr_d
        self.has_reuse_cache = mla_params.has_reuse_cache
        batch_reuse_info_host = mla_params.batch_reuse_info_host
        self._direct_attn_inputs = mla_params.attn_inputs

        self.max_q_len = max(self.q_lens)
        self.max_kv_len = max(self.kv_lens)
        self.total_kv_lens = sum(self.kv_lens)
        self.batch_reuse_info_host = batch_reuse_info_host
        self.batch_reuse_info_vec = mla_params.batch_reuse_info_vec_d
        self._forward_plan = plan_flashmla_forward(
            self.q_lens,
            prefix_lens,
            prefix_chunk_alignment_tokens=self.prefix_chunk_alignment_tokens,
            expanded_kv_budget_gib=self.expanded_kv_budget_gib,
            expanded_kv_bytes_per_token=self._expanded_kv_bytes_per_token(),
        )
        _log_page_rr_prefix_plan_once(
            self._forward_plan,
            self.page_rr_cache_adapter,
            q_lens=self.q_lens,
            prefix_lens=prefix_lens,
            alignment_tokens=self.prefix_chunk_alignment_tokens,
        )
        self._q_offsets = ()
        self._prefix_runtime_launches = ()
        self._page_rr_full_descriptor = None
        if (
            self.page_rr_cache_adapter is not None
            and any(prefix_lens)
            and self._forward_plan.route is FlashMLAForwardRoute.FULL
        ):
            request_indices = tuple(
                index for index, length in enumerate(prefix_lens) if length
            )
            nonempty_lens = tuple(prefix_lens[index] for index in request_indices)
            self._page_rr_full_descriptor = (
                self.page_rr_cache_adapter.build_prefix_chunk_descriptor(
                    request_indices=request_indices,
                    prefix_starts=(0,) * len(request_indices),
                    prefix_lens=nonempty_lens,
                    feature_width=self.kv_lora_rank + self.qk_rope_head_dim,
                )
            )
        self.release_forward_workspace()
        if self._forward_plan.route is FlashMLAForwardRoute.HYBRID:
            self._materialize_prefix_runtime_launches(mla_params.qo_indptr_d.device)

    def _expanded_kv_bytes_per_token(self) -> int:
        bytes_per_element = torch.bfloat16.itemsize
        if self.fp8_compute:
            # TokenSpeed quantizes the BF16 KV-up output for attention while the
            # BF16 staging tensor is still live, so budget the overlapping copy.
            bytes_per_element += torch.float8_e4m3fn.itemsize
        return (
            self.num_heads
            * (self.qk_nope_head_dim + self.qk_rope_head_dim + self.v_head_dim)
            * bytes_per_element
        )

    def _materialize_prefix_runtime_launches(self, device: torch.device) -> None:
        plan = cast(FlashMLAForwardPlan, self._forward_plan)

        q_offsets = tuple(_prefix_sum(self.q_lens))
        self._q_offsets = q_offsets
        runtime_launches = []
        for launch in plan.prefix_launches:
            owners = [item.request_idx for item in launch.slices]
            page_rr_descriptor = None
            if self.page_rr_cache_adapter is not None:
                page_rr_descriptor = (
                    self.page_rr_cache_adapter.build_prefix_chunk_descriptor(
                        request_indices=owners,
                        prefix_starts=tuple(
                            item.prefix_start for item in launch.slices
                        ),
                        prefix_lens=tuple(item.prefix_len for item in launch.slices),
                        feature_width=self.kv_lora_rank + self.qk_rope_head_dim,
                    )
                )
            qo_indptr = _prefix_sum(self.q_lens[owner] for owner in owners)
            kv_indptr = _prefix_sum(item.prefix_len for item in launch.slices)
            gather_qo_indptr = [0] * (len(owners) + 1)
            batch_reuse_info = []
            for local_row, (owner, item) in enumerate(
                zip(owners, launch.slices, strict=True)
            ):
                request_cache_page_start = self.batch_reuse_info_host[owner][2]
                batch_reuse_info.extend(
                    (
                        local_row,
                        item.prefix_len,
                        request_cache_page_start
                        + item.prefix_start // self.kernel_page_tokens,
                        (item.prefix_len + self.kernel_page_tokens - 1)
                        // self.kernel_page_tokens,
                    )
                )
            destination_starts = [q_offsets[owner] for owner in owners]
            flat_values = (
                qo_indptr
                + kv_indptr
                + gather_qo_indptr
                + batch_reuse_info
                + destination_starts
            )
            metadata = torch.tensor(flat_values, dtype=torch.int32, device=device)
            num_rows = len(owners)
            (
                launch_qo_indptr,
                launch_kv_indptr,
                launch_gather_qo_indptr,
                launch_batch_reuse_info,
                launch_destination_starts,
            ) = metadata.split(
                (num_rows + 1, num_rows + 1, num_rows + 1, num_rows * 4, num_rows)
            )
            q_range = None
            if owners == list(range(owners[0], owners[-1] + 1)):
                q_range = (
                    q_offsets[owners[0]],
                    q_offsets[owners[-1] + 1] - q_offsets[owners[0]],
                )
            metadata.record_stream(torch.cuda.current_stream(device))
            runtime_launches.append(
                _FlashMLAPrefixRuntimeLaunch(
                    spec=launch,
                    qo_indptr=launch_qo_indptr,
                    kv_indptr=launch_kv_indptr,
                    gather_qo_indptr=launch_gather_qo_indptr,
                    batch_reuse_info=launch_batch_reuse_info.view(num_rows, 4),
                    destination_starts=launch_destination_starts,
                    q_range=q_range,
                    max_q_len=max(self.q_lens[owner] for owner in owners),
                    max_kv_len=max(item.prefix_len for item in launch.slices),
                    page_rr_descriptor=page_rr_descriptor,
                )
            )
        self._prefix_runtime_launches = tuple(runtime_launches)

    def _live_reuse_cache_page_indices(self) -> torch.Tensor:
        block_table = cast(
            torch.Tensor,
            cast(Any, self._direct_attn_inputs).kv_cache_kernel_block_id_device,
        )
        reuse_cache_page_indice = block_table.contiguous().view(-1)
        reuse_cache_page_indice.record_stream(
            torch.cuda.current_stream(block_table.device)
        )
        return reuse_cache_page_indice

    def _gather_cache(self, *args):
        if self._prefix_producer is not None:
            return self._prefix_producer(*args)
        if self.fp8_compute:
            from .mla_fp8_kernels import gather_fp8_prefix

            gather_fp8_prefix(*args, scale=self.kv_scale)
            return args[0]
        rtp_llm_ops.reuse_kv_cache_indexed_batched(*args)
        return args[0]

    def _gather_reused_kv(
        self,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        page_rr_prefix: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_k_pe = k_pe.view(-1, self.qk_rope_head_dim)
        if page_rr_prefix is not None:
            prefix_lens = [row[1] for row in self.batch_reuse_info_host]

            def merge(prefix: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
                parts = zip(
                    prefix.split(prefix_lens), query.split(self.q_lens), strict=True
                )
                return torch.cat([rows for pair in parts for rows in pair])

            return (
                merge(
                    page_rr_prefix[:, : self.kv_lora_rank],
                    retained_bf16(compressed_kv),
                ),
                merge(page_rr_prefix[:, self.kv_lora_rank :], flat_k_pe),
            )
        if not self.has_reuse_cache:
            return compressed_kv, flat_k_pe
        kv_cache_base, reuse_cache_page_indice = self._reuse_cache_inputs(kv_cache)

        final_compressed_kv = torch.empty(
            (self.total_kv_lens, self.kv_lora_rank),
            dtype=torch.bfloat16,
            device=compressed_kv.device,
        )
        final_k_pe = torch.empty(
            (self.total_kv_lens, self.qk_rope_head_dim),
            dtype=flat_k_pe.dtype,
            device=flat_k_pe.device,
        )
        final_compressed_kv = self._gather_cache(
            final_compressed_kv,
            final_k_pe,
            compressed_kv,
            flat_k_pe.contiguous(),
            kv_cache_base,
            reuse_cache_page_indice,
            self.batch_reuse_info_vec,
            self.qo_indptr,
            self.kernel_page_tokens,
        )
        return final_compressed_kv, final_k_pe

    def _reuse_cache_inputs(
        self,
        kv_cache: Optional[LayerKVCache],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reuse_cache_page_indice = self._live_reuse_cache_page_indices()
        return cast(LayerKVCache, kv_cache).kv_cache_base, reuse_cache_page_indice

    def _read_page_rr_prefix(
        self,
        kv_cache: Optional[LayerKVCache],
        descriptor: MlaPageRRChunkDescriptor,
    ) -> torch.Tensor:
        adapter = cast(MlaPageRRCacheAdapter, self.page_rr_cache_adapter)
        if kv_cache is None:
            raise RuntimeError("MLA page-RR prefix reuse requires an MLA KV cache")
        block_table = getattr(
            self._direct_attn_inputs,
            "kv_cache_kernel_block_id_device",
            None,
        )
        if block_table is None:
            raise RuntimeError("MLA page-RR prefix reuse requires a block table")
        raw_cache = kv_cache.kv_cache_base
        expected_dtype = torch.float8_e4m3fn if self.fp8_compute else torch.bfloat16
        expected_width = self.kv_lora_rank + self.qk_rope_head_dim
        if (
            raw_cache.dtype != expected_dtype
            or not raw_cache.is_cuda
            or raw_cache.shape[-1] != expected_width
        ):
            raise RuntimeError(
                "MLA page-RR Prefill raw cache does not match the configured "
                f"precision; expected dtype={expected_dtype} "
                f"[kernel_blocks,kernel_page_tokens,{expected_width}], got "
                f"shape={tuple(raw_cache.shape)} dtype={raw_cache.dtype} "
                f"device={raw_cache.device}"
            )
        canonical = adapter.read_prefix_chunk(raw_cache, block_table, descriptor)
        if self.fp8_compute:
            canonical = canonical.to(torch.bfloat16)
            canonical.mul_(self.kv_scale)
        return canonical

    def _copy_page_rr_prefix_chunk(
        self,
        canonical: torch.Tensor,
        launch_compressed: torch.Tensor,
        launch_packed_kv: torch.Tensor,
    ) -> None:
        packed_head_dim = sum(_K3_PACKED_KV_HEAD_SPLITS)
        launch_compressed.copy_(canonical[:, : self.kv_lora_rank])
        launch_packed_kv.view(canonical.shape[0], self.num_heads, packed_head_dim)[
            ...,
            self.qk_nope_head_dim : self.qk_nope_head_dim + self.qk_rope_head_dim,
        ].copy_(canonical[:, self.kv_lora_rank :].unsqueeze(1))

    def _project_reused_kv_with_gap_fill(
        self,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        packed_projection: LinearBase,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Use the fused paged KPE gather before the packed KV projection."""

        fused_gather = getattr(rtp_llm_ops, "_gather_mla_latent_and_fill_k_pe", None)
        if (
            self._prefix_producer is not None
            or self.fp8_compute
            or not self.has_reuse_cache
            or not callable(fused_gather)
        ):
            return None

        flat_k_pe = k_pe.view(-1, self.qk_rope_head_dim)
        kv_cache_base, reuse_cache_page_indice = self._reuse_cache_inputs(kv_cache)
        head_splits = (
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        )
        packed_head_dim = sum(head_splits)
        final_compressed_kv = torch.empty(
            (self.total_kv_lens, self.kv_lora_rank),
            dtype=torch.bfloat16,
            device=compressed_kv.device,
        )
        packed_kv = torch.empty(
            (self.total_kv_lens, self.num_heads * packed_head_dim),
            dtype=torch.bfloat16,
            device=compressed_kv.device,
        )
        fused_gather(
            final_compressed_kv,
            packed_kv,
            compressed_kv,
            flat_k_pe,
            kv_cache_base,
            reuse_cache_page_indice,
            self.batch_reuse_info_vec,
            self.qo_indptr,
            self.kernel_page_tokens,
            packed_head_dim,
            self.qk_nope_head_dim,
        )
        packed_projection.forward_skip_head_mid(
            final_compressed_kv,
            head_splits,
            output=packed_kv,
        )
        packed_kv = packed_kv.view(self.total_kv_lens, self.num_heads, packed_head_dim)
        k = packed_kv[..., : -self.v_head_dim]
        value_states = packed_kv[..., -self.v_head_dim :]
        return k, value_states

    def _packed_kv_projection(
        self,
        compressed_kv: torch.Tensor,
        kv_b_proj: LinearBase,
    ) -> Optional[LinearBase]:
        head_splits = (
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        )
        if (
            head_splits == _K3_PACKED_KV_HEAD_SPLITS
            and kv_b_proj.supports_skip_head_mid(compressed_kv, head_splits)
        ):
            return kv_b_proj
        return None

    def _project_kv(
        self,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        layer_id: int,
        kv_b_proj: Optional[LinearBase] = None,
        packed_projection: Optional[LinearBase] = None,
        packed_output: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if kv_b_proj is None:
            kv_b_proj = self._create_kv_b_proj(layer_id)
            packed_projection = self._packed_kv_projection(compressed_kv, kv_b_proj)
        head_splits = (
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        )
        num_tokens = compressed_kv.shape[0]
        if packed_projection is not None:
            if packed_output is None:
                projected_kv = packed_projection.forward_skip_head_mid(
                    compressed_kv, head_splits
                )
            else:
                projected_kv = packed_projection.forward_skip_head_mid(
                    compressed_kv,
                    head_splits,
                    output=packed_output,
                )
            packed_kv = projected_kv.view(num_tokens, self.num_heads, sum(head_splits))
            packed_kv[..., self.qk_nope_head_dim : -self.v_head_dim].copy_(
                k_pe.view(num_tokens, 1, self.qk_rope_head_dim)
            )
            k = packed_kv[..., : -self.v_head_dim]
            value_states = packed_kv[..., -self.v_head_dim :]
            return k, value_states

        expanded_dim = self.qk_nope_head_dim + self.v_head_dim
        kv = kv_b_proj(compressed_kv).view(num_tokens, self.num_heads, expanded_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        value_states = kv[..., self.qk_nope_head_dim :]

        k = kv.new_empty(
            num_tokens,
            self.num_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
        )
        k[..., : self.qk_nope_head_dim].copy_(k_nope)
        k[..., self.qk_nope_head_dim :].copy_(
            k_pe.view(num_tokens, 1, self.qk_rope_head_dim)
        )
        return k, value_states

    def _create_kv_b_proj(self, layer_id: int) -> LinearBase:
        if self._kv_b_projections is not None:
            return self._kv_b_projections[layer_id]
        return LinearFactory.create_linear_from_weights(
            self.weights[layer_id],
            W.mla_kv_b_w,
            W.mla_kv_b_s,
            None,
            self.quant_config,
        )

    def _run_dense_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        value_states: torch.Tensor,
        *,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        causal: bool,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if out is None:
            out = torch.empty(
                q.shape[0],
                self.num_heads,
                self.v_head_dim,
                dtype=q.dtype,
                device=q.device,
            )
            lse = torch.empty(
                self.num_heads,
                q.shape[0],
                dtype=torch.float32,
                device=q.device,
            ).transpose(0, 1)
        else:
            lse = cast(torch.Tensor, lse)
        if self.fp8_compute:
            from .mla_fp8_kernels import quantize_fp8

            # Expanded K/V have their own ordinary unit-scale quantization.
            # Historical compressed KV was restored with kv_scale at gather.
            result, result_lse = self.tokenspeed_prefill(
                query=quantize_fp8(q, self.q_scale, name="prefill_q"),
                key=quantize_fp8(k, name="prefill_k"),
                value=quantize_fp8(value_states, name="prefill_v"),
                seq_lens=kv_indptr[1:] - kv_indptr[:-1],
                cum_seq_lens=kv_indptr,
                max_seq_len=max_kv_len,
                batch_size=qo_indptr.numel() - 1,
                softmax_scale=self.scale * self.q_scale,
                is_causal=causal,
                return_lse=True,
                cum_seq_lens_q=qo_indptr,
                max_seq_len_q=max_q_len,
                enable_pdl=False,
                out=out,
            )
            lse.copy_(result_lse)
            return result, lse
        self.flash_mla_cuda.dense_prefill_fwd(
            _workspace(q.device),
            q,
            k,
            value_states,
            qo_indptr,
            kv_indptr,
            out,
            lse,
            1 if causal else 0,
            self.scale,
            max_q_len,
            max_kv_len,
            True,
        )
        return out, lse

    def _dense_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        out, _ = self._run_dense_attention(
            q,
            k,
            value_states,
            qo_indptr=cast(torch.Tensor, self.qo_indptr),
            kv_indptr=cast(torch.Tensor, self.kv_indptr),
            max_q_len=self.max_q_len,
            max_kv_len=self.max_kv_len,
            causal=True,
        )
        return out

    def _forward_full(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_id: int,
        kv_b_proj: LinearBase,
        packed_projection: Optional[LinearBase],
    ) -> torch.Tensor:
        page_rr_prefix = None
        if self._page_rr_full_descriptor is not None:
            page_rr_prefix = self._read_page_rr_prefix(
                kv_cache,
                self._page_rr_full_descriptor,
            )
        projected_kv = None
        if packed_projection is not None and page_rr_prefix is None:
            projected_kv = self._project_reused_kv_with_gap_fill(
                compressed_kv, k_pe, kv_cache, packed_projection
            )
        if projected_kv is None:
            gathered_compressed_kv, gathered_k_pe = self._gather_reused_kv(
                compressed_kv, k_pe, kv_cache, page_rr_prefix
            )
            projected_kv = self._project_kv(
                gathered_compressed_kv,
                gathered_k_pe,
                layer_id,
                kv_b_proj,
                packed_projection=packed_projection,
            )
        return self._dense_attention(q, projected_kv[0], projected_kv[1])

    def _pack_prefix_q(
        self,
        q: torch.Tensor,
        launch: _FlashMLAPrefixRuntimeLaunch,
        workspace: _FlashMLAForwardWorkspace,
    ) -> torch.Tensor:
        if launch.q_range is not None:
            return q.narrow(0, launch.q_range[0], launch.q_range[1])
        packed_q = workspace.q_buffer(launch.spec.packed_q_tokens)
        q_offsets = self._q_offsets
        cursor = 0
        for item in launch.spec.slices:
            q_len = self.q_lens[item.request_idx]
            packed_q.narrow(0, cursor, q_len).copy_(
                q.narrow(0, q_offsets[item.request_idx], q_len)
            )
            cursor += q_len
        return packed_q

    def _forward_historical_prefix(
        self,
        workspace: _FlashMLAForwardWorkspace,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        packed_projection: LinearBase,
        canonical_output: torch.Tensor,
    ) -> None:
        flat_k_pe = k_pe.view(-1, self.qk_rope_head_dim)
        if self.page_rr_cache_adapter is None:
            fused_gather = rtp_llm_ops._gather_mla_latent_and_fill_k_pe
            kv_cache_base, reuse_cache_page_indice = self._reuse_cache_inputs(kv_cache)
        packed_head_dim = sum(_K3_PACKED_KV_HEAD_SPLITS)

        for launch in self._prefix_runtime_launches:
            kv_tokens = launch.spec.expanded_kv_tokens
            launch_compressed = workspace.compressed_kv_buffer(kv_tokens)
            launch_packed_kv = workspace.packed_kv_buffer(kv_tokens)
            if launch.page_rr_descriptor is not None:
                canonical = self._read_page_rr_prefix(
                    kv_cache,
                    launch.page_rr_descriptor,
                )
                self._copy_page_rr_prefix_chunk(
                    canonical,
                    launch_compressed,
                    launch_packed_kv,
                )
                del canonical
            elif self.fp8_compute or self._prefix_producer is not None:
                launch_rope = self._fp8_prefix_rope[:kv_tokens]
                launch_compressed = self._gather_cache(
                    launch_compressed,
                    launch_rope,
                    compressed_kv,
                    flat_k_pe,
                    kv_cache_base,
                    reuse_cache_page_indice,
                    launch.batch_reuse_info,
                    launch.gather_qo_indptr,
                    self.kernel_page_tokens,
                )
                launch_packed_kv.view(kv_tokens, self.num_heads, packed_head_dim)[
                    ...,
                    self.qk_nope_head_dim : self.qk_nope_head_dim
                    + self.qk_rope_head_dim,
                ].copy_(launch_rope[:, None, :])
            else:
                fused_gather(
                    launch_compressed,
                    launch_packed_kv,
                    compressed_kv,
                    flat_k_pe,
                    kv_cache_base,
                    reuse_cache_page_indice,
                    launch.batch_reuse_info,
                    launch.gather_qo_indptr,
                    self.kernel_page_tokens,
                    packed_head_dim,
                    self.qk_nope_head_dim,
                )
            packed_projection.forward_skip_head_mid(
                launch_compressed,
                _K3_PACKED_KV_HEAD_SPLITS,
                output=launch_packed_kv,
            )
            packed_kv = launch_packed_kv.view(
                kv_tokens, self.num_heads, packed_head_dim
            )
            launch_k = packed_kv[..., : -self.v_head_dim]
            launch_v = packed_kv[..., -self.v_head_dim :]
            launch_q = self._pack_prefix_q(q, launch, workspace)
            partial_buffers = workspace.attention_buffers(launch.spec.packed_q_tokens)
            partial_output, partial_lse = self._run_dense_attention(
                launch_q,
                launch_k,
                launch_v,
                qo_indptr=launch.qo_indptr,
                kv_indptr=launch.kv_indptr,
                max_q_len=launch.max_q_len,
                max_kv_len=launch.max_kv_len,
                causal=False,
                out=partial_buffers[0],
                lse=partial_buffers[1],
            )
            merge_attention_states_segmented_in_place(
                canonical_output,
                workspace.canonical_lse,
                partial_output,
                partial_lse,
                launch.qo_indptr,
                launch.destination_starts,
            )

    def _forward_hybrid(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_id: int,
        kv_b_proj: LinearBase,
        packed_projection: LinearBase,
    ) -> torch.Tensor:
        expected_tokens = sum(self.q_lens)
        # Initialize FlashMLA's device scratch before entering the physical
        # launch loops. Subsequent attention calls only retrieve this buffer.
        if not self.fp8_compute:
            _workspace(q.device)
        workspace = self._forward_workspace
        if workspace is None:
            packed_q_tokens = max(
                (
                    launch.spec.packed_q_tokens
                    for launch in self._prefix_runtime_launches
                    if launch.q_range is None
                ),
                default=0,
            )
            workspace = _FlashMLAForwardWorkspace.allocate(
                plan=cast(FlashMLAForwardPlan, self._forward_plan),
                compressed_kv=compressed_kv,
                q=q,
                kv_lora_rank=self.kv_lora_rank,
                num_heads=self.num_heads,
                qk_head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                packed_q_tokens=packed_q_tokens,
            )
            self._forward_workspace = workspace
            if self.fp8_compute or self._prefix_producer is not None:
                self._fp8_prefix_rope = retained_bf16(compressed_kv).new_empty(
                    (
                        cast(
                            FlashMLAForwardPlan, self._forward_plan
                        ).max_expanded_kv_tokens,
                        self.qk_rope_head_dim,
                    )
                )
        current_packed_kv = workspace.packed_kv_buffer(expected_tokens)
        current_k, current_v = self._project_kv(
            compressed_kv,
            k_pe,
            layer_id,
            kv_b_proj,
            packed_projection=packed_projection,
            packed_output=current_packed_kv,
        )
        current_output, _ = self._run_dense_attention(
            q,
            current_k,
            current_v,
            qo_indptr=cast(torch.Tensor, self.qo_indptr),
            kv_indptr=cast(torch.Tensor, self.qo_indptr),
            max_q_len=self.max_q_len,
            max_kv_len=self.max_q_len,
            causal=True,
            out=workspace.output_bf16,
            lse=workspace.canonical_lse,
        )
        canonical_output = (
            workspace.fp32_output
            if workspace.fp32_output is not None
            else workspace.output_bf16
        )
        if workspace.fp32_output is not None:
            workspace.fp32_output.copy_(current_output)
        self._forward_historical_prefix(
            workspace,
            q,
            compressed_kv,
            k_pe,
            kv_cache,
            packed_projection,
            canonical_output,
        )
        if workspace.fp32_output is not None:
            workspace.output_bf16.copy_(workspace.fp32_output)
        return workspace.output_bf16

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_id: int,
    ) -> torch.Tensor:
        if self._forward_plan is None:
            raise RuntimeError("FlashMLA Prefill must be planned before forward")
        plan = cast(FlashMLAForwardPlan, self._forward_plan)
        kv_b_proj = self._create_kv_b_proj(layer_id)
        packed_projection = self._packed_kv_projection(compressed_kv, kv_b_proj)
        if plan.route is FlashMLAForwardRoute.FULL:
            return self._forward_full(
                q,
                compressed_kv,
                k_pe,
                kv_cache,
                layer_id,
                kv_b_proj,
                packed_projection,
            )
        return self._forward_hybrid(
            q,
            compressed_kv,
            k_pe,
            kv_cache,
            layer_id,
            kv_b_proj,
            cast(LinearBase, packed_projection),
        )

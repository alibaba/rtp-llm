"""Dense causal MLA Prefill backed by FlashMLA's SM100 kernel.

This adapter keeps RTP's existing projections, RoPE/no-RoPE handling, cache
write and output projection unchanged; only the dense causal attention core is
replaced.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from rtp_llm.models.kimi_k3.mla_cache_tp import (
    kimi_k3_mla_cache_layout,
    mla_cache_tp_enabled,
)
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    has_bf16_gemm_nt_skip_head_mid,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_prefix_chunk import (
    FlashMLAPrefixChunkSpec,
    plan_flashmla_prefix_chunks,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_state_merge_triton import (
    merge_attention_states_in_place,
)
from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory
from rtp_llm.models_py.modules.factory.linear.linear_base import LinearBase
from rtp_llm.ops import KvCacheDataType
from rtp_llm.ops.compute_ops import LayerKVCache, rtp_llm_ops
from rtp_llm.utils.model_weight import W

_FLASHMLA_WORKSPACES: Dict[int, torch.Tensor] = {}
_FLASHMLA_LOGGED_DEVICES: set[int] = set()
# Log each static execution shape once. KV lengths are intentionally excluded:
# target verification grows them every iteration while reusing the same plan
# shape, so including them turns this guard into a per-step INFO log.
_FLASHMLA_LOGGED_CONFIGS: set[tuple[int, int, tuple[int, ...], bool, int]] = set()
_K3_PACKED_KV_HEAD_SPLITS = (128, 64, 128)


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


@dataclass(frozen=True)
class _FlashMLAPrefixChunk:
    spec: FlashMLAPrefixChunkSpec
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    gather_qo_indptr: torch.Tensor
    batch_reuse_info: torch.Tensor


@dataclass
class _FlashMLAPrefixChunkWorkspace:
    """Invocation-local storage reused by every historical-prefix chunk.

    The chunk loop is serialized on one CUDA stream, so one set of buffers is
    sufficient.  Keeping this workspace local to one invocation avoids both
    per-chunk allocator traffic and a large persistent allocation per MLA
    layer.
    """

    compressed_kv: torch.Tensor
    k_pe: Optional[torch.Tensor]
    packed_kv: Optional[torch.Tensor]
    attention_out: torch.Tensor
    attention_lse_storage: torch.Tensor
    num_heads: int

    @classmethod
    def allocate(
        cls,
        *,
        chunks: Sequence[_FlashMLAPrefixChunk],
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        q: torch.Tensor,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        num_heads: int,
        v_head_dim: int,
        packed_features: Optional[int],
        allocate_k_pe: bool,
    ) -> "_FlashMLAPrefixChunkWorkspace":
        if not chunks:
            raise ValueError("FlashMLA prefix workspace requires at least one chunk")
        max_kv_tokens = max(chunk.spec.kv_tokens for chunk in chunks)
        max_q_tokens = max(chunk.spec.q_tokens for chunk in chunks)
        packed_kv = (
            compressed_kv.new_empty((max_kv_tokens, packed_features))
            if packed_features is not None
            else None
        )
        return cls(
            compressed_kv=compressed_kv.new_empty((max_kv_tokens, kv_lora_rank)),
            k_pe=(
                k_pe.new_empty((max_kv_tokens, qk_rope_head_dim))
                if allocate_k_pe
                else None
            ),
            packed_kv=packed_kv,
            attention_out=q.new_empty((max_q_tokens, num_heads, v_head_dim)),
            attention_lse_storage=torch.empty(
                max_q_tokens * num_heads,
                dtype=torch.float32,
                device=q.device,
            ),
            num_heads=num_heads,
        )

    def gather_buffers(self, kv_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k_pe is None:
            raise RuntimeError(
                "FlashMLA prefix KPE workspace is unavailable for unfused gather"
            )
        return (
            self.compressed_kv.narrow(0, 0, kv_tokens),
            self.k_pe.narrow(0, 0, kv_tokens),
        )

    def packed_kv_buffer(self, kv_tokens: int) -> Optional[torch.Tensor]:
        if self.packed_kv is None:
            return None
        return self.packed_kv.narrow(0, 0, kv_tokens)

    def attention_buffers(self, q_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.attention_out.narrow(0, 0, q_tokens)
        # FlashMLA writes natural-log LSE in the layout produced by
        # empty([heads, tokens]).transpose(0, 1).  Reinterpret the reusable
        # flat storage with the exact logical token stride for this chunk.
        lse = torch.as_strided(
            self.attention_lse_storage,
            size=(q_tokens, self.num_heads),
            stride=(1, q_tokens),
        )
        return out, lse


def _indptr(lengths: Sequence[int]) -> list[int]:
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
        reuse_cache_page_indice_d: torch.Tensor,
        batch_reuse_info_vec_d: torch.Tensor,
        batch_reuse_info_host: tuple[tuple[int, int, int, int], ...],
        block_table_width: int,
    ) -> None:
        self.attn_inputs = attn_inputs
        self.q_lens_host = q_lens_host
        self.kv_lens_host = kv_lens_host
        self.prefix_lens_host = prefix_lens_host
        self.qo_indptr_d = qo_indptr_d
        self.prefill_ragged_kv_len_indptr_d = kv_indptr_d
        self.positions_d = positions_d
        self.batch_indice_d = batch_indice_d
        self.reuse_cache_page_indice_d = reuse_cache_page_indice_d
        self.batch_reuse_info_vec_d = batch_reuse_info_vec_d
        self.batch_reuse_info_host = batch_reuse_info_host
        self.block_table_width = block_table_width
        self.has_reuse_cache = any(prefix_lens_host)
        # MlaKVCacheWriteOp asks the wrapper to derive this from live device
        # positions and the currently selected HybridCache group.
        self.slot_mapping = None


def _host_i32_values(tensor: Optional[torch.Tensor], name: str) -> List[int]:
    if tensor is None or tensor.numel() == 0:
        raise RuntimeError(f"FlashMLA direct planner requires {name}")
    if tensor.is_cuda:
        raise RuntimeError(
            f"FlashMLA direct planner requires CPU {name}; refusing a hidden D2H"
        )
    if tensor.dtype != torch.int32:
        raise RuntimeError(
            f"FlashMLA direct planner requires int32 {name}, got {tensor.dtype}"
        )
    if tensor.dim() != 1:
        raise RuntimeError(f"FlashMLA {name} must be 1-D, got {tuple(tensor.shape)}")
    return [int(value) for value in tensor.tolist()]


def _cuda_i32_tensor(tensor: Optional[torch.Tensor], name: str) -> torch.Tensor:
    if tensor is None or tensor.numel() == 0:
        raise RuntimeError(f"FlashMLA direct planner requires CUDA {name}")
    if not tensor.is_cuda or tensor.dtype != torch.int32:
        raise RuntimeError(
            f"FlashMLA {name} must be CUDA int32, got "
            f"device={tensor.device} dtype={tensor.dtype}"
        )
    return tensor


def build_flashmla_device_params(
    attn_inputs: Any,
    page_size: int,
) -> FlashMLADeviceParams:
    """Build dense-prefill metadata without FlashInfer's pinned ``buf_h``.

    Host mirrors are used only for shape scalars required by FlashMLA's Python
    API.  Indptrs, positions, reuse metadata and page IDs stay on CUDA.  The
    fixed-width branch is the K3 MTP draft-prefill hot path (currently q_len=4).
    """

    if page_size <= 0:
        raise ValueError(f"FlashMLA page_size must be positive, got {page_size}")

    input_lengths_d = _cuda_i32_tensor(
        getattr(attn_inputs, "input_lengths", None), "input_lengths"
    )
    prefix_lengths_d = _cuda_i32_tensor(
        getattr(attn_inputs, "prefix_lengths", None), "prefix_lengths"
    )
    q_lens = _host_i32_values(
        getattr(attn_inputs, "input_lengths_host", None), "input_lengths_host"
    )
    prefix_lens = _host_i32_values(
        getattr(attn_inputs, "prefix_lengths_host", None), "prefix_lengths_host"
    )
    batch_size = len(q_lens)
    if batch_size == 0 or len(prefix_lens) != batch_size:
        raise RuntimeError(
            "FlashMLA host length batch mismatch: "
            f"q={len(q_lens)} prefix={len(prefix_lens)}"
        )
    if input_lengths_d.numel() != batch_size or prefix_lengths_d.numel() != batch_size:
        raise RuntimeError(
            "FlashMLA device length batch mismatch: "
            f"host={batch_size} input={input_lengths_d.numel()} "
            f"prefix={prefix_lengths_d.numel()}"
        )
    if any(q_len <= 0 for q_len in q_lens) or any(
        prefix_len < 0 for prefix_len in prefix_lens
    ):
        raise RuntimeError(
            f"FlashMLA lengths must be q>0 and prefix>=0, got "
            f"q={q_lens} prefix={prefix_lens}"
        )

    total_q = sum(q_lens)
    total_tokens = int(getattr(attn_inputs, "total_tokens", total_q))
    if total_tokens != total_q:
        raise RuntimeError(
            "FlashMLA packed query disagrees with host lengths: "
            f"total_tokens={total_tokens} sum_q={total_q}"
        )
    kv_lens = [
        q_len + prefix_len
        for q_len, prefix_len in zip(q_lens, prefix_lens, strict=True)
    ]

    qo_indptr_d = _cuda_i32_tensor(
        getattr(attn_inputs, "cu_seqlens", None), "cu_seqlens"
    )
    kv_indptr_d = _cuda_i32_tensor(
        getattr(attn_inputs, "cu_kv_seqlens", None), "cu_kv_seqlens"
    )
    if qo_indptr_d.numel() != batch_size + 1 or kv_indptr_d.numel() != batch_size + 1:
        raise RuntimeError(
            "FlashMLA indptr shape mismatch: "
            f"batch={batch_size} qo={qo_indptr_d.numel()} kv={kv_indptr_d.numel()}"
        )
    if (
        qo_indptr_d.device != input_lengths_d.device
        or kv_indptr_d.device != input_lengths_d.device
    ):
        raise RuntimeError("FlashMLA length tensors and indptrs must share one device")

    device = input_lengths_d.device
    packed_indices = torch.arange(total_q, dtype=torch.int32, device=device)
    fixed_q_len = q_lens[0] if all(q_len == q_lens[0] for q_len in q_lens) else 0
    if fixed_q_len:
        # K3 draft-prefill: q_len == propose_step + 1, currently 4.
        batch_indice_d = torch.div(packed_indices, fixed_q_len, rounding_mode="floor")
        local_positions = torch.remainder(packed_indices, fixed_q_len)
    else:
        padding_offset = _cuda_i32_tensor(
            getattr(attn_inputs, "padding_offset", None), "padding_offset"
        )
        if padding_offset.numel() != total_q:
            raise RuntimeError(
                "FlashMLA ragged padding_offset mismatch: "
                f"expected={total_q} actual={padding_offset.numel()}"
            )
        max_q_len = max(q_lens)
        padded_indices = packed_indices + padding_offset
        batch_indice_d = torch.div(padded_indices, max_q_len, rounding_mode="floor")
        local_positions = torch.remainder(padded_indices, max_q_len)
    positions_d = (
        prefix_lengths_d.index_select(0, batch_indice_d.to(torch.int64))
        + local_positions
    )

    block_table = getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)
    has_reuse_cache = any(prefix_lens)
    if block_table is None or block_table.numel() == 0:
        if has_reuse_cache:
            raise RuntimeError("FlashMLA cache reuse requires a CUDA block table")
        block_table = torch.empty((batch_size, 0), dtype=torch.int32, device=device)
    if (
        not block_table.is_cuda
        or block_table.dtype != torch.int32
        or block_table.dim() != 2
        or block_table.shape[0] != batch_size
        or block_table.device != device
    ):
        raise RuntimeError(
            "FlashMLA block table must be CUDA int32 [batch, max_blocks], "
            f"got device={block_table.device} dtype={block_table.dtype} "
            f"shape={tuple(block_table.shape)}"
        )
    max_blocks = int(block_table.shape[1])
    required_cache_pages = [(kv_len + page_size - 1) // page_size for kv_len in kv_lens]
    # A cache-less warmup/pure-attention call legitimately has no block table.
    # If a cache is supplied later, _device_slot_mapping() will require it at
    # the actual write site.  Whenever a table is present, validate the full
    # suffix write (not only the reused prefix) before indexing it.
    if max_blocks and any(
        page_count > max_blocks for page_count in required_cache_pages
    ):
        raise RuntimeError(
            "FlashMLA query write exceeds the selected block table: "
            f"required={required_cache_pages} max_blocks={max_blocks}"
        )

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
    flat_block_table = block_table.contiguous().view(-1)
    reuse_cache_page_indice_d = (
        flat_block_table if has_reuse_cache else flat_block_table[:0]
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
        reuse_cache_page_indice_d,
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
        reuse_cache_page_indice_d=reuse_cache_page_indice_d,
        batch_reuse_info_vec_d=batch_reuse_info_vec_d,
        batch_reuse_info_host=batch_reuse_info_host,
        block_table_width=max_blocks,
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
        page_size: int,
        softmax_extra_scale: float,
        use_mla: bool,
        weights: List[Dict[str, torch.Tensor]] | None,
        quant_config: Optional[object] = None,
        kv_cache_dtype: KvCacheDataType = KvCacheDataType.BASE,
        prefix_chunk_tokens: int = 0,
        parallelism_config: Optional[Any] = None,
    ) -> None:
        if weights is None:
            raise ValueError("FlashMLA Prefill requires MLA projection weights")
        if kv_cache_dtype != KvCacheDataType.BASE:
            raise ValueError("dense FlashMLA Prefill currently requires BF16 KV cache")

        # Import lazily: selecting FlashInfer must not require FlashMLA, while an
        # explicitly selected FlashMLA backend must fail rather than fall back.
        try:
            import flash_mla.cuda as flash_mla_cuda
        except ImportError as error:
            raise RuntimeError(
                "dense FlashMLA Prefill requires the CUDA13 flash-mla package"
            ) from error

        self.flash_mla_cuda = flash_mla_cuda
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.page_size = page_size
        self.prefix_chunk_tokens = int(prefix_chunk_tokens)
        if self.prefix_chunk_tokens < 0:
            raise ValueError(
                "FlashMLA prefix chunk capacity must be non-negative, got "
                f"{self.prefix_chunk_tokens}"
            )
        if self.prefix_chunk_tokens and (
            self.prefix_chunk_tokens < page_size or self.prefix_chunk_tokens % page_size
        ):
            raise ValueError(
                "FlashMLA prefix chunk capacity must be at least one cache "
                "page and divisible by it, got "
                f"chunk={self.prefix_chunk_tokens} page={page_size}"
            )
        self.scale = (
            (qk_nope_head_dim + qk_rope_head_dim) ** -0.5
        ) * softmax_extra_scale
        self.weights = weights
        self.quant_config = quant_config
        self.use_mla = use_mla
        self.qo_indptr: Optional[torch.Tensor] = None
        self.kv_indptr: Optional[torch.Tensor] = None
        self.max_q_len = 0
        self.max_kv_len = 0
        self.has_reuse_cache = False
        self.reuse_cache_page_indice: Optional[torch.Tensor] = None
        self.batch_reuse_info_vec: Optional[torch.Tensor] = None
        self.total_kv_lens = 0
        self.batch_size = 0
        self.q_lens: List[int] = []
        self.kv_lens: List[int] = []
        self.prefix_lens: List[int] = []
        self.batch_reuse_info_host: tuple[tuple[int, int, int, int], ...] = ()
        self.prefix_chunks: tuple[_FlashMLAPrefixChunk, ...] = ()
        self._prefix_chunk_metadata: Optional[torch.Tensor] = None
        self._direct_attn_inputs: Optional[Any] = None
        self._direct_block_table_width = 0
        self.prefix_lens: List[int] = []
        self.parallelism_config = parallelism_config

    def plan(self, mla_params: Any) -> None:
        if isinstance(mla_params, FlashMLADeviceParams):
            self.q_lens = list(mla_params.q_lens_host)
            self.kv_lens = list(mla_params.kv_lens_host)
            prefix_lens = list(mla_params.prefix_lens_host)
            self.qo_indptr = mla_params.qo_indptr_d
            self.kv_indptr = mla_params.prefill_ragged_kv_len_indptr_d
            self.has_reuse_cache = mla_params.has_reuse_cache
            batch_reuse_info_host = mla_params.batch_reuse_info_host
            self._direct_attn_inputs = mla_params.attn_inputs
            self._direct_block_table_width = mla_params.block_table_width
        else:
            qo_host = mla_params.qo_indptr_h
            kv_host = mla_params.prefill_ragged_kv_len_indptr_h
            qo_values = [int(value) for value in qo_host.tolist()]
            kv_values = [int(value) for value in kv_host.tolist()]
            if len(qo_values) != len(kv_values) or len(qo_values) < 2:
                raise ValueError(
                    f"invalid FlashMLA indptr: qo={qo_values}, kv={kv_values}"
                )
            self.qo_indptr = mla_params.qo_indptr_d
            self.kv_indptr = mla_params.prefill_ragged_kv_len_indptr_d
            self.q_lens = [
                qo_values[index + 1] - qo_values[index]
                for index in range(len(qo_values) - 1)
            ]
            self.kv_lens = [
                kv_values[index + 1] - kv_values[index]
                for index in range(len(kv_values) - 1)
            ]
            reuse_pages = mla_params.reuse_cache_page_indice_d
            self.has_reuse_cache = reuse_pages is not None and reuse_pages.numel() != 0
            reuse_host = mla_params.batch_reuse_info_vec_h
            if reuse_host is None or reuse_host.numel() == 0:
                prefix_lens = [0] * len(self.q_lens)
                batch_reuse_info_host = tuple(
                    (batch_idx, 0, 0, 0) for batch_idx in range(len(self.q_lens))
                )
            else:
                reuse_rows = reuse_host.reshape(-1, 4)
                if reuse_rows.shape[0] != len(self.q_lens):
                    raise ValueError(
                        "FlashMLA batch reuse metadata disagrees with qo_indptr: "
                        f"batch={len(self.q_lens)}, "
                        f"reuse_rows={reuse_rows.shape[0]}"
                    )
                batch_reuse_info_host = tuple(
                    (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
                    for row in reuse_rows.tolist()
                )
                prefix_lens = [row[1] for row in batch_reuse_info_host]
            self._direct_attn_inputs = None
            self._direct_block_table_width = 0

        self.max_q_len = max(self.q_lens)
        self.max_kv_len = max(self.kv_lens)
        self.total_kv_lens = sum(self.kv_lens)
        self.batch_size = len(self.q_lens)
        self.prefix_lens = prefix_lens
        self.batch_reuse_info_host = batch_reuse_info_host
        self.reuse_cache_page_indice = mla_params.reuse_cache_page_indice_d
        self.batch_reuse_info_vec = mla_params.batch_reuse_info_vec_d
        expected_kv_lens = [
            q_len + prefix_len for q_len, prefix_len in zip(self.q_lens, prefix_lens)
        ]
        if expected_kv_lens != self.kv_lens:
            raise ValueError(
                "FlashMLA Q/KV lengths disagree with cache reuse metadata: "
                f"expected={expected_kv_lens}, actual={self.kv_lens}"
            )
        self._materialize_prefix_chunks()

    def _materialize_prefix_chunks(self) -> None:
        specs = plan_flashmla_prefix_chunks(
            self.q_lens,
            self.prefix_lens,
            chunk_tokens=self.prefix_chunk_tokens,
            page_size=self.page_size,
        )
        if not specs:
            self.prefix_chunks = ()
            self._prefix_chunk_metadata = None
            return
        if self.qo_indptr is None:
            raise RuntimeError(
                "FlashMLA prefix chunks require query and cache-reuse metadata"
            )
        if len(self.batch_reuse_info_host) != self.batch_size or any(
            len(row) != 4 for row in self.batch_reuse_info_host
        ):
            raise RuntimeError(
                "FlashMLA host cache-reuse metadata must be [batch, 4], got "
                f"batch={self.batch_size} rows={self.batch_reuse_info_host}"
            )

        device = self.qo_indptr.device
        flat_values: list[int] = []
        descriptors: list[
            tuple[
                FlashMLAPrefixChunkSpec,
                tuple[int, int],
                tuple[int, int],
                tuple[int, int],
                tuple[int, int],
            ]
        ] = []

        def append(values: Sequence[int]) -> tuple[int, int]:
            offset = len(flat_values)
            flat_values.extend(int(value) for value in values)
            return offset, len(values)

        for spec in specs:
            q_lens = [self.q_lens[index] for index in spec.request_indices]
            batch_reuse_info: list[int] = []
            for local_idx, (request_idx, prefix_start, prefix_len) in enumerate(
                zip(
                    spec.request_indices,
                    spec.prefix_starts,
                    spec.prefix_lens,
                    strict=True,
                )
            ):
                base_row = self.batch_reuse_info_host[request_idx]
                batch_reuse_info.extend(
                    (
                        local_idx,
                        prefix_len,
                        base_row[2] + prefix_start // self.page_size,
                        (prefix_len + self.page_size - 1) // self.page_size,
                    )
                )
            descriptors.append(
                (
                    spec,
                    append(_indptr(q_lens)),
                    append(_indptr(spec.prefix_lens)),
                    append([0] * (len(spec.request_indices) + 1)),
                    append(batch_reuse_info),
                )
            )

        metadata = torch.tensor(flat_values, dtype=torch.int32, device=device)
        chunks: list[_FlashMLAPrefixChunk] = []
        for spec, qo_desc, kv_desc, gather_desc, reuse_desc in descriptors:

            def view(desc: tuple[int, int]) -> torch.Tensor:
                return metadata.narrow(0, desc[0], desc[1])

            chunks.append(
                _FlashMLAPrefixChunk(
                    spec=spec,
                    qo_indptr=view(qo_desc),
                    kv_indptr=view(kv_desc),
                    gather_qo_indptr=view(gather_desc),
                    batch_reuse_info=view(reuse_desc).view(-1, 4),
                )
            )
        metadata.record_stream(torch.cuda.current_stream(device))
        self._prefix_chunk_metadata = metadata
        self.prefix_chunks = tuple(chunks)

    def _current_reuse_cache_page_indices(self) -> torch.Tensor:
        if self.reuse_cache_page_indice is None:
            raise RuntimeError("FlashMLA cache reuse has no page indices")
        reuse_cache_page_indice = self.reuse_cache_page_indice
        if self._direct_attn_inputs is None:
            return reuse_cache_page_indice

        block_table = getattr(
            self._direct_attn_inputs,
            "kv_cache_kernel_block_id_device",
            None,
        )
        if (
            block_table is None
            or not block_table.is_cuda
            or block_table.dtype != torch.int32
            or block_table.dim() != 2
            or block_table.shape[0] != self.batch_size
            or block_table.shape[1] != self._direct_block_table_width
        ):
            raise RuntimeError(
                "FlashMLA current HybridCache group block table no longer "
                "matches its direct plan"
            )
        reuse_cache_page_indice = block_table.contiguous().view(-1)
        reuse_cache_page_indice.record_stream(
            torch.cuda.current_stream(block_table.device)
        )
        return reuse_cache_page_indice

    def _gather_reused_kv(
        self,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_k_pe = k_pe.view(-1, self.qk_rope_head_dim)
        if not self.has_reuse_cache:
            return compressed_kv, flat_k_pe
        if mla_cache_tp_enabled(self.parallelism_config):
            if kv_cache is None:
                raise RuntimeError("FlashMLA cache reuse requires an MLA KV cache")
            return self._gather_reused_kv_cache_tp(
                compressed_kv, flat_k_pe, kv_cache
            )

        kv_cache_base, reuse_cache_page_indice = self._reuse_cache_inputs(
            compressed_kv, kv_cache
        )

        final_compressed_kv = torch.empty(
            (self.total_kv_lens, self.kv_lora_rank),
            dtype=compressed_kv.dtype,
            device=compressed_kv.device,
        )
        final_k_pe = torch.empty(
            (self.total_kv_lens, self.qk_rope_head_dim),
            dtype=flat_k_pe.dtype,
            device=flat_k_pe.device,
        )
        rtp_llm_ops.reuse_kv_cache_indexed_batched(
            final_compressed_kv,
            final_k_pe,
            compressed_kv,
            flat_k_pe.contiguous(),
            kv_cache_base,
            reuse_cache_page_indice,
            self.batch_reuse_info_vec,
            self.qo_indptr,
            self.page_size,
        )
        return final_compressed_kv, final_k_pe

    def _reuse_cache_inputs(
        self,
        compressed_kv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if kv_cache is None:
            raise RuntimeError("FlashMLA cache reuse requires an MLA KV cache")
        if self.reuse_cache_page_indice is None:
            raise RuntimeError("FlashMLA cache reuse has no page indices")
        if self.batch_reuse_info_vec is None:
            raise RuntimeError("FlashMLA cache reuse has no batch metadata")
        if self.qo_indptr is None:
            raise RuntimeError("FlashMLA cache reuse has no query indptr")
        if self.total_kv_lens < compressed_kv.shape[0]:
            raise RuntimeError(
                "FlashMLA total KV length is smaller than the current suffix: "
                f"kv={self.total_kv_lens}, suffix={compressed_kv.shape[0]}"
            )

        reuse_cache_page_indice = self._current_reuse_cache_page_indices()
        return kv_cache.kv_cache_base, reuse_cache_page_indice

    def _gather_reused_kv_cache_tp(
        self,
        compressed_kv: torch.Tensor,
        flat_k_pe: torch.Tensor,
        kv_cache: LayerKVCache,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rebuild reused full-576 history from dimension-sharded cache pages.

        This is the explicit first-version repack path. It gathers only prefix
        tokens referenced by the current batch, not the whole resident pool.
        NCCL and repack remain separate PyTorch operations so phase-two tests
        can time them independently.
        """

        if self._direct_attn_inputs is None:
            raise RuntimeError(
                "K3 MLA cache TP prefix reuse requires direct K3 block metadata"
            )
        block_table = getattr(
            self._direct_attn_inputs, "kv_cache_kernel_block_id_device", None
        )
        if (
            block_table is None
            or not block_table.is_cuda
            or block_table.dtype != torch.int32
            or block_table.ndim != 2
            or block_table.shape[0] != self.batch_size
        ):
            raise RuntimeError("K3 MLA cache TP has invalid prefix block table")

        layout = kimi_k3_mla_cache_layout(self.parallelism_config)
        cache = kv_cache.kv_cache_base
        if cache.ndim != 3 or cache.shape[-1] != layout.local_width:
            raise RuntimeError(
                "K3 MLA cache TP expects [blocks,page,local_width], got "
                f"{tuple(cache.shape)}"
            )

        local_prefix_parts = []
        for request, prefix_len in enumerate(self.prefix_lens):
            if prefix_len <= 0:
                continue
            page_count = (prefix_len + self.page_size - 1) // self.page_size
            page_ids = block_table[request, :page_count].to(torch.long)
            request_cache = cache.index_select(0, page_ids).reshape(
                -1, layout.local_width
            )
            local_prefix_parts.append(request_cache[:prefix_len])
        local_prefix = torch.cat(local_prefix_parts, dim=0).contiguous()
        total_prefix = int(local_prefix.shape[0])
        gathered = all_gather(local_prefix, group=Group.TP).reshape(
            layout.tp_size, total_prefix, layout.local_width
        )
        full_prefix_ckv, full_prefix_pe = layout.reconstruct_rank_major(gathered)

        ckv_parts = []
        pe_parts = []
        prefix_offset = 0
        suffix_offset = 0
        for q_len, prefix_len in zip(
            self.q_lens, self.prefix_lens, strict=True
        ):
            if prefix_len:
                ckv_parts.append(
                    full_prefix_ckv[prefix_offset : prefix_offset + prefix_len]
                )
                pe_parts.append(
                    full_prefix_pe[prefix_offset : prefix_offset + prefix_len]
                )
                prefix_offset += prefix_len
            ckv_parts.append(compressed_kv[suffix_offset : suffix_offset + q_len])
            pe_parts.append(flat_k_pe[suffix_offset : suffix_offset + q_len])
            suffix_offset += q_len
        return torch.cat(ckv_parts, dim=0), torch.cat(pe_parts, dim=0)

    def _project_reused_kv_with_gap_fill(
        self,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        kv_b_proj: LinearBase,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Use the fused paged KPE gather before the packed KV projection."""

        fused_gather = getattr(rtp_llm_ops, "gather_mla_latent_and_fill_k_pe", None)
        packed_projection = self._packed_kv_projection(compressed_kv, kv_b_proj)
        if (
            not self.has_reuse_cache
            or not callable(fused_gather)
            or packed_projection is None
        ):
            return None

        flat_k_pe = k_pe.view(-1, self.qk_rope_head_dim)
        kv_cache_base, reuse_cache_page_indice = self._reuse_cache_inputs(
            compressed_kv, kv_cache
        )
        head_splits = (
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        )
        packed_head_dim = sum(head_splits)
        final_compressed_kv = torch.empty(
            (self.total_kv_lens, self.kv_lora_rank),
            dtype=compressed_kv.dtype,
            device=compressed_kv.device,
        )
        packed_kv = torch.empty(
            (self.total_kv_lens, self.num_heads * packed_head_dim),
            dtype=compressed_kv.dtype,
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
            self.page_size,
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

    def _gather_prefix_chunk(
        self,
        chunk: _FlashMLAPrefixChunk,
        compressed_kv: torch.Tensor,
        flat_k_pe: torch.Tensor,
        kv_cache: LayerKVCache,
        reuse_cache_page_indice: torch.Tensor,
        outputs: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if outputs is None:
            final_compressed_kv = torch.empty(
                (chunk.spec.kv_tokens, self.kv_lora_rank),
                dtype=compressed_kv.dtype,
                device=compressed_kv.device,
            )
            final_k_pe = torch.empty(
                (chunk.spec.kv_tokens, self.qk_rope_head_dim),
                dtype=flat_k_pe.dtype,
                device=flat_k_pe.device,
            )
        else:
            final_compressed_kv, final_k_pe = outputs
            expected_compressed_shape = (chunk.spec.kv_tokens, self.kv_lora_rank)
            expected_k_pe_shape = (chunk.spec.kv_tokens, self.qk_rope_head_dim)
            if (
                tuple(final_compressed_kv.shape) != expected_compressed_shape
                or final_compressed_kv.dtype != compressed_kv.dtype
                or final_compressed_kv.device != compressed_kv.device
                or not final_compressed_kv.is_contiguous()
                or tuple(final_k_pe.shape) != expected_k_pe_shape
                or final_k_pe.dtype != flat_k_pe.dtype
                or final_k_pe.device != flat_k_pe.device
                or not final_k_pe.is_contiguous()
            ):
                raise RuntimeError(
                    "FlashMLA prefix gather output buffer mismatch: "
                    f"compressed(shape={tuple(final_compressed_kv.shape)}, "
                    f"dtype={final_compressed_kv.dtype}, "
                    f"device={final_compressed_kv.device}, "
                    f"contiguous={final_compressed_kv.is_contiguous()}); "
                    f"k_pe(shape={tuple(final_k_pe.shape)}, "
                    f"dtype={final_k_pe.dtype}, device={final_k_pe.device}, "
                    f"contiguous={final_k_pe.is_contiguous()})"
                )
        rtp_llm_ops.reuse_kv_cache_indexed_batched(
            final_compressed_kv,
            final_k_pe,
            compressed_kv,
            flat_k_pe,
            kv_cache.kv_cache_base,
            reuse_cache_page_indice,
            chunk.batch_reuse_info,
            chunk.gather_qo_indptr,
            self.page_size,
        )
        return final_compressed_kv, final_k_pe

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
        packed_output: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if kv_b_proj is None:
            kv_b_proj = self._create_kv_b_proj(layer_id)
        head_splits = (
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        )
        packed_projection = self._packed_kv_projection(compressed_kv, kv_b_proj)
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

        k = compressed_kv.new_empty(
            compressed_kv.shape[0],
            self.num_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
        )
        k[..., : self.qk_nope_head_dim].copy_(k_nope)
        k[..., self.qk_nope_head_dim :].copy_(
            k_pe.view(num_tokens, 1, self.qk_rope_head_dim)
        )
        return k, value_states

    def _create_kv_b_proj(self, layer_id: int) -> LinearBase:
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
        if max_q_len <= 0 or max_kv_len <= 0:
            raise ValueError(
                "FlashMLA attention lengths must be positive, got "
                f"q={max_q_len} kv={max_kv_len}"
            )
        if (out is None) != (lse is None):
            raise ValueError("FlashMLA attention out and LSE must be provided together")
        if out is None or lse is None:
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
            expected_out_shape = (q.shape[0], self.num_heads, self.v_head_dim)
            expected_lse_shape = (q.shape[0], self.num_heads)
            if (
                tuple(out.shape) != expected_out_shape
                or out.dtype != q.dtype
                or out.device != q.device
                or not out.is_contiguous()
                or tuple(lse.shape) != expected_lse_shape
                or lse.dtype != torch.float32
                or lse.device != q.device
                or lse.stride() != (1, q.shape[0])
            ):
                raise RuntimeError(
                    "FlashMLA attention output buffer mismatch: "
                    f"out(shape={tuple(out.shape)}, dtype={out.dtype}, "
                    f"device={out.device}, contiguous={out.is_contiguous()}); "
                    f"lse(shape={tuple(lse.shape)}, dtype={lse.dtype}, "
                    f"device={lse.device}, stride={lse.stride()})"
                )
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
        if self.qo_indptr is None or self.kv_indptr is None:
            raise RuntimeError("FlashMLA Prefill must be planned before forward")
        out, _ = self._run_dense_attention(
            q,
            k,
            value_states,
            qo_indptr=self.qo_indptr,
            kv_indptr=self.kv_indptr,
            max_q_len=self.max_q_len,
            max_kv_len=self.max_kv_len,
            causal=True,
        )
        return out

    def _forward_chunked_prefix(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.qo_indptr is None:
            raise RuntimeError("FlashMLA Prefill must be planned before forward")
        if kv_cache is None:
            raise RuntimeError("FlashMLA prefix chunks require an MLA KV cache")
        expected_suffix_tokens = sum(self.q_lens)
        if compressed_kv.shape[0] != expected_suffix_tokens:
            raise RuntimeError(
                "FlashMLA current KV disagrees with packed Q lengths: "
                f"tensor={compressed_kv.shape[0]} expected={expected_suffix_tokens}"
            )

        kv_b_proj = self._create_kv_b_proj(layer_id)
        packed_projection = self._packed_kv_projection(compressed_kv, kv_b_proj)
        if packed_projection is None:
            raise RuntimeError(
                "FlashMLA historical-prefix chunking requires packed "
                "skip-head-mid projection capability"
            )
        suffix_k, suffix_v = self._project_kv(compressed_kv, k_pe, layer_id, kv_b_proj)
        output, output_lse = self._run_dense_attention(
            q,
            suffix_k,
            suffix_v,
            qo_indptr=self.qo_indptr,
            kv_indptr=self.qo_indptr,
            max_q_len=self.max_q_len,
            max_kv_len=self.max_q_len,
            causal=True,
        )
        del suffix_k, suffix_v
        if len(self.prefix_chunks) > 1:
            # FlashMLA emits BF16 partial states. Keep the recurrence in FP32
            # and round only once after the last historical-prefix chunk.
            output = output.float()

        flat_k_pe = k_pe.view(-1, self.qk_rope_head_dim).contiguous()
        reuse_cache_page_indice = self._current_reuse_cache_page_indices()
        fused_prefix_gather = getattr(
            rtp_llm_ops, "gather_mla_latent_and_fill_k_pe", None
        )
        use_fused_prefix_gather = callable(fused_prefix_gather)
        packed_features = self.num_heads * sum(_K3_PACKED_KV_HEAD_SPLITS)
        chunk_workspace = _FlashMLAPrefixChunkWorkspace.allocate(
            chunks=self.prefix_chunks,
            compressed_kv=compressed_kv,
            k_pe=flat_k_pe,
            q=q,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            num_heads=self.num_heads,
            v_head_dim=self.v_head_dim,
            packed_features=packed_features,
            allocate_k_pe=not use_fused_prefix_gather,
        )
        for chunk in self.prefix_chunks:
            if use_fused_prefix_gather:
                assert callable(fused_prefix_gather)
                chunk_compressed_kv = chunk_workspace.compressed_kv.narrow(
                    0, 0, chunk.spec.kv_tokens
                )
                chunk_packed_kv = chunk_workspace.packed_kv_buffer(chunk.spec.kv_tokens)
                assert chunk_packed_kv is not None
                packed_head_dim = sum(_K3_PACKED_KV_HEAD_SPLITS)
                fused_prefix_gather(
                    chunk_compressed_kv,
                    chunk_packed_kv,
                    compressed_kv,
                    flat_k_pe,
                    kv_cache.kv_cache_base,
                    reuse_cache_page_indice,
                    chunk.batch_reuse_info,
                    chunk.gather_qo_indptr,
                    self.page_size,
                    packed_head_dim,
                    self.qk_nope_head_dim,
                )
                packed_projection.forward_skip_head_mid(
                    chunk_compressed_kv,
                    _K3_PACKED_KV_HEAD_SPLITS,
                    output=chunk_packed_kv,
                )
                packed_kv = chunk_packed_kv.view(
                    chunk.spec.kv_tokens,
                    self.num_heads,
                    packed_head_dim,
                )
                chunk_k = packed_kv[..., : -self.v_head_dim]
                chunk_v = packed_kv[..., -self.v_head_dim :]
            else:
                gather_buffers = chunk_workspace.gather_buffers(chunk.spec.kv_tokens)
                chunk_compressed_kv, chunk_k_pe = self._gather_prefix_chunk(
                    chunk,
                    compressed_kv,
                    flat_k_pe,
                    kv_cache,
                    reuse_cache_page_indice,
                    outputs=gather_buffers,
                )
                chunk_k, chunk_v = self._project_kv(
                    chunk_compressed_kv,
                    chunk_k_pe,
                    layer_id,
                    kv_b_proj,
                    packed_output=chunk_workspace.packed_kv_buffer(
                        chunk.spec.kv_tokens
                    ),
                )
            q_chunk = q.narrow(0, chunk.spec.q_start, chunk.spec.q_tokens)
            attention_buffers = chunk_workspace.attention_buffers(chunk.spec.q_tokens)
            chunk_output, chunk_lse = self._run_dense_attention(
                q_chunk,
                chunk_k,
                chunk_v,
                qo_indptr=chunk.qo_indptr,
                kv_indptr=chunk.kv_indptr,
                max_q_len=max(
                    self.q_lens[index] for index in chunk.spec.request_indices
                ),
                max_kv_len=max(chunk.spec.prefix_lens),
                causal=False,
                out=attention_buffers[0],
                lse=attention_buffers[1],
            )
            output_view = output.narrow(0, chunk.spec.q_start, chunk.spec.q_tokens)
            lse_view = output_lse.narrow(0, chunk.spec.q_start, chunk.spec.q_tokens)
            merge_attention_states_in_place(
                output_view,
                lse_view,
                chunk_output,
                chunk_lse,
            )
            del (
                chunk_compressed_kv,
                chunk_k,
                chunk_v,
                chunk_output,
                chunk_lse,
            )
        if output.dtype != q.dtype:
            output = output.to(q.dtype)
        return output, output_lse

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_id: int,
    ) -> torch.Tensor:
        if self.qo_indptr is None or self.kv_indptr is None:
            raise RuntimeError("FlashMLA Prefill must be planned before forward")
        if self.prefix_chunks:
            out, _ = self._forward_chunked_prefix(
                q, compressed_kv, k_pe, kv_cache, layer_id
            )
        else:
            kv_b_proj = self._create_kv_b_proj(layer_id)
            projected_kv = self._project_reused_kv_with_gap_fill(
                compressed_kv, k_pe, kv_cache, kv_b_proj
            )
            if projected_kv is None:
                compressed_kv, k_pe = self._gather_reused_kv(
                    compressed_kv, k_pe, kv_cache
                )
                projected_kv = self._project_kv(
                    compressed_kv, k_pe, layer_id, kv_b_proj
                )
            if projected_kv[0].shape[0] != self.total_kv_lens:
                raise RuntimeError(
                    "FlashMLA gathered KV length disagrees with kv_indptr: "
                    f"tensor={projected_kv[0].shape[0]}, "
                    f"indptr={self.total_kv_lens}"
                )
            k, value_states = projected_kv
            out = self._dense_attention(q, k, value_states)

        device_index = q.device.index if q.device.index is not None else 0
        if device_index not in _FLASHMLA_LOGGED_DEVICES:
            logging.info(
                "dense Prefill MLA backend: FlashMLA "
                "device=%s heads=%d qk_dim=%d v_dim=%d",
                q.device,
                self.num_heads,
                self.qk_nope_head_dim + self.qk_rope_head_dim,
                self.v_head_dim,
            )
            _FLASHMLA_LOGGED_DEVICES.add(device_index)
        config = (
            device_index,
            self.batch_size,
            tuple(self.q_lens),
            self.has_reuse_cache,
            self.prefix_chunk_tokens,
        )
        if config not in _FLASHMLA_LOGGED_CONFIGS:
            logging.info(
                "dense FlashMLA Prefill plan: batch=%d q_lens=%s "
                "kv_lens=%s reuse_cache=%s prefix_chunk_tokens=%d "
                "prefix_chunks=%d",
                self.batch_size,
                self.q_lens,
                self.kv_lens,
                self.has_reuse_cache,
                self.prefix_chunk_tokens,
                len(self.prefix_chunks),
            )
            _FLASHMLA_LOGGED_CONFIGS.add(config)
        return out

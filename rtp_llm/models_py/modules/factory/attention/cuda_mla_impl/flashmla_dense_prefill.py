"""Dense causal MLA Prefill backed by FlashMLA's SM100 kernel.

Kimi K3 is a dense MLA model.  RTP's existing FlashMLA wrapper is sparse-only,
so it is not selected for K3 and the generic path falls back to FlashInfer.
This adapter keeps RTP's existing projections, RoPE/no-RoPE handling, cache
write and output projection unchanged; only the dense causal attention core is
replaced.
"""

import logging
from typing import Any, Dict, List, Optional

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    has_bf16_gemm_nt_skip_head_mid,
)
from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory
from rtp_llm.ops import KvCacheDataType
from rtp_llm.ops.compute_ops import LayerKVCache, rtp_llm_ops
from rtp_llm.utils.model_weight import W

_FLASHMLA_WORKSPACES: Dict[int, torch.Tensor] = {}
_FLASHMLA_LOGGED_DEVICES: set[int] = set()
# Log each static execution shape once. KV lengths are intentionally excluded:
# target verification grows them every iteration while reusing the same plan
# shape, so including them turns this guard into a per-step INFO log.
_FLASHMLA_LOGGED_CONFIGS: set[tuple[int, int, tuple[int, ...], bool]] = set()
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
    ) -> None:
        if weights is None:
            raise ValueError("FlashMLA Prefill requires MLA projection weights")
        if kv_cache_dtype != KvCacheDataType.BASE:
            raise ValueError(
                "Kimi K3 dense FlashMLA Prefill currently requires BF16 KV cache"
            )

        # Import lazily: selecting FlashInfer must not require FlashMLA, while an
        # explicitly selected FlashMLA backend must fail rather than fall back.
        try:
            import flash_mla.cuda as flash_mla_cuda
        except ImportError as error:
            raise RuntimeError(
                "K3 Prefill requires the CUDA13 flash-mla package"
            ) from error

        self.flash_mla_cuda = flash_mla_cuda
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.page_size = page_size
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
        self._direct_attn_inputs: Optional[Any] = None
        self._direct_block_table_width = 0

    def plan(self, mla_params: Any) -> None:
        if isinstance(mla_params, FlashMLADeviceParams):
            self.q_lens = list(mla_params.q_lens_host)
            self.kv_lens = list(mla_params.kv_lens_host)
            prefix_lens = list(mla_params.prefix_lens_host)
            self.qo_indptr = mla_params.qo_indptr_d
            self.kv_indptr = mla_params.prefill_ragged_kv_len_indptr_d
            self.has_reuse_cache = mla_params.has_reuse_cache
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
            else:
                reuse_rows = reuse_host.reshape(-1, 4)
                if reuse_rows.shape[0] != len(self.q_lens):
                    raise ValueError(
                        "FlashMLA batch reuse metadata disagrees with qo_indptr: "
                        f"batch={len(self.q_lens)}, "
                        f"reuse_rows={reuse_rows.shape[0]}"
                    )
                prefix_lens = [int(row[1]) for row in reuse_rows.tolist()]
            self._direct_attn_inputs = None
            self._direct_block_table_width = 0

        self.max_q_len = max(self.q_lens)
        self.max_kv_len = max(self.kv_lens)
        self.total_kv_lens = sum(self.kv_lens)
        self.batch_size = len(self.q_lens)
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

    def _gather_reused_kv(
        self,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_k_pe = k_pe.view(-1, self.qk_rope_head_dim)
        if not self.has_reuse_cache:
            return compressed_kv, flat_k_pe
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

        reuse_cache_page_indice = self.reuse_cache_page_indice
        if self._direct_attn_inputs is not None:
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
            kv_cache.kv_cache_base,
            reuse_cache_page_indice,
            self.batch_reuse_info_vec,
            self.qo_indptr,
            self.page_size,
        )
        return final_compressed_kv, final_k_pe

    def _project_kv(
        self,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv_b_proj = LinearFactory.create_linear_from_weights(
            self.weights[layer_id],
            W.mla_kv_b_w,
            W.mla_kv_b_s,
            None,
            self.quant_config,
        )
        head_splits = (
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        )
        packed_projection = getattr(kv_b_proj, "forward_skip_head_mid", None)
        num_tokens = compressed_kv.shape[0]
        if (
            head_splits == _K3_PACKED_KV_HEAD_SPLITS
            and callable(packed_projection)
            and has_bf16_gemm_nt_skip_head_mid()
            and compressed_kv.is_cuda
            and compressed_kv.dtype == torch.bfloat16
            and getattr(kv_b_proj, "bias", None) is None
            and torch.cuda.get_device_capability(compressed_kv.device)[0] == 10
        ):
            packed_kv = packed_projection(compressed_kv, head_splits).view(
                num_tokens, self.num_heads, sum(head_splits)
            )
            packed_kv[..., self.qk_nope_head_dim : -self.v_head_dim].copy_(
                k_pe.view(num_tokens, 1, self.qk_rope_head_dim)
            )
            k = packed_kv[..., : -self.v_head_dim]
            value_states = packed_kv[..., -self.v_head_dim :]
            return k, value_states

        expanded_dim = self.qk_nope_head_dim + self.v_head_dim
        kv = kv_b_proj(compressed_kv).view(
            num_tokens, self.num_heads, expanded_dim
        )
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

    def _dense_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.qo_indptr is None or self.kv_indptr is None:
            raise RuntimeError("FlashMLA Prefill must be planned before forward")
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
        self.flash_mla_cuda.dense_prefill_fwd(
            _workspace(q.device),
            q,
            k,
            value_states,
            self.qo_indptr,
            self.kv_indptr,
            out,
            lse,
            1,
            self.scale,
            self.max_q_len,
            self.max_kv_len,
            True,
        )
        return out

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
        compressed_kv, k_pe = self._gather_reused_kv(compressed_kv, k_pe, kv_cache)
        if compressed_kv.shape[0] != self.total_kv_lens:
            raise RuntimeError(
                "FlashMLA gathered KV length disagrees with kv_indptr: "
                f"tensor={compressed_kv.shape[0]}, indptr={self.total_kv_lens}"
            )
        k, value_states = self._project_kv(compressed_kv, k_pe, layer_id)
        out = self._dense_attention(q, k, value_states)

        device_index = q.device.index if q.device.index is not None else 0
        if device_index not in _FLASHMLA_LOGGED_DEVICES:
            logging.info(
                "Kimi K3 dense Prefill MLA backend: FlashMLA "
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
        )
        if config not in _FLASHMLA_LOGGED_CONFIGS:
            logging.info(
                "Kimi K3 FlashMLA Prefill plan: batch=%d q_lens=%s "
                "kv_lens=%s reuse_cache=%s",
                self.batch_size,
                self.q_lens,
                self.kv_lens,
                self.has_reuse_cache,
            )
            _FLASHMLA_LOGGED_CONFIGS.add(config)
        return out

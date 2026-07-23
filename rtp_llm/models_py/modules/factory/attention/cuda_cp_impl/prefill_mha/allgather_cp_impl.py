import logging
import os
from typing import Optional

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    ParamsBase,
    PyAttentionInputs,
    fill_mla_params,
)

logger = logging.getLogger(__name__)

_cp_trt_workspace_buffer: Optional[torch.Tensor] = None


def get_cp_trt_workspace_buffer() -> torch.Tensor:
    global _cp_trt_workspace_buffer
    if _cp_trt_workspace_buffer is None:
        _cp_trt_workspace_buffer = get_trt_workspace_buffer()
    return _cp_trt_workspace_buffer


from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from flashinfer.cascade import merge_state
from flashinfer.page import append_paged_kv_cache
from flashinfer.prefill import trtllm_batch_context_with_kv_cache

# FlashAttention-4 paged attention (Blackwell). Replaces trtllm_batch_context in
# the non-prefix CP paged path (links A/B); reads the HND pool via a zero-copy
# transpose view and, unlike trtllm, supports fp8 KV cache paged.
try:
    from flash_attn.cute import flash_attn_varlen_func as _fa4_varlen_func

    _HAS_FA4 = True
except Exception:  # pragma: no cover - FA4 only shipped on cuda13 x86 (Blackwell)
    _fa4_varlen_func = None
    _HAS_FA4 = False


def _use_fa4_cp_paged() -> bool:
    """FA4 replaces trtllm for the CP paged-context path unless disabled."""
    return _HAS_FA4 and os.environ.get(
        "RTP_LLM_CP_PREFILL_FA4", "1"
    ).strip().lower() not in ("0", "false", "no", "off")


def _match_q_to_kv(q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    """FA4 requires q/k/v to share one dtype. The KV cache is stored in its
    configured dtype (fp8 for the fp8-KV-cache layers, else bf16) and is read
    as-is (never re-cast); q is the bf16 model activation, so cast it to match
    the KV dtype. => fp8 cache yields uniform-fp8 attention automatically
    (scale=1.0, no descale); bf16 cache stays bf16.
    """
    return q if q.dtype == kv.dtype else q.to(kv.dtype)


from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
    cast_kv_for_cache_append,
    fill_fp8_kv_cache_scale,
    plan_prefix_paged_attention,
)
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.fmha_cp_kv_triton import (
    triton_build_dual_prefix_extend,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    get_py_flashinfer_workspace_buffer,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.trtllm_gen import (
    get_trt_workspace_buffer,
)


@triton.jit
def _fused_restore_packed_kv_kernel(
    packed_ptr,
    unpad_ptr,
    k_ptr,
    v_ptr,
    TOTAL: tl.constexpr,
    NK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    per_token = 2 * NK
    token = offs // per_token
    field = offs - token * per_token
    src_token = tl.load(unpad_ptr + token, mask=mask, other=0).to(tl.int64)
    vals = tl.load(packed_ptr + src_token * per_token + field, mask=mask, other=0.0)

    is_k = field < NK
    tl.store(k_ptr + token * NK + field, vals, mask=mask & is_k)
    tl.store(v_ptr + token * NK + (field - NK), vals, mask=mask & ~is_k)


def _fused_restore_packed_kv(
    packed_kv: torch.Tensor,
    unpad_indices: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_count = int(unpad_indices.numel())
    restore_k = torch.empty(
        token_count,
        num_kv_heads,
        head_dim,
        device=packed_kv.device,
        dtype=packed_kv.dtype,
    )
    restore_v = torch.empty_like(restore_k)
    if token_count == 0:
        return restore_k, restore_v

    nk = num_kv_heads * head_dim
    total = token_count * 2 * nk
    _fused_restore_packed_kv_kernel[(triton.cdiv(total, 256),)](
        packed_kv,
        unpad_indices,
        restore_k,
        restore_v,
        TOTAL=total,
        NK=nk,
        BLOCK=256,
    )
    return restore_k, restore_v


def _generate_q_indices_device(
    cp_chunk_lengths, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized ``generate_q_indices`` returning int64 CUDA tensors.

    Same first-half/second-half zigzag split as ``cp_utils.generate_q_indices``,
    but each per-chunk range is built with ``torch.arange`` on ``device`` and
    concatenated. This avoids the O(seq) Python ``list.extend(range(...))`` build
    plus ``torch.tensor(list)`` H2D, which stalls the GPU for milliseconds at the
    start of long-context prefill (65536-token part1 ranges).
    """
    parts0, parts1 = [], []
    offset = 0
    for chunk_len in cp_chunk_lengths:
        chunk_len = int(chunk_len)
        half0 = (chunk_len + 1) // 2
        parts0.append(
            torch.arange(offset, offset + half0, device=device, dtype=torch.int64)
        )
        parts1.append(
            torch.arange(
                offset + half0, offset + chunk_len, device=device, dtype=torch.int64
            )
        )
        offset += chunk_len
    empty = torch.empty(0, device=device, dtype=torch.int64)
    return (
        torch.cat(parts0) if parts0 else empty,
        torch.cat(parts1) if parts1 else empty,
    )


def _generate_full_causal_kv_indices_device(
    cp_chunk_lengths, cp_rank: int, cp_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized ``generate_full_causal_kv_indices`` returning int64 CUDA tensors.

    Matches ``cp_utils.generate_full_causal_kv_indices`` exactly (full causal KV
    range per Q-half under zigzag load balancing) but builds the contiguous
    ranges with ``torch.arange`` on ``device`` instead of Python lists.
    """
    parts0, parts1 = [], []
    seq_offset = 0
    for chunk_len in cp_chunk_lengths:
        chunk_len = int(chunk_len)
        assert chunk_len % 2 == 0
        h = chunk_len // 2
        end_part0 = h * (cp_rank + 1)
        if end_part0 > 0:
            parts0.append(
                torch.arange(
                    seq_offset, seq_offset + end_part0, device=device, dtype=torch.int64
                )
            )
        end_part1 = h * (2 * cp_size - cp_rank)
        if end_part1 > 0:
            parts1.append(
                torch.arange(
                    seq_offset, seq_offset + end_part1, device=device, dtype=torch.int64
                )
            )
        seq_offset += chunk_len * cp_size
    empty = torch.empty(0, device=device, dtype=torch.int64)
    return (
        torch.cat(parts0) if parts0 else empty,
        torch.cat(parts1) if parts1 else empty,
    )


class PCPAllGatherAttnOp:
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
        backend: str = "auto",  # "auto", "fa2", or "fa3"
        causal: bool = True,
        kv_layout: str = "NHD",  # "NHD" or "HND"
    ):
        """
        Args:
            config: Model configuration
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (for GQA/MQA)
            head_dim: Dimension of each head
            backend: FlashInfer backend ("auto", "fa2", or "fa3")
            causal: Whether to use causal masking
            kv_layout: KV cache layout ("NHD" or "HND")
        """
        super().__init__()
        self.attn_inputs = attn_inputs
        self.attn_configs = attn_configs
        self.num_qo_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.backend = backend
        self.kv_layout = kv_layout

        assert causal == True
        self.device = torch.cuda.current_device()
        self.workspace_buffer = get_py_flashinfer_workspace_buffer()

        self.cp_info = attn_inputs.context_parallel_info

        self.prefill_cp_rank = parallelism_config.tp_rank
        self.prefill_cp_size = parallelism_config.tp_size

        # CP page-RR KV sharding geometry (mirrors MSAAttention / DeviceData::props):
        #   sharded = prefill_cp enabled AND kv_cache_sharded AND raw tp_size>1.
        # When sharded, each rank's local pool / block table only holds the
        # 1/cp_size physical blocks it owns, so writing the full all-gathered
        # sequence with FlashInfer append_paged_kv_cache (which has no -1 skip and
        # reads the full block table) would index past the sharded block table.
        # In that case we route the local-pool write through the sharding-aware
        # C++ writer (mha_kv_write_cache) + cp_kv_slot_mapping (non-owned -> -1).
        cp_cfg = parallelism_config.prefill_cp_config
        self._kv_sharded = bool(
            getattr(cp_cfg, "kv_cache_sharded", False) and self.prefill_cp_size > 1
        )
        self._cp_size = self.prefill_cp_size if self._kv_sharded else 1
        self._cp_rank = self.prefill_cp_rank if self._kv_sharded else 0

        self.seq_size_per_block = (
            attn_configs.kernel_tokens_per_block or attn_configs.tokens_per_block
        )

        self.q0_idx = self.q1_idx = None
        self.kv0_idx = self.kv1_idx = None
        self.kv_restore_unpad_indices = None

        self.prefill_wrappers = {
            "ragged": {
                name: BatchPrefillWithRaggedKVCacheWrapper(
                    self.workspace_buffer,
                    kv_layout=kv_layout,
                    backend=backend,
                )
                for name in ["part0", "part1"]
            },
            "paged": {
                "prefix": BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    kv_layout="HND",
                    backend=backend,
                ),
            },
        }
        self._can_use_trtllm_paged_context = self._can_use_trtllm_paged_context()

    def _should_use_forward_opt(self) -> bool:
        value = os.environ.get("RTP_LLM_CP_PREFILL_FORWARD_OPT", "1").strip()
        if value.lower() in ("0", "false", "no", "off"):
            return False
        fa4 = _use_fa4_cp_paged()
        # Paged-context backend must be available: FA4 (preferred) or trtllm.
        if not fa4 and not self._can_use_trtllm_paged_context:
            return False
        # fp8 KV cache stays excluded: it is a mixed bf16-Q / fp8-KV path, and
        # FA4's flash_attn_varlen_func requires q/k/v to share one dtype (link B
        # deferred pending a mixed-dtype path). trtllm also lacks fp8 paged here.
        if (
            self.has_prefix
            or self._kv_sharded
            or self.attn_configs.kv_cache_dtype == KvCacheDataType.FP8
        ):
            return False

        return True

    def _can_use_trtllm_paged_context(self) -> bool:
        if torch.cuda.get_device_capability()[0] != 10:
            return False
        try:
            from flashinfer.artifacts import ArtifactPath, CheckSumHash
            from flashinfer.jit.attention.modules import get_artifact

            return bool(
                get_artifact(
                    f"{ArtifactPath.TRTLLM_GEN_FMHA}/checksums.txt",
                    CheckSumHash.TRTLLM_GEN_FMHA,
                )
            )
        except Exception as e:
            logger.warning("Disable CP prefill TRTLLM paged context: %s", e)
            return False

    def _physical_block_table(self) -> torch.Tensor:
        """Physical paged-cache block table (per-rank, CP-RR compact under
        sharding). Same table GLM5/DSV4/MSA use for paged cache I/O — addresses
        physical pages, not the (possibly token-level) kernel block table."""
        phys = getattr(self.attn_inputs, "kv_cache_block_id_device", None)
        if isinstance(phys, torch.Tensor) and phys.numel() > 0:
            return phys
        return self.attn_inputs.kv_cache_kernel_block_id_device

    def support(self, attention_inputs: PyAttentionInputs) -> bool:
        return attention_inputs.is_prefill

    def prepare(self, attention_inputs: PyAttentionInputs) -> ParamsBase:
        cu_seqlens = attention_inputs.cu_seqlens[
            : attention_inputs.input_lengths.size(0) + 1
        ]
        padding_mask = self.cp_info.prefill_qkv_padding_mask
        kv_restore_indices = self.cp_info.prefill_qkv_restore_indice
        self.kv_restore_unpad_indices = kv_restore_indices[padding_mask == 1]

        qo_indptr = cu_seqlens // 2

        q0_idx, q1_idx = _generate_q_indices_device(
            self.cp_info.prefill_cp_chunk_lengths, self.device
        )
        kv0_idx, kv1_idx = _generate_full_causal_kv_indices_device(
            self.cp_info.prefill_cp_chunk_lengths,
            self.prefill_cp_rank,
            self.prefill_cp_size,
            self.device,
        )

        self.kv0_idx = kv_restore_indices[kv0_idx]
        self.kv1_idx = kv_restore_indices[kv1_idx]
        self.q0_idx = q0_idx
        self.q1_idx = q1_idx

        kv_block_id_host = self.attn_inputs.kv_cache_kernel_block_id_host
        if kv_block_id_host is None:
            kv_block_id_host = self.attn_inputs.kv_cache_block_id_host
        tokens_per_block = (
            self.attn_configs.kernel_tokens_per_block
            or self.attn_configs.tokens_per_block
        )

        params = fill_mla_params(
            self.attn_inputs.prefix_lengths,
            self.attn_inputs.sequence_lengths,
            self.cp_info.prefill_actual_input_lengths_cpu,
            kv_block_id_host,
            tokens_per_block,
        )

        self._plan_ragged(qo_indptr, attention_inputs.prefix_lengths)
        q_lens = qo_indptr[1:] - qo_indptr[:-1]
        self._trtllm_max_q_len = int(q_lens.max().item())
        (
            self._trtllm_seq_lens_part0,
            self._trtllm_cu_kv_pages_part0,
            self._trtllm_max_kv_len_part0,
        ) = self._build_trtllm_paged_context_metadata(self.kv_indptr_part0)
        (
            self._trtllm_seq_lens_part1,
            self._trtllm_cu_kv_pages_part1,
            self._trtllm_max_kv_len_part1,
        ) = self._build_trtllm_paged_context_metadata(self.kv_indptr_part1)
        self.has_prefix = self.attn_inputs.prefix_lengths.any().item()
        self._use_forward_opt = self._should_use_forward_opt()
        if self.has_prefix:
            plan_prefix_paged_attention(
                self.prefill_wrappers["paged"]["prefix"],
                cu_seqlens,
                attention_inputs.prefix_lengths,
                params,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=self.seq_size_per_block,
                device=self.device,
                kv_data_type=(
                    torch.float8_e4m3fn
                    if self.attn_configs.kv_cache_dtype == KvCacheDataType.FP8
                    else torch.bfloat16
                ),
            )
        return params

    def _plan_ragged(
        self, qo_indptr: torch.Tensor, prefix_lengths: torch.Tensor
    ) -> None:
        self.qo_indptr = qo_indptr
        kv_indptr_part0 = qo_indptr * (self.prefill_cp_rank + 1)
        kv_indptr_part1 = qo_indptr * (2 * self.prefill_cp_size - self.prefill_cp_rank)
        self.kv_indptr_part0 = kv_indptr_part0
        self.kv_indptr_part1 = kv_indptr_part1
        common_params = {
            "num_qo_heads": self.num_qo_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim_qk": self.head_dim,
            "causal": True,
            "q_data_type": torch.bfloat16,
        }
        self.prefill_wrappers["ragged"]["part0"].plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr_part0,
            **common_params,
        )
        self.prefill_wrappers["ragged"]["part1"].plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr_part1,
            **common_params,
        )

    def _run_ragged_part(
        self,
        part: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        return_lse: bool = False,
    ):
        return self.prefill_wrappers["ragged"][part].run(q, k, v, return_lse=return_lse)

    def _build_trtllm_paged_context_metadata(
        self, kv_indptr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        seq_lens = (kv_indptr[1:] - kv_indptr[:-1]).to(
            device=self.device, dtype=torch.int32
        )
        pages_per_seq = (
            seq_lens + self.seq_size_per_block - 1
        ) // self.seq_size_per_block
        cu_kv_pages = torch.empty(
            seq_lens.numel() + 1, device=self.device, dtype=torch.int32
        )
        cu_kv_pages[0] = 0
        torch.cumsum(pages_per_seq, dim=0, out=cu_kv_pages[1:])
        return seq_lens, cu_kv_pages, int(seq_lens.max().item())

    def _run_trtllm_paged_context(
        self,
        q: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_lens: torch.Tensor,
        cu_kv_pages: torch.Tensor,
        max_kv_len: int,
        block_tables: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if block_tables is None:
            block_tables = self._physical_block_table()
        out = trtllm_batch_context_with_kv_cache(
            query=q,
            kv_cache=kv_cache_tensor,
            workspace_buffer=get_cp_trt_workspace_buffer(),
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_q_len=self._trtllm_max_q_len,
            max_kv_len=max_kv_len,
            bmm1_scale=self.head_dim**-0.5,
            bmm2_scale=1.0,
            batch_size=seq_lens.numel(),
            cum_seq_lens_q=self.qo_indptr,
            cum_seq_lens_kv=cu_kv_pages,
            window_left=-1,
            sinks=None,
            out_dtype=q.dtype,
        )
        return out

    def _run_fa4_paged_context(
        self,
        q: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_lens: torch.Tensor,
        max_kv_len: int,
        block_tables: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """FA4 paged causal attention over the HND pool (drop-in for trtllm).

        The HND pool ``[blocks, 2, H_kv, page, D]`` is fed to FA4 (which wants
        ``[num_pages, page, H_kv, D]``) via a zero-copy ``transpose(1, 2)`` view;
        FA4's TMA handles the strided K/V at full speed. Causal alignment and
        page-table/seqlen semantics match trtllm (verified numerically).
        """
        if block_tables is None:
            block_tables = self._physical_block_table()
        # Pool read as-is (fp8 when the KV cache is fp8 — zero-copy strided view);
        # q cast to match so fp8 cache => uniform-fp8 attention.
        k = kv_cache_tensor[:, 0].transpose(1, 2)
        v = kv_cache_tensor[:, 1].transpose(1, 2)
        q = _match_q_to_kv(q, k)
        out = _fa4_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=self.qo_indptr,
            max_seqlen_q=self._trtllm_max_q_len,
            max_seqlen_k=int(max_kv_len),
            seqused_k=seq_lens,
            page_table=block_tables,
            causal=True,
            softmax_scale=self.head_dim**-0.5,
        )
        return out[0] if isinstance(out, tuple) else out

    def _run_paged_context(
        self,
        q: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_lens: torch.Tensor,
        cu_kv_pages: torch.Tensor,
        max_kv_len: int,
    ) -> torch.Tensor:
        """Dispatch the CP paged-context attention to FA4 (default) or trtllm."""
        if _use_fa4_cp_paged():
            return self._run_fa4_paged_context(q, kv_cache_tensor, seq_lens, max_kv_len)
        return self._run_trtllm_paged_context(
            q, kv_cache_tensor, seq_lens, cu_kv_pages, max_kv_len
        )

    def _run_fa4_ragged(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
    ) -> torch.Tensor:
        """FA4 varlen (ragged) causal attention over contiguous [prefix||extend]
        K/V — drop-in for ``run_fmha_cp_ragged``. No page_table; k/v are the
        per-part concatenated tensors from ``triton_build_dual_prefix_extend``.
        Causal offset = seqlen_k - seqlen_q (bottom-right), matching fmha/trtllm.
        """
        max_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
        max_k = int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item())
        # k/v are the materialised [prefix||extend] (prefix in the cache dtype);
        # q cast to match so an fp8 cache yields uniform-fp8 attention.
        q = _match_q_to_kv(q, k)
        out = _fa4_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            causal=True,
            softmax_scale=self.head_dim**-0.5,
        )
        return out[0] if isinstance(out, tuple) else out

    def _forward_opt(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        params: ParamsBase = None,
    ) -> Optional[torch.Tensor]:
        qkv = qkv.reshape(qkv.shape[0], -1)
        q_size = self.head_dim * self.num_qo_heads
        packed_kv_size = 2 * self.head_dim * self.num_kv_heads
        q = qkv[:, :q_size]
        packed_kv = qkv[:, q_size : q_size + packed_kv_size].contiguous()

        all_packed_kv = all_gather(packed_kv, group=Group.TP).reshape(
            packed_kv.shape[0] * self.prefill_cp_size,
            2,
            self.num_kv_heads,
            self.head_dim,
        )
        q_reshaped = q.reshape(-1, self.num_qo_heads, self.head_dim)

        restore_k, restore_v = _fused_restore_packed_kv(
            all_packed_kv,
            self.kv_restore_unpad_indices,
            self.num_kv_heads,
            self.head_dim,
        )
        restore_token_count = restore_k.size(0)
        batch_indices = params.batch_indice_d.narrow(0, 0, restore_token_count)
        positions = params.positions_d.narrow(0, 0, restore_token_count)
        kv_cache_tensor = kv_cache.kv_cache_base.view(
            -1, 2, self.num_kv_heads, self.seq_size_per_block, self.head_dim
        )
        append_paged_kv_cache(
            append_key=restore_k,
            append_value=restore_v,
            batch_indices=batch_indices,
            positions=positions,
            paged_kv_cache=kv_cache_tensor,
            kv_indices=params.page_indice_d,
            kv_indptr=params.decode_page_indptr_d,
            kv_last_page_len=params.paged_kv_last_page_len_d,
            kv_layout="HND",
        )

        q0 = torch.index_select(q_reshaped, 0, self.q0_idx).contiguous()
        q1 = torch.index_select(q_reshaped, 0, self.q1_idx).contiguous()

        output = torch.empty_like(q_reshaped)
        output[self.q0_idx] = self._run_paged_context(
            q0,
            kv_cache_tensor,
            self._trtllm_seq_lens_part0,
            self._trtllm_cu_kv_pages_part0,
            self._trtllm_max_kv_len_part0,
        )
        output[self.q1_idx] = self._run_paged_context(
            q1,
            kv_cache_tensor,
            self._trtllm_seq_lens_part1,
            self._trtllm_cu_kv_pages_part1,
            self._trtllm_max_kv_len_part1,
        )
        return output

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        params: ParamsBase = None,
    ) -> torch.Tensor:
        if self._use_forward_opt:
            output = self._forward_opt(qkv, kv_cache, params)
            if output is not None:
                return output

        qkv = qkv.reshape(qkv.shape[0], -1)
        q, k, v = torch.split(
            qkv,
            [
                self.head_dim * self.num_qo_heads,
                self.head_dim * self.num_kv_heads,
                self.head_dim * self.num_kv_heads,
            ],
            dim=-1,
        )
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        all_keys = all_gather(k, group=Group.TP).reshape(
            k.shape[0] * self.prefill_cp_size, self.num_kv_heads, self.head_dim
        )
        all_values = all_gather(v, group=Group.TP).reshape(
            v.shape[0] * self.prefill_cp_size, self.num_kv_heads, self.head_dim
        )
        q_reshaped = q.reshape(-1, self.num_qo_heads, self.head_dim)
        del k, v

        # TODO: make write local kvcache async
        restore_k = all_keys[self.kv_restore_unpad_indices]
        restore_v = all_values[self.kv_restore_unpad_indices]
        nnz = restore_k.size(0)
        batch_indices = params.batch_indice_d.narrow(0, 0, nnz)
        positions = params.positions_d.narrow(0, 0, nnz)
        kv_cache_tensor = kv_cache.kv_cache_base.view(
            -1, 2, self.num_kv_heads, self.seq_size_per_block, self.head_dim
        )
        append_k, append_v = cast_kv_for_cache_append(
            restore_k, restore_v, kv_cache, self.attn_configs.kv_cache_dtype
        )
        if self._kv_sharded:
            # CP page-RR sharded: the local block table only holds this rank's
            # 1/cp_size owned blocks. Map each all-gathered token to its physical
            # slot via cp_kv_slot_mapping (non-owned / out-of-capacity -> -1) and
            # write through the C++ writer, which skips -1 slots. This stores
            # exactly this rank's owned page-RR shard, matching the MSA layers and
            # what decode's per-rank pool reader expects.
            from rtp_llm.models_py.modules.dsv4.fp8._cp_slot_mapping import (
                cp_kv_slot_mapping,
            )
            from rtp_llm.ops.compute_ops import rtp_llm_ops

            bt = self._physical_block_table().to(torch.int64)
            slot_mapping = cp_kv_slot_mapping(
                positions.to(torch.int64),
                bt,
                batch_indices.to(torch.int64),
                self.seq_size_per_block,  # tokens_per_block
                self.seq_size_per_block,  # kv_eb (entries per block, ratio=1)
                1,  # ratio (uncompressed MHA/GQA K/V)
                self._cp_size,
                self._cp_rank,
                owner_tokens_per_block=self.seq_size_per_block,
            )
            rtp_llm_ops.mha_kv_write_cache(
                append_k.contiguous(),
                append_v.contiguous(),
                kv_cache_tensor,
                slot_mapping,
            )
        else:
            append_paged_kv_cache(
                append_key=append_k,
                append_value=append_v,
                batch_indices=batch_indices,
                positions=positions,
                paged_kv_cache=kv_cache_tensor,
                kv_indices=params.page_indice_d,
                kv_indptr=params.decode_page_indptr_d,
                kv_last_page_len=params.paged_kv_last_page_len_d,
                kv_layout="HND",
            )
            # FP8 scale init only applies to the FlashInfer append path; the
            # sharded path's params.page_indice_d is full-length (would index past
            # the sharded block table), so it must not be used here. M3 uses a
            # bf16 paged pool (FP8_KV_CACHE=0) where this is a no-op anyway.
            fill_fp8_kv_cache_scale(
                kv_cache,
                params,
                batch_indices,
                positions,
                num_kv_heads=self.num_kv_heads,
                page_size=self.seq_size_per_block,
                kv_cache_dtype=self.attn_configs.kv_cache_dtype,
            )

        del restore_k, restore_v, append_k, append_v, batch_indices, positions

        if self.has_prefix:
            q0 = torch.index_select(q_reshaped, 0, self.q0_idx).contiguous()
            q1 = torch.index_select(q_reshaped, 0, self.q1_idx).contiguous()
            k0 = torch.index_select(all_keys, 0, self.kv0_idx).contiguous()
            k1 = torch.index_select(all_keys, 0, self.kv1_idx).contiguous()
            v0 = torch.index_select(all_values, 0, self.kv0_idx).contiguous()
            v1 = torch.index_select(all_values, 0, self.kv1_idx).contiguous()
            if _use_fa4_cp_paged():
                # Link C (varlen): reuse the fused [prefix||extend] build, then
                # FA4 varlen ragged per part (replaces the paged-prefix + ragged-
                # extend + merge_state hybrid, and the fmha_sm100 ragged path).
                prefix_lengths = self.attn_inputs.prefix_lengths
                # The prefix comes from the cache (fp8 for the fp8-KV-cache
                # layers); cast the bf16-gathered extend to the cache dtype so
                # the concatenated [prefix||extend] is uniform (fp8 => uniform
                # fp8 attention downstream). No-op for a bf16 cache.
                kv_dtype = kv_cache_tensor.dtype
                k0 = k0.to(kv_dtype)
                v0 = v0.to(kv_dtype)
                k1 = k1.to(kv_dtype)
                v1 = v1.to(kv_dtype)
                k0_cat, v0_cat, k1_cat, v1_cat = triton_build_dual_prefix_extend(
                    kv_cache_tensor,
                    prefix_lengths,
                    self._physical_block_table().to(torch.int32),
                    self.seq_size_per_block,
                    k0,
                    v0,
                    self.kv_indptr_part0,
                    k1,
                    v1,
                    self.kv_indptr_part1,
                )
                # cu_seqlens_k for [prefix||extend] = extend prefix-sum
                # (kv_indptr_partX) + prefix prefix-sum.
                pfx = torch.zeros_like(self.qo_indptr)
                pfx[1:] = prefix_lengths.to(self.qo_indptr.dtype).cumsum(0)
                output = torch.empty_like(q_reshaped)
                output[self.q0_idx] = self._run_fa4_ragged(
                    q0, k0_cat, v0_cat, self.qo_indptr, self.kv_indptr_part0 + pfx
                )
                output[self.q1_idx] = self._run_fa4_ragged(
                    q1, k1_cat, v1_cat, self.qo_indptr, self.kv_indptr_part1 + pfx
                )
                return output

            # Fallback (FA4 disabled): flashinfer paged prefix + ragged extend + LSE merge.
            prefix_out, prefix_lse = self.prefill_wrappers["paged"]["prefix"].run(
                q_reshaped, kv_cache_tensor, return_lse=True
            )

            out0, lse0 = self._run_ragged_part("part0", q0, k0, v0, return_lse=True)
            out1, lse1 = self._run_ragged_part("part1", q1, k1, v1, return_lse=True)
            out0, _ = merge_state(
                v_a=prefix_out[self.q0_idx],
                s_a=prefix_lse[self.q0_idx],
                v_b=out0,
                s_b=lse0,
            )
            out1, _ = merge_state(
                v_a=prefix_out[self.q1_idx],
                s_a=prefix_lse[self.q1_idx],
                v_b=out1,
                s_b=lse1,
            )
            output = torch.empty_like(q_reshaped)
            output[self.q0_idx] = out0
            output[self.q1_idx] = out1
            return output
        else:
            output = torch.empty_like(q_reshaped)
            q0 = torch.index_select(q_reshaped, 0, self.q0_idx).contiguous()
            k0 = torch.index_select(all_keys, 0, self.kv0_idx).contiguous()
            v0 = torch.index_select(all_values, 0, self.kv0_idx).contiguous()
            if _use_fa4_cp_paged():
                # No-prefix FA4 varlen on the gathered extend K/V. Cast to the
                # cache dtype so an fp8 KV cache yields uniform-fp8 attention
                # (fp8-cache non-prefix lands here since it is excluded from the
                # paged _forward_opt path); no-op for a bf16 cache.
                kv_dtype = kv_cache_tensor.dtype
                output[self.q0_idx] = self._run_fa4_ragged(
                    q0,
                    k0.to(kv_dtype),
                    v0.to(kv_dtype),
                    self.qo_indptr,
                    self.kv_indptr_part0,
                )
            else:
                output[self.q0_idx] = self._run_ragged_part("part0", q0, k0, v0)

            del q0, k0, v0
            q1 = torch.index_select(q_reshaped, 0, self.q1_idx).contiguous()
            k1 = torch.index_select(all_keys, 0, self.kv1_idx).contiguous()
            v1 = torch.index_select(all_values, 0, self.kv1_idx).contiguous()
            if _use_fa4_cp_paged():
                output[self.q1_idx] = self._run_fa4_ragged(
                    q1,
                    k1.to(kv_dtype),
                    v1.to(kv_dtype),
                    self.qo_indptr,
                    self.kv_indptr_part1,
                )
            else:
                output[self.q1_idx] = self._run_ragged_part("part1", q1, k1, v1)
            return output

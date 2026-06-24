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

from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
    cast_kv_for_cache_append,
    fill_fp8_kv_cache_scale,
    generate_full_causal_kv_indices,
    generate_q_indices,
    plan_prefix_paged_attention,
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
        if not self._can_use_trtllm_paged_context:
            return False
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

        q0_idx, q1_idx = generate_q_indices(self.cp_info.prefill_cp_chunk_lengths)
        kv0_idx, kv1_idx = generate_full_causal_kv_indices(
            self.cp_info.prefill_cp_chunk_lengths,
            self.prefill_cp_rank,
            self.prefill_cp_size,
        )

        self.kv0_idx = kv_restore_indices[kv0_idx]
        self.kv1_idx = kv_restore_indices[kv1_idx]
        self.q0_idx = torch.tensor(q0_idx, device=self.device)
        self.q1_idx = torch.tensor(q1_idx, device=self.device)

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

        self._plan_ragged(qo_indptr)
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

    def _plan_ragged(self, qo_indptr: torch.Tensor) -> None:
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
        output[self.q0_idx] = self._run_trtllm_paged_context(
            q0,
            kv_cache_tensor,
            self._trtllm_seq_lens_part0,
            self._trtllm_cu_kv_pages_part0,
            self._trtllm_max_kv_len_part0,
        )
        output[self.q1_idx] = self._run_trtllm_paged_context(
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

        q0 = torch.index_select(q_reshaped, 0, self.q0_idx).contiguous()
        q1 = torch.index_select(q_reshaped, 0, self.q1_idx).contiguous()

        k0 = torch.index_select(all_keys, 0, self.kv0_idx).contiguous()
        k1 = torch.index_select(all_keys, 0, self.kv1_idx).contiguous()
        v0 = torch.index_select(all_values, 0, self.kv0_idx).contiguous()
        v1 = torch.index_select(all_values, 0, self.kv1_idx).contiguous()
        if self.has_prefix:
            prefix_out, prefix_lse = self.prefill_wrappers["paged"]["prefix"].run(
                q_reshaped, kv_cache_tensor, return_lse=True
            )

            out0, lse0 = self.prefill_wrappers["ragged"]["part0"].run(
                q0, k0, v0, return_lse=True
            )
            out1, lse1 = self.prefill_wrappers["ragged"]["part1"].run(
                q1, k1, v1, return_lse=True
            )
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
            output[self.q0_idx] = self.prefill_wrappers["ragged"]["part0"].run(
                q0, k0, v0
            )
            output[self.q1_idx] = self.prefill_wrappers["ragged"]["part1"].run(
                q1, k1, v1
            )
            return output

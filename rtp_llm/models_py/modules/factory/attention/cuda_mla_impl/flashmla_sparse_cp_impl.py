"""
CP (context-parallel) variant of sparse MLA.
Mirrors flashmla_sparse_impl.py but with all-gather + restore + zig-zag q split.
"""

import copy
import logging
import os
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch

try:
    cuda_ver = torch.version.cuda or ""
    _major, _minor = (int(x) for x in (cuda_ver.split(".") + ["0", "0"])[:2])
    if (_major, _minor) >= (12, 9):
        from flash_mla import (
            flash_mla_sparse_fwd,
            flash_mla_with_kvcache,
            get_mla_metadata,
        )
except (ImportError, AttributeError, ValueError) as _e:
    logging.warning(f"flash_mla not available: {_e}. Requires CUDA >= 12.9")

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.triton_kv_scatter import (
    triton_kv_scatter,
)
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
    generate_q_indices,
)
from rtp_llm.ops import AttentionConfigs, FMHAConfig, FMHAType, ParallelismConfig
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs, rtp_llm_ops

from .flashmla_sparse_impl import (
    SparseMlaFp8DecodeParams,
    SparseMlaFp8Op,
    SparseMlaImpl,
    _as_uint8,
    _GatherWorkspace,
    _topk_2d,
)

_PD_DEBUG_PLAN_LOGGED: set[str] = set()


def _pd_debug_enabled() -> bool:
    return os.environ.get("RTP_LLM_PD_DEBUG", "0") == "1"


def _rank_tag() -> str:
    return (
        f"rank={os.environ.get('RANK', os.environ.get('WORLD_RANK', '?'))} "
        f"local_rank={os.environ.get('LOCAL_RANK', '?')}"
    )


def _cuda_graph_capturing() -> bool:
    try:
        return bool(
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
    except Exception:
        return False


def _tensor_summary(t: Optional[torch.Tensor]) -> str:
    if t is None:
        return "None"
    try:
        if t.is_cuda and _cuda_graph_capturing():
            return (
                f"shape={tuple(t.shape)} device={t.device} dtype={t.dtype} " "capture=1"
            )
        if t.numel() == 0:
            return f"shape={tuple(t.shape)} numel=0"
        tc = t.detach()
        if tc.is_cuda:
            tc = tc.cpu()
        if tc.numel() <= 16:
            return f"shape={tuple(t.shape)} values={tc.tolist()}"
        return (
            f"shape={tuple(t.shape)} numel={tc.numel()} "
            f"min={tc.min().item()} max={tc.max().item()} "
            f"head={tc[:4].tolist()} tail={tc[-4:].tolist()}"
        )
    except Exception as exc:
        return f"shape={tuple(t.shape)} summary_error={exc}"


class SparseMlaFp8CPOp(SparseMlaFp8Op):
    """Context-parallel sparse MLA prefill: all-gather KV, restore to global order,
    write to paged cache, then run attention only on q tokens this rank owns."""

    def __init__(
        self,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        page_size: int,
        softmax_extra_scale: float,
        top_k: int,
        parallelism_config: Optional[ParallelismConfig] = None,
    ):
        super().__init__(
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            page_size=page_size,
            softmax_extra_scale=softmax_extra_scale,
            top_k=top_k,
        )
        self.attn_inputs = None
        self.cp_info = None
        self.prefill_cp_rank = parallelism_config.tp_rank
        self.prefill_cp_size = parallelism_config.tp_size
        self.device = torch.cuda.current_device()

        # Filled per-forward in plan(); read by forward() and create_params()
        self.kv_restore_unpad_indices: Optional[torch.Tensor] = None
        self.total_global_ids: Optional[torch.Tensor] = None
        self.total_local_ids: Optional[torch.Tensor] = None
        self.cu_kv_seqlens_global: Optional[torch.Tensor] = None
        self.total_kv_len: int = 0
        self.precomputed_req_ids: Optional[torch.Tensor] = None
        self.full_rope_pos_ids: Optional[torch.Tensor] = None
        # Wired up by SparseMlaCpImpl post-construction
        self.kv_cache_write_op = None
        self.write_cache_store_impl = None

    def plan(
        self,
        mla_params: rtp_llm_ops.FlashInferMlaAttnParams,
        block_table: torch.Tensor,
        attn_inputs: Optional[PyAttentionInputs] = None,
    ) -> None:
        self.block_table = block_table
        self.mla_params = mla_params
        self.attn_inputs = attn_inputs
        self.cp_info = attn_inputs.context_parallel_info
        assert self.cp_info is not None, "context_parallel_info required for CP"

        chunk_lengths = self.cp_info.prefill_cp_chunk_lengths
        if isinstance(chunk_lengths, torch.Tensor):
            chunk_lengths_list = chunk_lengths.cpu().tolist()
        else:
            chunk_lengths_list = list(chunk_lengths)
        q0_idx, q1_idx = generate_q_indices(chunk_lengths_list)
        local_tokens = sum(chunk_lengths_list)

        # CPU tensors required by fill_cp_plan_params
        padding_mask_cpu = self.cp_info.prefill_qkv_padding_mask
        if padding_mask_cpu.is_cuda:
            padding_mask_cpu = padding_mask_cpu.cpu()
        kv_restore_cpu = self.cp_info.prefill_qkv_restore_indice
        if kv_restore_cpu.is_cuda:
            kv_restore_cpu = kv_restore_cpu.cpu()

        mla_params.fill_cp_plan_params(
            padding_mask_cpu,
            kv_restore_cpu,
            q0_idx,
            q1_idx,
            self.prefill_cp_rank,
            local_tokens,
            self.cp_info.prefill_actual_input_lengths_cpu,
            self.attn_inputs.prefix_lengths,
        )

        self.kv_restore_unpad_indices = mla_params.cp_kv_restore_unpad_indices
        self.total_global_ids = mla_params.cp_total_global_ids
        self.total_local_ids = mla_params.cp_total_local_ids
        self.cu_kv_seqlens_global = mla_params.cp_cu_kv_seqlens_global
        self.total_kv_len = mla_params.cp_total_kv_len

        n_q = self.total_global_ids.size(0)
        tile_sched_q0, num_splits_q0 = get_mla_metadata(  # type: ignore
            cache_seqlens=None,
            num_q_tokens_per_head_k=n_q * self.num_heads,
            topk=self.top_k,
            num_heads_q=self.num_heads,
            num_heads_k=1,
            is_fp8_kvcache=True,
        )
        self._fp8_kernel_metadata_q0 = SparseMlaFp8DecodeParams(
            tile_sched_q0, num_splits_q0
        )
        self.precomputed_req_ids = (
            mla_params.batch_indice_d[self.total_global_ids] if n_q > 0 else None
        )

        # Pack rope positions for the local q tokens into the full buffer; the
        # rope path consumes this as precomputed_pos_ids.
        if n_q > 0:
            positions_d = mla_params.positions_d
            full_rope = torch.zeros(
                positions_d.size(0),
                dtype=positions_d.dtype,
                device=positions_d.device,
            )
            full_rope[self.total_local_ids] = positions_d[self.total_global_ids]
            self.full_rope_pos_ids = full_rope
        else:
            self.full_rope_pos_ids = None

        # Gather path: prefill-only, gated by USE_GATHER_PATH (mirrors non-CP).
        gather_enabled = (
            os.environ.get("USE_GATHER_PATH", "0") == "1"
            and attn_inputs is not None
            and getattr(attn_inputs, "is_prefill", False)
        )
        self._gather = self._build_gather_workspace() if gather_enabled else None

        if _pd_debug_enabled():
            log_key = f"{os.getpid()}:{self.prefill_cp_rank}"
            if log_key not in _PD_DEBUG_PLAN_LOGGED:
                _PD_DEBUG_PLAN_LOGGED.add(log_key)
                logging.info(
                    "[PD_DEBUG][CP_MLA_PLAN] %s cp_rank=%s cp_size=%s "
                    "chunk_lengths=%s actual_lengths=%s prefix_lengths=%s "
                    "local_tokens=%s kv_restore=%s total_global=%s total_local=%s "
                    "cu_kv=%s total_kv_len=%s gather_enabled=%s",
                    _rank_tag(),
                    self.prefill_cp_rank,
                    self.prefill_cp_size,
                    chunk_lengths_list,
                    _tensor_summary(self.cp_info.prefill_actual_input_lengths_cpu),
                    _tensor_summary(self.attn_inputs.prefix_lengths),
                    local_tokens,
                    _tensor_summary(self.kv_restore_unpad_indices),
                    _tensor_summary(self.total_global_ids),
                    _tensor_summary(self.total_local_ids),
                    _tensor_summary(self.cu_kv_seqlens_global),
                    self.total_kv_len,
                    self._gather is not None,
                )

    def _build_gather_workspace(self) -> Optional[_GatherWorkspace]:
        """Allocate the BF16 workspace from prefill_ragged_kv_len_indptr_d.

        CP's prepare() sets input_lengths = prefill_actual_input_lengths, so the
        indptr reflects per-request full KV length (same as non-CP impl)."""
        assert self.mla_params is not None and self.block_table is not None
        batch_size = int(self.block_table.shape[0])
        indptr = self.mla_params.prefill_ragged_kv_len_indptr_d[: batch_size + 1]
        total_kv_len = int(indptr[batch_size].item())
        if total_kv_len == 0:
            return None
        return _GatherWorkspace(
            fused_kv=torch.empty(
                (total_kv_len, self.kv_lora_rank + self.qk_rope_head_dim),
                dtype=torch.bfloat16,
                device=self.block_table.device,
            ),
            workspace_starts=indptr[:batch_size],
            seq_lens=indptr[1:] - indptr[:batch_size],
            total_kv_len=total_kv_len,
            batch_size=batch_size,
        )

    def _convert_topk_indices_to_global(
        self, topk_indices: torch.Tensor
    ) -> torch.Tensor:
        """CP: topk rows align with total_local_ids; req_id is precomputed_req_ids
        (= mla_params.batch_indice_d[total_global_ids]) so row i maps to the
        request id of the i-th GLOBAL q token. Returns [T, 1, topk]."""
        from rtp_llm.models_py.triton_kernels.sparse_mla.block_index_to_global import (
            triton_convert_req_index_to_global_index,
        )

        assert self.block_table is not None and self.mla_params is not None
        assert self.precomputed_req_ids is not None
        topk_2d = _topk_2d(topk_indices)
        topk = topk_2d.shape[1]
        assert topk == self.top_k
        global_2d = triton_convert_req_index_to_global_index(
            req_id=self.precomputed_req_ids,
            block_table=self.block_table,
            token_indices=topk_2d,
            BLOCK_SIZE=self.token_per_block,
            NUM_TOPK_TOKENS=topk,
            BLOCK_N=min(128, topk),
            HAS_PREFILL_WORKSPACE=False,
        )
        return global_2d.unsqueeze(1)

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        topk: Optional[torch.Tensor],
        batch_indice_d: torch.Tensor,
        kv_cache=None,
        layer_id: int = 0,
    ) -> torch.Tensor:
        """CP prefill: all-gather → restore → write to kv_cache → attend on q tokens
        owned by this rank (q[total_local_ids]). Returns [total_q_len, H, kv_lora_rank]
        with non-owned positions zero (scattered later by total_local_ids)."""
        gathered_ckv = all_gather(compressed_kv.contiguous(), group=Group.TP)
        gathered_ckv = gathered_ckv.reshape(-1, compressed_kv.size(-1))
        gathered_k_pe = all_gather(k_pe.contiguous(), group=Group.TP)
        gathered_k_pe = gathered_k_pe.reshape(-1, k_pe.size(-1))

        restored_ckv = gathered_ckv[self.kv_restore_unpad_indices]
        restored_k_pe = gathered_k_pe[self.kv_restore_unpad_indices]
        self.kv_cache_write_op.forward(
            restored_ckv, restored_k_pe, kv_cache, self.mla_params
        )
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        if topk is None:
            return None
        assert q is not None and q.size(0) > 0

        q0 = q[self.total_local_ids].contiguous()
        if self._gather is not None:
            out0 = self._attend_gather(q0, kv_cache, topk)
        else:
            out0 = self._attend_with_kvcache(q0, kv_cache, topk, layer_id)

        out = triton_kv_scatter(out0, self.total_local_ids, q.size(0))
        return out

    def _attend_with_kvcache(
        self,
        q0: torch.Tensor,
        kv_cache,
        topk: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """flash_mla_with_kvcache on FP8 paged cache (CP equivalent of non-CP baseline)."""
        kv_cache_flat = _as_uint8(
            kv_cache.kv_cache_base.view(-1, 1, kv_cache.kv_cache_base.size(-1))
        )
        if kv_cache_flat.ndim == 3:
            kv_cache_flat = kv_cache_flat.unsqueeze(-2)
        global_topk = self._convert_topk_indices_to_global(topk).squeeze(1).unsqueeze(0)
        meta = self._fp8_kernel_metadata_q0
        attn_out, _ = flash_mla_with_kvcache(
            q=q0.unsqueeze(0),
            k_cache=kv_cache_flat,
            block_table=self.block_table,
            head_dim_v=self.kv_lora_rank,
            cache_seqlens=None,
            tile_scheduler_metadata=meta.tile_scheduler_metadata,
            num_splits=meta.num_splits,
            is_fp8_kvcache=True,
            indices=global_topk,
            softmax_scale=self.scale,
        )
        return attn_out.squeeze(0)

    def _attend_gather(
        self,
        q0: torch.Tensor,
        kv_cache,
        topk: torch.Tensor,
    ) -> torch.Tensor:
        """gather + flash_mla_sparse_fwd. After CP all-gather/restore/write, the paged
        cache has the full per-request KV; the only CP-specific bit is using
        precomputed_req_ids (req id per global q token) for the offset lookup."""
        ws = self._gather
        assert ws is not None and self.precomputed_req_ids is not None
        src = _as_uint8(kv_cache.kv_cache_base)
        if src.ndim == 4:
            src = src.squeeze(2)
        rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache_v2(
            src,
            ws.fused_kv,
            self.block_table.to(torch.int32),
            ws.seq_lens,
            ws.workspace_starts,
            ws.batch_size,
            ws.total_kv_len,
        )
        offsets = ws.workspace_starts[self.precomputed_req_ids]
        global_indices = (_topk_2d(topk) + offsets.unsqueeze(1)).unsqueeze(1)
        out, _, _ = flash_mla_sparse_fwd(
            q0,
            ws.fused_kv.unsqueeze(1),
            global_indices,
            self.scale,
            d_v=self.kv_lora_rank,
        )
        return out


class SparseMlaCpImpl(SparseMlaImpl):
    """Sparse MLA wrapper that selects SparseMlaFp8CPOp and packs CP indices."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        weights: List[Dict[str, torch.Tensor]],
        cos_sin_cache: torch.Tensor,
        fmha_config: Optional[FMHAConfig] = None,
        use_trt_fmha: bool = False,
        quant_config: Optional[object] = None,
        max_seq_len: int = 0,
        is_cuda_graph: bool = False,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        # ContextParallelProcessor leaves per-chunk lengths on shared attn_inputs;
        # sparse fill_params / cache_store need per-request actual lengths. Use a
        # shallow copy here so we don't mutate the caller's attn_inputs.
        attn_inputs_for_init = copy.copy(attn_inputs)
        attn_inputs_for_init.input_lengths = (
            attn_inputs.context_parallel_info.prefill_actual_input_lengths_cpu
        )
        super().__init__(
            attn_configs=attn_configs,
            attn_inputs=attn_inputs_for_init,
            weights=weights,
            cos_sin_cache=cos_sin_cache,
            fmha_config=fmha_config,
            use_trt_fmha=use_trt_fmha,
            quant_config=quant_config,
            max_seq_len=max_seq_len,
            is_cuda_graph=is_cuda_graph,
            parallelism_config=parallelism_config,
            fmha_impl=SparseMlaFp8CPOp,
        )
        self.fmha_impl.kv_cache_write_op = self.kv_cache_write_op
        self.fmha_impl.write_cache_store_impl = self.write_cache_store_impl

    def prepare(
        self, attn_inputs: PyAttentionInputs, forbid_realloc: bool = False
    ) -> None:
        cp_info = attn_inputs.context_parallel_info
        assert cp_info is not None
        attn_for_prepare = copy.copy(attn_inputs)
        attn_for_prepare.input_lengths = cp_info.prefill_actual_input_lengths_cpu
        super().prepare(attn_for_prepare, forbid_realloc=forbid_realloc)

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.CP_SPARSE_FLASHMLA

    @classmethod
    def support_prefill_cp(cls) -> bool:
        return True

    def create_params(self, attn_inputs: PyAttentionInputs):
        """Create fmha_params, run plan() via prepare(), then pack CP indices into
        cp_params for the indexer to consume. plan() already filled
        full_rope_pos_ids and precomputed_req_ids on fmha_impl."""
        self.fmha_params = rtp_llm_ops.SparseMlaParams()
        self.rope_params = self.fmha_params
        self.prepare(attn_inputs)

        gid = self.fmha_impl.total_global_ids
        has_tokens = gid is not None and gid.size(0) > 0

        def _pick(t):
            return t[gid] if has_tokens else None

        self.cp_params = SimpleNamespace(
            kv_restore_unpad_indices=self.fmha_impl.kv_restore_unpad_indices,
            total_global_ids=gid,
            total_local_ids=self.fmha_impl.total_local_ids,
            cu_kv_seqlens_global=self.fmha_impl.cu_kv_seqlens_global,
            total_kv_len=self.fmha_impl.total_kv_len,
            full_rope_pos_ids=self.fmha_impl.full_rope_pos_ids,
            precomputed_ks=_pick(self.fmha_params.ks),
            precomputed_ke=_pick(self.fmha_params.ke),
            precomputed_lengths=_pick(self.fmha_params.expanded_seq_lens),
            precomputed_topk_off=_pick(self.fmha_params.topk_indices_offset),
            precomputed_req_ids=self.fmha_impl.precomputed_req_ids,
        )

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
        topk_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """CP sparse MLA forward. q: [total_q_len, H, qk_head_dim]; topk_indices
        is request-local. Returns [total_q_len, H, nope_head_dim]."""
        assert kv_cache is not None

        # RoPE in-place on full q_pe / k_pe via full_rope_pos_ids. Padding rows
        # get pos=0 but never read: q is selected by total_local_ids; k_pe is
        # all-gathered then re-indexed by kv_restore_unpad_indices.
        q_pe = q[:, :, self.nope_head_dim :]
        if self.fmha_impl.full_rope_pos_ids is not None:
            self.rope_impl.forward(
                q_pe,
                k_pe,
                self.rope_params,
                precomputed_pos_ids=self.fmha_impl.full_rope_pos_ids,
            )

        q_transformed = self._apply_input_bmm(q, layer_id)
        attn_output = self.fmha_impl.forward(
            q_transformed,
            compressed_kv,
            k_pe,
            topk_indices,
            self.fmha_params.batch_indice_d,
            kv_cache,
            layer_id=layer_id,
        )
        if attn_output is None:
            return None
        return self._apply_output_bmm(attn_output, layer_id)

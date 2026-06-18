"""MiniMax-M3 sparse attention (MSA) module.

Wires the ported Triton MSA kernels (``rtp_llm/models_py/triton_kernels/
sparse_msa``) into rtp-llm's GenericMoe decoder for MiniMax-M3 *sparse* layers
(e.g. layers 3,4 in the 5-layer mini model). Dense layers keep using the
shared FlashInfer FMHA impl; only sparse layers are routed here.

Design (paged-only store):

* The persistent store for both the main K/V and the index-K is the standard
  cache-manager paged pool — there is NO self-built per-layer side cache.
  Main K/V live in ``kv_cache.kv_cache_base`` (HND paged pool) and idx_K lives
  in that pool's scale region ``kv_cache.kv_scale_base`` (reinterpreted as
  BF16). Both are addressed by the same block table and therefore travel
  together under PD separation.

* The MSA Triton kernels still consume flat *token-slot* tensors
  ``[max_slots, num_kv_heads, head_dim]`` addressed by a
  ``req_to_token [max_reqs, max_kv_len]`` map plus ``slot_ids [batch]``. Since
  that layout differs from the paged pool, each forward gathers the active
  sequence out of the paged pool into a process-wide *transient* scratch
  (``_MainKVScratch`` / ``_IdxKScratch``) that the kernel reads. The scratch is
  grown-on-demand and shared across all sparse layers (sparse layers run
  sequentially), so it is O(1) buffers, not O(num_sparse_layers) caches; it
  holds no state across steps.

* In the normal non-CP path the physical slot for ``(request b, token
  position p)`` is the paged block table::

      slot = block_table[b, p // page_size] * page_size + (p % page_size)

* In CP prefill, K/V are all-gathered into full sequence order while Q stays
  rank-local, then written into this rank's paged shard; the gather scratch is
  indexed by a compact ``b*seq_len + pos`` grid for the kernel.

The index branch (``index_q_proj`` / ``index_k_proj`` + per-head Gemma RMSNorm
+ partial RoPE) only selects top-k blocks; with ``disable_index_value=True``
(M3 default) it does not contribute to the attention value, so ``idx_v`` is
``None`` and the index output ``idx_o`` is discarded.
"""

import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

# Gated decode-path instrumentation (M3_MSA_DEBUG=1). Logs per-layer sparse
# decode block counts / lens / wall-clock so a block-count explosion vs pure
# eager-mode latency can be told apart without guessing.
_MSA_DEBUG = os.environ.get("M3_MSA_DEBUG", "0") == "1"
_MSA_DEBUG_MAX_STEPS = int(os.environ.get("M3_MSA_DEBUG_MAX_STEPS", "3"))
_MSA_DEBUG_STEP = {}
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather, all_reduce
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops import AttentionConfigs, HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs
from rtp_llm.utils.model_weight import W

device_type = get_device_type()
if device_type == DeviceType.ROCm:
    from rtp_llm.models_py.modules.base.rocm.norm import FusedQKRMSNorm
else:
    from rtp_llm.models_py.modules.base.cuda.norm import FusedQKRMSNorm


def _gemma_rmsnorm_per_head(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Per-head RMSNorm over the last dim using the loaded gamma.

    MiniMax-M3 weight loading already bakes Gemma's ``+1`` offset into norm
    weights, matching the dense Q/K norm path — so this is plain RMSNorm and
    we route it through flashinfer's fused kernel instead of a Python op
    chain (cast/pow/mean/rsqrt/mul/cast). Last-dim reduction means the (T,H,D)
    input can be reshaped to (T*H, D) where each row is normalized
    independently against the shared D-dim weight.
    """
    import flashinfer.norm

    orig_shape = x.shape
    return flashinfer.norm.rmsnorm(
        x.reshape(-1, orig_shape[-1]).contiguous(), weight, eps=eps
    ).view(orig_shape)


class _MainKVScratch:
    """Process-wide shared, transient gather scratch for MSA main K/V.

    The persistent store is the standard cache-manager paged pool; the MSA
    Triton kernels still need the active K/V in flat
    token-slot layout, so each forward we gather the full active sequence out
    of the paged pool into this scratch. Sparse layers run strictly
    sequentially within one model forward (layer i finishes before layer i+1),
    so a single buffer grown on demand serves all sparse layers — the scratch
    footprint is 1x, not num_sparse_layers x.
    """

    def __init__(self) -> None:
        self._k: Optional[torch.Tensor] = None
        self._v: Optional[torch.Tensor] = None

    def acquire(
        self,
        slots: int,
        heads: int,
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        if (
            self._k is None
            or self._k.shape[0] < slots
            or self._k.shape[1] != heads
            or self._k.shape[2] != dim
            or self._k.dtype != dtype
            or self._k.device != device
        ):
            self._k = torch.zeros(slots, heads, dim, dtype=dtype, device=device)
            self._v = torch.zeros_like(self._k)
        return self._k[:slots], self._v[:slots]


_MAIN_KV_SCRATCH = _MainKVScratch()


class _IdxKScratch:
    """Process-wide shared, transient gather scratch for MSA idx_K.

    Counterpart to ``_MainKVScratch`` for the index branch: the persistent
    store is the main paged pool's scale region (PD-transferable); the MSA
    Triton kernels still want idx_K in flat ``[slot, 1, idx_head_dim]`` layout,
    so each forward we gather the active sequence out of the scale region into
    this single buffer (one grown-on-demand buffer for all sparse layers).
    """

    def __init__(self) -> None:
        self._t: Optional[torch.Tensor] = None

    def acquire(
        self,
        slots: int,
        heads: int,
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        if (
            self._t is None
            or self._t.shape[0] < slots
            or self._t.shape[1] != heads
            or self._t.shape[2] != dim
            or self._t.dtype != dtype
            or self._t.device != device
        ):
            self._t = torch.zeros(slots, heads, dim, dtype=dtype, device=device)
        return self._t[:slots]


_IDX_K_SCRATCH = _IdxKScratch()


class MSAAttention(nn.Module):
    """MiniMax-M3 sparse attention for a single sparse layer."""

    # Class-level workspace shared across all sparse layers (trtllm-gen needs
    # a 256 MB scratch buffer; one per device is enough). Lazily allocated on
    # the first prefill that takes the trtllm-gen fast path.
    _trtllm_workspace: Dict[torch.device, torch.Tensor] = {}

    @classmethod
    def _get_trtllm_workspace(cls, device: torch.device) -> torch.Tensor:
        ws = cls._trtllm_workspace.get(device)
        if ws is None:
            ws = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=device)
            cls._trtllm_workspace[device] = ws
        return ws

    def __init__(
        self,
        attn_config: AttentionConfigs,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layernorm_eps: float,
        sparse_config: Dict[str, Any],
        layer_idx: int,
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.parallelism_config = parallelism_config
        self.tp_size = parallelism_config.get_attn_tp_size()
        self.tp_rank = parallelism_config.get_attn_tp_rank()
        self.layernorm_eps = layernorm_eps

        # CP (context parallelism) uses the raw TP dimension for sequence
        # splitting. get_attn_tp_size() returns 1 when CP is active so weights
        # are NOT sharded, but tp_size/tp_rank still identify the CP group.
        cp_cfg = parallelism_config.prefill_cp_config
        self.cp_enabled = cp_cfg.method.value != 0  # NONE = 0
        # CP page-RR KV sharding geometry. Mirrors C++ DeviceData::props:
        #   sharded = prefill_cp enabled AND kv_cache_sharded AND raw tp_size>1.
        # The CP group is the raw TP dimension (get_attn_tp_size()==1 under CP).
        # When not sharded, cp_size=1 makes the slot mapping a plain global-slot
        # passthrough (bit-equal to the pre-sharding global-slot behaviour).
        raw_tp_size = int(parallelism_config.tp_size)
        raw_tp_rank = int(parallelism_config.tp_rank)
        self._kv_sharded = bool(
            self.cp_enabled
            and getattr(cp_cfg, "kv_cache_sharded", False)
            and raw_tp_size > 1
        )
        self._cp_size = raw_tp_size if self._kv_sharded else 1
        self._cp_rank = raw_tp_rank if self._kv_sharded else 0
        self.head_num = attn_config.head_num
        self.kv_head_num = attn_config.kv_head_num
        self.head_dim = attn_config.size_per_head
        self.q_size = self.head_num * self.head_dim
        self.kv_size = self.kv_head_num * self.head_dim
        self.page_size = attn_config.kernel_tokens_per_block

        # --- main GQA branch (identical construction to CausalAttention) ---
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

        self.qk_fuse_norm = None
        if W.q_ln_gamma in weights and W.k_ln_gamma in weights:
            self.qk_fuse_norm = FusedQKRMSNorm(
                weights[W.q_ln_gamma],
                weights[W.k_ln_gamma],
                self.head_num,
                self.kv_head_num,
                self.head_dim,
                layernorm_eps,
            )

        # --- index branch (BF16; dequantized from MXFP8 at load) ---
        self.idx_head_dim = int(sparse_config["idx_head_dim"])
        full_idx_q_w = weights[W.msa_idx_q_w]  # [num_idx_heads*idx_dim, hidden]
        self.idx_k_w = weights[W.msa_idx_k_w]  # [idx_dim, hidden]
        self.idx_q_norm_w = weights[W.msa_idx_q_norm]  # [idx_dim]
        self.idx_k_norm_w = weights[W.msa_idx_k_norm]  # [idx_dim]
        self.total_idx_heads = int(
            sparse_config.get(
                "num_idx_heads", full_idx_q_w.shape[0] // self.idx_head_dim
            )
        )
        self.num_idx_heads = self._local_idx_heads()
        loaded_idx_heads = full_idx_q_w.shape[0] // self.idx_head_dim
        if loaded_idx_heads == self.total_idx_heads:
            start_head = self.idx_head_rank * self.num_idx_heads
            start = start_head * self.idx_head_dim
            end = start + self.num_idx_heads * self.idx_head_dim
            self.idx_q_w = full_idx_q_w[start:end].contiguous()
        elif loaded_idx_heads == self.num_idx_heads:
            self.idx_q_w = full_idx_q_w.contiguous()
        else:
            raise RuntimeError(
                "unexpected MSA index_q weight shape: "
                f"loaded_idx_heads={loaded_idx_heads}, "
                f"total_idx_heads={self.total_idx_heads}, "
                f"local_idx_heads={self.num_idx_heads}"
            )

        # --- sparse params ---
        self.topk_blocks = int(sparse_config["topk_blocks"])
        self.block_size = int(sparse_config["block_size"])
        self.init_blocks = int(sparse_config["init_blocks"])
        self.local_blocks = int(sparse_config["local_blocks"])
        self.score_type = str(sparse_config.get("score_type", "max"))
        self.disable_index_value = layer_idx in set(
            sparse_config.get("disable_value_layer_ids", [])
        )

        # --- partial RoPE cos/sin cache.  Match the dense C++ fused RoPE
        # path for M3: rope_style=1 uses the non-interleaved LLaMA layout.
        from rtp_llm.ops import get_rope_cache_once

        self._rope_theta = attn_config.rope_config.base
        self._rope_interleave = False
        try:
            rope_cache = get_rope_cache_once(
                attn_config.rope_config,
                attn_config.max_seq_len + attn_config.gen_num_per_cycle + 1,
                is_cuda=True,
                interleave=self._rope_interleave,
            )
            self.cos_sin_cache = rope_cache.data
        except Exception:
            self.cos_sin_cache = None

        # Paged-only store: the main K/V live in the standard cache-manager
        # paged pool (kv_cache_base) and idx_K in its scale region
        # (kv_scale_base, reinterpreted as BF16). Both are PD-transferable and
        # addressed by the same block table. The self-built persistent side
        # cache was removed; only a process-wide gather scratch is kept for the
        # MSA kernel.
        self._scratch_batch_size = 0
        self._scratch_seq_len = 0

        # Views into the process-wide shared gather scratch, refreshed each
        # forward by _source_main_kv_from_paged / _source_idx_k_from_paged.
        self._scratch_k: Optional[torch.Tensor] = None
        self._scratch_v: Optional[torch.Tensor] = None
        self._scratch_idx_k: Optional[torch.Tensor] = None
        # Allocated kernel slot span (anchors scratch sizing).
        self._scratch_slots = 0

    def _local_idx_heads(self) -> int:
        """Match SGLang's GQA-style sharding for sparse index-Q heads."""
        if self.total_idx_heads >= self.tp_size:
            if self.total_idx_heads % self.tp_size != 0:
                raise RuntimeError(
                    "MSA index heads must be divisible by TP size: "
                    f"idx_heads={self.total_idx_heads}, tp_size={self.tp_size}"
                )
            self.idx_head_tp_size = self.tp_size
            self.idx_replica_size = 1
        else:
            if self.tp_size % self.total_idx_heads != 0:
                raise RuntimeError(
                    "TP size must be divisible by MSA index heads when "
                    f"tp_size > idx_heads: tp_size={self.tp_size}, "
                    f"idx_heads={self.total_idx_heads}"
                )
            self.idx_head_tp_size = self.total_idx_heads
            self.idx_replica_size = self.tp_size // self.idx_head_tp_size
        self.idx_head_rank = self.tp_rank // self.idx_replica_size
        return self.total_idx_heads // self.idx_head_tp_size

    def _apply_rope(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> None:
        """In-place partial RoPE on q/k ([T, H, head_dim])."""
        import flashinfer.rope as fi_rope

        if self.cos_sin_cache is not None:
            fi_rope._apply_rope_pos_ids_cos_sin_cache(
                q=q,
                k=k,
                q_rope=q,
                k_rope=k,
                cos_sin_cache=self.cos_sin_cache,
                pos_ids=positions,
                interleave=self._rope_interleave,
            )
        else:
            import flashinfer

            flashinfer.apply_rope_pos_ids_inplace(
                q, k, positions, rope_theta=self._rope_theta
            )

    def _ensure_gather_scratch(
        self,
        kv_cache: LayerKVCache,
        device: torch.device,
        dtype: torch.dtype,
        bsz: Optional[int] = None,
        max_kv: Optional[int] = None,
        max_slot: Optional[int] = None,
    ) -> None:
        # Paged-only: main K/V and idx_K live in the cache-manager pool, so no
        # persistent side tensors are allocated here. This only tracks the
        # kernel slot span (and, under CP, the [bsz, seq_len] addressing grid)
        # that sizes the process-wide gather scratch acquired per forward.
        if self.cp_enabled:
            if bsz is None or max_kv is None:
                raise RuntimeError("CP MSA gather scratch requires batch size and kv length")
            if (
                self._scratch_slots > 0
                and self._scratch_batch_size >= int(bsz)
                and self._scratch_seq_len >= int(max_kv)
            ):
                return
            target_bsz = max(int(bsz), self._scratch_batch_size, 1)
            requested_seq_len = max(int(max_kv), self._scratch_seq_len, 1)
            grow_granularity = max(int(self.page_size), 256)
            target_seq_len = (
                (requested_seq_len + grow_granularity - 1)
                // grow_granularity
                * grow_granularity
            )
            self._scratch_batch_size = target_bsz
            self._scratch_seq_len = target_seq_len
            self._scratch_slots = target_bsz * target_seq_len
            return

        if max_slot is None:
            raise RuntimeError("non-CP MSA gather scratch requires max active slot")
        requested_slots = max(int(max_slot) + 1, 1)
        if self._scratch_slots >= requested_slots:
            return
        grow_granularity = max(int(self.page_size), 256)
        self._scratch_slots = (
            (requested_slots + grow_granularity - 1) // grow_granularity
        ) * grow_granularity

    def _get_lengths(self, attn_inputs: PyAttentionInputs):
        if attn_inputs.is_prefill:
            prefix = attn_inputs.prefix_lengths.to(torch.int64)
            inlen = attn_inputs.input_lengths.to(torch.int64)
            kv_lens = prefix + inlen
        else:
            seqlen = attn_inputs.sequence_lengths.to(torch.int64)
            kv_lens = seqlen + 1
            prefix = kv_lens - 1
            inlen = torch.ones_like(kv_lens)
        return kv_lens, prefix, inlen

    def _build_compact_addressing(
        self, attn_inputs: PyAttentionInputs, device: torch.device
    ):
        """CP path addressing over the compact per-request gather scratch."""
        if self._scratch_seq_len <= 0:
            raise RuntimeError("compact MSA gather scratch is not initialized")
        kv_lens, prefix, inlen = self._get_lengths(attn_inputs)
        bsz = int(kv_lens.numel())
        max_kv = int(kv_lens.max().item())
        pos = torch.arange(max_kv, device=device, dtype=torch.int32)
        row_offsets = (
            torch.arange(bsz, device=device, dtype=torch.int32)[:, None]
            * int(self._scratch_seq_len)
        )
        req_to_token = row_offsets + pos[None, :]
        slot_ids = torch.arange(bsz, device=device, dtype=torch.int64)

        prefix_cpu = prefix.detach().cpu().tolist()
        kv_cpu = kv_lens.detach().cpu().tolist()
        pos_parts = []
        slot_parts = []
        for b in range(bsz):
            p0, p1 = int(prefix_cpu[b]), int(kv_cpu[b])
            pos_parts.append(torch.arange(p0, p1, device=device, dtype=torch.int64))
            slot_parts.append(req_to_token[b, p0:p1])
        positions = torch.cat(pos_parts).to(torch.int32)
        write_slots = torch.cat(slot_parts).to(torch.int64)
        return req_to_token, slot_ids, kv_lens, positions, write_slots, prefix, inlen

    def _build_addressing(
        self, attn_inputs: PyAttentionInputs, device: torch.device
    ):
        """Return (req_to_token [B, max_kv], slot_ids [B], kv_lens [B],
        positions [T], write_slots [T]) from rtp-llm block table + lengths."""
        block_table = attn_inputs.kv_cache_kernel_block_id_device  # [B, max_blocks]
        bsz = block_table.size(0)
        max_blocks = block_table.size(1)
        kv_lens, prefix, inlen = self._get_lengths(attn_inputs)

        max_kv = int(kv_lens.max().item())
        pos = torch.arange(max_kv, device=device, dtype=torch.int64)
        blk_idx = (pos // self.page_size).clamp(max=max_blocks - 1)
        blk_off = pos % self.page_size
        bt = block_table.index_select(1, blk_idx).to(torch.int64)  # [B, max_kv]
        req_to_token = (bt * self.page_size + blk_off[None, :]).to(torch.int32)
        slot_ids = torch.arange(bsz, device=device, dtype=torch.int64)

        # token order: per-request concat of new tokens [prefix[b], kv_len[b])
        prefix_cpu = prefix.tolist()
        kv_cpu = kv_lens.tolist()
        pos_parts = []
        slot_parts = []
        for b in range(bsz):
            p0, p1 = prefix_cpu[b], kv_cpu[b]
            pos_parts.append(torch.arange(p0, p1, device=device, dtype=torch.int64))
            slot_parts.append(req_to_token[b, p0:p1])
        positions = torch.cat(pos_parts).to(torch.int32)
        write_slots = torch.cat(slot_parts).to(torch.int64)
        return req_to_token, slot_ids, kv_lens, positions, write_slots, prefix, inlen

    @staticmethod
    def _max_active_slot(req_to_token: torch.Tensor, kv_lens: torch.Tensor) -> int:
        """Return the largest physical slot read by the sparse kernels."""
        max_slot = 0
        kv_lens_cpu = kv_lens.detach().cpu().to(torch.int64).tolist()
        for b, kv_len in enumerate(kv_lens_cpu):
            kv_len = int(kv_len)
            if kv_len <= 0:
                continue
            row_max = int(req_to_token[b, :kv_len].max().item())
            max_slot = max(max_slot, row_max)
        return max_slot

    # ------------------------------------------------------------------
    # Source main K/V from the standard cache-manager paged pool.
    # The paged pool (kv_cache.kv_cache_base) is the persistent, PD-transferable
    # store; the per-step gather scratch (_scratch_k / _scratch_v) is filled
    # from it and read by the MSA kernel (req_to_token unchanged).
    # ------------------------------------------------------------------
    def _paged_main_views(self, kv_cache: LayerKVCache):
        """Token-major [block, page, head, dim] views of the standard HND paged
        pool [block, 2, head, page, head_dim] for K and V (non-contiguous views;
        fine for advanced-index read/write)."""
        base = kv_cache.kv_cache_base
        if base is None or base.dim() != 5:
            raise RuntimeError(
                "MSA paged main K/V requires a 5-D paged cache "
                "[block,2,head,page,dim], got "
                f"{None if base is None else tuple(base.shape)}"
            )
        kpv = base[:, 0].permute(0, 2, 1, 3)
        vpv = base[:, 1].permute(0, 2, 1, 3)
        return kpv, vpv

    def _physical_block_table(
        self, attn_inputs: PyAttentionInputs
    ) -> torch.Tensor:
        """Physical paged-cache block table (per-rank, CP-RR compact under
        sharding). Mirrors the GLM5/DSV4 indexer: cache I/O must address the
        physical pages, not the (possibly token-level) kernel block table."""
        phys = getattr(attn_inputs, "kv_cache_block_id_device", None)
        if isinstance(phys, torch.Tensor) and phys.numel() > 0:
            return phys
        return attn_inputs.kv_cache_kernel_block_id_device

    def _kernel_slots_to_paged(
        self, kernel_slots: torch.Tensor, attn_inputs: PyAttentionInputs
    ) -> torch.Tensor:
        """Map kernel-space slots to physical paged-pool slots.

        Three regimes (all addressed through the *physical* block table, the
        same table GLM5/DSV4 use for paged cache I/O):

        * non-CP: kernel slots are already global ``block*page+off`` → identity.
        * CP, full-replicated pool (``_cp_size == 1``): compact kernel slots
          ``b*scratch_seq_len + pos`` → resolve ``(b, pos)`` through the block
          table to the plain global slot.
        * CP page-RR sharded (``_cp_size > 1``): reuse GLM5/DSV4's
          ``cp_kv_slot_mapping`` (ratio=1, uncompressed MHA K/V). Non-owned
          tokens (and block-0 sentinels) become ``-1`` so the writer skips them.
        """
        ks = kernel_slots.to(torch.int64)
        if not self.cp_enabled:
            return ks
        seq_len = int(self._scratch_seq_len)
        b_idx = ks // seq_len
        positions = ks % seq_len
        bt = self._physical_block_table(attn_inputs).to(torch.int64)
        if not self._kv_sharded:
            blk = positions // self.page_size
            return bt[b_idx, blk] * self.page_size + (positions % self.page_size)
        from rtp_llm.models_py.modules.dsv4.fp8._cp_slot_mapping import (
            cp_kv_slot_mapping,
        )

        return cp_kv_slot_mapping(
            positions,
            bt,
            b_idx,
            self.page_size,  # tokens_per_block
            self.page_size,  # kv_eb (entries per block, ratio=1)
            1,               # ratio (uncompressed)
            self._cp_size,
            self._cp_rank,
            owner_tokens_per_block=self.page_size,
        )

    def _source_main_kv_from_paged(
        self,
        kv_cache: LayerKVCache,
        k: torch.Tensor,
        v: torch.Tensor,
        write_slots: torch.Tensor,
        req_to_token: torch.Tensor,
        kv_lens: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        device: torch.device,
    ) -> None:
        """Persist K/V into the standard paged pool (PD-store) and build the
        transient gather scratch the MSA kernel reads.

        The paged write goes through the sharding-aware C++ writer
        ``mha_kv_write_cache`` with a physical-block-table slot mapping, so it is
        correct under CP page-RR sharding (each rank stores only its 1/cp_size of
        the tokens; non-owned tokens get a -1 slot and are skipped).

        Scratch sourcing depends on sharding:
        * sharded (CP prefill): the local pool holds only 1/cp_size of the
          tokens, so the scratch is filled directly from the already-all-gathered
          full sequence (``k``/``v`` are the full sequence in CP prefill).
        * not sharded (non-CP prefill, or decode): the full active history is
          read back from the persistent (full) paged pool, exactly as before —
          required for decode where the current step only carries one token."""
        base = kv_cache.kv_cache_base
        if base is None or base.dim() != 5:
            raise RuntimeError(
                "MSA paged main K/V requires a 5-D paged cache "
                "[block,2,head,page,dim], got "
                f"{None if base is None else tuple(base.shape)}"
            )
        if base.dtype != k.dtype:
            raise RuntimeError(
                f"MSA paged main K/V dtype mismatch: paged={base.dtype} vs "
                f"act={k.dtype}; launch with FP8_KV_CACHE=0 for a bf16 paged pool"
            )

        # 1) persist into the paged pool via the sharding-aware C++ writer.
        slot_mapping = self._kernel_slots_to_paged(write_slots, attn_inputs)
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        rtp_llm_ops.mha_kv_write_cache(
            k.contiguous(), v.contiguous(), base, slot_mapping
        )

        # 2) build the transient gather scratch the MSA kernel reads.
        scratch_slots = int(self._scratch_slots)
        scratch_k, scratch_v = _MAIN_KV_SCRATCH.acquire(
            scratch_slots, self.kv_head_num, self.head_dim, k.dtype, device
        )
        if self._kv_sharded:
            scratch_k[write_slots] = k
            scratch_v[write_slots] = v
        else:
            kpv, vpv = self._paged_main_views(kv_cache)
            p = self.page_size
            max_kv = req_to_token.shape[1]
            ar = torch.arange(max_kv, device=device, dtype=torch.int64)
            mask = ar[None, :] < kv_lens.to(device=device, dtype=torch.int64)[:, None]
            dst_full = req_to_token.to(torch.int64)[mask]
            gf = self._kernel_slots_to_paged(dst_full, attn_inputs)
            scratch_k[dst_full] = kpv[gf // p, gf % p]
            scratch_v[dst_full] = vpv[gf // p, gf % p]
        self._scratch_k = scratch_k
        self._scratch_v = scratch_v

    # ------------------------------------------------------------------
    # Task-2: source idx_K from the main paged pool's scale region.
    # The C++ cache manager sizes the MHA scale region (kv_scale_base) to hold
    # one BF16 idx_K per token (indexer_head_dim). It is exposed to Python as
    # FP32; we reinterpret it as BF16 and view it as [block, page, idx_head_dim]
    # so idx_K is addressed by the same block table as the main K/V and travels
    # with it under PD separation.
    # ------------------------------------------------------------------
    def _idx_k_paged_view(self, kv_cache: LayerKVCache) -> torch.Tensor:
        """[block, page, idx_head_dim] BF16 view of the FP32 scale region."""
        scale = kv_cache.kv_scale_base
        if scale is None or scale.dim() != 2:
            raise RuntimeError(
                "MSA paged idx_K requires a 2-D kv_scale_base "
                "[block, scale_elems]; got "
                f"{None if scale is None else tuple(scale.shape)}. Launch with "
                "M3_IDX_PAGED=1 and a th_transformer built with the M3 MHA "
                "indexer scale sizing (indexer_head_dim set)."
            )
        blk = int(scale.shape[0])
        # FP32 storage reinterpreted as BF16: scale_elems fp32 -> 2*scale_elems
        # bf16 == page_size * idx_head_dim.
        sb = scale.view(torch.bfloat16)
        expect = self.page_size * self.idx_head_dim
        if int(sb.shape[1]) != expect:
            raise RuntimeError(
                f"MSA idx_K scale region mismatch: bf16 elems/block={int(sb.shape[1])} "
                f"!= page_size*idx_head_dim={expect} (page={self.page_size}, "
                f"idx_head_dim={self.idx_head_dim}); check C++ kv_scale_stride_bytes"
            )
        return sb.view(blk, self.page_size, self.idx_head_dim)

    def _source_idx_k_from_paged(
        self,
        kv_cache: LayerKVCache,
        idx_k: torch.Tensor,
        write_slots: torch.Tensor,
        req_to_token: torch.Tensor,
        kv_lens: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        device: torch.device,
    ) -> None:
        """Persist idx_K into the paged scale region (PD-store) and build the
        shared idx scratch the MSA kernel reads.

        Uses the same sharding-aware physical slot mapping as the main K/V: the
        scale region is a ``[block, page, idx_head_dim]`` view, so a token's
        flat row is exactly its physical slot. Non-owned tokens (slot == -1)
        are skipped. The kernel scratch is filled directly from the
        all-gathered ``idx_k`` (no paged read-back)."""
        idx_view = self._idx_k_paged_view(kv_cache)  # [block, page, idx_head_dim]
        if idx_view.dtype != idx_k.dtype:
            raise RuntimeError(
                f"MSA paged idx_K dtype mismatch: paged={idx_view.dtype} vs "
                f"act={idx_k.dtype} (scale region is reinterpreted as bf16)"
            )
        p = self.page_size
        idx_flat = idx_k.reshape(-1, self.idx_head_dim)
        scale_flat = idx_view.reshape(-1, self.idx_head_dim)  # [block*page, idx_dim]

        # 1) persist into the scale region at physical slots (skip -1 non-owned).
        #    The scale region is [block, page, idx_dim]; a token's flat row is
        #    exactly its physical slot (block*page + off), same mapping as K/V.
        slot_mapping = self._kernel_slots_to_paged(write_slots, attn_inputs)
        valid = slot_mapping >= 0
        scale_flat[slot_mapping[valid]] = idx_flat[valid]

        # 2) build the transient scratch the MSA kernel reads.
        scratch_slots = int(self._scratch_slots)
        idx_scratch = _IDX_K_SCRATCH.acquire(
            scratch_slots, 1, self.idx_head_dim, idx_k.dtype, device
        )
        if self._kv_sharded:
            idx_scratch[write_slots, 0] = idx_flat
        else:
            max_kv = req_to_token.shape[1]
            ar = torch.arange(max_kv, device=device, dtype=torch.int64)
            mask = ar[None, :] < kv_lens.to(device=device, dtype=torch.int64)[:, None]
            dst_full = req_to_token.to(torch.int64)[mask]
            gf = self._kernel_slots_to_paged(dst_full, attn_inputs)
            idx_scratch[dst_full, 0] = scale_flat[gf]
        self._scratch_idx_k = idx_scratch

    # ------------------------------------------------------------------
    def _forward_cp_prefill(
        self,
        hidden_states: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: LayerKVCache,
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """CP-aware prefill: local Q attends to all-gathered full-sequence KV.

        When ``x_fp8`` / ``x_scale`` are supplied (the upstream fused
        norm+quant path in GenericMoeDecoderLayer), feed them straight into
        ``qkv_proj`` to skip the per-token-group quant that the projection
        would otherwise run on its bf16 input. ``hidden_states`` still drives
        the index-branch F.linear paths (which are bf16 GEMMs).
        """
        from rtp_llm.models_py.triton_kernels.sparse_msa.minimax_sparse import (
            minimax_sparse_prefill,
        )

        cp_info = attn_inputs.context_parallel_info
        device = hidden_states.device
        local_tokens = hidden_states.shape[0]

        if x_fp8 is not None and x_scale is not None:
            qkv = self.qkv_proj(x_fp8, input_scales=x_scale)
        else:
            qkv = self.qkv_proj(hidden_states)
        if self.qk_fuse_norm is not None:
            qkv = self.qk_fuse_norm(qkv)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.reshape(local_tokens, self.head_num, self.head_dim)
        k = k.reshape(local_tokens, self.kv_head_num, self.head_dim)
        v = v.reshape(local_tokens, self.kv_head_num, self.head_dim)

        idx_q = F.linear(hidden_states, self.idx_q_w)
        idx_k = F.linear(hidden_states, self.idx_k_w)
        idx_q = idx_q.reshape(local_tokens, self.num_idx_heads, self.idx_head_dim)
        idx_k = idx_k.reshape(local_tokens, 1, self.idx_head_dim)
        idx_q = _gemma_rmsnorm_per_head(idx_q, self.idx_q_norm_w, self.layernorm_eps)
        idx_k = _gemma_rmsnorm_per_head(idx_k, self.idx_k_norm_w, self.layernorm_eps)

        # Coalesce the two small per-layer D2H transfers (chunk_lengths +
        # prefix_lengths) into one packed copy so we pay a single
        # cudaStreamSynchronize instead of two. The tiny H2D for chunk_lengths
        # when it originates on CPU is negligible (<1us pinned->device).
        _chunk_dev = cp_info.prefill_cp_chunk_lengths.detach().to(
            device=device, dtype=torch.int64
        )
        _prefix_dev = attn_inputs.prefix_lengths.detach().to(
            device=device, dtype=torch.int64
        )
        _packed_cpu = torch.cat([_chunk_dev, _prefix_dev]).cpu()
        _n_chunks = _chunk_dev.numel()
        chunk_lengths_cpu = _packed_cpu[:_n_chunks].tolist()
        prefix_cpu = _packed_cpu[_n_chunks:]
        prefix_cpu_list = prefix_cpu.tolist()
        if sum(int(x) for x in chunk_lengths_cpu) != local_tokens:
            raise RuntimeError(
                "MSA CP prefill expects rank-local token count to match "
                "prefill_cp_chunk_lengths; got "
                f"local_tokens={local_tokens}, chunks={chunk_lengths_cpu}"
            )

        # Bring shuffle_indices back to CPU once; we compute local_positions
        # entirely on CPU with numpy (cheap, all inputs are small ints) and
        # do a single H2D, eliminating the per-batch GPU op chain
        # (clamp + add + cat + cast = 2*bsz + 2 launches → 1 H2D).
        shuffle_cpu = cp_info.prefill_shuffle_indices.detach().cpu().to(torch.int64).tolist()

        # Vectorized CPU build of local_positions.
        # positions[i] = max(shuffle[i], 0) + prefix[batch_id[i]]
        # where batch_id[i] = which chunk the i-th token belongs to.
        chunk_arr = np.asarray(chunk_lengths_cpu, dtype=np.int64)
        prefix_arr = np.asarray(prefix_cpu_list, dtype=np.int64)
        shuffle_arr = np.asarray(shuffle_cpu, dtype=np.int64)
        bsz_py = chunk_arr.shape[0]
        batch_id = np.repeat(np.arange(bsz_py, dtype=np.int64), chunk_arr)
        positions_np = (np.maximum(shuffle_arr, 0) + prefix_arr[batch_id]).astype(
            np.int32, copy=False
        )
        # Force a fresh contiguous device tensor (single async H2D).
        local_positions = torch.from_numpy(np.ascontiguousarray(positions_np)).to(
            device=device, non_blocking=True
        )

        # Segment metadata: pure-Python list construction (no GPU op here).
        segment_lengths = []
        segment_starts = []
        segment_req_ids = []
        cursor = 0
        for b in range(bsz_py):
            chunk_len = int(chunk_arr[b])
            pair_len = chunk_len // 2
            req_prefix = int(prefix_arr[b])
            for rel_start in (0, pair_len):
                segment_lengths.append(pair_len)
                segment_req_ids.append(b)
                if pair_len > 0:
                    start_pos = int(shuffle_cpu[cursor + rel_start])
                    segment_starts.append(req_prefix + max(start_pos, 0))
                else:
                    segment_starts.append(req_prefix)
            cursor += chunk_len

        q = q.contiguous()
        k = k.contiguous()
        self._apply_rope(q, k, local_positions)
        idx_q = idx_q.contiguous()
        idx_k = idx_k.contiguous()
        self._apply_rope(idx_q, idx_k, local_positions)

        # Pack K, V, idx_k along the last dim and issue ONE all_gather
        # instead of three. NCCL launch + small-message latency dominates
        # over per-byte bandwidth here, so 3→1 saves ~75–130us / layer.
        # k / idx_k were already made contiguous above by the RoPE block;
        # v is fresh out of torch.split and needs realising once.
        v = v.contiguous()
        nk = self.kv_head_num * self.head_dim
        ni = self.idx_head_dim
        packed_kv = torch.cat(
            [
                k.reshape(local_tokens, nk),
                v.reshape(local_tokens, nk),
                idx_k.reshape(local_tokens, ni),
            ],
            dim=-1,
        )  # [local_tokens, 2*nk + ni], contiguous
        all_packed = all_gather(packed_kv, group=Group.TP)
        gathered_T = all_packed.shape[0]
        # last-dim slices are strided views; .reshape() auto-contiguous'es
        # so downstream fancy indexing (all_k[unpad_indices]) and the paged
        # scatter writes hit the fast path.
        all_k = all_packed[:, :nk].reshape(
            gathered_T, self.kv_head_num, self.head_dim
        )
        all_v = all_packed[:, nk : 2 * nk].reshape(
            gathered_T, self.kv_head_num, self.head_dim
        )
        all_idx_k = all_packed[:, 2 * nk :].reshape(
            gathered_T, 1, self.idx_head_dim
        )

        restore_indices = cp_info.prefill_qkv_restore_indice
        padding_mask = cp_info.prefill_qkv_padding_mask
        unpad_indices = restore_indices[padding_mask == 1].to(torch.long)
        full_k = all_k[unpad_indices]
        full_v = all_v[unpad_indices]
        full_idx_k = all_idx_k[unpad_indices]

        full_input_lengths_cpu = cp_info.prefill_actual_input_lengths_cpu.to(
            torch.int64
        )
        kv_lens_cpu = prefix_cpu + full_input_lengths_cpu

        bsz = int(kv_lens_cpu.numel())
        max_kv = int(kv_lens_cpu.max().item())
        self._ensure_gather_scratch(kv_cache, device, full_k.dtype, bsz=bsz, max_kv=max_kv)

        pos_range = torch.arange(max_kv, device=device, dtype=torch.int32)
        cache_row_offsets = (
            torch.arange(bsz, device=device, dtype=torch.int32)[:, None]
            * int(self._scratch_seq_len)
        )
        req_to_token = cache_row_offsets + pos_range[None, :]

        slot_parts = []
        for b in range(bsz):
            p0, p1 = int(prefix_cpu_list[b]), int(kv_lens_cpu[b].item())
            slot_parts.append(req_to_token[b, p0:p1])
        write_slots = torch.cat(slot_parts).to(torch.int64)

        # Persist into the cache-manager pool: idx_K -> scale region,
        # main K/V -> paged pool. Both are PD-transferable.
        self._source_idx_k_from_paged(
            kv_cache, full_idx_k, write_slots,
            req_to_token, kv_lens_cpu, attn_inputs, device,
        )
        self._source_main_kv_from_paged(
            kv_cache, full_k, full_v, write_slots,
            req_to_token, kv_lens_cpu, attn_inputs, device,
        )

        # PD separation: register this MSA layer's paged K/V (and idx_K on the
        # scale region) with the cache store, exactly like the non-CP prefill
        # path. cache_store_inputs already carries the prefill_cp_size / tp_rank
        # so the C++ writer stores only this rank's 1/cp_size page-RR shard.
        # Without this the decode side waits forever for the missing MSA blocks.
        if (
            kv_cache is not None
            and attn_inputs.is_prefill
            and attn_inputs.cache_store_inputs
        ):
            from rtp_llm.models_py.modules.factory.attention import common as _attn_common

            _write_impl = _attn_common.create_write_cache_store_impl(attn_inputs)
            _attn_common.apply_write_cache_store(_write_impl, attn_inputs, kv_cache)

        # Pack three small CPU lists (segment_req_ids/segment_lengths/
        # segment_starts) into ONE contiguous int64 buffer and do a single
        # pinned H2D, then split + cast on-device. Replaces 3 separate
        # torch.tensor(list, device=cuda) calls (each a pageable H2D).
        n_seg = len(segment_lengths)
        _packed_np = np.concatenate([
            np.asarray(segment_req_ids, dtype=np.int64),
            np.asarray(segment_lengths, dtype=np.int64),
            np.asarray(segment_starts, dtype=np.int64),
        ])
        _packed_dev = torch.from_numpy(_packed_np).to(device=device, non_blocking=True)
        # dim-0 slices of a 1-D tensor are always contiguous; .contiguous()
        # afterwards is a no-op but defensive in case allocator returns a view.
        segment_req_ids_t = _packed_dev[:n_seg].contiguous()                          # int64
        segment_lengths_t = _packed_dev[n_seg : 2 * n_seg].to(torch.int32)            # fresh contiguous
        prefix_i32 = _packed_dev[2 * n_seg :].to(torch.int32)                         # fresh contiguous

        req_to_token_segments = req_to_token.index_select(
            0, segment_req_ids_t
        ).contiguous()
        slot_ids = torch.arange(n_seg, device=device, dtype=torch.int64)
        cu_seqlens = torch.zeros(n_seg + 1, device=device, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(segment_lengths_t, dim=0)
        kv_lens_device = kv_lens_cpu.to(device=device, dtype=torch.int32)
        seq_lens_i32 = kv_lens_device.index_select(0, segment_req_ids_t)
        max_seqlen_q = max(int(x) for x in segment_lengths)
        max_seqlen_k = int(kv_lens_cpu.max().item())

        # Q is already in rank-local zigzag order. The Triton/trtllm-gen
        # kernel stores O by cu_seqlens offsets, so no output restore /
        # all-gather is needed here.
        main_k = self._scratch_k
        main_v = self._scratch_v
        idx_kc = self._scratch_idx_k
        _idx_o, o = minimax_sparse_prefill(
            q=q, k_cache=main_k, v_cache=main_v, sink=None,
            idx_q=idx_q, idx_k_cache=idx_kc, idx_v_cache=None,
            idx_sink=None, req_to_token=req_to_token_segments, slot_ids=slot_ids,
            cu_seqlens=cu_seqlens, seq_lens=seq_lens_i32,
            prefix_lens=prefix_i32,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            block_size_q=1, block_size_k=self.block_size,
            topk=self.topk_blocks, init_blocks=self.init_blocks,
            local_blocks=self.local_blocks, score_type=self.score_type,
            disable_index_value=self.disable_index_value,
            workspace=MSAAttention._get_trtllm_workspace(device),
        )

        output = self.o_proj(o.reshape(local_tokens, -1).contiguous())
        return output

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from rtp_llm.models_py.triton_kernels.sparse_msa.minimax_sparse import (
            minimax_sparse_decode,
            minimax_sparse_prefill,
        )

        assert kv_cache is not None, "MSAAttention requires a KV cache"
        assert (
            attn_inputs.kv_cache_kernel_block_id_device is not None
        ), "MSAAttention requires a block table"

        if (
            self.cp_enabled
            and attn_inputs.is_prefill
            and attn_inputs.context_parallel_info is not None
        ):
            return self._forward_cp_prefill(
                hidden_states, attn_inputs, kv_cache,
                x_fp8=x_fp8, x_scale=x_scale,
            )

        input_shape = hidden_states.shape[:-1]
        total_tokens = hidden_states.shape[0]
        device = hidden_states.device

        # --- main QKV + per-head Gemma QK norm ---
        if x_fp8 is not None and x_scale is not None:
            qkv = self.qkv_proj(x_fp8, input_scales=x_scale)
        else:
            qkv = self.qkv_proj(hidden_states)
        if self.qk_fuse_norm is not None:
            qkv = self.qk_fuse_norm(qkv)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.reshape(total_tokens, self.head_num, self.head_dim)
        k = k.reshape(total_tokens, self.kv_head_num, self.head_dim)
        v = v.reshape(total_tokens, self.kv_head_num, self.head_dim)

        # --- index branch: proj -> per-head Gemma norm ---
        idx_q = F.linear(hidden_states, self.idx_q_w)
        idx_k = F.linear(hidden_states, self.idx_k_w)
        idx_q = idx_q.reshape(total_tokens, self.num_idx_heads, self.idx_head_dim)
        idx_k = idx_k.reshape(total_tokens, 1, self.idx_head_dim)
        idx_q = _gemma_rmsnorm_per_head(idx_q, self.idx_q_norm_w, self.layernorm_eps)
        idx_k = _gemma_rmsnorm_per_head(idx_k, self.idx_k_norm_w, self.layernorm_eps)

        # --- addressing (req_to_token / slot_ids / positions / write slots) ---
        if self.cp_enabled:
            alloc_kv_lens, _, _ = self._get_lengths(attn_inputs)
            self._ensure_gather_scratch(
                kv_cache,
                device,
                k.dtype,
                bsz=int(alloc_kv_lens.numel()),
                max_kv=int(alloc_kv_lens.max().item()),
            )
            (
                req_to_token,
                slot_ids,
                kv_lens,
                positions,
                write_slots,
                prefix_lens,
                inlens,
            ) = self._build_compact_addressing(attn_inputs, device)
        else:
            (
                req_to_token,
                slot_ids,
                kv_lens,
                positions,
                write_slots,
                prefix_lens,
                inlens,
            ) = self._build_addressing(attn_inputs, device)
            self._ensure_gather_scratch(
                kv_cache,
                device,
                k.dtype,
                max_slot=self._max_active_slot(req_to_token, kv_lens),
            )

        # --- partial RoPE on main q/k and index q/k ---
        q = q.contiguous()
        k = k.contiguous()
        self._apply_rope(q, k, positions)
        idx_q = idx_q.contiguous()
        idx_k = idx_k.contiguous()
        self._apply_rope(idx_q, idx_k, positions)

        # --- write current tokens into the cache-manager pool ---
        # idx_K -> paged scale region; main K/V -> paged pool. Both are
        # PD-transferable and the scratch is filled from paged so the kernel
        # reads paged-sourced data.
        if kv_cache is not None:
            self._source_idx_k_from_paged(
                kv_cache, idx_k, write_slots, req_to_token, kv_lens, attn_inputs, device
            )
            self._source_main_kv_from_paged(
                kv_cache, k, v, write_slots, req_to_token, kv_lens, attn_inputs, device
            )

        # PD separation: register this MSA layer's paged K/V (and the idx_K
        # piggybacked on the scale region) with the cache store, exactly like
        # dense CausalAttention does through its fmha_impl. Without this the
        # decode side waits forever for the missing MSA-layer blocks.
        if (
            kv_cache is not None
            and attn_inputs.is_prefill
            and attn_inputs.cache_store_inputs
        ):
            from rtp_llm.models_py.modules.factory.attention import common as _attn_common

            _write_impl = _attn_common.create_write_cache_store_impl(attn_inputs)
            _attn_common.apply_write_cache_store(_write_impl, attn_inputs, kv_cache)

        # --- sparse attention via Triton MSA kernels ---
        main_k = self._scratch_k
        main_v = self._scratch_v
        idx_kc = self._scratch_idx_k
        max_seqlen_k = int(kv_lens.max().item())
        if attn_inputs.is_prefill:
            cu_seqlens = attn_inputs.cu_seqlens[: slot_ids.numel() + 1].to(torch.int32)
            seq_lens = kv_lens.to(torch.int32)
            prefix_i32 = prefix_lens.to(torch.int32)
            max_seqlen_q = int(inlens.max().item())
            _idx_o, o = minimax_sparse_prefill(
                q=q,
                k_cache=main_k,
                v_cache=main_v,
                sink=None,
                idx_q=idx_q,
                idx_k_cache=idx_kc,
                idx_v_cache=None,
                idx_sink=None,
                req_to_token=req_to_token,
                slot_ids=slot_ids,
                cu_seqlens=cu_seqlens,
                seq_lens=seq_lens,
                prefix_lens=prefix_i32,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                block_size_q=1,
                block_size_k=self.block_size,
                topk=self.topk_blocks,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                score_type=self.score_type,
                disable_index_value=self.disable_index_value,
                workspace=MSAAttention._get_trtllm_workspace(device),
            )
        else:
            seq_lens = kv_lens.to(torch.int32)
            # Decode = 1 query token per request: cu_seqlens is [0,1,...,batch].
            # Pass it + prefix_lens + the trtllm workspace so minimax_sparse_decode
            # can take the trtllm-gen sparse-decode fast path (same op as prefill)
            # instead of the legacy triton step3; falls back to triton when the
            # trtllm gate is not satisfied (e.g. multi-request batch).
            decode_bsz = int(slot_ids.numel())
            decode_cu_seqlens = torch.arange(
                decode_bsz + 1, device=device, dtype=torch.int32
            )
            _dbg = _MSA_DEBUG and _MSA_DEBUG_STEP.get(self.layer_idx, 0) < _MSA_DEBUG_MAX_STEPS
            if _dbg:
                _MSA_DEBUG_STEP[self.layer_idx] = _MSA_DEBUG_STEP.get(self.layer_idx, 0) + 1
                _bk = int(self.block_size)
                _nblk = (max_seqlen_k + _bk - 1) // max(_bk, 1)
                _kvl = kv_lens.detach().cpu().tolist()
                _r2t = tuple(req_to_token.shape)
                _mk = main_k.shape[0] if main_k is not None else -1
                _ik = idx_kc.shape[0] if idx_kc is not None else -1
                print(
                    f"[M3_MSA_DEBUG][decode] L{self.layer_idx} bsz={decode_bsz} "
                    f"kv_lens={_kvl} max_seqlen_k={max_seqlen_k} block_size_k={_bk} "
                    f"num_kv_blocks={_nblk} topk={self.topk_blocks} "
                    f"req_to_token={_r2t} scratch_k_slots={_mk} idx_scratch_slots={_ik} "
                    f"cp={self.cp_enabled} kv_sharded={self._kv_sharded} "
                    f"side_seq_len={self._scratch_seq_len}",
                    flush=True,
                )
                torch.cuda.synchronize(device)
                _t0 = time.perf_counter()
            _idx_o, o = minimax_sparse_decode(
                q=q,
                sink=None,
                k_cache=main_k,
                v_cache=main_v,
                idx_q=idx_q,
                idx_sink=None,
                idx_k_cache=idx_kc,
                idx_v_cache=None,
                req_to_token=req_to_token,
                slot_ids=slot_ids,
                seq_lens=seq_lens,
                max_seqlen=max_seqlen_k,
                block_size_q=1,
                block_size_k=self.block_size,
                topk=self.topk_blocks,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                score_type=self.score_type,
                disable_index_value=self.disable_index_value,
                workspace=MSAAttention._get_trtllm_workspace(device),
                cu_seqlens=decode_cu_seqlens,
                prefix_lens=prefix_lens.to(torch.int32),
            )
            if _dbg:
                torch.cuda.synchronize(device)
                print(
                    f"[M3_MSA_DEBUG][decode] L{self.layer_idx} "
                    f"sparse_decode_ms={(time.perf_counter() - _t0) * 1e3:.2f}",
                    flush=True,
                )

        attn_output = o.reshape(*input_shape, -1).contiguous()
        output = self.o_proj(attn_output)
        if self.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output

"""MiniMax-M3 sparse attention (MSA) module.

Wires the ported Triton MSA kernels (``rtp_llm/models_py/triton_kernels/
sparse_msa``) into rtp-llm's GenericMoe decoder for MiniMax-M3 *sparse* layers
(e.g. layers 3,4 in the 5-layer mini model). Dense layers keep using the
shared FlashInfer FMHA impl; only sparse layers are routed here.

Design (Option A — token-slot side cache):

* The MSA Triton kernels consume the main K/V and the index-K as flat
  *token-slot* tensors ``[max_slots, num_kv_heads, head_dim]`` addressed by a
  ``req_to_token [max_reqs, max_kv_len]`` map, plus ``slot_ids [batch]``. This
  layout is incompatible with rtp-llm's paged HND main cache, so instead of
  reading the paged cache we maintain *our own* per-layer token-slot side
  caches for the sparse layers and write K/V/index-K into them each step.

* In the normal non-CP path, the physical slot for ``(request b, token
  position p)`` reuses rtp-llm's block table exactly like the paged path::

      slot = block_table[b, p // page_size] * page_size + (p % page_size)

  so slots never collide across live requests and persist across decode steps
  (the block table is stable for a request's lifetime). ``req_to_token`` is
  built from the same formula, and ``slot_ids = arange(batch)`` because we
  build a per-batch ``req_to_token`` (row == batch index).

* In CP prefill, K/V are all-gathered into full sequence order while Q stays
  rank-local. The CP decode path uses a compact per-request side cache indexed
  by logical token position instead of the global KV-cache block pool; otherwise
  the full model would allocate multi-GB BF16 side caches per sparse layer.

The index branch (``index_q_proj`` / ``index_k_proj`` + per-head Gemma RMSNorm
+ partial RoPE) only selects top-k blocks; with ``disable_index_value=True``
(M3 default) it does not contribute to the attention value, so ``idx_v`` is
``None`` and the index output ``idx_o`` is discarded.
"""

from typing import Any, Dict, Optional

import torch
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


class MSAAttention(nn.Module):
    """MiniMax-M3 sparse attention for a single sparse layer."""

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

        # token-slot side caches (allocated lazily on first forward)
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None
        self.idx_k_cache: Optional[torch.Tensor] = None
        self._side_cache_batch_size = 0
        self._side_cache_seq_len = 0

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

    def _ensure_side_caches(
        self,
        kv_cache: LayerKVCache,
        device: torch.device,
        dtype: torch.dtype,
        bsz: Optional[int] = None,
        max_kv: Optional[int] = None,
        max_slot: Optional[int] = None,
    ) -> None:
        if self.cp_enabled:
            if bsz is None or max_kv is None:
                raise RuntimeError("CP MSA side cache requires batch size and kv length")
            target_bsz = max(int(bsz), self._side_cache_batch_size, 1)
            requested_seq_len = max(int(max_kv), self._side_cache_seq_len, 1)
            grow_granularity = max(int(self.page_size), 256)
            target_seq_len = (
                (requested_seq_len + grow_granularity - 1)
                // grow_granularity
                * grow_granularity
            )
            if (
                self.k_cache is not None
                and self._side_cache_batch_size >= int(bsz)
                and self._side_cache_seq_len >= int(max_kv)
            ):
                return

            old_k = self.k_cache
            old_v = self.v_cache
            old_idx_k = self.idx_k_cache
            old_bsz = self._side_cache_batch_size
            old_seq_len = self._side_cache_seq_len
            target_slots = target_bsz * target_seq_len
            new_k = torch.zeros(
                target_slots,
                self.kv_head_num,
                self.head_dim,
                dtype=dtype,
                device=device,
            )
            new_v = torch.zeros_like(new_k)
            new_idx_k = torch.zeros(
                target_slots,
                1,
                self.idx_head_dim,
                dtype=dtype,
                device=device,
            )
            if old_k is not None and old_v is not None and old_idx_k is not None:
                copy_bsz = min(old_bsz, target_bsz)
                copy_seq_len = min(old_seq_len, target_seq_len)
                if copy_bsz > 0 and copy_seq_len > 0:
                    new_k.view(
                        target_bsz, target_seq_len, self.kv_head_num, self.head_dim
                    )[:copy_bsz, :copy_seq_len].copy_(
                        old_k.view(
                            old_bsz, old_seq_len, self.kv_head_num, self.head_dim
                        )[:copy_bsz, :copy_seq_len]
                    )
                    new_v.view(
                        target_bsz, target_seq_len, self.kv_head_num, self.head_dim
                    )[:copy_bsz, :copy_seq_len].copy_(
                        old_v.view(
                            old_bsz, old_seq_len, self.kv_head_num, self.head_dim
                        )[:copy_bsz, :copy_seq_len]
                    )
                    new_idx_k.view(target_bsz, target_seq_len, 1, self.idx_head_dim)[
                        :copy_bsz, :copy_seq_len
                    ].copy_(
                        old_idx_k.view(old_bsz, old_seq_len, 1, self.idx_head_dim)[
                            :copy_bsz, :copy_seq_len
                        ]
                    )
            self.k_cache = new_k
            self.v_cache = new_v
            self.idx_k_cache = new_idx_k
            self._side_cache_batch_size = target_bsz
            self._side_cache_seq_len = target_seq_len
            return

        if max_slot is None:
            raise RuntimeError("non-CP MSA side cache requires max active slot")
        requested_slots = max(int(max_slot) + 1, 1)
        if self.k_cache is not None and self.k_cache.shape[0] >= requested_slots:
            return
        grow_granularity = max(int(self.page_size), 256)
        target_slots = (
            (requested_slots + grow_granularity - 1) // grow_granularity
        ) * grow_granularity
        old_k = self.k_cache
        old_v = self.v_cache
        old_idx_k = self.idx_k_cache
        new_k = torch.zeros(
            target_slots, self.kv_head_num, self.head_dim, dtype=dtype, device=device
        )
        new_v = torch.zeros_like(new_k)
        new_idx_k = torch.zeros(
            target_slots, 1, self.idx_head_dim, dtype=dtype, device=device
        )
        if old_k is not None and old_v is not None and old_idx_k is not None:
            copy_slots = min(old_k.shape[0], target_slots)
            if copy_slots > 0:
                new_k[:copy_slots].copy_(old_k[:copy_slots])
                new_v[:copy_slots].copy_(old_v[:copy_slots])
                new_idx_k[:copy_slots].copy_(old_idx_k[:copy_slots])
        self.k_cache = new_k
        self.v_cache = new_v
        self.idx_k_cache = new_idx_k

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
        """CP path addressing over the compact per-request side cache."""
        if self._side_cache_seq_len <= 0:
            raise RuntimeError("compact MSA side cache is not initialized")
        kv_lens, prefix, inlen = self._get_lengths(attn_inputs)
        bsz = int(kv_lens.numel())
        max_kv = int(kv_lens.max().item())
        pos = torch.arange(max_kv, device=device, dtype=torch.int32)
        row_offsets = (
            torch.arange(bsz, device=device, dtype=torch.int32)[:, None]
            * int(self._side_cache_seq_len)
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

        chunk_lengths_cpu = (
            cp_info.prefill_cp_chunk_lengths.detach().cpu().to(torch.int64).tolist()
        )
        prefix_cpu = attn_inputs.prefix_lengths.detach().cpu().to(torch.int64)
        prefix_cpu_list = prefix_cpu.tolist()
        if sum(int(x) for x in chunk_lengths_cpu) != local_tokens:
            raise RuntimeError(
                "MSA CP prefill expects rank-local token count to match "
                "prefill_cp_chunk_lengths; got "
                f"local_tokens={local_tokens}, chunks={chunk_lengths_cpu}"
            )

        local_shuffle = cp_info.prefill_shuffle_indices.to(
            device=device, dtype=torch.int64
        )
        local_pos_parts = []
        segment_lengths = []
        segment_starts = []
        segment_req_ids = []
        cursor = 0
        for b, chunk_len in enumerate(chunk_lengths_cpu):
            chunk_len = int(chunk_len)
            pair_len = chunk_len // 2
            req_prefix = int(prefix_cpu_list[b])
            chunk_positions = local_shuffle[cursor : cursor + chunk_len].clamp(min=0)
            local_pos_parts.append(chunk_positions + req_prefix)
            for rel_start in (0, pair_len):
                segment_lengths.append(pair_len)
                segment_req_ids.append(b)
                if pair_len > 0:
                    start_pos = int(local_shuffle[cursor + rel_start].item())
                    segment_starts.append(req_prefix + max(start_pos, 0))
                else:
                    segment_starts.append(req_prefix)
            cursor += chunk_len
        local_positions = torch.cat(local_pos_parts).to(torch.int32)

        q = q.contiguous()
        k = k.contiguous()
        self._apply_rope(q, k, local_positions)
        idx_q = idx_q.contiguous()
        idx_k = idx_k.contiguous()
        self._apply_rope(idx_q, idx_k, local_positions)

        all_k = all_gather(k, group=Group.TP)
        all_v = all_gather(v.contiguous(), group=Group.TP)
        all_idx_k = all_gather(idx_k, group=Group.TP)

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
        self._ensure_side_caches(kv_cache, device, full_k.dtype, bsz=bsz, max_kv=max_kv)
        assert self.k_cache is not None
        assert self.v_cache is not None
        assert self.idx_k_cache is not None

        pos_range = torch.arange(max_kv, device=device, dtype=torch.int32)
        cache_row_offsets = (
            torch.arange(bsz, device=device, dtype=torch.int32)[:, None]
            * int(self._side_cache_seq_len)
        )
        req_to_token = cache_row_offsets + pos_range[None, :]

        slot_parts = []
        for b in range(bsz):
            p0, p1 = int(prefix_cpu_list[b]), int(kv_lens_cpu[b].item())
            slot_parts.append(req_to_token[b, p0:p1])
        write_slots = torch.cat(slot_parts).to(torch.int64)

        self.k_cache[write_slots] = full_k
        self.v_cache[write_slots] = full_v
        self.idx_k_cache[write_slots] = full_idx_k

        segment_req_ids_t = torch.tensor(
            segment_req_ids, device=device, dtype=torch.long
        )
        req_to_token_segments = req_to_token.index_select(
            0, segment_req_ids_t
        ).contiguous()
        slot_ids = torch.arange(len(segment_req_ids), device=device, dtype=torch.int64)
        segment_lengths_t = torch.tensor(
            segment_lengths, device=device, dtype=torch.int32
        )
        cu_seqlens = torch.zeros(
            len(segment_lengths) + 1, device=device, dtype=torch.int32
        )
        cu_seqlens[1:] = torch.cumsum(segment_lengths_t, dim=0)
        kv_lens_device = kv_lens_cpu.to(device=device, dtype=torch.int32)
        seq_lens_i32 = kv_lens_device.index_select(0, segment_req_ids_t)
        prefix_i32 = torch.tensor(segment_starts, device=device, dtype=torch.int32)
        max_seqlen_q = max(int(x) for x in segment_lengths)
        max_seqlen_k = int(kv_lens_cpu.max().item())

        # Q is already in rank-local zigzag order. The Triton kernel stores O
        # by cu_seqlens offsets, so no output restore/all-gather is needed here.
        _idx_o, o = minimax_sparse_prefill(
            q=q, k_cache=self.k_cache, v_cache=self.v_cache, sink=None,
            idx_q=idx_q, idx_k_cache=self.idx_k_cache, idx_v_cache=None,
            idx_sink=None, req_to_token=req_to_token_segments, slot_ids=slot_ids,
            cu_seqlens=cu_seqlens, seq_lens=seq_lens_i32,
            prefix_lens=prefix_i32,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            block_size_q=1, block_size_k=self.block_size,
            topk=self.topk_blocks, init_blocks=self.init_blocks,
            local_blocks=self.local_blocks, score_type=self.score_type,
            disable_index_value=self.disable_index_value,
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
            self._ensure_side_caches(
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
            self._ensure_side_caches(
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

        # --- write current tokens into token-slot side caches ---
        assert self.k_cache is not None
        assert self.v_cache is not None
        assert self.idx_k_cache is not None
        self.k_cache[write_slots] = k
        self.v_cache[write_slots] = v
        self.idx_k_cache[write_slots] = idx_k

        # --- sparse attention via Triton MSA kernels ---
        max_seqlen_k = int(kv_lens.max().item())
        if attn_inputs.is_prefill:
            cu_seqlens = attn_inputs.cu_seqlens[: slot_ids.numel() + 1].to(torch.int32)
            seq_lens = kv_lens.to(torch.int32)
            prefix_i32 = prefix_lens.to(torch.int32)
            max_seqlen_q = int(inlens.max().item())
            _idx_o, o = minimax_sparse_prefill(
                q=q,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                sink=None,
                idx_q=idx_q,
                idx_k_cache=self.idx_k_cache,
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
            )
        else:
            seq_lens = kv_lens.to(torch.int32)
            _idx_o, o = minimax_sparse_decode(
                q=q,
                sink=None,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                idx_q=idx_q,
                idx_sink=None,
                idx_k_cache=self.idx_k_cache,
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
            )

        attn_output = o.reshape(*input_shape, -1).contiguous()
        output = self.o_proj(attn_output)
        if self.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output

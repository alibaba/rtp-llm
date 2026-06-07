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

* The physical slot for ``(request b, token position p)`` reuses rtp-llm's
  block table exactly like the paged path::

      slot = block_table[b, p // page_size] * page_size + (p % page_size)

  so slots never collide across live requests and persist across decode steps
  (the block table is stable for a request's lifetime). ``req_to_token`` is
  built from the same formula, and ``slot_ids = arange(batch)`` because we
  build a per-batch ``req_to_token`` (row == batch index).

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
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
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
    """Gemma-style RMSNorm (``* (1 + weight)``) over the last dim.

    Mirrors sglang ``GemmaRMSNorm.forward_native`` exactly (fp32 compute,
    1+weight scale, cast back to input dtype). ``x`` is ``[..., D]`` and
    ``weight`` is ``[D]``.
    """
    orig_dtype = x.dtype
    xf = x.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + eps)
    xf = xf * (1.0 + weight.float())
    return xf.to(orig_dtype)


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
        self.layernorm_eps = layernorm_eps

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
        self.idx_q_w = weights[W.msa_idx_q_w]  # [num_idx_heads*idx_dim, hidden]
        self.idx_k_w = weights[W.msa_idx_k_w]  # [idx_dim, hidden]
        self.idx_q_norm_w = weights[W.msa_idx_q_norm]  # [idx_dim]
        self.idx_k_norm_w = weights[W.msa_idx_k_norm]  # [idx_dim]
        self.num_idx_heads = self.idx_q_w.shape[0] // self.idx_head_dim

        # --- sparse params ---
        self.topk_blocks = int(sparse_config["topk_blocks"])
        self.block_size = int(sparse_config["block_size"])
        self.init_blocks = int(sparse_config["init_blocks"])
        self.local_blocks = int(sparse_config["local_blocks"])
        self.score_type = str(sparse_config.get("score_type", "max"))
        self.disable_index_value = layer_idx in set(
            sparse_config.get("disable_value_layer_ids", [])
        )

        # --- partial RoPE cos/sin cache (matches the dense FlashInfer path:
        # is_neox_style=False -> interleave=False; rope_config.dim is the
        # partial rotary dim, e.g. 64). ---
        from rtp_llm.ops import get_rope_cache_once

        self._rope_theta = attn_config.rope_config.base
        try:
            rope_cache = get_rope_cache_once(
                attn_config.rope_config,
                attn_config.max_seq_len + attn_config.gen_num_per_cycle + 1,
                is_cuda=True,
                interleave=False,
            )
            self.cos_sin_cache = rope_cache.data
        except Exception:
            self.cos_sin_cache = None

        # token-slot side caches (allocated lazily on first forward)
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None
        self.idx_k_cache: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
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
                interleave=False,
            )
        else:
            import flashinfer

            flashinfer.apply_rope_pos_ids_inplace(
                q, k, positions, rope_theta=self._rope_theta
            )

    def _ensure_side_caches(
        self, kv_cache: LayerKVCache, device: torch.device, dtype: torch.dtype
    ) -> None:
        if self.k_cache is not None:
            return
        base = kv_cache.kv_cache_base
        num_blocks = int(base.shape[0])
        max_slots = num_blocks * self.page_size
        self.k_cache = torch.zeros(
            max_slots, self.kv_head_num, self.head_dim, dtype=dtype, device=device
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        self.idx_k_cache = torch.zeros(
            max_slots, 1, self.idx_head_dim, dtype=dtype, device=device
        )

    def _build_addressing(
        self, attn_inputs: PyAttentionInputs, device: torch.device
    ):
        """Return (req_to_token [B, max_kv], slot_ids [B], kv_lens [B],
        positions [T], write_slots [T]) from rtp-llm block table + lengths."""
        block_table = attn_inputs.kv_cache_kernel_block_id_device  # [B, max_blocks]
        bsz = block_table.size(0)
        max_blocks = block_table.size(1)
        is_prefill = attn_inputs.is_prefill

        if is_prefill:
            prefix = attn_inputs.prefix_lengths.to(torch.int64)  # [B]
            inlen = attn_inputs.input_lengths.to(torch.int64)  # [B]
            kv_lens = prefix + inlen
        else:
            seqlen = attn_inputs.sequence_lengths.to(torch.int64)  # [B]
            kv_lens = seqlen + 1
            prefix = kv_lens - 1
            inlen = torch.ones_like(kv_lens)

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
        (
            req_to_token,
            slot_ids,
            kv_lens,
            positions,
            write_slots,
            prefix_lens,
            inlens,
        ) = self._build_addressing(attn_inputs, device)

        # --- partial RoPE on main q/k and index q/k ---
        q = q.contiguous()
        k = k.contiguous()
        self._apply_rope(q, k, positions)
        idx_q = idx_q.contiguous()
        idx_k = idx_k.contiguous()
        self._apply_rope(idx_q, idx_k, positions)

        # --- write current tokens into token-slot side caches ---
        self._ensure_side_caches(kv_cache, device, k.dtype)
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

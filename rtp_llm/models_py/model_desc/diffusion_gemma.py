"""DiffusionGemma (google/diffusiongemma-26B-A4B-it) model for RTP-LLM.

Architecture (HF reference: transformers models/diffusion_gemma):
  AR encoder + block-diffusion decoder sharing all backbone weights. Only
  ``layer_scalar`` buffers differ per mode, plus the decoder-only
  ``self_conditioning`` module. 30 MoE layers: 25 sliding_attention (window
  1024, Q16x256, KV8x256 with v_proj) alternating with 5 full_attention every
  6th layer (Q16x512, KV2x512 without v_proj: V = RMSNorm(k_proj(x))).
  Shared expert MLP (inter 2112) + routed experts (128 x inter 704, top-8)
  with a raw-residual router (norm -> *scale*hidden^-0.5 -> proj -> softmax
  -> top8 -> renorm -> *per_expert_scale).

This module implements the encoder-mode backbone (causal forward, KV cache
writes) used for prompt prefill and canvas commit (target-verify). The
decoder-mode canvas forward lives in diffusion_gemma_proposer.py which reuses
these layers with is_causal=False attention and decoder layer scalars.
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    Embedding,
    LinearFactory,
    MultimodalEmbeddingInjector,
    RMSNorm,
)
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyModelInputs,
    PyModelOutputs,
    rtp_llm_ops,
)
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)


SWA_TAG = "swa_kv"
GLOBAL_TAG = "global_kv"
LAYER_TYPE_SLIDING = "sliding_attention"
LAYER_TYPE_FULL = "full_attention"


def diffusion_gemma_layer_types(num_layers: int) -> list:
    """Gemma pattern: full attention every 6th layer, sliding otherwise."""
    types = []
    for idx in range(num_layers):
        if (idx + 1) % 6 == 0:
            types.append(LAYER_TYPE_FULL)
        else:
            types.append(LAYER_TYPE_SLIDING)
    return types


class DiffusionGemmaRope(nn.Module):
    """Per-layer-type RoPE.

    sliding: theta=1e4, default style, full head_dim (256).
    global: theta=1e6, "proportional" style — the first
    ``partial_rotary_factor * head_dim`` dims rotate with frequencies
    normalized by the FULL head_dim, remaining dims carry zero frequency
    (identity). Reference: transformers _compute_proportional_rope_parameters.
    Uses the non-interleaved (rotate-half) layout like Gemma/Qwen.
    """

    def __init__(
        self,
        theta: float,
        head_dim: int,
        partial_rotary_factor: float,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.head_dim = head_dim
        rope_angles = int(partial_rotary_factor * head_dim // 2)
        inv_freq_rotated = 1.0 / (
            theta
            ** (
                torch.arange(0, 2 * rope_angles, 2, dtype=torch.float64, device=device)
                / head_dim
            )
        )
        nope_angles = head_dim // 2 - rope_angles
        if nope_angles > 0:
            inv_freq = torch.cat(
                [inv_freq_rotated, torch.zeros(nope_angles, dtype=torch.float64, device=device)]
            )
        else:
            inv_freq = inv_freq_rotated
        self.register_buffer("inv_freq", inv_freq.to(dtype=torch.float32), persistent=False)
        self.max_seq_len_cached = 0
        self._device = device
        self._dtype = dtype
        self._cos: Optional[torch.Tensor] = None
        self._sin: Optional[torch.Tensor] = None
        self._build_cache(max(max_seq_len, 1))

    def _build_cache(self, seq_len: int) -> None:
        if seq_len <= self.max_seq_len_cached:
            return
        self.max_seq_len_cached = max(seq_len, self.max_seq_len_cached * 2, 16)
        positions = torch.arange(
            self.max_seq_len_cached, device=self._device, dtype=torch.float32
        )
        freqs = torch.outer(positions, self.inv_freq.to(torch.float32))
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos = emb.cos().to(self._dtype)
        self._sin = emb.sin().to(self._dtype)

    def apply(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate q/k (…, head_dim) in-place-safe with rotate-half layout."""
        self._build_cache(int(positions.max().item()) + 1 if positions.numel() else 1)
        cos = self._cos[positions.long()].to(q.dtype)
        sin = self._sin[positions.long()].to(q.dtype)
        while cos.dim() < q.dim():
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)

        def rotate(x: torch.Tensor) -> torch.Tensor:
            half = x.shape[-1] // 2
            x1 = x[..., :half]
            x2 = x[..., half:]
            return torch.cat((x1 * cos[..., :half] - x2 * sin[..., :half],
                              x2 * cos[..., :half] + x1 * sin[..., :half]), dim=-1)

        return rotate(q), rotate(k)


class DiffusionGemmaAttnOp:
    """Unified attention op for encoder (causal, writes KV) and canvas
    (non-causal, read-only prefix) paths, per cache-group tag.

    Splits attention like PyFlashinferHybridPrefillAttnOp: a ragged segment
    over the current query block and a paged segment over the committed
    prefix, merged via LSE. Sliding layers rely on canvas_length(256) <
    sliding_window(1024), so intra-canvas attention is fully bidirectional
    while the paged prefix read applies window_left=-window.
    """

    def __init__(
        self,
        layer_type: str,
        head_num: int,
        kv_head_num: int,
        head_dim: int,
        sliding_window: int,
        page_size: int,
        dtype: torch.dtype,
        kv_dtype: Optional[torch.dtype],
        backend: str = "auto",
    ):
        self.layer_type = layer_type
        self.head_num = head_num
        self.kv_head_num = kv_head_num
        self.head_dim = head_dim
        self.window = sliding_window if layer_type == LAYER_TYPE_SLIDING else 0
        self.page_size = page_size
        self.dtype = dtype
        self.kv_dtype = kv_dtype or dtype
        self.sm_scale = 1.0  # reference: self.scaling = 1.0 (no 1/sqrt(d))
        self.use_flashinfer = False
        try:
            from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
                get_py_flashinfer_workspace_buffer,
                release_py_flashinfer_workspace_buffer,
            )

            self.g_workspace_buffer = get_py_flashinfer_workspace_buffer()
            from flashinfer.prefill import (
                BatchPrefillWithPagedKVCacheWrapper,
                BatchPrefillWithRaggedKVCacheWrapper,
            )
            from flashinfer.cascade import merge_state_in_place

            self.ragged_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                self.g_workspace_buffer, backend=backend
            )
            self.paged_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self.g_workspace_buffer, "HND", backend=backend
            )
            self.merge_state_in_place = merge_state_in_place
            self._release_ws = release_py_flashinfer_workspace_buffer
            from rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op import (
                KVCacheWriteOp,
            )

            self.write_op = KVCacheWriteOp(
                num_kv_heads=self.kv_head_num,
                head_size=self.head_dim,
                token_per_block=self.page_size,
            )
            self.fmha_params = None
            self.use_flashinfer = True
        except Exception as error:  # CPU-only environments
            logger.info("DiffusionGemma attention falls back to torch: %s", error)

    def __del__(self):
        if self.use_flashinfer:
            try:
                self._release_ws(self.g_workspace_buffer)
            except Exception:
                pass

    def _window_left(self) -> int:
        # HF sliding-window convention: attend positions in
        # [q - window + 1, q + window - 1]; flashinfer's window_left is the
        # inclusive lower bound, so -window + 1 matches the reference mask.
        return -(self.window - 1) if self.window > 0 else -1

    def _torch_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        prefix_k: Optional[torch.Tensor],
        prefix_v: Optional[torch.Tensor],
        causal: bool,
        prefix_lens: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """CPU reference attention: [tokens, heads, dim] layout, per-request
        masked softmax. Supports bidirectional canvas + windowed prefix and
        GQA (kv heads repeated to the query head count)."""
        if k.shape[1] != q.shape[1]:
            groups = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
            if prefix_k is not None and prefix_k.shape[1] != q.shape[1]:
                prefix_k = prefix_k.repeat_interleave(groups, dim=1)
                prefix_v = prefix_v.repeat_interleave(groups, dim=1)
        outs = []
        total = q.shape[0]
        for b in range(cu_seqlens.numel() - 1):
            q_lo = int(cu_seqlens[b].item())
            q_hi = int(cu_seqlens[b + 1].item())
            if q_hi <= q_lo:
                continue
            plen = int(prefix_lens[b].item())
            keys = []
            values = []
            if plen > 0 and prefix_k is not None:
                keys.append(prefix_k[q_lo : q_lo + plen])
                values.append(prefix_v[q_lo : q_lo + plen])
            keys.append(k[q_lo:q_hi])
            values.append(v[q_lo:q_hi])
            key = torch.cat(keys, dim=0)
            value = torch.cat(values, dim=0)
            score = torch.einsum("qhd,khd->hqk", q[q_lo:q_hi], key) * self.sm_scale
            q_pos = torch.arange(q_hi - q_lo)
            k_pos = torch.cat(
                [torch.arange(plen), torch.arange(q_hi - q_lo) + plen]
            )
            allowed = torch.ones(q_hi - q_lo, key.shape[0], dtype=torch.bool)
            if causal:
                allowed &= (
                    k_pos.unsqueeze(0) <= (q_pos + plen).unsqueeze(1)
                )
            if self.window > 0:
                allowed &= (
                    k_pos.unsqueeze(0) >= (q_pos + plen).unsqueeze(1) - (self.window - 1)
                )
            score = score.masked_fill(~allowed.unsqueeze(0), float("-inf"))
            probs = torch.softmax(score.float(), dim=-1).to(q.dtype)
            out = torch.einsum("hqk,khd->qhd", probs, value).to(q.dtype)
            outs.append((q_lo, q_hi, out))
        merged = torch.empty(total, q.shape[1], q.shape[2], dtype=q.dtype, device=q.device)
        for lo, hi, out in outs:
            merged[lo:hi] = out
        return merged

    def _reshape_paged(self, paged: torch.Tensor) -> torch.Tensor:
        """[blocks, kv*2*page] flat -> [blocks, 2, H, P, D] view."""
        if paged.dim() == 5:
            return paged
        if paged.dim() != 2:
            raise RuntimeError(f"unexpected KV cache layout dim={paged.dim()}")
        blocks = paged.shape[0]
        return paged.view(blocks, 2, self.kv_head_num, self.page_size, self.head_dim)

    def _gather_prefix(
        self, kv_cache: LayerKVCache, attn_inputs: Any, prefix_lens: torch.Tensor, batch_size: int
    ):
        paged = self._reshape_paged(kv_cache.kv_cache_base)
        block_table = attn_inputs.kv_cache_kernel_block_id
        if block_table is None or block_table.numel() == 0:
            block_table = attn_inputs.kv_cache_kernel_block_id_device
        if block_table is None or block_table.numel() == 0:
            return None, None
        block_table = block_table.reshape(batch_size, -1)
        pages = block_table[:, : (int(prefix_lens.max().item()) + self.page_size - 1) // self.page_size + 1]
        gathered_k = paged[pages.to(torch.long), 0]  # [B, blocks, H, P, D]
        gathered_v = paged[pages.to(torch.long), 1]
        k_rows = []
        v_rows = []
        for b in range(batch_size):
            plen = int(prefix_lens[b].item())
            if plen <= 0:
                k_rows.append(gathered_k.new_zeros(0, self.kv_head_num, self.head_dim))
                v_rows.append(gathered_v.new_zeros(0, self.kv_head_num, self.head_dim))
                continue
            k_parts = []
            v_parts = []
            for pb in range((plen + self.page_size - 1) // self.page_size):
                take = min(plen - pb * self.page_size, self.page_size)
                k_parts.append(gathered_k[b, pb, :, :take, :])
                v_parts.append(gathered_v[b, pb, :, :take, :])
            k_rows.append(torch.cat(k_parts, dim=1).transpose(0, 1))
            v_rows.append(torch.cat(v_parts, dim=1).transpose(0, 1))
        return torch.cat(k_rows, dim=0), torch.cat(v_rows, dim=0)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        attn_inputs: Any,
        positions: torch.Tensor,
        causal: bool,
        write_kv: bool,
    ) -> torch.Tensor:
        """q/k/v: [tokens, heads, dim]; returns [tokens, heads, dim]."""
        prefix_lens = attn_inputs.prefix_lengths
        if prefix_lens is None:
            prefix_lens = torch.zeros(1, dtype=torch.int32, device=q.device)
        cu_seqlens = attn_inputs.cu_seqlens_device
        if cu_seqlens is None or cu_seqlens.numel() == 0:
            cu_seqlens = attn_inputs.cu_seqlens
        batch_size = int(prefix_lens.numel())
        qo_indptr = cu_seqlens[: batch_size + 1]

        has_prefix = (
            kv_cache is not None
            and int(prefix_lens.numel()) > 0
            and int(prefix_lens.max().item()) > 0
        )

        if not self.use_flashinfer or q.device.type != "cuda":
            prefix_k = prefix_v = None
            if has_prefix:
                prefix_k, prefix_v = self._gather_prefix(
                    kv_cache, attn_inputs, prefix_lens, batch_size
                )
            out = self._torch_attention(
                q, k, v, prefix_k, prefix_v, causal, prefix_lens, qo_indptr
            )
            if write_kv and kv_cache is not None:
                self._write_cache_torch(k, v, kv_cache, attn_inputs, prefix_lens, batch_size)
            return out

        qd = q
        kd = k
        vd = v
        new_out, new_lse = self.ragged_wrapper.run(
            qd, kd, vd,
            return_lse=True,
        )
        if write_kv and kv_cache is not None:
            self._write_cache_flashinfer(kd, vd, kv_cache, attn_inputs, positions)
        if has_prefix:
            paged = self._reshape_paged(kv_cache.kv_cache_base)
            prefix_out, prefix_lse = self.paged_wrapper.run(
                qd, paged,
                return_lse=True,
            )
            self.merge_state_in_place(new_out, new_lse, prefix_out, prefix_lse)
        return new_out

    def plan(self, attn_inputs: Any, causal: bool) -> None:
        """Plan flashinfer wrappers for this step (ragged + paged prefix)."""
        if not self.use_flashinfer:
            return
        batch_size = int(attn_inputs.input_lengths.numel())
        qo_indptr = attn_inputs.cu_seqlens_device
        if qo_indptr is None or qo_indptr.numel() == 0:
            qo_indptr = attn_inputs.cu_seqlens
        qo_indptr = qo_indptr[: batch_size + 1]
        self.ragged_wrapper.plan(
            qo_indptr,
            qo_indptr,
            self.head_num,
            self.kv_head_num,
            self.head_dim,
            causal=causal,
            window_left=self._window_left(),
            sm_scale=self.sm_scale,
            q_data_type=self.dtype,
            kv_data_type=self.kv_dtype,
            o_data_type=self.dtype,
        )
        prefix_lens = attn_inputs.prefix_lengths
        seq_lens = attn_inputs.sequence_lengths
        if seq_lens is None or seq_lens.numel() == 0:
            if prefix_lens is None or prefix_lens.numel() == 0:
                return
            seq_lens = prefix_lens + attn_inputs.input_lengths[:batch_size]
        if prefix_lens is None or int(prefix_lens.numel()) < batch_size:
            return
        block_table = attn_inputs.kv_cache_kernel_block_id
        if block_table is None or block_table.numel() == 0:
            block_table = attn_inputs.kv_cache_kernel_block_id_device
        if block_table is None or block_table.numel() == 0:
            return
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.fmha_params.fill_params(
            prefix_lens,
            seq_lens,
            attn_inputs.input_lengths,
            block_table,
            self.page_size,
            False,
        )
        self.write_op.set_params(self.fmha_params)
        if int(prefix_lens.max().item()) <= 0:
            return
        reuse_info = self.fmha_params.batch_reuse_info_vec_h
        prefix_lengths = reuse_info[:, 1]
        paged_kv_indptr = torch.empty(batch_size + 1, dtype=torch.int32, device="cpu")
        paged_kv_indptr[:-1].copy_(reuse_info[:, 2])
        paged_kv_indptr[-1] = self.fmha_params.reuse_cache_page_indice_h.numel()
        paged_kv_last_page_len = (prefix_lengths - 1) % self.page_size + 1
        self.paged_wrapper.plan(
            qo_indptr,
            paged_kv_indptr,
            self.fmha_params.reuse_cache_page_indice_d,
            paged_kv_last_page_len,
            self.head_num,
            self.kv_head_num,
            self.head_dim,
            self.page_size,
            causal=False,
            window_left=self._window_left(),
            sm_scale=self.sm_scale,
            q_data_type=self.dtype,
            kv_data_type=self.kv_dtype,
            o_data_type=self.dtype,
        )

    def _write_cache_flashinfer(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: LayerKVCache,
        attn_inputs: Any,
        positions: torch.Tensor,
    ) -> None:
        if getattr(self, "fmha_params", None) is not None:
            self.write_op.set_params(self.fmha_params)
        # SWA group layer caches arrive as the raw 2-D pool buffer
        # (makeLayerCache only expands FULL groups to 5-D); KVCacheWriteOp
        # indexes [:, 0, :, :, :], so reshape the per-call view first.
        kv_cache.kv_cache_base = self._reshape_paged(kv_cache.kv_cache_base)
        self.write_op.forward(k, v, kv_cache)

    def _write_cache_torch(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: LayerKVCache,
        attn_inputs: Any,
        prefix_lens: torch.Tensor,
        batch_size: int,
    ) -> None:
        """Write k/v rows [T, H, D] into the paged cache via the block table."""
        cache = self._reshape_paged(kv_cache.kv_cache_base)
        block_table = attn_inputs.kv_cache_kernel_block_id
        if block_table is None or block_table.numel() == 0:
            block_table = attn_inputs.kv_cache_kernel_block_id_device
        if block_table is None or block_table.numel() == 0:
            return
        block_table = block_table.reshape(batch_size, -1)
        rows_per_req = k.shape[0] // max(1, batch_size)
        pos = 0
        for b in range(batch_size):
            plen = int(prefix_lens[b].item())
            for r in range(rows_per_req):
                gpos = plen + r
                page = int(block_table[b, gpos // self.page_size].item())
                slot = gpos % self.page_size
                cache[page, 0, :, slot, :] = k[pos + r].to(cache.dtype)
                cache[page, 1, :, slot, :] = v[pos + r].to(cache.dtype)
            pos += rows_per_req


class DiffusionGemmaAttention(nn.Module):
    """Per-layer attention. Sliding layers have a real v_proj; global layers
    duplicate the k_proj output into the fused qkv's v slot at load time
    (V = RMSNorm(k_proj(x)) in the reference)."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        layer_type: str,
        hw_kernel_config: Optional[HWKernelConfig] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.parallelism_config = parallelism_config
        self.tp_size = parallelism_config.get_attn_tp_size()
        if layer_type == LAYER_TYPE_SLIDING:
            self.head_dim = config.dgm_head_dim
            self.kv_head_num = config.dgm_kv_head_num
        else:
            self.head_dim = config.dgm_global_head_dim
            self.kv_head_num = config.dgm_global_kv_head_num
        self.head_num = config.attn_config.head_num
        if self.tp_size > 1:
            self.head_num = self.head_num // self.tp_size
            if self.kv_head_num % self.tp_size == 0:
                self.kv_head_num = self.kv_head_num // self.tp_size
        self.q_size = self.head_num * self.head_dim
        self.kv_size = self.kv_head_num * self.head_dim
        self.qkv_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.attn_qkv_w,
            None,
            None,
            hw_kernel_config=hw_kernel_config,
        )
        self.o_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.attn_o_w,
            None,
            None,
            hw_kernel_config=hw_kernel_config,
        )
        self.qk_norm = _QKNorm(
            self.head_num, self.kv_head_num, self.head_dim,
            weights.get(W.q_ln_gamma), weights.get(W.k_ln_gamma),
            config.layernorm_eps,
        )
        self.v_norm = _NoScaleRMSNorm(config.layernorm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: DiffusionGemmaRope,
        positions: torch.Tensor,
        attn_op: DiffusionGemmaAttnOp,
        kv_cache: Optional[LayerKVCache],
        attn_inputs: Any,
        causal: bool,
        write_kv: bool,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        input_shape = hidden_states.shape[:-1]
        qkv = qkv.reshape(-1, self.q_size + 2 * self.kv_size)
        q = qkv[..., : self.q_size]
        k = qkv[..., self.q_size : self.q_size + self.kv_size]
        v = qkv[..., self.q_size + self.kv_size :]
        q, k = self.qk_norm(q, k, positions, rope, self.head_num, self.kv_head_num, self.head_dim)
        # v_norm is per-head (reference: RMSNorm without scale over head_dim).
        v = v.reshape(-1, self.head_dim)
        v = self.v_norm(v)
        v = v.reshape(-1, self.kv_head_num, self.head_dim)
        q = q.reshape(-1, self.head_num, self.head_dim)
        k = k.reshape(-1, self.kv_head_num, self.head_dim)
        attn_output = attn_op.forward(
            q, k, v, kv_cache, attn_inputs, positions, causal, write_kv
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        output = self.o_proj(attn_output)
        if self.tp_size > 1:
            from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

            output = all_reduce(output, group=Group.TP)
        return output


class _QKNorm(nn.Module):
    """Q/K RMSNorm + RoPE for the fused qkv slices (rotate-half layout).
    Returns (q, k) rotated."""

    def __init__(self, head_num, kv_head_num, head_dim, q_gamma, k_gamma, eps):
        super().__init__()
        self.q_norm = RMSNorm(q_gamma, eps) if q_gamma is not None else None
        self.k_norm = RMSNorm(k_gamma, eps) if k_gamma is not None else None

    def forward(self, q, k, positions, rope, head_num, kv_head_num, head_dim):
        if self.q_norm is not None:
            q = self.q_norm(q.reshape(-1, head_dim)).reshape(-1, head_num * head_dim)
        if self.k_norm is not None:
            k = self.k_norm(k.reshape(-1, head_dim)).reshape(-1, kv_head_num * head_dim)
        q = q.reshape(-1, head_num, head_dim)
        k = k.reshape(-1, kv_head_num, head_dim)
        q, k = rope.apply(q, k, positions)
        return q, k


class _NoScaleRMSNorm(nn.Module):
    """RMSNorm without weight (reference v_norm / router pre-norm)."""

    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps)).to(x.dtype)


class DiffusionGemmaGateMLP(nn.Module):
    """Gated GeGLU-tanh MLP (hidden_activation=gelu_pytorch_tanh): shared
    expert and self_conditioning share this structure."""

    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        gate_key: str,
        up_key: str,
        down_key: str,
        hw_kernel_config: Optional[HWKernelConfig] = None,
    ):
        super().__init__()
        self.gate_proj = LinearFactory.create_linear_from_weights(
            weights, gate_key, None, None, hw_kernel_config=hw_kernel_config
        )
        self.up_proj = LinearFactory.create_linear_from_weights(
            weights, up_key, None, None, hw_kernel_config=hw_kernel_config
        )
        self.down_proj = LinearFactory.create_linear_from_weights(
            weights, down_key, None, None, hw_kernel_config=hw_kernel_config
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            _gelu_tanh(self.gate_proj(x)) * self.up_proj(x)
        )


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x, approximate="tanh")


class DiffusionGemmaRouter(nn.Module):
    """Raw-residual router: no-scale RMSNorm -> *scale*hidden^-0.5 -> proj ->
    fp32 softmax -> top8 -> renormalize -> *per_expert_scale."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        hw_kernel_config: Optional[HWKernelConfig] = None,
    ):
        super().__init__()
        self.pre_norm = _NoScaleRMSNorm(config.layernorm_eps)
        # AtomicWeight implicitly unsqueezes 1-D checkpoint weights whose
        # name contains "scale" to [N, 1] (quant per-channel convention,
        # weight_module._load_raw_tensor); normalize back to 1-D here.
        self.scale = weights[W.dgm_router_scale].reshape(-1)
        self.scalar_root_size = config.hidden_size ** -0.5
        self.proj = LinearFactory.create_linear_from_weights(
            weights, W.moe_gate, None, None, hw_kernel_config=hw_kernel_config
        )
        self.per_expert_scale = weights[W.dgm_router_per_expert_scale].reshape(-1)
        self.top_k = config.moe_k
        self.num_experts = config.expert_num

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.pre_norm(hidden_states)
        h = h * self.scale * self.scalar_root_size
        scores = self.proj(h)
        probs = torch.softmax(scores.to(torch.float32), dim=-1)
        top_weights, top_idx = torch.topk(probs, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        top_weights = top_weights * self.per_expert_scale[top_idx]
        return top_weights.to(hidden_states.dtype), top_idx


class DiffusionGemmaExperts(nn.Module):
    """Routed experts: packed 3D gate_up [E, 2I, H] / down [E, H, I] with
    gelu_pytorch_tanh gating. v1 uses a gather-per-active-expert loop
    (reference semantics); the fused MoE executor replaces it once GeGLU
    activation lands in the shared executor (perf phase)."""

    def __init__(self, config: ModelConfig, weights: Dict[str, torch.Tensor]):
        super().__init__()
        self.gate_up_proj = weights[W.moe_w1]  # [E, 2I, H]
        self.down_proj = weights[W.moe_w2]  # [E, H, I]
        self.num_experts = config.expert_num
        self.inter = config.moe_inter_size // 2 * 2

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_weights: torch.Tensor,
        top_k_idx: torch.Tensor,
    ) -> torch.Tensor:
        # gate_up layout: [E, 2I, H] with gate first (loader orders it).
        E = self.num_experts
        tokens, hidden = hidden_states.shape
        flat_h = hidden_states
        final = torch.zeros_like(flat_h)
        weights_flat = top_k_weights.reshape(-1)
        idx_flat = top_k_idx.reshape(-1)
        # [tokens*K] -> per-expert grouping
        order = torch.argsort(idx_flat, stable=True)
        sorted_idx = idx_flat[order]
        sorted_h = flat_h[order // top_k_idx.shape[1]]
        sorted_w = weights_flat[order]
        boundaries = torch.searchsorted(
            sorted_idx, torch.arange(E + 1, device=sorted_idx.device)
        )
        for e in range(E):
            lo = int(boundaries[e].item())
            hi = int(boundaries[e + 1].item())
            if hi <= lo:
                continue
            rows = sorted_h[lo:hi]
            gate_up = torch.nn.functional.linear(rows, self.gate_up_proj[e])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_out = torch.nn.functional.linear(
                _gelu_tanh(gate) * up, self.down_proj[e]
            )
            expert_out = expert_out * sorted_w[lo:hi, None]
            final.index_add_(0, order[lo:hi] // top_k_idx.shape[1], expert_out.to(final.dtype))
        return final


class DiffusionGemmaLayer(nn.Module):
    """One MoE layer, mode-switchable (encoder vs decoder layer_scalar)."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        layer_type: str,
        hw_kernel_config: Optional[HWKernelConfig] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.hidden_size = config.hidden_size
        self.self_attn = DiffusionGemmaAttention(
            config, parallelism_config, weights, layer_idx, layer_type, hw_kernel_config
        )
        self.shared_mlp = DiffusionGemmaGateMLP(
            weights, W.ffn_w1, W.ffn_w3, W.ffn_w2, hw_kernel_config
        )
        self.input_layernorm = RMSNorm(weights[W.pre_ln_gamma], eps=config.layernorm_eps)
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            weights[W.dgm_pre_ffn_ln_gamma], eps=config.layernorm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            weights[W.dgm_post_ffn_ln_gamma], eps=config.layernorm_eps
        )
        self.pre_feedforward_layernorm_2 = RMSNorm(
            weights[W.dgm_pre_ffn_ln2_gamma], eps=config.layernorm_eps
        )
        self.post_feedforward_layernorm_1 = RMSNorm(
            weights[W.dgm_post_ffn_ln1_gamma], eps=config.layernorm_eps
        )
        self.post_feedforward_layernorm_2 = RMSNorm(
            weights[W.dgm_post_ffn_ln2_gamma], eps=config.layernorm_eps
        )
        self.router = DiffusionGemmaRouter(config, parallelism_config, weights, hw_kernel_config)
        self.experts = DiffusionGemmaExperts(config, weights)
        self.layer_scalar_enc = weights[W.dgm_layer_scalar_enc]
        self.layer_scalar_dec = weights[W.dgm_layer_scalar_dec]

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: DiffusionGemmaRope,
        positions: torch.Tensor,
        attn_op: DiffusionGemmaAttnOp,
        kv_cache: Optional[LayerKVCache],
        attn_inputs: Any,
        causal: bool,
        write_kv: bool,
        decoder_mode: bool,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, rope, positions, attn_op, kv_cache, attn_inputs, causal, write_kv
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.shared_mlp(hidden_states)
        hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

        hidden_states_2_for_routing = residual.reshape(-1, residual.shape[-1])
        hidden_states_2_for_experts = self.pre_feedforward_layernorm_2(
            hidden_states_2_for_routing
        )
        top_k_weights, top_k_idx = self.router(hidden_states_2_for_routing)
        hidden_states_2 = self.experts(
            hidden_states_2_for_experts, top_k_weights, top_k_idx
        )
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

        hidden_states = hidden_states_1 + hidden_states_2
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        scalar = self.layer_scalar_dec if decoder_mode else self.layer_scalar_enc
        return hidden_states * scalar


class DiffusionGemmaSelfConditioning(nn.Module):
    """Soft-embedding self-conditioning: pre_norm -> gated GeGLU -> add ->
    post RMSNorm (no scale)."""

    def __init__(
        self,
        config: ModelConfig,
        weights: Dict[str, torch.Tensor],
        hw_kernel_config: Optional[HWKernelConfig] = None,
    ):
        super().__init__()
        self.pre_norm = RMSNorm(
            weights[W.dgm_self_cond_pre_norm], eps=config.layernorm_eps
        )
        self.mlp = DiffusionGemmaGateMLP(
            weights,
            W.dgm_self_cond_gate_w,
            W.dgm_self_cond_up_w,
            W.dgm_self_cond_down_w,
            hw_kernel_config,
        )
        self.post_norm = _NoScaleRMSNorm(config.layernorm_eps)

    def forward(self, inputs_embeds, self_conditioning_signal):
        normed = self.pre_norm(self_conditioning_signal)
        sc = self.mlp(normed)
        return self.post_norm(inputs_embeds + sc)


class DiffusionGemmaModel(GptModelBase):
    """Encoder-mode backbone (causal). The proposer (decoder mode) reuses the
    layer modules with is_causal=False attention and decoder scalars."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        quant_config: Optional[object] = None,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self.embed_tokens = Embedding(
            config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.embed_scale = float(math.sqrt(config.hidden_size))
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        self.layer_types = diffusion_gemma_layer_types(config.num_layers)
        self.layers = nn.ModuleList(
            [
                DiffusionGemmaLayer(
                    config,
                    parallelism_config,
                    weights.weights[idx],
                    idx,
                    self.layer_types[idx],
                    py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )
        self_conditioning_weights = {
            W.dgm_self_cond_gate_w: weights.get_global_weight(W.dgm_self_cond_gate_w),
            W.dgm_self_cond_up_w: weights.get_global_weight(W.dgm_self_cond_up_w),
            W.dgm_self_cond_down_w: weights.get_global_weight(W.dgm_self_cond_down_w),
            W.dgm_self_cond_pre_norm: weights.get_global_weight(W.dgm_self_cond_pre_norm),
        }
        self.self_conditioning = DiffusionGemmaSelfConditioning(
            config, self_conditioning_weights, py_hw_kernel_config
        )
        embedding_weight = weights.get_global_weight(W.embedding)
        device = embedding_weight.device
        dtype = embedding_weight.dtype
        self.ropes = {
            LAYER_TYPE_SLIDING: DiffusionGemmaRope(
                config.dgm_swa_rope_theta,
                config.dgm_head_dim,
                1.0,
                config.max_seq_len,
                device,
                dtype,
            ),
            LAYER_TYPE_FULL: DiffusionGemmaRope(
                config.dgm_global_rope_theta,
                config.dgm_global_head_dim,
                config.dgm_global_partial_rope,
                config.max_seq_len,
                device,
                dtype,
            ),
        }
        self.attn_ops: Dict[str, DiffusionGemmaAttnOp] = {}
        self._init_capture_context(self._capture_canonical_layer, self._capture_canonical_final)

    def _capture_canonical_layer(self, hidden_states, residual=None):
        return hidden_states

    def _capture_canonical_final(self, hidden_states, residual=None):
        return self.norm(hidden_states)

    def get_attn_op(self, layer_type: str) -> DiffusionGemmaAttnOp:
        if layer_type not in self.attn_ops:
            cfg = self.config
            if layer_type == LAYER_TYPE_SLIDING:
                head_dim, kv_heads = cfg.dgm_head_dim, cfg.dgm_kv_head_num
            else:
                head_dim, kv_heads = cfg.dgm_global_head_dim, cfg.dgm_global_kv_head_num
            tp = self.parallelism_config.get_attn_tp_size()
            if tp > 1 and kv_heads % tp:
                kv_heads = kv_heads // math.gcd(kv_heads, tp)
            elif tp > 1:
                kv_heads = kv_heads // tp
            self.attn_ops[layer_type] = DiffusionGemmaAttnOp(
                layer_type,
                cfg.attn_config.head_num // tp,
                kv_heads,
                head_dim,
                cfg.dgm_sliding_window,
                cfg.attn_config.kernel_tokens_per_block,
                cfg.attn_config.dtype,
                None,
            )
        return self.attn_ops[layer_type]

    def _positions(self, inputs: PyModelInputs) -> torch.Tensor:
        attn = inputs.attention_inputs
        if isinstance(attn, dict):
            primary = next(iter(attn.values()))
        else:
            primary = attn
        pos = getattr(primary, "combo_position_ids", None)
        if pos is None or pos.numel() == 0:
            seq_len = getattr(primary, "sequence_lengths", None)
            total = int(inputs.input_ids.numel())
            base = 0 if seq_len is None or seq_len.numel() == 0 else 0
            pos = torch.arange(base, total, device=inputs.input_ids.device, dtype=torch.long)
        return pos.reshape(-1)

    def _select_inputs(self, inputs: PyModelInputs, layer_type: str):
        attn = inputs.attention_inputs
        tag = SWA_TAG if layer_type == LAYER_TYPE_SLIDING else GLOBAL_TAG
        if isinstance(attn, dict):
            if tag in attn:
                return attn[tag]
            return next(iter(attn.values()))
        return attn

    def _layer_kv_cache(self, layer_idx: int, layer_type: str):
        if self.kv_cache is None:
            return None
        tag = SWA_TAG if layer_type == LAYER_TYPE_SLIDING else GLOBAL_TAG
        groups = self.kv_cache.get_layer_cache_groups(layer_idx)
        for cache in groups:
            if str(cache.tag) == tag:
                return cache
        return groups[0] if groups else None

    def forward(
        self,
        inputs: PyModelInputs,
        fmha_impl: Any = None,
    ) -> PyModelOutputs:
        """Encoder-mode causal forward (prompt prefill / canvas commit)."""
        input_ids: torch.Tensor = inputs.input_ids
        text_tokens_mask = inputs.embedding_inputs.text_tokens_mask
        inputs_embeds = self.embed_tokens(
            input_ids, None, None, text_tokens_mask
        )
        hidden_states = inputs_embeds * self.embed_scale
        multimodal_inputs = getattr(inputs, "multimodal_inputs", None)
        mm_features = (
            multimodal_inputs.multimodal_features if multimodal_inputs is not None else None
        )
        mm_feature_locs = (
            multimodal_inputs.mm_features_locs if multimodal_inputs is not None else None
        )
        hidden_states = self.multimodal_embedding_injector(
            hidden_states, mm_features, mm_feature_locs
        )
        positions = self._positions(inputs)
        attn = inputs.attention_inputs
        primary = next(iter(attn.values())) if isinstance(attn, dict) else attn
        is_target_verify = bool(getattr(primary, "is_target_verify", False))
        capture = self.capture_context(inputs.capture_hidden_states)
        capture_aux_hidden = bool(self._mtp_aux_capture_layer_ids)
        if capture_aux_hidden:
            self.begin_aux_hidden_capture(hidden_states, is_target_verify)
        for layer_type in self.ropes:
            self.get_attn_op(layer_type).plan(
                self._select_inputs(inputs, layer_type), causal=True
            )
        for i, layer in enumerate(self.layers[: self.layer_num]):
            layer_type = self.layer_types[i]
            attn_inputs = self._select_inputs(inputs, layer_type)
            attn_op = self.get_attn_op(layer_type)
            hidden_states = layer(
                hidden_states,
                self.ropes[layer_type],
                positions,
                attn_op,
                self._layer_kv_cache(i, layer_type),
                attn_inputs,
                causal=True,
                write_kv=True,
                decoder_mode=False,
            )
            if i in self._mtp_aux_capture_layer_id_set:
                self.capture_aux_hidden(i, hidden_states)
            capture.capture_layer(i, hidden_states)
        if capture_aux_hidden:
            self.finish_aux_hidden_capture()
        outputs = capture.finalize(self.norm(hidden_states))
        if capture_aux_hidden:
            outputs.mtp_target_hidden_states = self._mtp_target_hidden_states
        return outputs

    def forward_canvas(
        self,
        inputs: PyModelInputs,
        canvas_tokens: torch.Tensor,
        canvas_positions: torch.Tensor,
        self_conditioning_logits: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Decoder-mode bidirectional canvas forward. Returns final hidden
        states [B*canvas, hidden]; caller applies lm_head + softcap."""
        batch = canvas_tokens.shape[0]
        embeds = self.embed_tokens(canvas_tokens.reshape(-1)) * self.embed_scale
        embeds = embeds.view(batch, -1, embeds.shape[-1])
        if self_conditioning_logits is not None:
            weight = self.embed_tokens.weight
            soft = torch.matmul(
                self_conditioning_logits.to(torch.float32).softmax(dim=-1).to(weight.dtype),
                weight,
            ) * self.embed_scale
            embeds = self.self_conditioning(embeds, soft)
        else:
            zeros = torch.zeros_like(embeds)
            embeds = self.self_conditioning(embeds, zeros)
        hidden_states = embeds.reshape(-1, embeds.shape[-1])
        attn = inputs.attention_inputs
        primary = next(iter(attn.values())) if isinstance(attn, dict) else attn
        for layer_type in self.ropes:
            self.get_attn_op(layer_type).plan(
                self._select_inputs(inputs, layer_type), causal=False
            )
        for i, layer in enumerate(self.layers[: self.layer_num]):
            layer_type = self.layer_types[i]
            attn_inputs = self._select_inputs(inputs, layer_type)
            attn_op = self.get_attn_op(layer_type)
            hidden_states = layer(
                hidden_states,
                self.ropes[layer_type],
                canvas_positions.reshape(-1),
                attn_op,
                self._layer_kv_cache(i, layer_type),
                attn_inputs,
                causal=False,
                write_kv=False,
                decoder_mode=True,
            )
        return self.norm(hidden_states)


__all__ = [
    "DiffusionGemmaModel",
    "DiffusionGemmaLayer",
    "SWA_TAG",
    "GLOBAL_TAG",
    "LAYER_TYPE_SLIDING",
    "LAYER_TYPE_FULL",
    "diffusion_gemma_layer_types",
]

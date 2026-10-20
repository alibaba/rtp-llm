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
import os
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

# opt r4 lane A debug A/B gates (default off). When set, the fused routing
# kernel and the flashinfer no-scale RMSNorm additionally compute the torch
# reference and print per-call comparison stats ([dgm-routing-ab] /
# [dgm-norm-ab]) so the existing GPU bazel lanes double as numerical A/B
# harnesses without touching the test files (opt_r4_laneA logs).
_ROUTING_AB_CHECK = os.environ.get("DGM_ROUTING_AB", "") not in ("", "0")
_NORM_AB_CHECK = os.environ.get("DGM_NORM_AB", "") not in ("", "0")

# opt r5 ⑥-clearing lane, item 1: fused residual-add + RMSNorm. The two
# in-layer "sum = residual + x; normed = RMSNorm(sum)" pairs (post-attention
# add feeding pre_feedforward_layernorm, and the h1+h2 add feeding
# post_feedforward_layernorm) run as ONE flashinfer.triton.norm
# rms_norm_add_residual launch: the 42.7 ms/window torch add cluster plus
# the paired norm read disappear into the fused kernel. The op updates the
# residual IN PLACE (it becomes x + residual) and writes the normed output
# to a fresh tensor; value contract: fp32 accumulation on both paths with
# the sum rounded to bf16 at the store exactly like the torch add, so the
# pair agrees within bf16 rounding noise (<= 1 ulp class, the
# dsv4/dsv41 rmsnorm-replace precedent; the residual's rsqrt uses the
# unrounded fp32 sum, the one allowed deviation). DGM_FUSED_ADD_NORM=0
# forces the torch two-step; DGM_ADDNORM_AB=1 additionally prints the
# fused-vs-torch comparison stats ([dgm-addnorm-ab]).
_FUSED_ADD_NORM_ENV = os.environ.get("DGM_FUSED_ADD_NORM", "1") != "0"
_ADDNORM_AB_CHECK = os.environ.get("DGM_ADDNORM_AB", "") not in ("", "0")
_FI_ADD_NORM = None
try:
    # explicit submodule import: flashinfer.triton's package __init__ does
    # not re-export .norm, so plain attribute access would miss it.
    from flashinfer.triton.norm import rms_norm_add_residual as _FI_ADD_NORM
except Exception as _addnorm_error:  # CPU-only / no flashinfer
    _FI_ADD_NORM = None
    logger.info(
        "DiffusionGemma flashinfer fused add+rmsnorm unavailable, torch path: %s",
        _addnorm_error,
    )
_addnorm_warned = False
_addnorm_ab_logged: set = set()


def _fused_add_norm_usable(x: torch.Tensor, residual: torch.Tensor, norm) -> bool:
    if not (_FUSED_ADD_NORM_ENV and _FI_ADD_NORM is not None):
        return False
    weight = getattr(norm, "weight", None)
    if weight is None or weight.dim() != 1 or weight.stride(0) != 1:
        return False
    # test harnesses patch layers with mock norm modules that do not carry
    # variance_epsilon; those must take the torch two-step path (bit-equal
    # to the pre-change behavior) instead of crashing in _fused_add_norm.
    if not hasattr(norm, "variance_epsilon"):
        return False
    return (
        x.is_cuda
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and residual.dtype == torch.bfloat16
        and x.shape == residual.shape
        and x.stride() == residual.stride()
        and x.stride(1) == 1
        # the op writes residual in place: x and residual must be distinct
        # storages or the fused read/write would alias.
        and x.data_ptr() != residual.data_ptr()
        and x.numel() > 0
    )


def _addnorm_ab(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out_normed: torch.Tensor,
    out_residual: torch.Tensor,
) -> None:
    """Debug A/B (DGM_ADDNORM_AB=1): fused add+rmsnorm vs the exact torch
    two-step the layer ran before (bf16 add, then the production
    flashinfer.norm.rmsnorm), logged once per shape."""
    key = (tuple(x.shape), x.dtype)
    if key in _addnorm_ab_logged:
        return
    _addnorm_ab_logged.add(key)
    try:
        import flashinfer

        ref_sum = residual + x
        ref_normed = flashinfer.norm.rmsnorm(ref_sum, weight, eps=eps)
        diff = (out_normed.to(torch.float32) - ref_normed.to(torch.float32)).abs()
        # bf16 bit-pattern distance: for same-sign finite values the int16
        # view distance counts ulps directly.
        ulp = (
            out_normed.view(torch.int16).to(torch.int32)
            - ref_normed.view(torch.int16).to(torch.int32)
        ).abs()
        mismatch = out_normed != ref_normed
        print(
            f"[dgm-addnorm-ab] shape={tuple(x.shape)} dtype={x.dtype} "
            f"max_diff={float(diff.max()):.3e} max_ulp={int(ulp.max())} "
            f"mismatch_frac={float(mismatch.float().mean()):.4f} "
            f"residual_equal={bool(torch.equal(out_residual, ref_sum))} "
            f"ref_absmax={float(ref_normed.to(torch.float32).abs().max()):.3f}",
            flush=True,
        )
    except Exception as error:
        logger.warning("DiffusionGemma add-norm A/B check failed: %s", error)


def _fused_add_norm(x: torch.Tensor, residual: torch.Tensor, norm):
    """(normed, summed): one fused residual-add + RMSNorm launch on CUDA bf16,
    the torch two-step otherwise.

    Returns ``norm(sum + x)`` and the sum itself. On the fused path the sum
    is written IN PLACE into ``residual`` (callers must pass a private
    tensor; the layer chain's residual tensors are always the previous
    op's fresh output). Any flashinfer/triton failure degrades to the torch
    two-step for the rest of the process (r4 norm contract).
    """
    global _addnorm_warned
    if _fused_add_norm_usable(x, residual, norm):
        weight = norm.weight
        eps = float(norm.variance_epsilon)
        out = torch.empty_like(x)
        # clone the inputs BEFORE the in-place launch for the A/B referee
        x_ref = x.clone() if _ADDNORM_AB_CHECK else None
        res_ref = residual.clone() if _ADDNORM_AB_CHECK else None
        try:
            _FI_ADD_NORM(x, residual, weight, eps, x_out=out)
            if _ADDNORM_AB_CHECK:
                _addnorm_ab(x_ref, res_ref, weight, eps, out, residual)
            return out, residual
        except Exception as error:
            if not _addnorm_warned:
                logger.warning(
                    "DiffusionGemma fused add+rmsnorm failed (%s); torch path "
                    "for this process",
                    error,
                )
                _addnorm_warned = True
    summed = residual + x
    return norm(summed), summed


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
        # rope apply() memo: the positions tensor seen last (object identity)
        # and its host-side max — see apply().
        self._pos_memo_ref: Optional[torch.Tensor] = None
        self._pos_memo_max: int = 0
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
        # positions.max().item() is a per-layer device sync, but every layer
        # in one forward receives the SAME positions tensor object — pay the
        # sync once per distinct tensor (memo on object identity) and reuse
        # the host-side max for the rest of the forward. Same values reach
        # _build_cache, zero semantic change.
        if positions is not self._pos_memo_ref:
            self._pos_memo_ref = positions
            self._pos_memo_max = int(positions.max().item()) if positions.numel() else 0
        need = self._pos_memo_max + 1 if positions.numel() else 1
        self._build_cache(need)
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
        # The paged KV write op only needs the flashinfer.page import, so it
        # is independent of the attention wrappers below: global layers
        # (head_dim 512 -> attention stays on the torch path) can still write
        # KV through the vectorized append kernel instead of the per-row
        # python loop, as long as plan() fills the block-table metadata.
        self.use_paged_write = False
        self.fmha_params = None
        try:
            from rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op import (
                KVCacheWriteOp,
            )

            self.write_op = KVCacheWriteOp(
                num_kv_heads=self.kv_head_num,
                head_size=self.head_dim,
                token_per_block=self.page_size,
            )
            self.use_paged_write = True
        except Exception as error:  # CPU-only environments
            logger.info("DiffusionGemma paged KV write falls back to torch: %s", error)
        try:
            from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
                get_py_flashinfer_workspace_buffer,
                release_py_flashinfer_workspace_buffer,
            )

            # flashinfer wrappers + the shared workspace buffer must be
            # normal tensors: the proposer builds its main model lazily
            # inside @inference_mode, and buffers allocated there poison
            # every later normal-mode plan/forward (inference tensors
            # reject inplace updates outside InferenceMode).
            with torch.inference_mode(False):
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
            self.use_flashinfer = True
        except Exception as error:  # CPU-only environments
            logger.info("DiffusionGemma attention falls back to torch: %s", error)
        if self.use_flashinfer and self.head_dim > 256:
            # flashinfer 0.6.9 ragged prefill rejects the global layers'
            # GQA config (head_dim 512, kv 2: "Invalid configuration
            # NUM_MMA_D_QK=32 ..."); run them on the verified torch path.
            # TODO(step-5): evaluate flashinfer fa2/trtllm-gen backends for
            # this shape before the zero-native-torch gate.
            logger.info(
                "DiffusionGemma attention (head_dim=%d) uses the torch path",
                self.head_dim,
            )
            self.use_flashinfer = False
        # Stage-level flashinfer diagnostics (bring-up): DGM_FI_DIAG=1
        # synchronizes after every GPU stage and labels it, so an async
        # illegal access surfaces at the kernel that caused it instead of
        # the next cublas GEMM.
        self._fi_diag = bool(os.environ.get("DGM_FI_DIAG", ""))
        self._fi_diag_first = True

    def __del__(self):
        if self.use_flashinfer:
            try:
                self._release_ws(self.g_workspace_buffer)
            except Exception:
                pass

    def _window_left(self) -> int:
        # flashinfer 0.6.9 ragged/paged prefill only compiles the sliding
        # window branch for window_left >= 0 (prefill.py use_sliding_window);
        # a negative value (the HF inclusive-lower-bound convention used
        # here historically) falls into the NO-window branch and the kernel
        # silently runs full causal — verified by the window A/B test
        # (test_diffusion_gemm_window_ab_prefill: both arms bit-identical,
        # W-arm matches the full-causal reference). Pass the positive window
        # (window-1: query attends kv positions >= q - (window-1), exactly
        # the HF [q-window+1, q] mask). ctx < window is unaffected (the
        # window never bites), so all existing goldens stay bit-stable.
        return (self.window - 1) if self.window > 0 else -1

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
        prefix_cu: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """CPU reference attention: [tokens, heads, dim] layout, per-request
        masked softmax. Supports bidirectional canvas + windowed prefix and
        GQA (kv heads repeated to the query head count).

        prefix_k/prefix_v are the batch-concatenated prefix rows; request b's
        rows live at [prefix_cu[b], prefix_cu[b]+plen) — NOT at the query
        concat offset q_lo (equal only for a single request)."""
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
                p_lo = 0 if prefix_cu is None else int(prefix_cu[b].item())
                keys.append(prefix_k[p_lo : p_lo + plen])
                values.append(prefix_v[p_lo : p_lo + plen])
            keys.append(k[q_lo:q_hi])
            values.append(v[q_lo:q_hi])
            key = torch.cat(keys, dim=0)
            value = torch.cat(values, dim=0)
            score = torch.einsum("qhd,khd->hqk", q[q_lo:q_hi], key) * self.sm_scale
            q_pos = torch.arange(q_hi - q_lo, device=q.device)
            k_pos = torch.cat(
                [torch.arange(plen, device=q.device), torch.arange(q_hi - q_lo, device=q.device) + plen]
            )
            allowed = torch.ones(
                q_hi - q_lo, key.shape[0], dtype=torch.bool, device=q.device
            )
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
        """Concatenate each request's prefix rows [plen, H, D] in batch order.

        Pure-torch vectorization of the former per-request python loop: one
        advanced-indexing gather per channel over the (b, row) grid picks
        paged[bt[b, row//P], ch, :, row%P, :], and a boolean mask keeps
        rows < plen_b. Row order and values are identical to the loop
        (verified bit-exact); max_plen needs the single .item() sync the
        loop's first line already had."""
        paged = self._reshape_paged(kv_cache.kv_cache_base)
        block_table = attn_inputs.kv_cache_kernel_block_id
        if block_table is None or block_table.numel() == 0:
            block_table = attn_inputs.kv_cache_kernel_block_id_device
        if block_table is None or block_table.numel() == 0:
            return None, None
        # normalize devices: the block table / prefix lens can arrive on
        # host while the paged cache is on device (the loop used per-row
        # .item() reads and never compared them directly).
        dev = paged.device
        block_table = block_table.reshape(batch_size, -1).to(device=dev, dtype=torch.long)
        max_plen = int(prefix_lens.max().item())
        rows = torch.arange(max_plen, device=dev)
        page_cols = rows // self.page_size  # [max_plen] column into block_table
        slots = rows % self.page_size  # [max_plen] slot within the page
        pages = block_table[:, page_cols]  # [B, max_plen]
        k_rows = paged[pages, 0, :, slots, :]  # [B, max_plen, H, D]
        v_rows = paged[pages, 1, :, slots, :]
        plens = prefix_lens.reshape(-1, 1).to(device=dev, dtype=torch.long)
        mask = rows.reshape(1, -1) < plens
        return k_rows[mask], v_rows[mask]

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
        # prefix rows are batch-concatenated by _gather_prefix; per-request
        # offsets = exclusive cumsum of prefix_lens (0 for b=0)
        if batch_size > 0:
            prefix_cu = torch.zeros(
                batch_size, dtype=prefix_lens.dtype, device=prefix_lens.device
            )
            if batch_size > 1:
                prefix_cu[1:] = torch.cumsum(prefix_lens[: batch_size - 1], dim=0)
        else:
            prefix_cu = None

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
                q, k, v, prefix_k, prefix_v, causal, prefix_lens, qo_indptr, prefix_cu
            )
            if write_kv and kv_cache is not None:
                self._write_cache(
                    k, v, kv_cache, attn_inputs, positions, prefix_lens, batch_size, qo_indptr
                )
            return out

        # The flashinfer wrappers inplace-update internal buffers in both
        # plan() and run(); run everything under inference_mode(False) so
        # buffers stay normal tensors regardless of the caller's mode
        # (propose runs under @inference_mode, the reference/engine paths
        # run under no_grad — a mode mix poisons the shared buffers).
        with torch.inference_mode(False):
            return self._forward_flashinfer(q, k, v, kv_cache, attn_inputs, positions, causal, write_kv)

    def _forward_flashinfer(
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
        qd = q
        kd = k
        vd = v
        if self._fi_diag and self._fi_diag_first:
            self._fi_diag_first = False
            block_table = attn_inputs.kv_cache_kernel_block_id
            if block_table is None or block_table.numel() == 0:
                block_table = attn_inputs.kv_cache_kernel_block_id_device
            logger.info(
                "[dgm-fi] first flashinfer call: q=%s k=%s cu=%s prefix=%s "
                "input=%s causal=%s write=%s bt=%s",
                tuple(q.shape),
                tuple(k.shape),
                qo_indptr.tolist(),
                prefix_lens.tolist() if prefix_lens is not None else None,
                attn_inputs.input_lengths.tolist()
                if attn_inputs.input_lengths is not None
                else None,
                causal,
                write_kv,
                tuple(block_table.shape) if block_table is not None else None,
            )
        if self._fi_diag:
            torch.cuda.synchronize()
        new_out, new_lse = self.ragged_wrapper.run(
            qd, kd, vd,
            return_lse=True,
        )
        if self._fi_diag:
            torch.cuda.synchronize()
            logger.info("[dgm-fi] ragged run ok qo=%d causal=%s", q.shape[0], causal)
        if write_kv and kv_cache is not None:
            self._write_cache_flashinfer(kd, vd, kv_cache, attn_inputs, positions)
            if self._fi_diag:
                torch.cuda.synchronize()
                logger.info("[dgm-fi] paged KV write ok qo=%d", q.shape[0])
        if has_prefix:
            paged = self._reshape_paged(kv_cache.kv_cache_base)
            if self._fi_diag:
                if getattr(self, "_fi_read_diag_first", True):
                    self._fi_read_diag_first = False
                    reuse_pages = (
                        self.fmha_params.reuse_cache_page_indice_d.tolist()
                        if getattr(self, "fmha_params", None) is not None
                        else []
                    )
                    if reuse_pages:
                        first = paged[reuse_pages[0]]
                        k_head = first[0, 0, :4, :4].float()
                        v_head = first[1, 0, :64, 0].float()
                        k_row_std = float(
                            first[0, 0].float().std(dim=0).mean().item()
                        )
                        v_row_std = float(
                            first[1, 0].float().std(dim=0).mean().item()
                        )
                        logger.info(
                            "[dgm-fi] prefix read probe: pages=%s | K "
                            "absmax=%.4f k_row_std=%.4f | V absmax=%.4f "
                            "v_row_std=%.4f V[:8]=%s | q_absmax=%.3f",
                            reuse_pages[:8],
                            float(first[0].abs().max().item()),
                            k_row_std,
                            float(first[1].abs().max().item()),
                            v_row_std,
                            v_head[:8].tolist(),
                            float(qd.abs().max().item()),
                        )
                torch.cuda.synchronize()
            prefix_out, prefix_lse = self.paged_wrapper.run(
                qd, paged,
                return_lse=True,
            )
            if self._fi_diag:
                torch.cuda.synchronize()
                self._fi_merge_diag_count = getattr(self, "_fi_merge_diag_count", 0) + 1
                if self._fi_merge_diag_count % 100 == 1:
                    new_row_std = float(
                        new_out.reshape(-1, q.shape[0], new_out.shape[-1])
                        .float()
                        .std(dim=1)
                        .mean()
                        .item()
                    )
                    prefix_row_std = float(
                        prefix_out.reshape(-1, q.shape[0], prefix_out.shape[-1])
                        .float()
                        .std(dim=1)
                        .mean()
                        .item()
                    )
                    logger.info(
                        "[dgm-fi] merge probe #%d: qo=%d causal=%s | "
                        "new_lse[0,:3]=%s new_out_absmax=%.4f "
                        "new_row_std=%.4f | prefix_lse[0,:3]=%s "
                        "prefix_out_absmax=%.4f prefix_row_std=%.4f",
                        self._fi_merge_diag_count,
                        q.shape[0],
                        causal,
                        new_lse[0, :3].tolist(),
                        float(new_out.abs().max().item()),
                        new_row_std,
                        prefix_lse[0, :3].tolist(),
                        float(prefix_out.abs().max().item()),
                        prefix_row_std,
                    )
                logger.info("[dgm-fi] paged prefix run ok qo=%d", q.shape[0])
            self.merge_state_in_place(new_out, new_lse, prefix_out, prefix_lse)
            if self._fi_diag:
                torch.cuda.synchronize()
                logger.info("[dgm-fi] merge_state ok qo=%d", q.shape[0])
        return new_out

    def plan(self, attn_inputs: Any, causal: bool) -> None:
        """Plan flashinfer wrappers for this step (ragged + paged prefix).

        flashinfer's plan() inplace-updates internal wrapper buffers; if the
        first call runs under torch.inference_mode (the proposer's
        @inference_mode decorator), those buffers become inference tensors
        and every later normal-mode plan dies with "Inplace update to
        inference tensor outside InferenceMode". Run the planning (and the
        lazy wrapper construction) under inference_mode(False) so the
        buffers stay normal tensors for both calling contexts.

        With use_paged_write, only the wrapper-free segment runs: the
        FlashInferMlaAttnParams fill + write_op.set_params + block table
        preparation, which the torch-attention layers need for the paged
        KV write. fill_params queries the CUDA device count, so the
        write-only plan requires a real device (CPU/stub-driver UTs keep
        the early return and the per-row write fallback).
        """
        if not (
            self.use_flashinfer
            or (self.use_paged_write and torch.cuda.is_available())
        ):
            return
        with torch.inference_mode(False):
            self._plan_locked(attn_inputs, causal)

    def _plan_locked(self, attn_inputs: Any, causal: bool) -> None:
        batch_size = int(attn_inputs.input_lengths.numel())
        qo_indptr = attn_inputs.cu_seqlens_device
        if qo_indptr is None or qo_indptr.numel() == 0:
            qo_indptr = attn_inputs.cu_seqlens
        qo_indptr = qo_indptr[: batch_size + 1]
        if self.use_flashinfer:
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
        if self.use_paged_write:
            self.write_op.set_params(self.fmha_params)
        if int(prefix_lens.max().item()) <= 0:
            return
        reuse_info = self.fmha_params.batch_reuse_info_vec_h
        prefix_lengths = reuse_info[:, 1]
        paged_kv_indptr = torch.empty(batch_size + 1, dtype=torch.int32, device="cpu")
        paged_kv_indptr[:-1].copy_(reuse_info[:, 2])
        paged_kv_indptr[-1] = self.fmha_params.reuse_cache_page_indice_h.numel()
        paged_kv_last_page_len = (prefix_lengths - 1) % self.page_size + 1
        if self._fi_diag and int(prefix_lengths.max().item()) > 0:
            block_table = attn_inputs.kv_cache_kernel_block_id
            if block_table is None or block_table.numel() == 0:
                block_table = attn_inputs.kv_cache_kernel_block_id_device
            logger.info(
                "[dgm-fi] paged plan: prefix=%s input=%s seq=%s | "
                "reuse_pages=%s indptr=%s lpl=%s | qo_indptr=%s | bt=%s",
                prefix_lengths.tolist(),
                attn_inputs.input_lengths.tolist()
                if attn_inputs.input_lengths is not None
                else None,
                seq_lens.tolist() if seq_lens is not None else None,
                self.fmha_params.reuse_cache_page_indice_h.tolist(),
                paged_kv_indptr.tolist(),
                paged_kv_last_page_len.tolist(),
                qo_indptr.tolist() if qo_indptr is not None else None,
                block_table.tolist()[:20] if block_table is not None else None,
            )
        if not self.use_flashinfer:
            # write-only plan: the block-table/fill_params segment above is
            # all the paged KV write needs; skip the paged attention plan.
            return
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
        if self.use_paged_write and getattr(self, "fmha_params", None) is not None:
            self.write_op.set_params(self.fmha_params)
        if self._fi_diag and getattr(self, "_fi_write_diag_first", True):
            self._fi_write_diag_first = False
            base = kv_cache.kv_cache_base
            params = self.fmha_params
            block_table = attn_inputs.kv_cache_kernel_block_id
            if block_table is None or block_table.numel() == 0:
                block_table = attn_inputs.kv_cache_kernel_block_id_device
            logger.info(
                "[dgm-fi] write geometry: raw=%s stride=%s dtype=%s | "
                "reshaped=%s | bt=%s | page_indice=%s indptr=%s lpl=%s | "
                "batch_ind=%s positions[:6]=%s pos_max=%s | k=%s",
                tuple(base.shape),
                tuple(base.stride()),
                base.dtype,
                tuple(self._reshape_paged(base).shape),
                block_table.tolist() if block_table is not None else None,
                params.page_indice_d.tolist() if params is not None else None,
                params.decode_page_indptr_d.tolist() if params is not None else None,
                params.paged_kv_last_page_len_d.tolist() if params is not None else None,
                tuple(params.batch_indice_d.shape) if params is not None else None,
                params.positions_d[:6].tolist() if params is not None else None,
                int(params.positions_d.max().item()) if params is not None else None,
                (tuple(k.shape), str(k.dtype), tuple(k.stride())),
            )
        # SWA group layer caches arrive as the raw 2-D pool buffer
        # (makeLayerCache only expands FULL groups to 5-D); KVCacheWriteOp
        # indexes [:, 0, :, :, :], so reshape the per-call view first.
        kv_cache.kv_cache_base = self._reshape_paged(kv_cache.kv_cache_base)
        self.write_op.forward(k, v, kv_cache)

    def _write_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: LayerKVCache,
        attn_inputs: Any,
        positions: torch.Tensor,
        prefix_lens: torch.Tensor,
        batch_size: int,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> None:
        """Dispatch the KV write on the torch-attention path: prefer the
        paged write op (same kernel the flashinfer path uses — see
        _write_cache_flashinfer), fall back to the reference per-row loop
        when the plan() metadata is missing or dtypes would trip the op's
        strict checks."""
        if self.use_paged_write and k.is_cuda:
            cache = self._reshape_paged(kv_cache.kv_cache_base)
            if (
                getattr(self, "fmha_params", None) is not None
                and k.dtype == cache.dtype
                and v.dtype == cache.dtype
            ):
                with torch.inference_mode(False):
                    # 2-D SWA pool buffers need the 5-D view the op indexes
                    # with [:, 0, :, :, :] (same as _write_cache_flashinfer).
                    kv_cache.kv_cache_base = cache
                    self.write_op.forward(k, v, kv_cache)
                return
        self._write_cache_torch(
            k, v, kv_cache, attn_inputs, prefix_lens, batch_size, cu_seqlens
        )

    def _write_cache_torch(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: LayerKVCache,
        attn_inputs: Any,
        prefix_lens: torch.Tensor,
        batch_size: int,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> None:
        """Write k/v rows [T, H, D] into the paged cache via the block table.

        Spans come from cu_seqlens — requests in a batched propose can have
        unequal spans, so k.shape[0] // batch_size is not a valid row count."""
        cache = self._reshape_paged(kv_cache.kv_cache_base)
        block_table = attn_inputs.kv_cache_kernel_block_id
        if block_table is None or block_table.numel() == 0:
            block_table = attn_inputs.kv_cache_kernel_block_id_device
        if block_table is None or block_table.numel() == 0:
            return
        block_table = block_table.reshape(batch_size, -1)
        pos = 0
        for b in range(batch_size):
            plen = int(prefix_lens[b].item())
            span = k.shape[0] if cu_seqlens is None else int(cu_seqlens[b + 1].item()) - int(cu_seqlens[b].item())
            for r in range(span):
                gpos = plen + r
                page = int(block_table[b, gpos // self.page_size].item())
                slot = gpos % self.page_size
                cache[page, 0, :, slot, :] = k[pos + r].to(cache.dtype)
                cache[page, 1, :, slot, :] = v[pos + r].to(cache.dtype)
            pos += span


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
    """RMSNorm without weight (reference v_norm / router pre-norm).

    CUDA bf16 contiguous inputs run flashinfer.norm.rmsnorm with a ones
    gamma (opt r4 lane A, contract #7): the same RMSNormKernel the
    production RMSNorm modules use, with gamma=1 — one kernel instead of
    the torch cast/pow/mean/rsqrt/mul chain. DGM_FI_NORM=0 forces the
    torch reference; any flashinfer failure degrades to it with a logged
    warning. Value contract: fp32 accumulation on both sides, agreement
    within bf16 rounding noise (dsv4/dsv41 rmsnorm-replace precedent)."""

    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps
        self.fi_enabled = os.environ.get("DGM_FI_NORM", "1") != "0"
        self._fi_norm = None
        if self.fi_enabled:
            try:
                import flashinfer

                self._fi_norm = flashinfer.norm.rmsnorm
            except Exception as error:  # CPU-only / no flashinfer
                logger.info(
                    "DiffusionGemma flashinfer no-scale RMSNorm unavailable, torch path: %s",
                    error,
                )
        self._fi_warned = False
        self._ones_cache: Dict[Tuple[int, str, int], torch.Tensor] = {}
        self._ab_logged: set = set()

    def _torch_forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps)).to(x.dtype)

    def _fi_usable(self, x: torch.Tensor) -> bool:
        return (
            self._fi_norm is not None
            and x.is_cuda
            and x.dim() >= 2
            and x.numel() > 0
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
        )

    def _ones_gamma(self, x: torch.Tensor) -> torch.Tensor:
        key = (x.shape[-1], str(x.dtype), x.device.index)
        ones = self._ones_cache.get(key)
        if ones is None:
            # Built outside inference mode: like the flashinfer workspace
            # buffers (bring-up lesson), an inference-tensor gamma would
            # leak inference-mode restrictions into later normal-mode reads.
            with torch.inference_mode(False):
                ones = torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
            self._ones_cache[key] = ones
        return ones

    def _norm_ab(self, x: torch.Tensor, out: torch.Tensor) -> None:
        """Debug A/B (DGM_NORM_AB=1): flashinfer vs torch reference, logged
        once per (shape, dtype)."""
        key = (tuple(x.shape), x.dtype)
        if key in self._ab_logged:
            return
        self._ab_logged.add(key)
        try:
            ref = self._torch_forward(x)
            diff = (out.to(torch.float32) - ref.to(torch.float32)).abs()
            print(
                f"[dgm-norm-ab] shape={tuple(x.shape)} dtype={x.dtype} "
                f"max_diff={float(diff.max()):.3e} "
                f"mean_diff={float(diff.mean()):.3e} "
                f"ref_absmax={float(ref.to(torch.float32).abs().max()):.3f}",
                flush=True,
            )
        except Exception as error:
            logger.warning("DiffusionGemma norm A/B check failed: %s", error)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fi_usable(x):
            try:
                flat = x.view(-1, x.shape[-1])
                out = self._fi_norm(flat, self._ones_gamma(x), eps=self.eps).view(x.shape)
                if _NORM_AB_CHECK:
                    self._norm_ab(x, out)
                return out
            except Exception as error:
                # Degrade to the torch reference for the rest of the process.
                if not self._fi_warned:
                    logger.warning(
                        "DiffusionGemma flashinfer no-scale RMSNorm failed (%s); "
                        "torch path for this process",
                        error,
                    )
                    self._fi_warned = True
        return self._torch_forward(x)


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


_ROUTING_TOPK_KERNEL = None
_ROUTING_E_LIMIT = 4096  # single-block lane budget for the fused epilogue


def _get_routing_topk_kernel():
    """Lazily define and cache the fused router-epilogue Triton kernel.

    The definition is deferred instead of a module-level ``@triton.jit`` so
    CPU-only sandboxes without triton never pay the import (same lazy
    discipline as DiffusionGemmaExperts' fused-MoE imports). The kernel
    replaces the torch router epilogue chain

      probs = softmax(scores.float())        # cast + softmax pass
      topk(probs, k)                         # radixSort/sbtopk + gather
      top_w / top_w.sum(-1)                  # reduce + div
      top_w * per_expert_scale[top_idx]      # gather + mul
      .to(bf16)                              # cast

    with a single launch: fp32 softmax, iterative-argmax top-k (descending
    order, lower-index tie-break — CUDA topk's gather order for exactly
    equal values), renormalization over the top-k and the per-expert-scale
    product, all in fp32 registers, cast to the output dtype only at the
    store. The selected probs/ids stay in registers between the two phases
    (no intermediate global round-trip), so the only rounding vs the torch
    chain is the fp32 softmax/topk implementation noise. In-repo kernel
    precedent: triton_kernels/moe/gate_fused.py (same per-row structure).
    """
    global _ROUTING_TOPK_KERNEL
    if _ROUTING_TOPK_KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit(do_not_specialize=["num_tokens"])
        def _softmax_topk_renorm_scale_kernel(
            scores_ptr,  # [M, E] router logits, row stride stride_scores_m
            per_expert_scale_ptr,  # [E] per-expert scales
            out_idx_ptr,  # [M, K] int64 (torch.topk index dtype)
            out_w_ptr,  # [M, K] output dtype (hidden dtype)
            num_tokens,
            stride_scores_m,
            E: tl.constexpr,
            K: tl.constexpr,
            BLOCK_E: tl.constexpr,  # >= E, power of two
            BLOCK_K: tl.constexpr,  # >= K, power of two
        ):
            row = tl.program_id(axis=0).to(tl.int64)
            if row >= num_tokens:
                return
            offs = tl.arange(0, BLOCK_E)
            lane = offs < E
            logits = tl.load(
                scores_ptr + row * stride_scores_m + offs,
                mask=lane,
                other=float("-inf"),
            ).to(tl.float32)
            # fp32 softmax, centered like torch.softmax(scores.float(), -1)
            row_max = tl.max(logits, axis=0)
            e = tl.exp(logits - row_max)
            e = tl.where(lane, e, 0.0)
            probs = e / tl.sum(e, axis=0)

            # Iterative argmax top-k: descending order, leftmost index on
            # exact ties. The k selections accumulate in BLOCK_K registers.
            k_offs = tl.arange(0, BLOCK_K)
            k_lane = k_offs < K
            top_p = tl.zeros([BLOCK_K], dtype=tl.float32)
            top_id = tl.zeros([BLOCK_K], dtype=tl.int32)
            cur = tl.where(lane, probs, -float("inf"))
            for k in tl.static_range(K):
                idx = tl.argmax(cur, axis=0)  # int32 scalar
                sel = tl.sum(tl.where(offs == idx, probs, 0.0), axis=0)
                top_p = tl.where(k_offs == k, sel, top_p)
                top_id = tl.where(k_offs == k, idx, top_id)
                cur = tl.where(offs == idx, -float("inf"), cur)

            # Renormalize over the top-k, apply the per-expert scale, cast
            # once at the store (identical algebra to the torch chain).
            pes = tl.load(
                per_expert_scale_ptr + top_id, mask=k_lane, other=0.0
            ).to(tl.float32)
            w = top_p / tl.sum(top_p, axis=0) * pes
            tl.store(
                out_w_ptr + row * K + k_offs,
                w.to(out_w_ptr.dtype.element_ty),
                mask=k_lane,
            )
            tl.store(out_idx_ptr + row * K + k_offs, top_id.to(tl.int64), mask=k_lane)

        _ROUTING_TOPK_KERNEL = _softmax_topk_renorm_scale_kernel
    return _ROUTING_TOPK_KERNEL


class DiffusionGemmaRouter(nn.Module):
    """Raw-residual router: no-scale RMSNorm -> *scale*hidden^-0.5 -> proj ->
    fp32 softmax -> top8 -> renormalize -> *per_expert_scale.

    The softmax -> top8 -> renormalize -> per_expert_scale epilogue runs as
    one fused Triton kernel on CUDA (opt r4 lane A): the torch chain behind
    it was the top-8 radixSort/sbtopk + gather + reduce cluster of the
    residual native-time list. DGM_FUSED_ROUTING=0 forces the torch
    reference path. Value contract: fp32 softmax and fp32 top-k
    renormalization on both sides, so weights agree modulo fp32
    implementation noise and top-8 selection flips only on near-ties
    (bf16-noise class, the K122 ruling; tie-break is lower-index on both
    paths)."""

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
        self.fused_routing = os.environ.get("DGM_FUSED_ROUTING", "1") != "0"
        self._routing_kernel = None
        self._routing_warned = False
        if self.fused_routing:
            try:
                self._routing_kernel = _get_routing_topk_kernel()
            except Exception as error:  # CPU-only sandboxes without triton
                logger.info(
                    "DiffusionGemma fused routing unavailable, torch path: %s", error
                )

    def _fused_routing_usable(self, scores: torch.Tensor) -> bool:
        if self._routing_kernel is None:
            return False
        if not (scores.is_cuda and scores.dim() == 2 and scores.numel() > 0):
            return False
        if scores.stride(-1) != 1 or self.per_expert_scale.stride(0) != 1:
            return False
        if self.per_expert_scale.device != scores.device:
            return False
        if not (0 < self.top_k <= self.num_experts <= _ROUTING_E_LIMIT):
            return False
        if scores.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return False
        if self.per_expert_scale.dtype not in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ):
            return False
        return True

    def _reference_route(
        self, scores: torch.Tensor, out_dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Torch reference chain (pre-r4 router math, kept verbatim)."""
        probs = torch.softmax(scores.to(torch.float32), dim=-1)
        top_weights, top_idx = torch.topk(probs, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        top_weights = top_weights * self.per_expert_scale[top_idx]
        return top_weights.to(out_dtype), top_idx

    def _forward_fused_routing(
        self, scores: torch.Tensor, out_dtype: torch.dtype
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        try:
            num_tokens = scores.shape[0]
            top_idx = torch.empty(
                num_tokens, self.top_k, device=scores.device, dtype=torch.int64
            )
            top_weights = torch.empty(
                num_tokens, self.top_k, device=scores.device, dtype=out_dtype
            )
            self._routing_kernel[(num_tokens,)](
                scores,
                self.per_expert_scale,
                top_idx,
                top_weights,
                num_tokens,
                scores.stride(0),
                E=self.num_experts,
                K=self.top_k,
                BLOCK_E=max(16, 1 << (self.num_experts - 1).bit_length()),
                BLOCK_K=max(1, 1 << (self.top_k - 1).bit_length()),
            )
            return top_weights, top_idx
        except Exception as error:
            # Degrade to the torch reference for the rest of the process
            # (same contract as the fused-entropy kernel in the proposer).
            if not self._routing_warned:
                logger.warning(
                    "DiffusionGemma fused routing failed (%s); torch path for this process",
                    error,
                )
                self._routing_warned = True
            return None

    def _routing_ab(
        self,
        scores: torch.Tensor,
        fused: Tuple[torch.Tensor, torch.Tensor],
        out_dtype: torch.dtype,
    ) -> None:
        """Debug A/B (DGM_ROUTING_AB=1): fused kernel vs torch reference on
        the same logits. Reports top-8 order/set agreement, weight deltas
        and the softmax boundary margin on mismatching rows — flips are
        near-tie class when mismatches only occur where the k-th / (k+1)-th
        probability gap is tiny."""
        try:
            ref_w, ref_i = self._reference_route(scores, out_dtype)
            fus_w, fus_i = fused
            rows = scores.shape[0]
            order_eq = (fus_i == ref_i).all(dim=-1)
            set_eq = (
                fus_i.sort(dim=-1).values == ref_i.sort(dim=-1).values
            ).all(dim=-1)
            w_diff = (fus_w.to(torch.float32) - ref_w.to(torch.float32)).abs()
            mismatch = ~(order_eq & set_eq)
            slot_gap_str = "n/a"
            margin_str = "n/a"
            if bool(mismatch.any()):
                probs = torch.softmax(scores.to(torch.float32), dim=-1)
                slot_diff = fus_i != ref_i
                r_idx, s_idx = slot_diff.nonzero(as_tuple=True)
                # Probability gap between the two experts disagreeing on a
                # slot: ~0 means the flip is an exact/near tie (bf16-logit
                # collision), a large gap would be a ranking bug.
                gaps = (
                    probs[r_idx, fus_i[r_idx, s_idx]] - probs[r_idx, ref_i[r_idx, s_idx]]
                ).abs()
                slot_gap_str = f"{float(gaps.max()):.3e}"
                if self.top_k < self.num_experts:
                    sorted_p = probs.sort(dim=-1, descending=True).values
                    boundary = sorted_p[:, self.top_k - 1] - sorted_p[:, self.top_k]
                    margin_str = f"{float(boundary[mismatch].min()):.3e}"
            print(
                f"[dgm-routing-ab] M={rows} E={self.num_experts} K={self.top_k} "
                f"exact_order={int(order_eq.sum())}/{rows} "
                f"same_set={int(set_eq.sum())}/{rows} "
                f"max_wdiff={float(w_diff.max()):.3e} "
                f"mean_wdiff={float(w_diff.mean()):.3e} "
                f"order_flip_max_prob_gap={slot_gap_str} "
                f"min_boundary_margin_on_mismatch={margin_str}",
                flush=True,
            )
        except Exception as error:
            logger.warning("DiffusionGemma routing A/B check failed: %s", error)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.pre_norm(hidden_states)
        h = h * self.scale * self.scalar_root_size
        scores = self.proj(h)
        if self._fused_routing_usable(scores):
            fused = self._forward_fused_routing(scores, hidden_states.dtype)
            if fused is not None:
                if _ROUTING_AB_CHECK:
                    self._routing_ab(scores, fused, hidden_states.dtype)
                return fused
        return self._reference_route(scores, hidden_states.dtype)


class DiffusionGemmaExperts(nn.Module):
    """Routed experts: packed 3D gate_up [E, 2I, H] / down [E, H, I] with
    gelu_pytorch_tanh gating.

    GPU bf16 forwards run the fused Triton MoE executor — the same
    moe_align_block_size_torch + get_default_config + invoke_fused_moe_kernel
    sequence as TritonFusedMoeExecutor.execute — with flashinfer's fused
    gelu_tanh_and_mul replacing the separate _gelu_tanh/mul. The reference
    gather-per-active-expert loop stays as the fallback for CPU / non-bf16
    and for DGM_FUSED_MOE=0. Both executors consume the same
    top_k_weights/top_k_idx products (the router epilogue itself is fused
    separately — see DiffusionGemmaRouter) and are value-equivalent modulo
    bf16 GEMM/accumulate rounding."""

    def __init__(self, config: ModelConfig, weights: Dict[str, torch.Tensor]):
        super().__init__()
        self.gate_up_proj = weights[W.moe_w1]  # [E, 2I, H]
        self.down_proj = weights[W.moe_w2]  # [E, H, I]
        self.num_experts = config.expert_num
        self.inter = config.moe_inter_size // 2 * 2
        self.top_k = config.moe_k
        self.fused_enabled = os.environ.get("DGM_FUSED_MOE", "1") != "0"
        self._fused_ready = False
        if self.fused_enabled:
            try:
                import triton.language as tl

                from flashinfer.activation import gelu_tanh_and_mul

                from rtp_llm.models_py.triton_kernels.moe.fused_moe_kernel import (
                    get_default_config,
                    invoke_fused_moe_kernel,
                    moe_align_block_size_torch,
                )

                self._tl_bfloat16 = tl.bfloat16
                self._gelu_tanh_and_mul = gelu_tanh_and_mul
                self._get_default_config = get_default_config
                self._invoke_fused_moe_kernel = invoke_fused_moe_kernel
                self._moe_align_block_size = moe_align_block_size_torch
                self._fused_ready = True
            except Exception as error:  # CPU-only / missing kernels
                logger.info("DiffusionGemma fused MoE unavailable, loop path: %s", error)

    def _fused_usable(self, hidden_states: torch.Tensor) -> bool:
        return (
            self._fused_ready
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_weights: torch.Tensor,
        top_k_idx: torch.Tensor,
    ) -> torch.Tensor:
        if self._fused_usable(hidden_states):
            return self._forward_fused(hidden_states, top_k_weights, top_k_idx)
        return self._forward_loop(hidden_states, top_k_weights, top_k_idx)

    def _forward_fused(
        self,
        hidden_states: torch.Tensor,
        top_k_weights: torch.Tensor,
        top_k_idx: torch.Tensor,
    ) -> torch.Tensor:
        # gate_up layout: [E, 2I, H] with gate first (loader orders it).
        M, K = hidden_states.shape
        if M == 0:
            return torch.zeros_like(hidden_states)
        top_k = top_k_idx.size(1)
        E = self.num_experts
        N = self.gate_up_proj.size(1)  # 2 * inter
        inter = N // 2
        device = hidden_states.device
        dtype = hidden_states.dtype
        compute_type = self._tl_bfloat16

        # Config selection for both GEMMs
        config1 = self._get_default_config(M, E, N, K, top_k)
        config2 = self._get_default_config(M, E, self.down_proj.size(1), inter, top_k)

        # Flat routing tensors for the kernel
        flat_topk_weights = top_k_weights.reshape(-1)  # [M * top_k]
        topk_ids = top_k_idx.to(torch.int32)
        flat_topk_ids = topk_ids.reshape(-1)

        # Token-expert alignment: compute once if block sizes match, else twice
        block_m1 = config1["BLOCK_SIZE_M"]
        block_m2 = config2["BLOCK_SIZE_M"]
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            self._moe_align_block_size(topk_ids, block_m1, E)
        )
        if block_m1 == block_m2:
            sorted_token_ids2, expert_ids2, num_tokens_post_padded2 = (
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
            )
        else:
            sorted_token_ids2, expert_ids2, num_tokens_post_padded2 = (
                self._moe_align_block_size(topk_ids, block_m2, E)
            )

        # GEMM1: hidden_states @ w1.T → intermediate1 [M*top_k, 2I]
        intermediate1 = torch.empty(M * top_k, N, device=device, dtype=dtype)
        self._invoke_fused_moe_kernel(
            hidden_states,
            self.gate_up_proj,
            intermediate1,
            flat_topk_weights,
            flat_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=top_k,
            config=config1,
            compute_type=compute_type,
        )

        # Activation: gelu_tanh(gate) * up → intermediate2 [M*top_k, I]
        intermediate2 = torch.empty(M * top_k, inter, device=device, dtype=dtype)
        self._gelu_tanh_and_mul(intermediate1, out=intermediate2)

        # GEMM2: intermediate2 @ w2.T → out [M*top_k, K]
        out = torch.empty(M * top_k, K, device=device, dtype=dtype)
        self._invoke_fused_moe_kernel(
            intermediate2,
            self.down_proj,
            out,
            flat_topk_weights,
            flat_topk_ids,
            sorted_token_ids2,
            expert_ids2,
            num_tokens_post_padded2,
            mul_routed_weight=True,
            top_k=1,
            config=config2,
            compute_type=compute_type,
        )

        # Sum reduce over top_k: out is [M*top_k, K], reshape and sum
        return out.view(M, top_k, K).sum(dim=1)

    def _forward_loop(
        self,
        hidden_states: torch.Tensor,
        top_k_weights: torch.Tensor,
        top_k_idx: torch.Tensor,
    ) -> torch.Tensor:
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
        # One batched D2H instead of 256 per-slot .item() reads: the expert
        # loop used to sync on boundaries[e]/[e+1] for every slot (including
        # empty ones) — 860k .item() launches across a decode window, the
        # measured #1 CPU-side bottleneck (workflow census cross-check).
        bounds = boundaries.tolist()
        for e in range(E):
            lo = bounds[e]
            hi = bounds[e + 1]
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
        # opt r5 item 1: h1 = residual + normed_attn followed by
        # pre_feedforward_layernorm(h1) is the fused add+rmsnorm pattern —
        # the torch add cluster (42.7 ms/window) and the paired norm read
        # collapse into one flashinfer.triton launch (torch two-step on
        # CPU/non-bf16/DGM_FUSED_ADD_NORM=0). The sum becomes the layer's
        # residual (h1 feeds the routing input and the final add), exactly
        # the values the torch chain produced.
        hidden_states, residual = _fused_add_norm(
            hidden_states, residual, self.pre_feedforward_layernorm
        )

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

        # opt r5 item 1, second site: h12 = h1 + h2 followed by
        # post_feedforward_layernorm(h12) — same fused pattern (the in-place
        # sum lands in the fresh post_attention_layernorm_1 output tensor).
        hidden_states, _h12 = _fused_add_norm(
            hidden_states_2, hidden_states_1, self.post_feedforward_layernorm
        )
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
        # finalize's canonical-final hook already applies self.norm once;
        # passing the normed hidden here would norm it twice (decoder.norm
        # weights reach |gamma|~588, so the double norm explodes the output).
        outputs = capture.finalize(hidden_states)
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
            # Contract #5-iii: the proposer keeps self_conditioning_logits in
            # fp32, so the former .to(torch.float32) upcast pass is gone;
            # softmax(dtype=torch.float32) additionally fuses the upcast into
            # the softmax kernel for any non-fp32 caller, leaving only the
            # weight-dtype cast for the matmul.
            soft = torch.matmul(
                self_conditioning_logits.softmax(dim=-1, dtype=torch.float32).to(weight.dtype),
                weight,
            ) * self.embed_scale
            embeds = self.self_conditioning(embeds, soft)
        else:
            zeros = torch.zeros_like(embeds)
            embeds = self.self_conditioning(embeds, zeros)
        hidden_states = embeds.reshape(-1, embeds.shape[-1])
        if bool(os.environ.get("DGM_FI_DIAG", "")):
            std = float(
                hidden_states.detach()
                .reshape(-1, 256, hidden_states.shape[-1])
                .std(dim=1)
                .mean()
                .item()
            )
            print(
                f"[dgm-fi] canvas entry: hidden_row_std={std:.4f} "
                f"absmax={float(hidden_states.abs().max().item()):.3f}",
                flush=True,
            )
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
            if bool(os.environ.get("DGM_FI_DIAG", "")) and i in (0, 1, 5, 29):
                # Reuse bring-up: the op-level mixed-prefix canvas is
                # verified correct (UT run 58), so the flat step-0 logits
                # on cache-hit queries originate above or inside the layer
                # stack. Track the per-position hidden std at the canvas
                # entry and after key layers: flat-from-start points at the
                # embedding/self-conditioning input, flat-after-L0 at the
                # attention input (paged prefix content), gradual at MoE.
                std = float(
                    hidden_states.detach()
                    .reshape(-1, 256, hidden_states.shape[-1])
                    .std(dim=1)
                    .mean()
                    .item()
                )
                print(
                    f"[dgm-fi] canvas layer {i} ({layer_type}): "
                    f"hidden_row_std={std:.4f} "
                    f"absmax={float(hidden_states.abs().max().item()):.3f}",
                    flush=True,
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

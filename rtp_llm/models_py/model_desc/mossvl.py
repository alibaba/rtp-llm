"""MOSS-VL python model desc: 36 self-attn layers + 12 cross-attn layers.

Reference semantics (modeling_moss_vl.py):
- vision features NEVER enter the text stream; the text keeps one image-token
  marker per frame whose embedding is the plain vocab row.
- cross-attn layers (layer_idx in cross_attention_layers) replace
  self-attention entirely: Q = q_norm(q_proj(LN(h))) with text interleaved
  mrope; K/V = k_norm(k_proj(features))/v_proj(features) with XRoPE vision
  positions; non-causal; h += tanh(attn_gate)*attn; h += tanh(mlp_gate)*MLP.
- pure-text requests skip cross layers entirely (their KV is empty).

Bring-up notes (D3 in the RUN plan): cross K/V live in a python-side
single-context store (prefill writes, decode reads); engine-level paged
vision KV, cache reuse and CUDA graph come with the marker-mode engine work.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import select_fmha_impl_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3 import Qwen3DecoderLayer
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    FMHAImplBase,
    RMSNorm,
)
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

# Fused Triton kernels for cross-attn optimization (step 5 work-in-progress;
# mrope_rope kernel has a numerical divergence — disabled pending fix, see
# models_py/model_desc/test/mossvl_triton_test.py)
_HAS_TRITON = False


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class MossVLInterleavedMrope(nn.Module):
    """Text/vision interleaved MRoPE frequencies (reference layout).

    position_ids: [..., 3] (t, h, w per token); output cos/sin [..., dim].
    Layout after apply_interleaved_mrope: freq pairs [t,h,w,t,h,w,...(20
    triplets), t,t,t,t] for section [24,20,20] on a 128-dim rotary region.
    """

    def __init__(self, dim: int, base: float, mrope_section: Tuple[int, int, int]):
        super().__init__()
        assert len(mrope_section) == 3
        assert sum(mrope_section) * 2 == dim, (mrope_section, dim)
        self.dim = dim
        self.mrope_section = list(mrope_section)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # position_ids [..., 3] -> per-axis freqs [3, ..., dim//2]
        pos = position_ids.to(torch.float)
        if pos.shape[-1] != 3:
            raise ValueError(f"mrope position_ids must end with 3 axes, got {pos.shape}")
        # inv_freq is created on CPU in __init__; move it once to the compute
        # device BEFORE CUDA graph capture would reject the copy
        if self.inv_freq.device != pos.device:
            self.inv_freq = self.inv_freq.to(pos.device)
        freqs = torch.einsum("d,...i->i...d", self.inv_freq, pos)
        freqs = self._apply_interleaved_mrope(freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def _apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


def apply_mrope_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate x [..., heads, dim] with interleaved-mrope cos/sin [..., dim]."""
    if cos.dim() == x.dim() - 1:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    # reference contract: cos/sin are cast to the hidden dtype before rotating
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return (x * cos) + (_rotate_half(x) * sin)


class MossVLQKNorm(nn.Module):
    """Per-head RMSNorm over head_dim (reference MossVLTextRMSNorm math)."""

    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        self.weight = nn.Parameter(weight.clone(), requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., head_dim]; fp32 statistics then cast back (reference order).
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight.to(input_dtype) * x.to(input_dtype))


class MossVLCrossKVStore:
    """Single-context cross-attention K/V store (bring-up contract).

    Prefill writes per-layer K/V (over vision tokens); decode reads them back
    without re-rotation. Multi-request batching, paged vision KV and CUDA
    graph-compatible storage arrive with the engine marker-mode work; until
    then the store assumes one active context (asserted on decode).
    """

    def __init__(self, num_layers: int):
        self._kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._owner_layers = set(range(num_layers))

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cached = self._kv.get(layer_idx)
        if cached is not None:
            k = torch.cat([cached[0], key_states], dim=0)
            v = torch.cat([cached[1], value_states], dim=0)
        else:
            k, v = key_states, value_states
        self._kv[layer_idx] = (k, v)
        return k, v

    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self._kv.get(layer_idx)

    def reset(self) -> None:
        self._kv.clear()


class MossVLCrossAttention(nn.Module):
    """Q from text hidden states; K/V from vision features (XRoPE, non-causal).

    The checkpoint stores cross_attn.{q,k,v,o}_proj separately; the loader
    merges them into the standard W.attn_qkv_w layout [hidden, q|k|v], which
    is sliced back apart here because Q and K/V consume different inputs.
    """

    def __init__(
        self,
        attn_config,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layernorm_eps: float,
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.parallelism_config = parallelism_config
        self.tp_size = parallelism_config.get_attn_tp_size()
        self.head_num = attn_config.head_num
        self.kv_head_num = attn_config.kv_head_num
        self.num_key_value_groups = self.head_num // self.kv_head_num
        self.head_dim = attn_config.size_per_head
        self.q_size = self.head_num * self.head_dim
        self.kv_size = self.kv_head_num * self.head_dim
        self.scaling = self.head_dim**-0.5

        merged_qkv = weights[W.attn_qkv_w]
        # merged layout (merge_qkv_hf): [hidden, q_cols | k_cols | v_cols]
        self.register_buffer(
            "q_weight", merged_qkv[:, : self.q_size].t().contiguous(), persistent=False
        )
        self.register_buffer(
            "k_weight",
            merged_qkv[
                :, self.q_size : self.q_size + self.kv_size
            ].t().contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "v_weight",
            merged_qkv[:, self.q_size + self.kv_size :].t().contiguous(),
            persistent=False,
        )
        self.o_proj = LinearFactory.create_linear_from_weights(
            weights, W.attn_o_w, W.attn_o_s, W.attn_o_b
        )
        self.q_norm = MossVLQKNorm(weights[W.q_ln_gamma], layernorm_eps)
        self.k_norm = MossVLQKNorm(weights[W.k_ln_gamma], layernorm_eps)

    def project_kv(
        self, cross_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """features [S, hidden] -> (K [S, kv, dim], V [S, kv, dim])."""
        k = F.linear(cross_states, self.k_weight).view(-1, self.kv_head_num, self.head_dim)
        v = F.linear(cross_states, self.v_weight).view(-1, self.kv_head_num, self.head_dim)
        # fused Triton per-head RMSNorm (1 launch vs ~4 native torch kernels)
        if _HAS_TRITON:
            k_normed = fused_qk_rmsnorm(k, self.k_norm.weight.data, self.k_norm.eps)
            if k_normed is not None:
                return k_normed, v
        return self.k_norm(k), v

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_states: Optional[torch.Tensor],
        q_cos: torch.Tensor,
        q_sin: torch.Tensor,
        k_cos: Optional[torch.Tensor],
        k_sin: Optional[torch.Tensor],
        kv_store: MossVLCrossKVStore,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        q = F.linear(hidden_states, self.q_weight).view(
            -1, self.head_num, self.head_dim
        )
        # fused Triton per-head RMSNorm for Q
        if _HAS_TRITON:
            q_normed = fused_qk_rmsnorm(q, self.q_norm.weight.data, self.q_norm.eps)
            if q_normed is not None:
                q = q_normed
            else:
                q = self.q_norm(q)
        else:
            q = self.q_norm(q)
        # fused Triton MRoPE rotation for Q (1 launch vs ~5 native kernels)
        if _HAS_TRITON:
            n_rows = q.shape[0] * q.shape[1]
            q_cos_exp = q_cos.unsqueeze(1).expand_as(q) if q_cos.dim() == 2 else q_cos
            q_sin_exp = q_sin.unsqueeze(1).expand_as(q) if q_sin.dim() == 2 else q_sin
            q_roped = fused_mrope_rope(q, q_cos_exp, q_sin_exp)
            if q_roped is not None:
                q = q_roped
            else:
                q = apply_mrope_rope(q, q_cos, q_sin)
        else:
            q = apply_mrope_rope(q, q_cos, q_sin)

        if cross_states is not None:
            k, v = self.project_kv(cross_states)
            if k_cos is not None:
                # fused Triton MRoPE rotation for K
                if _HAS_TRITON:
                    k_cos_exp = k_cos.unsqueeze(1).expand_as(k) if k_cos.dim() == 2 else k_cos
                    k_sin_exp = k_sin.unsqueeze(1).expand_as(k) if k_sin.dim() == 2 else k_sin
                    k_roped = fused_mrope_rope(k, k_cos_exp, k_sin_exp)
                    if k_roped is not None:
                        k = k_roped
                    else:
                        k = apply_mrope_rope(k, k_cos, k_sin)
                else:
                    k = apply_mrope_rope(k, k_cos, k_sin)
            k, v = kv_store.update(self.layer_idx, k, v)
        else:
            cached = kv_store.get(self.layer_idx)
            if cached is None:
                raise ValueError(
                    f"cross layer {self.layer_idx} has neither new vision states "
                    "nor a populated cross KV store"
                )
            k, v = cached

        # GQA broadcast: [S, kv, d] -> [S, heads, d]
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False, scale=self.scaling
        )
        attn_output = attn_output.transpose(0, 1).reshape(*input_shape, -1).contiguous()
        output = self.o_proj(attn_output)
        if self.tp_size > 1:
            from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

            output = all_reduce(output, group=Group.TP)
        return output


class MossVLCrossAttentionDecoderLayer(nn.Module):
    """Cross-attention layer with tanh-gated attention AND MLP residuals."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        layer_idx: int,
        weights: Dict[str, torch.Tensor],
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        attn_configs = config.getAttentionConfigs(
            parallelism_config.get_attn_tp_size()
        )
        # Pre-compute tanh(gate) once per layer: the gates are constant
        # scalars during inference; caching eliminates the torch.tanh() call
        # (a native elementwise kernel) per step while preserving exact
        # numerics (the multiply itself remains — see gate ⑥ note in the
        # performance analysis: weight folding was tried and reverted due to
        # bf16 rounding changes that break generation stopping).
        self._tanh_attn_gate = torch.tanh(
            weights[W.cross_attn_attn_gate].to(torch.float32).reshape(())
        )
        self._tanh_mlp_gate = torch.tanh(
            weights[W.cross_attn_mlp_gate].to(torch.float32).reshape(())
        )
        self.register_buffer(
            "attn_gate", weights[W.cross_attn_attn_gate].reshape(()), persistent=False
        )
        self.register_buffer(
            "mlp_gate", weights[W.cross_attn_mlp_gate].reshape(()), persistent=False
        )
        self.cross_attn = MossVLCrossAttention(
            attn_configs, parallelism_config, weights, config.layernorm_eps, layer_idx
        )
        self.mlp = DenseMLP(
            config.activation_type, parallelism_config, weights, quant_config, hw_kernel_config
        )
        self.input_layernorm = RMSNorm(weights[W.pre_ln_gamma], eps=config.layernorm_eps)
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_states: Optional[torch.Tensor],
        q_cos: torch.Tensor,
        q_sin: torch.Tensor,
        k_cos: Optional[torch.Tensor],
        k_sin: Optional[torch.Tensor],
        kv_store: MossVLCrossKVStore,
        attn_mask: Optional[torch.Tensor] = None,
        row_visible: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.cross_attn(
            hidden_states,
            cross_states,
            q_cos,
            q_sin,
            k_cos,
            k_sin,
            kv_store,
            attn_mask,
        )
        # reference full_text_row_masked_out_mask: rows with no visible vision
        # get their attention AND MLP contributions zeroed (residual survives)
        if row_visible is not None:
            hidden_states = hidden_states * row_visible.to(hidden_states.dtype).unsqueeze(-1)
        # fused Triton gated residual add (1 launch vs 2 native kernels)
        if _HAS_TRITON:
            fused = fused_gated_residual_add(residual, hidden_states, self._tanh_attn_gate)
            if fused is not None:
                hidden_states = fused
            else:
                hidden_states = residual + self._tanh_attn_gate.to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                ) * hidden_states
        else:
            hidden_states = residual + self._tanh_attn_gate.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            ) * hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if row_visible is not None:
            hidden_states = hidden_states * row_visible.to(hidden_states.dtype).unsqueeze(-1)
        # fused Triton gated residual add for MLP gate
        if _HAS_TRITON:
            fused = fused_gated_residual_add(residual, hidden_states, self._tanh_mlp_gate)
            if fused is not None:
                hidden_states = fused
            else:
                hidden_states = residual + self._tanh_mlp_gate.to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                ) * hidden_states
        else:
            hidden_states = residual + self._tanh_mlp_gate.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            ) * hidden_states
        return hidden_states


class MossVLCrossInputs:
    """Assembled cross-attention inputs for one forward.

    Built from per-media vision sequences (features with separators), the
    per-frame marker token positions in the text stream, and per-frame grid
    geometry. Vision 3D positions and the causal frame-visibility mask are
    derived here once per forward and shared by all 12 cross layers.
    """

    def __init__(
        self,
        features: Optional[torch.Tensor],
        k_cos: Optional[torch.Tensor],
        k_sin: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        row_visible: Optional[torch.Tensor],
    ):
        self.features = features
        self.k_cos = k_cos
        self.k_sin = k_sin
        self.attn_mask = attn_mask
        self.row_visible = row_visible


def build_moss_vl_cross_inputs(
    features_per_media: Optional[List[torch.Tensor]],
    frame_marker_positions: Optional[List[int]],
    frame_geometry: Optional[List[Tuple[int, int, int]]],
    combo_position_ids: Optional[torch.Tensor],
    rope: MossVLInterleavedMrope,
    device: torch.device,
    dtype: torch.dtype,
    old_rows: int = 0,
) -> MossVLCrossInputs:
    """features_per_media: [rows_with_separators, hidden] each; geometry per
    media: (num_frames, eh, ew); marker positions per FRAME in text order.
    old_rows: cross K/V rows already in the store from a prior conversation
    turn (prefix reuse) — prepended as all-visible mask columns.
    None/empty features -> decode step (no new vision)."""
    if features_per_media is None or len(features_per_media) == 0:
        return MossVLCrossInputs(None, None, None, None, None)
    assert combo_position_ids is not None
    if combo_position_ids.dim() == 1:
        # engine transport flattens [seq, 3] mrope ids to [seq*3]
        combo_position_ids = combo_position_ids.reshape(-1, 3)
    seq_len = combo_position_ids.shape[0]
    k_positions = []
    frame_spans = []  # (visible_from_text_pos, start_col, end_col)
    col = 0
    frame_idx = 0
    for media, geometry in zip(features_per_media, frame_geometry):
        num_frames, eh, ew = geometry
        rows = media.shape[0]
        stride = eh * ew + 1
        assert num_frames * stride == rows, (rows, num_frames, stride)
        max_hw = max(eh, ew)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(eh), torch.arange(ew), indexing="ij"
        )
        y_flat = y_grid.reshape(-1).to(torch.float32)
        x_flat = x_grid.reshape(-1).to(torch.float32)
        for _f in range(num_frames):
            marker = int(frame_marker_positions[frame_idx])
            marker_t = float(combo_position_ids[marker, 0])
            t_val = marker_t - max_hw
            frame_positions = torch.stack(
                (
                    torch.full((eh * ew,), t_val, dtype=torch.float32),
                    t_val + y_flat,
                    t_val + x_flat,
                ),
                dim=-1,
            )
            sep_position = torch.full((1, 3), marker_t, dtype=torch.float32)
            k_positions.append(torch.cat([frame_positions, sep_position], dim=0))
            frame_spans.append((marker, col, col + stride))
            col += stride
            frame_idx += 1
    features = torch.cat(features_per_media, dim=0).to(device=device, dtype=dtype)
    k_positions = torch.cat(k_positions, dim=0).to(device=device, dtype=torch.float32)
    k_cos, k_sin = rope(k_positions)
    # causal frame visibility: text token p sees frame f iff marker_f <= p.
    # Old rows (prior conversation turn, prefix reuse) are visible to ALL
    # current tokens — their markers precede this prefill's token range.
    # Masked entries use finfo.min (reference convention) so fully-masked
    # rows degrade to a uniform softmax instead of NaN; row_visible then
    # zeroes those contributions (full_text_row_masked_out_mask semantics).
    total_col = old_rows + col
    attn_mask = torch.full((seq_len, total_col), torch.finfo(dtype).min, device=device, dtype=dtype)
    if old_rows > 0:
        attn_mask[:, :old_rows] = 0.0
    for visible_from, start, end in frame_spans:
        attn_mask[visible_from:, old_rows + start : old_rows + end] = 0.0
    row_visible = (attn_mask != torch.finfo(dtype).min).any(dim=-1)
    return MossVLCrossInputs(features, k_cos, k_sin, attn_mask, row_visible)


class MossVLModel(GptModelBase):
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
        self.cross_attention_layers = list(
            config.mm_related_params.config.get("cross_attention_layers", [])
        )
        self._cross_layer_set = set(self.cross_attention_layers)
        rope_config = config.attn_config.rope_config
        self.rope = MossVLInterleavedMrope(
            rope_config.dim,
            float(rope_config.base),
            (
                rope_config.mrope_dim1,
                rope_config.mrope_dim2,
                rope_config.mrope_dim3,
            ),
        )
        self.cross_kv_store = MossVLCrossKVStore(config.num_layers)
        self._context_has_vision = False

        self.embed_tokens = Embedding(
            config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.layers = nn.ModuleList()
        for idx in range(self.layer_num):
            if idx in self._cross_layer_set:
                self.layers.append(
                    MossVLCrossAttentionDecoderLayer(
                        config,
                        parallelism_config,
                        idx,
                        weights.weights[idx],
                        quant_config,
                        py_hw_kernel_config,
                    )
                )
            else:
                self.layers.append(
                    Qwen3DecoderLayer(
                        config,
                        parallelism_config,
                        idx,
                        weights.weights[idx],
                        quant_config,
                        py_hw_kernel_config,
                    )
                )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )
        self._init_capture_context(
            self._capture_canonical_layer,
            self._capture_canonical_final,
        )

    def _capture_canonical_layer(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None
    ) -> torch.Tensor:
        return hidden_states

    def _capture_canonical_final(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None
    ) -> torch.Tensor:
        return self.norm(hidden_states)

    def reset_cross_kv(self) -> None:
        self.cross_kv_store.reset()
        self._context_has_vision = False

    @staticmethod
    def _is_prefill_step(inputs: PyModelInputs) -> bool:
        attention_inputs = getattr(inputs, "attention_inputs", None)
        if attention_inputs is None:
            by_tag = getattr(inputs, "attention_inputs_by_tag", None)
            if isinstance(by_tag, dict) and by_tag:
                attention_inputs = next(iter(by_tag.values()))
        if attention_inputs is None:
            return True
        return bool(getattr(attention_inputs, "is_prefill", True))

    @staticmethod
    def _get_prefix_length(inputs: PyModelInputs) -> int:
        attention_inputs = getattr(inputs, "attention_inputs", None)
        if attention_inputs is None:
            by_tag = getattr(inputs, "attention_inputs_by_tag", None)
            if isinstance(by_tag, dict) and by_tag:
                attention_inputs = next(iter(by_tag.values()))
        if attention_inputs is None:
            return 0
        prefix_lengths = getattr(attention_inputs, "prefix_lengths", None)
        if prefix_lengths is None:
            return 0
        if torch.is_tensor(prefix_lengths):
            if prefix_lengths.numel() == 0:
                return 0
            return int(prefix_lengths.reshape(-1)[0].item())
        return int(prefix_lengths)

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        capture = self.capture_context(inputs.capture_hidden_states)

        mm = inputs.multimodal_inputs
        # store lifecycle must be resolved BEFORE building cross inputs (the
        # builder counts old_rows from the store for the attention mask)
        is_prefill = self._is_prefill_step(inputs)
        if is_prefill:
            prefix_len = self._get_prefix_length(inputs)
            if prefix_len == 0:
                # fresh request: replace any stale cross KV from a previous
                # conversation
                self.cross_kv_store.reset()
        # build_moss_vl_cross_inputs reshapes the engine's flat [seq*3]
        # mrope ids itself
        cross = self._build_cross_inputs(
            inputs, mm, hidden_states.device, hidden_states.dtype
        )
        if is_prefill:
            if prefix_len == 0:
                self._context_has_vision = cross.features is not None
            else:
                # continuation of a conversation (text prefix reused): the
                # cross KV for reused-region frames was computed at the prior
                # prefill and stays in the store; new features append.
                self._context_has_vision = True
        position_ids = inputs.combo_position_ids
        if position_ids is None:
            # text-only fallback: plain (p, p, p) positions
            position_ids = (
                torch.arange(hidden_states.shape[0], device=hidden_states.device)
                .unsqueeze(-1)
                .expand(-1, 3)
                .contiguous()
            )
        elif position_ids.dim() == 1:
            # engine transport flattens [seq, 3] mrope ids to [seq*3]
            position_ids = position_ids.reshape(-1, 3)
        elif position_ids.dim() != 2 or position_ids.shape[-1] != 3:
            raise ValueError(
                f"moss_vl expects combo_position_ids [seq, 3], got {tuple(position_ids.shape)}"
            )
        q_cos, q_sin = self.rope(position_ids.to(hidden_states.device))

        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            if i in self._cross_layer_set:
                if cross.features is None and not self._context_has_vision:
                    # text-only request: the reference skips cross layers
                    # entirely (their per-generation cache is empty)
                    capture.capture_layer(i, hidden_states)
                    continue
                hidden_states = decoder_layer(
                    hidden_states,
                    cross.features,
                    q_cos,
                    q_sin,
                    cross.k_cos,
                    cross.k_sin,
                    self.cross_kv_store,
                    cross.attn_mask,
                    cross.row_visible,
                )
            else:
                layer_fmha_impl = select_fmha_impl_for_layer(fmha_impl, self.kv_cache, i)
                hidden_states = decoder_layer(
                    hidden_states,
                    layer_fmha_impl,
                    kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                )
            capture.capture_layer(i, hidden_states)
        return capture.finalize(hidden_states)

    def _build_cross_inputs(
        self,
        inputs: PyModelInputs,
        mm,
        device: torch.device,
        dtype: torch.dtype,
    ) -> MossVLCrossInputs:
        features = getattr(mm, "multimodal_features", None) if mm is not None else None
        if not features:
            return MossVLCrossInputs(None, None, None, None, None)
        extra = getattr(mm, "mm_extra_input", None) or []
        locs = getattr(mm, "mm_features_locs", None)
        frame_geometry = []
        for tensor in extra:
            values = tensor.reshape(-1).tolist()
            frame_geometry.append((int(values[0]), int(values[1]), int(values[2])))
        marker_positions = (
            [int(p) for p in locs.reshape(-1).tolist()] if locs is not None else []
        )
        # rows already in the store from a prior turn of the same
        # conversation (prefix reuse) — visible to all current tokens
        cached = self.cross_kv_store.get(self.cross_attention_layers[0]) if self.cross_attention_layers else None
        old_rows = int(cached[0].shape[0]) if cached is not None else 0
        return build_moss_vl_cross_inputs(
            list(features),
            marker_positions,
            frame_geometry,
            inputs.combo_position_ids,
            self.rope,
            device,
            dtype,
            old_rows=old_rows,
        )


__all__ = [
    "MossVLModel",
    "MossVLCrossAttention",
    "MossVLCrossAttentionDecoderLayer",
    "MossVLInterleavedMrope",
    "build_moss_vl_cross_inputs",
]


"""
MLA attention for DeepSeek V3.2, new-loader style.

Key design decisions:
  - __init__ creates independent nn.Module submodules named with HF ckpt keys
    (q_a_proj, kv_a_proj_with_mqa, q_a_layernorm, kv_a_layernorm, q_b_proj, o_proj).
    HF weights flow through RtpModule.load_weights directly into nn.Parameter.
  - process_weights_after_loading() fuses q_a_proj + kv_a_proj_with_mqa into
    _fused_qkv_a_w (following Qwen3Experts pattern), and stacks q_b_proj
    + kv_b_proj split into _fused_qkv_b_w.
  - _build_weights_dict() assembles the W.* dict that the MlaImplBase kernel
    factory expects at forward time.
  - forward() mirrors MlaAttention.forward() exactly; the Indexer is a
    separate submodule built by the DecoderLayer when is_sparse is True.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.layers.linear import ColumnParallelLinear, RowParallelLinear
from rtp_llm.models_py.layers.norm import RMSNorm
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules.factory.attention.attn_factory import MlaImplBase
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.ops.compute_ops import LayerKVCache
from rtp_llm.utils.model_weight import W


def _dequant_block_to_bf16(
    weight: torch.Tensor, scale: torch.Tensor, block: int = 128
) -> torch.Tensor:
    """Dequantize a DeepSeek FP8 per-block (128x128) weight [N,K] to bf16.

    scale is the [ceil(N/128), ceil(K/128)] block grid (standard orientation).
    """
    n, k = weight.shape
    s = scale.to(torch.float32)
    s = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    s = s[:n, :k]
    return (weight.to(torch.float32) * s).to(torch.bfloat16)


def _linear_weight_bf16(linear: nn.Module) -> torch.Tensor:
    """Return a linear's weight as bf16, dequantizing if it is fp8-per-block.

    Used by the MLA post-load derivations (fused qkv-a, kc/vc) which run
    through torch.cat / torch.bmm — neither supports fp8 — even though the
    forward projections themselves keep running the fp8 weights via DeepGEMM.
    Called from the attention module's post-load hook, which fires before the
    child linears' own hook (parent-before-child), so the scale is still under
    `weight_scale_inv`; `weight_scale` is checked too for robustness.
    """
    w = linear.weight.data
    if w.dtype != torch.float8_e4m3fn:
        return w
    scale = getattr(linear, "weight_scale_inv", None)
    if scale is None:
        scale = getattr(linear, "weight_scale", None)
    if scale is None:
        raise RuntimeError(
            f"fp8 linear missing block scale: {getattr(linear, 'prefix', '?')}"
        )
    return _dequant_block_to_bf16(w, scale.data)


class DeepSeekV32MlaAttention(RtpModule):
    """MLA attention for DeepSeek V3.2, new-loader style.

    HF ckpt keys consumed (per layer):
      model.layers.{i}.self_attn.q_a_proj.weight          → q_a_proj
      model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight → kv_a_proj_with_mqa
      model.layers.{i}.self_attn.q_a_layernorm.weight      → q_a_layernorm
      model.layers.{i}.self_attn.kv_a_layernorm.weight     → kv_a_layernorm
      model.layers.{i}.self_attn.q_b_proj.weight           → q_b_proj
      model.layers.{i}.self_attn.kv_b_proj.weight          → kv_b_proj
      model.layers.{i}.self_attn.o_proj.weight             → o_proj

    process_weights_after_loading fuses q_a_proj + kv_a_proj_with_mqa into
    _fused_qkv_a_w, copies q_b_proj into _fused_qkv_b_w, and splits
    kv_b_proj into _kc_w (nope half) and _vc_w (v half) using the same
    transpose+slice formula as the legacy loader.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        nope_head_dim: int,
        rope_head_dim: int,
        v_head_dim: int,
        layer_idx: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: torch.dtype = torch.bfloat16,
        layernorm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads // tp_size
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.nope_head_dim = nope_head_dim
        self.rope_head_dim = rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_head_dim = nope_head_dim + rope_head_dim
        self.layer_idx = layer_idx
        self.tp_size = tp_size
        self.softmax_scale = self.q_head_dim ** (-0.5)

        # --- Independent submodules matching HF ckpt names ---
        # q_a_proj is either the LoRA down-projection (hidden -> q_lora_rank)
        # or, when q_lora_rank == 0, the direct query projection
        # (hidden -> num_heads * q_head_dim). Both are replicated across TP here;
        # no-LoRA fused q/k-rope splitting is handled by the MLA kernels.
        q_a_output_size = (
            q_lora_rank if q_lora_rank > 0 else num_heads * self.q_head_dim
        )
        self.q_a_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=q_a_output_size,
            tp_size=1,
            tp_rank=0,
            quant_config=quant_config,
            prefix="q_a_proj" if q_lora_rank > 0 else "q_proj",
            bias=False,
            params_dtype=params_dtype,
        )
        # kv_a_proj_with_mqa: hidden → (kv_lora_rank + rope_head_dim).
        # MQA shared kv latent + k_pe — REPLICATED across TP ranks (tp_size=1).
        self.kv_a_proj_with_mqa = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=kv_lora_rank + rope_head_dim,
            tp_size=1,
            tp_rank=0,
            quant_config=quant_config,
            prefix="kv_a_proj_with_mqa",
            bias=False,
            params_dtype=params_dtype,
        )
        self.q_a_layernorm = RMSNorm(
            q_lora_rank, eps=layernorm_eps, params_dtype=params_dtype
        )
        self.kv_a_layernorm = RMSNorm(
            kv_lora_rank, eps=layernorm_eps, params_dtype=params_dtype
        )
        # q_b_proj: q_lora_rank → num_heads * q_head_dim
        self.q_b_proj = ColumnParallelLinear(
            input_size=q_lora_rank,
            output_size=num_heads * self.q_head_dim,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix="q_b_proj",
            bias=False,
            params_dtype=params_dtype,
        )
        # kv_b_proj: kv_lora_rank → num_heads * (nope_head_dim + v_head_dim).
        # Per-head k_nope / v up-projection — SHARDED by head along the output
        # dim (column parallel), matching the legacy loader's head_num split.
        # Each head contributes (nope+v) contiguous rows, so a plain column
        # split lands whole heads on each rank; self.num_heads (= num_heads //
        # tp_size) then matches the loaded weight in the kc/vc derivation below.
        self.kv_b_proj = ColumnParallelLinear(
            input_size=kv_lora_rank,
            output_size=num_heads * (nope_head_dim + v_head_dim),
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix="kv_b_proj",
            bias=False,
            params_dtype=params_dtype,
        )
        # o_proj: num_heads * v_head_dim → hidden
        self.o_proj = RowParallelLinear(
            input_size=num_heads * v_head_dim,
            output_size=hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix="o_proj",
            bias=False,
            params_dtype=params_dtype,
        )

        # --- Fused weights (built after loading) ---
        self._fused_qkv_a_w: Optional[nn.Parameter] = None
        self._fused_qkv_b_w: Optional[nn.Parameter] = None
        self._kv_b_w: Optional[nn.Parameter] = None
        self._kc_w: Optional[nn.Parameter] = None
        self._vc_w: Optional[nn.Parameter] = None

    def load_weights(self, weights):
        if self.q_lora_rank == 0:
            items = weights.items() if isinstance(weights, dict) else weights
            weights = {
                (
                    "q_a_proj." + name[len("q_proj.") :]
                    if name.startswith("q_proj.")
                    else name
                ): tensor
                for name, tensor in items
            }
        return super().load_weights(weights)

    def process_weights_after_loading(self):
        """Fuse q_a_proj + kv_a_proj_with_mqa into a single _fused_qkv_a_w.

        Also collect q_b_proj weight into _fused_qkv_b_w, and split
        kv_b_proj into _kc_w (nope) and _vc_w (v) using the same formula
        as the legacy loader's transpose_slice_k / transpose_slice_v
        (utils/model_weight.py).
        """
        # These derivations run through torch.cat / torch.bmm, which do not
        # support fp8, so dequantize the (possibly fp8-per-block) weights to
        # bf16 here. The forward projections still execute the fp8 weights via
        # the linear's DeepGEMM apply — this only affects the kc/vc + fused
        # views consumed by the MLA kernel.
        q_a_w = _linear_weight_bf16(self.q_a_proj)
        kv_a_w = _linear_weight_bf16(self.kv_a_proj_with_mqa)
        self._fused_qkv_a_w = nn.Parameter(
            torch.cat([q_a_w, kv_a_w], dim=0).contiguous(), requires_grad=False
        )
        if self.q_lora_rank > 0:
            # q_b_proj weight: [num_heads * q_head_dim, q_lora_rank]
            self._fused_qkv_b_w = nn.Parameter(
                _linear_weight_bf16(self.q_b_proj).clone(), requires_grad=False
            )

        # kv_b_proj weight: [num_heads * (nope + v_head), kv_lora_rank].
        # Reshape to [kv_lora_rank, num_heads, nope+v_head] then slice.
        kv_b_w = _linear_weight_bf16(self.kv_b_proj)
        head_num = self.num_heads
        nope = self.nope_head_dim
        v_head = self.v_head_dim
        t = (
            kv_b_w.transpose(0, 1)
            .contiguous()
            .view(self.kv_lora_rank, head_num, nope + v_head)
        )
        # _kv_b_w: [kv_lora_rank, head_num * (nope + v_head)] — transposed kv_b
        # for FlashInfer prefill (matches legacy DeepSeekV2's `transpose` rule).
        # Decode/absorb paths use _kc_w / _vc_w instead.
        self._kv_b_w = nn.Parameter(
            t.view(self.kv_lora_rank, head_num * (nope + v_head)),
            requires_grad=False,
        )
        # _kc_w shape: [head_num, nope, kv_lora_rank]
        self._kc_w = nn.Parameter(
            t[:, :, :nope].permute(1, 2, 0).contiguous(),
            requires_grad=False,
        )
        # _vc_w shape: [head_num, kv_lora_rank, v_head]
        self._vc_w = nn.Parameter(
            t[:, :, nope:].transpose(0, 1).contiguous(),
            requires_grad=False,
        )

    def _build_weights_dict(self) -> Dict[str, torch.Tensor]:
        """Assemble the W.* dict that MlaImplBase expects at forward time."""
        if self._fused_qkv_a_w is None:
            raise RuntimeError(
                "process_weights_after_loading() must be called before "
                "_build_weights_dict()"
            )
        weights: Dict[str, torch.Tensor] = {}
        if self.q_lora_rank > 0:
            weights[W.mla_fusedqkrope_w] = self._fused_qkv_a_w
            weights[W.mla_q_b_w] = self._fused_qkv_b_w
            weights[W.mla_q_a_ln_gamma] = self.q_a_layernorm.weight.data
        else:
            weights[W.mla_fusedqkrope_no_lora_w] = self._fused_qkv_a_w
        weights[W.mla_kv_a_ln_gamma] = self.kv_a_layernorm.weight.data
        weights[W.attn_o_w] = self.o_proj.weight.data
        weights[W.mla_kv_b_w] = self._kv_b_w
        weights[W.mla_kc] = self._kc_w
        weights[W.mla_vc] = self._vc_w
        return weights

    def _run_sparse_indexer(
        self,
        hidden_states: torch.Tensor,
        q_c: Optional[torch.Tensor],
        q_view: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        fmha_impl: MlaImplBase,
    ) -> Optional[torch.Tensor]:
        """Compute sparse top-k indices via the Indexer submodule.

        Mirrors legacy MlaAttention._run_sparse_indexer
        (modules/hybrid/mla_attention.py).  Returns None for dense layers
        (indexer not attached) so fmha_impl.forward gets a None and dense
        backends short-circuit; sparse backends require non-None.
        """
        indexer = getattr(self, "indexer", None)
        if indexer is None:
            return None
        q_for_indexer = q_c if self.q_lora_rank > 0 else q_view
        return indexer(
            hidden_states,
            q_for_indexer,
            kv_cache,
            fmha_impl.fmha_params,
            fmha_impl.attn_inputs,
            use_fast_path=not fmha_impl.is_sparse(),
            cp_params=fmha_impl.cp_params,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: MlaImplBase,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        q_c = None

        if self.q_lora_rank > 0:
            # q_a projection
            q = self.q_a_proj(hidden_states)
            # kv_a projection (with rope)
            kv_a = self.kv_a_proj_with_mqa(hidden_states)
            # split: q_a, then kv_a+rope
            compressed_kv = kv_a[..., : self.kv_lora_rank]
            k_pe = kv_a[..., self.kv_lora_rank :]
            # q_a layernorm
            q_c = self.q_a_layernorm(q.contiguous())
            # q_b projection
            q = self.q_b_proj(q_c)
        else:
            # No LoRA: match the legacy loader's fused q/k-rope projection.
            if self._fused_qkv_a_w is None:
                raise RuntimeError("process_weights_after_loading() must run first")
            fused_qkv = torch.nn.functional.linear(hidden_states, self._fused_qkv_a_w)
            q_offset = self.num_heads * self.q_head_dim
            q_output, kv_output = torch.split(
                fused_qkv,
                [q_offset, self.kv_lora_rank + self.rope_head_dim],
                dim=-1,
            )
            compressed_kv = kv_output[..., : self.kv_lora_rank]
            k_pe = kv_output[..., self.kv_lora_rank :]
            q = q_output

        q_view = q.reshape(-1, self.num_heads, self.q_head_dim)

        # kv_a layernorm
        compressed_kv = self.kv_a_layernorm(compressed_kv.contiguous())

        # Sparse Indexer (DSA) — runs only when self.indexer is attached
        # (DecoderLayer sets self.indexer when is_sparse=True).
        topk_indices = self._run_sparse_indexer(
            hidden_states, q_c, q_view, kv_cache, fmha_impl
        )

        attn_output = fmha_impl.forward(
            q_view, compressed_kv, k_pe, kv_cache, self.layer_idx, topk_indices
        )

        if attn_output is not None and attn_output.numel() != 0:
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        else:
            attn_output = torch.zeros(
                (*input_shape, self.num_heads * self.v_head_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        attn_output = self.o_proj(attn_output)
        if self.tp_size > 1:
            attn_output = all_reduce(attn_output, group=Group.TP)
        return attn_output

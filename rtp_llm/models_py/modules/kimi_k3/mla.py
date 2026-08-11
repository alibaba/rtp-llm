"""Kimi K3 gated MLA integration over RTP attention backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from rtp_llm.models_py.distributed.sequence_parallel import TokenShardLayout
from rtp_llm.models_py.modules.factory.linear.parallel import (
    all_gather_matmul,
    row_parallel_linear,
    should_use_fused_all_gather_matmul,
)
from rtp_llm.models_py.modules.base import RMSNorm
from rtp_llm.models_py.modules.hybrid.mla_attention import MlaAttention
from rtp_llm.ops import ParallelismConfig, RoleType
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
)
from rtp_llm.utils.model_weight import W

if TYPE_CHECKING:
    from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3ModelConfig

_MLA_LATENT_NORM_EPS = 1e-6


def prepare_mla_fmha_for_group(
    fmha_impl: Any,
    attention_inputs: PyAttentionInputs,
    selected_group_id: int,
    prepared_group_id: Optional[int],
) -> int:
    """Refresh cached MLA params when HybridCache switches FULL groups.

    FlashInfer MLA derives ``slot_mapping`` and its page table from the
    singular block-map fields during ``prepare``.  K3 owns several FULL cache
    groups, so changing only ``attention_inputs`` leaves the wrapper writing
    every later MLA layer through group 0's slot mapping.
    """

    if selected_group_id == prepared_group_id:
        return selected_group_id
    sequence_lengths = getattr(attention_inputs, "sequence_lengths", None)
    is_capturing = bool(
        sequence_lengths is not None
        and sequence_lengths.is_cuda
        and torch.cuda.is_current_stream_capturing()
    )
    if is_capturing:
        prepare_group = getattr(fmha_impl, "prepare_cuda_graph_group", None)
        if not callable(prepare_group):
            raise RuntimeError(
                "Kimi K3 HybridCache MLA requires graph-safe group refresh "
                "during CUDA Graph capture"
            )
        prepare_group(attention_inputs)
        return selected_group_id

    prepare = getattr(fmha_impl, "prepare", None)
    if not callable(prepare):
        raise RuntimeError(
            "Kimi K3 HybridCache MLA requires an FMHA implementation with prepare()"
        )
    prepare(attention_inputs)
    return selected_group_id


def _select_mla_attention_inputs(
    explicit_inputs: Optional[PyAttentionInputs],
    fmha_impl: Any,
) -> Optional[PyAttentionInputs]:
    """Select the group-current attention-input view for K3 MLA."""

    if explicit_inputs is not None:
        return explicit_inputs
    return getattr(fmha_impl, "attn_inputs", None)


class KimiK3MLA(MlaAttention):
    """K3 NoPE MLA over RTP's packed-token and compressed-cache layouts.

    The serving path reuses the framework MLA implementation and specializes
    only K3's packed Q-A/KV-A/output-gate projection and SP output layout.
    """

    def __init__(
        self,
        config: KimiK3ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int = -1,
    ) -> None:
        super().__init__(
            config.attn_config,
            parallelism_config,
            weights,
            layer_idx,
            _MLA_LATENT_NORM_EPS,
            config.quant_config,
        )
        # The framework RMSNorm consumes dense rows. The previous K3 wrapper
        # also materialized these split views before invoking the same kernel.
        self._perf_accepts_strided_latent = False
        tp_size = int(parallelism_config.get_attn_tp_size())
        self.attn_tp_size = tp_size
        total_heads = int(config.attn_config.head_num)
        if total_heads % tp_size:
            raise ValueError(
                f"MLA heads {total_heads} must be divisible by attention TP {tp_size}"
            )
        self.local_heads = total_heads // tp_size
        self.q_lora_rank = int(config.attn_config.q_lora_rank)
        self.kv_lora_rank = int(config.attn_config.kv_lora_rank)
        self.suffix_dim = int(config.attn_config.rope_head_dim)
        self.value_dim = int(config.attn_config.v_head_dim)
        # The source K3 MLA constructs q_a/kv_a KimiRMSNorm without passing
        # config.rms_norm_eps, so both intentionally use the module default
        # 1e-6.  Other decoder norms continue to use config.rms_norm_eps
        # (1e-5 for the real checkpoint).
        runtime = config.k3_runtime_config
        if not runtime.mla_use_nope:
            raise ValueError(
                "Kimi K3 requires the physical MLA suffix to remain no-RoPE"
            )
        self.use_output_gate = runtime.mla_use_output_gate
        # Prefill 走 FlashMLA(dense prefill),Decode 走 FlashInfer("kernel")。
        # 与 KDA 一样按 PD 角色固定后端。
        self._mla_backend = (
            "flashmla"
            if parallelism_config.role_type == RoleType.PREFILL
            else "kernel"
        )

        self._q_a_norm = weights[W.mla_q_a_ln_gamma]
        self._kv_a_norm = weights[W.mla_kv_a_ln_gamma]
        self._o_w = weights[W.attn_o_w]
        self._packed_qkv_gate_w = weights[W.mla_fusedqkrope_w]
        # These are only the two small MLA latent norms; decoder-wide norms keep
        # the framework kernel.
        self.q_a_layernorm = RMSNorm(self._q_a_norm, _MLA_LATENT_NORM_EPS)
        self.kv_a_layernorm = RMSNorm(self._kv_a_norm, _MLA_LATENT_NORM_EPS)
        self._sp_active_for_forward = False
        self._sp_padded_for_forward = False
        self._sp_prefill_input_is_sharded = False
        self._sp_prefill_layout_for_forward: Optional[TokenShardLayout] = None

    def _project_qkv_a_input(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        prefill_layout = getattr(self, "_sp_prefill_layout_for_forward", None)
        if self._sp_prefill_input_is_sharded:
            logical_tokens = (
                hidden_states.shape[0] * self.attn_tp_size
                if prefill_layout is None
                else prefill_layout.logical_tokens
            )
            packed = all_gather_matmul(
                hidden_states,
                [self._packed_qkv_gate_w],
                logical_tokens=logical_tokens,
                use_fused=should_use_fused_all_gather_matmul(
                    hidden_states.shape[0] * self.attn_tp_size
                ),
            )[0]
            return torch.split(
                packed,
                [
                    self.q_lora_rank + self.kv_lora_rank + self.suffix_dim,
                    self.local_heads * self.value_dim,
                ],
                dim=-1,
            )
        packed = self.fused_qkv_a_proj(hidden_states)
        return torch.split(
            packed,
            [
                self.q_lora_rank + self.kv_lora_rank + self.suffix_dim,
                self.local_heads * self.value_dim,
            ],
            dim=-1,
        )

    def _apply_output_gate(
        self,
        attn_output: torch.Tensor,
        output_gate: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """K3 sigmoid output gate, applied on the framework (kernel) path.

        ``attn_output`` is the framework context flattened to
        ``[tokens, local_heads * v_head_dim]`` (head-major), matching the flat
        layout of the rank-local gate projection, so the gate multiplies element
        wise per (head, value) exactly as K3 requires before o_proj.
        This runs before o_proj's TP all_reduce, so each rank gates only its
        local heads.
        """
        if not self.use_output_gate:
            return attn_output
        assert output_gate is not None
        return attn_output * torch.sigmoid(output_gate.reshape_as(attn_output))

    def _project_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        if self._sp_active_for_forward:
            tp_size = self.parallelism_config.get_attn_tp_size()
            return row_parallel_linear(
                attn_output,
                self._o_w,
                tp_size,
                reduce_scatter_tokens=True,
                pad_reduce_scatter_tokens=(
                    self._sp_padded_for_forward
                    or (
                        self._sp_prefill_input_is_sharded
                        and attn_output.shape[0] % tp_size != 0
                    )
                ),
                use_input_dtype_reduce_scatter=(self._sp_prefill_input_is_sharded),
            )
        return super()._project_output(attn_output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        sequence_parallel: bool = False,
        prefill_sp_layout: Optional[TokenShardLayout] = None,
    ) -> torch.Tensor:
        attn_inputs = _select_mla_attention_inputs(attention_inputs, fmha_impl)
        self._sp_active_for_forward = bool(
            sequence_parallel
            and self.parallelism_config.get_attn_tp_size() > 1
            and hidden_states.is_cuda
            and attn_inputs is not None
        )
        self._sp_prefill_input_is_sharded = prefill_sp_layout is not None
        self._sp_prefill_layout_for_forward = prefill_sp_layout
        if prefill_sp_layout is not None and (
            not self._sp_active_for_forward
            or attn_inputs is None
            or not attn_inputs.is_prefill
        ):
            raise ValueError(
                "prefill_sp_layout requires production CUDA MLA Prefill "
                "Sequence Parallel with TP>1"
            )
        self._sp_padded_for_forward = bool(
            self._sp_active_for_forward
            and attn_inputs is not None
            and not attn_inputs.is_prefill
        )
        if not hidden_states.is_cuda:
            raise RuntimeError("Kimi K3 MLA requires CUDA")
        try:
            return super().forward(hidden_states, fmha_impl, kv_cache)
        finally:
            self._sp_active_for_forward = False
            self._sp_padded_for_forward = False
            self._sp_prefill_input_is_sharded = False
            self._sp_prefill_layout_for_forward = None

__all__ = [
    "KimiK3MLA",
    "prepare_mla_fmha_for_group",
]

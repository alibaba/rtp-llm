from typing import Any, Optional

import torch

from rtp_llm.models_py.model_desc.generic_moe_mtp import GenericMoeMTPModel
from rtp_llm.models_py.model_desc.minimax_m3 import MiniMaxM3DecoderLayer
from rtp_llm.models_py.modules.hybrid.msa_attention import MSAAttention
from rtp_llm.ops.compute_ops import LayerKVCache


class _MiniMaxM3MTPRefreshContext:
    """Minimal runner contract for MiniMax-M3's sparse draft refresh.

    Sparse MSA owns RoPE, paged-cache writes, and attention metadata.  The
    generic dense FMHA object is therefore both unused and expensive to
    prepare.  CUDA Graph only requires this object to expose stable output
    metadata and a replay-prepare hook.
    """

    fmha_params = None

    def prepare_cuda_graph(self, _attn_inputs) -> None:
        return None


class MiniMaxM3MTPDecoderLayer(MiniMaxM3DecoderLayer):
    """MiniMax-M3 MTP layer with an explicit paged draft-refresh entry."""

    def _forward_attention(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache],
        prev_topk_indices: Optional[torch.Tensor],
        force_reuse_topk_indices: bool,
        attn_inputs: Optional[Any],
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
    ):
        if isinstance(fmha_impl, _MiniMaxM3MTPRefreshContext):
            if not isinstance(self.self_attn, MSAAttention):
                raise RuntimeError(
                    "MiniMax-M3 MTP draft refresh requires sparse MSA attention"
                )
            if kv_cache is None or attn_inputs is None:
                raise RuntimeError(
                    "MiniMax-M3 MTP draft refresh requires attention inputs and KV cache"
                )
            hidden_states = self.self_attn.forward_paged_continuation(
                hidden_states=hidden_states,
                attn_inputs=attn_inputs,
                kv_cache=kv_cache,
                x_fp8=x_fp8,
                x_scale=x_scale,
            )
            return hidden_states, None

        return super()._forward_attention(
            hidden_states,
            fmha_impl,
            kv_cache,
            prev_topk_indices,
            force_reuse_topk_indices,
            attn_inputs,
            x_fp8,
            x_scale,
        )


class MiniMaxM3MTPModel(GenericMoeMTPModel):
    """One recurrent MiniMax-M3 sparse-MSA/MoE MTP module."""

    decoder_layer_cls = MiniMaxM3MTPDecoderLayer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._draft_prefill_runtime = False

    def clone_for_cuda_graph(self):
        # MtpExecutor creates this clone exclusively for sp_prefill_draft_model_.
        # Model identity is therefore the reliable CG phase signal; ordinary
        # draft decode keeps using the original model instance.
        clone = super().clone_for_cuda_graph()
        clone._draft_prefill_runtime = True
        return clone

    def _is_draft_prefill(self, attention_inputs: Any) -> bool:
        return bool(getattr(self, "_draft_prefill_runtime", False)) or bool(
            getattr(attention_inputs, "is_mtp_draft_prefill", False)
        )

    def prepare_fmha_impl(self, inputs, is_cuda_graph: bool = False):
        if self._is_draft_prefill(inputs.attention_inputs):
            # The graph-owned attention object starts with default field values.
            # Mark it locally from the dedicated clone identity so MSA diagnostics
            # and model-owned phase checks remain truthful without GraphParams or
            # CudaGraphRunner propagation.
            inputs.attention_inputs.is_mtp_draft_prefill = True
            return _MiniMaxM3MTPRefreshContext()
        return super().prepare_fmha_impl(inputs, is_cuda_graph)

    def _mtp_iteration_step(self, inputs) -> int:
        if self._is_draft_prefill(inputs.attention_inputs):
            return 0
        return super()._mtp_iteration_step(inputs)

    def _mask_position_zero_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        fmha_impl: Any,
        attention_inputs: Any = None,
    ) -> torch.Tensor:
        """Apply MiniMax-M3's global position-zero mask after CP shuffling."""
        # A draft refresh runs only after target verification has committed a
        # non-empty prefix.  It cannot own global position zero, so avoid the
        # eager position/CP mask operator chain on this steady-state hot path.
        # Initial prompt prefill does not carry this phase flag and still takes
        # the full global-position correction below.
        if self._is_draft_prefill(attention_inputs):
            return inputs_embeds

        fmha_params = getattr(fmha_impl, "fmha_params", None)
        positions = getattr(fmha_params, "positions_d", None)
        if not torch.is_tensor(positions) or positions.numel() == 0:
            return inputs_embeds
        positions = positions.reshape(-1)
        if positions.size(0) != inputs_embeds.size(0):
            return inputs_embeds
        positions = positions.to(device=inputs_embeds.device)
        zero_mask = positions == 0

        cp_info = getattr(attention_inputs, "context_parallel_info", None)
        shuffle = getattr(cp_info, "prefill_shuffle_indices", None)
        chunk_lengths = getattr(cp_info, "prefill_cp_chunk_lengths", None)
        prefix_lengths = getattr(attention_inputs, "prefix_lengths", None)
        if not (
            torch.is_tensor(shuffle)
            and torch.is_tensor(chunk_lengths)
            and torch.is_tensor(prefix_lengths)
            and shuffle.numel() > 0
            and shuffle.numel() <= inputs_embeds.size(0)
            and chunk_lengths.numel() <= prefix_lengths.numel()
        ):
            return torch.where(zero_mask.unsqueeze(-1), 0, inputs_embeds)

        shuffle = shuffle.reshape(-1).to(device=inputs_embeds.device)
        chunk_lengths = chunk_lengths.reshape(-1).to(
            device=inputs_embeds.device, dtype=torch.long
        )
        prefill_streams = chunk_lengths.numel()
        prefill_prefixes = prefix_lengths.reshape(-1)[-prefill_streams:].to(
            device=inputs_embeds.device
        )
        row_prefixes = torch.repeat_interleave(
            prefill_prefixes, chunk_lengths, output_size=shuffle.numel()
        )
        if row_prefixes.numel() == shuffle.numel():
            prefill_zero_mask = (shuffle == 0) & (row_prefixes == 0)
            decode_rows = inputs_embeds.size(0) - shuffle.numel()
            zero_mask = torch.cat([zero_mask[:decode_rows], prefill_zero_mask], dim=0)
        return torch.where(zero_mask.unsqueeze(-1), 0, inputs_embeds)

"""Target-side MTP integration for Kimi K3.

The draft model lives in ``model_desc.kimi_k3_eagle3``.  This mixin keeps
Eagle3 auxiliary-hidden capture and chunk-Prefill hand-off state out of the
target model's operator graph while leaving a single Kimi K3 target forward.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence

import torch

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather_into,
    all_gather_trim,
)
from rtp_llm.models_py.distributed.sequence_parallel import SequenceParallelLayout
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInitResources


class KimiK3MTPTargetMixin:
    """Own target-only state and hooks required by the Eagle3 draft model."""

    def _initialize_mtp_state(self) -> None:
        self._mtp_hidden_buffer: Optional[torch.Tensor] = None
        self._mtp_hidden_valid_tokens = 0
        self._prefill_mtp_hidden_workspace: Optional[torch.Tensor] = None
        self._prefill_mtp_draft_workspace: Optional[torch.Tensor] = None
        self._whole_chunk_prefill_active = False

    def _initialize_mtp_runtime(self, init_resource: PyModelInitResources) -> None:
        if not self._is_decode_role or not self._mtp_runtime_enabled():
            return
        tokens_per_batch = max(int(self.config.gen_num_per_cycle) + 1, 1)
        graph_batch_capacity = int(
            getattr(init_resource, "max_decode_graph_batch_size", 1)
        )
        token_capacity = (
            max(self._max_generate_batch_size, graph_batch_capacity)
            * tokens_per_batch
        )
        self._mtp_hidden_buffer = self.embedding_weight.new_empty(
            token_capacity,
            3 * int(self.config.hidden_size),
        )
        logging.info(
            "[K3_EAGLE3] allocated Decode hidden buffer shape=%s",
            tuple(self._mtp_hidden_buffer.shape),
        )

    @staticmethod
    def _mtp_runtime_enabled() -> bool:
        return os.environ.get("SP_TYPE", "").lower() == "eagle3"

    def _mtp_aux_layer_ids(self, inputs: Any) -> tuple[int, ...]:
        """Return target layers whose outputs are consumed by Eagle3."""

        if not self._mtp_runtime_enabled() or bool(
            getattr(inputs, "force_disable_sp_run", False)
        ):
            return ()
        raw_aux_layers = os.environ.get("KIMI_K3_EAGLE3_AUX_LAYER_IDS")
        if raw_aux_layers:
            layer_ids = tuple(int(value) for value in raw_aux_layers.split(","))
        else:
            layer_ids = (0, max(0, self.layer_num // 2), self.layer_num - 1)
        if len(layer_ids) != 3 or any(
            layer_id < 0 or layer_id >= self.layer_num for layer_id in layer_ids
        ):
            raise ValueError(
                "KIMI_K3_EAGLE3_AUX_LAYER_IDS must contain three valid "
                f"zero-based layer ids for {self.layer_num} target layers"
            )
        return layer_ids

    def _publish_mtp_aux_hidden_states(
        self,
        hidden_states_by_layer: dict[int, torch.Tensor],
        layer_ids: Sequence[int],
        *,
        is_prefill: bool,
        token_layout: SequenceParallelLayout,
        attention_inputs: PyAttentionInputs,
    ) -> None:
        if not layer_ids:
            return
        mtp_hidden = torch.cat(
            [hidden_states_by_layer[layer_id] for layer_id in layer_ids],
            dim=-1,
        ).contiguous()
        if is_prefill and self._whole_chunk_prefill_active:
            physical_tokens = (
                int(mtp_hidden.size(0))
                * int(self.parallelism_config.get_attn_tp_size())
            )
            gathered = self._prefill_mtp_hidden_workspace.narrow(
                0, 0, physical_tokens
            )
            all_gather_into(mtp_hidden, gathered, group=Group.TP)
            self._mtp_hidden_buffer = gathered.narrow(
                0, 0, token_layout.logical_tokens
            )
            self._mtp_hidden_valid_tokens = token_layout.logical_tokens
            return
        mtp_hidden = all_gather_trim(
            mtp_hidden,
            token_layout.logical_tokens,
            group=Group.TP,
        )
        self._write_mtp_hidden_buffer(
            mtp_hidden,
            is_cuda_graph=(
                bool(getattr(attention_inputs, "is_cuda_graph", False))
                or (
                    mtp_hidden.is_cuda
                    and torch.cuda.is_current_stream_capturing()
                )
            ),
        )

    def _write_mtp_hidden_buffer(
        self, hidden_states: torch.Tensor, *, is_cuda_graph: bool
    ) -> None:
        rows = int(hidden_states.size(0))
        if self._is_decode_role:
            buffer = self._mtp_hidden_buffer
            # CUDA Graph replay does not execute Python. Every captured graph must
            # write into the same model-owned storage instead of replacing this
            # attribute with a tensor owned by one captured batch shape.
            buffer.narrow(0, 0, rows).copy_(hidden_states)
        else:
            # Prefill is not graph-captured and may be much larger than Decode.
            self._mtp_hidden_buffer = hidden_states
        if not is_cuda_graph:
            self._mtp_hidden_valid_tokens = rows

    def get_mtp_target_hidden_states(self, num_tokens: int) -> Optional[torch.Tensor]:
        if self._mtp_hidden_buffer is None:
            return None
        rows = (
            self._mtp_hidden_valid_tokens if int(num_tokens) < 0 else int(num_tokens)
        )
        return self._mtp_hidden_buffer.narrow(0, 0, rows)

    def abort_prefill_chunk_session(self) -> None:
        """Drop MTP state retained by an interrupted whole-chunk session."""

        from rtp_llm.models_py.modules.kimi_k3.chunk_prefill import (
            reset_kimi_k3_chunk_prefill,
        )

        reset_kimi_k3_chunk_prefill(self)
        self._mtp_hidden_buffer = None
        self._mtp_hidden_valid_tokens = 0


__all__ = ["KimiK3MTPTargetMixin"]

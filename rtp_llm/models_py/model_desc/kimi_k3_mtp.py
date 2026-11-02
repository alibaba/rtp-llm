"""Native recurrent K3 nextn layer; logits and recurrence have distinct outputs."""

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model
from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class KimiK3MtpModel(KimiK3Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("_mtp_prefill_last_hidden_buffer", None, persistent=False)
        self._mtp_prefill_last_hidden_valid_rows = 0
        self.media_token_id = self.config.k3_runtime_config.media_placeholder_token_id
        weights = self.weight.weights[0]
        # The BF16 MLA prefix path uses skip-head-mid KV-up. Keep its source
        # [in, out] weight contiguous so the linear's [out, in] transpose has
        # the layout required by the packed GEMM, including after Host reuse.
        weights[W.mla_kv_b_w] = weights[W.mla_kv_b_w].contiguous()
        eps = self.config.layernorm_eps
        self.enorm = RMSNorm(weights["kimi_k3.mtp.enorm"], eps)
        self.hnorm = RMSNorm(weights["kimi_k3.mtp.hnorm"], eps)
        self.eh_proj = LinearFactory.create_linear_from_weights(
            weights, "kimi_k3.mtp.eh_proj", hw_kernel_config=self.py_hw_kernel_config
        )

    def initialize(self, init_resource):
        ready = super().initialize(init_resource)
        self._initialize_mtp_last_handoff(init_resource)
        if self.tp_size > 1 and init_resource.is_decode_role:
            from rtp_llm.models_py.distributed.collective_torch import (
                Bf16TpCollectives,
            )

            max_tokens = self._max_moe_tokens
            attention = self.layers[0].attention
            # Q1 proposal and fixed-Q update run serially on the compute stream.
            # Keep MTP's BF16 boundary gathers independent of target FP8
            # projections and the global custom-AllReduce policy.
            # Proposal and update runners initialize this shared model.
            # Captured kernels retain raw pointers into the first workspace.
            existing = attention._mtp_bf16_collectives
            if existing is None:
                attention._mtp_bf16_collectives = Bf16TpCollectives(
                    attention.input.weight.device,
                    max_tokens=max_tokens,
                    hidden_size=self.config.hidden_size,
                )
            elif (
                existing.tp_size != self.tp_size
                or existing.hidden_size != self.config.hidden_size
                or existing.max_tokens < max_tokens
            ):
                raise ValueError(
                    "Cannot replace MTP workspaces retained by CUDA Graphs"
                )
        return ready

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False, cuda_graph_selection_mode=None):
        fmha = super().prepare_fmha_impl(inputs, is_cuda_graph, cuda_graph_selection_mode)
        if self._query_layout is not None and self.has_mtp_hidden_buffer():
            self._query_layout.prepare_last_hidden_rows()
        return fmha

    def _gather_modeling_boundary(self, local_input):
        if self.tp_size == 1:
            return local_input
        collectives = self.layers[0].attention._mtp_bf16_collectives
        if collectives is not None:
            return collectives.all_gather_modeling_boundary(local_input)
        return all_gather(local_input, Group.TP)

    def _embed_inputs(self, inputs: PyModelInputs, *, tp_gather=None) -> torch.Tensor:
        """Restore media placeholders in the already shifted draft token layout."""
        ids = inputs.input_ids
        if inputs.multimodal_inputs.multimodal_features:
            text_mask = inputs.embedding_inputs.text_tokens_mask
            if text_mask is None or text_mask.shape != ids.shape:
                raise ValueError(
                    "K3 MTP image features require one text mask entry per input token"
                )
            if self.media_token_id is None:
                raise ValueError(
                    "K3 MTP media tokens require media_placeholder_token_id"
                )
            # C++ has shifted the mask per request; preserve the original hash IDs.
            ids = ids.masked_fill(text_mask == 0, self.media_token_id)
        return self.embed_tokens(ids, tp_gather=tp_gather)

    def _project_local_mtp_input(self, inputs):
        physical_rows = inputs.input_ids.shape[0]
        layout = getattr(self, "_query_layout", None)
        if layout is not None:
            positions = inputs.combo_position_ids
            if positions is None or positions.numel() == 0:
                # The shared text gatherer omits explicit positions for K3.
                # CP's prepared map still describes real global positions,
                # including prefix reuse and invalid padding rows.
                positions = layout.local_positions
            if (
                physical_rows != layout.local_rows
                or inputs.input_hiddens.shape[0] != physical_rows
                or positions.numel() != physical_rows
            ):
                raise ValueError("MTP CP inputs must share the target rank-local rows")
            # The CP processor already selected tokens/positions and retained
            # this rank's target hidden. Replicated embedding needs no TP gather
            # and these rows must not be divided by TP a second time.
            embedded = self.embed_tokens(inputs.input_ids)
            valid = (positions.reshape(-1, 1) != 0) & layout.local_valid[:, None]
            embedded = embedded * valid
            return self.eh_proj(torch.cat((
                self.enorm(embedded), self.hnorm(inputs.input_hiddens.contiguous())
            ), dim=-1))
        if physical_rows % self.tp_size:
            raise ValueError("Native K3 MTP requires physical token padding for SP")
        local_rows = physical_rows // self.tp_size
        local_start = self.tp_rank * local_rows
        if inputs.input_hiddens.shape[0] != physical_rows:
            raise ValueError(
                "Native K3 MTP requires a hidden state for every physical token"
            )
        # The embedding gathers hidden shards across TP. Project only this
        # rank's token rows after that gather, before entering the SP layer.
        embedded = self._embed_inputs(inputs, tp_gather=self._gather_modeling_boundary)
        embedded = embedded.narrow(0, local_start, local_rows).contiguous()
        previous_h = inputs.input_hiddens.narrow(0, local_start, local_rows)
        return self.eh_proj(
            torch.cat(
                (self.enorm(embedded), self.hnorm(previous_h.contiguous())), dim=-1
            )
        )

    def _forward_single(self, inputs, fmha_impl=None):
        hidden = self._project_local_mtp_input(inputs)
        recurrent = self._forward_layers(
            hidden, inputs, fmha_impl, sequence_parallel_input=True, gather_output=False
        )
        if self._query_layout is None:
            recurrent = self._gather_modeling_boundary(recurrent)
        else:
            self._publish_mtp_last_hidden(recurrent)
        return PyModelOutputs(self.norm(recurrent), recurrent)

    def _initialize_mtp_last_handoff(self, init_resource):
        if not self.has_mtp_hidden_buffer():
            return
        capacity = max(self._max_generate_batch_size, int(init_resource.max_context_batch_size))
        shape = (capacity, self.config.hidden_size)
        buffer = self._mtp_prefill_last_hidden_buffer
        if buffer is None:
            self._mtp_prefill_last_hidden_buffer = torch.empty(
                shape, dtype=self.norm.weight.dtype, device=self.norm.weight.device
            )
        elif tuple(buffer.shape) != shape:
            raise ValueError("Cannot replace initialized MTP last-hidden storage")


    def _publish_mtp_last_hidden(self, recurrent):
        buffer = getattr(self, "_mtp_prefill_last_hidden_buffer", None)
        if buffer is None:
            return
        layout = self._query_layout
        rows = len(layout.lengths)
        if rows > buffer.shape[0]:
            raise ValueError("MTP per-request last-hidden capacity exceeded")
        local_last = layout.local_last_hidden(recurrent)
        # Only [requests, hidden] rows are gathered, with exactly one nonzero
        # owner per request. Never gather the complete local hidden sequence.
        gathered = all_gather(local_last, Group.TP) if self.tp_size > 1 else local_last
        torch.sum(gathered.view(self.tp_size, rows, recurrent.shape[1]), dim=0, out=buffer[:rows])
        self._mtp_prefill_last_hidden_valid_rows = rows


    def get_mtp_last_hidden_states(self, num_tokens):
        buffer = getattr(self, "_mtp_prefill_last_hidden_buffer", None)
        if buffer is None:
            return None
        valid = self._mtp_prefill_last_hidden_valid_rows
        rows = valid if num_tokens < 0 else int(num_tokens)
        if valid <= 0 or rows < 0 or rows > valid:
            raise ValueError(f"MTP last-hidden rows requested={rows}, written={valid}")
        return buffer[:rows]

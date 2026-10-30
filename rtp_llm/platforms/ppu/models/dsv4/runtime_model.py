"""PPU-owned runtime buffer sizes and operator binding."""

import torch

from rtp_llm.platforms.ppu.models.dsv4.impl.model import DeepSeekV4Model, Dsv4SharedRuntimeBufferStore, Dsv4MtpHiddenBufferSpec
from rtp_llm.platforms.ppu.models.dsv4.impl.prefill_workspace import tp_local_prefill_q_dim
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import resolve_moe_max_tokens_per_rank

from rtp_llm.platforms.ppu.models.dsv4.manifest import build_provider


class PpuDeepSeekV4Model(DeepSeekV4Model):
    def __init__(self, model_config, parallelism_config, weights, moe_config, max_generate_batch_size, **kwargs):
        provider = kwargs.pop("platform_provider", None)
        if provider is None:
            provider = build_provider(model_config, parallelism_config, moe_config, max_generate_batch_size)
        super().__init__(model_config, parallelism_config, weights, moe_config, max_generate_batch_size,
                         platform_provider=provider, **kwargs)
        self._prefill_scheduler_chunked = (getattr(model_config, "prefill_chunk_size", 0) or 0) > 0
        cap = getattr(model_config, "moe_prefill_max_tokens_per_rank", None)
        max_seq_len = int(getattr(model_config, "max_seq_len", 0) or 0)
        self._prefill_scheduler_token_capacity = int(cap if cap is not None else (max_seq_len or 4096))

    def _resolve_shared_token_capacity(self) -> int:
        if not self._is_decode_role:
            return super()._resolve_shared_token_capacity()
        return resolve_moe_max_tokens_per_rank(
            max_seq_len=int(self._v4_args.max_seq_len),
            current_max_tokens_per_rank=int(self._v4_args.max_tokens_per_rank),
            cp_size=1,
            max_generate_batch_size=int(self._max_generate_batch_size),
            is_decode_role=True,
            is_speculative=self._is_speculative,
            gen_num_per_cycle=self._gen_num_per_cycle,
            chunking_enabled=False,
        )

    def _resolve_mtp_hidden_token_capacity(self) -> int:
        decode_capacity = self._max_generate_batch_size * (
            max(self._gen_num_per_cycle + 1, 1) if self._is_speculative else 1
        )
        return max(self._resolve_prefill_q_token_capacity(), decode_capacity)

    def _resolve_prefill_q_token_capacity(self) -> int:
        if self._is_decode_role:
            return self._resolve_shared_token_capacity()
        budget = self._prefill_scheduler_token_capacity
        if not self._prefill_scheduler_chunked:
            # FIFO admits a singleton longer than the batch token budget.
            budget = max(budget, int(self._v4_args.max_seq_len))
        padded_capacity = resolve_moe_max_tokens_per_rank(
            max_seq_len=int(self._v4_args.max_seq_len),
            current_max_tokens_per_rank=budget,
            cp_size=self._prefill_cp_size,
            max_generate_batch_size=self._max_generate_batch_size,
            max_context_batch_size=self._max_context_batch_size,
            chunking_enabled=False,
        )
        return min(
            int(self._resolve_shared_token_capacity()),
            padded_capacity,
        )

    def _resolve_prefill_q_dim(self) -> int:
        """Return the TP-local dense-Q width backed by the prefill workspace."""
        return tp_local_prefill_q_dim(
            self._v4_args.n_heads,
            self._v4_args.head_dim,
            self._v4_args.tp_size,
        )

    def _bind_runtime_buffers(self, device: torch.device) -> None:
        assert self.v4 is not None
        mtp_hidden = None
        mtp_last_hidden_capacity = None
        mtp_hidden_enabled = self._is_speculative
        if mtp_hidden_enabled:
            # MTP rows are the pre-hc residual (hc_mult*dim); DSpARK rows are
            # the captured aux features (len(capture_ids)*dim). Target and
            # draft carry the same capture ids in their configs, so both
            # derive the same width regardless of bind order.
            hc_dim = (
                len(self._capture_aux_hidden_layer_ids) * int(self._v4_args.dim)
                if self._capture_aux_hidden_layer_ids
                else int(self._v4_args.hc_mult) * int(self._v4_args.dim)
            )
            mtp_hidden = Dsv4MtpHiddenBufferSpec(
                token_capacity=self._resolve_mtp_hidden_token_capacity(),
                hc_dim=hc_dim,
            )
            if self._is_speculative:
                mtp_last_hidden_capacity = (
                    self._resolve_mtp_last_hidden_token_capacity()
                )

        # Per-forward prefill workspace dims (max-sized; consumed by
        # ``forward_layers`` to build a per-forward ``PrefillWorkspace``). These
        # are plain ints — config, not buffers — so they carry no cross-forward
        # lifetime. CP gather/restore region is sized only when CP is active.
        cp_size = int(self._prefill_cp_size)
        q_rows = int(self._resolve_prefill_q_token_capacity())
        # Attention Q is row-sharded by heads under tensor parallelism.  The
        # workspace backs the local ``wq_b`` output, so sizing it with the
        # global head count makes ``_materialize_prefill_q`` try to reinterpret
        # a TP-size-larger buffer as the local Q tensor (for TP8: 32768 vs
        # 4096 elements per token).  Keep TP1 unchanged and fail loudly on an
        # invalid head partition.
        q_dim = self._resolve_prefill_q_dim()
        if cp_size > 1:
            full_rows = q_rows * cp_size
            main_w, idx_w = self._resolve_prefill_ws_gather_widths()
        else:
            full_rows = 0
            main_w = 0
            idx_w = 0
        self.v4._bind_prefill_workspace_dims(q_rows, q_dim, full_rows, main_w, idx_w)

        if self._shared_runtime_buffers is None:
            self._shared_runtime_buffers = Dsv4SharedRuntimeBufferStore(
                device=device,
                dtype=torch.bfloat16,
                mtp_hidden_enabled=mtp_hidden_enabled,
                mtp_hidden=mtp_hidden,
            )
        self._shared_runtime_buffers.bind(self.v4)

        if mtp_last_hidden_capacity is not None:
            self.v4._allocate_mtp_last_hidden_buffer(
                device,
                mtp_last_hidden_capacity,
            )

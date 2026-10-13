"""DeepSeek-V4 MTP draft model.

Thin subclass of :class:`DeepSeekV4Model`.  The MTP draft is structurally
a single regular V4 ``Block`` plus four MTP-only fusion tensors
(``enorm`` / ``hnorm`` / ``e_proj`` / ``h_proj``).  We keep the Block
inside the inherited ``self.v4`` (its ``W.*`` keys are the same as the
main model — see ``DeepSeekV4MtpWeight._get_weight_info``), and host the
fusion modules at the model level so the only code unique to this class
is the ``e_proj(enorm(masked_embed)) + h_proj(hnorm(prev_hidden))`` step
that produces the layer-loop input.

The rest — initialize, prepare_fmha_impl, forward dispatch, mHC reduce,
final norm, ``_mtp_hidden_buffer`` accessor — falls through to
``DeepSeekV4Model`` unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.deepseek_v4_model import (
    DeepSeekV4Model,
    Dsv4SharedRuntimeBufferStore,
)
from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.dsv4.chunk_env import (
    DEFAULT_DSV4_CHUNK_TOKENS,
    dsv4_chunk_tokens_from_env,
)
from rtp_llm.models_py.modules.dsv4.utils import _v4_fp8_linear
from rtp_llm.utils.model_weight import W


class DeepSeekV4MtpModel(DeepSeekV4Model):
    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config,
        weights: ModelWeights,
        moe_config,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            moe_config,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        Dsv4SharedRuntimeBufferStore.enable_mtp_hidden()
        # MTP overrides for V4Args. ``DeepSeekV4Mtp._create_config``
        # already sets ``num_layers=1`` and ``layer_compress_ratios=[0]``
        # on the ModelConfig; we additionally drop the hash-router count
        # so the lone draft layer always picks the noaux_tc path.
        self._v4_args.n_hash_layers = 0
        self._v4_args.compress_ratios = (
            [int(self._v4_args.compress_ratios[0])]
            if (self._v4_args.compress_ratios)
            else [0]
        )
        logging.info(
            "[DeepSeekV4MtpModel] V4Args: layers=%d hc_mult=%d compress_ratios=%s "
            "fp8_kv_cache=%s max_tokens_per_rank=%d",
            self._v4_args.n_layers,
            self._v4_args.hc_mult,
            list(self._v4_args.compress_ratios),
            self._v4_args.fp8_kv_cache,
            self._v4_args.max_tokens_per_rank,
        )

        # MTP-only fusion modules — populated in ``_load_extra_weights``.
        self.enorm: Optional[RMSNorm] = None
        self.hnorm: Optional[RMSNorm] = None
        self.e_proj = None
        self.h_proj = None
        self._mtp_fusion_chunk_logged = False

    def _resolve_mtp_last_hidden_token_capacity(self) -> Optional[int]:
        # Parent allocation is already gated by self._is_speculative.  Any
        # non-decode draft path, including PDFUSION, needs the CP prefill
        # per-request last-hidden handoff.
        if self._is_decode_role:
            return None
        return max(self._max_context_batch_size, self._max_generate_batch_size, 1)

    # ------------------------------------------------------------------
    # CUDA-graph gate — accept all cudagraph requests.  C++ MtpExecutor
    # creates two draft PyWrappedModel instances: ``draft_model_``
    # (q_len=1) and ``sp_prefill_draft_`` (q_len=gen+1).  The latter is
    # marked ``is_prefill=True`` even though it's functionally a
    # multi-token decode — the parent's default would reject it.
    # Both captures route through ``forward_decode`` + MTP's
    # ``_prepare_decode_hidden`` (``T = B*q_len``).  This requires
    # ``prepareCaptureInputs`` to size ``input_ids`` / ``input_hiddens``
    # and the output buffer to ``max_bs * num_tokens_per_bs`` (full
    # capacity), NOT the per-capture ``seq_len`` — handled in
    # ``cuda_graph_runner.cc`` / ``cuda_graph_prefill.cc`` for draft
    # prefill mode (num_tokens_per_bs != max_seq_len).
    # ------------------------------------------------------------------

    def _should_capture_cuda_graph(self, attn, is_target_verify: bool) -> bool:
        return True

    # ------------------------------------------------------------------
    # Weight loading hook (called by parent's _initialize_impl right
    # before ``del self.weight``).
    # ------------------------------------------------------------------

    def _load_extra_weights(self, weights: ModelWeights) -> None:
        gw = weights.global_weights
        eps = float(self._v4_args.norm_eps)
        self.enorm = RMSNorm(gw[W.v4_mtp_enorm], eps)
        self.hnorm = RMSNorm(gw[W.v4_mtp_hnorm], eps)
        self.e_proj = _v4_fp8_linear(gw[W.v4_mtp_e_proj_w], gw[W.v4_mtp_e_proj_s])
        self.h_proj = _v4_fp8_linear(gw[W.v4_mtp_h_proj_w], gw[W.v4_mtp_h_proj_s])

    # ------------------------------------------------------------------
    # Hidden-state preparation overrides — splice the e/h fusion stage in
    # front of the inherited layer loop.
    # ------------------------------------------------------------------

    def _apply_proj(self, layer, x: torch.Tensor) -> torch.Tensor:
        """``_v4_fp8_linear``-built layers want 2D input; reshape N-D
        tensors round-trip so callers can keep their natural rank."""
        if x.dim() > 2:
            shape = x.shape
            return layer(x.reshape(-1, shape[-1])).view(*shape[:-1], -1)
        return layer(x)

    def _mtp_fusion_chunk_tokens(self) -> int:
        return dsv4_chunk_tokens_from_env(
            "DSV4_MTP_FUSION_CHUNK_TOKENS",
            DEFAULT_DSV4_CHUNK_TOKENS,
            min_value=1,
        )

    def _build_fused_chunked(
        self,
        input_ids: torch.Tensor,
        pre_hc: torch.Tensor,
        positions: torch.Tensor,
        chunk_tokens: int,
    ) -> torch.Tensor:
        assert self.enorm is not None
        assert self.hnorm is not None
        assert self.e_proj is not None
        assert self.h_proj is not None

        T, hc, dim = pre_hc.shape
        # In-place reuse: pre_hc is either a fresh CP index_select tensor
        # (ContextParallelProcessorBase) or an allocBuf'd one-shot input
        # buffer (non-CP); neither has other live readers in this step.
        # The chunk loop reads pre_hc[start:end] into h_norm before writing
        # back to the same slice, so the per-chunk in-place copy is safe.
        # CUDA graph capture/replay is excluded by the caller guard.
        # Saves ~6 GiB peak at 1M ctx (T=125k, hc*dim=7*7168 bf16).
        fused = pre_hc
        if not self._mtp_fusion_chunk_logged:
            self._mtp_fusion_chunk_logged = True
            logging.info(
                "[DeepSeekV4MtpModel] chunked MTP fusion enabled: tokens=%d "
                "chunk_tokens=%d hc=%d dim=%d device=%s",
                T,
                chunk_tokens,
                hc,
                dim,
                pre_hc.device,
            )
        for start in range(0, T, chunk_tokens):
            end = min(start + chunk_tokens, T)
            input_ids_chunk = input_ids[start:end]
            positions_chunk = positions[start:end]
            embed_chunk = self.v4.embed(input_ids_chunk)
            embed_chunk = torch.where(
                positions_chunk.reshape(-1, 1) == 0,
                torch.zeros_like(embed_chunk),
                embed_chunk,
            )
            e_norm = self.enorm(embed_chunk)
            pre_hc_chunk = pre_hc[start:end]
            chunk_len = int(pre_hc_chunk.size(0))
            h_norm = self.hnorm(pre_hc_chunk.reshape(-1, dim)).view(chunk_len, hc, dim)
            fused_chunk = self._apply_proj(self.h_proj, h_norm)
            fused_chunk.add_(self._apply_proj(self.e_proj, e_norm).unsqueeze(1))
            fused[start:end].copy_(fused_chunk)
        return fused

    def _build_fused(
        self,
        input_ids: torch.Tensor,  # [T] int
        pre_hc: torch.Tensor,  # [T, hc, dim] bf16
        positions: torch.Tensor,  # [T] int (mask token at position 0)
    ) -> torch.Tensor:
        """``e_proj(enorm(masked_embed)) + h_proj(hnorm(prev_hidden))``.
        Returns ``[T, hc, dim]``."""
        assert self.enorm is not None
        assert self.hnorm is not None
        assert self.e_proj is not None
        assert self.h_proj is not None
        T, hc, dim = pre_hc.shape
        chunk_tokens = self._mtp_fusion_chunk_tokens()
        if T > chunk_tokens and not torch.cuda.is_current_stream_capturing():
            return self._build_fused_chunked(
                input_ids.reshape(-1), pre_hc, positions[:T], chunk_tokens
            )

        inputs_embeds = self.v4.embed(input_ids)  # [T, dim]
        # Suppress position-0 embedding (matches main-model "step 0 of a
        # brand-new request" behavior the official MTP impl relies on).
        inputs_embeds = torch.where(
            positions.reshape(-1, 1) == 0,
            torch.zeros_like(inputs_embeds),
            inputs_embeds,
        )
        e_norm = self.enorm(inputs_embeds)  # [T, dim]
        h_norm = self.hnorm(pre_hc.reshape(-1, dim)).view(T, hc, dim)
        return self._apply_proj(self.h_proj, h_norm) + self._apply_proj(
            self.e_proj, e_norm
        ).unsqueeze(
            1
        )  # [T, hc, dim]

    def _pre_hc_from_inputs(self, inputs, T: int) -> torch.Tensor:
        pre_hc_in = inputs.input_hiddens
        if pre_hc_in is None or pre_hc_in.numel() == 0:
            raise RuntimeError(
                "DeepSeekV4MtpModel expected pre-hc hidden states in input_hiddens"
            )
        hc = int(self._v4_args.hc_mult)
        dim = int(self._v4_args.dim)
        pre_hc = pre_hc_in.reshape(-1, pre_hc_in.size(-1))
        if int(pre_hc.size(-1)) != hc * dim:
            raise RuntimeError(
                f"DeepSeekV4MtpModel expected hidden dim {hc * dim}, "
                f"got {pre_hc.size(-1)}"
            )
        # CP layout is handled before Python sees the tensors: C++
        # handleInputs splits input_hiddens with the same zigzag plan as
        # input_ids.  This method only trims CUDA graph capacity to real T.
        if pre_hc.size(0) < T:
            raise RuntimeError(
                f"DeepSeekV4MtpModel: input_hiddens has {pre_hc.size(0)} rows "
                f"but {T} tokens required"
            )
        return pre_hc[:T].view(T, hc, dim).to(device=self.v4.embed.weight.device)

    def _prepare_decode_hidden(
        self,
        input_ids: torch.Tensor,
        meta: Any,
    ) -> torch.Tensor:
        B = int(meta.batch_size)
        q_len = int(meta.q_len_per_req)
        T = B * q_len
        hc = int(self._v4_args.hc_mult)
        dim = int(self._v4_args.dim)
        pre_hc = self._pre_hc_from_inputs(self._cur_inputs, T)
        positions = meta.position_ids[:T]
        fused = self._build_fused(input_ids.reshape(-1), pre_hc, positions)
        return fused.view(B, q_len, hc, dim)

    def _prepare_prefill_hidden(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        T = int(input_ids.numel())
        pre_hc = self._pre_hc_from_inputs(self._cur_inputs, T)
        return self._build_fused(input_ids.reshape(-1), pre_hc, positions[:T])

    # ------------------------------------------------------------------
    # forward — delegate to parent, just stash ``inputs`` so the prepare
    # hooks can pull ``input_hiddens`` off it.
    # ------------------------------------------------------------------

    def forward(self, inputs, fmha_impl: Any = None):
        self._cur_inputs = inputs
        try:
            return super().forward(inputs, fmha_impl)
        finally:
            self._cur_inputs = None


__all__ = ["DeepSeekV4MtpModel"]

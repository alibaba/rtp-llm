"""PPU MTP projections and hidden-shard embedding fusion."""

import logging

import torch

from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.deepseek_v4_mtp_model import DeepSeekV4MtpModel
from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.platforms.ppu.models.dsv4.impl.utils import _v4_fp8_linear
from rtp_llm.utils.model_weight import W

from rtp_llm.platforms.ppu.models.dsv4.runtime_model import PpuDeepSeekV4Model
from rtp_llm.platforms.ppu.models.dsv4.impl.platform_provider import Dsv4ProviderCapability


class PpuDeepSeekV4MtpModel(DeepSeekV4MtpModel, PpuDeepSeekV4Model):
    # The generic MTP class appears first in the MRO because it owns the
    # draft-specific forward. Explicitly route capacity and buffer binding to
    # the PPU implementation; otherwise the draft sizes its workspace from
    # its own max_seq_len instead of the shared PPU scheduler budget.
    def _resolve_shared_token_capacity(self) -> int:
        return PpuDeepSeekV4Model._resolve_shared_token_capacity(self)

    def _resolve_mtp_hidden_token_capacity(self) -> int:
        return PpuDeepSeekV4Model._resolve_mtp_hidden_token_capacity(self)

    def _resolve_prefill_q_token_capacity(self) -> int:
        return PpuDeepSeekV4Model._resolve_prefill_q_token_capacity(self)

    def _resolve_prefill_q_dim(self) -> int:
        return PpuDeepSeekV4Model._resolve_prefill_q_dim(self)

    def _bind_runtime_buffers(self, device: torch.device) -> None:
        PpuDeepSeekV4Model._bind_runtime_buffers(self, device)

    def _load_extra_weights(self, weights: ModelWeights) -> None:
        gw = weights.global_weights
        eps = float(self._v4_args.norm_eps)
        self.enorm = RMSNorm(gw[W.v4_mtp_enorm], eps)
        self.hnorm = RMSNorm(gw[W.v4_mtp_hnorm], eps)
        self.e_proj = _v4_fp8_linear(
            gw[W.v4_mtp_e_proj_w],
            gw[W.v4_mtp_e_proj_s],
            platform_provider=self._platform_provider,
        )
        self.h_proj = _v4_fp8_linear(
            gw[W.v4_mtp_h_proj_w],
            gw[W.v4_mtp_h_proj_s],
            platform_provider=self._platform_provider,
        )
        self._mtp_input_fusion = None
        if Dsv4ProviderCapability.MTP_INPUT_FUSION in self._platform_provider.capabilities:
            self._mtp_input_fusion = self._platform_provider.build_mtp_input_fusion(
                self.enorm, self.hnorm, self.e_proj, self.h_proj
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
            embed_chunk = self.v4._embed(input_ids_chunk)
            embed_chunk = torch.where(
                positions_chunk.reshape(-1, 1) == 0,
                torch.zeros_like(embed_chunk),
                embed_chunk,
            )
            e_norm = self.enorm(embed_chunk.contiguous())
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

        # The shared embedding can be sharded along hidden size under TP.
        # Use the target's gather path before the full-width fusion norms.
        inputs_embeds = self.v4._embed(input_ids)  # [T, dim]
        fusion = getattr(self, "_mtp_input_fusion", None)
        if fusion is not None:
            fused = fusion(inputs_embeds, pre_hc, positions)
            if fused is not None:
                return fused
        # Suppress position-0 embedding (matches main-model "step 0 of a
        # brand-new request" behavior the official MTP impl relies on).
        inputs_embeds = torch.where(
            positions.reshape(-1, 1) == 0,
            torch.zeros_like(inputs_embeds),
            inputs_embeds,
        )
        e_norm = self.enorm(inputs_embeds.contiguous())  # [T, dim]
        h_norm = self.hnorm(pre_hc.reshape(-1, dim)).view(T, hc, dim)
        return self._apply_proj(self.h_proj, h_norm) + self._apply_proj(
            self.e_proj, e_norm
        ).unsqueeze(
            1
        )  # [T, hc, dim]

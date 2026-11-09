"""PPU MTP projections and hidden-shard embedding fusion."""

import os

import torch

from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.deepseek_v4_mtp_model import DeepSeekV4MtpModel
from rtp_llm.models_py.model_desc.deepseek_v4_ppu.model import PpuDeepSeekV4Model
from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.dsv4.ppu.utils import _v4_fp8_linear
from rtp_llm.utils.model_weight import W


class PpuDeepSeekV4MtpModel(DeepSeekV4MtpModel, PpuDeepSeekV4Model):
    def prepare_fmha_impl(
        self, inputs, is_cuda_graph=False, cuda_graph_selection_mode=None
    ):
        impl = super().prepare_fmha_impl(
            inputs, is_cuda_graph, cuda_graph_selection_mode
        )
        if impl is None or os.environ.get("DSV4_PPU_MTP_COMMIT_LAYOUT") != "1":
            return impl
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
            primary_attention_inputs,
        )
        from rtp_llm.models_py.modules.dsv4.ppu.metadata.ppu_decode_metadata import (
            PpuDecodeMetadataGraph,
        )

        attn = primary_attention_inputs(inputs.attention_inputs, self.kv_cache)
        if (
            isinstance(impl, PpuDecodeMetadataGraph)
            and impl.config.q_len in (3, 4)
            and self._is_decode_role
            and impl.metadata.active_token_mask is not None
            and attn.is_prefill
            and not attn.is_target_verify
        ):
            impl.enable_mtp_commit_layout(attn.input_lengths_device)
        return impl

    def _pre_hc_from_inputs(self, inputs, T):
        rows = super()._pre_hc_from_inputs(inputs, T)
        layout = getattr(self, "_commit_layout", None)
        return (
            rows if layout is None else rows.index_select(0, layout.padded_to_compact)
        )

    def forward(self, inputs, fmha_impl=None):
        self._commit_layout = getattr(
            getattr(fmha_impl, "metadata", None), "mtp_commit_layout", None
        )
        try:
            return super().forward(inputs, fmha_impl)
        finally:
            self._commit_layout = None

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
        self._mtp_input_fusion = self._platform_provider.build_mtp_input_fusion(
            self.enorm, self.hnorm, self.e_proj, self.h_proj
        )

    def _embed_mtp_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Restore the full TP hidden width before the fusion norms.
        return self.v4._embed(input_ids).contiguous()

    def _build_fused(self, input_ids, pre_hc, positions):
        fusion = getattr(self, "_mtp_input_fusion", None)
        if fusion is not None:
            chunked = pre_hc.shape[0] > self._mtp_fusion_chunk_tokens()
            if not chunked or torch.cuda.is_current_stream_capturing():
                inputs_embeds = self._embed_mtp_inputs(input_ids)
                fused = fusion(inputs_embeds, pre_hc, positions)
                if fused is not None:
                    return fused
                # An unsupported fusion shape still reuses the TP gather above.
                tokens, hc, dim = pre_hc.shape
                inputs_embeds = torch.where(
                    positions.reshape(-1, 1) == 0,
                    torch.zeros_like(inputs_embeds),
                    inputs_embeds,
                )
                e_norm = self.enorm(inputs_embeds.contiguous())
                h_norm = self.hnorm(pre_hc.reshape(-1, dim)).view(tokens, hc, dim)
                return self._apply_proj(self.h_proj, h_norm) + self._apply_proj(
                    self.e_proj, e_norm
                ).unsqueeze(1)
        return super()._build_fused(input_ids, pre_hc, positions)

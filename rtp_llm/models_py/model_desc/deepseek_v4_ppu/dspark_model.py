"""PPU DSpARK draft: shared proposal geometry, PPU decode operators."""

from __future__ import annotations

from typing import Any

import torch

from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.deepseek_v4_dspark_model import (
    DeepSeekV4DSparkModel,
)
from rtp_llm.models_py.model_desc.deepseek_v4_ppu.model import PpuDeepSeekV4Model
from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.decode.attention_kernels import (
    attn_fp8_swa_paged,
)
from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.decode.compute_qkv import (
    decode_compute_qkv,
)
from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.decode.decode_attn_metadata import (
    get_or_build_sched_meta,
)
from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.decode.output_proj import (
    decode_output_proj,
)
from rtp_llm.models_py.modules.dsv4.ppu.utils import _v4_fp8_linear
from rtp_llm.utils.model_weight import W


class PpuDeepSeekV4DSparkModel(DeepSeekV4DSparkModel, PpuDeepSeekV4Model):
    """DSpARK draft on PPU.

    Proposal/commit geometry, paged-slot maths and the query-block index
    builder stay in the shared model; this class only swaps the four decode
    operators for their PPU implementations and builds the feature
    projection through the PPU provider.
    """

    def _load_extra_weights(self, weights: ModelWeights) -> None:
        gw = weights.global_weights
        self.main_norm = RMSNorm(
            gw[W.v4_dspark_main_norm], float(self._v4_args.norm_eps)
        )
        self.main_proj = _v4_fp8_linear(
            gw[W.v4_dspark_main_proj_w],
            gw[W.v4_dspark_main_proj_s],
            platform_provider=self._platform_provider,
        )

    def _dspark_compute_qkv(
        self, attn: Any, x: torch.Tensor, positions: torch.Tensor
    ):
        return decode_compute_qkv(attn, x, positions)

    def _dspark_sched_meta(self, graph_metadata: Any, *, attn: Any, **geometry: Any):
        return get_or_build_sched_meta(
            graph_metadata,
            extra_attn_type=None,
            platform_provider=getattr(attn, "_platform_provider", None),
            **geometry,
        )

    def _dspark_attend(
        self,
        attn: Any,
        q: torch.Tensor,
        pool: torch.Tensor,
        indices: torch.Tensor,
        sched_meta: Any,
        topk_length: torch.Tensor,
        block_table: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        return attn_fp8_swa_paged(
            q=q,
            swa_pool_3d=pool,
            attn_sink=attn.attn_sink,
            swa_topk_3d=indices,
            swa_block_table=block_table[:batch_size],
            sched_meta=sched_meta,
            fp8_op=attn._get_fp8_decode_op(),
            topk_length=topk_length,
            platform_provider=getattr(attn, "_platform_provider", None),
        )

    def _dspark_output_proj(
        self,
        attn: Any,
        output: torch.Tensor,
        freqs_cis: torch.Tensor,
        batch_size: int,
        gamma: int,
    ) -> torch.Tensor:
        return decode_output_proj(attn, output, freqs_cis, batch_size, gamma)

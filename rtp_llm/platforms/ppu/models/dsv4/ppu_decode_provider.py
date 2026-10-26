"""Decode composition with engine-owned DP/EP resources."""

import torch

from rtp_llm.platforms.ppu.models.dsv4.manifest import DECODE_EXECUTION_OPTIONS, _check_execution_options
from rtp_llm.platforms.ppu.models.dsv4.ppu_module_provider import PpuModuleProvider


class PpuDecodeProvider(PpuModuleProvider):
    name = "m890p-dsv4-fp4-decode-candidate"
    indexer_mode = "FP4"
    _moe_hint = "capacity"
    _moe_output_dtype = torch.bfloat16

    def build_moe(self, default_factory, *args, **kwargs):
        from rtp_llm.platforms.ppu.models.dsv4.ppu_deepep_fp4 import PpuDeepEPFP4Strategy
        from rtp_llm.platforms.ppu.models.dsv4.ppu_ep_moe import PpuEPMoE

        if not kwargs.get("is_decode_role"):
            raise ValueError("PPU Decode MoE requires Decode resources")
        return PpuEPMoE(
            *args,
            platform_provider=self,
            strategy_type=PpuDeepEPFP4Strategy,
            strategy_kwargs={
                "expected_m_policy": self._moe_hint,
                "output_dtype": self._moe_output_dtype,
                "gemm_tile": self._moe_tile,
            },
            **kwargs,
        )

    def __init__(self, execution_options):
        super().__init__(execution_options)
        from rtp_llm.platforms.ppu.runtime import PpuStreamPool

        support = _check_execution_options(
            self.execution_options, DECODE_EXECUTION_OPTIONS
        )
        if not support.supported:
            raise ValueError(support.reason)
        mtp_overlap = self.execution_options.get("DSV4_PPU_MTP_QUERY_OVERLAP", "0")
        if mtp_overlap not in ("0", "1"):
            raise ValueError("DSV4_PPU_MTP_QUERY_OVERLAP must be 0 or 1")
        self._mtp_query_overlap = mtp_overlap == "1"
        if self.execution_options.get("DSV4_PPU_MTP_METADATA_GRAPH", "0") != "0":
            raise ValueError("PPU metadata Graph is outside this implementation")
        mtp_moe_hint = self.execution_options.get("DSV4_PPU_MTP_MOE_HINT", "capacity")
        if mtp_moe_hint not in ("capacity", "batch"):
            raise ValueError("DSV4_PPU_MTP_MOE_HINT must be capacity or batch")
        self._moe_hint = mtp_moe_hint
        self._moe_tile = self.execution_options.get("DSV4_PPU_MTP_MOE_TILE", "auto")
        if self._moe_tile not in ("auto", "n128"):
            raise ValueError("DSV4_PPU_MTP_MOE_TILE must be auto or n128")
        mask_inactive = self.execution_options.get("DSV4_PPU_MTP_MASK_INACTIVE", "0")
        if mask_inactive != "0":
            raise ValueError("Inactive MoE masking requires PPU metadata Graph")
        batch_indexer = self.execution_options.get("DSV4_PPU_MTP_BATCH_INDEXER", "0")
        if batch_indexer not in ("0", "1"):
            raise ValueError("DSV4_PPU_MTP_BATCH_INDEXER must be 0 or 1")
        self._mtp_batch_indexer = batch_indexer == "1"
        self.stream_pool = PpuStreamPool()

    def build_fp8_linear(self, default_factory, *args, **kwargs):
        return super().build_fp8_linear(
            default_factory, *args, quantization="v2_column", **kwargs
        )

    def build_wo_a_fp8_linear(self, default_factory, *args, **kwargs):
        return super().build_wo_a_fp8_linear(
            default_factory, *args, quantization="v2_row", **kwargs
        )

    def build_attention(self, default_factory, *args, **kwargs):
        attention = super().build_attention(
            default_factory,
            *args,
            decode_stream_pool=self.stream_pool,
            decode_qkv_mode="merged",
            decode_indexer_mode="overlap",
            decode_mtp_overlap=self._mtp_query_overlap,
            decode_mtp_batch_indexer=self._mtp_batch_indexer,
            **kwargs,
        )
        return attention

    def build_shared_expert_executor(self, **kwargs):
        from rtp_llm.platforms.ppu.models.dsv4.ppu_shared_expert import PpuSharedExpertExecutor

        return PpuSharedExpertExecutor(
            stream_pool=self.stream_pool, start_before_routing=True
        )

    def build_hc_unit(self, *args, **kwargs):
        from rtp_llm.platforms.ppu.models.dsv4.ppu_hc import PpuHCUnit

        return PpuHCUnit(
            *args,
            options=self.execution_options,
            allow_graph=True,
            fuse_prenorm=True,
            fuse_norm=True,
            **kwargs,
        )

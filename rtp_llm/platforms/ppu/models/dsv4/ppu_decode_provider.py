"""Decode composition with engine-owned DP/EP resources."""

import torch
import weakref

from rtp_llm.platforms.ppu.models.dsv4.manifest import DECODE_EXECUTION_OPTIONS, _check_execution_options
from rtp_llm.platforms.ppu.models.dsv4.ppu_module_provider import PpuModuleProvider
from rtp_llm.platforms.ppu.models.dsv4.impl.platform_provider import Dsv4ProviderCapability


class PpuDecodeProvider(PpuModuleProvider):
    name = "m890p-dsv4-fp4-decode-candidate"
    indexer_mode = "FP4"
    capabilities = PpuModuleProvider.capabilities | frozenset({Dsv4ProviderCapability.MTP_INPUT_FUSION, Dsv4ProviderCapability.DECODE_METADATA})
    _moe_hint = "capacity"
    _moe_output_dtype = torch.bfloat16

    def build_moe(self, default_factory, *args, **kwargs):
        from rtp_llm.platforms.ppu.models.dsv4.ppu_deepep_fp4 import PpuDeepEPFP4Strategy
        from rtp_llm.platforms.ppu.models.dsv4.ppu_ep_moe import PpuEPMoE

        if not kwargs.get("is_decode_role"):
            raise ValueError("PPU Decode MoE requires Decode resources")
        strategy_type = PpuDeepEPFP4Strategy
        strategy_kwargs = {
            "expected_m_policy": self._moe_hint,
            "output_dtype": self._moe_output_dtype,
            "gemm_tile": self._moe_tile,
        }
        if self._moe_backend == "megamoe":
            from .ppu_mega_moe import PpuMegaMoEFP4Strategy

            strategy_type = PpuMegaMoEFP4Strategy
            strategy_kwargs["resources"] = self._mega_resources
            strategy_kwargs["module_name"] = self._mega_module_name
        return PpuEPMoE(
            *args,
            platform_provider=self,
            strategy_type=strategy_type,
            strategy_kwargs=strategy_kwargs,
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
        mtp_metadata = self.execution_options.get("DSV4_PPU_MTP_METADATA_GRAPH", "0")
        if mtp_metadata not in ("0", "1"):
            raise ValueError("DSV4_PPU_MTP_METADATA_GRAPH must be 0 or 1")
        self._mtp_metadata_graph = mtp_metadata == "1"
        mtp_moe_hint = self.execution_options.get("DSV4_PPU_MTP_MOE_HINT", "capacity")
        if mtp_moe_hint not in ("capacity", "batch"):
            raise ValueError("DSV4_PPU_MTP_MOE_HINT must be capacity or batch")
        self._moe_hint = mtp_moe_hint
        self._moe_tile = self.execution_options.get("DSV4_PPU_MTP_MOE_TILE", "auto")
        if self._moe_tile not in ("auto", "n128"):
            raise ValueError("DSV4_PPU_MTP_MOE_TILE must be auto or n128")
        mask_inactive = self.execution_options.get("DSV4_PPU_MTP_MASK_INACTIVE", "0")
        if mask_inactive not in ("0", "1"):
            raise ValueError("DSV4_PPU_MTP_MASK_INACTIVE must be 0 or 1")
        self._mask_inactive = mask_inactive == "1"
        if self._mask_inactive and not self._mtp_metadata_graph:
            raise ValueError("Inactive MoE masking requires PPU metadata Graph")
        batch_indexer = self.execution_options.get("DSV4_PPU_MTP_BATCH_INDEXER", "0")
        if batch_indexer not in ("0", "1"):
            raise ValueError("DSV4_PPU_MTP_BATCH_INDEXER must be 0 or 1")
        self._mtp_batch_indexer = batch_indexer == "1"
        self._decode_fusions = self._bool("DSV4_PPU_DECODE_FUSIONS", False)
        self._moe_backend = self.execution_options.get("DSV4_PPU_DECODE_MOE_BACKEND", "deepep")
        if self._moe_backend not in ("deepep", "megamoe"):
            raise ValueError("DSV4_PPU_DECODE_MOE_BACKEND must be deepep or megamoe")
        self._mega_resources = None
        self._mega_module_name = self.execution_options.get("DSV4_PPU_MEGAMOE_PACKAGE", "deep_gemm")
        if self._moe_backend == "megamoe":
            if self._mega_module_name not in ("deep_gemm", "deep_gemm_mega"):
                raise ValueError("DSV4_PPU_MEGAMOE_PACKAGE must be deep_gemm or deep_gemm_mega")
            from .ppu_mega_moe import PpuMegaMoEResources

            self._mega_resources = PpuMegaMoEResources()
        self.stream_pool = PpuStreamPool()
        self._decode_attention_refs = []

    def build_fp8_linear(self, default_factory, *args, **kwargs):
        return super().build_fp8_linear(
            default_factory, *args, quantization="v2_column", **kwargs
        )

    def build_wo_a_fp8_linear(self, default_factory, *args, **kwargs):
        linear = super().build_wo_a_fp8_linear(
            default_factory, *args, quantization="v2_row", **kwargs
        )
        linear.fuse_inverse_rope_quant = self._decode_fusions
        return linear

    def build_shared_expert(self, *args, **kwargs):
        shared = super().build_shared_expert(*args, **kwargs)
        shared.fuse_activation_quant = self._decode_fusions
        return shared

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
        attention._fuse_qkv_quant = self._decode_fusions
        self._decode_attention_refs.append(weakref.ref(attention))
        from rtp_llm.platforms.ppu.modules.linear.fp8_linear import PpuFp8Linear
        from .ppu_wo_a import PpuWoAFp8Linear

        for linear in (attention.wq_b, attention.wo_a, attention.wo_b):
            if type(linear) in (PpuFp8Linear, PpuWoAFp8Linear):
                linear._decode_attention_gemm = self._decode_fusions
        if attention.indexer is not None:
            attention.indexer.compressor._fuse_decode_queries = self._decode_fusions
            attention.indexer._fuse_query_slices = self._decode_fusions
            attention.indexer._fuse_topk_order = self._decode_fusions
            attention.indexer.compressor._fuse_decode_store = self._decode_fusions
        return attention

    def build_shared_expert_executor(self, **kwargs):
        from rtp_llm.platforms.ppu.models.dsv4.ppu_shared_expert import PpuSharedExpertExecutor

        return PpuSharedExpertExecutor(
            stream_pool=self.stream_pool, start_before_routing=True,
        )

    def build_hc_unit(self, *args, **kwargs):
        from rtp_llm.platforms.ppu.models.dsv4.ppu_hc import PpuHCUnit

        return PpuHCUnit(
            *args,
            options=self.execution_options,
            allow_graph=True,
            fuse_prenorm=True,
            fuse_norm=True,
            fuse_moe_sum=self._decode_fusions,
            fuse_pre_quant=self._decode_fusions,
            **kwargs,
        )

    def build_hc_head(self, *args, **kwargs):
        if not self._decode_fusions:
            return super().build_hc_head(*args, **kwargs)
        from .ppu_hc_head import PpuDecodeHCHead

        tp_size, tp_rank = kwargs.pop("tp_size"), kwargs.pop("tp_rank")
        head = PpuDecodeHCHead(*args, options=self.execution_options, **kwargs)
        head.tp_size, head.tp_rank = tp_size, tp_rank
        return head


    def build_mtp_input_fusion(self, enorm, hnorm, e_proj, h_proj):
        if not self._decode_fusions:
            return None
        from .ppu_mtp_input import PpuMtpInputFusion

        return PpuMtpInputFusion(enorm, hnorm, e_proj, h_proj)


    def build_decode_metadata(self, default_factory, *args, **kwargs):
        if not self._mtp_metadata_graph:
            return default_factory(*args, **kwargs)
        from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_fmha_impl import (
            DSv4DecodeFmhaImplFP8,
        )

        from .ppu_decode_metadata import PpuDecodeMetadataGraph

        if default_factory is not DSv4DecodeFmhaImplFP8:
            raise ValueError("PPU metadata Graph requires the FP8 Decode factory")
        tables = {}
        for reference in self._decode_attention_refs:
            attention = reference()
            if attention is None:
                raise RuntimeError(
                    "Decode Attention was released before metadata construction"
                )
            table = attention.freqs_cis
            tables[id(table)] = table
        if not tables:
            raise ValueError("Shared RoPE requires constructed Decode Attention layers")
        return PpuDecodeMetadataGraph(
            *args,
            fused_state_slots=True,
            mask_inactive=self._mask_inactive,
            shared_rope_tables=tuple(tables.values()),
            **kwargs,
        )

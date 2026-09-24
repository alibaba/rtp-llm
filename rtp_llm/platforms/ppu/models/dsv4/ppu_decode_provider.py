"""TP1 Decode composition with matching four- or eight-rank DP/EP resources."""

import weakref

import torch
from rtp_llm.models_py.modules.dsv4.platform_provider import Dsv4ProviderCapability

from .manifest import DECODE_EXECUTION_OPTIONS, _check_execution_options
from .ppu_module_provider import PpuModuleProvider


class PpuDecodeProvider(PpuModuleProvider):
    name = "m890p-dsv4-fp4-decode-candidate"
    indexer_mode = "FP4"
    capabilities = PpuModuleProvider.capabilities | frozenset(
        {Dsv4ProviderCapability.DECODE_METADATA}
    )
    _moe_hint = "capacity"
    _moe_output_dtype = torch.bfloat16

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
            raise ValueError("Inactive MoE masking requires MTP metadata Graph")
        batch_indexer = self.execution_options.get("DSV4_PPU_MTP_BATCH_INDEXER", "0")
        if batch_indexer not in ("0", "1"):
            raise ValueError("DSV4_PPU_MTP_BATCH_INDEXER must be 0 or 1")
        self._mtp_batch_indexer = batch_indexer == "1"
        self._decode_attention_refs = []
        self.stream_pool = PpuStreamPool()

    def build_decode_metadata(self, default_factory, *args, **kwargs):
        from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_fmha_impl import (
            DSv4DecodeFmhaImplFP8,
        )

        from .ppu_decode_metadata import PpuDecodeMetadataGraph

        if default_factory is not DSv4DecodeFmhaImplFP8:
            raise ValueError("PPU metadata Graph requires the FP8 Decode factory")
        config = args[0] if args else kwargs["config"]
        if config.q_len != 1 and not self._mtp_metadata_graph:
            # Verify and draft catch-up use the existing multi-token metadata
            # updater, whose slot mappings include speculative ring entries.
            return default_factory(*args, **kwargs)
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
        # Materialization can replace the RoPE table. Resolve it when binding
        # metadata, without a provider <-> attention ownership cycle.
        self._decode_attention_refs.append(weakref.ref(attention))
        return attention

    def build_shared_expert_executor(self, **kwargs):
        from .ppu_shared_expert import PpuSharedExpertExecutor

        return PpuSharedExpertExecutor(
            stream_pool=self.stream_pool, start_before_routing=True
        )

    def build_hc_unit(self, *args, **kwargs):
        from .ppu_hc import PpuHCUnit

        return PpuHCUnit(
            *args,
            options=self.execution_options,
            allow_graph=True,
            fuse_prenorm=True,
            fuse_norm=True,
            **kwargs,
        )

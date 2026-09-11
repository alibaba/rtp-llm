"""The published TP1/DP8/EP8 Decode composition and its instance resources."""

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
        self._decode_attention_refs = []
        self.stream_pool = PpuStreamPool()

    def build_decode_metadata(self, default_factory, *args, **kwargs):
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

"""Explicit TP1/DP8/EP8 Decode candidate using engine-owned communication."""

import weakref

import torch
from rtp_llm.models_py.modules.dsv4.platform_provider import Dsv4ProviderCapability

from .ppu_module_provider import PpuModuleProvider


class PpuDecodeProvider(PpuModuleProvider):
    name = "m890p-dsv4-fp4-decode-candidate"
    indexer_mode = "FP4"
    capabilities = PpuModuleProvider.capabilities | frozenset(
        {Dsv4ProviderCapability.DECODE_METADATA}
    )

    def __init__(self, execution_options=None):
        super().__init__(execution_options)
        from rtp_llm.platforms.ppu.runtime import PpuStreamPool

        self._fp8_quantization = self.execution_options.get(
            "DSV4_PPU_DECODE_FP8_QUANT", "auto"
        )
        if self._fp8_quantization not in ("auto", "v2"):
            raise ValueError("PPU Decode FP8 quantization must be auto or v2")
        self._qkv_mode = self.execution_options.get("DSV4_PPU_DECODE_QKV", "separate")
        if self._qkv_mode not in ("separate", "merged"):
            raise ValueError("PPU Decode QKV must be separate or merged")
        if (
            self._qkv_mode == "merged"
            and self.execution_options.get("DSV4_PPU_DECODE_ATTN_MODE", "sequential")
            != "overlap"
        ):
            raise ValueError("Merged PPU Decode QKV requires overlap mode")
        self._indexer_schedule = self.execution_options.get(
            "DSV4_PPU_DECODE_INDEXER", "sequential"
        )
        if self._indexer_schedule not in ("sequential", "overlap"):
            raise ValueError("PPU Decode Indexer must be sequential or overlap")
        if (
            self._indexer_schedule == "overlap"
            and self.execution_options.get("DSV4_PPU_DECODE_ATTN_MODE", "sequential")
            != "overlap"
        ):
            raise ValueError("PPU Indexer overlap requires Attention overlap")
        self._moe_hint = self.execution_options.get(
            "DSV4_PPU_DECODE_MOE_HINT", "capacity"
        )
        if self._moe_hint != "capacity":
            raise ValueError("PPU Decode MoE hint must be capacity")
        moe_output = self.execution_options.get("DSV4_PPU_DECODE_MOE_OUTPUT", "fp32")
        if moe_output not in ("fp32", "bf16"):
            raise ValueError("PPU Decode MoE output must be fp32 or bf16")
        self._moe_output_dtype = (
            torch.bfloat16 if moe_output == "bf16" else torch.float32
        )
        self._metadata_mode = self.execution_options.get(
            "DSV4_PPU_DECODE_METADATA", "eager"
        )
        if self._metadata_mode not in ("eager", "graph", "graph_fused"):
            raise ValueError(
                "PPU Decode metadata mode must be eager, graph or graph_fused"
            )
        self._rope_mode = self.execution_options.get("DSV4_PPU_DECODE_ROPE", "layer")
        if self._rope_mode not in ("layer", "shared"):
            raise ValueError("PPU Decode RoPE must be layer or shared")
        if self._rope_mode == "shared" and (
            self._metadata_mode == "eager"
            or self.execution_options.get("DSV4_PPU_DECODE_ATTN_MODE", "sequential")
            != "overlap"
        ):
            raise ValueError(
                "Shared PPU Decode RoPE requires metadata Graph and Attention overlap"
            )
        self._decode_attention_refs = []
        self._shared_schedule = self.execution_options.get(
            "DSV4_PPU_DECODE_SHARED_SCHEDULE", "after_route"
        )
        if self._shared_schedule not in ("after_route", "before_route"):
            raise ValueError("PPU shared schedule must be after_route or before_route")
        if (
            self._shared_schedule == "before_route"
            and self.execution_options.get("DSV4_SHARED_EXPERT_MODE", "sequential")
            != "overlap"
        ):
            raise ValueError("Early shared execution requires overlap mode")
        self.stream_pool = PpuStreamPool()

    def build_decode_metadata(self, default_factory, *args, **kwargs):
        if self._metadata_mode == "eager":
            return default_factory(*args, **kwargs)
        from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_fmha_impl import (
            DSv4DecodeFmhaImplFP8,
        )

        from .ppu_decode_metadata import PpuDecodeMetadataGraph

        if default_factory is not DSv4DecodeFmhaImplFP8:
            raise ValueError("PPU metadata Graph requires the FP8 Decode factory")
        tables = {}
        if self._rope_mode == "shared":
            for reference in self._decode_attention_refs:
                attention = reference()
                if attention is None:
                    raise RuntimeError(
                        "Decode Attention was released before metadata construction"
                    )
                table = attention.freqs_cis
                tables[id(table)] = table
            if not tables:
                raise ValueError(
                    "Shared RoPE requires constructed Decode Attention layers"
                )
        return PpuDecodeMetadataGraph(
            *args,
            fused_state_slots=self._metadata_mode == "graph_fused",
            shared_rope_tables=tuple(tables.values()),
            **kwargs,
        )

    def build_fp8_linear(self, default_factory, *args, **kwargs):
        return super().build_fp8_linear(
            default_factory,
            *args,
            quantization="v2_column" if self._fp8_quantization == "v2" else "auto",
            **kwargs,
        )

    def build_wo_a_fp8_linear(self, default_factory, *args, **kwargs):
        return super().build_wo_a_fp8_linear(
            default_factory,
            *args,
            quantization="v2_row" if self._fp8_quantization == "v2" else "auto",
            **kwargs,
        )

    def build_attention(self, default_factory, *args, **kwargs):
        mode = self.execution_options.get("DSV4_PPU_DECODE_ATTN_MODE", "sequential")
        if mode not in ("sequential", "overlap"):
            raise ValueError("PPU Decode attention mode must be sequential or overlap")
        attention = super().build_attention(
            default_factory,
            *args,
            decode_stream_pool=self.stream_pool if mode == "overlap" else None,
            decode_qkv_mode=self._qkv_mode,
            decode_indexer_mode=self._indexer_schedule,
            **kwargs,
        )
        if self._rope_mode == "shared":
            # Resolve each table after model materialization/reset_rope_cache.
            # Weak references avoid adding a provider <-> layer ownership cycle.
            self._decode_attention_refs.append(weakref.ref(attention))
        return attention

    def build_shared_expert_executor(self, **kwargs):
        from .ppu_shared_expert import PpuSharedExpertExecutor

        mode = self.execution_options.get("DSV4_SHARED_EXPERT_MODE", "sequential")
        if mode not in ("sequential", "overlap"):
            raise ValueError("PPU shared expert mode must be sequential or overlap")
        return PpuSharedExpertExecutor(
            stream_pool=self.stream_pool if mode == "overlap" else None,
            start_before_routing=self._shared_schedule == "before_route",
        )

    def build_hc_unit(self, *args, **kwargs):
        from .ppu_hc import PpuHCUnit

        reduction = self.execution_options.get("DSV4_PPU_DECODE_HC_REDUCTION", "torch")
        if reduction not in ("torch", "fused"):
            raise ValueError("PPU Decode HC reduction must be torch or fused")
        norm = self.execution_options.get("DSV4_PPU_DECODE_HC_NORM", "separate")
        if norm not in ("separate", "fused"):
            raise ValueError("PPU Decode HC norm must be separate or fused")
        return PpuHCUnit(
            *args,
            options=self.execution_options,
            allow_graph=True,
            fuse_prenorm=reduction == "fused",
            fuse_norm=norm == "fused",
            **kwargs,
        )

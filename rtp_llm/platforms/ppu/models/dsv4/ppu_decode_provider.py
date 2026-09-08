"""Explicit TP1/DP8/EP8 Decode candidate using engine-owned communication."""

from .ppu_module_provider import PpuModuleProvider


class PpuDecodeProvider(PpuModuleProvider):
    name = "m890p-dsv4-fp4-decode-candidate"
    indexer_mode = "FP4"

    def __init__(self, execution_options=None):
        super().__init__(execution_options)
        from rtp_llm.platforms.ppu.runtime import PpuStreamPool

        self.stream_pool = PpuStreamPool()

    def build_attention(self, default_factory, *args, **kwargs):
        mode = self.execution_options.get("DSV4_PPU_DECODE_ATTN_MODE", "sequential")
        if mode not in ("sequential", "overlap"):
            raise ValueError("PPU Decode attention mode must be sequential or overlap")
        return super().build_attention(
            default_factory,
            *args,
            decode_stream_pool=self.stream_pool if mode == "overlap" else None,
            **kwargs,
        )

    def build_shared_expert_executor(self, **kwargs):
        from .ppu_shared_expert import PpuSharedExpertExecutor

        mode = self.execution_options.get("DSV4_SHARED_EXPERT_MODE", "sequential")
        if mode not in ("sequential", "overlap"):
            raise ValueError("PPU shared expert mode must be sequential or overlap")
        return PpuSharedExpertExecutor(
            stream_pool=self.stream_pool if mode == "overlap" else None
        )

    def build_moe(self, default_factory, *args, **kwargs):
        from .ppu_deepep_fp4 import PpuDeepEPFP4Strategy

        if (
            kwargs.get("tp_size") != 1
            or kwargs.get("ep_size") != 8
            or not kwargs.get("is_decode_role")
        ):
            raise ValueError("PPU Decode MoE requires the TP1/EP8 Decode role")
        return default_factory(
            *args,
            platform_provider=self,
            execution_options=self.execution_options,
            strategy_type=PpuDeepEPFP4Strategy,
            **kwargs,
        )

    def build_hc_unit(self, *args, **kwargs):
        from .ppu_hc import PpuHCUnit

        return PpuHCUnit(
            *args, options=self.execution_options, allow_graph=True, **kwargs
        )

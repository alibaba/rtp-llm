"""PPU operators and buffers owned by one DeepSeek V4 model instance."""

from rtp_llm.platforms.ppu.models.dsv4.ppu_provider import M890PDsv4Provider


class PpuModuleProvider(M890PDsv4Provider):
    indexer_mode = "FP4"
    owns_runtime_buffers = True

    def build_attention(self, default_factory, *args, **kwargs):
        from rtp_llm.platforms.ppu.models.dsv4.ppu_fp4_indexer import PpuFP4Attention

        return super().build_attention(PpuFP4Attention, *args, **kwargs)

    def build_moe(self, default_factory, *args, **kwargs):
        if kwargs.get("ep_size") == 1:
            from rtp_llm.platforms.ppu.models.dsv4.ppu_tp_moe import PpuTPMoE

            return PpuTPMoE(*args, platform_provider=self, **kwargs)
        from rtp_llm.platforms.ppu.models.dsv4.ppu_ep_moe import PpuEPMoE
        from rtp_llm.platforms.ppu.models.dsv4.ppu_legacy_deepep import PpuLegacyDeepEPStrategy

        return PpuEPMoE(
            *args,
            platform_provider=self,
            strategy_type=PpuLegacyDeepEPStrategy,
            strategy_kwargs={"options": self.execution_options},
            **kwargs,
        )

    def build_hc_unit(self, *args, **kwargs):
        from rtp_llm.platforms.ppu.models.dsv4.ppu_hc import PpuHCUnit

        norm = self.execution_options.get("DSV4_PPU_PREFILL_HC_NORM", "fused")
        if norm not in ("separate", "fused"):
            raise ValueError("PPU Prefill HC norm must be separate or fused")
        return PpuHCUnit(
            *args,
            options=self.execution_options,
            fuse_norm=norm == "fused",
            **kwargs,
        )

    def build_hc_head(self, *args, **kwargs):
        from rtp_llm.platforms.ppu.models.dsv4.ppu_hc_head import PpuHCHead

        tp_size, tp_rank = kwargs.pop("tp_size"), kwargs.pop("tp_rank")
        head = PpuHCHead(*args, options=self.execution_options, **kwargs)
        head.tp_size, head.tp_rank = tp_size, tp_rank
        return head

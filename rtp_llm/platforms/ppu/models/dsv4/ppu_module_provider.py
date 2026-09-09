"""Adapters owned by one module-dispatch model, never installed globally."""

from .ppu_provider import M890PDsv4Provider


class PpuModuleProvider(M890PDsv4Provider):

    def build_hc_unit(self, *args, **kwargs):
        from .ppu_hc import PpuHCUnit

        return PpuHCUnit(*args, options=self.execution_options, **kwargs)

    def build_hc_head(self, *args, **kwargs):
        from rtp_llm.models_py.modules.dsv4.hc.fallback_impl import FallbackHCHead

        tp_size, tp_rank = kwargs.pop("tp_size"), kwargs.pop("tp_rank")
        head = FallbackHCHead(*args, options=self.execution_options, **kwargs)
        head.tp_size, head.tp_rank = tp_size, tp_rank
        return head

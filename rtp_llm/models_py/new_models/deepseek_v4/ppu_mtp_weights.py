"""V4 MTP's six checkpoint-owned parameters and their PPU consumers."""

from rtp_llm.models_py.layers.norm import RMSNorm
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.deepseek_v4.ppu_linear import PpuOwnedFP8Linear


class PpuMtpWeights(RtpModule):
    def __init__(self, dim, eps, dtype, provider):
        super().__init__()
        self.enorm = RMSNorm(dim, eps, dtype)
        self.hnorm = RMSNorm(dim, eps, dtype)
        self.e_proj = PpuOwnedFP8Linear(dim, dim, platform_provider=provider)
        self.h_proj = PpuOwnedFP8Linear(dim, dim, platform_provider=provider)
        self.eps = eps

    def bind(self, model, provider):
        from rtp_llm.models_py.modules import RMSNorm as RuntimeRMSNorm

        # Runtime norms reference our exact storage and use PPU's native path.
        # Expose concrete PpuFp8Linear objects: PpuMtpInputFusion checks the
        # exact type before entering its fused quantization kernel.
        enorm = RuntimeRMSNorm(self.enorm.weight, self.eps)
        hnorm = RuntimeRMSNorm(self.hnorm.weight, self.eps)
        e_proj, h_proj = self.e_proj.bind_runtime(), self.h_proj.bind_runtime()
        # These are runtime views, not second checkpoint owners.
        for name, op in (
            ("enorm", enorm),
            ("hnorm", hnorm),
            ("e_proj", e_proj),
            ("h_proj", h_proj),
        ):
            object.__setattr__(model, name, op)
        model._mtp_input_fusion = provider.build_mtp_input_fusion(
            enorm, hnorm, e_proj, h_proj
        )

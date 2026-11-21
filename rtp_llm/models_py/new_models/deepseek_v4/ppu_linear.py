"""Direct FP8 checkpoint projection backed by the instance's native PPU runtime."""

import torch

from rtp_llm.models_py.modules.dsv4.fp8.linear_owner import Dsv4FP8Parameters


class PpuOwnedFP8Linear(Dsv4FP8Parameters):
    def __init__(self, in_features, out_features, *, platform_provider, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        if platform_provider is None:
            raise ValueError("PPU FP8 projection requires a role-specific provider")
        object.__setattr__(self, "_platform_provider", platform_provider)

    def validate_runtime_device(self, device):
        if torch.device(device).type != "cuda":
            raise RuntimeError("V4 PPU FP8 execution requires an M890P device")
        from rtp_llm.device.device_type import is_ppu

        if not is_ppu():
            raise RuntimeError("V4 PPU FP8 execution requires an M890P device")
        from rtp_llm.models_py.modules.factory.linear.ppu_impl.fp8_linear import (
            _DENSE_SYMBOL,
            _resolve_deep_gemm_symbol,
            _resolve_ppu_quant_symbols,
        )

        _resolve_deep_gemm_symbol(_DENSE_SYMBOL)
        _resolve_ppu_quant_symbols()

    def bind_runtime(self):
        """Return the actual PpuFp8Linear for the vendor's type-based fusions."""
        if self._executor is None:
            self.validate_weights_loaded()
            self.validate_runtime_device(self.weight.device)
            executor = self._platform_provider.build_fp8_linear(
                None, self.weight, self.scale
            )
            object.__setattr__(self, "_executor", executor)
        return self._executor

    def process_weights_after_loading(self):
        self.bind_runtime()

    def forward(self, inputs, *args, **kwargs):
        if self._executor is None:
            raise RuntimeError("PPU projection must be bound after checkpoint loading")
        return self._executor(inputs, *args, **kwargs)

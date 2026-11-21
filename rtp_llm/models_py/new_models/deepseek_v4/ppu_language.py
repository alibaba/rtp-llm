"""Construct native M890P target/draft backbones from owned parameters."""

import torch

from rtp_llm.models_py.model_desc.deepseek_v4_ppu.model import PpuDeepSeekV4Model
from rtp_llm.models_py.module_base import RtpModule

from .language import OwnedV4ModelMixin
from .ppu_backbone import PpuOwnedV4Transformer


class PpuOwnedV4Mixin(OwnedV4ModelMixin):
    def create_backbone(self, args, load_config):
        return PpuOwnedV4Transformer(
            args,
            platform_provider=self._platform_provider,
            params_dtype=load_config.compute_dtype,
            head_dtype=(
                torch.float32
                if getattr(self.config, "enable_fp32_lm_head", False)
                else load_config.compute_dtype
            ),
        )

    def _build_runtime_backbone(self):
        from rtp_llm.models_py.model_loader import NewModelLoader

        NewModelLoader._validate_loaded_weights(self)
        self.v4.bind_runtime()


class PpuDeepSeekV4ForCausalLM(PpuOwnedV4Mixin, PpuDeepSeekV4Model, RtpModule):
    """Direct Flash target parameters with the native M890P serving runtime."""

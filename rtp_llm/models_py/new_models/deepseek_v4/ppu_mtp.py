"""Dedicated Flash MTP role: PPU hidden gathers, operators and graph buffers."""

import torch

from rtp_llm.models_py.model_desc.deepseek_v4_ppu.mtp_model import PpuDeepSeekV4MtpModel
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.deepseek_v4.mtp import (
    DeepSeekV4MtpForCausalLM,
    _MtpMapper,
)
from rtp_llm.models_py.new_models.deepseek_v4.ppu_backbone import PpuOwnedV4Transformer
from rtp_llm.models_py.new_models.deepseek_v4.ppu_language import PpuOwnedV4Mixin
from rtp_llm.models_py.new_models.deepseek_v4.ppu_mtp_weights import PpuMtpWeights


class _PpuMtpMapper(_MtpMapper):
    def map_name(self, name):
        mapped = super().map_name(name)
        return "fusion." + mapped if mapped in self.fusion_leaves else mapped


class PpuDeepSeekV4MtpForCausalLM(PpuOwnedV4Mixin, PpuDeepSeekV4MtpModel, RtpModule):
    checkpoint_weight_name_filter = (
        DeepSeekV4MtpForCausalLM.checkpoint_weight_name_filter
    )

    def create_backbone(self, args, load_config):
        if (
            args.n_layers != 1
            or list(args.compress_ratios) != [0]
            or args.n_hash_layers
        ):
            raise ValueError("PPU Flash MTP requires one non-hash SWA layer")
        return PpuOwnedV4Transformer(
            args,
            platform_provider=self._platform_provider,
            params_dtype=load_config.compute_dtype,
            head_dtype=torch.bfloat16,
            layer_prefix="mtp",
            global_prefix="mtp.0.",
        )

    def __init__(self, model_config, load_config):
        super().__init__(model_config, load_config)
        args = self._v4_args
        self.fusion = PpuMtpWeights(
            args.dim, args.norm_eps, load_config.compute_dtype, self._platform_provider
        )
        self.WEIGHTS_MAPPER = _PpuMtpMapper(self.v4)

    def _release_runtime_weight_source(self):
        super()._release_runtime_weight_source()
        self.fusion.bind(self, self._platform_provider)

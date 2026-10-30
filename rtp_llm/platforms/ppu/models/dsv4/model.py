"""PPU DSV4 model and checkpoint-layout specialization."""

import os

import torch

from rtp_llm.models.deepseek_v4 import (
    DeepSeekV4,
    DeepSeekV4Mtp,
    DeepSeekV4Weight,
    DeepSeekV4MtpWeight,
)
from rtp_llm.utils.model_weight import W

from rtp_llm.platforms.ppu.models.dsv4.cache import configure_indexer_cache
from rtp_llm.platforms.ppu.models.dsv4.manifest import validate_cache_config
from rtp_llm.platforms.ppu.models.dsv4.resources import configure_weight_splits


class _PpuWeightLayout:
    def _build_compressor(self, layer_id, ratio, ckpt_prefix, inner=False):
        weights = super()._build_compressor(layer_id, ratio, ckpt_prefix, inner)
        if inner:
            for weight in weights:
                if weight.name == W.v4_indexer_compressor_norm:
                    weight.data_type = torch.float32
        return weights

    def get_weight_info(self):
        info = super().get_weight_info()
        if self._moe_pure_tp_mode:
            configure_weight_splits(info)
        return info


class PpuDeepSeekV4Weight(_PpuWeightLayout, DeepSeekV4Weight):
    pass


class PpuDeepSeekV4MtpWeight(_PpuWeightLayout, DeepSeekV4MtpWeight):
    def _get_weight_info(self):
        info = super()._get_weight_info()
        # Quantized projection pairs own their UE8M0 scale bytes.
        info.weights = [
            weight for weight in info.weights
            if weight.name not in (W.v4_mtp_e_proj_s, W.v4_mtp_h_proj_s)
        ]
        return info


class _PpuModel:
    @classmethod
    def _apply_kv_cache_config(cls, model_config, kv_cache_config):
        validate_cache_config(kv_cache_config)
        super()._apply_kv_cache_config(model_config, kv_cache_config)

    def create_model_loader(self):
        validate_cache_config(self.kv_cache_config)
        configure_indexer_cache(self.model_config)
        if self.model_config.model_type == "deepseek_v4":
            from rtp_llm.platforms.ppu.models.dsv4.communication import maybe_warmup_ppu_tp_communication

            maybe_warmup_ppu_tp_communication(
                self.parallelism_config,
                options={"DSV4_PPU_TP_COMM_WARMUP": "1", **os.environ},
            )
        return super().create_model_loader()

    def _create_python_model(self):
        if self.model_config.model_type == "deepseek_v4_mtp":
            from rtp_llm.platforms.ppu.models.dsv4.mtp_model import PpuDeepSeekV4MtpModel as model_cls
        else:
            from rtp_llm.platforms.ppu.models.dsv4.runtime_model import PpuDeepSeekV4Model as model_cls

        self.py_model = model_cls(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )


class PpuDeepSeekV4(_PpuModel, DeepSeekV4):
    @staticmethod
    def get_weight_cls():
        return PpuDeepSeekV4Weight


class PpuDeepSeekV4Mtp(_PpuModel, DeepSeekV4Mtp):
    @staticmethod
    def get_weight_cls():
        return PpuDeepSeekV4MtpWeight

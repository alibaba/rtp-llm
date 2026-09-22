"""MiniMax-M3.1 text model and checkpoint-specific weight handling."""

import json
import os
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.minimax_m3 import MiniMaxM3, MiniMaxM3Weight, _env_flag


class MiniMaxM31Weight(MiniMaxM3Weight):
    """M3.1 loader boundary for the released per-expert NVFP4 checkpoint."""

    def __init__(self, *args: Any, **kwargs: Any):
        self._prepacked_nvfp4_routed = False
        self._mock_nvfp4_moe = False
        super().__init__(*args, **kwargs)

    def _process_meta(self, meta_dict, weight_keys):
        super()._process_meta(meta_dict, weight_keys)
        self._prepacked_nvfp4_routed = self._contains(
            weight_keys, ".block_sparse_moe.experts.0.w1.weight_packed"
        )
        self._mock_nvfp4_moe = _env_flag("M3_M31_MOCK_NVFP4_MOE")
        if self._prepacked_nvfp4_routed and not self._mock_nvfp4_moe:
            raise RuntimeError(
                "MiniMax-M3.1 per-expert packed NVFP4 routed-MoE weights are not "
                "supported yet. Set M3_M31_MOCK_NVFP4_MOE=1 only for structural "
                "bring-up; that mode skips routed experts, keeps the MXFP8 shared "
                "expert, and is not valid for quality evaluation."
            )

    def _get_hf_ffn_layer_weight_info(self, layer_id: int):
        layer_weights = super()._get_hf_ffn_layer_weight_info(layer_id)
        if self._prepacked_nvfp4_routed and self._mock_nvfp4_moe:
            # M3.1 structural bring-up deliberately materializes only the shared
            # expert. The M3 loader never sees this checkpoint-specific branch.
            return layer_weights[:1]
        return layer_weights


class MiniMaxM31(MiniMaxM3):
    """MiniMax-M3.1 text backbone, isolated from the legacy M3 runtime."""

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = super()._create_config(ckpt_path)
        config.model_type = "minimax_m31"
        return config

    @classmethod
    def _from_hf(cls, config: ModelConfig, ckpt_path: str):
        super()._from_hf(config, ckpt_path)
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return config
        with open(config_path) as reader:
            config_json = json.load(reader)
        cls._parse_nvfp4_mock_config(config, config_json)
        return config

    @staticmethod
    def _parse_nvfp4_mock_config(
        config: ModelConfig, config_json: Dict[str, Any]
    ) -> None:
        quant_cfg = config_json.get("quantization_config", {})
        packed_nvfp4 = (
            str(quant_cfg.get("moe_quant_algo", "")).upper() == "NVFP4"
            and str(quant_cfg.get("moe_quant_format", "")).lower()
            == "nvfp4-pack-quantized"
        )
        config.mock_nvfp4_moe = bool(
            packed_nvfp4 and _env_flag("M3_M31_MOCK_NVFP4_MOE")
        )

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.minimax_m31 import MiniMaxM31Model

        self.py_model = MiniMaxM31Model(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model

    @staticmethod
    def get_weight_cls():
        return MiniMaxM31Weight


register_model("minimax_m31", MiniMaxM31)


__all__ = ["MiniMaxM31", "MiniMaxM31Weight"]

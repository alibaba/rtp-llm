"""Qwen3 DFlash V1 checkpoint loader and model registration."""

from __future__ import annotations

from typing import Any, List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.qwen_v3 import QwenV3, QWenV3Weight
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, transpose
from rtp_llm.utils.util import get_config_from_path


class Qwen3DFlashWeight(QWenV3Weight):
    """DFlash omits embedding/lm-head and borrows those target tensors."""

    def _get_weight_info(self) -> ModelWeightInfo:
        info = super()._get_weight_info()
        # Keep embedding/lm-head descriptors so ModelLoader can validate the
        # target aliases and skip their absent draft checkpoint tensors.
        info.weights.extend(
            (
                AtomicWeight(
                    W.dspark_fc_w, [CkptWeightInfo("fc.weight", identity)], transpose
                ),
                AtomicWeight(
                    W.dspark_hidden_norm_gamma,
                    [CkptWeightInfo("hidden_norm.weight", identity)],
                    identity,
                ),
            )
        )
        return info


class Qwen3DFlash(QwenV3):
    """Dense mixed-mask Qwen3 DFlash draft model."""

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        raw = get_config_from_path(ckpt_path)
        if not isinstance(raw, dict):
            raise TypeError(f"DFlash config.json missing or invalid under {ckpt_path}")
        dflash = raw.get("dflash_config")
        if not isinstance(dflash, dict):
            raise ValueError("DFlash requires object dflash_config")
        config = cls._create_config_from_json(ckpt_path, raw)
        architectures = raw.get("architectures", ())
        if "DFlashDraftModel" not in architectures:
            raise ValueError(
                "DFlash requires architectures to include DFlashDraftModel"
            )
        config.input_vocab_size = int(raw["vocab_size"])
        config.dflash_mask_token_id = int(dflash["mask_token_id"])
        config.dflash_target_layer_ids = [
            int(value) for value in dflash["target_layer_ids"]
        ]
        layer_types = raw.get("layer_types", dflash.get("layer_types"))
        if not isinstance(layer_types, list):
            raise ValueError("DFlash requires a layer_types list")
        config.dflash_layer_types = [str(value) for value in layer_types]
        config.dflash_sliding_window = int(
            raw.get("sliding_window", dflash.get("sliding_window", 0)) or 0
        )
        block_size = raw.get("block_size", dflash.get("block_size"))
        if block_size is None:
            raise ValueError("DFlash requires native block_size metadata")
        config.dflash_native_block_size = int(block_size)
        # Attention is dispatched per layer by Qwen3DFlashModel.  Avoid
        # treating this mixed checkpoint as globally causal or non-causal.
        config.attn_config.is_causal = False
        return config

    @staticmethod
    def get_weight_cls():
        return Qwen3DFlashWeight

    @classmethod
    def speculative_weight_alias_names(cls, target_model, draft_model_config):
        target_config = target_model.model_config
        compatible = ("hidden_size", "vocab_size", "data_type", "enable_fp32_lm_head")
        mismatches = [
            name
            for name in compatible
            if getattr(target_config, name) != getattr(draft_model_config, name)
        ]
        if mismatches:
            details = ", ".join(
                f"{name}={getattr(target_config, name)!r}/{getattr(draft_model_config, name)!r}"
                for name in mismatches
            )
            raise ValueError(
                "DFlash cannot share semantically incompatible target embedding/lm-head: "
                + details
            )
        return (W.embedding, W.lm_head)

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.qwen3_dflash_model import Qwen3DFlashModel

        self.py_model = Qwen3DFlashModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            quant_config=self.model_config.quant_config,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model


register_model("qwen_3_dflash", Qwen3DFlash, ["DFlashDraftModel"])

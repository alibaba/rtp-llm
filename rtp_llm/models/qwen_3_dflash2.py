"""Qwen3 DFlash2 checkpoint loader.

The checkpoint omits embedding/lm-head tensors, as in DFlash V1. Dynamic
convolution and selector tensors retain their training layout and are replicated
on each TP rank; they operate after the backbone's residual all-reduces.
"""

from __future__ import annotations

import math

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.qwen_3_dflash import Qwen3DFlash, Qwen3DFlashWeight
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity
from rtp_llm.utils.util import get_config_from_path


class Qwen3DFlash2Weight(Qwen3DFlashWeight):
    def _get_weight_info(self) -> ModelWeightInfo:
        info = super()._get_weight_info()
        for runtime, checkpoint in (
            (W.dflash2_selector_predecessor, "predecessor_codebook"),
            (W.dflash2_selector_successor, "successor_codebook"),
            (W.dflash2_selector_projection, "hidden_projection.weight"),
        ):
            info.weights.append(
                AtomicWeight(
                    runtime,
                    [CkptWeightInfo("candidate_selector." + checkpoint, identity)],
                    identity,
                )
            )
        conv_weights = []
        for name, base, kernel in (
            (
                "attention_conv",
                W.dflash2_attention_conv_base,
                W.dflash2_attention_conv_kernel,
            ),
            ("mlp_conv", W.dflash2_mlp_conv_base, W.dflash2_mlp_conv_kernel),
        ):
            prefix = self.transformer_prefix + "layers.{i}." + name + "."
            conv_weights.extend(
                (
                    AtomicWeight(
                        base,
                        [CkptWeightInfo(prefix + "base_kernel", identity)],
                        identity,
                    ),
                    AtomicWeight(
                        kernel,
                        [CkptWeightInfo(prefix + "kernel_projection.weight", identity)],
                        identity,
                    ),
                )
            )
        # Qwen's mapper builds a descriptor list per layer. Keep the shared-list
        # variant supported too, matching ModelDeployWeightInfo.get_weight_info.
        if info.layer_weights and isinstance(info.layer_weights[0], list):
            for layer in info.layer_weights:
                layer.extend(conv_weights)
        else:
            info.layer_weights.extend(conv_weights)
        return info


class Qwen3DFlash2(Qwen3DFlash):
    checkpoint_architecture = "DFlash2DraftModel"

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = super()._create_config(ckpt_path)
        draft = get_config_from_path(ckpt_path)["dflash_config"]
        for source, target in (
            ("conv_kernel_size", "dflash2_conv_kernel_size"),
            ("conv_group_size", "dflash2_conv_group_size"),
            ("selector_rank", "dflash2_selector_rank"),
            ("selector_top_k", "dflash2_selector_top_k"),
        ):
            value = draft.get(source)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"DFlash2 requires positive integer {source}")
            setattr(config, target, value)
        if config.hidden_size % config.dflash2_conv_group_size:
            raise ValueError("DFlash2 conv_group_size must divide hidden_size")
        if config.dflash2_selector_top_k > config.vocab_size:
            raise ValueError("DFlash2 selector_top_k must not exceed vocab_size")
        scale = float(draft.get("input_embedding_scale", 1.0))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(
                "DFlash2 input_embedding_scale must be finite and positive"
            )
        config.dflash2_input_embedding_scale = scale
        return config

    @staticmethod
    def get_weight_cls():
        return Qwen3DFlash2Weight

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.qwen3_dflash2_model import Qwen3DFlash2Model

        self.py_model = Qwen3DFlash2Model(
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


register_model("qwen_3_dflash2", Qwen3DFlash2, ["DFlash2DraftModel"])

"""GLM-5.3-Flash draft model for the full checkpoint."""

import functools
import json
from pathlib import Path
from typing import Any, List, Optional

import torch

from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.attn_weight import MlaAttnAtomicWeight
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.deepseek_v2 import DeepSeekV2Weight
from rtp_llm.models.glm5_3_flash import (
    Glm53Flash,
    _configure_glm53_mtp_attention_layout,
    parse_glm53_flash_config,
    replicate_prefill_shared_weights,
)
from rtp_llm.utils.database import BaseDatabase
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, transpose, zeros


class Glm53FlashMtpWeight(DeepSeekV2Weight):
    """Weight loader for GLM-5 MTP that supports two checkpoint layouts.

    Layout A (legacy, extracted MTP-only checkpoint):
        config.json reports num_hidden_layers=1, MTP weights live at
        model.layers.0.* with embedding/lm_head also under model.layers.0.*.

    Layout B (full GLM-5 checkpoint, sglang/vllm style):
        config.json reports num_hidden_layers=N (e.g. 78) and
        num_nextn_predict_layers>=1, MTP weights at model.layers.{N}.*,
        embedding at model.embed_tokens.weight, lm_head at lm_head.weight.

    The mode is selected via model_config.mtp_layer_offset, set by
    Glm53FlashMtp._create_config(). offset==0 => Layout A; offset>0 => Layout B.
    """

    checkpoint_prefix = "model."

    def create_load_config(
        self,
        compute_dtype: torch.dtype,
        database: BaseDatabase,
        phy2log: Optional[List[List[int]]] = None,
        exported_device: Optional[Any] = None,
        force_cpu_load_weights: bool = False,
        moe_pure_tp_preshard: bool = False,
    ) -> LoadConfig:
        load_config = super().create_load_config(
            compute_dtype,
            database,
            phy2log=phy2log,
            exported_device=exported_device,
            force_cpu_load_weights=force_cpu_load_weights,
            moe_pure_tp_preshard=moe_pure_tp_preshard,
        )
        load_config.fp8_requant_preserve_fp32 = bool(
            getattr(self.model_config, "is_glm53_mtp", False)
        )
        return load_config

    def _get_hf_ffn_layer_weight_info(self, layer_id: int):
        weights = super()._get_hf_ffn_layer_weight_info(layer_id)
        if layer_id in self.moe_layer_index_:
            replicate_prefill_shared_weights(weights, self.model_config.mega_moe_policy)
        return weights

    def _process_meta(self, meta_dict, weight_keys):
        self.checkpoint_prefix = (
            "model.language_model."
            if any(key.startswith("model.language_model.") for key in weight_keys)
            else "model."
        )
        # The MTP layer index in the source checkpoint (78 for Layout B, 0 for
        # Layout A). DeepSeekV2Weight._process_meta only inspects layers
        # [0, _num_layers) which would miss the MTP layer in Layout B.
        mtp_layer_idx = int(getattr(self.model_config, "mtp_layer_offset", 0) or 0)
        layer_prefix = f"{self.checkpoint_prefix}layers.{mtp_layer_idx}."
        if f"{layer_prefix}self_attn.q_a_proj.weight" in weight_keys:
            self.q_use_lora = True
        if f"{layer_prefix}mlp.gate.e_score_correction_bias" in weight_keys:
            self.has_e_score_correction_bias = True

    @staticmethod
    def _prefix_checkpoint_names(weight: WeightModule, prefix: str) -> None:
        if prefix == "model.":
            return
        from rtp_llm.models.glm5_3_flash import Glm53FlashWeight

        Glm53FlashWeight._prefix_checkpoint_names(weight)

    def _get_hf_layer_weight_info(self, layer_id: int):
        weights = super()._get_hf_layer_weight_info(layer_id)
        if self.model_config.prefill_mla_cp:
            for weight in weights:
                if (
                    isinstance(weight, MlaAttnAtomicWeight)
                    and weight.config is not None
                ):
                    weight.config.replicate_for_prefill_cp = True
        if self.checkpoint_prefix == "model.language_model.":
            from rtp_llm.models.glm5_3_flash import (
                _glm53_indexer_compressor_weight_info,
            )

            weights.extend(_glm53_indexer_compressor_weight_info())
        return weights

    def _get_weight_info(self):
        mtp_layer_idx = int(getattr(self.model_config, "mtp_layer_offset", 0) or 0)

        if mtp_layer_idx == 0:
            embedding_ckpt = "model.layers.0.embed_tokens.weight"
            lm_head_ckpt = "model.layers.0.shared_head.head.weight"
        else:
            embedding_ckpt = "model.embed_tokens.weight"
            lm_head_ckpt = "lm_head.weight"

        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo(embedding_ckpt, identity)],
                identity,
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo(lm_head_ckpt, identity)],
                identity,
            ),
            AtomicWeight(
                W.multi_tokens_predict_final_ln_gamma,
                [
                    CkptWeightInfo(
                        f"model.layers.{mtp_layer_idx}.shared_head.norm.weight",
                        identity,
                    )
                ],
                identity,
            ),
            AtomicWeight(
                W.multi_tokens_predict_final_ln_beta,
                [],
                functools.partial(zeros, shape=[self._hidden_size]),
            ),
            AtomicWeight(
                W.multi_tokens_predict_enorm,
                [
                    CkptWeightInfo(
                        f"model.layers.{mtp_layer_idx}.enorm.weight", identity
                    )
                ],
                identity,
            ),
            AtomicWeight(
                W.multi_tokens_predict_hnorm,
                [
                    CkptWeightInfo(
                        f"model.layers.{mtp_layer_idx}.hnorm.weight", identity
                    )
                ],
                identity,
            ),
            AtomicWeight(
                W.multi_tokens_predict_eh_proj,
                [
                    CkptWeightInfo(
                        f"model.layers.{mtp_layer_idx}.eh_proj.weight", identity
                    )
                ],
                transpose,
            ),
        ]
        assert self._num_layers == 1
        layer_weights: List[List[WeightModule]] = []
        for layer in range(self._num_layers):
            per_layer = self._get_hf_layer_weight_info(layer)
            if mtp_layer_idx != 0:
                # Templates contain "{i}" placeholders that the framework will
                # later substitute with layer_id (= 0 here). Pre-substitute
                # them with the real MTP layer index so the loader looks at
                # model.layers.{mtp_layer_idx}.* in the full checkpoint.
                self._remap_layer_ckpt_names(per_layer, mtp_layer_idx)
            layer_weights.append(per_layer)

        weight_info = ModelWeightInfo(layer_weights=layer_weights, weights=weights)
        for weight in weight_info.weights:
            self._prefix_checkpoint_names(weight, self.checkpoint_prefix)
        for layer in weight_info.layer_weights:
            for weight in layer:
                self._prefix_checkpoint_names(weight, self.checkpoint_prefix)
        return weight_info

    @staticmethod
    def _remap_layer_ckpt_names(layer_weights, target_layer_idx: int) -> None:
        target = str(target_layer_idx)
        for wm in layer_weights:
            for component in wm.get_components():
                ckpt_list = getattr(component, "weights", None)
                if not ckpt_list:
                    continue
                for ckpt in ckpt_list:
                    if hasattr(ckpt, "name") and "{i}" in ckpt.name:
                        ckpt.name = ckpt.name.replace("{i}", target)


class Glm53FlashMtp(Glm53Flash):
    @classmethod
    def _create_config(cls, ckpt_path):
        raw = json.loads((Path(ckpt_path) / "config.json").read_text(encoding="utf-8"))
        text = raw["text_config"]
        # Draft receives multimodal features from the target request; it does
        # not construct or load a second vision tower.
        config = parse_glm53_flash_config({"text_config": text}, ckpt_path)
        count = int(text.get("num_nextn_predict_layers") or 1)
        if count != 1:
            raise ValueError("GLM-5.3-Flash expects one checkpoint draft layer")
        config.mtp_layer_offset = config.num_layers
        config.num_layers = count
        config.model_type = "glm5_3_flash_mtp"
        config.moe_layer_index = [0]
        config.is_mtp = True
        _configure_glm53_mtp_attention_layout(config, text, count)
        return config

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.glm53_flash_mtp import Glm53FlashMtpModel

        self.py_model = Glm53FlashMtpModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
        )
        return self.py_model

    @staticmethod
    def get_weight_cls():
        return Glm53FlashMtpWeight


register_model("glm5_3_flash_mtp", Glm53FlashMtp)

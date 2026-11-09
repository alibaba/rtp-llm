"""HY V4 draft checkpoint mapping."""

from __future__ import annotations

import functools
from typing import Any, List

from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.hy_v4.weight import Hy4Weight
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, transpose, zeros


class Hy4MtpWeight(Hy4Weight):
    """Load the single draft block stored below ``model.mtp_layers.0``."""

    def _process_meta(self, meta_dict: Any, weight_keys: List[str]):
        # Parent metadata probing only visits backbone layer names.
        self.q_use_lora = self.q_use_lora or (
            "model.mtp_layers.0.self_attn.q_a_proj.weight" in weight_keys
        )
        self.has_e_score_correction_bias = self.has_e_score_correction_bias or (
            "model.mtp_layers.0.mlp.gate.e_score_correction_bias" in weight_keys
        )
        self.has_fused_experts = self.has_fused_experts or any(
            key.startswith("model.mtp_layers.0.mlp.experts.gate_up_proj")
            for key in weight_keys
        )
        self.has_direct_mxfp4_experts = getattr(
            self, "has_direct_mxfp4_experts", False
        ) or ("model.mtp_layers.0.mlp.experts.w13_weight" in weight_keys)
        self.has_fused_experts = self.has_fused_experts or self.has_direct_mxfp4_experts
        if "model.mtp_layers.0.mlp.experts.gate_up_proj.weight" in weight_keys:
            self.fused_gate_up_suffix = ".weight"
        if "model.mtp_layers.0.mlp.experts.down_proj.weight" in weight_keys:
            self.fused_down_suffix = ".weight"

    @staticmethod
    def _remap_to_mtp(layer_weights: List[WeightModule]) -> None:
        for weight in layer_weights:
            for component in weight.get_components():
                for ckpt in getattr(component, "weights", ()):
                    ckpt.name = ckpt.name.replace(
                        "model.layers.{i}", "model.mtp_layers.0"
                    )

    def _get_weight_info(self):
        assert self._num_layers == 1
        layer_weights = [self._get_hf_layer_weight_info(0)]
        self._remap_to_mtp(layer_weights[0])
        weights: List[WeightModule] = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("model.embed_tokens.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("lm_head.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.multi_tokens_predict_enorm,
                [CkptWeightInfo("model.mtp_layers.0.enorm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.multi_tokens_predict_hnorm,
                [CkptWeightInfo("model.mtp_layers.0.hnorm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.multi_tokens_predict_eh_proj,
                [CkptWeightInfo("model.mtp_layers.0.eh_proj.weight", identity)],
                transpose,
            ),
            AtomicWeight(
                W.multi_tokens_predict_final_ln_gamma,
                [CkptWeightInfo("model.mtp_layers.0.final_layernorm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.multi_tokens_predict_final_ln_beta,
                [],
                functools.partial(zeros, shape=[self._hidden_size]),
            ),
        ]
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)

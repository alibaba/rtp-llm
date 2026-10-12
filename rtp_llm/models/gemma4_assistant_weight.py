from typing import Any, Dict, List

from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, transpose

GEMMA4_ASSISTANT_Q_PROJ_W = W.gemma4_assistant_q_proj
GEMMA4_ASSISTANT_PRE_PROJECTION_W = W.gemma4_assistant_pre_proj
GEMMA4_ASSISTANT_POST_PROJECTION_W = W.gemma4_assistant_post_proj


class Gemma4AssistantWeightInfo(ModelDeployWeightInfo):
    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]):
        super().__init__(*args, **kwargs)
        self.prefix = "model."

    def _get_weight_info(self) -> ModelWeightInfo:
        global_weights: List[WeightModule] = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("model.embed_tokens.weight", identity)],
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("lm_head.weight", identity)],
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo("model.norm.weight", identity)],
            ),
            AtomicWeight(
                GEMMA4_ASSISTANT_PRE_PROJECTION_W,
                [CkptWeightInfo("pre_projection.weight", identity)],
                process_fun=transpose,
            ),
            AtomicWeight(
                GEMMA4_ASSISTANT_POST_PROJECTION_W,
                [CkptWeightInfo("post_projection.weight", identity)],
                process_fun=transpose,
            ),
        ]
        layer_weights = [
            self._get_layer_weight_info(layer_id)
            for layer_id in range(self._num_layers)
        ]
        return ModelWeightInfo(weights=global_weights, layer_weights=layer_weights)

    def _get_layer_weight_info(self, layer_id: int) -> List[WeightModule]:
        del layer_id
        prefix = self.prefix + "layers.{i}."
        return [
            AtomicWeight(
                W.pre_ln_gamma,
                [CkptWeightInfo(prefix + "input_layernorm.weight", identity)],
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [CkptWeightInfo(prefix + "post_attention_layernorm.weight", identity)],
            ),
            AtomicWeight(
                W.pre_ffn_ln_gamma,
                [CkptWeightInfo(prefix + "pre_feedforward_layernorm.weight", identity)],
            ),
            AtomicWeight(
                W.post_ffn_ln_gamma,
                [
                    CkptWeightInfo(
                        prefix + "post_feedforward_layernorm.weight", identity
                    )
                ],
            ),
            AtomicWeight(
                W.layer_scalar,
                [CkptWeightInfo(prefix + "layer_scalar", identity)],
            ),
            AtomicWeight(
                GEMMA4_ASSISTANT_Q_PROJ_W,
                [CkptWeightInfo(prefix + "self_attn.q_proj.weight", identity)],
                process_fun=transpose,
            ),
            AtomicWeight(
                W.attn_o_w,
                [CkptWeightInfo(prefix + "self_attn.o_proj.weight", identity)],
                process_fun=transpose,
            ),
            AtomicWeight(
                W.q_ln_gamma,
                [CkptWeightInfo(prefix + "self_attn.q_norm.weight", identity)],
            ),
            AtomicWeight(
                W.ffn_w1,
                [CkptWeightInfo(prefix + "mlp.gate_proj.weight", identity)],
                process_fun=transpose,
            ),
            AtomicWeight(
                W.ffn_w3,
                [CkptWeightInfo(prefix + "mlp.up_proj.weight", identity)],
                process_fun=transpose,
            ),
            AtomicWeight(
                W.ffn_w2,
                [CkptWeightInfo(prefix + "mlp.down_proj.weight", identity)],
                process_fun=transpose,
            ),
        ]


__all__ = [
    "GEMMA4_ASSISTANT_POST_PROJECTION_W",
    "GEMMA4_ASSISTANT_PRE_PROJECTION_W",
    "GEMMA4_ASSISTANT_Q_PROJ_W",
    "Gemma4AssistantWeightInfo",
]

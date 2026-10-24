"""DiffusionGemma HF checkpoint -> RTP-LLM weight mapping.

Checkpoint keys (safetensors):
  model.decoder.embed_tokens.weight                -> embedding (tied lm_head)
  model.decoder.norm.weight                        -> final norm
  model.decoder.self_conditioning.{pre_norm,gate_proj,up_proj,down_proj}.weight
  model.decoder.layers.{i}.self_attn.{q,k,v}_proj.weight / {q,k}_norm.weight
  model.decoder.layers.{i}.{mlp.gate,mlp.up,mlp.down}_proj.weight
  model.decoder.layers.{i}.experts.{gate_up,down}_proj          (packed 3D)
  model.decoder.layers.{i}.router.{proj.weight,scale,per_expert_scale}
  model.decoder.layers.{i}.{input,post_attention,pre_feedforward,
      post_feedforward,pre_feedforward_2,post_feedforward_1,
      post_feedforward_2}_layernorm*.weight
  model.decoder.layers.{i}.layer_scalar              (decoder mode scalar)
  model.encoder.language_model.layers.{i}.layer_scalar (encoder mode scalar)
  model.encoder.vision_tower.* + model.encoder.embed_vision.*  (vision phase)

Global-attention layers (every 6th) have no v_proj: the loader duplicates the
k_proj output into the fused qkv's v slot (reference: V = v_norm(k_proj(x))).
"""

from typing import Any, List

import torch

from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    merge_qkv_hf,
    transpose,
    zeros,
)

DECODER_PREFIX = "model.decoder.layers.{i}."
ENCODER_PREFIX = "model.encoder.language_model.layers.{i}."


def merge_qkv_global(ts: List[torch.Tensor]) -> torch.Tensor:
    """Fused qkv for global layers: V is the k_proj output duplicated."""
    q, k = ts
    return torch.concat([q.T, k.T, k.T], dim=1).contiguous()


class DiffusionGemmaWeightInfo(ModelDeployWeightInfo):
    def __init__(self, **kwargs: Any):
        self.prefix = ""
        super().__init__(**kwargs)
        layer_types = getattr(self.model_config, "dgm_layer_types", None) or []
        self._layer_types = list(layer_types)

    def _layer_type(self, layer_id: int) -> str:
        if layer_id < len(self._layer_types):
            return self._layer_types[layer_id]
        return "sliding_attention"

    def _attn_config(self, layer_id: int) -> AttnConfig:
        config = AttnConfig(
            hidden_size=self._hidden_size,
            size_per_head=self._size_per_head,
            head_num=self._head_num,
            head_num_kv=self._head_num_kv,
            need_post_ln=False,
        )
        if self._layer_type(layer_id) == "full_attention":
            config.size_per_head = int(
                getattr(self.model_config, "dgm_global_head_dim", self._size_per_head)
            )
            config.head_num_kv = int(
                getattr(self.model_config, "dgm_global_kv_head_num", self._head_num_kv)
            )
        return config

    def _get_layer_weight_info(self, layer_id: int) -> List:
        is_global = self._layer_type(layer_id) == "full_attention"
        attn_config = self._attn_config(layer_id)
        layer_weights: List = [
            AttnAtomicWeight(
                W.attn_qkv_w,
                (
                    [
                        CkptWeightInfo(DECODER_PREFIX + "self_attn.q_proj.weight"),
                        CkptWeightInfo(DECODER_PREFIX + "self_attn.k_proj.weight"),
                        CkptWeightInfo(DECODER_PREFIX + "self_attn.v_proj.weight"),
                    ]
                    if not is_global
                    else [
                        CkptWeightInfo(DECODER_PREFIX + "self_attn.q_proj.weight"),
                        CkptWeightInfo(DECODER_PREFIX + "self_attn.k_proj.weight"),
                    ]
                ),
                merge_qkv_hf if not is_global else merge_qkv_global,
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.attn_o_w,
                [CkptWeightInfo(DECODER_PREFIX + "self_attn.o_proj.weight")],
                transpose,
                config=attn_config,
            ),
            AtomicWeight(
                W.q_ln_gamma,
                [CkptWeightInfo(DECODER_PREFIX + "self_attn.q_norm.weight")],
                identity,
            ),
            AtomicWeight(
                W.k_ln_gamma,
                [CkptWeightInfo(DECODER_PREFIX + "self_attn.k_norm.weight")],
                identity,
            ),
            AtomicWeight(
                W.pre_ln_gamma,
                [CkptWeightInfo(DECODER_PREFIX + "input_layernorm.weight")],
                identity,
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [CkptWeightInfo(DECODER_PREFIX + "post_attention_layernorm.weight")],
                identity,
            ),
            AtomicWeight(
                W.dgm_pre_ffn_ln_gamma,
                [CkptWeightInfo(DECODER_PREFIX + "pre_feedforward_layernorm.weight")],
                identity,
            ),
            AtomicWeight(
                W.dgm_post_ffn_ln_gamma,
                [CkptWeightInfo(DECODER_PREFIX + "post_feedforward_layernorm.weight")],
                identity,
            ),
            AtomicWeight(
                W.dgm_pre_ffn_ln2_gamma,
                [CkptWeightInfo(DECODER_PREFIX + "pre_feedforward_layernorm_2.weight")],
                identity,
            ),
            AtomicWeight(
                W.dgm_post_ffn_ln1_gamma,
                [CkptWeightInfo(DECODER_PREFIX + "post_feedforward_layernorm_1.weight")],
                identity,
            ),
            AtomicWeight(
                W.dgm_post_ffn_ln2_gamma,
                [CkptWeightInfo(DECODER_PREFIX + "post_feedforward_layernorm_2.weight")],
                identity,
            ),
            # Shared expert MLP (gated GeGLU).
            AtomicWeight(
                W.ffn_w1,
                [CkptWeightInfo(DECODER_PREFIX + "mlp.gate_proj.weight")],
                transpose,
            ),
            AtomicWeight(
                W.ffn_w3,
                [CkptWeightInfo(DECODER_PREFIX + "mlp.up_proj.weight")],
                transpose,
            ),
            AtomicWeight(
                W.ffn_w2,
                [CkptWeightInfo(DECODER_PREFIX + "mlp.down_proj.weight")],
                transpose,
            ),
            # Routed experts, packed 3D in the HF layout [E, 2I, H] / [E, H, I];
            # the v1 executor consumes them directly (gate first).
            AtomicWeight(
                W.moe_w1,
                [CkptWeightInfo(DECODER_PREFIX + "experts.gate_up_proj")],
                identity,
            ),
            AtomicWeight(
                W.moe_w2,
                [CkptWeightInfo(DECODER_PREFIX + "experts.down_proj")],
                identity,
            ),
            # Router.
            AtomicWeight(
                W.moe_gate,
                [CkptWeightInfo(DECODER_PREFIX + "router.proj.weight")],
                transpose,
            ),
            AtomicWeight(
                W.dgm_router_scale,
                [CkptWeightInfo(DECODER_PREFIX + "router.scale")],
                identity,
            ),
            AtomicWeight(
                W.dgm_router_per_expert_scale,
                [CkptWeightInfo(DECODER_PREFIX + "router.per_expert_scale")],
                identity,
            ),
            # Mode-specific layer scalars.
            AtomicWeight(
                W.dgm_layer_scalar_dec,
                [CkptWeightInfo(DECODER_PREFIX + "layer_scalar")],
                identity,
            ),
            AtomicWeight(
                W.dgm_layer_scalar_enc,
                [CkptWeightInfo(ENCODER_PREFIX + "layer_scalar")],
                identity,
            ),
        ]
        return layer_weights

    def _get_weight_info(self) -> ModelWeightInfo:
        weights: List = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("model.decoder.embed_tokens.weight")],
                identity,
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("model.decoder.embed_tokens.weight")],
                identity,
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo("model.decoder.norm.weight")],
                identity,
            ),
            AtomicWeight(
                W.dgm_self_cond_pre_norm,
                [CkptWeightInfo("model.decoder.self_conditioning.pre_norm.weight")],
                identity,
            ),
            AtomicWeight(
                W.dgm_self_cond_gate_w,
                [CkptWeightInfo("model.decoder.self_conditioning.gate_proj.weight")],
                transpose,
            ),
            AtomicWeight(
                W.dgm_self_cond_up_w,
                [CkptWeightInfo("model.decoder.self_conditioning.up_proj.weight")],
                transpose,
            ),
            AtomicWeight(
                W.dgm_self_cond_down_w,
                [CkptWeightInfo("model.decoder.self_conditioning.down_proj.weight")],
                transpose,
            ),
        ]
        layer_weights = [
            self._get_layer_weight_info(layer) for layer in range(self._num_layers)
        ]
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)

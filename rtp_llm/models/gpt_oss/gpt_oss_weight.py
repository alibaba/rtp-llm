"""gpt-oss weight info: HF checkpoint -> framework W.* mapping.

Checkpoint layout (openai/gpt-oss-20b & 120b, verified from safetensors
headers, SOURCE_FACT):
  model.embed_tokens.weight                      BF16 [vocab, 2880]
  model.norm.weight                              BF16 [2880]
  lm_head.weight                                 BF16 [vocab, 2880]
  model.layers.{i}.input_layernorm.weight        BF16 [2880]
  model.layers.{i}.post_attention_layernorm.weight BF16 [2880]
  model.layers.{i}.self_attn.{q,k,v,o}_proj.weight BF16 [4096|512|512|2880, 2880]
  model.layers.{i}.self_attn.{q,k,v,o}_proj.bias   BF16 [4096|512|512|2880]
  model.layers.{i}.self_attn.sinks               BF16 [64] (per-head)
  model.layers.{i}.mlp.router.weight             BF16 [E, 2880]
  model.layers.{i}.mlp.router.bias               BF16 [E]
  model.layers.{i}.mlp.experts.gate_up_proj_blocks U8 [E, 2*inter, K/32, 16]
  model.layers.{i}.mlp.experts.gate_up_proj_scales U8 [E, 2*inter, K/32]
  model.layers.{i}.mlp.experts.gate_up_proj_bias  BF16 [E, 2*inter] (rows interleaved gate/up)
  model.layers.{i}.mlp.experts.down_proj_blocks    U8 [E, D, K/32, 16]
  model.layers.{i}.mlp.experts.down_proj_scales   U8 [E, D, K/32]
  model.layers.{i}.mlp.experts.down_proj_bias     BF16 [E, D]

The routed experts are MXFP4: E2M1 nibble-packed (low nibble = even K,
high nibble = odd K) with UE8M0 group-32 scales — byte-identical to the
DeepSeek V4 `V4PerGroupFp4Weight` scheme, so they are declared int8 (not
uint8) to stay on the DeepGEMM fp8_fp4 path (uint8 falls back to BF16
dequant). gate/up rows are interleaved in the checkpoint (even=gate,
odd=up); `deinterleave_stack_moe_w1` reorders them into the canonical
[gate; up] moe_w1 layout (biases likewise).
"""

from typing import List

import torch

from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    deinterleave_stack_moe_w1,
    flatten_stack_moe,
    identity,
    merge_qkv_b,
    merge_qkv_hf,
    transpose,
)

# NOTE on expert scale dtype: the checkpoint stores UE8M0 scales as raw U8.
# They are declared (and stay) uint8 through the loader — a numeric
# ``.to(float8_e8m0fnu)`` would corrupt the exponent bytes (e8m0 has no
# mantissa), and the loader's post-process ``.to(declared_dtype)`` would
# equally corrupt an e8m0 view. The grouped-fp4 executor bit-views
# ``W.moe_s1/W.moe_s2`` as float8_e8m0fnu when it consumes them.


class GptOssWeightInfo(ModelDeployWeightInfo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix = "model."

    def _get_weight_info(self) -> ModelWeightInfo:
        weights: List[WeightModule] = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo(self.prefix + "embed_tokens.weight", identity)],
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("lm_head.weight", identity)],
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo(self.prefix + "norm.weight", identity)],
            ),
        ]
        all_layer_weights = [
            self._get_hf_layer_weight_info(layer_id)
            for layer_id in range(self._num_layers)
        ]
        return ModelWeightInfo(layer_weights=all_layer_weights, weights=weights)

    def _get_hf_layer_weight_info(self, layer_id: int) -> List[WeightModule]:
        # layer_id kept for interface parity; gpt-oss layers are homogeneous.
        weights: List[WeightModule] = []
        weights.extend(self._create_layer_norm_weight())
        weights.extend(self._create_attention_weight())
        weights.extend(self._create_moe_weight())
        return weights

    def _create_layer_norm_weight(self) -> List[WeightModule]:
        # Plain RMSNorm (no +1 shift, unlike Gemma/Qwen3Next plus_one).
        return [
            AtomicWeight(
                W.pre_ln_gamma,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.input_layernorm.weight", identity
                    )
                ],
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.post_attention_layernorm.weight",
                        identity,
                    )
                ],
            ),
        ]

    def _create_attention_weight(self) -> List[WeightModule]:
        attn_config: AttnConfig = self.attn_config
        prefix = self.prefix + "layers.{i}.self_attn."
        return [
            AttnAtomicWeight(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(prefix + "q_proj.weight", identity),
                    CkptWeightInfo(prefix + "k_proj.weight", identity),
                    CkptWeightInfo(prefix + "v_proj.weight", identity),
                ],
                process_fun=merge_qkv_hf,
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.attn_qkv_b,
                [
                    CkptWeightInfo(prefix + "q_proj.bias", identity),
                    CkptWeightInfo(prefix + "k_proj.bias", identity),
                    CkptWeightInfo(prefix + "v_proj.bias", identity),
                ],
                process_fun=merge_qkv_b,
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.attn_o_w,
                [CkptWeightInfo(prefix + "o_proj.weight", identity)],
                process_fun=transpose,
                config=attn_config,
            ),
            AtomicWeight(
                W.attn_o_b,
                [CkptWeightInfo(prefix + "o_proj.bias", identity)],
            ),
            AtomicWeight(
                W.attn_sink,
                [CkptWeightInfo(prefix + "sinks", identity)],
            ),
        ]

    def _create_moe_weight(self) -> List[WeightModule]:
        moe_config = MoeConfig(
            expert_num=self.model_config.expert_num,
            align_size=0,
        )
        expert_prefix = self.prefix + "layers.{i}.mlp.experts."
        return [
            # Router: BF16 Linear [E, H] + bias; ranking on raw biased logits,
            # weights = softmax over the top-k selected logits (gpt-oss).
            AtomicWeight(
                W.moe_gate,
                [CkptWeightInfo(self.prefix + "layers.{i}.mlp.router.weight", identity)],
            ),
            AtomicWeight(
                W.moe_gate_bias,
                [CkptWeightInfo(self.prefix + "layers.{i}.mlp.router.bias", identity)],
            ),
            # Routed experts: MXFP4 (int8-packed E2M1 + UE8M0 group-32 scales),
            # already stacked [E, N, ...] per tensor -> stacked_ckpt_keys.
            MoeAtomicWeight(
                W.moe_w1,
                [CkptWeightInfo(expert_prefix + "gate_up_proj_blocks", identity)],
                process_fun=deinterleave_stack_moe_w1,
                config=moe_config,
                data_type=torch.int8,
                stacked_ckpt_keys=True,
            ),
            MoeAtomicWeight(
                W.moe_s1,
                [CkptWeightInfo(expert_prefix + "gate_up_proj_scales", identity)],
                process_fun=deinterleave_stack_moe_w1,
                config=moe_config,
                data_type=torch.uint8,
                stacked_ckpt_keys=True,
            ),
            MoeAtomicWeight(
                W.moe_b1,
                [CkptWeightInfo(expert_prefix + "gate_up_proj_bias", identity)],
                process_fun=deinterleave_stack_moe_w1,
                config=moe_config,
                data_type=torch.bfloat16,
                stacked_ckpt_keys=True,
            ),
            MoeAtomicWeight(
                W.moe_w2,
                [CkptWeightInfo(expert_prefix + "down_proj_blocks", identity)],
                process_fun=flatten_stack_moe,
                config=moe_config,
                data_type=torch.int8,
                stacked_ckpt_keys=True,
            ),
            MoeAtomicWeight(
                W.moe_s2,
                [CkptWeightInfo(expert_prefix + "down_proj_scales", identity)],
                process_fun=flatten_stack_moe,
                config=moe_config,
                data_type=torch.uint8,
                stacked_ckpt_keys=True,
            ),
            MoeAtomicWeight(
                W.moe_b2,
                [CkptWeightInfo(expert_prefix + "down_proj_bias", identity)],
                process_fun=flatten_stack_moe,
                config=moe_config,
                data_type=torch.bfloat16,
                stacked_ckpt_keys=True,
            ),
        ]

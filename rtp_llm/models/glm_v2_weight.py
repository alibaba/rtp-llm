import functools
from typing import List

from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, FfnConfig, FfnWeight
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.utils.chatglm2_quantization import extract_weight_to_half
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    concat_1,
    identity,
    transpose,
    w_half1_t,
    w_half2_t,
    zeros,
)


class GlmV2WeightInfo(ModelDeployWeightInfo):
    def _process_meta(self, meta_dicts, weight_keys):
        if "transformer.prefix_encoder.embedding.weight" in weight_keys:
            self._has_prefix_encoder = True

    def _get_weight_info(self):
        weights = [
            AtomicWeight(
                W.embedding,
                [
                    CkptWeightInfo(
                        "transformer.embedding.word_embeddings.weight", concat_1
                    )
                ],
                identity,
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("transformer.output_layer.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [
                    CkptWeightInfo(
                        "transformer.encoder.final_layernorm.weight", identity
                    )
                ],
                identity,
            ),
            AtomicWeight(
                W.final_ln_beta, [], functools.partial(zeros, shape=[self._hidden_size])
            ),
        ]
        if self._has_prefix_encoder:
            weights.append(
                AtomicWeight(
                    W.prefix_w,
                    [
                        CkptWeightInfo(
                            "transformer.prefix_encoder.embedding.weight", identity
                        )
                    ],
                    identity,
                )
            )

        attn_config: AttnConfig = self.attn_config
        ffn_config: FfnConfig = self.ffn_config
        layer_weights: List[WeightModule] = [
            AtomicWeight(
                W.pre_ln_gamma,
                [
                    CkptWeightInfo(
                        "transformer.encoder.layers.{i}.input_layernorm.weight",
                        identity,
                    )
                ],
                identity,
            ),
            AttnAtomicWeight(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(
                        "transformer.encoder.layers.{i}.self_attention.query_key_value.weight",
                        identity,
                    )
                ],
                transpose,
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.attn_qkv_b,
                [
                    CkptWeightInfo(
                        "transformer.encoder.layers.{i}.self_attention.query_key_value.bias",
                        identity,
                    )
                ],
                identity,
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.attn_o_w,
                [
                    CkptWeightInfo(
                        "transformer.encoder.layers.{i}.self_attention.dense.weight",
                        identity,
                    )
                ],
                transpose,
                config=attn_config,
            ),
            FfnWeight(
                sub_weights=[
                    FfnAtomicWeight(
                        W.ffn_w1,
                        [
                            CkptWeightInfo(
                                "transformer.encoder.layers.{i}.mlp.dense_h_to_4h.weight",
                                identity,
                            )
                        ],
                        functools.partial(w_half1_t, inter_size=self._align_size),
                        config=ffn_config,
                    ),
                    FfnAtomicWeight(
                        W.ffn_w3,
                        [
                            CkptWeightInfo(
                                "transformer.encoder.layers.{i}.mlp.dense_h_to_4h.weight",
                                identity,
                            )
                        ],
                        functools.partial(w_half2_t, inter_size=self._align_size),
                        config=ffn_config,
                    ),
                    FfnAtomicWeight(
                        W.ffn_w2,
                        [
                            CkptWeightInfo(
                                "transformer.encoder.layers.{i}.mlp.dense_4h_to_h.weight",
                                identity,
                            )
                        ],
                        transpose,
                        config=ffn_config,
                    ),
                ],
                config=ffn_config,
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [
                    CkptWeightInfo(
                        "transformer.encoder.layers.{i}.post_attention_layernorm.weight",
                        identity,
                    )
                ],
                identity,
            ),
        ]

        assert self._src_quantization_bit == 0 or self._src_quantization_bit in [4, 8]
        if self._src_quantization_bit in [4, 8]:
            for idx, layer_weight in enumerate(layer_weights):
                new_weight = layer_weight.weights + [
                    CkptWeightInfo(
                        layer_weight.weights[0].name + "_scale",
                        functools.partial(identity, allow_empty=True),
                    )
                ]
                layer_weights[idx] = AtomicWeight(
                    layer_weight.name,
                    new_weight,
                    functools.partial(
                        extract_weight_to_half,
                        source_bit_width=self._src_quantization_bit,
                        sufix_func=layer_weight.process_fun,
                    ),
                )

        model_weight_info = ModelWeightInfo(
            layer_weights=layer_weights, weights=weights
        )

        return model_weight_info

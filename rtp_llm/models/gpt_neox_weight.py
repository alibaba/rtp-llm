import functools

from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, FfnConfig, FfnWeight
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    trans_qkv,
    trans_qkv_b,
    transpose,
    zeros,
)


class GPTNeoxWeight(ModelDeployWeightInfo):
    def _get_weight_info(self):
        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("gpt_neox.embed_in.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.lm_head, [CkptWeightInfo("embed_out.weight", identity)], identity
            ),
        ]

        attn_config: AttnConfig = self.attn_config
        ffn_config: FfnConfig = self.ffn_config
        layer_weights = []
        for _ in range(self._num_layers):
            layer_weight = [
                AttnAtomicWeight(
                    W.attn_qkv_w,
                    [
                        CkptWeightInfo(
                            "gpt_neox.layers.{i}.attention.query_key_value.weight",
                            identity,
                        )
                    ],
                    functools.partial(
                        trans_qkv,
                        hidden_size=self._hidden_size,
                        head_num=self._head_num,
                    ),
                    config=attn_config,
                ),
                AttnAtomicWeight(
                    W.attn_qkv_b,
                    [
                        CkptWeightInfo(
                            "gpt_neox.layers.{i}.attention.query_key_value.bias",
                            identity,
                        )
                    ],
                    functools.partial(
                        trans_qkv_b,
                        hidden_size=self._hidden_size,
                        head_num=self._head_num,
                    ),
                    config=attn_config,
                ),
                AttnAtomicWeight(
                    W.attn_o_w,
                    [
                        CkptWeightInfo(
                            "gpt_neox.layers.{i}.attention.dense.weight", identity
                        )
                    ],
                    transpose,
                    config=attn_config,
                ),
                AttnAtomicWeight(
                    W.attn_o_b,
                    [
                        CkptWeightInfo(
                            "gpt_neox.layers.{i}.attention.dense.bias", identity
                        )
                    ],
                    identity,
                    config=attn_config,
                ),
                FfnWeight(
                    sub_weights=[
                        FfnAtomicWeight(
                            W.ffn_w3,
                            [
                                CkptWeightInfo(
                                    "gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight",
                                    identity,
                                )
                            ],
                            transpose,
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_b3,
                            [
                                CkptWeightInfo(
                                    "gpt_neox.layers.{i}.mlp.dense_h_to_4h.bias",
                                    identity,
                                )
                            ],
                            identity,
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_w2,
                            [
                                CkptWeightInfo(
                                    "gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight",
                                    identity,
                                )
                            ],
                            transpose,
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_b2,
                            [
                                CkptWeightInfo(
                                    "gpt_neox.layers.{i}.mlp.dense_4h_to_h.bias",
                                    identity,
                                )
                            ],
                            identity,
                            config=ffn_config,
                        ),
                    ],
                    config=ffn_config,
                ),
            ]

            # default use parallel residual: x = x + attn(ln1(x)) + mlp(ln2(x))

            if self.model_config.norm == "rmsnorm":
                weights.extend(
                    [
                        AtomicWeight(
                            W.final_ln_gamma,
                            [
                                CkptWeightInfo(
                                    "gpt_neox.final_layer_norm.scale", identity
                                )
                            ],
                            identity,
                        ),
                        AtomicWeight(
                            W.final_ln_beta,
                            [],
                            functools.partial(zeros, shape=[self._hidden_size]),
                        ),
                    ]
                )
                layer_weights.extend(
                    [
                        AtomicWeight(
                            W.pre_attn_ln_gamma,
                            [
                                CkptWeightInfo(
                                    "gpt_neox.layers.{i}.input_layernorm.scale",
                                    identity,
                                )
                            ],
                            identity,
                        ),
                        AtomicWeight(
                            W.pre_ln_gamma,
                            [
                                CkptWeightInfo(
                                    "gpt_neox.layers.{i}.post_attention_layernorm.scale",
                                    identity,
                                )
                            ],
                            identity,
                        ),
                    ]
                )
            elif self.model_config.norm == "layernorm":
                weights.extend(
                    [
                        AtomicWeight(
                            W.final_ln_gamma,
                            [
                                CkptWeightInfo(
                                    "gpt_neox.final_layer_norm.weight", identity
                                )
                            ],
                            identity,
                        ),
                        AtomicWeight(
                            W.final_ln_beta,
                            [
                                CkptWeightInfo(
                                    "gpt_neox.final_layer_norm.bias", identity
                                )
                            ],
                            identity,
                        ),
                    ]
                )
                layer_weights.extend(
                    [
                        AtomicWeight(
                            W.pre_attn_ln_gamma,
                            [
                                CkptWeightInfo(
                                    "gpt_neox.layers.{i}.input_layernorm.weight",
                                    identity,
                                )
                            ],
                            identity,
                        ),
                        AtomicWeight(
                            W.pre_attn_ln_beta,
                            [
                                CkptWeightInfo(
                                    "gpt_neox.layers.{i}.input_layernorm.bias", identity
                                )
                            ],
                            identity,
                        ),
                        AtomicWeight(
                            W.pre_ln_gamma,
                            [
                                CkptWeightInfo(
                                    "gpt_neox.layers.{i}.post_attention_layernorm.weight",
                                    identity,
                                )
                            ],
                            identity,
                        ),
                        AtomicWeight(
                            W.pre_ln_beta,
                            [
                                CkptWeightInfo(
                                    "gpt_neox.layers.{i}.post_attention_layernorm.bias",
                                    identity,
                                )
                            ],
                            identity,
                        ),
                    ]
                )
                layer_weights.append(layer_weight)
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


class GPTNeox13BWeight(ModelDeployWeightInfo):
    def _get_weight_info(self):
        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("gpt_neox.embed_in.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.lm_head, [CkptWeightInfo("embed_out.weight", identity)], identity
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo("gpt_neox.final_layer_norm.scale", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_beta, [], functools.partial(zeros, shape=[self._hidden_size])
            ),
        ]
        attn_config: AttnConfig = self.attn_config
        ffn_config: FfnConfig = self.ffn_config

        layer_weights = []
        for _ in range(self._num_layers):
            layer_weight = [
                AtomicWeight(
                    W.pre_ln_gamma,
                    [
                        CkptWeightInfo(
                            "gpt_neox.layers.{i}.input_layernorm.scale", identity
                        )
                    ],
                    identity,
                ),
                AttnAtomicWeight(
                    W.attn_qkv_w,
                    [
                        CkptWeightInfo(
                            "gpt_neox.layers.{i}.attention.query_key_value.weight",
                            identity,
                        )
                    ],
                    functools.partial(
                        trans_qkv,
                        hidden_size=self._hidden_size,
                        head_num=self._head_num,
                    ),
                    config=attn_config,
                ),
                AttnAtomicWeight(
                    W.attn_qkv_b,
                    [
                        CkptWeightInfo(
                            "gpt_neox.layers.{i}.attention.query_key_value.bias",
                            identity,
                        )
                    ],
                    functools.partial(
                        trans_qkv_b,
                        hidden_size=self._hidden_size,
                        head_num=self._head_num,
                    ),
                    config=attn_config,
                ),
                AttnAtomicWeight(
                    W.attn_o_w,
                    [
                        CkptWeightInfo(
                            "gpt_neox.layers.{i}.attention.dense.weight", identity
                        )
                    ],
                    transpose,
                    config=attn_config,
                ),
                AttnAtomicWeight(
                    W.attn_o_b,
                    [
                        CkptWeightInfo(
                            "gpt_neox.layers.{i}.attention.dense.bias", identity
                        )
                    ],
                    identity,
                    config=attn_config,
                ),
                FfnWeight(
                    sub_weights=[
                        FfnAtomicWeight(
                            W.ffn_w3,
                            [
                                CkptWeightInfo(
                                    "gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight",
                                    identity,
                                )
                            ],
                            transpose,
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_b3,
                            [
                                CkptWeightInfo(
                                    "gpt_neox.layers.{i}.mlp.dense_h_to_4h.bias",
                                    identity,
                                )
                            ],
                            identity,
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_w2,
                            [
                                CkptWeightInfo(
                                    "gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight",
                                    identity,
                                )
                            ],
                            transpose,
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_b2,
                            [
                                CkptWeightInfo(
                                    "gpt_neox.layers.{i}.mlp.dense_4h_to_h.bias",
                                    identity,
                                )
                            ],
                            identity,
                            config=ffn_config,
                        ),
                    ],
                    config=ffn_config,
                ),
                AtomicWeight(
                    W.post_ln_gamma,
                    [
                        CkptWeightInfo(
                            "gpt_neox.layers.{i}.post_attention_layernorm.scale",
                            identity,
                        )
                    ],
                    identity,
                ),
            ]
            layer_weights.append(layer_weight)

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)

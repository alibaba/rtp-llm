"""Checkpoint layouts for the EmbeddingGemma 2 encoder (no LM head)."""

from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, FfnWeight
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight, CustomAtomicWeight
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    merge_qkv_hf,
    transpose,
)

PLE_PROJECTION = "embeddinggemma2.ple_projection"
PLE_NORM = "embeddinggemma2.ple_norm"
PLE_GATE = "embeddinggemma2.ple_gate"
PLE_OUTPUT = "embeddinggemma2.ple_output"
PLE_OUTPUT_NORM = "embeddinggemma2.ple_output_norm"
EMBEDDING_PROJECTION = "embeddinggemma2.embedding_projection"


class EmbeddingGemma2WeightInfo(ModelDeployWeightInfo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._quant_algo.isQuant():
            raise ValueError(
                "EmbeddingGemma2 currently supports unquantized BF16/FP32 weights"
            )
        if self.tp_size != 1 or self.ffn_tp_size != 1 or self.ep_size != 1:
            raise ValueError("EmbeddingGemma2 currently requires TP/EP=1")
        self.prefix = "language_model."

    def _process_meta(self, meta_dicts, weight_keys):
        suffix = "ple.per_layer_model_projection.weight"
        matches = [key[: -len(suffix)] for key in weight_keys if key.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError("EmbeddingGemma2 requires exactly one text PLE projection")
        self.prefix = matches[0]

    def _weight(self, name, source, process=identity, custom=False):
        cls = CustomAtomicWeight if custom else AtomicWeight
        return cls(
            name, [CkptWeightInfo(self.prefix + source, identity)], process_fun=process
        )

    def _get_weight_info(self):
        global_weights = [
            self._weight(W.embedding, "embed_tokens.weight"),
            self._weight(W.final_ln_gamma, "norm.weight"),
            self._weight(
                PLE_PROJECTION, "ple.per_layer_model_projection.weight", transpose, True
            ),
            self._weight(PLE_NORM, "ple.per_layer_projection_norm.weight", custom=True),
            self._weight(
                EMBEDDING_PROJECTION, "embedding_projection.weight", transpose, True
            ),
        ]
        layers = []
        for i, geometry in enumerate(
            self.model_config.mm_related_params.config["embeddinggemma2_layers"][
                : self._num_layers
            ]
        ):
            attn = AttnConfig(
                hidden_size=self._hidden_size,
                size_per_head=geometry["head_dim"],
                head_num=self._head_num,
                head_num_kv=geometry["kv_heads"],
            )
            root = self.prefix + "layers.{i}."
            weights = [
                AttnAtomicWeight(
                    W.attn_qkv_w,
                    [
                        CkptWeightInfo(root + f"self_attn.{part}_proj.weight", identity)
                        for part in ("q", "k", "v")
                    ],
                    process_fun=merge_qkv_hf,
                    config=attn,
                ),
                AttnAtomicWeight(
                    W.attn_o_w,
                    [CkptWeightInfo(root + "self_attn.o_proj.weight", identity)],
                    process_fun=transpose,
                    config=attn,
                ),
            ]
            for name, source in (
                (W.pre_ln_gamma, "input_layernorm.weight"),
                (W.post_ln_gamma, "post_attention_layernorm.weight"),
                (W.pre_ffn_ln_gamma, "pre_feedforward_layernorm.weight"),
                (W.post_ffn_ln_gamma, "post_feedforward_layernorm.weight"),
                (W.q_ln_gamma, "self_attn.q_norm.weight"),
                (W.k_ln_gamma, "self_attn.k_norm.weight"),
                (W.layer_scalar, "layer_scalar"),
            ):
                weights.append(self._weight(name, "layers.{i}." + source))
            for name, source in (
                (PLE_GATE, "ple_block.per_layer_input_gate.weight"),
                (PLE_OUTPUT, "ple_block.per_layer_projection.weight"),
                (PLE_OUTPUT_NORM, "ple_block.post_per_layer_input_norm.weight"),
            ):
                weights.append(
                    self._weight(
                        name,
                        "layers.{i}." + source,
                        identity if name == PLE_OUTPUT_NORM else transpose,
                        True,
                    )
                )
            ffn = self.ffn_config
            weights.append(
                FfnWeight(
                    [
                        FfnAtomicWeight(
                            name,
                            [
                                CkptWeightInfo(
                                    root + "mlp." + source + ".weight", identity
                                )
                            ],
                            process_fun=transpose,
                            config=ffn,
                        )
                        for name, source in (
                            (W.ffn_w1, "gate_proj"),
                            (W.ffn_w3, "up_proj"),
                            (W.ffn_w2, "down_proj"),
                        )
                    ],
                    config=ffn,
                )
            )
            layers.append(weights)
        return ModelWeightInfo(weights=global_weights, layer_weights=layers)

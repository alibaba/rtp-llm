"""Qwen4-Exp one-layer MTP draft model."""

import json
import os
from typing import Any, Collection, Dict, List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.qwen4_exp.qwen4_exp import Qwen4Exp
from rtp_llm.models.qwen4_exp.qwen4_exp_weight import Qwen4ExpWeight
from rtp_llm.ops import HybridAttentionType, RopeStyle
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, transpose

_MTP_PREFIX = "mtp."

_MTP_INDEXER_WEIGHTS = (
    (W.qwen4_indexer_qk_proj_w, "indexer.index_qk_proj.weight"),
    (W.qwen4_indexer_q_ln_gamma, "indexer.q_layernorm.weight"),
    (W.qwen4_indexer_k_ln_gamma, "indexer.k_layernorm.weight"),
)


class Qwen4ExpMTPWeight(Qwen4ExpWeight):
    """Map the real ``mtp.*`` namespace without borrowing Qwen3.5's FC ABI."""

    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]):
        super().__init__(*args, **kwargs)
        self.prefix = _MTP_PREFIX

    def _process_meta(self, meta_dict: Any, weight_keys: Collection[str]):
        anchor = _MTP_PREFIX + "layers.0.attn_hyper_connection.hc_norm.weight"
        if anchor not in weight_keys:
            raise ValueError(
                "Qwen4ExpMTPWeight: checkpoint is missing the one-layer MTP "
                f"anchor {anchor!r}"
            )
        self.prefix = _MTP_PREFIX
        self._has_stacked_ckpt = self._contains(
            weight_keys, _MTP_PREFIX + "layers.0.mlp.experts.gate_up_proj"
        )

    def _append_indexer_weights(self, layer_weights: List[List[WeightModule]]) -> None:
        # The released draft layer always owns these tensors.  Declare them even
        # while QSA execution is feature-gated so manifest validation cannot
        # silently lose part of the checkpoint.
        if len(layer_weights) != 1:
            raise ValueError(
                "qwen4_exp MTP weight mapping requires exactly one draft layer"
            )
        layer_weights[0].extend(
            AtomicWeight(
                name,
                [CkptWeightInfo(_MTP_PREFIX + "layers.{i}.self_attn." + source)],
            )
            for name, source in _MTP_INDEXER_WEIGHTS
        )

    def _append_ple_weights(self, layer_weights: List[List[WeightModule]]) -> None:
        # PLE belongs to the target backbone only.  There are no ``mtp.*.ple``
        # tensors in the released manifest.
        return

    def _create_global_weights(self) -> List[WeightModule]:
        mixer = _MTP_PREFIX + "hyper_connection_mixer."
        return [
            # The draft checkpoint intentionally omits both vocabulary matrices;
            # BaseModel aliases these descriptors from the validated target owner.
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("model.language_model.embed_tokens.weight", identity)],
            ),
            AtomicWeight(W.lm_head, [CkptWeightInfo("lm_head.weight", identity)]),
            # All Qwen4 plain RMSNorm tensors stay raw.  Their consumers perform
            # ``1 + gamma.float()``; folding here in BF16 changes real values.
            AtomicWeight(
                W.multi_tokens_predict_enorm,
                [CkptWeightInfo(_MTP_PREFIX + "pre_fc_norm_embedding.weight")],
            ),
            AtomicWeight(
                W.multi_tokens_predict_hnorm,
                [CkptWeightInfo(_MTP_PREFIX + "pre_fc_norm_hidden.weight")],
            ),
            AtomicWeight(
                W.qwen4_mtp_fc_embedding_w,
                [CkptWeightInfo(_MTP_PREFIX + "fc_embedding.weight")],
                transpose,
            ),
            AtomicWeight(
                W.qwen4_mtp_fc_hidden_w,
                [CkptWeightInfo(_MTP_PREFIX + "fc_hidden.weight")],
                transpose,
            ),
            AtomicWeight(
                W.qwen4_hc_mixer_norm,
                [CkptWeightInfo(mixer + "hc_norm.weight")],
            ),
            AtomicWeight(
                W.qwen4_hc_mixer_mix_down,
                [CkptWeightInfo(mixer + "input_mix_weight_down.weight")],
            ),
            AtomicWeight(
                W.qwen4_hc_mixer_mix_up,
                [CkptWeightInfo(mixer + "input_mix_weight_up.weight")],
            ),
        ]


class Qwen4ExpMTP(Qwen4Exp):
    """Released one-layer draft using the restricted QSA MTP correctness path."""

    @staticmethod
    def get_weight_cls():
        return Qwen4ExpMTPWeight

    def support_cuda_graph(self) -> bool:
        return False

    def load(self, skip_python_model: bool = False):
        config = getattr(self, "model_config", None)
        if not bool(getattr(config, "enable_qwen4_qsa", False)):
            raise RuntimeError(
                "qwen4_exp MTP requires RTP_LLM_ENABLE_QWEN4_EXP_QSA=true; "
                "dense draft fallback is forbidden"
            )
        return super().load(skip_python_model=skip_python_model)

    def _py_model_class(self):
        from rtp_llm.models_py.model_desc.qwen4_exp_mtp import Qwen4ExpMTPModel

        return Qwen4ExpMTPModel

    @classmethod
    def speculative_weight_alias_names(
        cls, target_model: Qwen4Exp, draft_model_config: ModelConfig
    ) -> tuple[str, ...]:
        if not isinstance(target_model, Qwen4Exp) or isinstance(
            target_model, Qwen4ExpMTP
        ):
            raise TypeError("qwen4_exp MTP requires a Qwen4Exp target owner")
        if getattr(draft_model_config, "_qwen4_mtp_use_dedicated_embeddings", True):
            raise ValueError(
                "qwen4_exp MTP cannot alias a checkpoint that requests dedicated "
                "draft embeddings"
            )

        target_config = target_model.model_config
        if bool(getattr(target_config, "enable_output_vocab_pruning", False)) or bool(
            getattr(draft_model_config, "enable_output_vocab_pruning", False)
        ):
            raise ValueError(
                "qwen4_exp MTP cannot alias lm_head when output-vocab pruning is "
                "enabled on the target or draft"
            )
        compatible_fields = (
            "vocab_size",
            "hidden_size",
            "data_type",
            "enable_fp32_lm_head",
        )
        mismatches = [
            name
            for name in compatible_fields
            if getattr(target_config, name) != getattr(draft_model_config, name)
        ]
        if mismatches:
            details = ", ".join(
                f"{name}={getattr(target_config, name)!r}/"
                f"{getattr(draft_model_config, name)!r}"
                for name in mismatches
            )
            raise ValueError(
                "qwen4_exp MTP cannot alias semantically incompatible target "
                f"weights: {details}"
            )
        return (W.embedding, W.lm_head)

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = super()._create_config(ckpt_path)
        with open(os.path.join(ckpt_path, "config.json")) as reader:
            root_config = json.load(reader)
        text_config = root_config["text_config"]
        mtp_config = text_config.get("mtp")
        if not isinstance(mtp_config, dict):
            raise ValueError("qwen4_exp checkpoint has no MTP config")
        layer_types = list(mtp_config.get("layer_types", ()))
        if (
            mtp_config.get("hybrid") is not True
            or int(mtp_config.get("num_hidden_layers", 0)) != 1
            or layer_types != ["full_attention"]
        ):
            raise ValueError(
                "qwen4_exp MTP requires one full-attention draft layer, got "
                f"hybrid={mtp_config.get('hybrid')!r}, "
                f"num_hidden_layers={mtp_config.get('num_hidden_layers')!r}, "
                f"layer_types={layer_types!r}"
            )
        use_dedicated = bool(text_config.get("mtp_use_dedicated_embeddings", False))
        if use_dedicated:
            raise ValueError(
                "qwen4_exp MTP dedicated embeddings are not present in the "
                "released checkpoint"
            )
        hidden_source_layer = mtp_config.get("mtp_use_hidden_state_from_layer")
        if hidden_source_layer is not None:
            raise ValueError(
                "qwen4_exp MTP currently requires "
                "mtp_use_hidden_state_from_layer=null so the draft consumes the "
                f"target's final hidden state, got {hidden_source_layer!r}"
            )

        config.model_type = "qwen4_exp_mtp"
        config.num_layers = 1
        config.moe_layer_index = [0]
        config.is_mtp = True
        config.mtp_input_hidden_size = config.hidden_size * config.hc_mult
        config.hybrid_attention_config.enable_hybrid_attention = True
        config.hybrid_attention_config.hybrid_attention_types = [
            HybridAttentionType.NONE
        ]
        config.attn_config.rope_config.style = RopeStyle.Base
        config.attn_config.rope_config.base = int(mtp_config["rope_theta"])
        config.enable_qwen4_ple = False
        config._qwen4_ple_layer_ids = []
        # QSA is the released draft attention contract. Config construction may
        # keep it disabled for checkpoint-inspection tools, but load() refuses a
        # silent dense fallback.
        config.attn_config.is_sparse = config.enable_qwen4_qsa
        config.attn_config.use_sparse_gqa_fmha = config.enable_qwen4_qsa
        # The draft consumes text ids and aliases only the target vocabulary
        # matrices.  Leaving the target's multimodal flag set would make generic
        # sizing/loading code look up a non-existent qwen4_exp_mtp vision mixin.
        config.mm_model_config.is_multimodal = False
        config._extra_weight_bytes = 0.0
        config._tp_sharded_extra_weight_bytes = 0.0
        config._replicated_extra_weight_bytes = 0.0
        config._extra_weight_param_count = 0
        config._qwen4_mtp_use_dedicated_embeddings = use_dedicated
        return config


register_model("qwen4_exp_mtp", Qwen4ExpMTP)

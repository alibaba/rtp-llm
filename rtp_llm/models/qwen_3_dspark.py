"""Qwen3 DSpark draft-model registration."""

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.qwen_v3 import QwenV3
from rtp_llm.models_py.speculative.qwen3_dspark_checkpoint import (
    target_layer_ids as checkpoint_target_layer_ids,
)
from rtp_llm.utils.util import get_config_from_path


class Qwen3DSpark(QwenV3):
    @staticmethod
    def get_weight_cls():
        from rtp_llm.models.qwen_3_dspark_ft import Qwen3DSparkWeight

        return Qwen3DSparkWeight

    def support_pp(self) -> bool:
        return False

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        dspark = get_config_from_path(ckpt_path)
        assert dspark is not None, f"config.json missing under {ckpt_path}"
        backbone = dspark.get("transformer_layer_config", dspark)
        if not isinstance(backbone, dict):
            raise TypeError("transformer_layer_config must be an object")
        config = cls._create_config_from_json(ckpt_path, backbone)
        config.attn_config.is_causal = False
        config.input_vocab_size = int(backbone["vocab_size"])
        config.vocab_size = int(dspark.get("draft_vocab_size", backbone["vocab_size"]))
        config.dspark_noise_token_id = int(dspark["mask_token_id"])
        config.dspark_target_layer_ids = checkpoint_target_layer_ids(dspark)
        config.dspark_markov_rank = int(dspark.get("markov_rank", 0) or 0)
        if config.dspark_markov_rank <= 0:
            raise ValueError("Qwen3 DSpark requires markov_rank > 0")
        config.dspark_sample_from_anchor = bool(dspark.get("sample_from_anchor", False))
        # This opt-in assigns target ownership deliberately. No sharing is
        # inferred for existing checkpoints with their own language-model head.
        shares = dspark.get("dspark_shared_target_weights", [])
        if (
            not isinstance(shares, list)
            or any(role not in ("embedding", "lm_head") for role in shares)
            or len(set(shares)) != len(shares)
        ):
            raise ValueError(
                "dspark_shared_target_weights must list unique embedding/lm_head roles"
            )
        object.__setattr__(config, "_dspark_shared_target_weights", tuple(shares))
        return config

    @classmethod
    def speculative_weight_alias_names(cls, target_model, draft_model_config):
        # Vocab matrices in a self-contained draft (including the five-layer
        # training export) may be trained independently. Never substitute the
        # target just because two tensors have the same size or one is absent.
        roles = getattr(draft_model_config, "_dspark_shared_target_weights", ())
        if not roles:
            return ()
        target = target_model.model_config
        if target_model.py_model is None or target_model.weight is None or getattr(target, "is_mtp", False):
            raise ValueError("Qwen3 DSpARK loans require a loaded target, not a draft")
        if target.hidden_size != draft_model_config.hidden_size:
            raise ValueError("DSpARK target/draft hidden sizes differ")
        target_input_vocab = target.input_vocab_size or target.vocab_size
        if (
            "embedding" in roles
            and target_input_vocab != draft_model_config.input_vocab_size
        ):
            raise ValueError("DSpARK target/draft input vocabularies differ")
        if "lm_head" in roles:
            if (
                draft_model_config.vocab_size != draft_model_config.input_vocab_size
                or target.vocab_size != draft_model_config.vocab_size
                or getattr(target, "enable_output_vocab_pruning", False)
            ):
                raise ValueError(
                    "DSpARK cannot borrow a head with a different output vocabulary"
                )
        missing = [
            role for role in roles
            if target_model.weight.get_global_weight_or_none(role) is None
        ]
        if missing:
            raise ValueError(
                f"DSpARK cannot borrow {missing}: the target weights must be on the same pipeline stage as the draft"
            )
        return roles


register_model(
    "qwen_3_dspark", Qwen3DSpark, ["Qwen3DSparkForCausalLM", "Qwen3DSparkModel"]
)

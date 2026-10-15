"""MiniCPM-V-4.7 DSpark draft-model registration.

Draft checkpoint (openbmb/MiniCPM-V-4.7-35B-A3B-DSpark): 5 full-attention
Qwen3-style decoder layers reading the target model's hidden states at
``dflash_config.target_layer_ids`` (0-based, one per draft layer), a single
``fc`` projecting the concatenated target features into the draft hidden
space, a ``hidden_norm``, a low-rank Markov head (rank 256) and a
``confidence_head.proj`` (adaptive-verification input; loaded but not used by
the v1 verification path).
"""

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.qwen_3_dspark import Qwen3DSpark, Qwen3DSparkWeight
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, transpose
from rtp_llm.utils.util import get_config_from_path


class MiniCPMV47DSparkWeight(Qwen3DSparkWeight):
    """Qwen3 DSpark weights plus the confidence head projection."""

    def _get_weight_info(self):
        info = super()._get_weight_info()
        info.weights.extend(
            [
                AtomicWeight(
                    W.dspark_confidence_w,
                    [CkptWeightInfo("confidence_head.proj.weight", transpose)],
                ),
                AtomicWeight(
                    W.dspark_confidence_b,
                    [CkptWeightInfo("confidence_head.proj.bias", identity)],
                ),
            ]
        )
        return info


class MiniCPMV47DSpark(Qwen3DSpark):
    @classmethod
    def speculative_weight_alias_names(cls, target_model, draft_model_config):
        """Borrow embedding + lm_head from the MiniCPM-V-4.7 target owner.

        The draft checkpoint ships only the 5 backbone layers, fc,
        hidden_norm, markov and confidence heads (62 tensors, verified) —
        no vocabulary matrices.
        """
        from rtp_llm.models.minicpmv4_7 import MiniCPMV4_7

        if not isinstance(target_model, MiniCPMV4_7):
            raise TypeError("MiniCPM-V-4.7 DSpark requires a MiniCPM-V-4.7 target owner")
        target_config = target_model.model_config
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
                "MiniCPM-V-4.7 DSpark cannot alias semantically incompatible "
                f"target weights: {details}"
            )
        return (W.embedding, W.lm_head)

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        dspark = get_config_from_path(ckpt_path)
        assert dspark is not None, f"config.json missing under {ckpt_path}"
        dflash = dspark.get("dflash_config", {})
        backbone = dspark.get("transformer_layer_config", dspark)
        if not isinstance(backbone, dict):
            raise TypeError("transformer_layer_config must be an object")
        config = cls._create_config_from_json(ckpt_path, backbone)
        config.attn_config.is_causal = False
        config.input_vocab_size = int(backbone["vocab_size"])
        config.vocab_size = int(dspark.get("draft_vocab_size", backbone["vocab_size"]))
        config.dspark_noise_token_id = int(dflash["mask_token_id"])
        # MiniCPM DSpark config uses 0-based layer ids directly
        # (dflash_config.target_layer_ids, no speculators 1-based adjustment).
        target_layer_ids = [int(x) for x in dflash["target_layer_ids"]]
        config.dspark_target_layer_ids = target_layer_ids
        config.dspark_markov_rank = int(dspark.get("markov_rank", dflash.get("markov_rank", 0)) or 0)
        if config.dspark_markov_rank <= 0:
            raise ValueError("MiniCPM DSpark requires markov_rank > 0")
        config.dspark_sample_from_anchor = bool(dspark.get("sample_from_anchor", False))
        # Match the target's lm_head compute dtype so the borrowed
        # embed/lm_head weights stay semantically compatible (the alias check
        # compares this field).
        config.enable_fp32_lm_head = False
        return config

    @staticmethod
    def get_weight_cls():
        return MiniCPMV47DSparkWeight


register_model("minicpmv4_7_dspark", MiniCPMV47DSpark)

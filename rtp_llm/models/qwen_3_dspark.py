"""Qwen3 DSpark draft-model registration."""

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.qwen_v3 import QwenV3, QWenV3Weight
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, transpose
from rtp_llm.utils.util import get_config_from_path


class Qwen3DSparkWeight(QWenV3Weight):
    def _get_weight_info(self):
        info = super()._get_weight_info()
        extras = (
            (W.dspark_fc_w, "fc.weight", transpose),
            (W.dspark_hidden_norm_gamma, "hidden_norm.weight", identity),
            (W.dspark_markov_w1, "markov_head.markov_w1.weight", identity),
            (W.dspark_markov_w2, "markov_head.markov_w2.weight", identity),
        )
        info.weights.extend(
            AtomicWeight(name, [CkptWeightInfo(path, identity)], transform)
            for name, path, transform in extras
        )
        return info


class Qwen3DSpark(QwenV3):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        dspark = get_config_from_path(ckpt_path)
        assert dspark is not None, f"config.json missing under {ckpt_path}"
        backbone = dspark["transformer_layer_config"]
        if not isinstance(backbone, dict):
            raise TypeError("transformer_layer_config must be an object")
        config = cls._create_config_from_json(ckpt_path, backbone)
        config.attn_config.is_causal = False
        config.input_vocab_size = int(backbone["vocab_size"])
        config.vocab_size = int(dspark["draft_vocab_size"])
        config.dspark_noise_token_id = int(dspark["mask_token_id"])
        target_layer_ids = [int(x) for x in dspark["aux_hidden_state_layer_ids"]]
        # Speculators stores extraction points as 1-based layer outputs
        # (e.g. 8 means the hidden state after decoder layer index 7).  RTP's
        # capture hook is called with the zero-based decoder index.  Dense
        # DSpark configs already use RTP's zero-based convention.
        if dspark.get("speculators_model_type") == "dspark":
            target_layer_ids = [x - 1 for x in target_layer_ids]
            if any(x < 0 for x in target_layer_ids):
                raise ValueError(
                    "Speculators DSpark aux_hidden_state_layer_ids must be positive"
                )
        config.dspark_target_layer_ids = target_layer_ids
        config.dspark_markov_rank = int(dspark.get("markov_rank", 0) or 0)
        if config.dspark_markov_rank <= 0:
            raise ValueError("Qwen3 DSpark requires markov_rank > 0")
        # The released Speculators-format checkpoints use the legacy
        # bonus-anchor layout: one anchor row followed by K mask rows.  Dense
        # DSpark checkpoints default to treating the anchor as prediction 0.
        # An explicit checkpoint value always wins.
        if "sample_from_anchor" in dspark:
            sample_from_anchor = dspark["sample_from_anchor"]
        elif dspark.get("speculators_model_type") == "dspark":
            sample_from_anchor = False
        else:
            sample_from_anchor = not bool(dspark.get("dspark_bonus_anchor", False))
        config.dspark_sample_from_anchor = bool(sample_from_anchor)
        return config

    @staticmethod
    def get_weight_cls():
        return Qwen3DSparkWeight

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.qwen3_dspark_model import (
            Qwen3DSparkModel,
        )

        self.py_model = Qwen3DSparkModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            quant_config=self.model_config.quant_config,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model


register_model("qwen_3_dspark", Qwen3DSpark, ["Qwen3DSparkForCausalLM"])

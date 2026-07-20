"""qwen_3_dflash / qwen_3_dspark draft-model registration.

Loads the converted DFlash/DSpark draft checkpoint produced by
rtp_llm/tools/convert/dspark_ckpt_convert.py: a Qwen3 (qk-norm) backbone under
model.* plus top-level draft extras (fc, hidden_norm inside model.*, lm_head,
markov_head.*).  confidence_head.* tensors are intentionally unmapped — the
loader only pulls what the weight info declares, so they are ignored
(phase-1 No Goal, see docs/dspark-phase1-design-2026-07-14.md).

DSpark extends DFlash (mirroring upstream vLLM's qwen3_dflash/qwen3_dspark
split): Qwen3DFlash is the backbone + feature-KV base; Qwen3DSpark adds the
Markov head.  The two model types differ only in that extension — DFlash omits
the markov_head weights and uses greedy argmax at stage D, DSpark loads the
markov head and applies the low-rank transition-bias chain.
"""

from typing import List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.qwen_v3 import QwenV3, QWenV3Weight
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, transpose
from rtp_llm.utils.util import get_config_from_path


class Qwen3DFlashWeight(QWenV3Weight):
    """DFlash draft weights: Qwen3 backbone + fc + hidden_norm (no Markov head)."""

    def _draft_extra_weights(self) -> List[AtomicWeight]:
        """Draft-head weights beyond the shared DFlash base; empty for DFlash."""
        return []

    def _get_weight_info(self):
        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("model.embed_tokens.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("lm_head.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo("model.norm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.dspark_fc_w,
                [CkptWeightInfo("fc.weight", identity)],
                transpose,
            ),
            AtomicWeight(
                W.dspark_hidden_norm_gamma,
                [CkptWeightInfo("model.hidden_norm.weight", identity)],
                identity,
            ),
        ]
        weights.extend(self._draft_extra_weights())

        layer_weights: List[List[WeightModule]] = [
            self._get_hf_layer_weight_info(layer_id)
            for layer_id in range(self._num_layers)
        ]
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


class Qwen3DSparkWeight(Qwen3DFlashWeight):
    """DSpark draft weights: DFlash base + the low-rank Markov head."""

    def _draft_extra_weights(self) -> List[AtomicWeight]:
        return [
            AtomicWeight(
                W.dspark_markov_w1,
                [CkptWeightInfo("markov_head.markov_w1.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.dspark_markov_w2,
                [CkptWeightInfo("markov_head.markov_w2.weight", identity)],
                identity,
            ),
        ]


class Qwen3DFlash(QwenV3):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        # Deferred import: keeps rtp_llm.models importable without pulling the
        # models_py CUDA module stack (same pattern as _create_python_model).
        from rtp_llm.models_py.model_desc.qwen3_dflash import DSparkDraftParams

        config = super()._create_config(ckpt_path)  # qk_norm=True + Qwen3 fields
        # DFlash block visibility: feature prefix fully visible + intra-block
        # bidirectional == non-causal attention.
        config.attn_config.is_causal = False
        config_json = get_config_from_path(ckpt_path)
        assert config_json is not None, f"config.json missing under {ckpt_path}"
        config.dspark_config = DSparkDraftParams.from_ckpt_config(config_json)
        return config

    @staticmethod
    def get_weight_cls():
        return Qwen3DFlashWeight

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.qwen3_dflash import Qwen3DFlashModel

        self.py_model = Qwen3DFlashModel(
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


class Qwen3DSpark(Qwen3DFlash):
    """DFlash backbone + Markov head; inherits the DFlash config wiring."""

    @staticmethod
    def get_weight_cls():
        return Qwen3DSparkWeight

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.qwen3_dspark import Qwen3DSparkModel

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


register_model("qwen_3_dflash", Qwen3DFlash, ["Qwen3DFlashForCausalLM"])
register_model("qwen_3_dspark", Qwen3DSpark, ["Qwen3DSparkForCausalLM"])

"""MiniMax-M3.1 DSpARK checkpoint adapter.

The initial mock checkpoint reuses the bundled MiniMax-M3 MTP sparse block and
adds only a feature projection plus Vanilla Markov weights.  It is a structural
bring-up artifact, not a trained DSpARK checkpoint.
"""

import json
import os
from typing import Any

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.minimax_m3 import add_unit_offset
from rtp_llm.models.minimax_m3_mtp import MiniMaxM3MTP, MiniMaxM3MTPWeight
from rtp_llm.ops import KvCacheDataType, SpeculativeType
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, transpose


class MiniMaxM31DSparkWeight(MiniMaxM3MTPWeight):
    def _get_weight_info(self):
        if self._num_layers != 1:
            raise ValueError(
                "MiniMax-M3.1 DSpARK mock checkpoint exposes one physical sparse layer"
            )
        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo(self.prefix + "model.embed_tokens.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo(self.prefix + "lm_head.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo(self._mtp_root + "final_layernorm.weight", identity)],
                add_unit_offset,
            ),
            AtomicWeight(
                W.dspark_fc_w,
                [CkptWeightInfo("dspark.fc.weight", identity)],
                transpose,
                disable_quantization=True,
            ),
            AtomicWeight(
                W.dspark_markov_w1,
                [CkptWeightInfo("dspark.markov_w1.weight", identity)],
                identity,
                disable_quantization=True,
            ),
            AtomicWeight(
                W.dspark_markov_w2,
                [CkptWeightInfo("dspark.markov_w2.weight", identity)],
                identity,
                disable_quantization=True,
            ),
        ]
        return ModelWeightInfo(
            layer_weights=[self._get_hf_layer_weight_info(0)], weights=weights
        )


class MiniMaxM31DSpark(MiniMaxM3MTP):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = super()._create_config(ckpt_path)
        with open(os.path.join(ckpt_path, "config.json")) as reader:
            raw = json.load(reader)
        dspark = raw.get("dspark_config", {})
        config.model_type = "minimax_m31_dspark"
        config.dspark_noise_token_id = int(dspark["mask_token_id"])
        config.dspark_target_layer_ids = [
            int(layer_id) for layer_id in dspark["aux_hidden_state_layer_ids"]
        ]
        config.dspark_markov_rank = int(dspark["markov_rank"])
        config.dspark_sample_from_anchor = bool(dspark.get("sample_from_anchor", True))
        config.input_vocab_size = int(dspark.get("input_vocab_size", config.vocab_size))
        config.use_opaque_kv_cache_store = True
        return config

    @classmethod
    def configure_speculative_model(
        cls, sp_config, target_config: ModelConfig, draft_config: ModelConfig
    ) -> None:
        if sp_config.type != SpeculativeType.DSPARK:
            raise ValueError("MiniMax-M3.1 DSpARK requires SP_TYPE=dspark")
        if int(sp_config.gen_num_per_cycle) <= 0:
            raise ValueError("MiniMax-M3.1 DSpARK requires a positive proposal width")
        if target_config.hidden_size != draft_config.hidden_size:
            raise ValueError("MiniMax-M3.1 DSpARK target/draft hidden size mismatch")
        if target_config.vocab_size != draft_config.vocab_size:
            raise ValueError("MiniMax-M3.1 DSpARK target/draft vocabulary mismatch")

    @staticmethod
    def _validate_kv_cache_dtype(model_config: ModelConfig) -> None:
        if model_config.attn_config.kv_cache_dtype not in (
            KvCacheDataType.BASE,
            KvCacheDataType.FP8,
        ):
            raise ValueError("MiniMax-M3.1 DSpARK mock supports BF16 or FP8 draft KV")

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.minimax_m31_dspark import (
            MiniMaxM31DSparkModel,
        )

        self._validate_kv_cache_dtype(self.model_config)
        self.py_model = MiniMaxM31DSparkModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model

    @staticmethod
    def get_weight_cls() -> type[MiniMaxM31DSparkWeight]:
        return MiniMaxM31DSparkWeight


register_model(
    "minimax_m31_dspark",
    MiniMaxM31DSpark,
    ["MiniMaxM31DSpark"],
)

__all__ = [
    "MiniMaxM31DSpark",
    "MiniMaxM31DSparkWeight",
]

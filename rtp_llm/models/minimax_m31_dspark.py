"""Released five-layer MiniMax-M3.1 DSpARK checkpoint declarations."""

import json
import logging
from pathlib import Path

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.weight_module import AtomicWeight, CustomAtomicWeight
from rtp_llm.models.minimax_m3 import (
    MiniMaxM3,
    MiniMaxM3Weight,
    _get_target_embedding,
    _get_target_lm_head,
)
from rtp_llm.models.minimax_m31_dspark_checkpoint import PREFIX, inspect_checkpoint
from rtp_llm.ops import SpeculativeType
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, transpose

DSPARK_HIDDEN_NORM = "dspark_hidden_norm.raw_weight"
DSPARK_FINAL_NORM = "dspark_final_norm.raw_weight"
DSPARK_CONFIDENCE_WEIGHT = W.dspark_confidence_w
DSPARK_CONFIDENCE_BIAS = W.dspark_confidence_b


class TargetSharedDSparkWeight(AtomicWeight):
    """Return an already-sharded target tensor without draft file reads."""

    def __init__(self, name):
        if name not in (W.embedding, W.lm_head):
            raise ValueError(f"unsupported shared DSpARK weight {name}")
        super().__init__(name, [], identity, disable_quantization=True)

    def load(self, tensor_source, layer_id, device, load_config):
        del tensor_source, layer_id, load_config
        getter = (
            _get_target_embedding if self.name == W.embedding else _get_target_lm_head
        )
        return {self.name: getter(device)}


class MiniMaxM31DSparkWeight(MiniMaxM3Weight):
    """Map draft namespaces, preserving raw Gemma weights for FP32 scaling."""

    def _process_meta(self, meta_dict, weight_keys):
        super()._process_meta(meta_dict, weight_keys)
        if self._num_layers != 5 or not any(
            key.startswith(PREFIX) for key in weight_keys
        ):
            raise ValueError(
                "expected released five-layer MiniMax-M3.1 DSpARK checkpoint"
            )
        if self._sparse_layer_set or self.moe_layer_index_:
            raise ValueError(
                "released DSpARK uses SWA/dense layers, not sparse MSA/MoE"
            )

    def _get_hf_layer_weight_info(self, layer_id):
        if not 0 <= layer_id < 5:
            raise ValueError(f"invalid DSpARK layer {layer_id}")
        modules = super()._get_hf_layer_weight_info(layer_id)
        for module in modules:
            for component in module.get_components():
                if component.name in (
                    W.pre_ln_gamma,
                    W.post_ln_gamma,
                    W.q_ln_gamma,
                    W.k_ln_gamma,
                ):
                    # The draft kernel adds one after converting to FP32.
                    # Baking the offset here rounds its scale to BF16 early.
                    component.process_fun = identity
                for weight in getattr(component, "weights", ()) or ():
                    weight.name = weight.name.replace(
                        self.prefix + "model.layers.{i}.",
                        PREFIX + "layers.{i}.decoder_layer.",
                    )
        return modules

    def _get_weight_info(self):
        info = super()._get_weight_info()
        # Embedding, lm_head and target norm are absent from the draft shard.
        info.weights = [
            TargetSharedDSparkWeight(W.embedding),
            TargetSharedDSparkWeight(W.lm_head),
        ]
        for name, suffix, transform in (
            (W.dspark_fc_w, "fc.weight", transpose),
            (W.dspark_markov_w1, "markov_head.markov_w1.weight", identity),
            (W.dspark_markov_w2, "markov_head.markov_w2.weight", identity),
            (DSPARK_HIDDEN_NORM, "hidden_norm.weight", identity),
            (DSPARK_FINAL_NORM, "final_norm.weight", identity),
            (DSPARK_CONFIDENCE_WEIGHT, "confidence_head.proj.weight", identity),
            (DSPARK_CONFIDENCE_BIAS, "confidence_head.proj.bias", identity),
        ):
            info.weights.append(
                CustomAtomicWeight(
                    name,
                    [CkptWeightInfo(PREFIX + suffix, identity)],
                    transform,
                    disable_quantization=True,
                )
            )
        return info


class MiniMaxM31DSpark(MiniMaxM3):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        report = inspect_checkpoint(ckpt_path)
        raw = json.loads((Path(ckpt_path) / "config.json").read_text())
        text = raw["text_config"]
        config = super()._create_config(ckpt_path)
        config.model_type = "minimax_m31_dspark"
        config.dspark_noise_token_id = text["dspark_noise_token_id"]
        config.dspark_target_layer_ids = report["target_layer_ids"]
        config.dspark_markov_rank = text["dspark_markov_rank"]
        config.dspark_checkpoint_metadata = report
        config.prepacked_nvfp4_moe = False
        config.expert_num = 0
        config.moe_k = 0
        config.moe_style = 0
        config.is_mtp = True
        # One physical draft instance contains all five transformer layers.
        # This is NOT the layer count: CacheConfigCreator repeats the entire
        # draft cache per physical module, independently of proposal width.
        config.physical_mtp_module_num = 1
        return config

    @classmethod
    def configure_speculative_model(cls, sp_config, target_config, draft_config):
        if sp_config.type != SpeculativeType.DSPARK:
            raise ValueError("MiniMax-M3.1 DSpARK requires SP_TYPE=dspark")
        if target_config.hidden_size != draft_config.hidden_size:
            raise ValueError("MiniMax-M3.1 DSpARK target/draft hidden size mismatch")
        if target_config.vocab_size != draft_config.vocab_size:
            raise ValueError("MiniMax-M3.1 DSpARK target/draft vocabulary mismatch")
        if max(draft_config.dspark_target_layer_ids) >= target_config.num_layers:
            raise ValueError("MiniMax-M3.1 DSpARK target hidden layer does not exist")

    def _create_python_model(self):
        import torch

        if torch.version.hip is not None:
            raise RuntimeError(
                "MiniMax-M3.1 DSpARK currently requires the CUDA backend"
            )

        from rtp_llm.models_py.model_desc.minimax_m31_dspark import (
            MiniMaxM31DSparkMath,
            MiniMaxM31DSparkModel,
        )

        config = self.model_config
        metadata = config.dspark_checkpoint_metadata
        block_size = int(metadata["block_size"])
        sliding_window = int(metadata["sliding_window"])
        layer_types = metadata["layer_types"]
        use_gemma_norm = bool(metadata["use_gemma_norm"])
        if (
            int(config.gen_num_per_cycle) != block_size
            or not config.dspark_sample_from_anchor
        ):
            raise ValueError(
                "MiniMax-M3.1 DSpARK requires GEN_NUM_PER_CIRCLE to match "
                "checkpoint dspark_block_size and sp_dspark_sample_from_anchor=true"
            )
        if layer_types != ["sliding_attention"] * 5:
            raise ValueError(
                "MiniMax-M3.1 DSpARK requires five sliding-attention layers"
            )
        if not use_gemma_norm:
            raise ValueError("MiniMax-M3.1 DSpARK requires Gemma RMSNorm semantics")
        logging.info(
            "M31 DSPARK reference math: Gemma RMSNorm; non-causal query; "
            "sliding_window=%d; gamma=%d includes anchor logit; target capture IDs=%s",
            sliding_window,
            block_size,
            config.dspark_target_layer_ids,
        )
        self.py_model = MiniMaxM31DSparkModel(
            config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
            math_contract=MiniMaxM31DSparkMath(
                hidden_norm_gemma=True,
                final_norm_gemma=True,
                # Training demo MiniMaxDSparkAttention is ENCODER_ONLY:
                # query tokens can attend later query tokens within the SWA.
                causal_query=False,
                window_left=sliding_window - 1,
            ),
        )

    @staticmethod
    def get_weight_cls():
        return MiniMaxM31DSparkWeight


register_model("minimax_m31_dspark", MiniMaxM31DSpark, ["DSparkMiniMaxDraftModel"])

__all__ = ["MiniMaxM31DSpark", "MiniMaxM31DSparkWeight"]

import functools
import json
import os
from typing import List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.deepseek_v2 import DeepSeekV2, DeepSeekV2Weight
from rtp_llm.utils.dsa_indexing import dsa_layer_has_indexer
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, transpose, zeros


class Glm5Weight(DeepSeekV2Weight):
    def _get_hf_layer_weight_info(self, layer_id):
        weights = super()._get_hf_layer_weight_info(layer_id)
        if dsa_layer_has_indexer(self.model_config, layer_id):
            return weights
        indexer_weights = {
            W.mla_indexer_qb_w,
            W.mla_indexer_k_w,
            W.mla_indexer_k_norm_w,
            W.mla_indexer_k_norm_b,
            W.mla_indexer_weights_proj_w,
        }
        return [weight for weight in weights if weight.name not in indexer_weights]


class Glm5(DeepSeekV2):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        config_path = os.path.join(ckpt_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as reader:
                architecture = json.load(reader)
            config.swiglu_limit = float(architecture.get("swiglu_limit", 0.0))
            config.index_topk_freq = int(architecture.get("index_topk_freq", 1))
            config.index_skip_topk_offset = architecture.get("index_skip_topk_offset")
            config.index_topk_pattern = architecture.get("index_topk_pattern")
            config.indexer_types = architecture.get("indexer_types")
        return config

    @classmethod
    def _post_build_model_config(cls, model_config: ModelConfig) -> None:
        if model_config.kv_cache_spec_descs:
            return
        super()._post_build_model_config(model_config)
        if not (
            model_config.attn_config.is_sparse and model_config.attn_config.use_mla
        ):
            return
        if model_config.num_layers and not dsa_layer_has_indexer(model_config, 0):
            raise ValueError("DSA layer 0 cannot share Indexer KV before a full layer")
        model_config.kv_cache_spec_descs = [
            [
                desc
                for desc in layer
                if desc.tag != "indexer_kv" or dsa_layer_has_indexer(model_config, i)
            ]
            for i, layer in enumerate(model_config.kv_cache_spec_descs)
        ]

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.glm5 import Glm5MoeModel

        self.py_model = Glm5MoeModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
        )

    @staticmethod
    def get_weight_cls():
        return Glm5Weight


class Glm5MtpWeight(Glm5Weight):
    """Weight loader for GLM-5 MTP that supports two checkpoint layouts.

    Layout A (legacy, extracted MTP-only checkpoint):
        config.json reports num_hidden_layers=1, MTP weights live at
        model.layers.0.* with embedding/lm_head also under model.layers.0.*.

    Layout B (full GLM-5 checkpoint, sglang/vllm style):
        config.json reports num_hidden_layers=N (e.g. 78) and
        num_nextn_predict_layers>=1, MTP weights at model.layers.{N}.*,
        embedding at model.embed_tokens.weight, lm_head at lm_head.weight.

    The mode is selected via model_config.mtp_layer_offset, set by
    Glm5Mtp._create_config(). offset==0 => Layout A; offset>0 => Layout B.
    """

    def _process_meta(self, meta_dict, weight_keys):
        # The MTP layer index in the source checkpoint (78 for Layout B, 0 for
        # Layout A). DeepSeekV2Weight._process_meta only inspects layers
        # [0, _num_layers) which would miss the MTP layer in Layout B.
        mtp_layer_idx = int(getattr(self.model_config, "mtp_layer_offset", 0) or 0)
        if f"model.layers.{mtp_layer_idx}.self_attn.q_a_proj.weight" in weight_keys:
            self.q_use_lora = True
        if (
            f"model.layers.{mtp_layer_idx}.mlp.gate.e_score_correction_bias"
            in weight_keys
        ):
            self.has_e_score_correction_bias = True

    def _get_weight_info(self):
        mtp_layer_idx = int(getattr(self.model_config, "mtp_layer_offset", 0) or 0)

        if mtp_layer_idx == 0:
            embedding_ckpt = "model.layers.0.embed_tokens.weight"
            lm_head_ckpt = "model.layers.0.shared_head.head.weight"
        else:
            embedding_ckpt = "model.embed_tokens.weight"
            lm_head_ckpt = "lm_head.weight"

        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo(embedding_ckpt, identity)],
                identity,
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo(lm_head_ckpt, identity)],
                identity,
            ),
            AtomicWeight(
                W.multi_tokens_predict_final_ln_gamma,
                [
                    CkptWeightInfo(
                        f"model.layers.{mtp_layer_idx}.shared_head.norm.weight",
                        identity,
                    )
                ],
                identity,
            ),
            AtomicWeight(
                W.multi_tokens_predict_final_ln_beta,
                [],
                functools.partial(zeros, shape=[self._hidden_size]),
            ),
            AtomicWeight(
                W.multi_tokens_predict_enorm,
                [
                    CkptWeightInfo(
                        f"model.layers.{mtp_layer_idx}.enorm.weight", identity
                    )
                ],
                identity,
            ),
            AtomicWeight(
                W.multi_tokens_predict_hnorm,
                [
                    CkptWeightInfo(
                        f"model.layers.{mtp_layer_idx}.hnorm.weight", identity
                    )
                ],
                identity,
            ),
            AtomicWeight(
                W.multi_tokens_predict_eh_proj,
                [
                    CkptWeightInfo(
                        f"model.layers.{mtp_layer_idx}.eh_proj.weight", identity
                    )
                ],
                transpose,
            ),
        ]
        assert self._num_layers == 1
        layer_weights: List[List[WeightModule]] = []
        for layer in range(self._num_layers):
            per_layer = self._get_hf_layer_weight_info(layer)
            if mtp_layer_idx != 0:
                # Templates contain "{i}" placeholders that the framework will
                # later substitute with layer_id (= 0 here). Pre-substitute
                # them with the real MTP layer index so the loader looks at
                # model.layers.{mtp_layer_idx}.* in the full checkpoint.
                self._remap_layer_ckpt_names(per_layer, mtp_layer_idx)
            layer_weights.append(per_layer)

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)

    @staticmethod
    def _remap_layer_ckpt_names(layer_weights, target_layer_idx: int) -> None:
        target = str(target_layer_idx)
        for wm in layer_weights:
            for component in wm.get_components():
                ckpt_list = getattr(component, "weights", None)
                if not ckpt_list:
                    continue
                for ckpt in ckpt_list:
                    if hasattr(ckpt, "name") and "{i}" in ckpt.name:
                        ckpt.name = ckpt.name.replace("{i}", target)


class Glm5Mtp(Glm5):

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        num_main_layers = config.num_layers
        num_mtp_layers = 1
        config_path = os.path.join(ckpt_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as reader:
                config_json = json.loads(reader.read())
            num_main_layers = int(config_json.get("num_hidden_layers", num_main_layers))
            num_mtp_layers = int(config_json.get("num_nextn_predict_layers", 1)) or 1
        # Layout A: standalone extracted MTP checkpoint (single layer only).
        # Layout B: full checkpoint where MTP layer is at index num_hidden_layers.
        if num_main_layers <= 1:
            config.mtp_layer_offset = 0
        else:
            config.mtp_layer_offset = num_main_layers
        config.num_layers = num_mtp_layers
        config.moe_layer_index = [i for i in range(config.num_layers)]
        config.is_mtp = True
        # Keep DSA enabled on the MTP layer to match HF ckpt (layer 78 ships
        # indexer weights) and to match sglang/vllm (both build the indexer
        # gated on ``hasattr(config, "index_topk")`` / ``is_deepseek_dsa(config)``
        # independent of ``is_nextn``). Disabling it here drops draft quality
        # (acceptance ~2.0/3 vs expected ~2.7/3) because the draft attention
        # sees full KV instead of the trained top-2048 sparse pattern.
        return config

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.generic_moe_mtp import GenericMoeMTPModel

        self.py_model = GenericMoeMTPModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
        )

    @staticmethod
    def get_weight_cls():
        return Glm5MtpWeight


register_model("glm_5", Glm5, ["GlmMoeDsaForCausalLM"])
register_model("glm_5_mtp", Glm5Mtp, ["GlmMoeDsaMtpForCausalLM"])

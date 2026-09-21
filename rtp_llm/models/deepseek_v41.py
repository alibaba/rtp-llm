"""DeepSeek V4.1 Flash configuration and checkpoint weight descriptors."""

import json
import os

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight, MMAtomicWeight
from rtp_llm.models.deepseek_v4 import (
    DeepSeekV4,
    DeepSeekV4DSpark,
    DeepSeekV4DSparkWeight,
    DeepSeekV4Weight,
)
from rtp_llm.ops import VitSeparation
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, sp_id


class DeepSeekV41Weight(DeepSeekV4Weight):
    def get_weight_info(self):
        # V41 declares its descriptors directly, without a BaseVitWeights module.
        if self.vit_separation == VitSeparation.VIT_SEPARATION_ROLE:
            return ModelWeightInfo(self._build_vision_weights(), [])
        return super().get_weight_info()

    def _build_vision_weights(self):
        """Declare the checkpoint's vision/aligner/delimiter tensors.

        Names mirror ``DeepSeekV41VisionEmbedding.state_dict`` under the
        ``v41.`` namespace; RMSNorm tensors keep their checkpoint FP32 dtype.
        """
        vision = self.model_config.deepseek_v41_config["vision_config"]
        weights = []

        def add(name, dtype=torch.bfloat16):
            # MMAtomicWeight + sp_id: the ViT/aligner stack runs replicated on
            # every TP rank (encode_image computes the full image features and
            # splices rank-local rows), so each rank keeps the full tensor.
            # Plain AtomicWeight falls through to W.gpt_style_tp_strategy[name]
            # and KeyError under TP>1.
            weights.append(
                MMAtomicWeight(
                    "v41." + name,
                    [CkptWeightInfo(name, identity)],
                    identity,
                    data_type=dtype,
                    split_func=sp_id,
                )
            )

        add("image_start")
        add("image_newline")
        add("image_end")
        add("vision.patch_embed.proj.weight")
        add("vision.patch_embed.proj.bias")
        add("vision.norm.weight", torch.float32)
        for layer in range(int(vision["num_hidden_layers"])):
            prefix = f"vision.blocks.{layer}."
            add(prefix + "norm1.weight", torch.float32)
            add(prefix + "norm2.weight", torch.float32)
            for name in ("attn.wqkv", "attn.wo"):
                add(prefix + name + ".weight")
                add(prefix + name + ".bias")
            add(prefix + "mlp.w1.weight")
            add(prefix + "mlp.w2.weight")
        for name in ("w1", "w2"):
            add(f"aligner.{name}.weight")
            add(f"aligner.{name}.bias")
        return weights

    def _get_hf_layer_weight_info(self, layer_id):
        # Reuse V4's common projections/MoE/mHC, whose checkpoint names match.
        # Ratios 1/2 deliberately do not enter V4's 4/128 compressor builders.
        weights = super()._get_hf_layer_weight_info(layer_id)
        config = self.model_config.deepseek_v41_config
        if layer_id in config["kv_source_layer_ids"]:
            compressor = [
                (W.v4_compressor_wkv, "wkv.weight"),
                (W.v4_compressor_norm, "norm.weight"),
            ]
            if self._compress_ratio(layer_id) == 2:
                compressor.append((W.v4_compressor_wgate, "wgate.weight"))
            for name, suffix in compressor:
                weights.append(
                    AtomicWeight(
                        name,
                        [
                            CkptWeightInfo(
                                self._key(f"attn.compressor.{suffix}"), identity
                            )
                        ],
                        identity,
                        data_type=torch.bfloat16,
                    )
                )
            for name, suffix in (
                (W.v41_indexer_wk, "wk.weight"),
                (W.v41_indexer_k_norm, "k_norm.weight"),
            ):
                weights.append(
                    AtomicWeight(
                        name,
                        [CkptWeightInfo(self._key(f"attn.indexer.{suffix}"), identity)],
                        identity,
                        data_type=torch.bfloat16,
                    )
                )
        if layer_id in config["index_source_layer_ids"]:
            weights += self._build_indexer(layer_id)
        # Engram tables and projection are loaded by the host-memory Engram
        # module. Declaring the 196 GB tables here would copy them to every GPU.
        return weights

    def _get_weight_info(self):
        info = super()._get_weight_info()
        obsolete_head = {W.v4_hc_head_base, W.v4_hc_head_fn, W.v4_hc_head_scale}
        info.weights = [w for w in info.weights if w.name not in obsolete_head]
        if self.vit_separation != VitSeparation.VIT_SEPARATION_REMOTE:
            info.weights.extend(self._build_vision_weights())
        return info


class DeepSeekV41(DeepSeekV4):
    def _as_multimodal_model(self):
        return self if type(self) is DeepSeekV41 else None

    @classmethod
    def _create_config(cls, ckpt_path):
        with open(os.path.join(ckpt_path, "config.json")) as reader:
            raw = json.load(reader)
        text = {**raw, **raw.get("text_config", {})}
        text.pop("text_config", None)
        text["checkpoint_path"] = ckpt_path
        config = ModelConfig()
        config.norm_type = "rmsnorm"
        config.has_post_decoder_layernorm = True
        config.activation_type = "SiGLU"
        DeepSeekV4._from_hf(config, ckpt_path, config_json=text)
        config.is_deepseek_v41 = True
        config.mm_model_config.is_multimodal = True
        config.deepseek_v41_config = text
        config.max_seq_len = int(text["max_position_embeddings"])
        config.config_dtype = text.get("dtype", text.get("torch_dtype", "bfloat16"))
        config.attn_config.v41_kv_source_layer_ids = list(text["kv_source_layer_ids"])
        # V4.1 reference scales attention by head_dim**-0.5; YaRN is in RoPE.
        config.attn_config.softmax_extra_scale = 1.0
        config.special_tokens.bos_token_id = int(text.get("bos_token_id", 0))
        config.special_tokens.eos_token_id = int(text.get("eos_token_id", 1))
        return config

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model

        self.py_model = DeepSeekV41Model(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )

    def init_multimodal(self, mm_model_config, vit_config, device):
        # The framework's weight loader owns the vision tensors; bind them
        # after loading in load_mm_weight. Subclasses (the DSpark draft)
        # never carry the ViT part: their loader passes no vit_config.
        self.mm_part = None

    def _may_init_multimodal(self):
        if type(self) is not DeepSeekV41:
            return
        super()._may_init_multimodal()

    def load_mm_weight(self, model_config, ctype, tp_size, tp_rank, device):
        from rtp_llm.config.dsv41_config import V41Config
        from rtp_llm.models.multimodal.deepseek_v41_vision import (
            DeepSeekV41VisionEmbedding,
        )

        self.mm_part = DeepSeekV41VisionEmbedding.from_model_weights(
            V41Config.from_path(model_config.ckpt_path), self.weight.global_weights
        )

    @staticmethod
    def get_weight_cls():
        return DeepSeekV41Weight


class DeepSeekV41DSparkWeight(DeepSeekV4DSparkWeight):
    def _get_weight_info(self):
        info = super()._get_weight_info()
        obsolete_head = {W.v4_hc_head_base, W.v4_hc_head_fn, W.v4_hc_head_scale}
        info.weights = [w for w in info.weights if w.name not in obsolete_head]
        # The new checkpoint renamed the Markov embedding and projection.
        for weight in info.weights:
            if weight.name == W.v4_dspark_markov_w1:
                weight.weights = [
                    CkptWeightInfo(
                        f"mtp.{self._num_layers - 1}.markov_head.embed.weight", identity
                    )
                ]
            elif weight.name == W.v4_dspark_markov_w2:
                weight.weights = [
                    CkptWeightInfo(
                        f"mtp.{self._num_layers - 1}.markov_head.head.weight", identity
                    )
                ]
        return info


class DeepSeekV41DSpark(DeepSeekV41, DeepSeekV4DSpark):
    @classmethod
    def _create_config(cls, ckpt_path):
        config = DeepSeekV41._create_config(ckpt_path)
        config.mm_model_config.is_multimodal = False
        text = dict(config.deepseek_v41_config)
        layers = int(text["num_nextn_predict_layers"])
        ratios = text["compress_ratios"][config.num_layers :]
        if len(ratios) != layers or any(ratios):
            raise ValueError("V4.1 DSpark requires trailing SWA-only draft layers")
        config.num_layers = layers
        config.attn_config.layer_compress_ratios = [0] * layers
        config.attn_config.v41_kv_source_layer_ids = []
        config.expert_num = int(text["dspark_n_routed_experts"])
        config.moe_k = int(text["dspark_num_experts_per_tok"])
        config.moe_layer_index = list(range(layers))
        config.num_hash_layers = 0
        config.is_mtp = True
        text.update(
            kv_source_layer_ids=[], index_source_layer_ids=[], engram_layer_ids=[]
        )
        config.deepseek_v41_config = text
        return config

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.deepseek_v41_dspark_model import (
            DeepSeekV41DSparkModel,
        )

        self.py_model = DeepSeekV41DSparkModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )

    @staticmethod
    def get_weight_cls():
        return DeepSeekV41DSparkWeight


register_model("deepseek_v41", DeepSeekV41, ["DeepseekV41ForCausalLM"])
register_model("deepseek_v41_dspark", DeepSeekV41DSpark)


import os
import json
import functools
from typing import List, Any, Tuple, Dict, Union
from transformers import AutoTokenizer

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.models.qwen_vl import QWen_VL
from maga_transformer.models.base_model import BaseModel, MultimodalInput
from maga_transformer.models.multimodal.multimodal_mixin import MultiModalMixin
from maga_transformer.models.qwen2_vl.qwen2_vl_vit import Qwen2VLImageEmbedding
from maga_transformer.model_factory_register import register_model

from maga_transformer.models.multimodal.multimodal_mixin import BaseVitWeights, BaseMultiModalWeightInfo

from maga_transformer.utils.model_weight import (W, CkptWeightInfo, identity, zeros, transpose, merge_qkv_b, merge_qkv_hf, transpose_pad)
from maga_transformer.model_loader.weight_module import WeightModule, AtomicWeight
from maga_transformer.model_loader.ffn_weight import FfnAtomicWeight, FfnConfig, FfnWeight
from maga_transformer.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from maga_transformer.model_loader.model_weight_info import ModelWeightInfo, ModelDeployWeightInfo

class QwenVL2VitWeight(BaseVitWeights):
    def _set_weight_prefix(self):
        self._ckpt_prefix = "visual."
        self._ft_prefix = "self.mm_part.visual."

class QWen2VLWeightInfo(ModelDeployWeightInfo, BaseMultiModalWeightInfo):
    def __init__(self, config, tp_size, tp_rank):
        ModelDeployWeightInfo.__init__(self, config, tp_size, tp_rank)
        BaseMultiModalWeightInfo.__init__(self, config)

    def _get_weight_info(self):
        weights = self._get_hf_weight_info()
        weights = self._get_vit_info(weights)
        return weights

    def _get_hf_weight_info(self):
        inter_padding_size = self._inter_padding_size
        weights = [
            AtomicWeight(W.embedding, [CkptWeightInfo('model.embed_tokens.weight', identity)], identity),
            AtomicWeight(W.lm_head, [CkptWeightInfo('lm_head.weight', identity)], identity),
            AtomicWeight(W.final_ln_gamma, [CkptWeightInfo('model.norm.weight', identity)], identity),
            AtomicWeight(W.final_ln_beta, [], functools.partial(zeros, shape=[self._hidden_size])),
        ]

        layer_weights: List[List[WeightModule]] = []
        for layer in range(self._num_layers):
            w = self._get_hf_layer_weight_info(layer)
            layer_weights.append(w)

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)

    def _get_hf_layer_weight_info(self, layer_id):
        inter_padding_size = self._layer_inter_padding_size[layer_id] if self._layer_inter_padding_size else self._inter_padding_size
        attn_config: AttnConfig=self.attn_config
        ffn_config: FfnConfig=self.ffn_config
        layer_weights = [
            AtomicWeight(W.pre_ln_gamma, [CkptWeightInfo('model.layers.{i}.input_layernorm.weight', identity)], identity),
            AttnAtomicWeight(W.attn_qkv_w, [CkptWeightInfo('model.layers.{i}.self_attn.q_proj.weight', identity),
                                      CkptWeightInfo('model.layers.{i}.self_attn.k_proj.weight', identity),
                                      CkptWeightInfo('model.layers.{i}.self_attn.v_proj.weight', identity)],
                                      functools.partial(merge_qkv_hf), config=attn_config),
            AttnAtomicWeight(W.attn_qkv_b, [CkptWeightInfo('model.layers.{i}.self_attn.q_proj.bias', identity),
                                      CkptWeightInfo('model.layers.{i}.self_attn.k_proj.bias', identity),
                                      CkptWeightInfo('model.layers.{i}.self_attn.v_proj.bias', identity)],
                                      functools.partial(merge_qkv_b), config=attn_config),
            AttnAtomicWeight(W.attn_o_w, [CkptWeightInfo('model.layers.{i}.self_attn.o_proj.weight', identity)], transpose, config=attn_config),
            FfnWeight(sub_weights=[
                FfnAtomicWeight(W.ffn_w1, [CkptWeightInfo('model.layers.{i}.mlp.gate_proj.weight', identity)], 
                                functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=0), config=ffn_config),
                FfnAtomicWeight(W.ffn_w3, [CkptWeightInfo('model.layers.{i}.mlp.up_proj.weight', identity)], 
                                functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=0), config=ffn_config),
                FfnAtomicWeight(W.ffn_w2, [CkptWeightInfo('model.layers.{i}.mlp.down_proj.weight', identity)], 
                                functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=1), config=ffn_config)
            ], config=ffn_config),
            AtomicWeight(W.post_ln_gamma, [CkptWeightInfo('model.layers.{i}.post_attention_layernorm.weight', identity)], identity)
        ]
        return layer_weights


class QWen2_VL(QWen_VL, MultiModalMixin):
    @staticmethod
    def multimodal_modify_prompt_plugin(prompt: Union[List[Dict[str, Any]], str], images: List[str],
                                        img_token: str, **kwargs: Any) -> Tuple[str, List[MultimodalInput]]:
        return MultiModalMixin.multimodal_modify_prompt_plugin(prompt, images, img_token, **kwargs)

    def _init_multimodal(self, config: GptInitModelParameters):
        self.mm_part = Qwen2VLImageEmbedding(config)
        config.mm_related_params.vit_weights = QwenVL2VitWeight({"vit": self.mm_part.visual})

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = GptInitModelParameters(
            head_num=0,
            size_per_head=0,
            layer_num=0,
            max_seq_len=0,
            vocab_size=0
        )

        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path) as reader:
            content = reader.read()
            config_json = json.loads(content)

        QWen2_VL._from_hf(config, config_json)
        QWen2_VL._load_vit_param(config, config_json)
        config.mm_related_params.config["ckpt_path"] = ckpt_path

        return config

    @staticmethod
    def _load_vit_param(config: GptInitModelParameters, config_json: Dict[str, Any]):
        config.mm_related_params.config = config_json["vision_config"]
        config.mm_related_params.special_tokens.update({'default_mm_token': '<img/>'})
        config.mm_sep_tokens = [[config_json['vision_start_token_id'], config_json['vision_end_token_id']]]

    @staticmethod
    def _from_hf(config: GptInitModelParameters, config_json: Dict[str, Any]):
        config.vocab_size = config_json["vocab_size"]
        config.rotary_embedding_base = config_json["rope_theta"]
        config.max_seq_len = 10240
        config.activation_type = 'SiGLU'
        config.head_num = config_json["num_attention_heads"]
        config.head_num_kv = config_json["num_key_value_heads"]
        config.hidden_size = config_json["hidden_size"]
        config.size_per_head = config_json["hidden_size"] // config.head_num
        config.layer_num = config_json["num_hidden_layers"]
        config.inter_size = config_json["intermediate_size"]
        config.norm_type = 'rmsnorm'
        config.layernorm_eps = config_json["rms_norm_eps"]
        config.has_post_decoder_layernorm=True
        config.special_tokens.bos_token_id = config_json["bos_token_id"]
        config.special_tokens.eos_token_id = config_json["eos_token_id"]
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)

        config.rotary_embedding_style = 7
        config.mrope_section = config_json["rope_scaling"].get("mrope_section", [16, 24, 24])
        config.mm_position_ids_style = 2
        config.position_id_len_factor = len(config.mrope_section)
        config.rotary_embedding_dim = 128

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        return AutoTokenizer.from_pretrained(config.tokenizer_path, trust_remote_code=True)

    @staticmethod
    def get_weight_cls():
        return QWen2VLWeightInfo

    @staticmethod
    def eval_model_size(config: GptInitModelParameters):
        llm_size = BaseModel.eval_model_size(config)

        data_width = 4
        llm_size += QWen2_VL.eval_vit_param_count(config) * data_width
        return llm_size

    @staticmethod
    def eval_vit_param_count(config: GptInitModelParameters):
        vit_config = config.mm_related_params.config
        embed_dim = vit_config.get("embed_dim", 1280)
        hidden_size = vit_config.get("hidden_size", 3584)
        vit_size = vit_config.get("temporal_patch_size", 2) * vit_config.get("spatial_patch_size", 14) ** 2 * vit_config.get("in_chans", 3) * embed_dim
        patch_merger_size = embed_dim * vit_config.get("spatial_merge_size", 2) ** 2
        vit_size += patch_merger_size ** 2 + patch_merger_size * hidden_size + embed_dim
        mlp_hidden_dim = embed_dim * vit_config.get("mlp_ratio", 4)
        vit_size += vit_config.get("depth", 32) * (embed_dim * 2 + embed_dim * mlp_hidden_dim + embed_dim * embed_dim * 4)

        return vit_size

    @staticmethod
    def eval_model_param_count(config: GptInitModelParameters):
        llm_param_count = BaseModel.eval_model_param_count(config)
        llm_param_count += QWen2_VL.eval_vit_param_count(config)

        return llm_param_count

register_model('qwen2_vl', QWen2_VL, ["Qwen2VLForConditionalGeneration"])

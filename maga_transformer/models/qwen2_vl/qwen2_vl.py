
import torch
import os
import json
import functools
from typing import List, Any, Tuple, Dict, Union
from transformers import AutoTokenizer, AutoProcessor

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.models.qwen import QWen
from maga_transformer.models.qwen_vl_weight import QWenVLWeightInfo, QwenVLVitWeight
from maga_transformer.models.qwen_vl import QWen_VL, QWen_VL_ViT
from maga_transformer.models.base_model import BaseModel, MultimodalInput
from maga_transformer.models.multimodal.multimodal_mixin import MultiModalMixin
from maga_transformer.models.qwen2_vl.qwen2_vl_vit import Qwen2VLImageEmbedding
from maga_transformer.model_factory_register import register_model
from maga_transformer.utils.util import to_torch_dtype

from maga_transformer.models.qwen import QWenWeight
from maga_transformer.models.multimodal.multimodal_mixin import BaseVitWeights, BaseMultiModalWeightInfo

from maga_transformer.utils.model_weight import (W, WeightInfo, ModelWeightInfo,
                                                 ModelDeployWeightInfo, CkptWeightInfo, concat_1,
                                                 concat_0, identity, zeros, transpose, merge_qkv_lora_A,
                                                 merge_qkv_lora_B, shift_one, pad, merge_qkv_b, merge_qkv_hf)
from maga_transformer.utils.model_weight import W, WeightInfo, ModelWeightInfo,\
    ModelDeployWeightInfo, CkptWeightInfo, \
    concat_0, concat_1, identity, zeros, transpose, trans_qkv, trans_qkv_b, trans_lora_qkv, transpose_pad, pad, ones

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
        self._get_vit_info(weights)
        return weights
    
    def _get_hf_weight_info(self):
        weights = [
            WeightInfo(W.embedding, [CkptWeightInfo('model.embed_tokens.weight', identity)], identity),
            WeightInfo(W.lm_head, [CkptWeightInfo('lm_head.weight', identity)], identity),
            WeightInfo(W.final_ln_gamma, [CkptWeightInfo('model.norm.weight', identity)], identity),
            WeightInfo(W.final_ln_beta, [], functools.partial(zeros, shape=[self._hidden_size])),
        ]

        layer_weights = [
            WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('model.layers.{i}.input_layernorm.weight', identity)], identity),
            WeightInfo(W.attn_qkv_w, [CkptWeightInfo('model.layers.{i}.self_attn.q_proj.weight', identity),
                                      CkptWeightInfo('model.layers.{i}.self_attn.k_proj.weight', identity),
                                      CkptWeightInfo('model.layers.{i}.self_attn.v_proj.weight', identity)],
                                      functools.partial(merge_qkv_hf)),
            WeightInfo(W.attn_qkv_b, [CkptWeightInfo('model.layers.{i}.self_attn.q_proj.bias', identity),
                                      CkptWeightInfo('model.layers.{i}.self_attn.k_proj.bias', identity),
                                      CkptWeightInfo('model.layers.{i}.self_attn.v_proj.bias', identity)],
                                      functools.partial(merge_qkv_b)),
            WeightInfo(W.attn_o_w, [CkptWeightInfo('model.layers.{i}.self_attn.o_proj.weight', identity)], transpose),
            WeightInfo(W.ffn_w1, [CkptWeightInfo('model.layers.{i}.mlp.gate_proj.weight', identity)], transpose),
            WeightInfo(W.ffn_w3, [CkptWeightInfo('model.layers.{i}.mlp.up_proj.weight', identity)], transpose),
            WeightInfo(W.ffn_w2, [CkptWeightInfo('model.layers.{i}.mlp.down_proj.weight', identity)], transpose),
            WeightInfo(W.post_ln_gamma, [CkptWeightInfo('model.layers.{i}.post_attention_layernorm.weight', identity)], identity)   
        ]

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights,
                               tp_strategy=self._get_gpt_style_tp_strategy())

class QWen2_VL(QWen_VL, MultiModalMixin):
    @staticmethod
    def multimodal_modify_prompt_plugin(prompt: Union[List[Dict[str, Any]], str], images: List[str],
                                        img_token: str, **kwargs: Any) -> Tuple[str, List[MultimodalInput]]:
        return MultiModalMixin.multimodal_modify_prompt_plugin(prompt, images, img_token, **kwargs)

    def init_multimodal(self, config: GptInitModelParameters):
        with torch.device(g_parallel_info.device):
            self.mm_part = Qwen2VLImageEmbedding(config.mm_related_params.config)
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
        
        if config_json.get("rope_scaling"):
            if config_json["rope_scaling"].get("type", "") == "mrope":
                config.rotary_embedding_style = 7
                config.mrope_section = config_json["rope_scaling"].get("mrope_section", [16, 24, 24])
                config.mm_position_ids_style = 2
                config.position_id_len_factor = len(config.mrope_section)
                config.rotary_embedding_dim = 128

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        return AutoProcessor.from_pretrained(config.tokenizer_path, trust_remote_code=True)

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
        embed_dim = vit_config["embed_dim"]
        hidden_size = vit_config["hidden_size"]
        vit_size = vit_config["temporal_patch_size"] * vit_config["spatial_patch_size"] ** 2 * vit_config["in_chans"] * embed_dim
        patch_merger_size = embed_dim * vit_config["spatial_merge_size"] ** 2
        vit_size += patch_merger_size ** 2 + patch_merger_size * hidden_size + embed_dim
        mlp_hidden_dim = embed_dim * vit_config["mlp_ratio"]
        vit_size += vit_config["depth"] * (embed_dim * 2 + embed_dim * mlp_hidden_dim + embed_dim * embed_dim * 4)

        return vit_size

    @staticmethod
    def eval_model_param_count(config: GptInitModelParameters):
        llm_param_count = BaseModel.eval_model_param_count(config)
        llm_param_count += QWen2_VL.eval_vit_param_count(config)

        return llm_param_count

register_model('qwen2_vl', QWen2_VL, ["Qwen2VLForConditionalGeneration"])

import os
import logging
import json
import math
import torch

from typing import Any, Dict, List

from transformers import AutoTokenizer
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.models.qwen_v2 import QWenV2
from maga_transformer.models.llama import Llama
from maga_transformer.models.gpt import GPT
from maga_transformer.models.multimodal.multimodal_mixin import MultiModalMixin
from maga_transformer.models.internvl_weight import InternVLVitWeight, InternVLWeightInfo
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.model_factory_register import register_model
from maga_transformer.models.internvl_vit import InternVLImageEmbedding

class InternVLTokenizer:
    def __init__(self, tokenizer_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def encode(self, prompt: str, **kwargs):
        prompt_slices = prompt.split("<image>")
        new_prompt = prompt_slices[0]
        for slice in prompt_slices[1:]:
            new_prompt += "<img></img>" + slice
        return self.tokenizer.encode(new_prompt, **kwargs)

    def decode(self, token_id: List[int], **kwargs):
        return self.tokenizer.decode(token_id, **kwargs)

class InternVL(GPT, MultiModalMixin):
    def __init__(self, config: GptInitModelParameters):
        if g_parallel_info.tp_rank == 0:
            with torch.cuda.device(torch.device(g_parallel_info.device)):
                self.mm_part = InternVLImageEmbedding(config.mm_related_params.config)
            config.mm_related_params.vit_weights = InternVLVitWeight({"vision_model": self.mm_part.vision_model,
                                                                      "mlp1": self.mm_part.mlp1}, True)
            
        GPT.__init__(self, config)
        self.config.mm_sep_tokens = [self.tokenizer.encode("<img>")[0], self.tokenizer.encode("</img>")[0]]

    @staticmethod
    def get_weight_cls():
        return InternVLWeightInfo

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        return InternVLTokenizer(config.tokenizer_path)

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = GptInitModelParameters(
            head_num=0,
            head_num_kv=0,
            size_per_head=0,
            layer_num=0,
            inter_size=0,
            vocab_size=0,
            max_seq_len=0,
            ckpt_path=ckpt_path,
            rotary_embedding_dim=128,
            rotary_embedding_style=1,
            activation_type='SiGLU',
            has_pre_decoder_layernorm=False,
            has_post_decoder_layernorm=True,
            norm_type='rmsnorm'
            )
        
        config_path = os.path.join(ckpt_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path) as reader:
                content = reader.read()
                config_json = json.loads(content)
                llm_config = config_json["llm_config"]
                if llm_config["architectures"][0] == "Qwen2ForCausalLM":
                    InternVL._qwen2_create_config(config, llm_config)
                elif llm_config["architectures"][0] == "InternLM2ForCausalLM" or \
                    llm_config["architectures"][0] == "LlamaForCausalLM":
                    InternVL._internlm2_create_config(config, llm_config)
                else:
                    raise Exception("unknown language model architecture")
                InternVL._init_vit_params(config, config_json)
        else:
            raise Exception("no config.json found")
        config.mm_related_params.special_tokens.update({'default_mm_token': '<image>'})
        return config

    @staticmethod           
    def _qwen2_create_config(config: GptInitModelParameters, config_json: Dict[str, Any]):
        config.special_tokens.bos_token_id = -1
        config.special_tokens.eos_token_id = 151643
        # <|im_start|> and <|im_end|>
        config.special_tokens.stop_words_list = [[151645], [151644]]
        config.special_tokens.system.token_ids = [151644, 8948, 198] # '<|im_start|>system\n'
        config.special_tokens.system.eos_token_ids = [151645, 198] # '<|im_end|>\n'
        config.special_tokens.user.token_ids = [151644, 872, 198] # '<|im_start|>user\n'
        config.special_tokens.user.eos_token_ids = [151645, 198]  # '<|im_end|>\n'
        config.special_tokens.assistant.token_ids = [151644, 77091, 198] # '<|im_start|>assistant\n'
        config.special_tokens.assistant.eos_token_ids = [151645, 198] # '<|im_end|>\n'
        QWenV2._from_config_json(config, config_json)
        assert config.head_num > 0 and config.head_num_kv > 0 and config.size_per_head > 0 and config.layer_num > 0 and config.inter_size > 0, "error config"
    
    @staticmethod
    def _internlm2_create_config(config: GptInitModelParameters, config_json: Dict[str, Any]):
        Llama.from_huggingface(config, config_json)

    @staticmethod
    def _init_vit_params(config: GptInitModelParameters, config_json: Dict[str, Any]):
        config.mm_related_params.config = config_json["vision_config"]
        config.mm_related_params.config["select_layer"] = config_json["select_layer"]
        config.mm_related_params.config["llm_hidden_size"] = config_json["llm_config"]["hidden_size"]
        config.mm_related_params.config["downsample_ratio"] = config_json["downsample_ratio"]
        config.mm_related_params.config["ps_version"] = config_json["ps_version"]

register_model("internvl", InternVL, ["InternVLChatModel"])
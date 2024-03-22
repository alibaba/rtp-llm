import os
from typing import Any, Dict
import torch

from maga_transformer.utils.util import get_config_from_path
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.tokenizer.tokenization_chatglm import ChatGLMTokenizer
from maga_transformer.models.glm_weight import GlmWeightInfo
from maga_transformer.models.gpt import GPT
from maga_transformer.model_factory_register import register_model

class ChatGlm(GPT):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if os.environ.get('REUSE_CACHE', None) == "1":
            raise Exception("chatglm kvcache style not support reuse block cache")

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        return ChatGLMTokenizer.from_pretrained(config.tokenizer_path)

    @staticmethod
    def get_weight_cls():
        return GlmWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config_dict = get_config_from_path(ckpt_path)
        if config_dict is not None:
            config = ChatGlm.from_huggingface(ChatGlm, config_dict)
        else:
            config = ChatGlm.default_config()
        return config

    
    @staticmethod
    def default_config():
        config = GptInitModelParameters(
            head_num=32,
            size_per_head=128,
            inter_size=4 * 4096,
            activation_type='gelu',
            norm_type='alphanorm',
            rotary_embedding_dim=128,
            rotary_embedding_style=2,
            add_bias_linear=True,
            layer_num=28,
            max_seq_len=2048,
            has_post_decoder_layernorm=True,
            vocab_size=130528,
            use_norm_input_residual=True,
            use_norm_attn_out_residual=True)
        config.special_tokens.bos_token_id = 130004
        config.special_tokens.eos_token_id = 130005

        assert config.norm_type == 'alphanorm'
        return config


    @staticmethod
    def from_huggingface(cls, config_json: Dict[str, Any]):
        '''
        "apply_query_key_layer_scaling": true,
        "apply_residual_connection_post_layernorm": false,
        "attention_softmax_in_fp32": true,
        "fp32_residual_connection": false,
        "original_rope": true,
        '''
        config = ChatGlm.default_config()
        config.head_num = config_json.get('num_attention_heads', config.head_num)
        config.hidden_size = config_json.get('hidden_size', 4096)
        if config_json.get('multi_query_attention', False):
            config.head_num_kv = config_json['multi_query_group_num']
        else:
            config.head_num_kv = config.head_num
        config.size_per_head = config.hidden_size // config.head_num
        config.layer_num = config_json.get('num_layers', config.layer_num)
        config.max_seq_len = config_json.get('max_sequence_length', config.max_seq_len)
        config.vocab_size = config_json.get('vocab_size', config.vocab_size)
        config.weights_data_type = config_json.get('torch_dtype', config.weights_data_type)
        config.layernorm_eps = config_json.get('layernorm_epsilon', config.layernorm_eps)
        config.inter_size = config_json.get('inner_hidden_size', config.inter_size)
        config.rotary_embedding_dim = config.size_per_head
        config.special_tokens.bos_token_id = config_json.get('bos_token_id', config.special_tokens.bos_token_id)
        config.special_tokens.eos_token_id = config_json.get('eos_token_id', config.special_tokens.eos_token_id)
        config.src_quantization_bit = config_json.get('quantization_bit', 0)
        return config

    # override
    def create_context_decoder_mask(self, input_lengths: torch.Tensor):
        batch_size = len(input_lengths)
        max_input_length = max(input_lengths)
        context_lengths = [int(x) - 1 for x in input_lengths]
        attention_mask = torch.ones(
            (max_input_length, max_input_length), dtype=torch.bool, device=self.device)\
            .tril().unsqueeze(0)
        # attention_mask = ~attention_mask
        attention_mask = attention_mask.tile(batch_size, 1, 1).to(self.dtype)
        for i, context_length in enumerate(context_lengths):
            attention_mask[i, :, :context_length] = 1
        for b, input_length in enumerate(input_lengths):
            attention_mask[b, input_length:, ...] = 0
        return attention_mask

register_model('chatglm', ChatGlm, [], ["THUDM/chatglm-6b", "THUDM/chatglm-6b-int4", "THUDM/chatglm-6b-int4-qe", "THUDM/chatglm-6b-int8"])
register_model('chat_glm', ChatGlm)

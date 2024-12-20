import os
import json
import torch
import logging
from typing import Any, Dict, List, Optional

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.config.task_type import TaskType
from maga_transformer.models.base_model import BaseModel
from maga_transformer.models.downstream_modules.custom_module import CustomModule
from maga_transformer.models.downstream_modules.classifier.roberta_classifier import RobertaClassifierModule
from maga_transformer.models.downstream_modules.classifier.bert_classifier import BertClassifierModule
from maga_transformer.models.downstream_modules import RobertaRerankerModule, BertRerankerModule
from maga_transformer.models.bert_weight import BertWeightInfo, RobertaWeightInfo
from maga_transformer.model_factory_register import register_model
from transformers import AutoTokenizer, BertTokenizer

class Bert(BaseModel):
    @staticmethod
    def get_weight_cls():
        return BertWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = GptInitModelParameters(
            head_num=0,
            size_per_head=0,
            layer_num=0,
            max_seq_len=0,
            vocab_size=0,
            ckpt_path=ckpt_path,
            activation_type='gelu',
            norm_type='layernorm',
            rotary_embedding_dim=0,
            rotary_embedding_style=0,
            has_positional_encoding=True,
            has_pre_decoder_layernorm=True,
            layernorm_type='post_layernorm',
            is_causal=False
        )
        # hugggingface
        config_path = os.path.join(ckpt_path, 'config.json')
        if not os.path.exists(config_path):
            raise Exception(f"failed to find config json from {ckpt_path}")
        with open(config_path) as reader:
            content = reader.read()
            config_json = json.loads(content)
            cls.from_huggingface(config, config_json)
        return config

    def load_custom_module(self) -> Optional[CustomModule]:
        if self.task_type == TaskType.SEQ_CLASSIFICATION:
            logging.info("using BertClassifierModule as custom module")
            return BertClassifierModule(self.config, self.tokenizer)
        if self.task_type == TaskType.RERANKER:
            logging.info("using BertRerankerModule as custom module")
            return BertRerankerModule(self.config, self.tokenizer)
        return super().load_custom_module()

    @classmethod
    def from_huggingface(cls, config: GptInitModelParameters, config_json: Dict[str, Any]):
        # check position_embedding_type == absolute
        config.head_num = config_json['num_attention_heads']
        # bert has no group attention
        config.head_num_kv = config.head_num
        config.size_per_head = config_json['hidden_size'] // config_json['num_attention_heads']
        config.hidden_size = config_json['hidden_size']
        config.layer_num = config_json['num_hidden_layers']
        config.max_seq_len = config_json.get('max_position_embeddings', 512)
        config.vocab_size = config_json['vocab_size']
        config.type_vocab_size = config_json.get('type_vocab_size', 0)
        config.layernorm_eps = config_json['layer_norm_eps']
        config.inter_size = config_json['intermediate_size']

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        try:
            return AutoTokenizer.from_pretrained(config.tokenizer_path, trust_remote_code=True)
        except:
            logging.warning("failed to load bert tokenizer using AutoTokenizer, try using BertTokenizer instead")
            return BertTokenizer.from_pretrained(config.tokenizer_path)

class Roberta(Bert):
    @staticmethod
    def get_weight_cls():
        return RobertaWeightInfo

    def load_custom_module(self) -> Optional[CustomModule]:
        if self.task_type == TaskType.SEQ_CLASSIFICATION:
            logging.info("using RobertaClassifierModule as custom module")
            return RobertaClassifierModule(self.config, self.tokenizer)
        elif self.task_type == TaskType.RERANKER:
            logging.info("using RobertaRerankerModule as custom module")
            return RobertaRerankerModule(self.config, self.tokenizer)
        return super().load_custom_module()

    @classmethod
    def from_huggingface(cls, config: GptInitModelParameters, config_json: Dict[str, Any]):
        Bert.from_huggingface(config, config_json)
        config.special_tokens.pad_token_id = config_json['pad_token_id']
        config.position_ids_style = 1

    def create_context_position_ids(self, input_lengths: List[int]):
        pad_index = self.config.special_tokens.pad_token_id
        return torch.concat([torch.arange(pad_index + 1, input_length + pad_index + 1, dtype=torch.int32) for input_length in input_lengths], dim=0)

register_model('bert', Bert, ['BertModel', 'BertForMaskedLM', 'BertForSequenceClassification'])
register_model('roberta', Roberta, ['XLMRobertaModel', 'RobertaModel', "XLMRobertaForSequenceClassification"])

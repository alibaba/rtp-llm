import os
import json
import torch

from typing import Any, Dict, List, Type, Optional

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.config.task_type import TaskType
from maga_transformer.models.base_model import BaseModel
from maga_transformer.models.bert import Bert
from maga_transformer.models.downstream_modules.custom_module import CustomModule
from maga_transformer.models.downstream_modules.classifier.roberta_classifier import RobertaClassifierModule
from maga_transformer.models.bert_weight import BertWeightInfo, RobertaWeightInfo
from maga_transformer.models.megatron_bert_weight import MegatronBertWeightInfo
from maga_transformer.model_factory_register import register_model
from transformers import AutoTokenizer

class MegatronBert(Bert):
    @staticmethod
    def get_weight_cls():
        return MegatronBertWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = Bert._create_config(ckpt_path)
        config.has_pre_decoder_layernorm = False
        config.layernorm_type = 'pre_layernorm'
        return config

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        return AutoTokenizer.from_pretrained(config.tokenizer_path, trust_remote_code=True)

register_model('megatron_bert', MegatronBert, ['MegatronBertModel'])
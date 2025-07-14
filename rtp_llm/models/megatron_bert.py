import json
import os
from typing import Any, Dict, List, Optional, Type

import torch
from transformers import AutoTokenizer

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.config.task_type import TaskType
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.bert import Bert
from rtp_llm.models.bert_weight import BertWeightInfo, RobertaWeightInfo
from rtp_llm.models.downstream_modules.classifier.roberta_classifier import (
    RobertaClassifierModule,
)
from rtp_llm.models.downstream_modules.custom_module import CustomModule
from rtp_llm.models.megatron_bert_weight import MegatronBertWeightInfo


class MegatronBert(Bert):
    @staticmethod
    def get_weight_cls():
        return MegatronBertWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = Bert._create_config(ckpt_path)
        config.has_pre_decoder_layernorm = False
        config.layernorm_type = "pre_layernorm"
        return config

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        return AutoTokenizer.from_pretrained(
            config.tokenizer_path, trust_remote_code=True
        )


register_model("megatron_bert", MegatronBert, ["MegatronBertModel"])

import json
import logging
import os
from typing import Any, Dict, List, Optional

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.bert_weight import BertWeightInfo, RobertaWeightInfo
from rtp_llm.models.downstream_modules import BertRerankerModule, RobertaRerankerModule
from rtp_llm.models.downstream_modules.classifier.bert_classifier import (
    BertClassifierModule,
)
from rtp_llm.models.downstream_modules.classifier.roberta_classifier import (
    RobertaClassifierModule,
)
from rtp_llm.models.downstream_modules.custom_module import CustomModule
from rtp_llm.models_py.model_desc.bert import BertModel
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops import TaskType


class Bert(BaseModel):
    @staticmethod
    def get_weight_cls():
        return BertWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        from rtp_llm.model_config_creators.bert import create_bert_config

        config = create_bert_config(ckpt_path)
        return config

    def support_cuda_graph(self) -> bool:
        return True

    def _create_python_model(self) -> Optional[GptModelBase]:
        model_config = self.model_config
        parallelism_config = self.parallelism_config
        quant_config = self.model_config.quant_config
        fmha_config = self.fmha_config
        py_hw_kernel_config = self.hw_kernel_config
        max_generate_batch_size = self.max_generate_batch_size

        self.py_model = BertModel(
            model_config,
            parallelism_config,
            self.weight,
            max_generate_batch_size=max_generate_batch_size,
            quant_config=quant_config,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )

    def _init_custom_module(self) -> Optional[CustomModule]:
        if self.model_config.task_type == TaskType.SEQ_CLASSIFICATION:
            logging.info("using BertClassifierModule as custom module")
            return BertClassifierModule(self.model_config, self.tokenizer)
        if self.model_config.task_type == TaskType.RERANKER:
            logging.info("using BertRerankerModule as custom module")
            return BertRerankerModule(self.model_config, self.tokenizer)
        return super()._init_custom_module()


class Roberta(Bert):
    @staticmethod
    def get_weight_cls():
        return RobertaWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str):
        from rtp_llm.model_config_creators.bert import create_roberta_config

        config = create_roberta_config(ckpt_path)
        return config

    def _init_custom_module(self) -> Optional[CustomModule]:
        logging.info(f"task_type : {self.model_config.task_type}")
        if self.model_config.task_type == TaskType.SEQ_CLASSIFICATION:
            logging.info("using RobertaClassifierModule as custom module")
            return RobertaClassifierModule(self.model_config, self.tokenizer)
        elif self.model_config.task_type == TaskType.RERANKER:
            logging.info("using RobertaRerankerModule as custom module")
            return RobertaRerankerModule(self.model_config, self.tokenizer)
        return super()._init_custom_module()

    def support_cuda_graph(self) -> bool:
        return False


register_model(
    "bert", Bert, ["BertModel", "BertForMaskedLM", "BertForSequenceClassification"]
)
register_model(
    "roberta",
    Roberta,
    ["XLMRobertaModel", "RobertaModel", "XLMRobertaForSequenceClassification"],
)

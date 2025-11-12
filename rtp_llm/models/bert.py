import json
import logging
import os
from typing import Any, Dict, List, Optional

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.ops import TaskType
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


class Bert(BaseModel):
    @staticmethod
    def get_weight_cls():
        return BertWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.activation_type = "gelu"
        config.norm_type = "layernorm"
        config.rotary_embedding_dim_ = 0
        config.rotary_embedding_style_ = 0
        config.has_positional_encoding_ = True
        config.has_pre_decoder_layernorm_ = True
        config.layernorm_type = "post_layernorm"
        config.is_causal_ = False
        # hugggingface
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise Exception(f"failed to find config json from {ckpt_path}")
        with open(config_path) as reader:
            content = reader.read()
            config_json = json.loads(content)
            cls.from_huggingface(config, config_json)
        return config

    def support_cuda_graph(self) -> bool:
        return True

    def _create_python_model(self) -> Optional[GptModelBase]:
        model_config = self.model_config
        parallelism_config = self.engine_config.parallelism_config
        device_resource_config = self.engine_config.device_resource_config
        quant_config = self.model_config.quant_config
        vocab_size = self.model_config.vocab_size
        fmha_config = self.engine_config.fmha_config
        py_hw_kernel_config = self.engine_config.hw_kernel_config
        
        self.py_model = BertModel(
            model_config,
            parallelism_config,
            device_resource_config,
            self.weight,
            quant_config,
            vocab_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
        )

    def _init_custom_module(self) -> Optional[CustomModule]:
        if self.py_model_config.task_type == TaskType.SEQ_CLASSIFICATION:
            logging.info("using BertClassifierModule as custom module")
            return BertClassifierModule(self.model_config, self.tokenizer)
        if self.py_model_config.task_type == TaskType.RERANKER:
            logging.info("using BertRerankerModule as custom module")
            return BertRerankerModule(self.model_config, self.tokenizer)
        return super()._init_custom_module()

    @classmethod
    def from_huggingface(
        cls, config: ModelConfig, config_json: Dict[str, Any]
    ):
        # check position_embedding_type == absolute
        config.head_num_ = config_json["num_attention_heads"]
        # bert has no group attention
        config.head_num_kv_ = config.head_num_
        config.size_per_head_ = (
            config_json["hidden_size"] // config_json["num_attention_heads"]
        )
        config.hidden_size = config_json["hidden_size"]
        config.num_layers = config_json["num_hidden_layers"]
        config.max_seq_len = config_json.get("max_position_embeddings", 512)
        config.vocab_size = config_json["vocab_size"]
        config.type_vocab_size_ = config_json.get("type_vocab_size", 0)
        config.layernorm_eps_ = config_json["layer_norm_eps"]
        config.inter_size = config_json["intermediate_size"]
        config.config_dtype_ = config_json.get("torch_dtype", None)


class Roberta(Bert):
    @staticmethod
    def get_weight_cls():
        return RobertaWeightInfo

    def _init_custom_module(self) -> Optional[CustomModule]:
        logging.info(f"task_type : {self.py_model_config.task_type}")
        if self.py_model_config.task_type == TaskType.SEQ_CLASSIFICATION:
            logging.info("using RobertaClassifierModule as custom module")
            return RobertaClassifierModule(self.model_config, self.tokenizer)
        elif self.py_model_config.task_type == TaskType.RERANKER:
            logging.info("using RobertaRerankerModule as custom module")
            return RobertaRerankerModule(self.model_config, self.tokenizer)
        return super()._init_custom_module()

    def support_cuda_graph(self) -> bool:
        return False

    @classmethod
    def from_huggingface(
        cls, config: ModelConfig, config_json: Dict[str, Any]
    ):
        Bert.from_huggingface(config, config_json)
        config.special_tokens.pad_token_id = config_json["pad_token_id"]
        config.position_ids_style = 1

register_model(
    "bert", Bert, ["BertModel", "BertForMaskedLM", "BertForSequenceClassification"]
)
register_model(
    "roberta",
    Roberta,
    ["XLMRobertaModel", "RobertaModel", "XLMRobertaForSequenceClassification"],
)

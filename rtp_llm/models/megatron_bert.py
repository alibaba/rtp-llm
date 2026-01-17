from transformers import AutoTokenizer

from rtp_llm.model_factory_register import register_model
from rtp_llm.models.bert import Bert
from rtp_llm.models.megatron_bert_weight import MegatronBertWeightInfo


class MegatronBert(Bert):
    @staticmethod
    def get_weight_cls():
        return MegatronBertWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str):
        from rtp_llm.model_config_creators.bert import create_megatron_bert_config

        config = create_megatron_bert_config(ckpt_path)
        return config


register_model("megatron_bert", MegatronBert, ["MegatronBertModel"])

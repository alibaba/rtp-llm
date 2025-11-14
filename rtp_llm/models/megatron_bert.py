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
        config = Bert._create_config(ckpt_path)
        config.has_pre_decoder_layernorm = False
        config.layernorm_type = "pre_layernorm"
        return config


register_model("megatron_bert", MegatronBert, ["MegatronBertModel"])

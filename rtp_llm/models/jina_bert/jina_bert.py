from rtp_llm.model_factory_register import register_model
from rtp_llm.models.bert import Bert
from rtp_llm.models.jina_bert.jina_bert_weight import JinaBertWeightInfo

# jina bert相比于bert有3点区别
"""
1. qk_norm(optional)
2. gated gelu
3. alibi
"""


class JinaBert(Bert):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = Bert._create_config(ckpt_path)
        config.activation_type = "gated-gelu"
        config.use_attention_linear_bias = True
        config.has_positional_encoding = False
        config.qk_norm = True
        return config

    @staticmethod
    def get_weight_cls():
        return JinaBertWeightInfo


register_model("jina_bert_code", JinaBert, [], ["jinaai/jina-bert-v2-qk-post-norm"])

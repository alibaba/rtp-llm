from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.config.task_type import TaskType
from maga_transformer.models.bert import Bert
from maga_transformer.models.downstream_modules.custom_module import CustomModule
from maga_transformer.models.downstream_modules.classifier.roberta_classifier import RobertaClassifierModule
from maga_transformer.models.downstream_modules import RobertaRerankerModule
from maga_transformer.models.jina_bert.jina_bert_weight import JinaBertWeightInfo
from maga_transformer.model_factory_register import register_model
from transformers import AutoTokenizer

# jina bert相比于bert有3点区别
'''
1. qk_norm(optional)
2. gated gelu
3. alibi
'''
class JinaBert(Bert):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = Bert._create_config(ckpt_path)
        config.activation_type = 'gated-gelu'
        config.use_attention_linear_bias = True
        config.has_positional_encoding = False
        config.qk_norm = True
        return config

    @staticmethod
    def get_weight_cls():
        return JinaBertWeightInfo
    
register_model('jina_bert_code', JinaBert, [], ["jinaai/jina-bert-v2-qk-post-norm"])    
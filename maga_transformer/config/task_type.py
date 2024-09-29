import os
from enum import Enum
from maga_transformer.utils.util import get_config_from_path
'''
embedding: return model embedding output
language_model: return token by lm_head
sequence_classfication: return label and score by classifier
'''
class TaskType(Enum):
    DENSE_EMBEDDING           = "DENSE_EMBEDDING"
    ALL_EMBEDDING             = "ALL_EMBEDDING"
    SPARSE_EMBEDDING          = "SPARSE_EMBEDDING"
    COLBERT_EMBEDDING         = "COLBERT_EMBEDDING"
    LANGUAGE_MODEL            = "LANGUAGE_MODEL"
    SEQ_CLASSIFICATION        = "SEQ_CLASSIFICATION"
    RERANKER                  = "RERANKER"
    LINEAR_SOFTMAX            = "LINEAR_SOFTMAX"
    PLUGIN_TASK               = "PLUGIN_TASK"
    BGE_M3                    = "BGE_M3"
    
    @staticmethod
    def from_str(task_type: str):
        for val in TaskType:
            if val.value == task_type:
                return val
        raise Exception(f"unknown task type: {task_type}")
    
def check_task_type(ckpt_path: str):        
    def _is_dense_embedding_task(ckpt_path: str) -> bool:
        def _check_is_sentence_transformer_repo() -> bool:
            if os.path.exists(os.path.join(ckpt_path, "config_sentence_transformers.json")):
                return True
            module_file_path = os.path.join(ckpt_path, "modules.json")
            if os.path.exists(module_file_path):
                with open(module_file_path, 'r') as reader:
                    content = reader.read()
                if 'sentence_transformers' in content:
                    return True
            return False
        return os.environ.get('EMBEDDING_MODEL', '0') == '1' or _check_is_sentence_transformer_repo()

    def _is_classifier_task(ckpt_path: str) -> bool:
        config_json = get_config_from_path(ckpt_path)
        if not config_json:
            return False
        if 'architectures' in config_json and len(config_json['architectures']) > 0:
                model_type = config_json['architectures'][0]
                if 'SequenceClassification' in model_type:
                    return True
        return False

    # from_env
    if 'TASK_TYPE' in os.environ and os.environ['TASK_TYPE'] != '':
        return TaskType.from_str(os.environ['TASK_TYPE'])
    # from config
    elif _is_dense_embedding_task(ckpt_path):
        return TaskType.DENSE_EMBEDDING
    elif _is_classifier_task(ckpt_path):
        return TaskType.SEQ_CLASSIFICATION
    else:
        return TaskType.LANGUAGE_MODEL
import os
from typing import Optional
from transformers import PreTrainedTokenizerBase

from maga_transformer.config.task_type import TaskType
from maga_transformer.utils.util import get_config_from_path
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.models.downstream_modules.plugin_loader import UserModuleLoader
from maga_transformer.models.downstream_modules import SparseEmbeddingModule, DenseEmbeddingModule, ColBertEmbeddingModule, ClassifierModule

def load_task_type(param: GptInitModelParameters) -> TaskType:
    # from_env
    if 'TASK_TYPE' in os.environ and os.environ['TASK_TYPE'] != '':
        return TaskType.from_str(os.environ['TASK_TYPE'])
    # from config
    if _is_dense_embedding_task(param.ckpt_path):
        return TaskType.DENSE_EMBEDDING
    if _is_classifier_task(param.ckpt_path):
        return TaskType.SEQ_CLASSIFICATION
    return TaskType.LANGUAGE_MODEL

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

def create_custom_module(task_type: TaskType, config: GptInitModelParameters, tokenizer: Optional[PreTrainedTokenizerBase]):
    if task_type == TaskType.LANGUAGE_MODEL:
        return None
    if task_type == TaskType.PLUGIN_TASK:
        if 'USER_MODULE_PATH' not in os.environ:
            raise Exception("USER_MODULE_PATH is needed when use task_type==PLUGIN_TASK")
        module_path = os.environ['USER_MODULE_PATH']
        return UserModuleLoader.load(os.path.join(config.ckpt_path, module_path))(config, tokenizer)

    assert tokenizer is not None, "tokenizer should not be None"
    if task_type == TaskType.DENSE_EMBEDDING:
        return DenseEmbeddingModule(config, tokenizer)
    elif task_type == TaskType.SPARSE_EMBEDDING:
        return SparseEmbeddingModule(config, tokenizer)
    elif task_type == TaskType.COLBERT_EMBEDDING:
        return ColBertEmbeddingModule(config, tokenizer)
    elif task_type == TaskType.SEQ_CLASSIFICATION:
        return ClassifierModule(config, tokenizer)
    raise Exception(f"unknown task_type: {task_type}")
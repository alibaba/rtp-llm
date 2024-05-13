from enum import Enum
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
    PLUGIN_TASK               = "PLUGIN_TASK"
    
    @staticmethod
    def from_str(task_type: str):
        for val in TaskType:
            if val.value == task_type:
                return val
        raise Exception(f"unknown task type: {task_type}")

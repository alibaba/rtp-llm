"""RoBERTa/XLM-RoBERTa direct checkpoint owner.

The service config keeps Roberta.from_huggingface's position_ids_style=1.
EmbeddingExecutor supplies packed positions with pad_token_id + 1 added once;
the shared encoder consumes those positions unchanged. Checkpoint rows, including
padding rows, are loaded verbatim and are not zeroed or shifted by the loader.

Classifier and reranker projections are declared by the existing custom handler.
Sparse/ColBERT sidecar files remain owned by their existing handlers; no alternate
W.* encoder tree or handler runtime is introduced here.
"""

from rtp_llm.models_py.new_models.bert.language import BertEncoderOwner


class RobertaEncoderOwner(BertEncoderOwner):
    checkpoint_prefix = "roberta"

import importlib
from typing import Any, Dict

_CLASS_TO_MODULE: Dict[str, str] = {
    "ALLEmbeddingModule": "rtp_llm.models.downstream_modules.embedding.all_embedding_module",
    "BertRerankerModule": "rtp_llm.models.downstream_modules.reranker.reranker_module",
    "BgeM3EmbeddingModule": "rtp_llm.models.downstream_modules.embedding.bge_m3_embedding_module",
    "ClassifierModule": "rtp_llm.models.downstream_modules.classifier.classifier",
    "ColBertEmbeddingModule": "rtp_llm.models.downstream_modules.embedding.colbert_embedding_module",
    "DenseEmbeddingModule": "rtp_llm.models.downstream_modules.embedding.dense_embedding_module",
    "RerankerModule": "rtp_llm.models.downstream_modules.reranker.reranker_module",
    "RobertaClassifierModule": "rtp_llm.models.downstream_modules.classifier.roberta_classifier",
    "RobertaRerankerModule": "rtp_llm.models.downstream_modules.reranker.reranker_module",
    "SparseEmbeddingModule": "rtp_llm.models.downstream_modules.embedding.sparse_emebdding_module",
}

__all__ = sorted(_CLASS_TO_MODULE)


def __getattr__(name: str) -> Any:
    module_path = _CLASS_TO_MODULE.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value

import os as _os

# Phase-25 namespace merge: extend `rtp_llm.models.downstream_modules.__path__`
# with the sibling internal_source counterpart so internal modules (mainse,
# biencoder_flot_module, ...) are reachable without `internal_source.` prefix.
_internal_dir = _os.path.normpath(
    _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "internal_source",
        "rtp_llm",
        "models",
        "downstream_modules",
    )
)
if _os.path.isdir(_internal_dir) and _internal_dir not in __path__:
    __path__.append(_internal_dir)
del _os, _internal_dir

from .classifier.classifier import ClassifierModule
from .classifier.roberta_classifier import RobertaClassifierModule
from .embedding.all_embedding_module import ALLEmbeddingModule
from .embedding.bge_m3_embedding_module import BgeM3EmbeddingModule
from .embedding.colbert_embedding_module import ColBertEmbeddingModule
from .embedding.dense_embedding_module import DenseEmbeddingModule
from .embedding.sparse_emebdding_module import SparseEmbeddingModule
from .reranker.reranker_module import (
    BertRerankerModule,
    RerankerModule,
    RobertaRerankerModule,
)

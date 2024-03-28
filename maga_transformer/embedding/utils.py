import torch
import numpy as np
from typing import List, Any, Callable, Dict
from maga_transformer.embedding.embedding_config import EmbeddingType
from maga_transformer.embedding.api_datatype import EmbeddingResponseFormat, OpenAIEmbeddingResponse, SimilarityResponse

def calc_dense_similarity(left: EmbeddingResponseFormat, right: EmbeddingResponseFormat) -> float:
    assert isinstance(left.embedding, list) and isinstance(right.embedding, list)
    return float(np.array(left.embedding) @ np.array(right.embedding).T)

def calc_sparse_similarity(left: EmbeddingResponseFormat, right: EmbeddingResponseFormat) -> float:
    assert isinstance(left.embedding, dict) and isinstance(right.embedding, dict), "sparse similaritey datatype error"
    left_embedding = left.embedding
    right_embedding = right.embedding
    result: float = 0
    for key in left_embedding.keys():
        if key not in right_embedding:
            continue
        result += left_embedding[key] * right_embedding[key]
    return result

def calc_colbert_similarity(left: EmbeddingResponseFormat, right: EmbeddingResponseFormat) -> float:
    assert isinstance(left.embedding, list) and isinstance(right.embedding, list), "colbert similaritey datatype error"
    q_reps, p_reps = torch.tensor(left.embedding), torch.tensor(right.embedding)
    token_scores = torch.einsum('in,jn->ij', q_reps, p_reps)
    scores, _ = token_scores.max(-1)
    scores = torch.sum(scores) / q_reps.size(0)
    return float(scores)

_FUNC_MAP: Dict[EmbeddingType, Callable[[EmbeddingResponseFormat, EmbeddingResponseFormat],  float]] = {
    EmbeddingType.DENSE: calc_dense_similarity,
    EmbeddingType.SPARSE: calc_sparse_similarity,
    EmbeddingType.COLBERT: calc_colbert_similarity
}

def calc_similarity(left: OpenAIEmbeddingResponse, right: OpenAIEmbeddingResponse, type: EmbeddingType) -> SimilarityResponse:        
    compare_func = _FUNC_MAP[type]
    batch_results: List[List[float]] = []
    for l_item in left.data:
        result: List[float] = []
        for r_item in right.data:
            result.append(compare_func(l_item, r_item))
        batch_results.append(result)
    return SimilarityResponse(type=type, similarity=batch_results)


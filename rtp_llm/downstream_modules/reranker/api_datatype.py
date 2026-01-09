from typing import List, Optional

from rtp_llm.config.base_model_config import PyDanticModelBase


class VoyageRerankerRequest(PyDanticModelBase):
    query: str
    documents: List[str]
    instruction: Optional[str] = None
    model: Optional[str] = None
    top_k: Optional[int] = None
    sorted: bool = True
    truncation: bool = True
    return_documents: bool = True
    normalize: bool = False


class RankingItem(PyDanticModelBase):
    index: int
    document: Optional[str] = None
    relevance_score: float


class VoyageRerankerResponse(PyDanticModelBase):
    results: List[RankingItem]
    total_tokens: int
    model: Optional[str] = None

from typing import List, Optional
from maga_transformer.config.base_model_config import PyDanticModelBase

class VoyageRerankerRequest(PyDanticModelBase):
    query: str
    documents: List[str]
    model: Optional[str] = None
    top_k: Optional[int] = None
    truncation: bool = True
    
class RankingItem(PyDanticModelBase):
    index: int
    document: str
    relevance_score: float
    
class VoyageRerankerResponse(PyDanticModelBase):
    results: List[RankingItem]
    total_tokens: int
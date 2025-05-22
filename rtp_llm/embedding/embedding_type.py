from enum import Enum

class EmbeddingType(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    COLBERT = "colbert"

TYPE_STR = '__TYPE'
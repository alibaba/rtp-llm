"""
Base modules - independent modules without dependencies on other modules.
Device-specific ops are loaded via device routing.
"""

from rtp_llm.device import get_current_device

# Import common base modules (architecture-independent)
from rtp_llm.models_py.modules.base.common.embedding import Embedding, EmbeddingBert
from rtp_llm.models_py.modules.base.common.kvcache_store import WriteCacheStoreOp
from rtp_llm.models_py.modules.base.common.norm import (
    AddBiasResLayerNormTorch,
    LayerNorm,
    LayerNormTorch,
    RMSNormTorch,
    RMSResNormTorch,
)

# Device-specific ops (routed via device.get_base_ops())
_ops = get_current_device().get_base_ops()

FusedSiluAndMul = _ops.FusedSiluAndMul
RMSNorm = _ops.RMSNorm
RMSResNorm = _ops.RMSResNorm
AddBiasResLayerNorm = _ops.AddBiasResLayerNorm
FusedQKRMSNorm = _ops.FusedQKRMSNorm
QKRMSNorm = _ops.QKRMSNorm
SelectTopk = _ops.SelectTopk
GroupTopK = _ops.GroupTopK
FakeBalanceExpert = _ops.FakeBalanceExpert
IndexerOp = _ops.IndexerOp
SigmoidGateScaleAdd = _ops.SigmoidGateScaleAdd

__all__ = [
    "Embedding",
    "EmbeddingBert",
    "WriteCacheStoreOp",
    "AddBiasResLayerNorm",
    "AddBiasResLayerNormTorch",
    "LayerNorm",
    "LayerNormTorch",
    "RMSNormTorch",
    "RMSResNormTorch",
    "FusedQKRMSNorm",
    "QKRMSNorm",
    "RMSNorm",
    "RMSResNorm",
    "SelectTopk",
    "GroupTopK",
    "FakeBalanceExpert",
    "FusedSiluAndMul",
    "IndexerOp",
    "SigmoidGateScaleAdd",
]

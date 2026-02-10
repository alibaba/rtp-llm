"""
Base modules - independent modules without dependencies on other modules.
Different architectures may have different implementations.
"""

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
from rtp_llm.ops.compute_ops import DeviceType, get_device

# Determine device type and import architecture-specific modules
device_type = get_device().get_device_type()

if device_type == DeviceType.ROCm:
    from rtp_llm.models_py.modules.base.rocm.activation import FusedSiluAndMul
    from rtp_llm.models_py.modules.base.rocm.norm import (
        AddBiasResLayerNorm,
        FusedQKRMSNorm,
        QKRMSNorm,
        RMSNorm,
    )

    # Import NotImplementedOp placeholders for ROCm
    from rtp_llm.models_py.modules.base.rocm.not_implemented_ops import (
        GroupTopK,
        IndexerOp,
        RMSResNorm,
    )
    from rtp_llm.models_py.modules.base.rocm.select_topk import SelectTopk
else:
    from rtp_llm.models_py.modules.base.cuda.activation import FusedSiluAndMul
    from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
    from rtp_llm.models_py.modules.base.cuda.norm import (
        AddBiasResLayerNorm,
        FusedQKRMSNorm,
        QKRMSNorm,
        RMSNorm,
        RMSResNorm,
    )
    from rtp_llm.models_py.modules.base.cuda.select_topk import GroupTopK, SelectTopk

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
        "FusedSiluAndMul",
        "IndexerOp",
    ]

from importlib import import_module

__all__ = [
    # Base modules
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
    # Factory modules
    "FusedMoeFactory",
    "LinearFactory",
    "AttnImplFactory",
    "FMHAImplBase",
    # Hybrid modules
    "CausalAttention",
    "MlaAttention",
    "DenseMLP",
    # MoE gating ops
    "SigmoidGateScaleAdd",
    # Multimodal modules
    "MultimodalDeepstackInjector",
    "MultimodalEmbeddingInjector",
    "reshape_extra_input_to_deepstack",
]

_FACTORY_EXPORTS = {
    "FusedMoeFactory",
    "LinearFactory",
    "AttnImplFactory",
    "FMHAImplBase",
}
_HYBRID_EXPORTS = {"CausalAttention", "MlaAttention", "DenseMLP"}


def __getattr__(name: str):
    """Load runtime modules only when their public symbols are requested.

    Importing a leaf module such as fused_moe.defs.config_adapter must not load
    CUDA operators merely because Python first executes this package file.
    """
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in _FACTORY_EXPORTS:
        module_name = "rtp_llm.models_py.modules.factory"
    elif name in _HYBRID_EXPORTS:
        module_name = "rtp_llm.models_py.modules.hybrid"
    else:
        module_name = "rtp_llm.models_py.modules.base"
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value

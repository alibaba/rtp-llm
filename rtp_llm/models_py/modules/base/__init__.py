"""Base components with platform implementations loaded on first use."""

from importlib import import_module

from rtp_llm.device.device_type import DeviceType, get_device_type

_COMMON_MODULES = {
    "Embedding": "embedding",
    "EmbeddingBert": "embedding",
    "WriteCacheStoreOp": "kvcache_store",
    "MultimodalDeepstackInjector": "multimodal_embedding",
    "MultimodalEmbeddingInjector": "multimodal_embedding",
    "reshape_extra_input_to_deepstack": "multimodal_embedding",
    "AddBiasResLayerNormTorch": "norm",
    "LayerNormTorch": "norm",
    "RMSNormTorch": "norm",
    "RMSResNormTorch": "norm",
}
_PLATFORM_MODULES = {
    "FusedSiluAndMul": "activation",
    "SigmoidGateScaleAdd": "moe_gating",
    "AddBiasResLayerNorm": "norm",
    "FusedQKRMSNorm": "norm",
    "LayerNorm": "norm",
    "QKRMSNorm": "norm",
    "RMSNorm": "norm",
    "RMSResNorm": "norm",
    "FakeBalanceExpert": "select_topk",
    "GroupTopK": "select_topk",
    "SelectTopk": "select_topk",
    "IndexerOp": "indexer_op",
}
__all__ = list(_COMMON_MODULES) + list(_PLATFORM_MODULES)


def __getattr__(name: str):
    # A leaf import such as common.fp8_linear must not initialize unrelated
    # platform operators. Keep the existing package-level exports available.
    if name in _COMMON_MODULES:
        module = "common." + _COMMON_MODULES[name]
    elif name in _PLATFORM_MODULES:
        platform = "rocm" if get_device_type() == DeviceType.ROCm else "cuda"
        operation = _PLATFORM_MODULES[name]
        if platform == "rocm" and name in (
            "FakeBalanceExpert",
            "GroupTopK",
            "IndexerOp",
        ):
            operation = "not_implemented_ops"
        module = platform + "." + operation
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value

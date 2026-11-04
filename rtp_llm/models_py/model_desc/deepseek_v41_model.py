"""Device dispatch for V4.1. Import no private CUDA model before selection."""

from rtp_llm.config.v41_runtime import select_v41_backend


def configure_v41(model, model_config) -> None:
    config = dict(getattr(model_config, "deepseek_v41_config", None) or {})
    if not config:
        raise ValueError("DeepSeek V4.1 requires deepseek_v41_config")
    model._v4_args.v41_config = config
    model._v4_args.n_hash_layers = 0


def get_v41_model_class():
    backend = select_v41_backend()
    if backend == "cuda":
        from rtp_llm.models_py.model_desc.deepseek_v41_cuda_model import (
            DeepSeekV41Model,
        )

        return DeepSeekV41Model
    from rtp_llm.models_py.modules.dsv41.backend import get_v41_backend

    return get_v41_backend(backend).model_cls


def __getattr__(name):
    if name == "DeepSeekV41Model":
        return get_v41_model_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DeepSeekV41Model", "configure_v41", "get_v41_model_class"]

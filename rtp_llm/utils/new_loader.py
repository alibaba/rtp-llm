from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from rtp_llm.config.model_config import ModelConfig
    from rtp_llm.ops import DeviceResourceConfig, ParallelismConfig


class NewLoaderConfigSource(Protocol):
    use_new_loader: Optional[bool]


def is_new_loader_enabled(
    model_config: NewLoaderConfigSource, *, default_enabled: bool = False
) -> bool:
    """Resolve an explicit loader override against the model-specific default."""
    if not isinstance(default_enabled, bool):
        raise TypeError("default_enabled must be a bool")
    configured = model_config.use_new_loader
    if configured is None:
        return default_enabled
    if not isinstance(configured, bool):
        raise TypeError("model_config.use_new_loader must be a bool or None")
    return configured


def new_loader_unsupported_reason(
    model_config: "ModelConfig",
    *,
    skip_python_model: bool = False,
    force_cpu_load_weights: bool = False,
    device_resource_config: Optional["DeviceResourceConfig"] = None,
    parallelism_config: Optional["ParallelismConfig"] = None,
) -> Optional[str]:
    """Return why a runtime configuration still requires the legacy loader.

    Shared by language and multimodal loading so automatic routing cannot
    select different loaders for the two halves of one deployment.
    """
    if skip_python_model:
        return "newloader requires the Python model runtime"
    if force_cpu_load_weights:
        return "force_cpu_load_weights is not supported by this newloader slice"
    if model_config.enable_output_vocab_pruning:
        return "output vocabulary pruning is not supported by this newloader slice"
    if model_config.eplb_config.enable_eplb():
        return "EPLB is not supported by this newloader slice"
    if model_config.ptuning_path:
        return "p-tuning is not supported by this newloader slice"
    if model_config.lora_infos:
        return "LoRA loading is not supported by this newloader slice"
    if model_config.require_weight_update:
        return (
            "online UpdateWeights is required but is not supported by NewLoader; "
            "use --use_new_loader false"
        )
    quant_config = model_config.quant_config
    if quant_config is not None:
        runtime_method = quant_config.get_runtime_method_key()
        if not isinstance(runtime_method, str) or not runtime_method.strip():
            return (
                f"quantization config {type(quant_config).__name__} does not "
                "provide a supported NewLoader runtime method"
            )
    if (
        device_resource_config is not None
        and device_resource_config.enable_layer_micro_batch != 0
    ):
        return "layer micro-batch is not supported by this newloader slice"
    if parallelism_config is None:
        return None

    attn_tp = (
        parallelism_config.get_attn_tp_size(),
        parallelism_config.get_attn_tp_rank(),
    )
    ffn_tp = (
        parallelism_config.get_ffn_tp_size(),
        parallelism_config.get_ffn_tp_rank(),
    )
    physical_tp = (parallelism_config.tp_size, parallelism_config.tp_rank)
    if parallelism_config.prefill_cp_config.is_enabled() or attn_tp != physical_tp:
        return "Context parallelism is not supported by this newloader slice"
    if ffn_tp != attn_tp:
        return (
            "Independent FFN TP/sequence parallelism is not supported by this "
            "newloader slice"
        )
    if parallelism_config.ffn_disaggregate_config.enable_ffn_disaggregate:
        return "FFN disaggregation is not supported by this newloader slice"
    return None

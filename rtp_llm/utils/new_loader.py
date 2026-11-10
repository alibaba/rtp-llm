import json
import os
from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from rtp_llm.config.model_config import ModelConfig
    from rtp_llm.ops import MoeConfig, ParallelismConfig


class NewLoaderConfigSource(Protocol):
    use_new_loader: Optional[bool]
    require_weight_update: Optional[bool]


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
    parallelism_config: Optional["ParallelismConfig"] = None,
    moe_config: Optional["MoeConfig"] = None,
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
    if (
        model_config.use_new_loader is None
        and model_config.require_weight_update is None
    ):
        return (
            "the online UpdateWeights policy is undeclared; set "
            "--require_weight_update false to opt in to automatic NewLoader "
            "routing, or true to retain the legacy loader"
        )
    if model_config.require_weight_update:
        return (
            "online UpdateWeights is required but is not supported by NewLoader; "
            "use --use_new_loader false"
        )
    checkpoint_reason = new_loader_checkpoint_unsupported_reason(model_config)
    if checkpoint_reason is not None:
        return checkpoint_reason
    if moe_config is not None and moe_config.use_deepep_low_latency:
        from rtp_llm.device.device_type import DeviceType, get_device_type

        if get_device_type() == DeviceType.ROCm:
            return "DeepEP low-latency MoE is not supported by NewLoader on ROCm"
    quant_config = model_config.quant_config
    if quant_config is not None:
        runtime_method = quant_config.get_runtime_method_key()
        if not isinstance(runtime_method, str) or not runtime_method.strip():
            return (
                f"quantization config {type(quant_config).__name__} does not "
                "provide a supported NewLoader runtime method"
            )
        if model_config.expert_num > 0:
            from rtp_llm.device.device_type import DeviceType, get_device_type

            moe_reason = quant_config.get_new_loader_moe_unsupported_reason()
            if moe_reason is not None:
                return moe_reason
            device_type = get_device_type()
            moe_runtime_method = quant_config.get_moe_runtime_method_key()
            # Keep the routing capability matrix aligned with the strategies
            # registered by fused_moe/__init__.py.  A non-empty linear runtime
            # key alone does not prove that the device has a fused-MoE executor
            # for the same checkpoint representation.
            if device_type == DeviceType.ROCm:
                supported_moe_methods = {
                    "FP8_PER_BLOCK",
                    "FP8_PER_CHANNEL_COMPRESSED",
                    "FP8_PER_CHANNEL_QUARK",
                }
            elif device_type in (DeviceType.Cuda, DeviceType.Ppu):
                supported_moe_methods = {
                    "FP8_DYNAMIC_PER_TENSOR",
                    "FP8_PER_BLOCK",
                    "W4A8_INT4_PER_CHANNEL",
                    "W4A8_INT4_PER_CHANNEL_COMPRESSED",
                }
            else:
                supported_moe_methods = set()
            if moe_runtime_method not in supported_moe_methods:
                return (
                    f"{device_type.name} {moe_runtime_method} MoE is unsupported "
                    "by NewLoader; no matching fused-MoE strategy is registered"
                )
    if parallelism_config is None:
        return None

    tp_size = int(parallelism_config.tp_size)
    vocab_size = int(model_config.vocab_size)
    if (
        model_config.model_type not in ("bert", "roberta")
        and tp_size > 1
        and vocab_size > 0
        and vocab_size % tp_size != 0
    ):
        return (
            f"vocabulary size {vocab_size} is not divisible by tensor-parallel "
            f"size {tp_size}; NewLoader does not yet implement legacy vocabulary "
            "padding"
        )

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
    return None


def new_loader_checkpoint_unsupported_reason(model_config) -> Optional[str]:
    """Check on-disk layouts before selecting either half of a multimodal model.

    FT dumps already contain legacy W-name/rank-local tensors. They are not HF
    weights and must not be dispatched through the streaming HF loader.
    """
    ckpt_path = getattr(model_config, "ckpt_path", None)
    if not ckpt_path:
        return None
    index_path = os.path.join(ckpt_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as source:
            index = json.load(source)
        if isinstance(index, dict) and index.get("is_ft_style_weight", False):
            return "FT-style pre-sharded checkpoints require the legacy loader"
    if model_config.model_type == "llama":
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.isfile(config_path):
            return "Llama NewLoader requires a Hugging Face config.json, not Meta params.json"
        with open(config_path, encoding="utf-8") as source:
            config = json.load(source)
        if not isinstance(config, dict):
            raise ValueError("Llama checkpoint config.json must contain an object")
        architectures = config.get("architectures")
        if architectures is not None and (
            not isinstance(architectures, list)
            or not all(isinstance(name, str) for name in architectures)
        ):
            raise TypeError("architectures must be a list of strings")
        if config.get("model_type", "llama") != "llama" or (
            architectures is not None and "LlamaForCausalLM" not in architectures
        ):
            return "Llama NewLoader supports the Hugging Face LlamaForCausalLM checkpoint layout only"
        if config.get("attention_bias", False) or config.get("mlp_bias", False):
            return "Biased Llama projections are not supported by NewLoader"
        if config.get("hidden_act", "silu") != "silu":
            return "Llama NewLoader requires hidden_act='silu'"
    return None

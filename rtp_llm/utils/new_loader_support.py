"""Capability boundary for the first, unquantized dense-model rollout."""

import json
import os


def new_loader_unsupported_reason(
    config,
    *,
    parallelism_config,
    force_cpu_load_weights=False,
    skip_python_model=False,
    load_method=None
):
    if skip_python_model:
        return "NewLoader requires the Python model runtime"
    if config.use_new_loader is None and config.require_weight_update is None:
        return "online UpdateWeights policy is undeclared; set --require_weight_update false to opt in"
    if config.require_weight_update:
        return "online UpdateWeights is required but is not supported by NewLoader"
    if force_cpu_load_weights:
        return "force_cpu_load_weights is not supported by this NewLoader batch"
    if config.quant_config is not None:
        return "quantized checkpoints and online quantization are not supported by this NewLoader batch"
    if config.expert_num > 0:
        return "MoE is not supported by this NewLoader batch"
    if config.mm_model_config.is_multimodal:
        return "multimodal runtimes are not included in this dense-text batch"
    if config.enable_output_vocab_pruning:
        return "output vocabulary pruning is not supported by this NewLoader batch"
    if config.eplb_config.enable_eplb():
        return "EPLB is not supported by this NewLoader batch"
    if config.ptuning_path or config.lora_infos:
        return "p-tuning and LoRA are not supported by this NewLoader batch"
    from rtp_llm.ops import TaskType

    if config.task_type != TaskType.LANGUAGE_MODEL:
        return "only the language-model task is supported by this NewLoader batch"
    from rtp_llm.models_py.model_loader import NewLoaderLoadMethod, NewModelLoader

    if (
        NewModelLoader.resolve_requested_load_method(load_method)
        == NewLoaderLoadMethod.FASTSAFETENSORS
    ):
        return "fastsafetensors is not supported by this NewLoader version; use scratch or legacy"
    parallel = parallelism_config
    if parallel.pp_size != 1:
        return "pipeline parallelism is not supported by this NewLoader batch"
    if parallel.dp_size != 1 or parallel.ep_size != 1:
        return "independent DP/EP is not supported by this dense-text batch"
    physical_tp = (parallel.tp_size, parallel.tp_rank)
    attn_tp = (parallel.get_attn_tp_size(), parallel.get_attn_tp_rank())
    ffn_tp = (parallel.get_ffn_tp_size(), parallel.get_ffn_tp_rank())
    if parallel.prefill_cp_config.is_enabled() or attn_tp != physical_tp:
        return "Context Parallelism / independent attention TP is not supported by this NewLoader batch"
    if ffn_tp != attn_tp:
        return "independent FFN TP / sequence parallelism is not supported by this NewLoader batch"
    if config.vocab_size % parallel.tp_size:
        return "vocabulary padding for non-divisible TP vocabularies is not supported by this NewLoader batch"
    index_path = os.path.join(config.ckpt_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
        if isinstance(index, dict) and index.get("is_ft_style_weight", False):
            return "FT-style pre-sharded checkpoints require the legacy loader"
    if config.model_type == "llama":
        from rtp_llm.models_py.new_models.llama.language import _validate_llama_config

        try:
            _validate_llama_config(config)
        except (ValueError, TypeError, FileNotFoundError) as exc:
            return str(exc)
    return None

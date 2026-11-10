"""Fail-closed configuration checks for the Q36 Megatron copying launcher."""

from collections.abc import Mapping
import os
from pathlib import Path


def get(config, *keys):
    value = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError("missing config " + ".".join(keys))
        value = value[key]
    return value


def validate(config):
    checks = {
        ("model_engine",): "megatron",
        ("trainer", "nnodes"): 1,
        ("trainer", "n_gpus_per_node"): 4,
        ("rllm", "backend"): "verl",
        ("rllm", "algorithm", "adv_estimator"): "grpo",
        ("actor_rollout_ref", "actor", "_target_"): "verl.workers.config.McoreActorConfig",
        ("actor_rollout_ref", "actor", "strategy"): "megatron",
        ("actor_rollout_ref", "actor", "megatron", "tensor_model_parallel_size"): 4,
        ("actor_rollout_ref", "actor", "megatron", "expert_model_parallel_size"): 1,
        ("actor_rollout_ref", "actor", "megatron", "expert_tensor_parallel_size"): 4,
        ("actor_rollout_ref", "actor", "megatron", "pipeline_model_parallel_size"): 1,
        ("actor_rollout_ref", "actor", "megatron", "context_parallel_size"): 1,
        ("actor_rollout_ref", "actor", "megatron", "use_megatron_fsdp"): False,
        ("actor_rollout_ref", "actor", "megatron", "param_offload"): False,
        ("actor_rollout_ref", "actor", "megatron", "optimizer_offload"): False,
        ("actor_rollout_ref", "actor", "megatron", "use_distributed_optimizer"): False,
        ("actor_rollout_ref", "actor", "megatron", "override_ddp_config", "grad_reduce_in_fp32"): False,
        ("actor_rollout_ref", "actor", "megatron", "override_ddp_config", "overlap_grad_reduce"): False,
        ("actor_rollout_ref", "actor", "megatron", "override_transformer_config", "gradient_accumulation_fusion"): False,
        ("actor_rollout_ref", "actor", "megatron", "override_transformer_config", "attention_backend"): "fused",
        ("actor_rollout_ref", "actor", "megatron", "override_transformer_config", "mtp_num_layers"): 0,
        ("actor_rollout_ref", "actor", "megatron", "dtype"): "bfloat16",
        ("actor_rollout_ref", "actor", "ppo_micro_batch_size_per_gpu"): 1,
        ("actor_rollout_ref", "actor", "optim", "optimizer"): "adam",
        ("actor_rollout_ref", "actor", "optim", "use_precision_aware_optimizer"): False,
        ("actor_rollout_ref", "actor", "optim", "override_optimizer_config", "optimizer_cpu_offload"): True,
        ("actor_rollout_ref", "actor", "optim", "override_optimizer_config", "optimizer_offload_fraction"): 1.0,
        ("actor_rollout_ref", "actor", "optim", "override_optimizer_config", "overlap_cpu_optimizer_d2h_h2d"): False,
        ("actor_rollout_ref", "model", "lora_rank"): 0,
        ("actor_rollout_ref", "model", "use_fused_kernels"): False,
        ("actor_rollout_ref", "model", "mtp", "enable"): False,
        ("actor_rollout_ref", "model", "mtp", "enable_train"): False,
        ("actor_rollout_ref", "model", "mtp", "enable_rollout"): False,
        ("actor_rollout_ref", "rollout", "name"): "rtp",
        ("actor_rollout_ref", "rollout", "mode"): "async",
        ("actor_rollout_ref", "rollout", "tensor_model_parallel_size"): 4,
        ("actor_rollout_ref", "rollout", "data_parallel_size"): 1,
        ("actor_rollout_ref", "rollout", "pipeline_model_parallel_size"): 1,
    }
    for path, expected in checks.items():
        actual = get(config, *path)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"{'.'.join(path)} expected {expected!r}, got {actual!r}")
    model_path = get(config, "actor_rollout_ref", "model", "path")
    configured = os.environ.get('Q36_MODEL_PATH')
    if (not isinstance(model_path, str) or not configured or
            Path(model_path).resolve() != Path(configured).resolve() or
            not (Path(model_path) / 'config.json').is_file()):
        raise ValueError('model path must match Q36_MODEL_PATH and contain config.json')

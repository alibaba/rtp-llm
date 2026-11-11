"""The pinned two-update GRPO recipe for one 64-GPU actor and one 32-GPU policy."""

from dataclasses import dataclass
from ipaddress import IPv4Address
import json
from pathlib import Path


TRAIN_NODES = 16
ROLLOUT_NODES = 8
GPUS_PER_NODE = 4
ROLLOUTS = 4
PROMPTS = 8
STEPS = 2
MAX_PROMPT = 2048
MAX_RESPONSE = 2048


@dataclass(frozen=True)
class Recipe:
    metadata: str
    gateway_host: str
    gateway_port: int

    def __post_init__(self):
        path = Path(self.metadata)
        ip = IPv4Address(self.gateway_host)
        if not path.is_absolute() or any(ch in self.metadata for ch in ('\n','\r')):
            raise ValueError('use the absolute official checkpoint metadata directory')
        if not ip.is_private or ip.is_unspecified or ip.is_multicast or ip.is_loopback:
            raise ValueError('the Python Tool gateway must use the assigned training IP')
        if type(self.gateway_port) is not int or not 1024 <= self.gateway_port < 65536:
            raise ValueError('invalid dedicated Python Tool gateway port')

    def overrides(self):
        # rLLM uses its synchronous training loop and calls our split checkpoint
        # manager directly. The framework's unused actor-local checkpoint object
        # remains its default naive instance.
        return [
            'rllm/backend=verl', 'model_engine=megatron',
            'rllm.algorithm.adv_estimator=grpo', 'rllm.algorithm.kl_beta=0.0',
            'rllm.algorithm.rollout_correction.bypass_mode=true',
            'rllm.algorithm.lr_schedule=constant',
            'rllm.algorithm.warmup_steps_ratio=0.0',
            f'rllm.data.train_batch_size={PROMPTS}',
            f'rllm.data.max_prompt_length={MAX_PROMPT}',
            f'rllm.data.max_response_length={MAX_RESPONSE}',
            'rllm.disable_thinking=false', f'rllm.rollout.n={ROLLOUTS}',
            'rllm.rollout.train.temperature=1.0', 'rllm.rollout.train.top_p=1.0',
            'rllm.workflow.n_parallel_tasks=8', 'rllm.workflow.retry_limit=1',
            'rllm.workflow.raise_on_error=true', 'rllm.async_training.enable=false',
            f'rllm.gateway.host={self.gateway_host}', f'rllm.gateway.port={self.gateway_port}',
            f'rllm.trainer.total_batches={STEPS}', f'rllm.trainer.total_epochs={STEPS}',
            'rllm.trainer.save_freq=-1', 'rllm.trainer.test_freq=-1',
            'rllm.trainer.val_before_train=false', 'trainer.resume_mode=disable', 'trainer.balance_batch=true',
            'actor_rollout_ref.model.path='+json.dumps(self.metadata),
            'actor_rollout_ref.model.lora_rank=0',
            'actor_rollout_ref.model.mtp.enable=false',
            'actor_rollout_ref.model.mtp.enable_train=false',
            'actor_rollout_ref.model.mtp.enable_rollout=false',
            'actor_rollout_ref.actor.megatron.tensor_model_parallel_size=8',
            'actor_rollout_ref.actor.megatron.expert_model_parallel_size=64',
            'actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1',
            'actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1',
            'actor_rollout_ref.actor.megatron.context_parallel_size=1',
            'actor_rollout_ref.actor.megatron.use_megatron_fsdp=false',
            'actor_rollout_ref.actor.megatron.param_offload=false',
            'actor_rollout_ref.actor.megatron.optimizer_offload=false',
            'actor_rollout_ref.actor.megatron.use_distributed_optimizer=false',
            '+actor_rollout_ref.actor.megatron.override_ddp_config.grad_reduce_in_fp32=false',
            '+actor_rollout_ref.actor.megatron.override_ddp_config.overlap_grad_reduce=false',
            '+actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=false',
            'actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=fused',
            '+actor_rollout_ref.actor.megatron.override_transformer_config.mtp_num_layers=0',
            '++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full',
            '++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform',
            '++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1',
            'actor_rollout_ref.actor.megatron.dtype=bfloat16',
            '++actor_rollout_ref.actor.router_replay.mode=disabled',
            'actor_rollout_ref.actor.optim.optimizer=adam', 'actor_rollout_ref.actor.optim.lr=1e-6',
            'actor_rollout_ref.actor.optim.use_precision_aware_optimizer=false',
            '+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=true',
            '+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0',
            '+actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=false',
            f'actor_rollout_ref.actor.ppo_mini_batch_size={PROMPTS}',
            f'actor_rollout_ref.actor.ppo_max_token_len_per_gpu={MAX_PROMPT + MAX_RESPONSE}',
            'actor_rollout_ref.actor.ppo_epochs=1',
            'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1',
            'actor_rollout_ref.actor.use_dynamic_bsz=false',
            'actor_rollout_ref.rollout.name=rtp', 'actor_rollout_ref.rollout.mode=async',
            'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1',
            'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false',
            'actor_rollout_ref.rollout.tensor_model_parallel_size=8',
            'actor_rollout_ref.rollout.data_parallel_size=4',
            'actor_rollout_ref.rollout.pipeline_model_parallel_size=1',
            f'actor_rollout_ref.rollout.nnodes={ROLLOUT_NODES}',
            f'actor_rollout_ref.rollout.n_gpus_per_node={GPUS_PER_NODE}',
            'actor_rollout_ref.rollout.checkpoint_engine.backend=naive',
            f'rollout.nnodes={ROLLOUT_NODES}', f'rollout.n_gpus_per_node={GPUS_PER_NODE}',
            f'trainer.nnodes={TRAIN_NODES}', f'trainer.n_gpus_per_node={GPUS_PER_NODE}',
        ]


def compose(recipe):
    import hydra
    from omegaconf import OmegaConf, open_dict
    from rllm.trainer.verl.utils import sync_config

    args = recipe.overrides()
    with hydra.initialize_config_module(version_base=None, config_module='rllm.trainer.config'):
        config = hydra.compose(config_name='unified', overrides=args)
    if not OmegaConf.has_resolver('mul'):
        OmegaConf.register_new_resolver('mul', lambda x, y: int(x) * int(y))
    sync_config(config, hydra_overrides=args)
    actor = config.actor_rollout_ref.actor
    # The pinned sync_config still looks up the pre-MCore-router-replay path.
    # The current actor dataclass takes the switch only within its engine config.
    with open_dict(actor):
        if actor.pop('router_replay')['mode'] != 'disabled':
            raise ValueError('router replay is disabled for original-policy score collection')
    OmegaConf.resolve(config)
    validate(config)
    return config


def validate(config):
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config

    actor = config.actor_rollout_ref.actor
    model = config.actor_rollout_ref.model
    rollout = config.actor_rollout_ref.rollout
    mcore = actor.megatron
    if (config.trainer.nnodes, config.trainer.n_gpus_per_node, rollout.nnodes,
            rollout.n_gpus_per_node, rollout.tensor_model_parallel_size,
            rollout.data_parallel_size, rollout.pipeline_model_parallel_size) != (16,4,8,4,8,4,1):
        raise ValueError('invalid 64+32 deployment size or native policy topology')
    if (mcore.tensor_model_parallel_size, mcore.expert_model_parallel_size,
            mcore.expert_tensor_parallel_size, mcore.pipeline_model_parallel_size,
            mcore.context_parallel_size, mcore.dtype) != (8,64,1,1,1,'bfloat16'):
        raise ValueError('invalid Qwen3.8 Megatron actor topology or precision')
    if any((mcore.use_distributed_optimizer, mcore.use_megatron_fsdp,
            mcore.param_offload, mcore.optimizer_offload, mcore.override_ddp_config.grad_reduce_in_fp32,
            actor.optim.use_precision_aware_optimizer, model.lora_rank, model.mtp.enable,
            config.rllm.async_training.enable, need_critic(config), need_reference_policy(config))):
        raise ValueError('unexpected optimizer, model or reference actor option')
    if (not actor.optim.override_optimizer_config.optimizer_cpu_offload or
            config.rllm.workflow.raise_on_error is not True or config.rllm.workflow.retry_limit != 1 or
            config.rllm.trainer.total_batches != STEPS or config.rllm.rollout.n != ROLLOUTS or
            config.rllm.data.train_batch_size != PROMPTS or not config.actor_rollout_ref.hybrid_engine or
            rollout.checkpoint_engine.backend != 'naive'):
        raise ValueError('invalid synchronous GRPO recipe or worker compatibility settings')
    recompute = mcore.override_transformer_config
    correction = config.rllm.algorithm.rollout_correction
    if (not correction.bypass_mode or correction.tis_mode is not None
            or not actor.use_rollout_log_probs or not config.trainer.balance_batch
            or actor.ppo_epochs != 1 or actor.ppo_micro_batch_size_per_gpu != 1
            or actor.ppo_max_token_len_per_gpu != MAX_PROMPT + MAX_RESPONSE
            or mcore.router_replay.mode != 'disabled'
            or (recompute.recompute_granularity, recompute.recompute_method,
                recompute.recompute_num_layers) != ('full', 'uniform', 1)):
        raise ValueError('invalid on-policy scores or training memory configuration')
    validate_config(config=config, use_reference_policy=False, use_critic=False)

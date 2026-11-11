"""Shorten only the driver-side rollout budget for the resident BF16 actor."""

import os


MAX_RESPONSE = 1024
TOOL_TURNS = 2
TOOL_TOKENS = 512
PARALLEL_TASKS = 32


def configure_rollouts(config):
    from omegaconf import OmegaConf
    from q38_driver_recipe import PROMPTS, ROLLOUTS, STEPS, validate

    validate(config)
    actor = config.actor_rollout_ref.actor
    if (config.data.train_batch_size, config.rllm.data.train_batch_size,
            config.actor_rollout_ref.rollout.n, config.rllm.rollout.n,
            config.rllm.trainer.total_batches, config.data.max_prompt_length,
            actor.ppo_max_token_len_per_gpu, actor.ppo_micro_batch_size_per_gpu,
            actor.ppo_mini_batch_size, actor.optim.lr) != (PROMPTS, PROMPTS, ROLLOUTS, ROLLOUTS,
                                                         STEPS, 2048, 4096, 1, PROMPTS, 1e-6):
        raise RuntimeError("short rollout requires the verified resident actor and two-step recipe")
    resident = OmegaConf.to_container(config.actor_rollout_ref, resolve=True)
    config.rllm.workflow.n_parallel_tasks = PARALLEL_TASKS
    config.rllm.data.max_response_length = MAX_RESPONSE
    config.data.max_response_length = MAX_RESPONSE
    # Gateway session sampling takes precedence over each Tool HTTP request.
    config.rllm.rollout.train.max_tokens = TOOL_TOKENS
    validate(config)
    if (config.rllm.algorithm.adv_estimator != "grpo"
            or config.rllm.workflow.n_parallel_tasks != PROMPTS * ROLLOUTS
            or config.rllm.rollout.train.max_tokens * TOOL_TURNS != MAX_RESPONSE
            or config.data.max_prompt_length + config.data.max_response_length > actor.ppo_max_token_len_per_gpu
            or OmegaConf.to_container(config.actor_rollout_ref, resolve=True) != resident):
        raise RuntimeError("the short rollout profile changed the resident actor or update contract")
    return config


def configure_tool_budget():
    """Apply after the pinned driver's normal _tool_env setup."""
    os.environ.update(
        RLLM_PYTHON_TOOL_MAX_TURNS=str(TOOL_TURNS),
        RLLM_PYTHON_TOOL_CALL_TOKENS=str(TOOL_TOKENS),
        RLLM_PYTHON_TOOL_TOTAL_TOKENS=str(MAX_RESPONSE),
    )

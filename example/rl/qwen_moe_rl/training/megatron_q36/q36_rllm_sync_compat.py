"""Adapt rLLM's router-replay key to this pinned verl Megatron schema."""
OLD = ('actor_rollout_ref.actor.router_replay.mode', 'rllm.algorithm.router_replay')
NEW = ('actor_rollout_ref.actor.megatron.router_replay.mode', 'rllm.algorithm.router_replay')


def install():
    import rllm.trainer.verl.utils as utils
    keys = utils._SHARED_KEYS
    if keys.count(NEW) == 1 and OLD not in keys:
        return
    if keys.count(OLD) != 1 or NEW in keys:
        raise RuntimeError('unexpected rLLM/verl router-replay mapping')
    keys[keys.index(OLD)] = NEW

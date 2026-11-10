"""Adapt pinned rLLM configuration synchronization for the Q36 profiles."""
import functools
import os
OLD = ('actor_rollout_ref.actor.router_replay.mode', 'rllm.algorithm.router_replay')
NEW = ('actor_rollout_ref.actor.megatron.router_replay.mode', 'rllm.algorithm.router_replay')


def install():
    import rllm.trainer.verl.utils as utils
    keys = utils._SHARED_KEYS
    if not (keys.count(NEW) == 1 and OLD not in keys):
        if keys.count(OLD) != 1 or NEW in keys:
            raise RuntimeError('unexpected rLLM/verl router-replay mapping')
        keys[keys.index(OLD)] = NEW

    original = utils.sync_config
    if getattr(original, '_q36_standalone_sync', False):
        return

    @functools.wraps(original)
    def sync_config(config, *args, **kwargs):
        if os.environ.get('Q36_ONPOLICY_NO_CONCURRENT_ROLLOUT') != '1':
            return original(config, *args, **kwargs)
        if (os.environ.get('Q36_STRICT_4PLUS4_CLAIM') != '1' or
                os.environ.get('Q36_DISAGG_RESIDENT_KV') != '1'):
            raise ValueError('standalone on-policy sync requires strict claim and resident KV')
        if config.rllm.get('async_training', {}).get('enable', False):
            raise ValueError('standalone on-policy sync requires async_training disabled')
        checkpoint = config.actor_rollout_ref.rollout.checkpoint_engine
        if (checkpoint.backend != 'rtp_nixl_guarded' or
                checkpoint.custom_backend_module != 'verl_nixl_guarded'):
            raise ValueError('standalone on-policy sync requires the guarded NIXL backend')
        result = original(config, *args, **kwargs)
        if (config.actor_rollout_ref.rollout.checkpoint_engine is not checkpoint or
                checkpoint.backend != 'naive' or config.actor_rollout_ref.hybrid_engine is not True or
                checkpoint.custom_backend_module != 'verl_nixl_guarded' or
                config.rllm.get('async_training', {}).get('enable', False)):
            raise RuntimeError('unexpected pinned rLLM synchronous configuration conversion')
        checkpoint.backend = 'rtp_nixl_guarded'
        print('Q36_ONPOLICY_SYNC backend=rtp_nixl_guarded hybrid_engine=true async_training=false', flush=True)
        return result

    sync_config._q36_standalone_sync = True
    utils.sync_config = sync_config

"""Adapt rLLM's trainer keyword to this pinned verl's actor_wg keyword."""
import functools
import inspect


def install(module):
    cls = module.CheckpointEngineManager
    original = cls.__init__
    if getattr(original, '_q36_trainer_keyword_compat', False): return
    params = inspect.signature(original).parameters
    if 'actor_wg' not in params or 'trainer' in params:
        raise RuntimeError('unexpected checkpoint manager constructor')

    @functools.wraps(original)
    def initialize(self, *args, trainer=None, **kwargs):
        if trainer is not None:
            if 'actor_wg' in kwargs or len(args) > 1:
                raise TypeError('trainer and actor_wg cannot both be supplied')
            kwargs['actor_wg'] = trainer
        return original(self, *args, **kwargs)

    initialize._q36_trainer_keyword_compat = True
    cls.__init__ = initialize

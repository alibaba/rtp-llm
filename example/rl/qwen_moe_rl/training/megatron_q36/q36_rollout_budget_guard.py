"""Reject, never silently clip, native requests exceeding this run's budget."""
import functools
import json


def install(module):
    cls = module.RTPServer
    original = cls.generate
    if getattr(original, '_q36_budget_guard', False): return

    @functools.wraps(original)
    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        params = dict(sampling_params)
        budget = params.get('max_tokens', params.get('max_new_tokens', 2048))
        if type(budget) is not int or not 0 < budget <= 1024:
            raise ValueError('actual RTP request response budget must be 1..1024 tokens')
        if len(prompt_ids) + budget > 8192:
            raise ValueError('actual RTP prompt plus response budget exceeds 8192 tokens')
        print('Q36_ROLLOUT_BUDGET ' + json.dumps({'prompt_tokens': len(prompt_ids),
              'max_new_tokens': budget}), flush=True)
        return await original(self, request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, **kwargs)

    generate._q36_budget_guard = True
    cls.generate = generate

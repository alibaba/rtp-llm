"""Keep the stricter client/session token limit in the qualified 8+4 gateway."""
import functools
import hashlib
import json
import os
from pathlib import Path

EXPECTED_GATEWAY_SHA256 = '220f0dbc43b5bbecec11f3f3dfc3023adc80d8ce1eb42439587fb3d274695efd'
TOKEN_FIELDS = ('max_tokens', 'max_completion_tokens')


def install(module):
    source = Path(module.__file__).resolve(strict=True)
    expected = (Path(os.environ['Q36_RUN_ROOT']).resolve(strict=True).parent / 'manager-probe-env/gateway-src/rllm_model_gateway/middleware.py').resolve(strict=True)
    if source != expected or hashlib.sha256(source.read_bytes()).hexdigest() != EXPECTED_GATEWAY_SHA256:
        raise RuntimeError('gateway middleware source identity mismatch')
    cls = module.SessionRoutingMiddleware
    original = cls._mutate
    if getattr(original, '_q36_token_minimum', False):
        return

    def limits(payload):
        values = []
        for key in TOKEN_FIELDS:
            value = payload.get(key)
            if value is None:
                continue
            if type(value) is not int or value <= 0:
                raise ValueError('token limit must be a strictly positive integer')
            values.append(value)
        return values

    @functools.wraps(original)
    def mutate(self, payload, session_id=None):
        client = limits(payload)
        original(self, payload, session_id)
        merged = limits(payload)
        values = client + merged
        if values:
            effective = min(values)
            payload['max_tokens'] = effective
            payload.pop('max_completion_tokens', None)
            print('Q36_GATEWAY_BUDGET ' + json.dumps({'client': client, 'after_session': merged,
                  'effective': effective}), flush=True)
    mutate._q36_token_minimum = True
    cls._mutate = mutate

"""Record a small real optimizer witness without changing update decisions."""
import hashlib
import json
import time
import types


def attach(optimizer, chunks, torch, dist):
    if not hasattr(optimizer, 'step'): return optimizer
    candidates = [(name, p) for c in chunks for name, p in c.named_parameters()
                  if name.endswith('decoder.layers.0.mlp.router.weight')]
    original = optimizer.step

    def digest(tensor):
        value = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return hashlib.sha256(value).hexdigest()

    def snapshot(parameter, grad=False):
        torch.cuda.synchronize()
        result = {'weight_sha256': digest(parameter)}
        if grad:
            g = getattr(parameter, 'main_grad', None)
            if g is None:
                result['gradient'] = None
            else:
                result['gradient'] = {'dtype': str(g.dtype), 'finite': bool(torch.isfinite(g).all().item()),
                                      'nonzero': int(torch.count_nonzero(g).item()), 'sha256': digest(g)}
        return result

    def step(self, *args, **kwargs):
        started = time.monotonic(); before = None; error = None
        if len(candidates) == 1:
            try: before = snapshot(candidates[0][1], grad=True)
            except Exception as exc: error = 'before: ' + repr(exc)
        else: error = 'expected one layer0 router, observed %d' % len(candidates)
        before_done = time.monotonic()
        try:
            outcome = original(*args, **kwargs)
        except Exception as exc:
            print('Q36_OPTIMIZER_WITNESS ' + json.dumps({'rank': dist.get_rank(), 'step_error': repr(exc),
                  'observation_error': error, 'before': before}), flush=True)
            raise
        after = None; state = {}
        if len(candidates) == 1:
            try:
                p = candidates[0][1]; after = snapshot(p)
                for part in getattr(self, 'chained_optimizers', ()):
                    if getattr(part, 'is_stub_optimizer', False): continue
                    opt = getattr(part, 'optimizer', None)
                    found = getattr(opt, 'state', {}).get(p) if opt is not None else None
                    if found:
                        for key in ('exp_avg', 'exp_avg_sq'):
                            v = found.get(key)
                            if v is not None:
                                state[key] = {'dtype': str(v.dtype), 'device': str(v.device), 'sha256': digest(v)}
            except Exception as exc: error = (error or '') + ' after: ' + repr(exc)
        print('Q36_OPTIMIZER_WITNESS ' + json.dumps({'rank': dist.get_rank(), 'before': before,
              'after': after, 'states': state, 'outcome': str(outcome), 'observation_error': error,
              'before_seconds': before_done - started, 'total_seconds': time.monotonic() - started}), flush=True)
        return outcome

    optimizer.step = types.MethodType(step, optimizer)
    return optimizer

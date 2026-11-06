"""Read-only actual GRPO loss audit adapted from the pushed qwen_moe_rl hooks.

Provenance: example/rl/qwen_moe_rl/training/overrides/qwen36_training_overrides.py
at RTP branch HEAD 6fd1d2618573af5d826fe79cacf2c450560515c0.
No loss, reward, advantage, or gradient tensor is replaced by this observer.
"""
from __future__ import annotations
import contextvars,functools,hashlib,inspect,json,math,os,struct,sys,time
from pathlib import Path
_POLICY_AUDIT_CONTEXT=contextvars.ContextVar('qwen38_policy_audit_context',default=None)

def _policy_audit_directory():
    explicit = os.environ.get('Q38_POLICY_AUDIT_DIR')
    if explicit:
        return Path(explicit)
    # The dedicated model-audit destination itself opts into run auditing and
    # is explicitly forwarded to every Ray worker by the existing launcher.
    model_audit = os.environ.get('Q38_TRAINING_AUDIT_DIR')
    if model_audit:
        return Path(model_audit).parent/'policy-loss'
    return None

def _observed_tensor(tensor, mask=None, include_values=False):
    """Read detached tensor values; never replace a tensor used by the loss."""
    import torch
    if not isinstance(tensor, torch.Tensor):
        return None
    value = tensor.detach()
    if value.is_nested:
        value = value.values()
    flat = value.reshape(-1)
    selected = flat
    if mask is not None:
        active = mask.detach()
        if active.is_nested:
            active = active.values()
        active = active.reshape(-1).to(bool)
        if active.numel() != flat.numel():
            raise RuntimeError('Policy audit mask and observed tensor sizes differ')
        selected = flat[active]
    numbers = selected.float().cpu().tolist()
    finite = all(math.isfinite(x) for x in numbers)
    bits = struct.pack(f'<{len(numbers)}f', *numbers)
    result = dict(shape=list(tensor.shape) if not tensor.is_nested else ['nested'],
                  dtype=str(tensor.dtype), device=str(tensor.device),
                  numel=flat.numel(), selected_numel=len(numbers), finite=finite,
                  nonzero=sum(x != 0 for x in numbers),
                  positive=sum(x > 0 for x in numbers), negative=sum(x < 0 for x in numbers),
                  minimum=min(numbers) if numbers and finite else None,
                  maximum=max(numbers) if numbers and finite else None,
                  sum=sum(numbers) if finite else None,
                  selected_fp32_sha256=hashlib.sha256(bits).hexdigest())
    if include_values:
        result['selected_fp32_bits_hex'] = bits.hex()
    return result

def _observed_native_amp_log_probs(model_output, data):
    """Compare existing loss inputs only; no extra model forward or gate.

    Current PPO log probabilities may belong to a legitimately newer policy.
    This receipt deliberately makes no same-policy claim when no policy
    version is present at the real loss boundary.
    """
    import torch
    from verl.workers.utils.padding import no_padding_2_padding
    observed = dict(available_batch_fields=[str(key) for key in data.keys()],
                    available_model_output_fields=[str(key) for key in model_output.keys()],
                    status='unavailable', hard_threshold_applied=False,
                    extra_model_forward=False, tensors_or_loss_modified=False,
                    same_policy_version_established=False)
    needed = ['rollout_log_probs', 'response_mask']
    missing = [key for key in needed if key not in data]
    if 'log_probs' not in model_output:
        missing.append('model_output.log_probs')
    if missing:
        observed['missing_fields'] = missing
        return observed
    try:
        with torch.no_grad():
            source = model_output['log_probs']
            current = no_padding_2_padding(source.detach(), data)
            original = data.select(*needed).to_padded_tensor()
            native = original['rollout_log_probs'].detach()
            mask = original['response_mask'].detach().to(bool)
            if current.shape != native.shape or native.shape != mask.shape:
                raise RuntimeError(f'Actual response layout mismatch: current={tuple(current.shape)} native={tuple(native.shape)} mask={tuple(mask.shape)}')
            lhs = current[mask].float()
            rhs = native.to(current.device)[mask].float()
            finite = bool(torch.isfinite(lhs).all() and torch.isfinite(rhs).all())
            errors = lhs - rhs
            count = errors.numel()
            observed.update(status='recorded', active_response_tokens=count,
                finite=finite, current_log_probs_dtype=str(source.dtype),
                native_rollout_log_probs_dtype=str(native.dtype), response_shape=list(current.shape),
                current_device=str(source.device),
                autocast_current_device_enabled=torch.is_autocast_enabled(source.device.type),
                maximum_absolute_error=float(errors.abs().max().cpu()) if count and finite else None,
                RMS_error=float(errors.double().square().mean().sqrt().cpu()) if count and finite else None,
                current_selected_fp32_sha256=_observed_tensor(current, mask)['selected_fp32_sha256'],
                native_selected_fp32_sha256=_observed_tensor(native, mask)['selected_fp32_sha256'],
                interpretation='Only an externally verified same-policy pair is a numerical-consistency test; PPO policy updates can legitimately differ.')
    except Exception as exc:
        # Optional observation cannot replace a valid loss with an audit error.
        observed['layout_error'] = f'{type(exc).__name__}: {exc}'
    return observed

def _observed_batch_context(data, model_output=None):
    """Only record fields that are really present at the PPO-loss boundary."""
    import torch
    from verl.utils import tensordict_utils as tu
    observed = {'available_batch_fields': [str(key) for key in data.keys()],
                'rewards': {}, 'group_ids': {}, 'step_fields': {}, 'unreadable_fields': {}}
    for key in ('token_level_rewards', 'token_level_scores', 'rewards', 'returns'):
        value = data.get(key, None)
        if isinstance(value, torch.Tensor):
            observed['rewards'][key] = _observed_tensor(value, include_values=True)
    for key in ('uid', 'group_id', 'group_ids', 'task_id', 'episode_id'):
        try:
            value = tu.get_non_tensor_data(data=data, key=key, default=None)
            if value is not None:
                value = value.tolist() if hasattr(value, 'tolist') else value
                observed['group_ids'][key] = value
        except Exception as exc:
            observed['unreadable_fields'][key] = f'{type(exc).__name__}: {exc}'
    for key in ('global_step', 'global_steps', 'training_step', 'rtp_global_step',
                'weight_version', 'policy_version', 'rollout_policy_version'):
        try:
            value = tu.get_non_tensor_data(data=data, key=key, default=None)
            if value is None:
                value = data.get(key, None)
            if value is not None:
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().tolist()
                observed['step_fields'][key] = value
        except Exception as exc:
            observed['unreadable_fields'][key] = f'{type(exc).__name__}: {exc}'
    if model_output is not None:
        observed['native_rollout_vs_actual_training_log_probs'] = _observed_native_amp_log_probs(model_output, data)
    return observed

def install_policy_loss_audit():
    """Observe actual loss inputs and outputs without changing their graph."""
    import torch
    import torch.distributed as dist
    import verl.trainer.ppo.core_algos as core
    import verl.workers.utils.losses as losses

    for mode, original in list(core.POLICY_LOSS_REGISTRY.items()):
        if getattr(original, '_qwen38_readonly_policy_audit', False):
            continue
        signature = inspect.signature(original)

        def make_wrapper(function, policy_mode, function_signature):
            @functools.wraps(function)
            def audited(*args, **kwargs):
                directory = _policy_audit_directory()
                if directory is None:
                    return function(*args, **kwargs)
                arguments = function_signature.bind(*args, **kwargs).arguments
                result = function(*args, **kwargs)
                advantages = arguments.get('advantages')
                mask = arguments.get('response_mask')
                loss = result[0] if isinstance(result, tuple) else result
                stamp = time.time_ns()
                record = dict(schema='qwen38-real-policy-loss-v1', pid=os.getpid(),
                    rank=dist.get_rank() if dist.is_initialized() else 0,
                    world_size=dist.get_world_size() if dist.is_initialized() else 1,
                    time_ns=stamp, loss_mode=policy_mode,
                    observation_boundary='actual registered policy loss arguments and returned autograd tensor',
                    advantages=_observed_tensor(advantages, mask, include_values=True),
                    response_mask=_observed_tensor(mask),
                    old_log_probs=_observed_tensor(arguments.get('old_log_prob'), mask),
                    current_log_probs=_observed_tensor(arguments.get('log_prob'), mask),
                    loss=_observed_tensor(loss, include_values=True),
                    loss_requires_grad=bool(isinstance(loss, torch.Tensor) and loss.requires_grad),
                    loss_grad_fn=type(loss.grad_fn).__name__ if isinstance(loss, torch.Tensor) and loss.grad_fn is not None else None,
                    loss_agg_mode=arguments.get('loss_agg_mode'),
                    batch_context=_POLICY_AUDIT_CONTEXT.get(),
                    cuda_initialized=torch.cuda.is_initialized())
                directory.mkdir(parents=True, exist_ok=True)
                target = directory/f'policy-loss-rank{record["rank"]}-{stamp}.json'
                target.write_text(json.dumps(record, ensure_ascii=False, default=str, allow_nan=False)+'\n')
                return result
            audited._qwen38_readonly_policy_audit = True
            return audited

        core.POLICY_LOSS_REGISTRY[mode] = make_wrapper(original, mode, signature)

    original_ppo = losses.ppo_loss
    if not getattr(original_ppo, '_qwen38_readonly_batch_audit', False):
        @functools.wraps(original_ppo)
        def observe_ppo(config, model_output, data, *args, **kwargs):
            if _policy_audit_directory() is None:
                return original_ppo(config, model_output, data, *args, **kwargs)
            token = _POLICY_AUDIT_CONTEXT.set(_observed_batch_context(data, model_output))
            try:
                return original_ppo(config, model_output, data, *args, **kwargs)
            finally:
                _POLICY_AUDIT_CONTEXT.reset(token)
        observe_ppo._qwen38_readonly_batch_audit = True
        losses.ppo_loss = observe_ppo
        loaded = sys.modules.get('verl.workers.engine_workers')
        if loaded is not None and loaded.ppo_loss is original_ppo:
            loaded.ppo_loss = observe_ppo

"""Opt-in text-only Qwen3.5/3.6 training hooks for the existing verl engine.

Inject through actor_rollout_ref.model.external_lib. No installed package is edited.
The conditional-generation parameter names are retained for RTP weight refit.
"""
from __future__ import annotations

import importlib.util
import inspect
import contextvars
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import struct
import textwrap
import time

MODEL_TYPES = {'qwen3_5', 'qwen3_5_moe'}
_INSTALLED = False
_POLICY_AUDIT_CONTEXT = contextvars.ContextVar('qwen36_policy_audit_context', default=None)


def _policy_audit_directory():
    explicit = os.environ.get('RLLM_QWEN36_POLICY_AUDIT_DIR')
    if explicit:
        return Path(explicit)
    # The dedicated model-audit destination itself opts into run auditing and
    # is explicitly forwarded to every Ray worker by the existing launcher.
    model_audit = os.environ.get('RLLM_QWEN36_MODEL_AUDIT_DIR')
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


def _observed_batch_context(data):
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
    for key in ('global_step', 'global_steps', 'training_step'):
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
    return observed


def install_policy_loss_audit():
    """Observe actual loss inputs and outputs without changing their graph."""
    import torch
    import torch.distributed as dist
    import verl.trainer.ppo.core_algos as core
    import verl.workers.utils.losses as losses

    for mode, original in list(core.POLICY_LOSS_REGISTRY.items()):
        if getattr(original, '_qwen36_readonly_policy_audit', False):
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
                record = dict(schema='qwen36-real-policy-loss-v1', pid=os.getpid(),
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
            audited._qwen36_readonly_policy_audit = True
            return audited

        core.POLICY_LOSS_REGISTRY[mode] = make_wrapper(original, mode, signature)

    original_ppo = losses.ppo_loss
    if not getattr(original_ppo, '_qwen36_readonly_batch_audit', False):
        @functools.wraps(original_ppo)
        def observe_ppo(config, model_output, data, *args, **kwargs):
            if _policy_audit_directory() is None:
                return original_ppo(config, model_output, data, *args, **kwargs)
            token = _POLICY_AUDIT_CONTEXT.set(_observed_batch_context(data))
            try:
                return original_ppo(config, model_output, data, *args, **kwargs)
            finally:
                _POLICY_AUDIT_CONTEXT.reset(token)
        observe_ppo._qwen36_readonly_batch_audit = True
        losses.ppo_loss = observe_ppo
        loaded = sys.modules.get('verl.workers.engine_workers')
        if loaded is not None and loaded.ppo_loss is original_ppo:
            loaded.ppo_loss = observe_ppo


def export_dtype(name):
    import torch
    return torch.float32 if name.endswith(('.linear_attn.A_log', '.linear_attn.dt_bias')) else torch.bfloat16


def install_export_dtype_hook():
    import verl.workers.engine.fsdp.transformer_impl as engine
    original = engine.FSDPEngine.get_per_tensor_param
    if getattr(original, '_qwen36_preserves_gdn_fp32', False):
        return
    source = textwrap.dedent(inspect.getsource(original))
    pattern = '.full_tensor().to(torch.bfloat16, non_blocking=True)'
    if source.count(pattern) != 1:
        raise RuntimeError('Unexpected verl export implementation; review the GDN FP32 export hook')
    source = source.replace(pattern, '.full_tensor().to(_qwen36_export_dtype(name), non_blocking=True)')
    namespace = dict(original.__globals__)
    namespace['_qwen36_export_dtype'] = export_dtype
    exec(compile(source, '<qwen36-verl-fp32-export-overlay>', 'exec'), namespace)
    patched = namespace[original.__name__]

    def export(self, *args, **kwargs):
        config = getattr(self.model_config, 'hf_config', None)
        if getattr(config, 'model_type', None) in MODEL_TYPES:
            return patched(self, *args, **kwargs)
        return original(self, *args, **kwargs)

    export._qwen36_preserves_gdn_fp32 = True
    engine.FSDPEngine.get_per_tensor_param = export


def configure_text_only(config):
    if getattr(config, 'model_type', None) not in MODEL_TYPES:
        raise ValueError(f'Expected Qwen3.5/3.6, got {getattr(config, "model_type", None)}')
    config.use_cache = False
    for subconfig in (config, getattr(config, 'text_config', None)):
        if subconfig is None:
            continue
        subconfig.use_cache = False
        for attr in ('mtp_num_hidden_layers', 'num_nextn_predict_layers'):
            if hasattr(subconfig, attr):
                setattr(subconfig, attr, 0)
        if hasattr(subconfig, 'output_router_logits'):
            subconfig.output_router_logits = False
        # Correctness first: no grouped-expert or Hub-kernel dependencies needed.
        subconfig._experts_implementation = os.environ.get('RLLM_QWEN36_EXPERTS_IMPL', 'eager')
    return config


def prepare_text_only(model, checkpoint_load_mode=None):
    configure_text_only(model.config)
    visual_numel = mtp_numel = trainable_numel = 0
    for name, parameter in model.named_parameters():
        parameter._qwen36_parameter_name = name
        if name.startswith('model.visual.') or '.visual.' in name:
            parameter.requires_grad_(False)
            visual_numel += parameter.numel()
        elif name.startswith('mtp.') or '.mtp.' in name:
            parameter.requires_grad_(False)
            mtp_numel += parameter.numel()
        elif parameter.requires_grad:
            trainable_numel += parameter.numel()
    if trainable_numel == 0:
        raise RuntimeError('The text-only Qwen actor has no trainable parameters')
    model._qwen36_text_only = True
    from qwen397_streamed_checkpoint import observed_node_ip
    master_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    audit = {'schema': 'qwen36-text-only-load-v1', 'pid': os.getpid(),
             'hostname': socket.gethostname(), 'node_ip': observed_node_ip(),
             'master_dtype': master_dtypes[0] if len(master_dtypes) == 1 else None,
             'master_dtypes': master_dtypes,
             'model_class': type(model).__name__, 'model_type': model.config.model_type,
             'checkpoint_load_mode': checkpoint_load_mode,
             'pre_shard_parameter_devices': sorted({str(p.device) for p in model.parameters()}),
             'text_trainable_numel': trainable_numel, 'visual_frozen_numel': visual_numel,
             'mtp_frozen_numel': mtp_numel, 'use_cache': False,
             'mtp_layers': getattr(model.config.get_text_config(), 'mtp_num_hidden_layers', 0),
             'fla_available': importlib.util.find_spec('fla') is not None,
             'causal_conv1d_available': importlib.util.find_spec('causal_conv1d') is not None,
             'parameter_prefix_preserved': 'model.language_model', 'time_ns': time.time_ns()}
    try:
        import torch
        import torch.distributed as dist
        audit.update(rank=dist.get_rank() if dist.is_initialized() else 0,
                     world_size=dist.get_world_size() if dist.is_initialized() else 1)
        if torch.cuda.is_initialized():
            properties = torch.cuda.get_device_properties(torch.cuda.current_device())
            audit['gpu_uuid'] = str(getattr(properties, 'uuid', 'unknown'))
            audit['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    except ImportError:
        audit.update(rank=0, world_size=1)
    target = os.environ.get('RLLM_QWEN36_MODEL_AUDIT_DIR')
    if target:
        path = Path(target)
        path.mkdir(parents=True, exist_ok=True)
        (path/f'model-load-rank{audit["rank"]}-pid{os.getpid()}.json').write_text(json.dumps(audit, indent=2)+'\n')
    print('QWEN36_TEXT_ONLY_LOAD', json.dumps(audit), flush=True)
    return model


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import verl.utils.model as model_utils
    import verl.models.transformers.monkey_patch as monkey_patch

    original_factory = model_utils.get_hf_auto_model_class
    original_patch = monkey_patch.apply_monkey_patch

    def factory(hf_config):
        base = original_factory(hf_config)
        if getattr(hf_config, 'model_type', None) not in MODEL_TYPES:
            return base
        configure_text_only(hf_config)

        class TextOnlyAutoModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                import torch
                import torch.distributed as dist

                from qwen397_streamed_checkpoint import is_qwen397, meta_checkpoint_model
                if is_qwen397(hf_config):
                    return prepare_text_only(meta_checkpoint_model(base, hf_config, *args, **kwargs),
                                             'meta_sharded_checkpoint_slice_world32')
                text = hf_config.get_text_config()
                is_target = (hf_config.model_type == 'qwen3_5_moe' and text.num_hidden_layers == 48
                             and text.hidden_size == 3072 and text.num_experts == 256)
                if not (is_target and dist.is_initialized() and dist.get_world_size() == 8):
                    return prepare_text_only(base.from_pretrained(*args, **kwargs))
                if dist.get_rank() == 0:
                    model = base.from_pretrained(*args, **kwargs)
                    if any(p.device.type != 'cpu' for p in model.parameters()):
                        raise RuntimeError('Rank 0 must hold its full unsharded checkpoint on CPU')
                    return prepare_text_only(model, 'checkpoint_cpu_rank0')
                # Our pinned FSDP2 engine broadcasts rank 0's checkpoint one
                # tensor at a time after wrapping. Other ranks need only the
                # module structure here. Skipping their HF weight reads saves
                # seven full FP32 copies and prevents startup RSS exhaustion.
                with torch.device('meta'):
                    model = base.from_config(configure_text_only(hf_config),
                                             dtype=kwargs.get('torch_dtype', torch.float32))
                if any(p.device.type != 'meta' for p in model.parameters()):
                    raise RuntimeError('Nonzero ranks must start with sharded-broadcast meta parameters')
                return prepare_text_only(model, 'meta_receiver_from_rank0')

            @classmethod
            def from_config(cls, config, **kwargs):
                return prepare_text_only(base.from_config(configure_text_only(config), **kwargs))

        return TextOnlyAutoModel

    def patch(model, ulysses_sp_size=1, use_remove_padding=True, use_fused_kernels=False, **kwargs):
        if getattr(model.config, 'model_type', None) not in MODEL_TYPES:
            return original_patch(model, ulysses_sp_size=ulysses_sp_size,
                                  use_remove_padding=use_remove_padding,
                                  use_fused_kernels=use_fused_kernels, **kwargs)
        if use_remove_padding or ulysses_sp_size != 1 or use_fused_kernels:
            raise ValueError('Qwen3.5 acceptance requires padded/SP1/unfused forward; validate packed GDN separately')
        # Native HF text forwarding handles plain text without running the vision
        # tower. Avoid installing verl's multimodal dummy-vision forward here.
        if not getattr(model, '_qwen36_text_only', False):
            prepare_text_only(model)
        return None

    model_utils.get_hf_auto_model_class = factory
    monkey_patch.apply_monkey_patch = patch
    # The FSDP module may already have imported the function by name.
    loaded = sys.modules.get('verl.workers.engine.fsdp.transformer_impl')
    if loaded is not None:
        loaded.apply_monkey_patch = patch
    install_export_dtype_hook()
    from qwen397_streamed_checkpoint import install_streamed_loader
    install_streamed_loader()
    install_policy_loss_audit()
    _INSTALLED = True


install()

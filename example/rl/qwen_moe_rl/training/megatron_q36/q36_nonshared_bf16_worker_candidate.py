"""Opt-in verl injection for the private nonshared BF16/SR factory.

An independently qualified live 35B logical coordinate mapper is mandatory;
this module does not implement one. Install before creating the optimizer.
"""
import functools
import hashlib
import inspect
from pathlib import Path

SOURCE_HASHES = {
    'bf16_hybrid_adam_candidate.py': '4c530ee5c00b391ffbd5f67c1645cfe423bf64d6f295f9c3dea507088956b71b',
    'bf16_streaming_optimizer_candidate.py': '0a60bc34d8936858fa8f77a5bccd0f9086b3897724380677886eb12855bcc1a2',
    'bf16_finite_candidate.py': 'bcc6b280bba286981b7062c72895a8945ff7ed492a64f2ec1738bef5eaadc3f8',
    'bf16_route_activity_candidate.py': '7b43ebf38c906e3188db8e62cf141fe46ff956fa6b0745ebba6f21035d7f2cbc',
    'bf16_nonsharded_megatron_candidate.py': '51071aab338b03d98358aba6a06ea73fb5b36e82b4f4b4148ef88593f80ac189',
    'bf16_sr_coordinates_candidate.py': '52e6d187b08b93e83fd56e3155eff6c904484e306a409d3bd62d25ec86b7e47c',
    'bf16_streaming_optimizer_coords_candidate.py': '2309225fa7feaaaa4a0f553f1017ab795e3a8b1f2f6b9337f63a77aa3c3348da',
    'bf16_megatron_factory_nonshared_candidate.py': '413fc97ed9c261ee2f37c72eba7d7d1b31bcda4353c99bf26100b3ee03c40361',
    'q36_bounded_clip_candidate.py': '9deb41fbc628fbff1e621a41dcf46c7237aa9114a2be87b5dcc7bd469c2dc8ef',
    'bf16_fixed_tp4_coordinates_candidate.py': 'cbd5d0763541736aea21439121175a21e3ae92e854e3cf2a2b23ec8a15dbc3f8',
    'bf16_megatron_factory_fixed_tp4_candidate.py': '2642b52ecc5870b60775e064460c98adde3ff2c5fffa5f148bcac0be1792bb5b',
    'q36_fixed_tp4_sr_binder_ddp_candidate.py': 'cb5593ab33c08306bbd299b2832188db8de64ff72310330f4eeb80e2fcd244a2',
}


def verify_candidate_sources(directory):
    for name, expected in SOURCE_HASHES.items():
        path = Path(directory) / name
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'nonshared BF16 candidate source mismatch: {name}')


def install_hooks(verl_optim, core_optim, factory_candidate, torch_module, dist_module, coordinate_mapper):
    """Patch verl's saved factory alias; mapper is required on every rank.

    A world vote ensures a mapper failure on one rank does not leave peers
    entering the patched optimizer's collective preflight alone. The real
    mapper must be based on the live model and independently qualified source
    coordinates. This function cannot prove mapper correctness.
    """
    if not callable(coordinate_mapper):
        raise TypeError('a qualified 35B logical-coordinate mapper is required')
    existing = verl_optim.init_megatron_optim_config
    if getattr(existing, '_q36_nonshared_bf16_injection', False):
        raise RuntimeError('nonshared BF16 worker hooks are already installed')
    signature = inspect.signature(existing)
    factory_candidate.install()
    if not getattr(core_optim, '_q38_candidate_installed', False):
        raise RuntimeError('candidate failed to install MCore optimizer factory')
    patched_native = core_optim.get_megatron_optimizer

    @functools.wraps(existing)
    def initialize(*args, **kwargs):
        bound = signature.bind(*args, **kwargs); bound.apply_defaults()
        if (bound.arguments['use_distributed_optimizer'] is not False or
                bound.arguments['fp16'] is not False or bound.arguments['bf16'] is not True):
            raise ValueError('requires nonsharded BF16 Megatron optimizer')
        config = existing(*args, **kwargs)
        if not (config.optimizer == 'adam' and config.bf16 and not config.use_distributed_optimizer
                and not config.use_precision_aware_optimizer and config.optimizer_cpu_offload
                and config.optimizer_offload_fraction == 1.0
                and not config.overlap_cpu_optimizer_d2h_h2d
                and not config.overlap_param_gather_with_optimizer_step):
            raise ValueError('optimizer does not match qualified BF16 candidate configuration')
        config.main_params_dtype = config.main_grads_dtype = torch_module.bfloat16
        config.exp_avg_dtype = config.exp_avg_sq_dtype = torch_module.bfloat16
        config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 = True
        config.q38_bf16_hdo = config.q38_nonsharded_bf16 = True
        return config

    @functools.wraps(patched_native)
    def native(config, model_chunks, *args, **kwargs):
        error = None; bindings = None
        try:
            bindings = coordinate_mapper(model_chunks)
            parameters = {id(p) for chunk in model_chunks for p in chunk.parameters() if p.requires_grad}
            if not parameters or not isinstance(bindings, dict) or set(bindings) != parameters:
                raise ValueError('logical SR coordinates do not cover exactly the live trainable parameters')
        except Exception as exc:
            error = exc
        ready = torch_module.tensor(int(error is None), dtype=torch_module.int32,
                                    device=torch_module.cuda.current_device())
        dist_module.all_reduce(ready, op=dist_module.ReduceOp.MIN)
        if not bool(ready.item()):
            raise RuntimeError('all-rank 35B logical SR coordinate preflight failed') from error
        config.q38_sr_bindings = bindings
        optimizer = patched_native(config, model_chunks, *args, **kwargs)
        from q36_update_witness import attach
        return attach(optimizer, model_chunks, torch_module, dist_module)

    initialize._q36_nonshared_bf16_injection = True
    native._q36_nonshared_bf16_injection = True
    verl_optim.get_megatron_optimizer_native = native
    verl_optim.init_megatron_optim_config = initialize

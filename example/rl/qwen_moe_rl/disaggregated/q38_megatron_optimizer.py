"""Install the BF16 streaming Adam path for the loaded TP8/EP64 actor."""

import functools
import inspect

from q38_megatron_loader import TRAIN_LOCAL_EXPERTS, capture_groups, _world_gate


def optimizer_coordinates(binding, groups):
    from bf16_sr_coordinates_candidate import SRCoordinates

    parameter = binding.parameter
    if binding.expert is not None:
        coordinate = binding.expert
        begin = groups.ep_rank*TRAIN_LOCAL_EXPERTS
        if not begin <= coordinate.global_expert < begin+TRAIN_LOCAL_EXPERTS:
            raise ValueError(f"expert update identity differs from EP ownership at {binding.name}")
        key = f"{coordinate.hf_name}/global-expert-{coordinate.global_expert}"
    elif getattr(parameter, "tensor_model_parallel", False):
        key = f"{binding.name}/tp8-shard-{groups.tp_rank}"
    else:
        key = f"{binding.name}/tp8-replica"
    shape = tuple(parameter.shape)
    return SRCoordinates(key, shape, shape).validate()


def _bind(chunks):
    import torch
    from megatron.bridge.models.conversion.utils import unwrap_model
    from megatron.core.distributed import DistributedDataParallel

    groups = capture_groups()
    model = unwrap_model(chunks)
    if len(chunks) != 1 or len(model) != 1 or not isinstance(chunks[0], DistributedDataParallel):
        raise ValueError("expected one live Megatron DDP actor")
    config = chunks[0].ddp_config
    if config.grad_reduce_in_fp32 or config.use_distributed_optimizer or config.use_megatron_fsdp:
        raise ValueError("expected BF16 DDP main gradients and nonsharded optimizer ownership")
    dense, experts = model[0]._q38_import_bindings
    expected = {id(p) for p in chunks[0].parameters() if p.requires_grad}
    actual = {id(binding.parameter) for binding in dense+experts}
    if actual != expected or len(actual) != len(dense)+len(experts):
        raise ValueError("optimizer ownership differs from the verified checkpoint destinations")
    for binding in dense+experts:
        parameter = binding.parameter
        gradient = getattr(parameter, "main_grad", None)
        if (parameter.dtype != torch.bfloat16 or not parameter.is_cuda
                or gradient is None or gradient.dtype != torch.bfloat16
                or gradient.device != parameter.device or gradient.shape != parameter.shape):
            raise ValueError(f"live BF16 parameter/main gradient mismatch at {binding.name}")
        parameter._q38_sr_coordinates = optimizer_coordinates(binding, groups)


def install_for_verl():
    import torch
    import torch.distributed as dist
    from megatron.core import optimizer as core_optimizer
    import verl.utils.megatron.optimizer as verl_optimizer
    import bf16_megatron_factory_nonshared_candidate as factory
    from bf16_streaming_optimizer_coords_candidate import StreamingBf16AdamWCoordinates
    from q36_update_witness import attach

    initialize = verl_optimizer.init_megatron_optim_config
    if getattr(initialize, "_q38_ep64_optimizer", False):
        return
    if getattr(core_optimizer, "_q38_candidate_installed", False):
        raise RuntimeError("a different BF16 Megatron optimizer is already installed")
    signature = inspect.signature(initialize)
    factory.StreamingBf16AdamW = StreamingBf16AdamWCoordinates
    factory.install()
    get_optimizer = core_optimizer.get_megatron_optimizer

    @functools.wraps(initialize)
    def initialize_bf16(*args, **kwargs):
        values = signature.bind(*args, **kwargs)
        values.apply_defaults()
        if (values.arguments["use_distributed_optimizer"] is not False
                or values.arguments["fp16"] is not False or values.arguments["bf16"] is not True):
            raise ValueError("expected the nonsharded BF16 Megatron optimizer configuration")
        config = initialize(*args, **kwargs)
        if config.use_precision_aware_optimizer or not config.optimizer_cpu_offload:
            raise ValueError("expected CPU moment storage under the streaming BF16 optimizer")
        config.main_params_dtype = config.main_grads_dtype = torch.bfloat16
        config.exp_avg_dtype = config.exp_avg_sq_dtype = torch.bfloat16
        config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 = True
        config.q38_bf16_hdo = config.q38_nonsharded_bf16 = True
        factory._check_config(config)
        return config

    @functools.wraps(get_optimizer)
    def build_bf16(config, model_chunks, *args, **kwargs):
        error = None
        try:
            _bind(model_chunks)
        except Exception as exc:
            error = exc
        _world_gate(error, "BF16 optimizer ownership")
        optimizer = get_optimizer(config, model_chunks, *args, **kwargs)
        return attach(optimizer, model_chunks, torch, dist)

    initialize_bf16._q38_ep64_optimizer = True
    verl_optimizer.get_megatron_optimizer_native = build_bf16
    verl_optimizer.init_megatron_optim_config = initialize_bf16

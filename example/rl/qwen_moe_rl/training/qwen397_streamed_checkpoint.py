"""Load the pinned BF16 checkpoint directly into 32 FP32 FSDP2 shards.

No rank constructs a full CPU/GPU model. FSDP wraps meta parameters first;
safetensors then reads just the interval owned by each real DTensor shard.
The accepted CPU-moment AdamW and name-dependent RTP export are unchanged.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import resource
import socket
import sys
import time


WORLD_SIZE = 32
REVISION = '8472618112abcbd45acbcdc58436aff4233c23f7'
CONFIG_SHA256 = '3ae7fa89c2f7d1354096418ddaf1331e9e0898a8ca11e804fb6dbaa087efb7da'
INDEX_SHA256 = '407d6a184a29469034ad92bcfaa6b72b582f8ad6afcbfad18a38a1c59ed296f8'
TEXT_NUMEL = 396346350336
VISUAL_NUMEL = 456010480


def observed_node_ip():
    try:
        import ray
        if ray.is_initialized():
            return ray.util.get_node_ip_address()
    except ImportError:
        pass
    return socket.gethostbyname(socket.gethostname())


def is_qwen397(config):
    text = config.get_text_config()
    return (config.model_type == 'qwen3_5_moe' and text.num_hidden_layers == 60
            and text.hidden_size == 4096 and text.num_experts == 512)


def validate_checkpoint_identity(checkpoint):
    checkpoint = Path(checkpoint).resolve(strict=True)
    for name, expected in [('config.json', CONFIG_SHA256),
                           ('model.safetensors.index.json', INDEX_SHA256)]:
        actual = hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Pinned official 397B {name} SHA256 differs: {actual}')
    return checkpoint


def validate_torch_api():
    import torch
    from packaging.version import Version
    from torch.distributed.tensor import DTensor, Shard
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    version = Version(torch.__version__.split('+')[0])
    if not Version('2.11') <= version < Version('2.14'):
        raise RuntimeError(f'Review 397B checkpoint streaming with Torch {torch.__version__}; expected 2.11–2.13')
    if not callable(compute_local_shape_and_global_offset) or not hasattr(DTensor, 'to_local'):
        raise RuntimeError('Required DTensor shard/offset API is unavailable')


def bind_meta_checkpoint_call(factory_config, args, kwargs):
    """Bind the 397B-only HF call without dropping conflicting model options."""
    options = dict(kwargs)
    supplied_config = options.pop('config', factory_config)
    if supplied_config is not factory_config:
        raise ValueError('397B from_pretrained config must be the same instance selected by the model factory')
    if len(args) > 1:
        raise TypeError('397B meta initialization accepts only the checkpoint path as a positional model argument')
    if args and 'pretrained_model_name_or_path' in options:
        raise TypeError('397B checkpoint path was supplied both positionally and by keyword')
    path = args[0] if args else options.pop('pretrained_model_name_or_path', None)
    if path is None:
        raise ValueError('A pinned local checkpoint path is required')
    unexpected = set(options) - {'torch_dtype', 'trust_remote_code'}
    if unexpected:
        raise TypeError(f'Unsupported 397B meta initialization options: {sorted(unexpected)}')
    return path, options


def meta_checkpoint_model(auto_class, factory_config, *args, **kwargs):
    """Construct meta parameters and real, small initialized CPU buffers."""
    # verl passes config=... to HF from_pretrained. It must not collide with
    # this helper's already positional factory config, nor be silently ignored.
    path, options = bind_meta_checkpoint_call(factory_config, args, kwargs)
    config = factory_config
    import torch
    import torch.distributed as dist
    from accelerate import init_empty_weights

    validate_torch_api()
    if not dist.is_initialized() or dist.get_world_size() != WORLD_SIZE:
        raise ValueError('397B training must initialize all 32 FSDP ranks before constructing the model')
    checkpoint = validate_checkpoint_identity(path)
    if options.get('torch_dtype', torch.float32) != torch.float32:
        raise ValueError('397B training masters must be FP32; BF16 remains the forward/rollout dtype')
    if config.get_text_config().mtp_num_hidden_layers != 0:
        raise ValueError('The original text-only path disables MTP before initialization')
    # Parameter-only meta initialization is essential: torch.device(meta) also
    # destroys the initialized nonpersistent rotary-buffer values.
    with torch.device('cpu'), init_empty_weights(include_buffers=False):
        model = auto_class.from_config(config, dtype=torch.float32,
                                      trust_remote_code=options.get('trust_remote_code', False))
    if any(parameter.device.type != 'meta' for parameter in model.parameters()):
        raise RuntimeError('An unsharded 397B parameter escaped meta initialization')
    if any(buffer.device.type != 'cpu' for buffer in model.buffers()):
        raise RuntimeError('All initialized nonpersistent buffers must remain materialized on CPU')
    model._qwen397_checkpoint_path = str(checkpoint)
    model._qwen397_initial_parameter_devices = sorted({str(p.device) for p in model.parameters()})
    return model


def shard_interval(size, world_size, coordinate):
    """Match PyTorch Shard/torch.chunk semantics, including empty tail ranks."""
    if size < 0 or world_size < 1 or not 0 <= coordinate < world_size:
        raise ValueError('Invalid shard extent or coordinate')
    chunk = math.ceil(size/world_size)
    start = min(coordinate*chunk, size)
    return start, min(start+chunk, size)


def source_slice(parameter, global_shape, expected_world=WORLD_SIZE):
    """Validate real placement/offsets and return the source shard selection."""
    from torch.distributed.tensor import DTensor, Shard
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    if not isinstance(parameter, DTensor):
        raise TypeError('Every parameter, including frozen vision, must be FSDP-sharded before loading')
    mesh, placements = parameter.device_mesh, parameter.placements
    if mesh.ndim != 1 or mesh.size() != expected_world or len(placements) != 1:
        raise ValueError(f'Expected a one-dimensional FSDP{expected_world} mesh')
    placement = placements[0]
    if type(placement) is not Shard or placement.dim != 0:
        raise ValueError(f'Expected ordinary first-axis FSDP sharding, got {placements}')
    if list(parameter.shape) != list(global_shape):
        raise ValueError('Runtime global parameter shape differs from the official checkpoint')
    coordinate = mesh.get_coordinate()[0]
    begin, end = shard_interval(global_shape[0], expected_world, coordinate)
    shape, offset = compute_local_shape_and_global_offset(global_shape, mesh, placements)
    expected_shape = (end-begin, *global_shape[1:])
    expected_offset = (begin, *(0 for _ in global_shape[1:]))
    if tuple(shape) != expected_shape or tuple(offset) != expected_offset:
        raise RuntimeError(f'DTensor offset API disagrees with its first-axis shard: {shape}, {offset}')
    if tuple(parameter.detach().to_local().shape) != expected_shape:
        raise RuntimeError('Actual DTensor local storage differs from its checkpoint interval')
    return (slice(begin, end), *(slice(None) for _ in global_shape[1:])), begin, end


def capture_buffers(model):
    """Keep small buffer values and plain attribute aliases across to_empty()."""
    import torch

    saved, ids, aliases = {}, {}, []
    for name, buffer in model.named_buffers():
        if buffer.is_meta:
            raise ValueError(f'Initialized buffer was discarded on meta: {name}')
        saved[name] = buffer.detach().cpu().clone()
        ids[id(buffer)] = name
    for module_name, module in model.named_modules():
        for attr, value in vars(module).items():
            if isinstance(value, torch.Tensor) and id(value) in ids:
                aliases.append((module_name, attr, ids[id(value)]))
    return saved, aliases


def restore_buffers(model, saved, aliases):
    import torch

    current = dict(model.named_buffers())
    if set(current) != set(saved):
        raise RuntimeError('Buffer names changed during FSDP materialization')
    with torch.no_grad():
        for name, buffer in current.items():
            buffer.copy_(saved[name])
            if not torch.equal(buffer.detach().cpu(), saved[name]):
                raise RuntimeError(f'Initialized buffer values differ after materialization: {name}')
    modules = dict(model.named_modules())
    for module_name, attr, buffer_name in aliases:
        setattr(modules[module_name], attr, current[buffer_name])


def copy_checkpoint_parameter(parameter, source, name, expected_world=WORLD_SIZE):
    """Read/cast only one local BF16/F32 slice, and verify sampled real values."""
    import torch

    shape = source.get_shape()
    selection, begin, end = source_slice(parameter, shape, expected_world)
    # Do not create a differentiable ToTorchTensor custom-function view before
    # writing checkpoint values. Its view/version guard rejects later reads
    # after an in-place copy even when copy_ itself runs under no_grad.
    # Detach shares the actual shard storage; it does not clone the parameter.
    local = parameter.detach().to_local()
    if local.dtype != torch.float32:
        raise ValueError(f'{name}: expected an FP32 training master shard')
    part = source[selection]
    if tuple(part.shape) != tuple(local.shape) or part.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f'{name}: checkpoint slice shape/dtype does not match the real local shard')
    source_bytes = part.numel()*part.element_size()
    count = min(part.numel(), 64)
    indices = torch.arange(count, dtype=torch.int64)
    if count > 1:
        indices = indices*(part.numel()-1)//(count-1)
    flat = part.reshape(-1)
    expected = flat[indices].float().numpy().tobytes()
    if not torch.isfinite(flat[indices].float()).all().item():
        raise RuntimeError(f'{name}: checkpoint sample contains nonfinite values')
    with torch.no_grad():
        local.copy_(part, non_blocking=False)
    observed = local.reshape(-1)[indices.to(local.device)].float().cpu().numpy().tobytes()
    if expected != observed:
        raise RuntimeError(f'{name}: real loaded master coordinates differ from the official checkpoint')
    return dict(name=name, global_shape=list(shape), shard_dim=0, shard_begin=begin, shard_end=end,
                local_shape=list(local.shape), local_numel=local.numel(), local_bytes=local.numel()*4,
                target_dtype=str(local.dtype), source_dtype=str(part.dtype), source_slice_bytes=source_bytes,
                source_sample_sha256=hashlib.sha256(expected).hexdigest(), sample_count=count,
                sample_coordinates_match=True)


def _rss_peak_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == 'darwin' else value*1024


def checkpoint_communication_preflight(device):
    """Verify the existing real training group before any checkpoint weight open."""
    import torch
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() != WORLD_SIZE:
        raise ValueError('Checkpoint communication preflight requires the existing real FSDP32 process group')
    rank = dist.get_rank()
    configured_gid = os.environ.get('NCCL_IB_GID_INDEX')
    expected_gid = os.environ.get('Q397_NCCL_IB_GID_INDEX', '3')
    if configured_gid != expected_gid:
        raise RuntimeError(f'Training Ray worker RoCE GID was not forwarded: expected {expected_gid}, got {configured_gid}')
    if os.environ.get('NCCL_IB_DISABLE', '0') != '0':
        raise RuntimeError('The verified training transport keeps IB/RoCE enabled; NCCL_IB_DISABLE must be unset or 0')
    from q397_nccl_scope import inspect_pid_nccl
    nccl_scope = inspect_pid_nccl(torch_module=torch)
    started = time.time()
    print('Q397_CHECKPOINT_COMM_PREFLIGHT_START', rank, configured_gid, flush=True)
    coverage = torch.tensor([rank+1, 1], device=device, dtype=torch.int64)
    dist.all_reduce(coverage)
    torch.cuda.synchronize(device)
    observed = coverage.cpu().tolist()
    if observed != [528, 32]:
        raise RuntimeError(f'The real 32-rank CUDA communication preflight differs: {observed}')
    properties = torch.cuda.get_device_properties(device)
    audit = dict(schema='qwen397-pre-checkpoint-communication-v1', passed=True,
                 rank=rank, world_size=WORLD_SIZE, hostname=socket.gethostname(),
                 node_ip=observed_node_ip(), pid=os.getpid(), torch_version=torch.__version__,
                 cuda_version=torch.version.cuda, nccl_version=torch.cuda.nccl.version(),
                 nccl_version_kind='compile_time_header', nccl_scope=nccl_scope,
                 gpu_uuid=str(getattr(properties, 'uuid', 'unknown')),
                 cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                 nccl_ib_gid_index=configured_gid, nccl_ib_disable=os.environ.get('NCCL_IB_DISABLE'),
                 gid_evidence='Explicit NCCL_IB_GID_INDEX; actual RoCE path independently verified by the small Ray32 probe',
                 shard_weight_files_opened_before_preflight=0,
                 collective='existing default process group CUDA int64 all_reduce',
                 coverage=observed, started_unix=started, finished_unix=time.time())
    directory = Path(os.environ['RLLM_QWEN36_MODEL_AUDIT_DIR'])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'comm-preflight-rank{rank}.json'
    with path.open('x') as stream:
        json.dump(audit, stream, separators=(',', ':'))
        stream.write('\n')
    print('Q397_CHECKPOINT_COMM_PREFLIGHT_READY', rank, configured_gid,
          audit['nccl_version'], observed, flush=True)


def load_checkpoint_shards(model, full_state, device_mesh=None, cpu_offload=None):
    import torch
    import torch.distributed as dist
    from safetensors import safe_open

    validate_torch_api()
    if not is_qwen397(model.config) or not hasattr(model, '_qwen397_checkpoint_path'):
        raise ValueError('397B streamed initialization received another model or an unpinned source')
    if not dist.is_initialized() or dist.get_world_size() != WORLD_SIZE:
        raise ValueError('397B checkpoint loading requires all 32 initialized FSDP ranks')
    if device_mesh is None or device_mesh.ndim != 1 or device_mesh.size() != WORLD_SIZE:
        raise ValueError('397B checkpoint loading requires an actual one-dimensional FSDP32 mesh')
    if cpu_offload is not None:
        raise ValueError('FP32 training master/gradient shards must remain on their GPU')
    if any(not tensor.is_meta for tensor in full_state.values()):
        raise RuntimeError('Pre-FSDP full state must contain only meta tensors, never complete CPU weights')
    checkpoint = validate_checkpoint_identity(model._qwen397_checkpoint_path)
    mapping = json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    expected = {name for name in mapping if name.startswith(('model.language_model.', 'model.visual.'))
                or name == 'lm_head.weight'}
    parameters = dict(model.named_parameters())
    if set(parameters) != expected:
        raise ValueError('Runtime parameter names differ from the complete official text+vision checkpoint: '
                         f'missing={sorted(expected-set(parameters))[:10]}, extra={sorted(set(parameters)-expected)[:10]}')
    text_numel = sum(p.numel() for name, p in parameters.items() if not name.startswith('model.visual.'))
    vision_numel = sum(p.numel() for name, p in parameters.items() if name.startswith('model.visual.'))
    if text_numel != TEXT_NUMEL or vision_numel != VISUAL_NUMEL:
        raise ValueError('397B runtime global parameter counts differ from the official meta contract')
    saved_buffers, aliases = capture_buffers(model)
    device = torch.device('cuda', torch.cuda.current_device())
    if os.environ.get('Q397_REQUIRE_COMM_PREFLIGHT') == '1':
        checkpoint_communication_preflight(device)
    started = time.time()
    model.to_empty(device=device)
    records = []
    for name, parameter in model.named_parameters():
        local = parameter.detach().to_local()
        if local.device != device or local.dtype != torch.float32:
            raise ValueError(f'{name}: expected local CUDA FP32 master storage')
        with safe_open(str(checkpoint/mapping[name]), framework='pt', device='cpu') as shard:
            record = copy_checkpoint_parameter(parameter, shard.get_slice(name), name)
        parameter._qwen36_parameter_name = name
        record.update(checkpoint_file=mapping[name], requires_grad=bool(parameter.requires_grad))
        records.append(record)
        if len(records) == 1 or len(records) % 60 == 0:
            print('Q397_CHECKPOINT_SLICE_LOADED', dist.get_rank(), len(records), len(parameters), flush=True)
    restore_buffers(model, saved_buffers, aliases)
    text_source_count = sum(row['name'] == 'lm_head.weight'
                            or row['name'].startswith('model.language_model.') for row in records)
    visual_source_count = sum(row['name'].startswith('model.visual.') for row in records)
    if (text_source_count, visual_source_count) != (1038, 333):
        raise RuntimeError('Actual loaded 397B text/vision tensor counts differ from the official contract')
    local_numel = sum(row['local_numel'] for row in records)
    trainable_local_numel = sum(row['local_numel'] for row in records if row['requires_grad'])
    total = torch.tensor([local_numel, trainable_local_numel], device=device, dtype=torch.int64)
    dist.all_reduce(total)
    if total.tolist() != [TEXT_NUMEL+VISUAL_NUMEL, TEXT_NUMEL]:
        raise RuntimeError('The real 32-rank local shards do not cover the complete model exactly once')
    dist.barrier()
    directory = Path(os.environ['RLLM_QWEN36_MODEL_AUDIT_DIR'])
    directory.mkdir(parents=True, exist_ok=True)
    properties = torch.cuda.get_device_properties(device)
    audit = dict(schema='qwen397-streamed-checkpoint-load-v2', hostname=socket.gethostname(),
                 node_ip=observed_node_ip(), pid=os.getpid(), rank=dist.get_rank(),
                 world_size=dist.get_world_size(), torch_version=torch.__version__,
                 gpu_uuid=str(getattr(properties, 'uuid', 'unknown')),
                 model_id='Qwen/Qwen3.5-397B-A17B', model_revision=REVISION, checkpoint_path=str(checkpoint),
                 config_sha256=CONFIG_SHA256, checkpoint_index_sha256=INDEX_SHA256,
                 source_tensor_count=len(records), text_source_count=text_source_count,
                 visual_source_count=visual_source_count,
                 loaded_local_numel=local_numel, loaded_local_bytes=4*local_numel,
                 trainable_local_numel=trainable_local_numel, shard_coverage_allreduce=total.tolist(),
                 maximum_checkpoint_slice_bytes=max(row['source_slice_bytes'] for row in records),
                 pre_shard_parameter_devices=model._qwen397_initial_parameter_devices,
                 rank0_full_model_materialized=False,
                 buffers_verified=True, buffer_count=len(saved_buffers), buffer_alias_count=len(aliases),
                 master_dtype='torch.float32', master_device=str(device), rss_peak_bytes=_rss_peak_bytes(),
                 source_slices=records, started_unix=started, finished_unix=time.time())
    path = directory/f'checkpoint-load-rank{dist.get_rank()}.json'
    temporary = path.with_suffix('.new')
    temporary.write_text(json.dumps(audit, separators=(',', ':'))+'\n')
    temporary.replace(path)
    model._qwen397_streaming_verified = True
    print('Q397_CHECKPOINT_SHARD_READY', dist.get_rank(), local_numel,
          audit['maximum_checkpoint_slice_bytes'], flush=True)


def install_streamed_loader():
    import verl.workers.engine.fsdp.transformer_impl as engine
    import verl.utils.fsdp_utils as fsdp_utils

    original = engine.fsdp2_load_full_state_dict
    if getattr(original, '_qwen397_checkpoint_streaming', False):
        return
    if list(inspect.signature(original).parameters) != ['model', 'full_state', 'device_mesh', 'cpu_offload']:
        raise RuntimeError('The pinned verl FSDP2 checkpoint hook signature changed; review before training')

    def load(model, *args, **kwargs):
        if is_qwen397(model.config):
            return load_checkpoint_shards(model, *args, **kwargs)
        return original(model, *args, **kwargs)

    load._qwen397_checkpoint_streaming = True
    engine.fsdp2_load_full_state_dict = load
    if fsdp_utils.fsdp2_load_full_state_dict is original:
        fsdp_utils.fsdp2_load_full_state_dict = load

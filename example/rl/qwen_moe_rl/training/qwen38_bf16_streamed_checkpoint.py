"""Pinned BF16 source slices directly into the real 72 BF16 FSDP shards.

The source map changes storage only: every original policy coordinate is loaded
once globally. No rank constructs the entire policy, and MTP is not trained.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import time
from collections import defaultdict

from qwen38_training_contract import (POLICY_NUMEL, WORLD_SIZE, load_metadata,
                                      flat_source_fragments, shard_interval)


def is_qwen38(config):
    return (getattr(config, 'model_type', None) == 'qwen3_5_moe_text'
            and getattr(config, 'num_hidden_layers', None) == 92
            and getattr(config, 'hidden_size', None) == 8192
            and getattr(config, 'num_experts', None) == 512)


def bind_meta_call(config, args, kwargs):
    options = dict(kwargs)
    if options.pop('config', config) is not config:
        raise ValueError('The selected Qwen3.8 config instance must be preserved')
    if len(args) > 1 or (args and 'pretrained_model_name_or_path' in options):
        raise ValueError('Exactly one explicit checkpoint path is required')
    checkpoint = args[0] if args else options.pop('pretrained_model_name_or_path', None)
    if checkpoint is None or set(options) - {'torch_dtype', 'dtype', 'trust_remote_code'}:
        raise ValueError('Review unsupported Qwen3.8 meta factory arguments')
    return checkpoint, options


def meta_checkpoint_model(auto_class, config, *args, **kwargs):
    import torch
    import torch.distributed as dist
    from accelerate import init_empty_weights
    from qwen38_chunked_experts import install_meta_chunks
    from qwen38_vocab_tiles import install_meta_vocab_tiles
    checkpoint, options = bind_meta_call(config, args, kwargs)
    load_metadata(checkpoint)
    if not dist.is_initialized() or dist.get_world_size() != WORLD_SIZE:
        raise RuntimeError('All 72 actual FSDP ranks must exist before model construction')
    if options.get('torch_dtype', options.get('dtype', torch.bfloat16)) != torch.bfloat16:
        raise ValueError('Training keeps one BF16 master shard, with BF16 compute')
    config.use_cache = False; config.output_router_logits = False
    config.mtp_num_hidden_layers = 0; config._experts_implementation = 'eager'
    with torch.device('cpu'), init_empty_weights(include_buffers=False):
        model = auto_class.from_config(config, dtype=torch.bfloat16,
                    trust_remote_code=options.get('trust_remote_code', False))
    if any(not p.is_meta for p in model.parameters()):
        raise RuntimeError('A policy parameter escaped meta construction')
    if any(b.device.type != 'cpu' for b in model.buffers()):
        raise RuntimeError('Initialized rotary and other small buffers must remain real CPU values')
    remember_initialized_buffers(model)
    for name, parameter in model.named_parameters():
        if parameter.dtype != torch.bfloat16:
            raise RuntimeError(f'HF created a non-BF16 persistent policy master: {name} {parameter.dtype}')
    install_meta_chunks(model)
    install_meta_vocab_tiles(model)
    # Whole-decoder recomputation keeps every FSDP expert chunk unsharded
    # until backward (48GiB of BF16 experts in one real layer). Bound each
    # checkpoint to one chunk, with separate attention/router/shared regions.
    from qwen38_bounded_checkpoint import install_bounded_checkpoints
    install_bounded_checkpoints(model)
    if sum(p.numel() for p in model.parameters()) != POLICY_NUMEL:
        raise ValueError('HF meta model differs from the pinned main-policy coordinates')
    model._qwen38_checkpoint_path = str(Path(checkpoint).resolve(strict=True))
    return model


def source_interval(parameter, expected_shape, *, world_size=WORLD_SIZE):
    from torch.distributed.tensor import DTensor, Shard
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
    if not isinstance(parameter, DTensor):
        raise TypeError('The policy must be FSDP2-sharded before checkpoint reads')
    mesh, placements = parameter.device_mesh, parameter.placements
    if (mesh.ndim != 1 or mesh.size() != world_size or len(placements) != 1
            or type(placements[0]) is not Shard or placements[0].dim != 0):
        raise ValueError('Expected the explicit 1D ordinary first-axis shard policy')
    if tuple(parameter.shape) != tuple(expected_shape):
        raise ValueError('Runtime global shape differs from its source-map contract')
    begin, end = shard_interval(expected_shape[0], world_size, mesh.get_coordinate()[0])
    shape, offset = compute_local_shape_and_global_offset(expected_shape, mesh, placements)
    if (tuple(shape) != (end-begin, *expected_shape[1:])
            or tuple(offset) != (begin, *(0 for _ in expected_shape[1:]))
            or tuple(parameter.to_local().shape) != tuple(shape)):
        raise ValueError('Actual DTensor interval disagrees with the pinned source selection')
    return begin, end


def _verify_copy(destination, part):
    import torch
    if part.dtype != torch.bfloat16 or destination.dtype != torch.bfloat16:
        raise ValueError('Read original BF16 bytes directly into persistent BF16 masters')
    if tuple(part.shape) != tuple(destination.shape):
        raise ValueError('Bounded checkpoint selection differs from its real destination')
    count = min(part.numel(), 64)
    indices = torch.arange(count, dtype=torch.int64)
    if count > 1:indices = indices * (part.numel()-1) // (count-1)
    expected = part.reshape(-1)[indices].float()
    if not torch.isfinite(expected).all().item():
        raise ValueError('A sampled official checkpoint coordinate is nonfinite')
    with torch.no_grad():destination.copy_(part, non_blocking=False)
    observed = destination.reshape(-1)[indices.to(destination.device)].float().cpu()
    if not torch.equal(expected, observed):
        raise RuntimeError('Loaded BF16 shard does not reproduce sampled BF16 source coordinates')
    sample=hashlib.sha256(expected.numpy().tobytes()).hexdigest()
    if os.environ.get('Q38_REPLAY_VERIFY_FULL_SHARD_BYTES')=='1':
        # Fresh recovery requires every bounded BF16 destination byte, not
        # only the historical 64-coordinate sample, to match its source.
        actual=destination.detach().cpu().contiguous()
        expected_bytes=part.contiguous().view(torch.uint8)
        actual_bytes=actual.view(torch.uint8)
        if not torch.equal(actual_bytes,expected_bytes):
            raise RuntimeError('Current BF16 destination differs from the complete bounded official source slice')
        payload=memoryview(actual_bytes.numpy()).cast('B')
        full=hashlib.sha256(payload).hexdigest()
        del payload,actual_bytes,expected_bytes,actual
        return {'sample_sha256':sample,'full_slice_bytes_sha256':full}
    return sample


def copy_source_slice(parameter, source, source_map=None, *, max_source_bytes=128*1024**2,
                      world_size=WORLD_SIZE):
    """Bound every safetensors read, including a flat chunk crossing an expert."""
    import torch
    source_shape = tuple(source.get_shape())
    expected = tuple(source_map['runtime_shape']) if source_map else source_shape
    if source_map and source_shape != tuple(source_map['source_shape']):
        raise ValueError('Official packed tensor shape differs from its flat mapping')
    begin, end = source_interval(parameter, expected, world_size=world_size)
    target = parameter.detach().to_local()
    if target.dtype != torch.bfloat16 or not target.is_contiguous():
        raise ValueError('A contiguous BF16 local master is required')
    row_elements = 1
    for size in expected[1:]:row_elements *= size
    rows_per_read = max_source_bytes // (2*row_elements)
    if rows_per_read < 1:raise ValueError('One source row exceeds the bounded reader budget')
    vocab=source_map is not None and 'vocab_row_begin' in source_map
    if vocab:
        fragments=[{'feature_begin':source_map['vocab_row_begin']+begin,
                    'feature_end':source_map['vocab_row_begin']+end,'destination_row_begin':0}]
    elif source_map:
        fragments=flat_source_fragments(source_shape,source_map['expert_begin'],
                                       source_map['expert_end'],begin,end)
    else:fragments=[{'feature_begin':begin,'feature_end':end,'destination_row_begin':0}]
    hashes = []; full_hashes = []; peak = 0; reads = 0
    for fragment in fragments:
        low, high = fragment['feature_begin'], fragment['feature_end']
        for cursor in range(low, high, rows_per_read):
            stop = min(high, cursor+rows_per_read)
            selection = ((fragment['expert'], slice(cursor, stop), slice(None))
                         if source_map and not vocab else (slice(cursor, stop), *(slice(None) for _ in expected[1:])))
            part = source[selection]
            dest_begin = fragment['destination_row_begin'] + cursor-low
            destination = target[dest_begin:dest_begin+stop-cursor]
            check=_verify_copy(destination,part)
            if isinstance(check,dict):
                hashes.append(check['sample_sha256']);full_hashes.append(check['full_slice_bytes_sha256'])
            else:hashes.append(check)
            reads += 1
            peak = max(peak, part.numel()*part.element_size())
            del part
    return {'local_numel': target.numel(), 'local_bytes': target.numel()*2,
            'source_begin': begin, 'source_end': end, 'source_reads': reads,
            'maximum_source_slice_bytes': peak, 'sample_sha256': hashes,
            'source_coordinates_verified': True,
            'full_destination_BF16_bytes_verified':os.environ.get('Q38_REPLAY_VERIFY_FULL_SHARD_BYTES')=='1' and len(full_hashes)==reads,
            'full_slice_bytes_sha256':full_hashes}


def save_buffers(model):
    import torch
    saved = {}; ids = {}; aliases = []
    for name, buffer in model.named_buffers():
        if buffer.is_meta:raise ValueError('An initialized model buffer was discarded on meta')
        saved[name] = buffer.detach().cpu().clone(); ids[id(buffer)] = name
    for prefix, module in model.named_modules():
        for key, value in vars(module).items():
            if isinstance(value, torch.Tensor) and id(value) in ids:
                aliases.append((prefix, key, ids[id(value)]))
    return saved, aliases


def restore_buffers(model, saved, aliases):
    import torch
    current = dict(model.named_buffers())
    if set(current) != set(saved):raise RuntimeError('Buffer names changed during materialization')
    with torch.no_grad():
        for name, value in current.items():
            value.copy_(saved[name])
            if not torch.equal(value.cpu(), saved[name]):raise RuntimeError('Initialized buffer changed')
    for prefix, key, name in aliases:setattr(model.get_submodule(prefix), key, current[name])


def remember_initialized_buffers(model):
    """Keep only initialized tiny non-parameter buffers across verl's cast."""
    saved,aliases=save_buffers(model)
    if sum(value.numel()*value.element_size() for value in saved.values())>1024*1024:
        raise ValueError('The initial-buffer preservation budget must stay below one MiB')
    model._q38_initialized_buffer_state=(saved,aliases)


def restore_initialized_buffers_after_engine_cast(model):
    """verl module.to(BF16) must not quantize original FP32 rotary constants."""
    import torch
    saved,aliases=model._q38_initialized_buffer_state
    current=dict(model.named_buffers())
    if set(current)!=set(saved):raise RuntimeError('Engine casting changed initialized buffer names')
    with torch.no_grad():
        for name,value in current.items():
            if value.is_meta or value.device.type!='cpu':
                raise RuntimeError('Restore initialized buffers before real FSDP materialization')
            # Tensor identity is preserved for any still-bound Python aliases;
            # the explicit alias map also reconnects aliases that module._apply
            # left pointing at the preceding FP32 Tensor object.
            value.data=saved[name].clone()
    restore_buffers(model,saved,aliases)
    del model._q38_initialized_buffer_state
    model._q38_original_buffer_precision_restored=True


class LocalPinnedShardProvider:
    """Existing snapshot reader; whole-file integrity is an external gate."""
    def __init__(self, checkpoint):self.checkpoint=Path(checkpoint)

    def open_file(self, filename):
        from safetensors import safe_open
        return safe_open(str(self.checkpoint/filename),framework='pt',device='cpu')

    def all_rank_file_consumed(self, filename):
        # A fallback owner-download/scatter provider overrides this boundary:
        # all ranks ACK their slices before the owner deletes the local file.
        return None

    def close(self):
        return {'source':'existing_snapshot','persistent_OSS_snapshot_proven_here':False,
                'local_snapshot_integrity_must_be_verified_externally':True}


def load_checkpoint_shards(model, full_state, device_mesh=None, cpu_offload=None):
    import torch
    import torch.distributed as dist
    if not is_qwen38(model.config) or not hasattr(model, '_qwen38_checkpoint_path'):
        raise ValueError('The streamed loader is restricted to the pinned Qwen3.8 policy')
    if (not dist.is_initialized() or dist.get_world_size() != WORLD_SIZE
            or device_mesh is None or device_mesh.ndim != 1 or device_mesh.size() != WORLD_SIZE):
        raise ValueError('The actual FSDP72 process group and mesh are required')
    if cpu_offload is not None or any(not t.is_meta for t in full_state.values()):
        raise ValueError('Do not initialize a full real model or enable full CPU master offload')
    checkpoint = Path(model._qwen38_checkpoint_path)
    _, index, policy, _ = load_metadata(checkpoint)
    sources = model._qwen38_expert_sources
    canonical = {sources[name]['canonical_name'] if name in sources else name
                 for name, _ in model.named_parameters()}
    if canonical != set(policy):raise ValueError('Flattening omitted or added official policy keys')
    device = torch.device('cuda', torch.cuda.current_device())
    coverage = torch.tensor([dist.get_rank()+1, 1], device=device, dtype=torch.int64)
    dist.all_reduce(coverage)
    if coverage.tolist() != [WORLD_SIZE*(WORLD_SIZE+1)//2, WORLD_SIZE]:
        raise RuntimeError('Actual training-group communication preflight failed before weight-file open')
    saved, aliases = save_buffers(model); started = time.time()
    model.to_empty(device=device)
    records = [];files=defaultdict(list)
    for name,parameter in model.named_parameters():
        mapped=sources.get(name);canonical_name=mapped['canonical_name'] if mapped else name
        files[index['weight_map'][canonical_name]].append((name,parameter,mapped,canonical_name))
    selection=os.environ.get('Q38_SOURCE_PROVIDER')
    if selection in ('platform_mounted','existing_snapshot'):
        if os.environ.get('Q38_EXISTING_SNAPSHOT_VERIFIED')!='1':
            raise RuntimeError('Explicit external platform snapshot availability verification is required')
        if checkpoint.resolve()!=Path('/mnt/fuse/Qwen/Qwen3.8-2.4T-A95B').resolve():
            raise ValueError('Use the user-approved current platform model mount')
        provider=LocalPinnedShardProvider(checkpoint)
    else:
        raise ValueError('Select Q38_SOURCE_PROVIDER=platform_mounted; model HTTP downloading is disabled in this run')
    # Every rank visits every official filename in the same deterministic
    # order. A bounded owner/scatter provider can synchronize this boundary
    # without retaining the complete 4.9TB checkpoint on local NVMe.
    for filename in sorted(files):
        with provider.open_file(filename) as stream:
            for name,parameter,mapped,canonical_name in sorted(files[filename],key=lambda row:row[0]):
                if parameter.to_local().device != device:raise RuntimeError('Wrong allocated GPU destination')
                record=copy_source_slice(parameter,stream.get_slice(canonical_name),mapped)
                parameter._qwen36_parameter_name=name
                record.update(name=name,canonical_name=canonical_name,checkpoint_file=filename)
                records.append(record)
                if len(records)==1 or len(records)%256==0:
                    print('Q38_CHECKPOINT_SLICE_LOADED',dist.get_rank(),len(records),flush=True)
        provider.all_rank_file_consumed(filename)
    # This boundary is reached only after every file context closed and its
    # all-rank consumption ACK succeeded. Never `finally close()` an incomplete
    # transport and report it as successful consumption.
    source_close=provider.close()
    restore_buffers(model, saved, aliases)
    owned = sum(row['local_numel'] for row in records)
    total = torch.tensor(owned, device=device, dtype=torch.int64); dist.all_reduce(total)
    if total.item() != POLICY_NUMEL:raise RuntimeError('Actual shards do not cover the policy exactly once')
    audit = {'schema': 'qwen38-flat-sharded-checkpoint-v1', 'rank': dist.get_rank(),
             'world_size': WORLD_SIZE, 'hostname': socket.gethostname(), 'pid': os.getpid(),
             'started_unix': started, 'finished_unix': time.time(), 'loaded_local_numel': owned,
             'global_policy_numel': total.item(), 'source_keys': len(canonical),
             'runtime_parameters': len(records), 'rank0_full_model_materialized': False,
             'source_provider':selection,'main_policy_source_files_consumed':len(files),
             'model_weight_downloading_performed':False,
             'whole_file_body_sha256_verified_here':False,
             'source_provider_close_receipt':source_close,
             'complete_224_object_OSS_snapshot_claimed':False,
             'all_master_dtype': 'bfloat16', 'official_source_dtype': 'bfloat16',
             'maximum_source_slice_bytes': max(row['maximum_source_slice_bytes'] for row in records),
             'buffer_values_and_aliases_verified': True, 'source_slices': records}
    target = Path(os.environ['Q38_TRAINING_AUDIT_DIR']); target.mkdir(parents=True, exist_ok=True)
    with (target/f'checkpoint-load-rank{dist.get_rank()}.json').open('x') as stream:
        json.dump(audit, stream);stream.write('\n')
    model._qwen38_streaming_verified = True
    print('Q38_CHECKPOINT_SHARD_READY', dist.get_rank(), owned, flush=True)

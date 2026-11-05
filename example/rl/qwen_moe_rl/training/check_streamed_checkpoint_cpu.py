"""CPU-only regression for real DTensor offsets, partial reads and buffers.

Run with CUDA_VISIBLE_DEVICES='' python -m torch.distributed.run --standalone
--nproc_per_node=4 training/check_streamed_checkpoint_cpu.py --output ...json
This validates loader mechanics, never claims a full-model/GPU training pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, Replicate
    from safetensors import safe_open
    from safetensors.torch import save_file
    from qwen397_streamed_checkpoint import (capture_buffers, copy_checkpoint_parameter,
                                            restore_buffers, shard_interval, source_slice,
                                            validate_torch_api)

    if torch.cuda.is_initialized():
        raise RuntimeError('This regression must never initialize CUDA')
    validate_torch_api()
    dist.init_process_group('gloo')
    world, rank = dist.get_world_size(), dist.get_rank()
    mesh = init_device_mesh('cpu', (world,))
    temporary = [tempfile.mkdtemp(prefix='q397-checkpoint-cpu-') if rank == 0 else None]
    dist.broadcast_object_list(temporary, src=0)
    directory = Path(temporary[0])
    sources = {'even': torch.arange(24, dtype=torch.float32).reshape(8, 3).to(torch.bfloat16),
               'uneven': (torch.arange(14, dtype=torch.float32)/16).reshape(7, 2).to(torch.bfloat16),
               'empty_tail': torch.arange(5, dtype=torch.float32).reshape(1, 5).to(torch.bfloat16),
               'packed_experts': (torch.arange(4096, dtype=torch.float32)/64).reshape(512, 4, 2).to(torch.bfloat16),
               'gdn_fp32': torch.linspace(-0.117, 1.743, 64, dtype=torch.float32)}
    if rank == 0:
        save_file(sources, str(directory/'model.safetensors'))
    dist.barrier()
    records = []
    with safe_open(str(directory/'model.safetensors'), framework='pt', device='cpu') as checkpoint:
        for name, original in sources.items():
            begin, end = shard_interval(original.shape[0], world, rank)
            local = torch.empty((end-begin, *original.shape[1:]), dtype=torch.float32)
            dtensor = DTensor.from_local(local, device_mesh=mesh, placements=[Shard(0)],
                                        shape=original.shape, stride=original.stride(), run_check=False)
            parameter = torch.nn.Parameter(dtensor)
            record = copy_checkpoint_parameter(parameter, checkpoint.get_slice(name), name, expected_world=world)
            if not torch.equal(parameter.detach().to_local(), original[begin:end].float()):
                raise AssertionError('Complete loaded local shard differs: '+name)
            gathered = parameter.detach().full_tensor()
            if not torch.equal(gathered, original.float()):
                raise AssertionError('Full tensor reconstructed from all actual DTensor shards differs: '+name)
            # Create the real autograd graph only after initialization. A local
            # shard loss has an exact 2*x gradient, including empty tail ranks;
            # it verifies that detached checkpoint writes did not sever the
            # trainable DTensor leaf or corrupt its view/version bookkeeping.
            loss = parameter.to_local().square().sum()
            loss.backward()
            if parameter.grad is None or not torch.equal(
                    parameter.grad.detach().to_local(), 2*original[begin:end].float()):
                raise AssertionError('Post-load DTensor autograd gradient differs: '+name)
            record['post_load_backward_checked'] = True
            records.append(record)

        replicated = DTensor.from_local(torch.zeros((8, 3)), device_mesh=mesh,
                                        placements=[Replicate()], run_check=False)
        try:
            source_slice(replicated, (8, 3), expected_world=world)
        except ValueError:
            pass
        else:
            raise AssertionError('Replicated parameters must be rejected before consuming full-model memory')

    model = torch.nn.Module()
    model.register_buffer('inv_freq', torch.tensor([0.1, 0.23, 0.91]), persistent=False)
    model.original_inv_freq = model.inv_freq  # Older HF versions used a plain alias.
    saved, aliases = capture_buffers(model)
    model.to_empty(device='cpu')
    restore_buffers(model, saved, aliases)
    if model.original_inv_freq is not model.inv_freq or not torch.equal(model.inv_freq, saved['inv_freq']):
        raise AssertionError('Initialized nonpersistent buffer or plain tensor alias was lost')
    # Exercise real 32-way interval arithmetic without starting 32 test processes.
    for size in (0, 1, 2, 16, 31, 32, 33, 63, 64, 512, 4096, 248320):
        intervals = [shard_interval(size, 32, coordinate) for coordinate in range(32)]
        if intervals[0][0] != 0 or intervals[-1][1] != size:
            raise AssertionError('32-rank intervals do not reach tensor boundaries')
        if any(left[1] != right[0] for left, right in zip(intervals, intervals[1:])):
            raise AssertionError('32-rank intervals overlap or omit checkpoint values')
        if sum(stop-start for start, stop in intervals) != size:
            raise AssertionError('32-rank interval coverage differs from tensor extent')
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU regression initialized CUDA')
    all_records = [None for _ in range(world)]
    dist.all_gather_object(all_records, dict(rank=rank, records=records))
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = dict(schema='qwen397-streamed-checkpoint-cpu-regression-v1', passed=True,
                      torch_version=torch.__version__, world_size=world, cuda_initialized=False,
                      checks=['real DTensor source offsets', 'even and uneven shards', 'zero-length tail shards',
                              'BF16 source to FP32 master', 'FP32 GDN source preservation',
                              'full reconstruction from actual distributed shards',
                              'post-load DTensor backward with exact local gradients', 'replicate rejected',
                              'nonpersistent buffers and plain alias preserved', '32-way interval coverage'],
                      source_file_sha256=hashlib.sha256((directory/'model.safetensors').read_bytes()).hexdigest(),
                      ranks=all_records)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print('Q397_STREAMED_CHECKPOINT_CPU_PASSED', json.dumps({k: v for k, v in result.items() if k != 'ranks'}), flush=True)
        (directory/'model.safetensors').unlink()
        directory.rmdir()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

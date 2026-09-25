"""Run with torchrun --standalone --nproc-per-node=8; correctness only."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.modules.kimi_k3.collectives import reduce_scatter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    dist.init_process_group('nccl', timeout=timedelta(seconds=180))
    assert dist.get_world_size() == 8
    collective._parallelism_config = SimpleNamespace(tp_size=8, dp_size=1, world_size=8)
    collective._initialized = True
    collective._group_map[collective.Group.DP_AND_TP] = dist.group.WORLD
    args.output.mkdir(exist_ok=True)
    generator = torch.Generator().manual_seed(48149)
    operands = (torch.randint(-64, 65, (8, 5032, 128), generator=generator)
                .float().div_(256).bfloat16())
    report = {'rank': rank, 'world_size': 8, 'scope': 'K3 collective correctness only', 'shapes': {}}

    for name, offset, count in [
        ('full', 0, 5032), ('reuse', 4096, 936),
        ('below_old_boundary', 0, 1168),
        ('above_old_boundary', 0, 1176), ('decode', 0, 8),
    ]:
        cpu = operands[rank, offset:offset+count].repeat(1, 56)
        backing = torch.empty((count, 14336), device='cuda', dtype=torch.bfloat16)
        x = backing[:, ::2]
        x.copy_(cpu)
        assert not x.is_contiguous()
        assert x.shape == (count, 7168)

        def expected(scale=1.):
            # Compare with direct BF16 NCCL. Its reduction order is backend-owned,
            # so do not assume a ring order or replace it with an FP32 sum.
            source = (cpu * scale).bfloat16().cuda().contiguous()
            result = torch.empty((count // 8, 7168), device='cuda',
                                 dtype=torch.bfloat16)
            dist.reduce_scatter_tensor(result, source, group=dist.group.WORLD)
            return result.cpu()

        eager = reduce_scatter(x, collective.Group.TP)
        assert torch.equal(eager.cpu(), expected()), (name, 'BF16 NCCL oracle')
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                reduce_scatter(x, collective.Group.TP)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = reduce_scatter(x, collective.Group.TP)
        for scale in [1., .5, 1.25]:
            x.copy_((cpu * scale).cuda())
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(captured.cpu(), expected(scale)), (name, 'replay', scale)
        x.copy_(cpu)
        actual_fp32 = reduce_scatter(x.float(), collective.Group.TP)
        start = offset + rank * (count // 8)
        expected_fp32 = operands[:, start:start+count//8].float().sum(0).repeat(1, 56)
        assert actual_fp32.dtype == torch.float32
        assert torch.equal(actual_fp32.cpu(), expected_fp32)
        report['shapes'][name] = {'physical_tokens': count, 'width': 7168,
                                 'communication_dtype': 'BF16', 'backend': 'NCCL',
                                 'noncontiguous': True, 'bf16_nccl_oracle_equal': True,
                                 'replay_scales': [1., .5, 1.25], 'fp32_fallback_equal': True}
        del graph, captured, eager, x, backing
    # Full/reuse equality is checked separately at model level.
    for member in range(8):
        subgroup = dist.new_group([member])
        if member == rank:
            singleton = subgroup
    collective._parallelism_config = SimpleNamespace(tp_size=1, dp_size=1, world_size=1)
    collective._group_map[collective.Group.DP_AND_TP] = singleton
    sample = torch.ones((1, 8), device='cuda', dtype=torch.bfloat16)
    assert reduce_scatter(sample, collective.Group.TP) is sample
    module_path = Path(__import__(reduce_scatter.__module__, fromlist=['__file__']).__file__)
    report.update(tp1_identity=True, implementation_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest())
    with (args.output / f'rank{rank}.json').open('x') as out:
        json.dump(report, out, indent=2)
    dist.barrier()
    if rank == 0:
        print('PASS: TP8 BF16 NCCL, boundary shapes, strided inputs, 15 graph replays per rank, FP32 fallback, TP1', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

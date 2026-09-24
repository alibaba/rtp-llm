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
    # Use a real NCCL group without starting the unrelated RTP server transport.
    collective._parallelism_config = SimpleNamespace(tp_size=8, dp_size=1, world_size=8)
    collective._initialized = True
    collective._group_map[collective.Group.DP_AND_TP] = dist.group.WORLD
    args.output.mkdir(exist_ok=True)
    generator = torch.Generator().manual_seed(48149)
    operands = (torch.randint(-64, 65, (8, 5032, 128), generator=generator)
                .float().div_(256).bfloat16())
    # Emulate the native BF16 rank-order contract with FP64 oracle additions
    # and explicit rounding after each step. This differs from one final cast.
    full_reference = operands[0].clone()
    for contribution in operands[1:]:
        full_reference = (full_reference.double() + contribution.double()).bfloat16()
    assert not torch.equal(full_reference, operands.double().sum(0).bfloat16())
    report = {'rank': rank, 'world_size': 8, 'scope': 'K3 collective correctness only', 'shapes': {}}
    results = {}

    def gather(value):
        result = torch.empty((value.shape[0]*8, value.shape[1]), dtype=value.dtype, device=value.device)
        dist.all_gather_into_tensor(result, value.contiguous())
        return result.cpu()

    for name, offset, count in [('full', 0, 5032), ('reuse', 4096, 936)]:
        cpu = operands[rank, offset:offset+count]
        # A strided input checks the production contiguous conversion too.
        backing = torch.empty((count, 256), device='cuda', dtype=torch.bfloat16)
        x = backing[:, ::2]
        x.copy_(cpu)
        assert not x.is_contiguous()
        expected = full_reference[offset:offset+count]
        eager = reduce_scatter(x, collective.Group.TP)
        actual = gather(eager)
        assert torch.equal(actual, expected), (name, 'native BF16 rounding oracle')
        results[name] = actual
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
        for scale in [1., .5, 1.]:
            x.copy_((cpu * scale).cuda())
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(gather(captured), expected * scale), (name, 'replay', scale)
        # Non-BF16 must retain the existing reduction rather than downcast.
        fp32 = reduce_scatter(x.float(), collective.Group.TP)
        assert fp32.dtype == torch.float32
        assert torch.equal(gather(fp32), operands[:, offset:offset+count].float().sum(0))
        report['shapes'][name] = {'physical_tokens': count, 'noncontiguous': True,
                                'native_bf16_oracle_equal': True, 'replay_scales': [1, .5, 1],
                                'fp32_fallback_equal': True}
        del graph, captured, eager, x, backing
    assert torch.equal(results['full'][4096:5030], results['reuse'][:934])
    # TP1 retains the input (no allocation or collective); restore the fixture.
    for member in range(8):
        subgroup = dist.new_group([member])
        if member == rank:
            singleton = subgroup
    collective._parallelism_config = SimpleNamespace(tp_size=1, dp_size=1, world_size=1)
    collective._group_map[collective.Group.DP_AND_TP] = singleton
    sample = torch.ones((1, 8), device='cuda', dtype=torch.bfloat16)
    assert reduce_scatter(sample, collective.Group.TP) is sample
    module_path = Path(__import__(reduce_scatter.__module__, fromlist=['__file__']).__file__)
    report.update(full_reuse_equal=True, tp1_identity=True,
                  implementation_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest())
    with (args.output / f'rank{rank}.json').open('x') as out:
        json.dump(report, out, indent=2)
    dist.barrier()
    if rank == 0:
        print('PASS: TP8 native BF16 rounding oracle, strided input, full/reuse, six graph replays, FP32 fallback, TP1', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

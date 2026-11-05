"""Compare staged AdamW to torch AdamW across chunks, skipped gradients and reload."""
import argparse
import json
import os

import torch
import torch.distributed as dist

from qwen35_streamed_adamw import CpuMomentAdamW, _local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--distributed', action='store_true')
    args = parser.parse_args()
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    if args.device == 'cuda':
        torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}' if args.device == 'cuda' else 'cpu')
    if args.distributed:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import distribute_tensor, Shard
        dist.init_process_group(backend='nccl' if args.device == 'cuda' else 'gloo')
        mesh = init_device_mesh(args.device, (dist.get_world_size(),))

    def shard(tensor):
        tensor = tensor.to(device)
        if args.distributed:
            return distribute_tensor(tensor, mesh, [Shard(0)])
        return tensor

    torch.manual_seed(875)
    shapes = [(32, 19), (23, 13), (16, 27)]
    actor = [torch.nn.Parameter(shard(torch.randn(shape))) for shape in shapes]
    expected = [torch.nn.Parameter(_local(param.detach()).clone()) for param in actor]
    kwargs = dict(lr=1e-6, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)
    staged = CpuMomentAdamW(actor, chunk_bytes=52, **kwargs)
    reference = torch.optim.AdamW(expected, fused=args.device == 'cuda', foreach=False, **kwargs)
    # CPU single-tensor kernels can round scalar tails and SIMD blocks
    # differently for our deliberately 13-element chunks. CUDA fused kernels
    # are required to match the resident-state control bit for bit.
    fp_tolerance = dict(rtol=0, atol=0) if args.device == 'cuda' else dict(rtol=1e-6, atol=6e-8)
    max_delta = max_moment_delta = 0.0
    for step in range(1, 6):
        if step == 3:
            restore = CpuMomentAdamW(actor, chunk_bytes=52, **kwargs)
            restore.load_state_dict(staged.state_dict())
            staged = restore
        for index, shape in enumerate(shapes):
            if index == 2 or (index == 1 and step == 2):
                actor[index].grad = expected[index].grad = None
                continue
            grad = shard(torch.randn(shape))
            actor[index].grad = grad
            expected[index].grad = _local(grad).clone()
        staged.step()
        reference.step()
        for actual, target in zip(actor, expected, strict=True):
            value = _local(actual.detach())
            max_delta = max(max_delta, (value - target).abs().max().item())
            torch.testing.assert_close(value, target, **fp_tolerance)
            if actual not in staged.state:
                assert target not in reference.state
                continue
            for key in ('exp_avg', 'exp_avg_sq', 'step'):
                a, b = staged.state[actual][key], reference.state[target][key].cpu()
                assert a.device.type == 'cpu' and a.dtype == torch.float32
                max_moment_delta = max(max_moment_delta, (a - b).abs().max().item())
                torch.testing.assert_close(a, b, **(dict(rtol=0, atol=0) if key == 'step' else fp_tolerance))
    info = staged.moment_placement()
    assert info['moment_devices'] == ['cpu']
    assert info['max_staged_moment_bytes_per_step'] == 104
    assert info['persistent_moment_bytes'] == 8 * sum(_local(p).numel() for p in actor[:2])
    print(json.dumps(dict(passed=True, rank=rank, world=dist.get_world_size() if args.distributed else 1,
                          device=str(device), torch_version=torch.__version__, steps=5,
                          max_weight_abs_difference=max_delta, max_moment_abs_difference=max_moment_delta,
                          **info)), flush=True)
    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

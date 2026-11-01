"""Small-input numerical validation of unchanged production quant kernels.

Run under an external timeout and a single explicit CUDA_VISIBLE_DEVICES UUID.
No model import, no collectives, no EGM, no production module replacement.
"""
import argparse
import json
import math
import os
import subprocess

import torch

from build_quant_probe import load_probe, DEFAULT_REPO, default_build_directory


def emit(**record):
    print(json.dumps(record, sort_keys=True), flush=True)


def expected(cpu, v2):
    """Mathematical group32 power-of-two scaling and CPU FP8 rounding oracle.

    Legacy applies a 1e-10 scale floor. V2's integer exponent extraction
    saturates subnormal scales to minimum normal float32 (observable at zero).
    Neither convention is changed by this probe.
    """
    x = cpu.float().reshape(cpu.shape[0], -1, 32)
    amax = x.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    unrounded = amax / 448.0
    floor = torch.finfo(torch.float32).tiny if v2 else 1e-10
    exponent = torch.ceil(torch.log2(unrounded.clamp_min(floor)))
    scales = torch.pow(2.0, exponent)
    q = (x / scales.unsqueeze(-1)).clamp(-448.0, 448.0).reshape(cpu.shape)
    return q.to(torch.float8_e4m3fn).view(torch.uint8), (exponent + 127).to(torch.uint8)


def packed_scales(m, k):
    aligned_m = (m + 3) // 4 * 4
    packs = (k // 32 + 3) // 4
    return torch.zeros((packs, aligned_m), device='cuda', dtype=torch.int32).T[:m]


def unpack_scale_bytes(scales, groups):
    words = scales.cpu().to(torch.int64) & 0xffffffff
    return torch.stack([(words >> (8 * shift)) & 255 for shift in range(4)], -1).reshape(scales.shape[0], -1)[:, :groups].to(torch.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default=str(DEFAULT_REPO))
    p.add_argument('--build', default=default_build_directory())
    a = p.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible.startswith('GPU-') or ',' in visible:
        raise RuntimeError('exactly one GPU UUID in CUDA_VISIBLE_DEVICES is required')
    # nvidia-smi visibility is physical. Check the exact UUID before CUDA init.
    line = subprocess.check_output(['nvidia-smi', '-i', visible,
                                   '--query-gpu=memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True).strip()
    memory, utilization = [int(x.strip()) for x in line.split(',')]
    if memory >= 64 or utilization != 0:
        emit(status='GPU_TEST_SKIPPED', reason='selected GPU is occupied', memory_mib=memory, utilization=utilization)
        raise SystemExit(77)
    mod = load_probe(a.repo, a.build)
    torch.cuda.set_device(0)
    emit(stage='environment', torch=torch.__version__, cuda_runtime=torch.version.cuda,
         device_name=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0),
         visible_uuid=visible, max_jobs=os.environ.get('MAX_JOBS'))
    gen = torch.Generator(device='cpu').manual_seed(41016)
    cases = []
    for m, k in [(4, 32), (7, 256), (32, 512)]:
        cases.append((f'random_{m}x{k}', torch.randn((m, k), generator=gen).to(torch.bfloat16)))
    cases.append(('zero_7x256', torch.zeros((7, 256), dtype=torch.bfloat16)))
    cases.append(('tiny_7x256', (torch.randn((7, 256), generator=gen) * 1e-12).to(torch.bfloat16)))
    boundary = torch.tensor([0., 1., -1., 448., -448., 224., -224., 1e4], dtype=torch.bfloat16)
    cases.append(('boundary_8x256', boundary[:, None].expand(8, 256).contiguous()))
    total = 0
    for name, cpu in cases:
        x = cpu.cuda()
        for v2 in (False, True):
            q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
            scales = packed_scales(*x.shape)
            mod.quant(x, q, scales, v2)
            torch.cuda.synchronize()
            ref_q, ref_s = expected(cpu, v2)
            q_bad = int((q.view(torch.uint8).cpu() != ref_q).sum())
            s_bad = int((unpack_scale_bytes(scales, x.shape[1] // 32) != ref_s).sum())
            emit(stage='case', case=name, variant='v2' if v2 else 'legacy',
                 q_elements=x.numel(), scale_elements=ref_s.numel(),
                 q_mismatches=q_bad, scale_mismatches=s_bad)
            if q_bad or s_bad:
                raise AssertionError(f'{name} variant v2={v2}: q={q_bad}, scales={s_bad}')
            total += x.numel()
    torch.cuda.synchronize()
    emit(status='PASS', cases=len(cases) * 2, compared_fp8_values=total,
         scope='unchanged group32 BF16->E4M3+packedUE8M0 kernels only; not full Engram/model')


if __name__ == '__main__':
    main()

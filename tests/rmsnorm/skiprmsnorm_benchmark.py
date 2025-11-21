from time import time

import torch
import torch.nn.functional as F

from rtp_llm.models_py.kernels import atex_skiprmsnorm

M = [1, 3, 9, 15, 32, 256]
N = [256, 512, 768, 1024, 1536, 2048, 4096, 8192, 16384]
ITER = 100


def tensor_size(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


f = open("/root/projects/rtp-llm/rmsnorm.csv", "w")

for m in M:
    for n in N:
        x = torch.randn(size=[m, n], device="cuda", dtype=torch.float16)
        r = torch.randn(size=[m, n], device="cuda", dtype=torch.float16)
        w = torch.randn(size=[n], device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()

        tik = time()
        for iter in range(ITER):
            o1, o2 = atex_skiprmsnorm(x, r, w, eps=1e-9, normailize_shape=n)
        torch.cuda.synchronize()
        tok = time()

        bandwidth = tensor_size(w) + tensor_size(o1) + tensor_size(x) + tensor_size(o2)
        bandwidth = bandwidth / 1024 / 1024 / 1024 * ITER
        bandwidth = bandwidth / (tok - tik)
        print(f"Rmsnorm Benchmark: M={m}, N={n}, bandwidth={bandwidth: .2f} GB/s")
        f.write(f"{m},{n},{bandwidth}\n")

        real = F.rms_norm(x + r, [n], w, eps=1e-9)
        sum = x + r
        check1 = torch.allclose(real, o1, rtol=0.001, atol=0.001)
        check2 = torch.allclose(sum, o2, rtol=0.001, atol=0.001)
        if not check1 or not check2:
            raise Exception("result is not correct")


for m in M:
    for n in N:
        x = torch.randn(size=[m, n], device="cuda", dtype=torch.bfloat16)
        r = torch.randn(size=[m, n], device="cuda", dtype=torch.bfloat16)
        w = torch.randn(size=[n], device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()

        tik = time()
        for iter in range(ITER):
            o1, o2 = atex_skiprmsnorm(x, r, w, eps=1e-9, normailize_shape=n)
        torch.cuda.synchronize()
        tok = time()

        bandwidth = tensor_size(w) + tensor_size(o1) + tensor_size(x) + tensor_size(o2)
        bandwidth = bandwidth / 1024 / 1024 / 1024 * ITER
        bandwidth = bandwidth / (tok - tik)
        print(f"Rmsnorm Benchmark: M={m}, N={n}, bandwidth={bandwidth: .2f} GB/s")
        f.write(f"{m},{n},{bandwidth}\n")

        real = F.rms_norm(x + r, [n], w, eps=1e-9)
        sum = x + r
        check1 = torch.allclose(real, o1, rtol=0.01, atol=0.01)
        check2 = torch.allclose(sum, o2, rtol=0.01, atol=0.01)
        if not check1 or not check2:
            raise Exception("result is not correct")

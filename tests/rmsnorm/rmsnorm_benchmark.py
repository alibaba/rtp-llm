from time import time

import torch
import torch.nn.functional as F

from rtp_llm.models_py.kernels import atex_rmsnorm

M = [1, 3, 9, 15, 32, 256, 512, 1039]
N = [256, 512, 768, 1024, 1536, 2048, 4096, 8192, 16384]
ITER = 128


def tensor_size(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


f = open("/root/projects/rtp-llm/rmsnorm.csv", "w")
for m in M:
    for n in N:
        x = torch.randn(size=[m, n], device="cuda", dtype=torch.float16)
        w = torch.randn(size=[n], device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()

        tik = time()
        for iter in range(ITER):
            o = atex_rmsnorm(x, w, eps=1e-9, normailize_shape=n)
        torch.cuda.synchronize()
        tok = time()

        bandwidth = tensor_size(w) + tensor_size(o) + tensor_size(x)
        bandwidth = bandwidth / 1024 / 1024 / 1024 * ITER
        bandwidth = bandwidth / (tok - tik)
        print(f"Rmsnorm Benchmark: M={m}, N={n}, bandwidth={bandwidth:.2f} GB/s")
        f.write(f"{m},{n},{bandwidth}\n")

        real = F.rms_norm(x, [n], w, eps=1e-9)
        if not torch.allclose(real, o, rtol=0.001, atol=0.001):
            raise Exception("result is not correct")


for m in M:
    for n in N:
        x = torch.randn(size=[m, n], device="cuda", dtype=torch.bfloat16)
        w = torch.randn(size=[n], device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()

        tik = time()
        for iter in range(ITER):
            o = atex_rmsnorm(x, w, eps=1e-9, normailize_shape=n)
        torch.cuda.synchronize()
        tok = time()

        bandwidth = tensor_size(w) + tensor_size(o) + tensor_size(x)
        bandwidth = bandwidth / 1024 / 1024 / 1024 * ITER
        bandwidth = bandwidth / (tok - tik)
        print(f"Rmsnorm Benchmark: M={m}, N={n}, bandwidth={bandwidth:.2f} GB/s")
        f.write(f"{m},{n},{bandwidth}\n")

        real = F.rms_norm(x, [n], w, eps=1e-9)
        if not torch.allclose(real, o, rtol=0.01, atol=0.01):
            raise Exception("result is not correct")

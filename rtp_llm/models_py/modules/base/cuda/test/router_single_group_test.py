import unittest

import torch
from rtp_llm.models_py.triton_kernels.moe.router_single_group import (
    try_single_group_router,
)
from rtp_llm.ops import compute_ops


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SingleGroupRouterTest(unittest.TestCase):
    def test_exact_cuda_boundary(self):
        op = compute_ops.GroupTopKOp()
        torch.manual_seed(892)
        for n in (1, 17, 2024):
            for dtype in (torch.bfloat16, torch.float32):
                for kind in ("random", "ties", "extreme", "invalid"):
                    x = torch.randn(n, 256, dtype=dtype, device="cuda")
                    bias = torch.randn(256, device="cuda")
                    if kind == "ties":
                        x.zero_()
                        bias.zero_()
                    if kind == "extreme":
                        x *= 90
                        bias.zero_()
                    if kind == "invalid":
                        bias[0] = float("inf")
                        x[0].fill_(float("nan"))
                    for norm in (False, True):
                        for idtype in (torch.int32, torch.int64):
                            ref_w = torch.empty((n, 8), device="cuda")
                            ref_i = torch.empty((n, 8), device="cuda", dtype=idtype)
                            w, ids = torch.empty_like(ref_w), torch.empty_like(ref_i)
                            s = x.float().sigmoid()
                            op.forward(ref_w, ref_i, s, s + bias, 1, 1, 8, norm, 2.5)
                            self.assertTrue(
                                try_single_group_router(
                                    x, bias, w, ids, 1, 1, 8, norm, 2.5
                                )
                            )
                            self.assertTrue(
                                torch.equal(ids, ref_i),
                                f"ids {n} {dtype} {kind} {norm}",
                            )
                            self.assertTrue(
                                torch.equal(
                                    w.view(torch.int32), ref_w.view(torch.int32)
                                ),
                                f"weights {n} {dtype} {kind} {norm}, maxdiff={(w-ref_w).abs().max().item()}",
                            )

    def test_subnormal_scores_and_bias_cancellation(self):
        op = compute_ops.GroupTopKOp()
        for value in (-88.0, -89.0, -90.0, 0.0):
            x = torch.full((17, 256), value, device="cuda")
            bias = torch.zeros(256, device="cuda")
            bias[::3] = 1.0e-38
            bias[1::3] = -1.0e-38
            for factor in (1.0, 1.3, 2.5):
                w = torch.empty(17, 8, device="cuda")
                ids = torch.empty(17, 8, device="cuda", dtype=torch.int64)
                rw, ri = torch.empty_like(w), torch.empty_like(ids)
                scores = x.sigmoid()
                op.forward(rw, ri, scores, scores + bias, 1, 1, 8, True, factor)
                self.assertTrue(
                    try_single_group_router(x, bias, w, ids, 1, 1, 8, True, factor)
                )
                self.assertTrue(torch.equal(ids, ri), f"ids {value}")
                self.assertTrue(
                    torch.equal(w.view(torch.int32), rw.view(torch.int32)),
                    f"weights {value} {factor}",
                )


if __name__ == "__main__":
    unittest.main()

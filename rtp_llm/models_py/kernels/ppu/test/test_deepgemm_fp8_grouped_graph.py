"""Real grouped FP8 kernels must follow the capture stream and replay inputs."""

from unittest import TestCase, main

import numpy as np
import torch

from rtp_llm.models_py.kernels.ppu import deepgemm_wrapper as gemm


class Fp8GroupedGraphTest(TestCase):
    def test_contiguous_and_nopad_replay_changed_inputs(self):
        torch.cuda.set_device(0)
        groups, rows, n, k = 4, 512, 256, 256
        lhs = torch.ones(rows, k, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.ones(rows, k // 128, device="cuda")
        rhs = torch.ones(groups, n, k, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
        rhs_scales = torch.ones(groups, n // 128, k // 128, device="cuda")
        indices = torch.arange(groups, dtype=torch.int32, device="cuda").repeat_interleave(rows // groups)
        counts = torch.full((groups,), rows // groups, dtype=torch.int32, device="cuda")
        for layout in ("contiguous", "nopad"):
            with self.subTest(layout=layout):
                impl = getattr(gemm, "m_grouped_fp8_gemm_nt_" + layout)
                output = torch.zeros(rows, n, dtype=torch.bfloat16, device="cuda")
                def forward():
                    args = [counts] if layout == "nopad" else []
                    impl((lhs, scales), (rhs, rhs_scales), output, indices, *args)
                lhs.fill_(1)
                forward()
                torch.cuda.synchronize()
                self.assertTrue(np.all(output.float().cpu().numpy() == k))
                output.zero_()
                torch.cuda.synchronize()
                graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
                with torch.cuda.graph(graph, stream=stream):
                    forward()
                torch.cuda.synchronize()
                self.assertTrue(np.all(output.float().cpu().numpy() == 0))
                for value in (2, 3):
                    lhs.fill_(value)
                    output.zero_()
                    torch.cuda.synchronize()
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertTrue(np.all(output.float().cpu().numpy() == value * k))


if __name__ == "__main__":
    main()

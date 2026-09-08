"""GLM TP2/4/8 AG/GEMM and GEMM/RS with padding and stream handoff."""

import os
import unittest

import torch
import torch.distributed as dist
from rtp_llm.models_py.distributed.glm53_collective_gemm import (
    all_gather_projections,
    can_fuse_input,
    configure_glm53_collective_gemm,
    project_reduce_scatter,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Glm53CollectiveGemmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = int(os.environ.get("WORLD_SIZE", "1"))
        if cls.world not in (2, 4, 8):
            raise unittest.SkipTest("run with torchrun --nproc-per-node=2, 4 or 8")
        cls.rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(cls.rank)
        dist.init_process_group("nccl", device_id=torch.device("cuda", cls.rank))
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            raise unittest.SkipTest("SM100/103 required")
        os.environ.update(GLM53_PREFILL_AG_GEMM="1", GLM53_PREFILL_GEMM_RS="1")
        configure_glm53_collective_gemm(dist.group.WORLD, 4096)
        torch.manual_seed(9330 + cls.rank)

    @classmethod
    def tearDownClass(cls):
        dist.barrier()
        dist.destroy_process_group()

    def test_input_token_order_and_ragged_padding(self):
        weights = [
            torch.randn(4096, n, device="cuda", dtype=torch.bfloat16) * 0.02
            for n in (3072, 8, 128, 128)
        ]
        projections = [CudaF16Linear(w) for w in weights]
        self.assertFalse(can_fuse_input(projections, 7))
        for logical in (32768, 32769):
            local_m = (logical + self.world - 1) // self.world
            x = torch.randn(local_m, 4096, device="cuda", dtype=torch.bfloat16) * 0.1
            if self.rank == self.world - 1:
                x[logical - local_m * self.rank :].zero_()
            gathered = x.new_empty(local_m * self.world, 4096)
            dist.all_gather_into_tensor(gathered, x)
            actual = all_gather_projections(x, projections, logical)
            for p, y in zip(projections, actual):
                expected = p(gathered[:logical])
                self.assertEqual(y.shape, expected.shape)
                torch.testing.assert_close(y, expected, rtol=1 / 128, atol=2e-3)

    def test_output_padding_and_stream_handoff(self):
        weight = torch.randn(1024, 4096, device="cuda", dtype=torch.bfloat16) * 0.02
        projection = CudaF16Linear(weight)
        self.assertIsNone(project_reduce_scatter(weight[:7], projection))
        stream = torch.cuda.Stream()
        for logical in (32768, 32769, 32768):
            physical = (logical + self.world - 1) // self.world * self.world
            x = torch.randn(logical, 1024, device="cuda", dtype=torch.bfloat16) * 0.1
            # FP32 reduction oracle avoids imposing NCCL's BF16 addition tree.
            local = projection(x).float()
            padded = local.new_zeros((physical, 4096))
            padded[:logical].copy_(local)
            expected = local.new_empty((physical // self.world, 4096))
            dist.reduce_scatter_tensor(expected, padded)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                actual = project_reduce_scatter(x, projection)
            torch.cuda.current_stream().wait_stream(stream)
            torch.testing.assert_close(
                actual, expected.to(torch.bfloat16), rtol=1 / 128, atol=2e-3
            )
            # The next launch moves back to the default stream; workspace ownership
            # must serialize this use without synchronizing the whole device.
            again = project_reduce_scatter(x, projection)
            torch.testing.assert_close(again, actual, rtol=0, atol=0)
            if self.rank == self.world - 1 and physical > logical:
                self.assertEqual(
                    int(
                        torch.count_nonzero(
                            actual[logical - self.rank * (physical // self.world) :]
                        )
                    ),
                    0,
                )


if __name__ == "__main__":
    unittest.main()

"""GLM TP2/4/8 AG/GEMM and GEMM/RS with padding and stream handoff."""

import os
import unittest

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.glm53_collective_gemm import (
    all_gather_kda_projections,
    all_gather_projections,
    can_fuse_input,
    configure_glm53_collective_gemm,
    project_reduce_scatter,
    reduce_scatter_glm53,
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
        os.environ.update(
            GLM53_PREFILL_AG_GEMM="1",
            GLM53_PREFILL_GEMM_RS="1",
            GLM53_PREFILL_STABLE_RS="1",
        )
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

    def test_local_low_rank_token_order_and_replicated_weights(self):
        weights = [
            torch.randn(4096, n, device="cuda", dtype=torch.bfloat16) * 0.02
            for n in (3072, 8, 128, 256)
        ]
        for weight in weights[2:]:
            dist.broadcast(weight, src=0)
        projections = [CudaF16Linear(weight) for weight in weights]
        for logical in (32768, 32769):
            local_m = (logical + self.world - 1) // self.world
            x = torch.randn(local_m, 4096, device="cuda", dtype=torch.bfloat16) * 0.1
            valid = max(0, min(local_m, logical - self.rank * local_m))
            x[valid:].zero_()
            gathered = x.new_empty((local_m * self.world, 4096))
            dist.all_gather_into_tensor(gathered, x)
            baseline = all_gather_projections(x, projections, logical)
            actual = all_gather_kda_projections(x, projections, logical)
            for old, new, projection in zip(baseline, actual, projections):
                reference = torch.nn.functional.linear(
                    gathered[:logical].float(), projection.weight.float()
                )
                torch.testing.assert_close(
                    new.float(), reference, rtol=1 / 128, atol=2e-3
                )
                torch.testing.assert_close(new, old, rtol=1 / 128, atol=2e-3)
                self.assertTrue(new.is_contiguous())

    def test_stable_rs_padding_order_and_stream_handoff(self):
        lengths = [0, 1, self.world - 1, self.world + 1, 32769]
        if os.environ.get("GLM53_TEST_LARGE_STABLE_RS") == "1":
            lengths.append(1048576)
        stream = torch.cuda.Stream()
        for logical in lengths:
            with self.subTest(logical=logical):
                physical = (logical + self.world - 1) // self.world * self.world
                source = torch.randn(logical, 4096, device="cuda", dtype=torch.bfloat16)
                padded = source.new_zeros((physical, 4096))
                padded[:logical].copy_(source)
                if logical == 0:
                    self.assertEqual(reduce_scatter_glm53(source).shape, (0, 4096))
                    continue
                # NCCL transports each source's destination slice without doing
                # arithmetic. Sum those slices in explicit source rank order.
                received = torch.empty_like(padded)
                dist.all_to_all_single(received, padded)
                pieces = received.reshape(self.world, physical // self.world, 4096)
                expected = pieces[0].float()
                for piece in pieces[1:]:
                    expected.add_(piece.float())
                expected = expected.bfloat16()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    actual = reduce_scatter_glm53(source)
                torch.cuda.current_stream().wait_stream(stream)
                again = reduce_scatter_glm53(source)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(again, expected, rtol=0, atol=0)
                valid = max(
                    0, min(actual.shape[0], logical - self.rank * actual.shape[0])
                )
                self.assertEqual(torch.count_nonzero(actual[valid:]).item(), 0)
                if logical == 32769:
                    strided = torch.stack((source, source), dim=-1)[..., 0]
                    self.assertFalse(strided.is_contiguous())
                    torch.testing.assert_close(
                        reduce_scatter_glm53(strided), expected, rtol=0, atol=0
                    )

    def test_stable_rs_rejects_invalid_input_before_launch(self):
        for tensor in (
            torch.empty(1, 4096),
            torch.empty(1, 4096, device="cuda", dtype=torch.float32),
            torch.empty(1, 2048, device="cuda", dtype=torch.bfloat16),
            torch.empty(1, device="cuda", dtype=torch.bfloat16).expand(1048577, 4096),
        ):
            with self.assertRaises(ValueError):
                reduce_scatter_glm53(tensor)

    def test_stable_rs_interleaves_with_gemm_rs(self):
        # Dense FFN/shared RS and KDA GEMM/RS reuse the same staging area
        # within a model forward. Exercise both directions across streams.
        projection = CudaF16Linear(
            torch.randn(1024, 4096, device="cuda", dtype=torch.bfloat16) * 0.02
        )
        stream = torch.cuda.Stream()
        for logical in (32768, 32769, 65536):
            x = torch.randn(logical, 1024, device="cuda", dtype=torch.bfloat16) * 0.1
            source = projection(x)
            expected = project_reduce_scatter(x, projection)
            reduced = reduce_scatter_glm53(source)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                gemm_again = project_reduce_scatter(x, projection)
                rs_again = reduce_scatter_glm53(source)
            torch.cuda.current_stream().wait_stream(stream)
            final = project_reduce_scatter(x, projection)
            torch.testing.assert_close(gemm_again, expected, rtol=0, atol=0)
            torch.testing.assert_close(final, expected, rtol=0, atol=0)
            torch.testing.assert_close(rs_again, reduced, rtol=0, atol=0)

        from rtp_llm.models_py.distributed import glm53_collective_gemm

        workspace = glm53_collective_gemm._STATE.workspace
        alias = workspace.buffer[workspace._data_offset_bytes :][:8192]
        alias = alias.view(torch.bfloat16).reshape(1, 4096)
        with self.assertRaisesRegex(ValueError, "alias"):
            reduce_scatter_glm53(alias)


if __name__ == "__main__":
    unittest.main()

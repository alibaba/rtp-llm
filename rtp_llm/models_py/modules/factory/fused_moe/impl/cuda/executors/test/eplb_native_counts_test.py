import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    get_cutlass_moe_mm_without_permute_info,
)
from rtp_llm.models_py.utils.expert_load_monitor import ExpertLoadContext


class EplbNativeCountsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("This test requires a CUDA GPU")

    def _recorder(self, experts=4):
        return ExpertLoadContext(torch.zeros(experts, dtype=torch.int32, device="cuda"))

    def test_problem_size_kernel_records_counts_before_padding(self):
        recorder = self._recorder()
        ids = torch.tensor([[0, 1], [1, -1], [4, 3]], device="cuda")
        offsets = torch.empty(4, dtype=torch.int32, device="cuda")
        problem1 = torch.empty((4, 3), dtype=torch.int32, device="cuda")
        problem2 = torch.empty_like(problem1)

        def record(route_ids, swap1, swap2):
            get_cutlass_moe_mm_without_permute_info(
                route_ids,
                offsets,
                problem1,
                problem2,
                4,
                32,
                64,
                swap1,
                swap2,
                expert_token_counts=recorder.expert_token_counts,
            )

        for swap1 in (False, True):
            for swap2 in (False, True):
                recorder.expert_token_counts.zero_()
                record(ids, swap1, swap2)
                self.assertEqual(recorder.expert_token_counts.tolist(), [1, 2, 0, 1])
                self.assertEqual(problem1[:, int(swap1)].tolist(), [1, 2, 0, 1])
                self.assertEqual(problem2[:, int(swap2)].tolist(), [1, 2, 0, 1])
                self.assertEqual(offsets.tolist(), [0, 1, 3, 3])
        recorder.expert_token_counts.zero_()
        record(ids[:1], False, False)
        record(ids[1:], False, False)
        self.assertEqual(recorder.expert_token_counts.tolist(), [1, 2, 0, 1])
        record(ids[:0], False, False)
        self.assertEqual(recorder.expert_token_counts.tolist(), [1, 2, 0, 1])

    def test_triton_gemm_records_only_local_routes(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_executor import (
            TritonFusedMoeExecutor,
        )
        from rtp_llm.utils.model_weight import W

        for ep_size in (1, 2):
            rank = ep_size - 1
            config = SimpleNamespace(
                ep_size=ep_size,
                ep_rank=rank,
                expert_num=4,
                physical_expert_num=4,
                moe_k=2,
            )
            experts = 4 // ep_size
            executor = TritonFusedMoeExecutor(
                config,
                SimpleNamespace(),
                {
                    W.moe_w1: torch.randn(
                        experts, 64, 32, dtype=torch.bfloat16, device="cuda"
                    ),
                    W.moe_w2: torch.randn(
                        experts, 32, 32, dtype=torch.bfloat16, device="cuda"
                    ),
                },
            )
            recorder = self._recorder(experts=experts)
            ids = torch.tensor([[0, 2], [3, 2], [1, 0]], device="cuda")
            payload = ExpertForwardPayload(
                expert_x=torch.randn(3, 32, dtype=torch.bfloat16, device="cuda"),
                expert_topk_ids=ids,
                expert_topk_weights=torch.ones((3, 2), device="cuda"),
            )

            def run(expert_load):
                payload.expert_load = expert_load
                return executor.execute(
                    payload, "SiGLU", None, None, False, None
                ).fused_expert_output

            reference = run(None)
            actual = run(recorder)
            torch.testing.assert_close(actual, reference)
            expected = [2, 1, 2, 1] if ep_size == 1 else [2, 1]
            self.assertEqual(recorder.expert_token_counts.tolist(), expected)
            recorder.expert_token_counts.zero_()
            run(recorder)
            run(recorder)
            self.assertEqual(
                recorder.expert_token_counts.tolist(), [2 * c for c in expected]
            )
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(recorder)
            ids.fill_(rank * experts)
            for _ in range(2):
                recorder.expert_token_counts.zero_()
                graph.replay()
                self.assertEqual(
                    recorder.expert_token_counts.tolist(), [6] + [0] * (experts - 1)
                )

    def test_scatter_records_assignments_without_expert_alignment_padding(self):
        from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
            ep_scatter,
            ep_scatter_v2,
        )

        x = torch.ones((3, 128), dtype=torch.bfloat16, device="cuda")
        scale = torch.ones((3, 1), device="cuda")
        ids = torch.tensor([[0, 1], [-1, 1], [0, -1]], device="cuda")
        padded_counts = torch.tensor([128, 128], dtype=torch.int32, device="cuda")
        starts = torch.empty_like(padded_counts)
        output = torch.empty((256, 128), dtype=x.dtype, device="cuda")
        output_scale = torch.empty((256, 1), device="cuda")
        expert_ids = torch.empty((256,), dtype=torch.int32, device="cuda")
        output_index = torch.empty_like(ids)
        counts = torch.zeros((2,), dtype=torch.int32, device="cuda")

        for version in (1, 2):
            with self.subTest(scatter_version=version):

                def scatter():
                    if version == 1:
                        ep_scatter(
                            x,
                            scale,
                            ids,
                            padded_counts,
                            starts,
                            output,
                            output_scale,
                            expert_ids,
                            output_index,
                            expert_token_counts=counts,
                        )
                    else:
                        ep_scatter_v2(
                            x,
                            scale,
                            ids,
                            128,
                            starts,
                            output,
                            output_scale.view(2, 128, 1),
                            output_index,
                            expert_token_counts=counts,
                        )

                ids.copy_(torch.tensor([[0, 1], [-1, 1], [0, -1]], device="cuda"))
                counts.zero_()
                scatter()
                self.assertEqual(counts.cpu().tolist(), [2, 2])
                scatter()
                self.assertEqual(counts.cpu().tolist(), [4, 4])
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    scatter()
                for value, expected in ((-1, [0, 0]), (0, [6, 0]), (1, [0, 6])):
                    ids.fill_(value)
                    for _ in range(2):
                        counts.zero_()
                        graph.replay()
                        self.assertEqual(counts.cpu().tolist(), expected)


if __name__ == "__main__":
    unittest.main()

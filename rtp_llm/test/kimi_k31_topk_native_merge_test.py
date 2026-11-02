"""Exact bounded merge and prepare/Graph contracts; no model golden."""

import math
import unittest

import torch

from rtp_llm.models_py.modules.dsv4.indexer_topk import NativeExactIndexerTopKBackend
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_workspace import (
    DsaPhaseWorkspace,
    indexer_candidate_workspace_shapes,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class NativeRunningTopKTest(unittest.TestCase):
    def test_single_block_has_no_unused_candidate_pair(self):
        shapes = indexer_candidate_workspace_shapes(
            8192, 65536, 2048, prefix="block", native=True,
            needs_block_output=False,
        )
        self.assertEqual(shapes, {})

    def _fixture(self, topk):
        rows, width = 6, 2 * topk
        generator = torch.Generator().manual_seed(3101)
        score = torch.randn((rows, width), generator=generator).mul_(2).round_()
        score[0].zero_()
        score[0, ::2] = -0.0
        # A -0 coarse bin must not let zero-valued candidates displace the
        # larger values at the end of the row.
        score[1].fill_(-2.0)
        score[1, : topk + topk // 2] = -0.0
        score[1, : topk // 4] = 0.0
        score[1, -topk // 4 :] = 2.0
        # Exact midpoint of FP16 0xbc0f and 0xbc10. Round-to-even places it
        # below the upper bin; FP32 collect bounds must agree with that.
        score[2].fill_(-10.0)
        score[2, : topk + topk // 4] = -1.01513671875
        score[2, -topk // 4 :] = 2.0
        score[3, 19:] = float("nan")
        score[4].fill_(-float("inf"))
        positions = ((torch.arange(width, dtype=torch.int32) * 17) % 65536).repeat(rows, 1)
        expected = []
        for values, indices in zip(score.tolist(), positions.tolist()):
            selected = sorted(
                ((value, index) for value, index in zip(values, indices) if math.isfinite(value)),
                key=lambda pair: (-pair[0], pair[1]),
            )[:topk]
            expected.append(sorted([index for _, index in selected] + [-1] * (topk - len(selected))))
        return score.cuda(), positions.cuda(), torch.tensor(expected, device="cuda", dtype=torch.int32)

    def test_exact_members_with_prepared_buffers_and_graph(self):
        selector = NativeExactIndexerTopKBackend(stable_ties=True)
        for topk in (512, 1024, 2048):
            with self.subTest(topk=topk):
                score, positions, expected = self._fixture(topk)
                rows = score.shape[0]
                shapes = indexer_candidate_workspace_shapes(
                    rows, 2 * topk, topk, prefix="merge", native=True
                )
                self.assertNotIn("merge.keys", shapes)
                shapes.update({
                    "merge.input_scores": ((rows, 2 * topk), torch.float32),
                    "merge.input_positions": ((rows, 2 * topk), torch.int32),
                    "running.scores": ((rows, topk), torch.float32),
                    "running.positions": ((rows, topk), torch.int32),
                })
                workspace = DsaPhaseWorkspace("cuda")
                workspace.reserve(shapes)
                running = (
                    workspace("running.scores", (rows, topk), torch.float32),
                    workspace("running.positions", (rows, topk), torch.int32),
                )
                current = (score[:, topk:], positions[:, topk:])

                def reset():
                    running[0].copy_(score[:, :topk])
                    running[1].copy_(positions[:, :topk])

                reset()
                selector.merge_candidates(running, current, workspace=workspace)
                torch.testing.assert_close(running[1].sort(1).values, expected)
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated()
                reset()
                selector.merge_candidates(running, current, workspace=workspace)
                torch.cuda.synchronize()
                self.assertEqual(torch.cuda.memory_allocated(), allocated)
                reset()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    selector.merge_candidates(running, current, workspace=workspace)
                reset()
                graph.replay()
                torch.testing.assert_close(running[1].sort(1).values, expected)


if __name__ == "__main__":
    unittest.main()

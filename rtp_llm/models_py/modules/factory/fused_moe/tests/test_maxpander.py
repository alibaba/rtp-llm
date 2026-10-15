"""Static placement and GPU dispatch contracts; no model weights required."""

import unittest

import numpy as np
import torch

from rtp_llm.eplb.maxpander_plan import build_circulant_plan, draft_redundant_experts
from rtp_llm.models_py.modules.maxpander import MaxPanderRouter


class PlacementTest(unittest.TestCase):
    def test_draft_ratio_is_not_absolute_target_replica_count(self):
        self.assertEqual(draft_redundant_experts(256, 64, 128), 32)
        self.assertEqual(draft_redundant_experts(256, 64, 256), 64)
        self.assertEqual(draft_redundant_experts(256, 256, 128), 128)
        self.assertEqual(draft_redundant_experts(256, 0, 128), 0)
        with self.assertRaises(ValueError):
            draft_redundant_experts(256, 64, 7)

    def test_all_requested_ratios_and_ep_sizes(self):
        for g in (2, 4, 8):
            for p in (256, 320, 384, 512):
                with self.subTest(ep=g, physical=p):
                    phi, lp, cnt = build_circulant_plan(256, p, g)
                    np.testing.assert_array_equal(
                        phi, build_circulant_plan(256, p, g)[0]
                    )
                    self.assertEqual(len(phi), p)
                    self.assertEqual(int(cnt.sum()), p)
                    self.assertTrue(np.all((cnt >= 1) & (cnt <= 2)))
                    for e in range(256):
                        copies = lp[e, : cnt[e]]
                        np.testing.assert_array_equal(phi[copies], e)
                        self.assertEqual(len(set(copies // (p // g))), cnt[e])

    def test_invalid_layouts(self):
        for e, p, g in (
            (256, 320, 1),
            (256, 321, 4),
            (255, 320, 4),
            (256, 768, 4),
            (0, 0, 2),
            (256, 128, 2),
        ):
            with self.subTest(layout=(e, p, g)), self.assertRaises(ValueError):
                build_circulant_plan(e, p, g)

    def test_zero_redundancy_can_still_need_mapping(self):
        phi, _, _ = build_circulant_plan(256, 256, 2)
        self.assertFalse(np.array_equal(phi, np.arange(256)))


class DispatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This target is explicitly a GPU test: unavailable CUDA is a failure,
        # not a passing skip masking missing graph coverage.
        if not torch.cuda.is_available():
            raise RuntimeError("MaxPander dispatch tests require an allocated CUDA GPU")

    def make_router(self, e=16, p=20, g=2):
        phi, lp, cnt = build_circulant_plan(e, p, g)
        router = MaxPanderRouter(
            torch.tensor(lp, device="cuda", dtype=torch.int32),
            torch.tensor(cnt, device="cuda", dtype=torch.int32),
            num_logical=e,
            num_physical=p,
            ep_size=g,
        )
        return router, torch.tensor(phi, device="cuda"), cnt

    def check_mapping(self, ids, physical, phi):
        valid = (ids >= 0) & (ids < 16)
        self.assertTrue(torch.equal(phi[physical[valid].long()], ids[valid].long()))
        self.assertTrue(torch.all(physical[~valid] == -1).item())
        self.assertEqual(ids.dtype, physical.dtype)
        self.assertEqual(ids.shape, physical.shape)

    def test_mapping_sentinels_empty_and_quotas(self):
        for p in (16, 20, 24, 32):
            for dtype in (torch.int32, torch.int64):
                router, phi, _ = self.make_router(p=p)
                ids = (torch.arange(2048, device="cuda") % 19 - 1).to(dtype).view(-1, 4)
                physical = router(ids)
                self.check_mapping(ids, physical, phi)
                hist = torch.bincount(ids[(ids >= 0) & (ids < 16)].long(), minlength=16)
                self.assertTrue(torch.equal(router.quota.sum(1).long(), hist))
                empty = ids[:0]
                self.assertEqual(router(empty).shape, empty.shape)

    def test_forces_extra_physical_slots(self):
        router, phi, counts = self.make_router()
        e = int(np.flatnonzero(counts == 2)[0])
        ids = torch.full((128, 4), e, device="cuda", dtype=torch.int64)
        physical = router(ids)
        self.check_mapping(ids, physical, phi)
        self.assertEqual(physical.unique().numel(), 2)
        self.assertTrue(
            torch.equal(router.quota[e], torch.tensor([256, 256], device="cuda"))
        )

    def test_graph_replay_recomputes_routes(self):
        router, phi, _ = self.make_router()
        ids = torch.zeros((256, 4), device="cuda", dtype=torch.int64)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                router(ids)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            physical = router(ids)
        for value in (0, 5, 15, -1):
            ids.fill_(value)
            graph.replay()
            torch.cuda.synchronize()
            self.check_mapping(ids, physical, phi)

    def test_invalid_table_fails_before_dispatch(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            MaxPanderRouter(
                torch.tensor([[0, 1], [2, 3]], device="cuda"),
                torch.tensor([2, 2], device="cuda"),
                num_logical=2,
                num_physical=4,
                ep_size=2,
            )

    def test_flash_shapes_and_weighted_replica_equivalence(self):
        # Models CP rank-local subsets / decode microbatches without a real EP
        # collective. This is a routing contract, not native FP4 correctness.
        for g in (2, 4, 8):
            for p in (256, 320, 384, 512):
                router, phi, _ = self.make_router(e=256, p=p, g=g)
                logical = torch.arange(4096, device="cuda").view(-1, 8) % 256
                scores = torch.linspace(0.1, 1.0, 8, device="cuda").expand_as(logical)
                expert_outputs = torch.arange(256, device="cuda", dtype=torch.float32)
                physical = router(logical)
                self.assertTrue(torch.equal(phi[physical], logical))
                baseline = (expert_outputs[logical] * scores).sum(1)
                actual = (expert_outputs[phi][physical] * scores).sum(1)
                torch.testing.assert_close(actual, baseline, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

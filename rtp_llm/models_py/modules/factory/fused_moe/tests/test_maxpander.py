"""Static placement and GPU dispatch contracts; no model weights required."""

import unittest
from contextlib import ExitStack
from unittest import mock

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

    def make_router(self, e=16, p=20, g=2, *, seed=0, identity=False):
        phi, lp, cnt = build_circulant_plan(e, p, g, seed=seed)
        if identity:
            self.assertEqual(e, p)
            phi = np.arange(e, dtype=np.int64)
            lp = np.full((e, 2), -1, dtype=np.int64)
            lp[:, 0] = phi
        router = MaxPanderRouter(
            torch.tensor(lp, device="cuda", dtype=torch.int32),
            torch.tensor(cnt, device="cuda", dtype=torch.int32),
            num_logical=e,
            num_physical=p,
            ep_size=g,
        )
        return router, torch.tensor(phi, device="cuda"), cnt

    def direct_reference(self, logical_ids, physical_to_logical):
        """Invert physical placement independently for single-candidate WF routing."""
        logical_to_physical = torch.argsort(physical_to_logical)
        valid = (logical_ids >= 0) & (logical_ids < physical_to_logical.numel())
        expected = torch.full_like(logical_ids, -1)
        expected[valid] = logical_to_physical[logical_ids[valid].long()].to(
            logical_ids.dtype
        )
        return expected

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
                if p > router.num_logical:
                    hist = torch.bincount(
                        ids[(ids >= 0) & (ids < 16)].long(), minlength=16
                    )
                    self.assertTrue(torch.equal(router.quota.sum(1).long(), hist))
                else:
                    self.assertIsNone(router.quota)
                empty = ids[:0]
                self.assertEqual(router(empty).shape, empty.shape)

    def test_zero_redundancy_matches_independent_lookup(self):
        for num_logical, ep_size, dtype in (
            (256, 2, torch.int32),
            (256, 4, torch.int64),
            (384, 2, torch.int64),
            (384, 4, torch.int32),
        ):
            for seed, identity in ((0, False), (17, False), (41, True)):
                with self.subTest(
                    experts=num_logical,
                    ep=ep_size,
                    dtype=dtype,
                    seed=seed,
                    identity=identity,
                ):
                    router, physical_to_logical, _ = self.make_router(
                        e=num_logical,
                        p=num_logical,
                        g=ep_size,
                        seed=seed,
                        identity=identity,
                    )
                    self.assertEqual(
                        torch.equal(
                            physical_to_logical,
                            torch.arange(num_logical, device="cuda"),
                        ),
                        identity,
                    )
                    generator = torch.Generator(device="cuda").manual_seed(seed)
                    logical = torch.randint(
                        -2,
                        num_logical + 2,
                        (257, 7),
                        device="cuda",
                        dtype=dtype,
                        generator=generator,
                    )
                    logical.view(-1)[:num_logical] = torch.arange(
                        num_logical, device="cuda", dtype=dtype
                    )
                    original = logical.clone()
                    expected = self.direct_reference(logical, physical_to_logical)
                    torch.testing.assert_close(
                        router(logical), expected, rtol=0, atol=0
                    )
                    torch.testing.assert_close(logical, original, rtol=0, atol=0)
                    self.assertIs(router.remap_inplace(logical), logical)
                    torch.testing.assert_close(logical, expected, rtol=0, atol=0)

    def test_zero_redundancy_uses_only_direct_kernel_without_workspace(self):
        from rtp_llm.models_py.modules import maxpander_kernels

        router, physical_to_logical, _ = self.make_router(e=256, p=256)
        self.assertTrue(router._direct_mapping)
        self.assertEqual(set(dict(router.named_buffers())), {"cand_phys"})
        self.assertEqual(router.cand_phys.shape, (256, 2))
        for name in (
            "cand_gpu",
            "rep_count",
            "counts",
            "hot_order",
            "quota",
            "alloc",
            "rank_counter",
        ):
            self.assertIn(name, router._buffers)
            self.assertIsNone(getattr(router, name))
        logical = torch.arange(1799, device="cuda").view(257, 7) % 259 - 1
        expected = self.direct_reference(logical, physical_to_logical)
        packed = logical.clone()
        router.remap_inplace(packed)
        packed.copy_(logical)
        empty = packed[:0]
        candidate_address = router.cand_phys.data_ptr()
        direct_kernel = maxpander_kernels._direct_remap_kernel
        with ExitStack() as stack:
            for name in (
                "_wf_count_kernel",
                "_wf_count_hist_kernel",
                "_wf_hot_order_kernel",
                "_wf_prep_kernel",
                "_wf_quota_kernel",
                "_wf_scatter_kernel",
            ):
                stack.enter_context(mock.patch.object(maxpander_kernels, name, None))
            direct_mock = stack.enter_context(
                mock.patch.object(maxpander_kernels, "_direct_remap_kernel")
            )
            direct_mock.__getitem__.side_effect = direct_kernel.__getitem__
            for name in (
                "empty",
                "empty_like",
                "zeros",
                "zeros_like",
                "full",
                "full_like",
                "tensor",
            ):
                stack.enter_context(
                    mock.patch.object(
                        torch, name, side_effect=AssertionError("unexpected allocation")
                    )
                )
            for name in ("cpu", "item", "tolist", "numpy", "clone", "to"):
                stack.enter_context(
                    mock.patch.object(
                        torch.Tensor,
                        name,
                        side_effect=AssertionError("unexpected tensor conversion"),
                    )
                )
            stack.enter_context(
                mock.patch.object(
                    torch.cuda,
                    "synchronize",
                    side_effect=AssertionError("unexpected CPU synchronization"),
                )
            )
            self.assertIs(router.remap_inplace(packed), packed)
            self.assertIs(router.remap_inplace(empty), empty)
            direct_mock.__getitem__.assert_called_once()
        torch.testing.assert_close(packed, expected, rtol=0, atol=0)
        self.assertEqual(candidate_address, router.cand_phys.data_ptr())
        self.assertEqual(set(dict(router.named_buffers())), {"cand_phys"})

    def test_zero_redundancy_strides_empty_and_integer_edges(self):
        router, physical_to_logical, _ = self.make_router(e=384, p=384, g=4)
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                edge_values = [-1, -2, 0, 1, 383, 384, 385]
                edge_values.extend([torch.iinfo(dtype).min, torch.iinfo(dtype).max])
                if dtype == torch.int64:
                    edge_values.extend(
                        [
                            2**31,
                            2**32,
                            2**32 + 1,
                            -(2**31) - 1,
                            -(2**32),
                            -(2**32) + 1,
                            2**40 + 383,
                            -(2**40) + 383,
                        ]
                    )
                logical = torch.tensor(edge_values, dtype=dtype, device="cuda")
                logical = logical.repeat(1799 // len(edge_values) + 1)[:1799].view(
                    257, 7
                )
                storage = torch.full((515, 13), -123, device="cuda", dtype=dtype)
                packed = storage[1::2, 2:9]
                packed.copy_(logical)
                expected_storage = storage.clone()
                expected_storage[1::2, 2:9] = self.direct_reference(
                    logical, physical_to_logical
                )
                self.assertIs(router.remap_inplace(packed), packed)
                torch.testing.assert_close(storage, expected_storage, rtol=0, atol=0)
                for empty in (packed[:0], packed[:, :0]):
                    self.assertIs(router.remap_inplace(empty), empty)
                    self.assertEqual(router(empty).shape, empty.shape)

    def test_zero_redundancy_graph_replay_reloads_live_inputs(self):
        for num_logical, ep_size, dtype in (
            (256, 2, torch.int32),
            (384, 4, torch.int64),
        ):
            with self.subTest(experts=num_logical, ep=ep_size, dtype=dtype):
                router, physical_to_logical, _ = self.make_router(
                    e=num_logical, p=num_logical, g=ep_size, seed=17
                )
                source = torch.zeros((257, 7), device="cuda", dtype=dtype)
                storage = torch.full((257, 11), -123, device="cuda", dtype=dtype)
                packed = storage[:, 1:8]
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    packed.copy_(source)
                    router.remap_inplace(packed)
                    router(source)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    packed.copy_(source)
                    router.remap_inplace(packed)
                    physical = router(source)
                replay_values = [0, num_logical - 1, -1, num_logical]
                if dtype == torch.int64:
                    replay_values.extend([2**32 + 1, -(2**32) + 1])
                for value in replay_values:
                    source.fill_(value)
                    graph.replay()
                    torch.cuda.synchronize()
                    expected = self.direct_reference(source, physical_to_logical)
                    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
                    torch.testing.assert_close(physical, expected, rtol=0, atol=0)
                    self.assertTrue(torch.all(storage[:, :1] == -123).item())
                    self.assertTrue(torch.all(storage[:, 8:] == -123).item())

    def test_float32_count_limit_applies_only_to_redundancy(self):
        direct_router, physical_to_logical, _ = self.make_router(e=256, p=256)
        redundant_router, _, _ = self.make_router(e=256, p=320)
        logical = torch.full((2**22 + 1, 4), 7, device="cuda", dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "exact float32 integer range"):
            redundant_router.remap_inplace(logical)
        self.assertIs(direct_router.remap_inplace(logical), logical)
        expected_physical = torch.argsort(physical_to_logical)[7]
        self.assertTrue(torch.all(logical == expected_physical).item())

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

    def test_inplace_strided_rows_and_repeated_calls(self):
        router, phi, _ = self.make_router()
        storage = torch.full((257, 9), -123, device="cuda", dtype=torch.int64)
        packed = storage[:, 2:6]
        scratch_addresses = [
            router.counts.data_ptr(),
            router.hot_order.data_ptr(),
        ]
        for value in (0, 7, 15, -1, 19):
            logical = torch.full_like(packed, value)
            packed.copy_(logical)
            result = router.remap_inplace(packed)
            self.assertIs(result, packed)
            self.check_mapping(logical, packed, phi)
            self.assertTrue(torch.all(storage[:, :2] == -123))
            self.assertTrue(torch.all(storage[:, 6:] == -123))
            self.assertEqual(
                scratch_addresses,
                [
                    router.counts.data_ptr(),
                    router.hot_order.data_ptr(),
                ],
            )

    def test_inplace_graph_replay_reloads_logical_ids(self):
        router, phi, _ = self.make_router()
        source = torch.zeros((64, 4), device="cuda", dtype=torch.int64)
        storage = torch.empty((64, 8), device="cuda", dtype=torch.int64)
        packed = storage[:, :4]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            packed.copy_(source)
            router.remap_inplace(packed)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            packed.copy_(source)
            router.remap_inplace(packed)
        for value in (0, 5, 15, -1):
            source.fill_(value)
            graph.replay()
            torch.cuda.synchronize()
            self.check_mapping(source, packed, phi)

    def test_invalid_inplace_layout_rejected(self):
        for num_physical in (16, 20):
            router, _, _ = self.make_router(p=num_physical)
            with self.subTest(physical=num_physical):
                with self.assertRaisesRegex(ValueError, "contiguous columns"):
                    router.remap_inplace(
                        torch.zeros((8, 4), device="cuda", dtype=torch.int64).t()
                    )
                with self.assertRaisesRegex(ValueError, "overlap"):
                    router.remap_inplace(
                        torch.zeros((1, 4), device="cuda", dtype=torch.int64).expand(
                            8, 4
                        )
                    )

    def test_hot_order_matches_stable_sort_with_ties_and_padding(self):
        for experts in (24, 256):
            router, _, _ = self.make_router(e=experts, p=experts + experts // 4)
            for logical in (
                torch.full((257, 6), -1, device="cuda", dtype=torch.int64),
                torch.arange(1536, device="cuda").view(-1, 6) % experts,
                torch.randint(-1, experts + 2, (257, 6), device="cuda"),
            ):
                router(logical)
                expected = torch.argsort(router.counts, descending=True, stable=True)
                self.assertTrue(torch.equal(router.hot_order.long(), expected))

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

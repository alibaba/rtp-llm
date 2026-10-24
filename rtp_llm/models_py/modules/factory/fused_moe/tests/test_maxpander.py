"""Static placement and GPU dispatch contracts; no model weights required."""

import unittest
import weakref
from contextlib import ExitStack
from itertools import product
from unittest import mock

import numpy as np
import torch
import triton
import triton.language as tl

from rtp_llm.eplb.maxpander_plan import build_circulant_plan, draft_redundant_experts
from rtp_llm.models_py.modules.factory.fused_moe.tests.maxpander_reference import (
    LegacyFullWF,
)
from rtp_llm.models_py.modules.maxpander import MaxPanderRouter
from rtp_llm.models_py.modules.maxpander_kernels import _round2

_WIDE_INVALID_INT64_IDS = (
    2**31,
    2**32,
    2**32 + 1,
    2**32 + 4,
    -(2**31) - 1,
    -(2**32),
    -(2**32) + 1,
    2**40 + 4,
    -(2**40) + 4,
    -(2**63),
    2**63 - 1,
)


@triton.jit
def _round2_test_kernel(
    demands_ptr,
    replicas_ptr,
    allocations_ptr,
    quotas_ptr,
    num_cases,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    valid = offsets < num_cases
    demand = tl.load(demands_ptr + offsets, mask=valid, other=0)
    replicas = tl.load(replicas_ptr + offsets, mask=valid, other=1)
    allocation0 = tl.load(allocations_ptr + offsets * 2, mask=valid, other=0)
    allocation1 = tl.load(allocations_ptr + offsets * 2 + 1, mask=valid, other=0)
    quota0, quota1 = _round2(demand, replicas, allocation0, allocation1)
    tl.store(quotas_ptr + offsets * 2, quota0, mask=valid)
    tl.store(quotas_ptr + offsets * 2 + 1, quota1, mask=valid)


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

    def check_split_only_counter(self, logical, quota, rank_counter):
        experts = quota.size(0)
        valid = (logical >= 0) & (logical < experts)
        counts = torch.bincount(logical[valid].long(), minlength=experts)
        torch.testing.assert_close(quota.sum(1), counts, rtol=0, atol=0)
        split = (quota[:, 0] > 0) & (quota[:, 1] > 0)
        expected = torch.where(split, counts, 0).to(rank_counter.dtype)
        torch.testing.assert_close(rank_counter, expected, rtol=0, atol=0)

    def check_routing_counts_and_quotas(
        self, logical, packed, router, physical_to_logical
    ):
        valid = (logical >= 0) & (logical < router.num_logical)
        expected_counts = torch.bincount(
            logical[valid].long(), minlength=router.num_logical
        )
        torch.testing.assert_close(
            router.counts, expected_counts.float(), rtol=0, atol=0
        )
        self.check_split_only_counter(logical, router.quota, router.rank_counter)
        expected_order = torch.argsort(expected_counts, descending=True, stable=True)
        torch.testing.assert_close(
            router.hot_order.long(), expected_order, rtol=0, atol=0
        )
        active = expected_counts[expected_order] > 0
        refinable = active & (router.rep_count[expected_order] == 2)
        expected_work_counts = torch.stack((active.sum(), refinable.sum())).int()
        torch.testing.assert_close(
            router.work_counts, expected_work_counts, rtol=0, atol=0
        )
        expected_refine_order = expected_order[refinable].int()
        torch.testing.assert_close(
            router.refine_order[: expected_refine_order.numel()],
            expected_refine_order,
            rtol=0,
            atol=0,
        )
        self.assertTrue(torch.all(packed[~valid] == -1))
        physical_ids = packed[valid].long()
        self.assertTrue(
            torch.all(
                (physical_ids >= 0) & (physical_ids < physical_to_logical.numel())
            )
        )
        torch.testing.assert_close(
            physical_to_logical[physical_ids], logical[valid].long(), rtol=0, atol=0
        )
        expected_histogram = torch.zeros_like(physical_to_logical, dtype=torch.int64)
        present = router.cand_phys >= 0
        expected_histogram[router.cand_phys[present].long()] = router.quota[
            present
        ].long()
        torch.testing.assert_close(
            torch.bincount(physical_ids, minlength=physical_to_logical.numel()),
            expected_histogram,
            rtol=0,
            atol=0,
        )

    def test_partial_count_workspace_is_bounded_and_nonpersistent(self):
        from rtp_llm.models_py.modules import maxpander_kernels

        self.assertEqual(maxpander_kernels.MAX_PARTITIONS, 64)
        self.assertEqual(maxpander_kernels.COUNT_BLOCK_SIZE, 1024)
        self.assertEqual(maxpander_kernels.REDUCE_PARTITION_TILE, 4)
        for experts, physical in ((16, 20), (384, 480), (768, 960), (1536, 1920)):
            with self.subTest(experts=experts):
                router = self.make_router(e=experts, p=physical, g=4)[0]
                self.assertIs(
                    dict(router.named_buffers())["partial_counts"],
                    router.partial_counts,
                )
                self.assertEqual(
                    router.partial_counts.shape, (64, triton.next_power_of_2(experts))
                )
                self.assertEqual(router.partial_counts.dtype, torch.int32)
                self.assertEqual(router.partial_counts.device, router.counts.device)
                self.assertTrue(router.partial_counts.is_contiguous())
                self.assertIn("partial_counts", router._non_persistent_buffers_set)
                self.assertNotIn("partial_counts", router.state_dict())

    def test_partial_counts_ignore_poison_and_cover_more_than_64_blocks(self):
        cases = (
            (1, 1, "mixed"),
            (1025, 1, "mixed"),
            (4097, 1, "sparse"),
            (65535, 1, "hot"),
            (65536, 1, "invalid"),
            (65537, 1, "mixed"),
            (16385, 5, "sparse"),
            (1025, 1, "invalid"),
            (65537, 1, "hot"),
        )
        poison = torch.iinfo(torch.int32).max
        for (experts, physical), dtype in product(
            ((384, 480), (768, 960)), (torch.int32, torch.int64)
        ):
            router, physical_to_logical, replicas = self.make_router(
                e=experts, p=physical, g=4
            )
            reference = LegacyFullWF(
                router.cand_phys,
                router.rep_count,
                num_logical=experts,
                num_physical=physical,
                ep_size=4,
                wf_passes=router.passes,
            )
            replicated_expert = int(np.flatnonzero(replicas == 2)[0])
            for rows, topk, pattern in cases:
                with self.subTest(
                    experts=experts, dtype=dtype, rows=rows, pattern=pattern
                ):
                    numel = rows * topk
                    offsets = torch.arange(numel, device="cuda", dtype=dtype)
                    logical = (offsets * 13 % (experts + 3) - 1).view(rows, topk)
                    if pattern == "hot":
                        logical.fill_(replicated_expert)
                    elif pattern == "sparse":
                        logical.copy_((offsets * 7 % 16).view(rows, topk))
                    elif pattern == "invalid":
                        invalid_ids = (-1, experts, experts + 3)
                        if dtype == torch.int64:
                            invalid_ids += _WIDE_INVALID_INT64_IDS
                        invalid_values = torch.tensor(
                            invalid_ids, device="cuda", dtype=dtype
                        )
                        logical.copy_(
                            invalid_values[offsets.long() % len(invalid_ids)].view(
                                rows, topk
                            )
                        )
                    storage = torch.full(
                        (rows * 2 + 1, topk + 4), -777, device="cuda", dtype=dtype
                    )
                    packed = storage[1::2, 2 : topk + 2]
                    packed.copy_(logical)
                    router.partial_counts.fill_(poison)
                    router.counts.fill_(float("nan"))
                    router.rank_counter.fill_(-777)
                    router.alloc.fill_(-777)
                    router.quota.fill_(-777)
                    self.assertIs(router.remap_inplace(packed), packed)
                    self.check_routing_counts_and_quotas(
                        logical, packed, router, physical_to_logical
                    )
                    valid = (logical >= 0) & (logical < experts)
                    expected = torch.where(valid, logical, -1)
                    reference.remap_inplace(expected)
                    for name in ("counts", "hot_order", "quota", "alloc"):
                        torch.testing.assert_close(
                            getattr(router, name),
                            getattr(reference, name),
                            rtol=0,
                            atol=0,
                        )
                    partitions = (
                        min(64, triton.cdiv(numel, 1024))
                        if numel > 1024 or experts > 512
                        else 0
                    )
                    if partitions:
                        partition_ids = offsets.long() // 1024 % partitions
                        bins = router.partial_counts.size(1)
                        histogram_ids = (
                            partition_ids[valid.view(-1)] * bins + logical[valid].long()
                        )
                        expected_partial = (
                            torch.bincount(histogram_ids, minlength=partitions * bins)
                            .view(partitions, bins)
                            .int()
                        )
                        torch.testing.assert_close(
                            router.partial_counts[:partitions],
                            expected_partial,
                            rtol=0,
                            atol=0,
                        )
                    self.assertTrue(
                        torch.all(router.partial_counts[partitions:] == poison)
                    )
                    self.assertTrue(torch.all(storage[::2] == -777))
                    self.assertTrue(torch.all(storage[:, :2] == -777))
                    self.assertTrue(torch.all(storage[:, topk + 2 :] == -777))

    def test_round2_preserves_exact_integer_demands_near_float32_limit(self):
        cases = [
            (0, 1, 0.0, 0.0, (0, 0)),
            (1, 2, 0.5, 0.5, (1, 0)),
            (3, 2, 1.25, 1.75, (1, 2)),
            (3, 2, 1.75, 1.25, (2, 1)),
        ]
        for demand in (2**23 - 1, 2**23, 2**23 + 1, 2**23 + 2, 2**24 - 1, 2**24):
            cases.extend(
                (
                    (demand, 1, float(demand), 0.0, (demand, 0)),
                    (demand, 2, float(demand), 0.0, (demand, 0)),
                    (demand, 2, 0.0, float(demand), (0, demand)),
                    (
                        demand,
                        2,
                        demand / 2,
                        demand / 2,
                        ((demand + 1) // 2, demand // 2),
                    ),
                )
            )
        demands = torch.tensor(
            [case[0] for case in cases], device="cuda", dtype=torch.float32
        )
        replicas = torch.tensor(
            [case[1] for case in cases], device="cuda", dtype=torch.int32
        )
        allocations = torch.tensor(
            [(case[2], case[3]) for case in cases], device="cuda", dtype=torch.float32
        )
        expected = torch.tensor(
            [case[4] for case in cases], device="cuda", dtype=torch.int32
        )
        quotas = torch.empty_like(expected)
        _round2_test_kernel[(1,)](
            demands,
            replicas,
            allocations,
            quotas,
            len(cases),
            BLOCK_SIZE=triton.next_power_of_2(len(cases)),
        )
        torch.testing.assert_close(quotas, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            quotas.sum(1),
            torch.tensor([case[0] for case in cases], device="cuda", dtype=torch.int64),
            rtol=0,
            atol=0,
        )

    def test_int32_wide_stride_routing_uses_int64_element_addresses(self):
        layouts = ((3, 4, 2**30 + 16), (257, 4, 2**23 + 16))
        backing_elements = max(
            (rows - 1) * row_stride + topk + 4 for rows, topk, row_stride in layouts
        )
        required_bytes = backing_elements * 4
        headroom_bytes = 512 * 1024**2
        device = torch.device("cuda", torch.cuda.current_device())
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        if free_bytes < required_bytes + headroom_bytes:
            self.skipTest(
                f"wide-stride int32 regression requires {required_bytes} backing bytes "
                f"plus {headroom_bytes} headroom; "
                f"only {free_bytes}/{total_bytes} bytes free"
            )
        try:
            backing = torch.empty(backing_elements, device=device, dtype=torch.int32)
        except torch.cuda.OutOfMemoryError as error:
            self.skipTest(
                f"wide-stride int32 backing allocation of {required_bytes} bytes "
                f"failed: {error}"
            )
        router, physical_to_logical, replicas = self.make_router()
        reference = LegacyFullWF(
            router.cand_phys,
            router.rep_count,
            num_logical=16,
            num_physical=20,
            ep_size=2,
            wf_passes=router.passes,
        )
        replicated_expert = int(np.flatnonzero(replicas == 2)[0])
        for rows, topk, row_stride in layouts:
            sampled_rows = backing.as_strided((rows, topk + 4), (row_stride, 1))
            packed = sampled_rows[:, 2 : topk + 2]
            self.assertLess(row_stride, 2**31)
            self.assertGreater((rows - 1) * row_stride + topk - 1, 2**31)
            for pattern in ("mixed", "hot"):
                with self.subTest(rows=rows, topk=topk, pattern=pattern):
                    logical = (
                        torch.arange(rows * topk, device=device, dtype=torch.int32)
                        * 13
                        % (router.num_logical + 3)
                        - 1
                    ).view(rows, topk)
                    if pattern == "hot":
                        logical.fill_(replicated_expert)
                    sampled_rows.fill_(-777)
                    packed.copy_(logical)
                    valid = (logical >= 0) & (logical < router.num_logical)
                    expected_counts = torch.bincount(
                        logical[valid].long(), minlength=router.num_logical
                    ).float()
                    router.counts.fill_(float("nan"))
                    router.partial_counts.fill_(torch.iinfo(torch.int32).max)
                    self.assertIs(router.remap_inplace(packed), packed)
                    torch.testing.assert_close(
                        router.counts, expected_counts, rtol=0, atol=0
                    )
                    self.check_routing_counts_and_quotas(
                        logical, packed, router, physical_to_logical
                    )
                    reference.remap_inplace(logical.clone())
                    for name in ("counts", "hot_order", "quota", "alloc"):
                        torch.testing.assert_close(
                            getattr(router, name),
                            getattr(reference, name),
                            rtol=0,
                            atol=0,
                        )
                    self.assertTrue(torch.all(sampled_rows[:, :2] == -777))
                    self.assertTrue(torch.all(sampled_rows[:, topk + 2 :] == -777))

    def test_active_worklists_preserve_full_wf_results(self):
        generator = torch.Generator(device="cuda").manual_seed(271)
        layouts = (
            (16, 20, 2),
            (12, 15, 3),
            (256, 320, 2),
            (384, 480, 4),
            (384, 768, 4),
            (384, 480, 8),
            (1536, 1920, 8),
        )
        for experts, physical, ep_size in layouts:
            placement, candidates, replicas = build_circulant_plan(
                experts, physical, ep_size, seed=11
            )
            candidate_table = torch.tensor(candidates, device="cuda", dtype=torch.int32)
            replica_counts = torch.tensor(replicas, device="cuda", dtype=torch.int32)
            physical_to_logical = torch.tensor(placement, device="cuda")
            dense = torch.randint(experts, (257, 6), device="cuda", generator=generator)
            dense[0] = torch.tensor(
                [-1, experts, 0, experts - 1, -1, experts + 3], device="cuda"
            )
            sparse = dense.remainder(min(experts, 16))
            patterns = [dense, sparse, torch.full_like(dense, -1)]
            single_copy = np.flatnonzero(replicas == 1)
            if len(single_copy):
                patterns.append(torch.full_like(dense, int(single_copy[0])))
            patterns.append(sparse)
            for passes in (1, 2, 4, 8, 64):
                arguments = dict(
                    num_logical=experts,
                    num_physical=physical,
                    ep_size=ep_size,
                    wf_passes=passes,
                )
                router = MaxPanderRouter(candidate_table, replica_counts, **arguments)
                reference = LegacyFullWF(candidate_table, replica_counts, **arguments)
                for dtype in (torch.int32, torch.int64):
                    storage = torch.full((257, 10), -777, device="cuda", dtype=dtype)
                    reference_storage = torch.full_like(storage, -777)
                    routed = storage[:, 2:8]
                    expected = reference_storage[:, 2:8]
                    for pattern_index, logical in enumerate(patterns):
                        with self.subTest(
                            experts=experts,
                            physical=physical,
                            ep=ep_size,
                            passes=passes,
                            dtype=dtype,
                            pattern=pattern_index,
                        ):
                            routed.copy_(logical)
                            expected.copy_(logical)
                            router.remap_inplace(routed)
                            reference.remap_inplace(expected)
                            for name in (
                                "counts",
                                "hot_order",
                                "quota",
                                "alloc",
                            ):
                                self.assertTrue(
                                    torch.equal(
                                        getattr(router, name), getattr(reference, name)
                                    ),
                                    name,
                                )
                            self.check_split_only_counter(
                                logical, router.quota, router.rank_counter
                            )
                            valid = (logical >= 0) & (logical < experts)
                            self.assertTrue(torch.all(routed[~valid] == -1))
                            self.assertTrue(
                                torch.equal(
                                    physical_to_logical[routed[valid].long()],
                                    logical[valid],
                                )
                            )
                            self.assertTrue(
                                torch.equal(
                                    torch.bincount(
                                        routed[valid].long(), minlength=physical
                                    ),
                                    torch.bincount(
                                        expected[valid].long(), minlength=physical
                                    ),
                                )
                            )
                            ordered = router.hot_order.long()
                            active = router.counts[ordered] > 0
                            refinable = active & (replica_counts[ordered] == 2)
                            active_count, refine_count = router.work_counts.tolist()
                            self.assertEqual(active_count, int(active.sum()))
                            self.assertEqual(refine_count, int(refinable.sum()))
                            self.assertTrue(
                                torch.equal(
                                    router.refine_order[:refine_count],
                                    router.hot_order[refinable],
                                )
                            )
                            self.assertTrue(torch.all(storage[:, :2] == -777))
                            self.assertTrue(torch.all(storage[:, 8:] == -777))

    def test_fused_count_order_boundaries_preserve_full_wf_results(self):
        from rtp_llm.models_py.modules import maxpander_kernels

        shapes = ((341, 3), (256, 4), (205, 5))
        layouts = ((384, 480, 4), (512, 640, 4), (768, 960, 4))
        for experts, physical, ep_size in layouts:
            placement, candidates, replicas = build_circulant_plan(
                experts, physical, ep_size, seed=11
            )
            candidate_table = torch.tensor(candidates, device="cuda", dtype=torch.int32)
            replica_counts = torch.tensor(replicas, device="cuda", dtype=torch.int32)
            physical_to_logical = torch.tensor(placement, device="cuda")
            for (rows, topk), dtype, passes in product(
                shapes, (torch.int32, torch.int64), (1, 2)
            ):
                arguments = dict(
                    num_logical=experts,
                    num_physical=physical,
                    ep_size=ep_size,
                    wf_passes=passes,
                )
                router = MaxPanderRouter(candidate_table, replica_counts, **arguments)
                reference = LegacyFullWF(candidate_table, replica_counts, **arguments)
                numel = rows * topk
                logical = (
                    torch.arange(numel, device="cuda", dtype=dtype) * 13 % (experts + 2)
                    - 1
                ).view(rows, topk)
                storage = torch.full(
                    (rows * 2 + 1, topk + 4), -777, device="cuda", dtype=dtype
                )
                reference_storage = torch.full_like(storage, -777)
                routed = storage[1::2, 2 : topk + 2]
                expected = reference_storage[1::2, 2 : topk + 2]
                with self.subTest(
                    experts=experts, numel=numel, dtype=dtype, passes=passes
                ), ExitStack() as stack:
                    dispatches = {
                        name: stack.enter_context(
                            mock.patch.object(
                                getattr(maxpander_kernels, name),
                                "run",
                                wraps=getattr(maxpander_kernels, name).run,
                            )
                        )
                        for name in (
                            "_wf_count_order_kernel",
                            "_wf_partial_count_kernel",
                            "_wf_reduce_order_kernel",
                        )
                    }
                    for pattern in (logical, torch.full_like(logical, -1), logical):
                        routed.copy_(pattern)
                        expected.copy_(pattern)
                        router.remap_inplace(routed)
                        reference.remap_inplace(expected)
                        for name in ("counts", "hot_order", "quota", "alloc"):
                            self.assertTrue(
                                torch.equal(
                                    getattr(router, name), getattr(reference, name)
                                ),
                                name,
                            )
                        self.check_split_only_counter(
                            pattern, router.quota, router.rank_counter
                        )
                        valid = (pattern >= 0) & (pattern < experts)
                        self.assertTrue(torch.all(routed[~valid] == -1))
                        self.assertTrue(
                            torch.equal(
                                physical_to_logical[routed[valid].long()],
                                pattern[valid],
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                torch.bincount(
                                    routed[valid].long(), minlength=physical
                                ),
                                torch.bincount(
                                    expected[valid].long(), minlength=physical
                                ),
                            )
                        )
                        ordered = router.hot_order.long()
                        active = router.counts[ordered] > 0
                        refinable = active & (replica_counts[ordered] == 2)
                        active_count, refine_count = router.work_counts.tolist()
                        self.assertEqual(active_count, int(active.sum()))
                        self.assertEqual(refine_count, int(refinable.sum()))
                        self.assertTrue(
                            torch.equal(
                                router.refine_order[:refine_count],
                                router.hot_order[refinable],
                            )
                        )
                        self.assertTrue(torch.all(storage[::2] == -777))
                        self.assertTrue(torch.all(storage[:, :2] == -777))
                        self.assertTrue(torch.all(storage[:, topk + 2 :] == -777))
                    expected_kernels = (
                        {"_wf_count_order_kernel"}
                        if experts <= 512 and numel <= 1024
                        else {"_wf_partial_count_kernel", "_wf_reduce_order_kernel"}
                    )
                    for name, dispatch in dispatches.items():
                        self.assertEqual(
                            dispatch.call_count, int(name in expected_kernels), name
                        )

    def test_scatter_split_only_counter_and_exact_physical_histogram(self):
        from rtp_llm.models_py.modules.maxpander_kernels import _wf_scatter_kernel

        candidates = torch.tensor(
            [[0, -1], [1, 4], [2, 5], [3, 6], [7, -1]],
            device="cuda",
            dtype=torch.int32,
        )
        physical_to_logical = torch.tensor([0, 1, 2, 3, 1, 2, 3, 4], device="cuda")
        cases = (
            ("single_replica", (0, 4)),
            ("first_replica_only", (1,)),
            ("second_replica_only", (2,)),
            ("split_replicas", (3,)),
            ("mixed", (0, 1, 2, 3, 4)),
            ("invalid", ()),
        )
        for dtype in (torch.int32, torch.int64):
            invalid_ids = [-1, -2, 5, 9, torch.iinfo(dtype).min, torch.iinfo(dtype).max]
            if dtype == torch.int64:
                invalid_ids.extend(_WIDE_INVALID_INT64_IDS)
            storage = torch.full((515, 13), -777, device="cuda", dtype=dtype)
            packed = storage[1::2, 2:9]
            rank_counter = torch.zeros(5, device="cuda", dtype=torch.int32)
            for name, experts in cases:
                values = torch.tensor(
                    (*experts, *invalid_ids), device="cuda", dtype=dtype
                )
                logical = values.repeat(packed.numel() // values.numel() + 1)[
                    : packed.numel()
                ].view_as(packed)
                valid = (logical >= 0) & (logical < 5)
                counts = torch.bincount(logical[valid].long(), minlength=5)
                quota = torch.stack((counts, torch.zeros_like(counts)), dim=1).int()
                quota[2, 0] = 0
                quota[2, 1] = counts[2]
                quota[3, 0] = counts[3] // 3
                quota[3, 1] = counts[3] - quota[3, 0]
                expected_histogram = torch.zeros(8, device="cuda", dtype=torch.int64)
                present = candidates >= 0
                expected_histogram[candidates[present].long()] = quota[present].long()
                for block_size in (128, 1024):
                    with self.subTest(dtype=dtype, case=name, block=block_size):
                        packed.copy_(logical)
                        rank_counter.zero_()
                        _wf_scatter_kernel[
                            ((packed.numel() + block_size - 1) // block_size,)
                        ](
                            packed,
                            quota,
                            candidates,
                            rank_counter,
                            packed,
                            5,
                            packed.numel(),
                            packed.stride(0),
                            TOPK=packed.size(1),
                            BLOCK_SIZE=block_size,
                        )
                        self.check_split_only_counter(logical, quota, rank_counter)
                        self.assertTrue(torch.all(packed[~valid] == -1))
                        self.assertTrue(
                            torch.equal(
                                physical_to_logical[packed[valid].long()],
                                logical[valid],
                            )
                        )
                        torch.testing.assert_close(
                            torch.bincount(packed[valid].long(), minlength=8),
                            expected_histogram,
                            rtol=0,
                            atol=0,
                        )
                        self.assertTrue(torch.all(storage[::2] == -777))
                        self.assertTrue(torch.all(storage[:, :2] == -777))
                        self.assertTrue(torch.all(storage[:, 9:] == -777))

    def test_partial_counts_reject_wide_int64_invalid_ids(self):
        from rtp_llm.models_py.modules.maxpander_kernels import _wf_partial_count_kernel

        for experts in ((0, 1, 4), ()):
            values = torch.tensor(
                (*experts, -1, -2, 5, 9, *_WIDE_INVALID_INT64_IDS),
                device="cuda",
                dtype=torch.int64,
            )
            logical = values.repeat(1799 // values.numel() + 1)[:1799]
            valid = (logical >= 0) & (logical < 5)
            expected = torch.bincount(logical[valid], minlength=5).float()
            for block_size in (128, 1024):
                with self.subTest(experts=experts, block=block_size):
                    partitions = min(4, triton.cdiv(logical.numel(), block_size))
                    partial_counts = torch.empty(
                        (partitions, 8), device="cuda", dtype=torch.int32
                    )
                    _wf_partial_count_kernel[(partitions,)](
                        logical,
                        partial_counts,
                        5,
                        logical.numel(),
                        1,
                        TOPK=1,
                        BLOCK_SIZE=block_size,
                        BLOCK_EXPERTS=8,
                    )
                    counts = partial_counts.sum(0)[:5].float()
                    torch.testing.assert_close(counts, expected, rtol=0, atol=0)

    def test_wide_int64_invalid_ids_preserve_counts_quotas_and_graph_replay(self):
        shapes = ((341, 3), (256, 4), (205, 5), (257, 7))
        for capture, (rows, topk) in product((False, True), shapes):
            numel = rows * topk
            router, physical_to_logical, replicas = self.make_router()
            single = int(np.flatnonzero(replicas == 1)[0])
            double = int(np.flatnonzero(replicas == 2)[0])
            values = torch.tensor(
                (-1, -2, 16, 19, *_WIDE_INVALID_INT64_IDS),
                device="cuda",
                dtype=torch.int64,
            )
            invalid = values.repeat(numel // values.numel() + 1)[:numel].view(
                rows, topk
            )
            source = invalid.clone()
            storage = torch.full(
                (rows * 2 + 1, topk + 4), -777, device="cuda", dtype=torch.int64
            )
            packed = storage[1::2, 2 : topk + 2]
            if capture:
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
            for expert, count in (
                (single, 0),
                (single, 513),
                (double, 514),
                (double, 513),
                (double, 0),
                (double, 514),
            ):
                with self.subTest(
                    capture=capture, numel=numel, expert=expert, count=count
                ):
                    source.copy_(invalid)
                    source.view(-1)[:count].fill_(expert)
                    if capture:
                        graph.replay()
                        torch.cuda.synchronize()
                    else:
                        packed.copy_(source)
                        router.remap_inplace(packed)
                    expected_counts = torch.zeros_like(router.counts)
                    expected_counts[expert] = count
                    expected_quota = torch.zeros_like(router.quota)
                    expected_alloc = torch.zeros_like(router.alloc)
                    copies = int(replicas[expert])
                    expected_alloc[expert, :copies] = count / copies
                    if copies == 1:
                        expected_quota[expert, 0] = count
                    else:
                        expected_quota[expert, 0] = (count + 1) // 2
                        expected_quota[expert, 1] = count // 2
                    torch.testing.assert_close(
                        router.counts, expected_counts, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        router.quota, expected_quota, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        router.alloc, expected_alloc, rtol=0, atol=0
                    )
                    self.check_split_only_counter(
                        source, router.quota, router.rank_counter
                    )
                    valid = (source >= 0) & (source < 16)
                    self.assertTrue(torch.all(packed[~valid] == -1))
                    self.assertTrue(
                        torch.equal(
                            physical_to_logical[packed[valid].long()], source[valid]
                        )
                    )
                    expected_histogram = torch.zeros(
                        20, device="cuda", dtype=torch.int64
                    )
                    present = router.cand_phys >= 0
                    expected_histogram[router.cand_phys[present].long()] = (
                        expected_quota[present].long()
                    )
                    torch.testing.assert_close(
                        torch.bincount(packed[valid].long(), minlength=20),
                        expected_histogram,
                        rtol=0,
                        atol=0,
                    )
                    self.assertTrue(torch.all(storage[::2] == -777))
                    self.assertTrue(torch.all(storage[:, :2] == -777))
                    self.assertTrue(torch.all(storage[:, topk + 2 :] == -777))

    def test_split_only_counter_repeated_calls_and_graph_replay(self):
        candidates = torch.tensor(
            [[0, -1], [1, 4], [2, 5], [3, 6], [7, -1]],
            device="cuda",
            dtype=torch.int32,
        )
        replicas = torch.tensor([1, 2, 2, 2, 1], device="cuda", dtype=torch.int32)
        physical_to_logical = torch.tensor([0, 1, 2, 3, 1, 2, 3, 4], device="cuda")
        patterns = (
            (
                "mixed_second",
                [0] * 800 + [1] * 600 + [2] * 400,
                [[800, 0], [0, 600], [100, 300], [0, 0], [0, 0]],
            ),
            (
                "single_replica",
                [4] * 1800,
                [[0, 0], [0, 0], [0, 0], [0, 0], [1800, 0]],
            ),
            (
                "rounded_first_only",
                [1],
                [[0, 0], [1, 0], [0, 0], [0, 0], [0, 0]],
            ),
            (
                "mixed_first",
                [4] * 800 + [1] * 600 + [2] * 400,
                [[0, 0], [600, 0], [300, 100], [0, 0], [800, 0]],
            ),
            (
                "split_replicas",
                [3] * 1800,
                [[0, 0], [0, 0], [0, 0], [900, 900], [0, 0]],
            ),
            ("invalid", [], [[0, 0]] * 5),
        )
        for passes in (1, 2, 8):
            arguments = dict(num_logical=5, num_physical=8, ep_size=2, wf_passes=passes)
            for dtype in (torch.int32, torch.int64):
                for capture in (False, True):
                    router = MaxPanderRouter(candidates, replicas, **arguments)
                    reference = LegacyFullWF(candidates, replicas, **arguments)
                    source = torch.zeros((259, 7), device="cuda", dtype=dtype)
                    storage = torch.full((519, 13), -777, device="cuda", dtype=dtype)
                    reference_storage = torch.full_like(storage, -777)
                    packed = storage[1::2, 2:9]
                    expected = reference_storage[1::2, 2:9]
                    if capture:
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
                    for name, values, expected_quota in (
                        *patterns,
                        patterns[0],
                        patterns[0],
                    ):
                        with self.subTest(
                            passes=passes, dtype=dtype, capture=capture, case=name
                        ):
                            invalid = [-1, -2, 5, 9] * (source.numel() // 4 + 1)
                            logical = values + invalid[: source.numel() - len(values)]
                            source.copy_(
                                torch.tensor(
                                    logical, device="cuda", dtype=dtype
                                ).view_as(source)
                            )
                            router.rank_counter.fill_(-777)
                            if capture:
                                graph.replay()
                                torch.cuda.synchronize()
                            else:
                                packed.copy_(source)
                                router.remap_inplace(packed)
                            expected.copy_(source)
                            reference.remap_inplace(expected)
                            for workspace in ("counts", "hot_order", "quota", "alloc"):
                                self.assertTrue(
                                    torch.equal(
                                        getattr(router, workspace),
                                        getattr(reference, workspace),
                                    ),
                                    workspace,
                                )
                            torch.testing.assert_close(
                                router.quota,
                                torch.tensor(
                                    expected_quota, device="cuda", dtype=torch.int32
                                ),
                                rtol=0,
                                atol=0,
                            )
                            self.check_split_only_counter(
                                source, router.quota, router.rank_counter
                            )
                            valid = (source >= 0) & (source < 5)
                            self.assertTrue(torch.all(packed[~valid] == -1))
                            self.assertTrue(
                                torch.equal(
                                    physical_to_logical[packed[valid].long()],
                                    source[valid],
                                )
                            )
                            torch.testing.assert_close(
                                torch.bincount(packed[valid].long(), minlength=8),
                                torch.bincount(expected[valid].long(), minlength=8),
                                rtol=0,
                                atol=0,
                            )
                            self.assertTrue(torch.all(storage[::2] == -777))
                            self.assertTrue(torch.all(storage[:, :2] == -777))
                            self.assertTrue(torch.all(storage[:, 9:] == -777))

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

    def test_default_passes_is_two(self):
        router, _, _ = self.make_router()
        self.assertEqual(router.passes, 2)

    def test_cached_launchers_reload_inputs_on_current_stream(self):
        from rtp_llm.models_py.modules import maxpander_kernels

        router, physical_to_logical, _ = self.make_router()
        packed = torch.zeros((257, 4), device="cuda", dtype=torch.int64)
        router.remap_inplace(packed)
        self.assertEqual(len(router._launch_cache), 1)
        with ExitStack() as stack:
            for name in (
                "_wf_partial_count_kernel",
                "_wf_reduce_order_kernel",
                "_wf_count_order_kernel",
                "_wf_quota_kernel",
                "_wf_scatter_kernel",
            ):
                stack.enter_context(
                    mock.patch.object(
                        getattr(maxpander_kernels, name),
                        "run",
                        side_effect=AssertionError("unexpected JIT dispatch"),
                    )
                )
            for value in (7, -1, 15, 0):
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    packed.fill_(value)
                    router.remap_inplace(packed)
                    completed = torch.cuda.Event()
                    completed.record()
                completed.synchronize()
                logical = torch.full_like(packed, value)
                self.check_mapping(logical, packed, physical_to_logical)
                valid = logical[(logical >= 0) & (logical < 16)]
                expected = torch.bincount(valid, minlength=16)
                torch.testing.assert_close(
                    router.quota.sum(1), expected, rtol=0, atol=0
                )

    def test_launcher_specializations_are_isolated(self):
        router, physical_to_logical, _ = self.make_router()
        for dtype in (torch.int32, torch.int64):
            storage = torch.empty((263, 13), device="cuda", dtype=dtype)
            for packed in (storage[:, :4], storage[:, 1:5], storage[:17, :7]):
                for passes in (2, 8, 1, 2):
                    router.passes = passes
                    logical = torch.randint_like(packed, -1, 18)
                    packed.copy_(logical)
                    router.remap_inplace(packed)
                    self.check_mapping(logical, packed, physical_to_logical)
                    valid = logical[(logical >= 0) & (logical < 16)].long()
                    torch.testing.assert_close(
                        router.quota.sum(1),
                        torch.bincount(valid, minlength=16),
                        rtol=0,
                        atol=0,
                    )

    def test_launcher_cache_is_bounded_and_does_not_retain_inputs(self):
        router, physical_to_logical, _ = self.make_router()
        storage = torch.zeros((32, 36), device="cuda", dtype=torch.int32)
        for offset in range(32):
            packed = storage[:, offset : offset + 4]
            logical = torch.full_like(packed, offset % 16)
            packed.copy_(logical)
            router.remap_inplace(packed)
            self.check_mapping(logical, packed, physical_to_logical)
            self.assertLessEqual(len(router._launch_cache), 16)
        self.assertEqual(len(router._launch_cache), 16)
        reference = weakref.ref(packed)
        del packed
        self.assertIsNone(reference())
        router.to(device="cuda")
        self.assertEqual(len(router._launch_cache), 0)

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
            "partial_counts",
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
                "_wf_partial_count_kernel",
                "_wf_reduce_order_kernel",
                "_wf_count_order_kernel",
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

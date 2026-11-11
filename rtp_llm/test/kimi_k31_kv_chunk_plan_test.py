"""KV budget/coverage contracts, independent of checkpoint precision."""

import os
import unittest
from unittest import mock
from unittest.mock import Mock, patch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_chunk_plan import (
    DsaKVCapacityError,
    plan_dsa_kv_chunks,
    prefill_indexer_score_bytes,
)
from rtp_llm.models_py.modules.factory.attention.mla_prefill_plan import (
    Bf16MlaWorkspaceBudget,
    plan_prefix_chunks,
    prefill_kv_workspace_budget_bytes,
)


class DsaKVChunkPlanTest(unittest.TestCase):
    def test_fp8_prefill_score_budget_includes_storage_padding(self):
        # The allocator snapshot has 8 MiB of row padding beyond the logical score.
        self.assertEqual(prefill_indexer_score_bytes(8192, 154624), 4840 * 2**20)

    def test_full_history_that_fits_is_not_split(self):
        plan = plan_dsa_kv_chunks(
            [17, 2048, 4097],
            page_tokens=1024,
            budget_bytes=100000,
            fixed_bytes=1000,
            temporary_bytes=lambda b: b.tokens * 4,
        )
        self.assertFalse(plan.chunked)
        self.assertEqual(plan.blocks[0].lengths, (17, 2048, 4097))

    def test_ragged_pages_are_consumed_exactly_once(self):
        lengths = [17, 2300, 0, 4097]
        plan = plan_dsa_kv_chunks(
            lengths,
            page_tokens=1024,
            budget_bytes=9500,
            fixed_bytes=1000,
            temporary_bytes=lambda b: b.tokens * 4,
        )
        self.assertTrue(plan.chunked)
        seen = [set() for _ in lengths]
        for block in plan.blocks:
            self.assertLessEqual(1000 + block.tokens * 4, 9500)
            for request, start, length in zip(
                block.requests, block.starts, block.lengths
            ):
                self.assertEqual(start % 1024, 0)
                positions = set(range(start, start + length))
                self.assertFalse(seen[request] & positions)
                seen[request].update(positions)
        self.assertEqual(seen, [set(range(n)) for n in lengths])
        # Increasing the chosen page capacity by one must violate this budget.
        bigger = plan_dsa_kv_chunks(
            lengths,
            page_tokens=1024,
            budget_bytes=1000000,
            fixed_bytes=1000,
            temporary_bytes=lambda b: b.tokens * 4,
        )
        self.assertGreater(bigger.page_capacity, plan.page_capacity)

    def test_independent_phase_budgets_and_fixed_live_data(self):
        def plan(budget):
            return plan_dsa_kv_chunks(
                [65536],
                page_tokens=1024,
                budget_bytes=budget,
                fixed_bytes=1000,
                temporary_bytes=lambda b: b.tokens * 4,
            )

        self.assertNotEqual(len(plan(34000).blocks), len(plan(66000).blocks))
        with self.assertRaisesRegex(ValueError, "fixed live"):
            plan(999)
        with self.assertRaisesRegex(ValueError, "one physical KV page"):
            plan(1001)

    def test_fixed_overflow_skips_callbacks_and_block_construction(self):
        for lengths in ((), (0, 0), (65536,)):
            for whole_plan in (False, True):
                with self.subTest(lengths=lengths, whole_plan=whole_plan):
                    temporary = Mock(
                        side_effect=AssertionError("must not price a block")
                    )
                    plan_temporary = Mock(
                        side_effect=AssertionError("must not price a plan")
                    )
                    with patch(
                        f"{plan_dsa_kv_chunks.__module__}._blocks",
                        side_effect=AssertionError("must not construct blocks"),
                    ) as blocks:
                        for _ in range(2):
                            with self.assertRaises(DsaKVCapacityError) as caught:
                                plan_dsa_kv_chunks(
                                    lengths,
                                    page_tokens=64,
                                    budget_bytes=999,
                                    fixed_bytes=1000,
                                    temporary_bytes=temporary,
                                    plan_temporary_bytes=(
                                        plan_temporary if whole_plan else None
                                    ),
                                )
                            error = caught.exception
                            self.assertEqual(error.lower_bound_bytes, 1000)
                            self.assertIsNone(error.min_required_bytes)
                            self.assertIsNone(error.min_page_capacity)
                            self.assertIsNone(error.one_page_required_bytes)
                            for field in (
                                "diagnostic_reason=fixed_exceeds_budget,",
                                "lower_bound_bytes=1000,",
                                "shortfall_at_least_bytes=1,",
                                "min_required_bytes=unknown,",
                                "additional_workspace_bytes=unknown,",
                                "kv_costs=not_computed,",
                                "one_page_required_bytes=unknown",
                            ):
                                self.assertIn(field, str(error))
                        blocks.assert_not_called()
                    temporary.assert_not_called()
                    plan_temporary.assert_not_called()

    def test_exact_one_page_budget_and_fixed_only_is_not_an_allowance(self):
        for budget in (1000, 1255, 1256):
            with self.subTest(budget=budget):
                kwargs = dict(
                    page_tokens=64,
                    budget_bytes=budget,
                    fixed_bytes=1000,
                    temporary_bytes=lambda b: b.tokens * 4,
                )
                if budget == 1256:
                    self.assertEqual(
                        plan_dsa_kv_chunks([128], **kwargs).required_bytes, 1256
                    )
                    continue
                with self.assertRaises(DsaKVCapacityError) as caught:
                    plan_dsa_kv_chunks([128], **kwargs)
                error = caught.exception
                self.assertEqual(error.min_required_bytes, 1256)
                self.assertEqual(error.one_page_required_bytes, 1256)
                self.assertEqual(error.min_page_capacity, 1)
                self.assertIn(
                    f"remaining_after_fixed_bytes={budget - 1000}", str(error)
                )
                self.assertIn(f"shortfall_bytes={1256 - budget}", str(error))

    def test_whole_plan_union_and_metadata_define_exact_capacity(self):
        # Per-block maxima alone miss the persistent union/metadata. The
        # additional charge is deliberately non-monotonic across capacities.
        def whole_plan(blocks):
            return {1: 500, 2: 200, 3: 300}[len(blocks)]

        kwargs = dict(
            page_tokens=64,
            fixed_bytes=1000,
            temporary_bytes=lambda b: b.tokens,
            plan_temporary_bytes=whole_plan,
        )
        plan = plan_dsa_kv_chunks([192], budget_bytes=1200, **kwargs)
        self.assertEqual(plan.page_capacity, 2)
        self.assertEqual(plan.required_bytes, 1200)
        for budget in (1000, 1199):
            with self.subTest(budget=budget), self.assertRaises(
                DsaKVCapacityError
            ) as caught:
                plan_dsa_kv_chunks([192], budget_bytes=budget, **kwargs)
            self.assertEqual(caught.exception.min_required_bytes, 1200)
            self.assertEqual(caught.exception.min_page_capacity, 2)
            self.assertEqual(caught.exception.one_page_required_bytes, 1300)
        with self.assertRaises(DsaKVCapacityError) as caught:
            plan_dsa_kv_chunks([192], budget_bytes=1200, max_page_capacity=1, **kwargs)
        self.assertEqual(caught.exception.min_required_bytes, 1300)

    def test_empty_history_failure_includes_resident_phase_bytes(self):
        with self.assertRaises(DsaKVCapacityError) as caught:
            plan_dsa_kv_chunks(
                [0, 0],
                page_tokens=64,
                budget_bytes=1100,
                fixed_bytes=1000,
                temporary_bytes=lambda b: self.fail("no KV block"),
                plan_temporary_bytes=lambda blocks: 101,
            )
        self.assertEqual(caught.exception.min_required_bytes, 1101)
        self.assertEqual(caught.exception.min_page_capacity, 0)
        self.assertIsNone(caught.exception.one_page_required_bytes)

    def test_full_phase_and_kv_caps_must_fit_the_same_nonmonotonic_plan(self):
        # Full-phase-only chooses 4, KV-only chooses 3, but only capacity 1
        # satisfies both. A binary search or independent minimum is unsound.
        total = {4: 90, 3: 120, 2: 80, 1: 100}
        kv = {4: 60, 3: 40, 2: 55, 1: 50}
        capacity = lambda blocks: max(b.tokens for b in blocks)
        plan = plan_dsa_kv_chunks(
            [4], page_tokens=1, budget_bytes=100, fixed_bytes=10,
            temporary_bytes=lambda b: 0,
            plan_temporary_bytes=lambda blocks: total[capacity(blocks)] - 10,
            kv_budget_bytes=50,
            kv_temporary_bytes=lambda blocks: kv[capacity(blocks)],
        )
        self.assertEqual(plan.page_capacity, 1)
        self.assertEqual(plan.required_bytes, 100)
        self.assertEqual(sum(b.tokens for b in plan.blocks), 4)

    def test_dual_budget_failure_reports_the_constraining_scope(self):
        for total_budget, kv_budget, expected_scope, minimum, capacity in (
            (1000, 9, "prefill_kv", 10, 1),
            (99, 10, "phase", 100, 1),
        ):
            with self.subTest(scope=expected_scope), self.assertRaises(DsaKVCapacityError) as caught:
                plan_dsa_kv_chunks(
                    [2], page_tokens=1, budget_bytes=total_budget, fixed_bytes=10,
                    temporary_bytes=lambda b: 0,
                    plan_temporary_bytes=lambda blocks: 90 if len(blocks) == 2 else 20,
                    kv_budget_bytes=kv_budget,
                    kv_temporary_bytes=lambda blocks: 10 if len(blocks) == 2 else 20,
                )
            self.assertEqual(caught.exception.budget_scope, expected_scope)
            self.assertEqual(caught.exception.min_required_bytes, minimum)
            self.assertEqual(caught.exception.min_page_capacity, capacity)

    def test_fixed_overflow_does_not_price_either_budget(self):
        callback = Mock(side_effect=AssertionError("fixed overflow must short-circuit"))
        with self.assertRaises(DsaKVCapacityError) as caught:
            plan_dsa_kv_chunks(
                [1024], page_tokens=64, budget_bytes=9, fixed_bytes=10,
                temporary_bytes=callback, plan_temporary_bytes=callback,
                kv_budget_bytes=1, kv_temporary_bytes=callback,
            )
        self.assertEqual(caught.exception.budget_scope, "phase")
        callback.assert_not_called()

    def test_dual_budget_requires_a_positive_cap_and_matching_callback(self):
        for options in (
            {"kv_budget_bytes": 1},
            {"kv_temporary_bytes": lambda blocks: 0},
            {"kv_budget_bytes": 0, "kv_temporary_bytes": lambda blocks: 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                plan_dsa_kv_chunks(
                    [], page_tokens=64, budget_bytes=1, fixed_bytes=0,
                    temporary_bytes=lambda block: 0, **options,
                )

    def test_empty_history_preserves_fixed_budget(self):
        plan = plan_dsa_kv_chunks(
            [0, 0],
            page_tokens=1024,
            budget_bytes=2048,
            fixed_bytes=1024,
            temporary_bytes=lambda b: b.tokens,
        )
        self.assertEqual(plan.blocks, ())
        self.assertEqual(plan.required_bytes, 1024)


class SharedPrefillKVBudgetTest(unittest.TestCase):
    def test_budget_defaults_to_eight_gib_and_accepts_override(self):
        key = "RTP_PREFILL_KV_WORKSPACE_BUDGET_GIB"
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(prefill_kv_workspace_budget_bytes(), 8 * 2**30)
        with mock.patch.dict(os.environ, {key: "4.5"}):
            self.assertEqual(prefill_kv_workspace_budget_bytes(), 4608 * 2**20)
        for value in ("0", "-1", "nan", "inf", "invalid"):
            with self.subTest(value=value), mock.patch.dict(os.environ, {key: value}):
                with self.assertRaisesRegex(ValueError, key):
                    prefill_kv_workspace_budget_bytes()

    def test_mtp_budget_charges_the_overlapping_reader_and_kv_up_buffers(self):
        plan = plan_prefix_chunks(
            (8192,), (1040384,), page_tokens=1024, bytes_per_token=7680,
            workspace_budget=Bf16MlaWorkspaceBudget(6 * 2**30, 8, 512, 64),
        )
        self.assertTrue(plan.chunked)
        self.assertEqual(plan.capacity_tokens, 576512)
        self.assertEqual(
            [(s.start, s.length) for s in plan.slices],
            [(0, 576512), (576512, 463872)],
        )

    def test_mtp_full_path_checks_padding_per_request(self):
        # Eight tiny prefixes still need eight independent owner pages if read
        # together. Reading each through the existing slice loop fits 20 MiB.
        workspace = Bf16MlaWorkspaceBudget(20 * 2**20, 8, 512, 64)
        small = plan_prefix_chunks(
            (8,), (1,), page_tokens=1024, bytes_per_token=7680,
            workspace_budget=workspace,
        )
        self.assertFalse(small.chunked)
        ragged = plan_prefix_chunks(
            (1,) * 8, (1,) * 8, page_tokens=1024, bytes_per_token=7680,
            pack_requests=True, workspace_budget=workspace,
        )
        self.assertTrue(ragged.chunked)
        self.assertEqual([s.owner for s in ragged.slices], list(range(8)))

    def test_one_million_history_fits_one_gib_without_omitting_tokens(self):
        query, prefix = 8192, 1040384
        workspace = Bf16MlaWorkspaceBudget(2**30, 8, 512, 64, query)
        plan = plan_prefix_chunks(
            (query,), (prefix,), page_tokens=1024, bytes_per_token=7680,
            workspace_budget=workspace,
        )
        self.assertTrue(plan.chunked)
        self.assertTrue(plan.requires_fp32_accumulator)
        cursor = 0
        for part in plan.slices:
            self.assertEqual((part.owner, part.start), (0, cursor))
            self.assertLessEqual(part.length, plan.capacity_tokens)
            cursor += part.length
        self.assertEqual(cursor + query, 1048576)
        rows = plan.capacity_tokens
        owner_pages = (rows + 8191) // 8192
        live_bytes = (
            max(query, rows) * 7680 + rows * 512 * 2
            + (rows + 9 * owner_pages * 1024) * 1152
        )
        self.assertLessEqual(live_bytes, workspace.budget_bytes)

    def test_mtp_cold_query_must_fit_packed_kv_without_history(self):
        with self.assertRaisesRegex(ValueError, "current-query BF16 MLA KV"):
            plan_prefix_chunks(
                (8192,), (0,), page_tokens=1024, bytes_per_token=7680,
                workspace_budget=Bf16MlaWorkspaceBudget(60 * 2**20, 8, 512, 64),
            )


if __name__ == "__main__":
    unittest.main()

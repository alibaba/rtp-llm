"""KV budget/coverage contracts, independent of checkpoint precision."""

import os
import unittest
from unittest import mock

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_chunk_plan import (
    fp8_sfa_prefill_query_shape,
    plan_dsa_kv_chunks,
    prefill_indexer_score_bytes,
)
from rtp_llm.models_py.modules.factory.attention.mla_prefill_plan import (
    Bf16MlaWorkspaceBudget,
    plan_prefix_chunks,
    prefill_kv_workspace_budget_bytes,
)


class DsaKVChunkPlanTest(unittest.TestCase):
    def test_fp8_sfa_prefill_packs_full_query_shard_into_one_launch(self):
        shared_memory = 227 * 1024
        for rows, expected in (
            (0, (0, 1)),
            (8192, (8192, 1)),
            (16384, (8192, 2)),
            (16385, (8193, 2)),
            (65536, (10923, 6)),
        ):
            with self.subTest(rows=rows):
                batch, width = fp8_sfa_prefill_query_shape(rows, shared_memory)
                self.assertEqual((batch, width), expected)
                self.assertLessEqual(20 * batch + 4, shared_memory)
                self.assertIn(batch * width - rows, range(width))

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

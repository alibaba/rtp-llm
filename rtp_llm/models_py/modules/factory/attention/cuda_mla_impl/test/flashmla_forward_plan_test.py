import random
import unittest

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_forward_plan import (
    FlashMLAForwardRoute,
    plan_flashmla_forward,
)

PAGE_SIZE = 4
BYTES_PER_TOKEN = 16


def make_plan(q_lens, prefix_lens, *, capacity_tokens, page_size=PAGE_SIZE):
    return plan_flashmla_forward(
        q_lens,
        prefix_lens,
        page_size=page_size,
        expanded_kv_budget_bytes=capacity_tokens * BYTES_PER_TOKEN,
        expanded_kv_bytes_per_token=BYTES_PER_TOKEN,
    )


def launch_slices(plan):
    return [
        tuple(
            (item.request_idx, item.prefix_start, item.prefix_len)
            for item in launch.slices
        )
        for launch in plan.prefix_launches
    ]


def assert_prefix_invariants(test_case, plan, q_lens, prefix_lens, *, page_size):
    if plan.route is FlashMLAForwardRoute.FULL:
        test_case.assertEqual(plan.prefix_launches, ())
        test_case.assertEqual(plan.max_expanded_kv_tokens, 0)
        test_case.assertEqual(plan.max_packed_q_tokens, 0)
        test_case.assertEqual(plan.max_partial_state_tokens, 0)
        test_case.assertFalse(plan.requires_fp32_accumulator)
        return

    cursors = [0] * len(prefix_lens)
    owner_launch_counts = [0] * len(prefix_lens)
    active = {index for index, length in enumerate(prefix_lens) if length}

    for launch in plan.prefix_launches:
        owners = [item.request_idx for item in launch.slices]
        test_case.assertEqual(len(owners), len(set(owners)))
        test_case.assertEqual(owners, sorted(active)[: len(owners)])
        test_case.assertLessEqual(launch.expanded_kv_tokens, plan.capacity_tokens)
        test_case.assertEqual(
            launch.expanded_kv_tokens,
            sum(item.prefix_len for item in launch.slices),
        )
        test_case.assertEqual(
            launch.packed_q_tokens,
            sum(q_lens[item.request_idx] for item in launch.slices),
        )

        for slice_index, item in enumerate(launch.slices):
            test_case.assertEqual(item.prefix_start, cursors[item.request_idx])
            test_case.assertEqual(item.prefix_start % page_size, 0)
            test_case.assertGreater(item.prefix_len, 0)
            cursor_after = item.prefix_start + item.prefix_len
            test_case.assertLessEqual(cursor_after, prefix_lens[item.request_idx])
            if cursor_after < prefix_lens[item.request_idx]:
                test_case.assertEqual(slice_index, len(launch.slices) - 1)
                test_case.assertEqual(item.prefix_len % page_size, 0)
            cursors[item.request_idx] = cursor_after
            owner_launch_counts[item.request_idx] += 1
            if cursor_after == prefix_lens[item.request_idx]:
                active.remove(item.request_idx)

        if active:
            test_case.assertLess(
                plan.capacity_tokens - launch.expanded_kv_tokens,
                page_size,
            )

    test_case.assertEqual(cursors, list(prefix_lens))
    test_case.assertEqual(
        plan.max_expanded_kv_tokens,
        max(launch.expanded_kv_tokens for launch in plan.prefix_launches),
    )
    test_case.assertEqual(
        plan.max_packed_q_tokens,
        max(launch.packed_q_tokens for launch in plan.prefix_launches),
    )
    test_case.assertEqual(
        plan.max_partial_state_tokens,
        plan.max_packed_q_tokens,
    )
    test_case.assertEqual(
        plan.requires_fp32_accumulator,
        any(count > 1 for count in owner_launch_counts),
    )


class FlashMLAForwardPlanTest(unittest.TestCase):
    def test_prefix_budget_must_fit_one_page(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one prefix page"):
            make_plan((1,), (8,), capacity_tokens=PAGE_SIZE - 1)

    def test_route_selection_and_budget_boundaries(self) -> None:
        cases = (
            ("zero_budget", (4096,), (1 << 20,), 0, FlashMLAForwardRoute.FULL, 0),
            ("exact_budget", (4,), (4,), 128, FlashMLAForwardRoute.FULL, 8),
            (
                "one_byte_over",
                (4,),
                (4,),
                127,
                FlashMLAForwardRoute.HYBRID,
                4,
            ),
            ("q_over_budget", (9,), (1,), 128, FlashMLAForwardRoute.HYBRID, 8),
            ("no_prefix", (7, 1), (0, 0), 128, FlashMLAForwardRoute.FULL, 8),
        )
        for name, q_lens, prefix_lens, budget, route, capacity in cases:
            with self.subTest(name=name):
                plan = plan_flashmla_forward(
                    q_lens,
                    prefix_lens,
                    page_size=PAGE_SIZE,
                    expanded_kv_budget_bytes=budget,
                    expanded_kv_bytes_per_token=BYTES_PER_TOKEN,
                )
                self.assertIs(plan.route, route)
                self.assertEqual(plan.capacity_tokens, capacity)
                assert_prefix_invariants(
                    self,
                    plan,
                    q_lens,
                    prefix_lens,
                    page_size=PAGE_SIZE,
                )

    def test_request_major_packing_splits_only_the_boundary_request(self) -> None:
        cases = (
            (
                (1, 1),
                (1024, 128),
                1024,
                [((0, 0, 1024),), ((1, 0, 128),)],
            ),
            (
                (1, 1),
                (1, 385),
                256,
                [
                    ((0, 0, 1), (1, 0, 128)),
                    ((1, 128, 256),),
                    ((1, 384, 1),),
                ],
            ),
            (
                (1, 1, 1),
                (129, 257, 256),
                256,
                [
                    ((0, 0, 129),),
                    ((1, 0, 256),),
                    ((1, 256, 1), (2, 0, 128)),
                    ((2, 128, 128),),
                ],
            ),
            (
                (1, 1, 1),
                (256, 127, 1),
                256,
                [((0, 0, 256),), ((1, 0, 127), (2, 0, 1))],
            ),
        )
        for q_lens, prefix_lens, capacity, expected in cases:
            with self.subTest(prefix_lens=prefix_lens):
                plan = make_plan(
                    q_lens,
                    prefix_lens,
                    capacity_tokens=capacity,
                    page_size=128,
                )
                self.assertEqual(launch_slices(plan), expected)
                assert_prefix_invariants(
                    self,
                    plan,
                    q_lens,
                    prefix_lens,
                    page_size=128,
                )

    def test_large_batches_preserve_coverage_without_special_cases(self) -> None:
        for batch_size in (8, 63, 96):
            with self.subTest(batch_size=batch_size):
                q_lens = (1,) * batch_size
                prefix_lens = tuple(
                    PAGE_SIZE * (1 + index % 5) + index % PAGE_SIZE
                    for index in range(batch_size)
                )
                plan = make_plan(
                    q_lens,
                    prefix_lens,
                    capacity_tokens=PAGE_SIZE * batch_size,
                )
                self.assertIs(plan.route, FlashMLAForwardRoute.HYBRID)
                self.assertLessEqual(
                    sum(len(launch.slices) for launch in plan.prefix_launches),
                    4 * batch_size,
                )
                assert_prefix_invariants(
                    self,
                    plan,
                    q_lens,
                    prefix_lens,
                    page_size=PAGE_SIZE,
                )

    def test_equivalent_shapes_reuse_cached_plan(self) -> None:
        kwargs = {
            "page_size": 128,
            "expanded_kv_budget_bytes": 5 * 1024**3,
            "expanded_kv_bytes_per_token": 7680,
        }
        first = plan_flashmla_forward([1, 64], [1 << 20, 128], **kwargs)
        second = plan_flashmla_forward((1, 64), (1 << 20, 128), **kwargs)

        self.assertIs(first, second)

    def test_seeded_general_batches_are_deterministic_and_legal(self) -> None:
        rng = random.Random(20260903)
        checked = 0
        for _ in range(120):
            batch_size = rng.randint(1, 96)
            page_size = rng.choice((4, 8, 16))
            q_lens = tuple(rng.randint(1, 3) for _ in range(batch_size))
            prefix_lens = tuple(
                rng.randint(0, page_size * 6 + page_size - 1) for _ in range(batch_size)
            )
            capacity_tokens = page_size * rng.randint(2, 12)
            if (
                sum(q_lens) > capacity_tokens
                or sum(q_lens) + sum(prefix_lens) <= capacity_tokens
            ):
                continue
            first = make_plan(
                q_lens,
                prefix_lens,
                capacity_tokens=capacity_tokens,
                page_size=page_size,
            )
            second = make_plan(
                q_lens,
                prefix_lens,
                capacity_tokens=capacity_tokens,
                page_size=page_size,
            )
            self.assertEqual(first, second)
            assert_prefix_invariants(
                self,
                first,
                q_lens,
                prefix_lens,
                page_size=page_size,
            )
            checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()

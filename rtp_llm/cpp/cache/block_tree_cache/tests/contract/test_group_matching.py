"""Check sparse, mixed-stride Device plans against an independent page enumeration."""

import random
import unittest

from block_tree_cache import Binding, Cache, Group, Limits, Page, PageState, Position, Receipt, Result, Rule, Status


GROUPS = (Group(7, Rule.FULL, 2), Group(11, Rule.LINEAR, 4), Group(19, Rule.SWA, 2, window_tokens=5))


def required_positions(group, end):
    positions = [Position(group.id, token) for token in range(group.page_tokens, end + 1, group.page_tokens)]
    if group.rule == Rule.LINEAR:
        return positions[-1:]
    if group.rule == Rule.SWA:
        return [p for p in positions if p.end_token > end - group.window_tokens]
    return positions


def expected_plan(pages, selected, limit, boundaries):
    for end in range(limit, 0, -1):
        if boundaries is not None and end not in boundaries:
            continue
        if any(end % group.page_tokens for group in selected):
            continue
        positions = [p for group in selected for p in required_positions(group, end)]
        if all(p in pages for p in positions):
            return end, {p: pages[p] for p in positions}
    return 0, {}


class MixedGroupMatchTest(unittest.TestCase):
    def test_sparse_prefixes(self):
        keys = [b"same-prefix-64bits" + i.to_bytes(3, "big") for i in range(12)]
        selections = ((7,), (11,), (19,), (7, 19), (11, 19), (7, 11, 19))
        for seed in range(48):
            rng = random.Random(643192 + seed)
            pages = {
                Position(group.id, end): Page(group.id, end, 3, 100 * group.id + end)
                for group in GROUPS for end in range(group.page_tokens, 25, group.page_tokens)
                if rng.random() < 0.79
            }
            cache = Cache(b"coverage", 2, GROUPS)
            cache.publish(keys, 24, [Binding(p, (page,)) for p, page in pages.items()])
            for ids in selections:
                selected = [group for group in GROUPS if group.id in ids]
                for boundaries in (None, (4, 12, 20, 24)):
                    for limit in (4, 9, 16, 23, 24):
                        with self.subTest(seed=seed, groups=ids, boundaries=boundaries, limit=limit):
                            end, expected = expected_plan(pages, selected, limit, boundaries)
                            plan = cache.match(1, keys, Limits(limit, ids, boundaries, 0, False))
                            self.assertIsNotNone(plan)
                            try:
                                self.assertEqual((plan.ready_tokens, plan.recoverable_tokens), (end, end))
                                for rows in (plan.pages, plan.ready_pages):
                                    self.assertEqual(len(rows), len(expected))
                                    self.assertEqual({p.position: p.pages for p in rows},
                                                     {p: (page,) for p, page in expected.items()})
                                    self.assertTrue(all(p.state == PageState.READY for p in rows))
                            finally:
                                plan.close()
            cache.drain()
            for work in cache.poll().work:
                cache.complete([Receipt(work.id, actor, Result.SUCCESS, True) for actor in work.participants])
            self.assertEqual(cache.close(), Status.OK)


if __name__ == "__main__":
    unittest.main()

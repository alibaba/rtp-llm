"""Exercise the physical allocator/evictor boundary through the installed C ABI."""

import unittest

from block_tree_cache import Binding, Cache, Group, Limits, Page, Position, Receipt, Result, Rule, Status, WorkKind


def open_cache(groups=(Group(0, Rule.FULL, 1),), host_slots=8, *, remote=False):
    routes = {WorkKind.DEVICE_TO_HOST: (0, 1), WorkKind.HOST_TO_DEVICE: (0, 1)}
    if remote:
        routes.update({kind: (3,) for kind in (WorkKind.REMOTE_QUERY, WorkKind.REMOTE_TO_HOST,
                                              WorkKind.HOST_TO_REMOTE)})
    return Cache(b"tiered", 1, groups, host_slots=host_slots, routes=routes, tiered_eviction=True)


def settle(cache, work, *, committed=False):
    cache.complete([Receipt(work.id, actor, Result.SUCCESS, True, committed) for actor in work.participants])


class TieredEvictionTest(unittest.TestCase):
    def close_cache(self, cache):
        while cache.drain() == Status.PENDING:
            remaining = cache.poll().work
            self.assertTrue(remaining)
            for work in remaining:
                self.assertEqual(work.kind, WorkKind.RELEASE)
                settle(cache, work)
        self.assertEqual(cache.close(), Status.OK)

    def test_only_one_physical_release_is_needed(self):
        cache = open_cache()
        bindings = [Binding(Position(0, 1), (Page(17, i, 1, i + 1),)) for i in range(4)]
        keys = [bytes([65 + i]) for i in range(4)]
        for key, binding in zip(keys, bindings):
            cache.publish([key], 1, [binding])
        self.assertEqual(cache.poll().work, ())
        pages = [b.pages[0] for b in bindings]
        self.assertFalse(cache.device_balance_needed(17, 10, 2))
        self.assertTrue(cache.device_balance_needed(17, 10, 1))
        cache.balance_device(17, 10, 1, pages + pages)
        copy, = cache.poll().work
        self.assertEqual(copy.kind, WorkKind.DEVICE_TO_HOST)
        item, = copy.pages
        cache.complete([Receipt(copy.id, 0, Result.SUCCESS, True)])
        cache.balance_device(17, 10, 1, pages)
        self.assertEqual([w.id for w in cache.poll().work], [copy.id])
        i = keys.index(item.key)
        self.assertEqual(cache.installed_device_pages([keys[i]], [bindings[i]]), (True,))
        self.assertEqual(cache.host_capacity(0), (8, 7))
        cache.complete([Receipt(copy.id, 1, Result.SUCCESS, True)])
        self.assertEqual(cache.installed_device_pages([keys[i]], [bindings[i]]), (False,))
        cache.balance_device(17, 10, 1, pages)
        release, = cache.poll().work
        self.assertEqual(release.kind, WorkKind.RELEASE)
        self.assertEqual(release.pages[0].source, (pages[i],))
        settle(cache, release)
        cache.balance_device(17, 10, 2, [p for p in pages if p != pages[i]])
        self.assertFalse(cache.device_balance_needed(17, 10, 2))

        plan = cache.match(20, [keys[i]], Limits(1, allow_remote=False))
        self.assertEqual((plan.ready_tokens, plan.host_tokens), (0, 1))
        new = Binding(Position(0, 1), (Page(17, 7, 2, 70),))
        self.assertEqual(cache.bind(plan, [new]), Status.WAITING)
        plan.close()
        load, = cache.poll().work
        self.assertEqual(load.kind, WorkKind.HOST_TO_DEVICE)
        cache.complete([Receipt(load.id, 0, Result.SUCCESS, True)])
        self.assertEqual(cache.host_capacity(0), (8, 7))
        self.assertEqual(cache.installed_device_pages([keys[i]], [new]), (False,))
        cache.complete([Receipt(load.id, 1, Result.SUCCESS, True)])
        self.assertEqual(cache.installed_device_pages([keys[i]], [new, bindings[i]]), (True, False))
        self.assertEqual(cache.host_capacity(0), (8, 8))
        cache.cancel(20)
        self.close_cache(cache)

    def test_reused_source_survives_both_successful_dma_receipts(self):
        cache = open_cache()
        page = Page(0, 1, 9, 100)
        binding = Binding(Position(0, 1), (page,))
        cache.publish([b"A"], 1, [binding])
        cache.balance_device(0, 10, 0, [page])
        copy, = cache.poll().work
        cache.accept(copy.id, 0)
        cache.accept(copy.id, 1)
        cache.balance_device(0, 10, 0, [])
        settle(cache, copy)
        self.assertEqual(cache.installed_device_pages([b"A"], [binding]), (True,))
        self.assertEqual(cache.host_capacity(0), (8, 8))
        self.assertEqual(cache.poll().work, ())
        self.close_cache(cache)

    def test_two_distinct_holds_for_one_physical_slot_only_credit_one_release(self):
        cache = open_cache((Group(0, Rule.FULL, 1), Group(1, Rule.FULL, 1)))
        a, alias, b, c = (Page(0, 2, 1, 11), Page(0, 2, 1, 12), Page(0, 3, 1, 13), Page(0, 4, 1, 14))
        cache.publish([b"A"], 1, [Binding(Position(0, 1), (a,)), Binding(Position(1, 1), (alias,))])
        cache.publish([b"B"], 1, [Binding(Position(0, 1), (b,))])
        cache.publish([b"C"], 1, [Binding(Position(0, 1), (c,))])
        cache.balance_device(0, 10, 0, [a])
        copies = cache.poll().work
        self.assertEqual({p.source[0] for w in copies for p in w.pages}, {a, alias})
        self.assertTrue(all(w.kind == WorkKind.DEVICE_TO_HOST for w in copies))
        cache.balance_device(0, 10, 0, [a, alias, b, c])
        all_copies = cache.poll().work
        new = [w for w in all_copies if w.id not in {old.id for old in copies}]
        self.assertEqual(sum(len(w.pages) for w in new), 1)
        for work in all_copies:
            settle(cache, work)
        self.close_cache(cache)

    def test_group_moves_require_all_members_and_credit_physical_slots(self):
        cache = open_cache((Group(0, Rule.FULL, 1, members=2),))
        pairs = [(Page(4, i, 1, i + 1), Page(4, i + 1, 1, i + 2)) for i in (0, 2, 4)]
        for key, pair in zip((b"A", b"B", b"C"), pairs):
            cache.publish([key], 1, [Binding(Position(0, 1), pair)])
        cache.balance_device(4, 10, 0, [pair[0] for pair in pairs])
        self.assertEqual(cache.poll().work, ())
        pages = [page for pair in pairs for page in pair]
        cache.balance_device(4, 10, 0, pages)
        copy, = cache.poll().work
        row, = copy.pages
        self.assertIn(row.source, pairs)
        self.assertEqual(cache.host_capacity(0), (8, 7))
        cache.balance_device(4, 10, 0, pages)
        self.assertEqual([work.id for work in cache.poll().work], [copy.id])
        settle(cache, copy)
        self.close_cache(cache)

    def test_shared_host_pool_watermark_selects_one_victim_across_groups(self):
        cache = open_cache((Group(0, Rule.FULL, 1), Group(1, Rule.FULL, 1)), host_slots=6, remote=True)
        for i in range(6):
            page = Page(0, i, 1, i + 1)
            cache.publish([bytes([65 + i])], 1, [Binding(Position(i % 2, 1), (page,))])
            cache.demote(0, 1, [page])
            copy, = cache.poll().work
            settle(cache, copy)
            for work in cache.poll().work:
                if work.kind == WorkKind.RELEASE:
                    settle(cache, work)
        writes = cache.poll().work
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0].kind, WorkKind.HOST_TO_REMOTE)
        self.assertEqual(len(writes[0].pages), 1)
        self.assertEqual(cache.host_capacity(1), (6, 0))
        settle(cache, writes[0], committed=True)
        self.assertEqual(cache.host_capacity(0), (6, 1))
        self.assertEqual(cache.host_capacity(1), (6, 1))
        self.close_cache(cache)


if __name__ == "__main__":
    unittest.main()

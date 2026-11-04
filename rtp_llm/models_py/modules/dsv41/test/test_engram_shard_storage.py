# SPDX-License-Identifier: Apache-2.0
import unittest

from rtp_llm.models_py.modules.dsv41.engram_shard import plan_shards
from rtp_llm.models_py.modules.dsv41.engram_shard_storage import ShardedLayerMapping

DIM = 256
ROWS = 1_048_576  # 4 x 262144, no padding


class FakeOps:
    def __init__(self, fail_on: str = ""):
        self.calls = []
        self.fail_on = fail_on
        self._next = 0x1000_0000_0000
        self.views = []

    def reserve(self, size, alignment):
        self.calls.append(("reserve", size))
        pointer = self._next
        self._next += 1 << 34
        return pointer

    def import_handle(self, raw):
        self.calls.append(("import", raw))
        if self.fail_on == "import":
            raise RuntimeError("import failed")
        return int.from_bytes(raw[:8], "little")

    def map(self, va, size, offset, handle):
        self.calls.append(("map", va, size, offset, handle))
        if self.fail_on == "map":
            raise RuntimeError("map failed")

    def set_access(self, va, size):
        self.calls.append(("set_access", va, size))
        if getattr(self, "fail_on", "") == "set_access":
            raise RuntimeError("set_access failed")

    def release(self, handle):
        self.calls.append(("release", handle))

    def synchronize(self):
        self.calls.append(("synchronize",))

    def unmap(self, va, size):
        self.calls.append(("unmap", va, size))

    def free(self, va, size):
        self.calls.append(("free", va, size))

    def view(self, va, shape, owner):
        self.views.append((va, shape))
        return (va, shape)


class ShardedLayerMappingTest(unittest.TestCase):
    def setUp(self):
        self.plan = plan_shards(ROWS, DIM, 4)
        self.handles = [bytes([i + 1]) * 8 for i in range(4)]

    def test_attach_maps_every_shard(self):
        ops = FakeOps()
        mapping = ShardedLayerMapping(self.plan, self.handles, self.handles, 0, ops=ops).attach()
        self.assertEqual(mapping.state, "MAPPED")
        reserves = [c for c in ops.calls if c[0] == "reserve"]
        self.assertEqual([c[1] for c in reserves], [self.plan.weight_span, self.plan.scale_span])
        maps = [c for c in ops.calls if c[0] == "map"]
        self.assertEqual(len(maps), 8)
        for _, va, size, offset, _ in maps:
            self.assertEqual(va % self.plan.granularity, 0)
            self.assertEqual(size % self.plan.granularity, 0)
            self.assertEqual(offset, 0)  # fabric handles: offset must be zero

    def test_views_cover_padded_rows(self):
        ops = FakeOps()
        mapping = ShardedLayerMapping(self.plan, self.handles, self.handles, 0, ops=ops).attach()
        weight = mapping.weight_view()
        scale = mapping.scale_view()
        self.assertEqual(weight[1], (self.plan.padded_rows, DIM))
        self.assertEqual(scale[1], (self.plan.padded_rows, DIM // 32))

    def test_close_unmaps_and_frees(self):
        ops = FakeOps()
        mapping = ShardedLayerMapping(self.plan, self.handles, self.handles, 0, ops=ops).attach()
        mapping.close()
        self.assertEqual(mapping.state, "CLOSED")
        unmaps = [c for c in ops.calls if c[0] == "unmap"]
        frees = [c for c in ops.calls if c[0] == "free"]
        self.assertEqual(len(unmaps), 8)
        self.assertEqual(len(frees), 2)
        self.assertEqual(
            sorted(c[2] for c in frees),
            sorted([self.plan.weight_span, self.plan.scale_span]),
        )

    def test_failed_attach_rolls_back(self):
        ops = FakeOps(fail_on="map")
        mapping = ShardedLayerMapping(self.plan, self.handles, self.handles, 0, ops=ops)
        with self.assertRaises(RuntimeError):
            mapping.attach()
        self.assertEqual(mapping.state, "CLOSED")
        # Whatever was reserved must have been freed.
        self.assertEqual(len([c for c in ops.calls if c[0] == "free"]), 2)

    def test_rejects_bad_handles(self):
        with self.assertRaises(ValueError):
            ShardedLayerMapping(self.plan, self.handles[:3], self.handles, 0, ops=FakeOps())
        with self.assertRaises(ValueError):
            ShardedLayerMapping(self.plan, [b""] * 4, self.handles, 0, ops=FakeOps())

    def test_double_attach_rejected(self):
        mapping = ShardedLayerMapping(self.plan, self.handles, self.handles, 0, ops=FakeOps()).attach()
        with self.assertRaises(RuntimeError):
            mapping.attach()

    def test_view_before_attach_rejected(self):
        mapping = ShardedLayerMapping(self.plan, self.handles, self.handles, 0, ops=FakeOps())
        with self.assertRaises(RuntimeError):
            mapping.weight_view()

    def test_failure_at_each_partial_attach_stage_balances_resources(self):
        for operation, positions in (("reserve", (1, 2)), ("import_handle", (1, 4, 8)),
                                     ("map", (1, 4, 8)), ("set_access", (1, 4, 8))):
            for position in positions:
                with self.subTest(operation=operation, position=position):
                    ops = FakeOps()
                    original = getattr(ops, operation)
                    calls = 0
                    def fail_at(*args):
                        nonlocal calls
                        calls += 1
                        if calls == position:
                            raise RuntimeError("injected stage failure")
                        return original(*args)
                    setattr(ops, operation, fail_at)
                    mapping = ShardedLayerMapping(self.plan, self.handles, self.handles, 0, ops=ops)
                    with self.assertRaisesRegex(RuntimeError, "injected stage"):
                        mapping.attach()
                    names = [c[0] for c in ops.calls]
                    self.assertEqual(names.count("import"), names.count("release"))
                    self.assertEqual(names.count("map"), names.count("unmap"))
                    self.assertEqual(names.count("reserve"), names.count("free"))
                    self.assertEqual(mapping.info()["state"], "CLOSED")

    def test_authorization_failure_unmaps_and_releases_import(self):
        ops = FakeOps(fail_on="set_access")
        mapping = ShardedLayerMapping(self.plan, self.handles, self.handles, 0, ops=ops)
        with self.assertRaisesRegex(RuntimeError, "set_access failed"):
            mapping.attach()
        self.assertEqual(mapping.state, "CLOSED")
        names = [c[0] for c in ops.calls]
        self.assertEqual(names.count("unmap"), 1)
        self.assertEqual(names.count("release"), 1)
        self.assertLess(names.index("synchronize"), names.index("unmap"))
        self.assertLess(names.index("unmap"), names.index("release"))
        self.assertLess(names.index("release"), names.index("free"))

    def test_cleanup_retries_each_operation_without_double_free(self):
        for operation in ("synchronize", "unmap", "release", "free"):
            with self.subTest(operation=operation):
                ops = FakeOps()
                mapping = ShardedLayerMapping(self.plan, self.handles, self.handles, 0, ops=ops).attach()
                original = getattr(ops, operation)
                calls = 0
                def fail_once(*args):
                    nonlocal calls
                    calls += 1
                    if calls == 2 or (operation == "synchronize" and calls == 1):
                        raise RuntimeError("transient cleanup failure")
                    return original(*args)
                setattr(ops, operation, fail_once)
                with self.assertRaises(RuntimeError):
                    mapping.close()
                setattr(ops, operation, original)
                mapping.close()
                before = len(ops.calls)
                mapping.close()
                self.assertEqual(len(ops.calls), before)
                names = [c[0] for c in ops.calls]
                self.assertEqual(names.count("unmap"), 8)
                self.assertEqual(names.count("release"), 8)
                self.assertEqual(names.count("free"), 2)

    def test_failed_attach_retains_cleanup_resource_and_original_cause(self):
        from rtp_llm.models_py.modules.dsv41.engram_storage import CleanupPendingError
        ops = FakeOps(fail_on="set_access")
        original = ops.unmap
        def fail(*args):
            raise RuntimeError("unmap failed")
        ops.unmap = fail
        mapping = ShardedLayerMapping(self.plan, self.handles, self.handles, 0, ops=ops)
        with self.assertRaises(CleanupPendingError) as caught:
            mapping.attach()
        self.assertIs(caught.exception.resource, mapping)
        self.assertIn("set_access", str(caught.exception.__cause__))
        ops.unmap = original
        mapping.close()
        self.assertEqual(mapping.state, "CLOSED")

    def test_exporter_lease_blocks_close_until_last_alias_is_gone(self):
        from rtp_llm.models_py.modules.dsv41.engram_shard_storage import ArrayViewLease
        from rtp_llm.models_py.modules.dsv41.engram_storage import StorageBusyError
        ops = FakeOps()
        mapping = ShardedLayerMapping(self.plan, self.handles, self.handles, 0, ops=ops).attach()
        lease = ArrayViewLease(mapping._weight_ptr, (1, 256), mapping)
        alias = lease
        del lease
        with self.assertRaises(StorageBusyError):
            mapping.close()
        self.assertFalse(any(c[0] == "unmap" for c in ops.calls))
        del alias
        mapping.close()
        self.assertEqual(mapping.info()["active_views"], 0)


if __name__ == "__main__":
    unittest.main()

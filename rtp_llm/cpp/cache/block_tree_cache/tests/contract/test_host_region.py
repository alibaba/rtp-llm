import unittest

from block_tree_cache import (
    Binding, Cache, Error, Group, HostRegion, Page, Position, Receipt, Result, Rule, Status, WorkKind,
)


def open_region(storage):
    return Cache(
        b"host-mapping", 1, [Group(0, Rule.FULL, 1)], host_slots=2,
        routes={WorkKind.HOST_TO_DEVICE: [0], WorkKind.DEVICE_TO_HOST: [0]},
        host_region=HostRegion(storage, stride_bytes=8, payload_bytes=3),
    )


class HostRegionTest(unittest.TestCase):
    def test_native_rows_alias_the_exporter_across_a_pending_close(self):
        storage = bytearray(b"." * 16)
        cache = open_region(storage)
        cache.publish([b"key"], 1, [Binding(Position(0, 1), (Page(0, 1, 1, 1),))])
        cache.demote(0, 1, [Page(0, 1, 1, 1)], remove_device=False)
        work, = cache.poll().work
        cache.accept(work.id, 0)
        self.assertEqual(cache.close(), Status.PENDING)
        with self.assertRaises(BufferError):
            storage.extend(b"x")

        for slot, data in enumerate((b"one", b"two")):
            with cache.host_slot_buffer(0, slot) as row:
                row[:] = data
        self.assertEqual(storage, b"one.....two.....")
        with self.assertRaises(Error):
            cache.host_slot_buffer(0, 2)

        cache.complete([Receipt(work.id, 0, Result.SUCCESS, True)])
        cache.drain()
        for release in cache.poll().work:
            self.assertEqual(release.kind, WorkKind.RELEASE)
            cache.complete([Receipt(release.id, 0, Result.SUCCESS, True)])
        self.assertEqual(cache.close(), Status.OK)
        storage.extend(b"x")  # The cache relinquished its buffer export.

    def test_failed_open_releases_the_export(self):
        storage = bytearray(8)
        with self.assertRaises(Error) as caught:
            open_region(storage)
        self.assertEqual(caught.exception.status, Status.INVALID)
        storage.extend(b"x")


if __name__ == "__main__":
    unittest.main()

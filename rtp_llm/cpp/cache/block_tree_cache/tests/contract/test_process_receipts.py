import json
import subprocess
import sys
import unittest

from block_tree_cache import (
    Binding, Cache, Group, Operation, Page, Position, Receipt, Result, Rule, Status, WorkKind,
)


def pending_store():
    routes = {WorkKind.DEVICE_TO_HOST: [0], WorkKind.HOST_TO_DEVICE: [0]}
    cache = Cache(b"deployment", 1, [Group(0, Rule.FULL, 1)], host_slots=1, routes=routes)
    cache.publish([b"prefix"], 1, [Binding(Position(0, 1), (Page(0, 1, 1, 1),))])
    cache.demote(0, 1, [Page(0, 1, 1, 1)], remove_device=False)
    store, = cache.poll().work
    return cache, store.id


class ProcessReceiptTest(unittest.TestCase):
    def test_previous_process_cannot_complete_a_new_store(self):
        raw = subprocess.check_output([sys.executable, __file__, "--operation"], text=True)
        previous = Operation(*json.loads(raw))
        cache, current = pending_store()
        self.assertEqual(previous.operation, current.operation)
        cache.complete([Receipt(previous, 0, Result.SUCCESS, True)])
        self.assertEqual([work.id for work in cache.poll().work], [current])
        cache.complete([Receipt(current, 0, Result.SUCCESS, True)])
        cache.drain()
        for work in cache.poll().work:
            self.assertEqual(work.kind, WorkKind.RELEASE)
            cache.complete([Receipt(work.id, 0, Result.SUCCESS, True)])
        self.assertEqual(cache.close(), Status.OK)


if __name__ == "__main__":
    if sys.argv[1:] == ["--operation"]:
        # Exit with a work item outstanding, as on an engine restart.
        cache, operation = pending_store()
        print(json.dumps([operation.instance, operation.operation]))
    else:
        unittest.main()

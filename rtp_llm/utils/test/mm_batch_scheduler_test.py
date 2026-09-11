import threading
import time
import unittest
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass

from rtp_llm.utils.mm_batch_scheduler import MMBatchScheduler


@dataclass
class Image:
    identity: object
    num_patches: int = 1


class MMBatchSchedulerTest(unittest.TestCase):
    def scheduler(self, forward, images=4, patches=16, wait_ms=50):
        scheduler = MMBatchScheduler(forward, wait_ms, images, patches)
        self.addCleanup(scheduler.close)
        return scheduler

    def test_concurrent_requests_share_forward_and_keep_identity(self):
        calls = []

        def forward(images):
            calls.append([image.identity for image in images])
            return [image.identity for image in images]

        scheduler = self.scheduler(forward)
        barrier = threading.Barrier(4)

        def request(index):
            barrier.wait()
            return scheduler.submit(Image((index, index % 4))).result(timeout=2)

        with ThreadPoolExecutor(4) as executor:
            results = list(executor.map(request, range(4)))
        self.assertEqual(
            [result.output for result in results], [(i, i % 4) for i in range(4)]
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual({result.batch_size for result in results}, {4})
        self.assertEqual(len({result.batch_id for result in results}), 1)

    def test_patch_budget_splits_without_reordering_or_losing_items(self):
        calls = []

        def forward(images):
            calls.append(sum(image.num_patches for image in images))
            return [image.identity for image in images]

        scheduler = self.scheduler(forward, patches=5)
        futures = [
            scheduler.submit(Image(i, patches)) for i, patches in enumerate((3, 3, 2))
        ]
        self.assertEqual([future.result(2).output for future in futures], [0, 1, 2])
        self.assertTrue(all(size <= 5 for size in calls))
        self.assertEqual(len(calls), 2)
        with self.assertRaises(ValueError):
            scheduler.submit(Image("oversized", 6))

    def test_expired_and_cancelled_queue_items_never_run(self):
        entered, release, cancelled = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        calls = []

        def forward(images):
            calls.extend(image.identity for image in images)
            if images[0].identity == "first":
                entered.set()
                release.wait(2)
            return [image.identity for image in images]

        scheduler = self.scheduler(forward, images=1)
        first = scheduler.submit(Image("first"))
        self.assertTrue(entered.wait(1))
        expired = scheduler.submit(Image("expired"), deadline=time.monotonic() + 0.02)
        removed = scheduler.submit(Image("cancelled"), cancelled=cancelled)
        cancelled.set()
        time.sleep(0.03)
        release.set()
        self.assertEqual(first.result(2).output, "first")
        with self.assertRaises(TimeoutError):
            expired.result(2)
        with self.assertRaises(CancelledError):
            removed.result(2)
        self.assertEqual(scheduler.submit(Image("next")).result(2).output, "next")
        self.assertEqual(calls, ["first", "next"])

    def test_cancel_running_request_keeps_other_batch_result(self):
        entered, release, cancelled = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def forward(images):
            entered.set()
            release.wait(2)
            return [image.identity for image in images]

        scheduler = self.scheduler(forward, images=2)
        first = scheduler.submit(Image("cancel"), cancelled=cancelled)
        second = scheduler.submit(Image("keep"))
        self.assertTrue(entered.wait(1))
        cancelled.set()
        release.set()
        with self.assertRaises(CancelledError):
            first.result(2)
        self.assertEqual(second.result(2).output, "keep")

    def test_full_queue_obeys_submission_deadline(self):
        entered, release = threading.Event(), threading.Event()

        def forward(images):
            entered.set()
            release.wait(2)
            return [image.identity for image in images]

        scheduler = self.scheduler(forward, images=1)
        running = scheduler.submit(Image(0))
        self.assertTrue(entered.wait(1))
        queued = [scheduler.submit(Image(i)) for i in (1, 2)]
        try:
            with self.assertRaises(TimeoutError):
                scheduler.submit(Image(3), deadline=time.monotonic() + 0.02)
        finally:
            release.set()
        self.assertEqual(
            [future.result(2).output for future in [running] + queued], [0, 1, 2]
        )

    def test_output_count_error_fails_request_and_executor_survives(self):
        def forward(images):
            return (
                []
                if images[0].identity == "bad"
                else [image.identity for image in images]
            )

        scheduler = self.scheduler(forward, images=1)
        with self.assertLogs(level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "0 outputs for 1 images"):
                scheduler.submit(Image("bad")).result(2)
        self.assertEqual(scheduler.submit(Image("good")).result(2).output, "good")
        scheduler.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            scheduler.submit(Image("late"))

    def test_close_interrupts_batch_collection_and_fails_waiters(self):
        calls = []
        scheduler = self.scheduler(lambda images: calls.append(images), wait_ms=10000)
        pending = scheduler.submit(Image("pending"))
        scheduler.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            pending.result(1)
        self.assertFalse(scheduler._thread.is_alive())
        self.assertFalse(calls)

    def test_close_fails_queue_but_allows_running_forward_to_finish(self):
        entered, release = threading.Event(), threading.Event()

        def forward(images):
            entered.set()
            release.wait(2)
            return [image.identity for image in images]

        scheduler = self.scheduler(forward, images=1)
        running = scheduler.submit(Image("running"))
        self.assertTrue(entered.wait(1))
        pending = scheduler.submit(Image("pending"))
        closer = threading.Thread(target=scheduler.close)
        closer.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "closed"):
                pending.result(1)
        finally:
            release.set()
            closer.join(2)
        self.assertEqual(running.result(1).output, "running")
        self.assertFalse(closer.is_alive())
        self.assertFalse(scheduler._thread.is_alive())


if __name__ == "__main__":
    unittest.main()

"""CPU scheduling/ownership gates. These are not GPU numerical/perf evidence."""

import queue
import threading
import unittest
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.mm_batch_pipeline import PrefetchedMMBatch
from rtp_llm.multimodal.mm_scheduler import (
    MMScheduler,
    MMSchedulerExecutionError,
    MMSchedulerTimeoutError,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.gpu_input_transfer import (
    _pack_rgb_bytes,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.packed_graph import (
    PackedVisionGraph,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
    PreparedMMInputs,
)
from rtp_llm.utils.base_model_datatypes import MMUrlType


class Item:
    def __init__(self, value, timeout_ms=5000):
        self.preprocess_result = value
        self.mm_type = MMUrlType.IMAGE
        self.mm_inputs = [None]
        self.work_estimate = None
        self.mm_timeout_ms = timeout_ms
        self.embedding_result = None
        self.completed = False
        self.defer_feature_hashes = True
        self.feature_hashes = None

    def complete_cache(self, result):
        self.completed = True


class Backend(MultiModalEmbeddingInterface):
    def __init__(self, block=False, fail_prepare=False, fail_forward=False):
        self.entered = threading.Event()
        self.release = threading.Event()
        if not block:
            self.release.set()
        self.prepared_events = [threading.Event() for _ in range(8)]
        self.prepared = []
        self.forwarded = []
        self.closed = []
        self.waited = []
        self.stage_threads = []
        self.forward_threads = []
        self.fail_prepare = fail_prepare
        self.fail_forward = fail_forward

    def prepare_embedding_batch(self, data_list, mm_types):
        if self.fail_prepare:
            self.fail_prepare = False
            raise ValueError("upload failure")
        owner = self
        slot = len(self.prepared)
        values = [value + 100 for value in data_list]

        class Prepared(PreparedMMInputs):
            def wait(self):
                owner.waited.append(slot)

            def close(self):
                if self.data_list:
                    owner.closed.append(slot)
                super().close()

        result = Prepared(values)
        self.prepared.append(values)
        self.stage_threads.append(threading.current_thread().name)
        self.prepared_events[slot].set()
        return result

    def batched_embedding(self, data_list, mm_types):
        self.forwarded.append(list(data_list))
        self.forward_threads.append(threading.current_thread().name)
        self.entered.set()
        if not self.release.wait(10):
            raise TimeoutError("test release missing")
        if self.fail_forward:
            self.fail_forward = False
            raise ValueError("forward failure")
        return list(data_list)


class RGBBytePackingTest(unittest.TestCase):
    def test_mixed_shapes_and_nonzero_storage_offset_are_byte_exact(self):
        first = torch.arange(120, dtype=torch.uint8).reshape(4, 10, 3)[1:]
        second = torch.tensor([0, 127, 128, 254, 255, 1], dtype=torch.uint8).reshape(
            1, 2, 3
        )
        images = [second, first, second]
        expected = torch.cat([image.reshape(-1) for image in images])
        # A subview catches writes outside the destination range.
        storage = torch.full((expected.numel() + 6,), 77, dtype=torch.uint8)
        _pack_rgb_bytes(images, storage[3:-3])
        self.assertTrue(torch.equal(storage[3:-3], expected))
        self.assertEqual(storage[:3].tolist(), [77] * 3)
        self.assertEqual(storage[-3:].tolist(), [77] * 3)

    def test_upload_thread_avoids_aten_copy_without_changing_thread_count(self):
        images = [
            torch.full((8, 7, 3), 255, dtype=torch.uint8),
            torch.full((3, 2, 3), 128, dtype=torch.uint8),
        ]
        destination = torch.empty(sum(rgb.numel() for rgb in images), dtype=torch.uint8)
        expected = torch.cat([rgb.view(-1) for rgb in images])
        original_pointer = destination.data_ptr()
        thread_count = torch.get_num_threads()
        with patch.object(
            torch.Tensor, "copy_", side_effect=AssertionError("per-image copy")
        ):
            with patch.object(
                torch,
                "set_num_threads",
                side_effect=AssertionError("global thread change"),
            ):
                with ThreadPoolExecutor(1, thread_name_prefix="mm-upload") as executor:
                    executor.submit(_pack_rgb_bytes, images, destination).result(5)
        self.assertEqual(torch.get_num_threads(), thread_count)
        self.assertEqual(destination.data_ptr(), original_pointer)
        self.assertTrue(torch.equal(destination, expected))

    def test_distinct_batches_do_not_alias_source_or_each_other(self):
        image = torch.arange(18, dtype=torch.uint8).reshape(2, 3, 3)
        first = torch.empty(18, dtype=torch.uint8)
        second = torch.empty(18, dtype=torch.uint8)
        _pack_rgb_bytes([image], first)
        retained = first.clone()
        image.fill_(255)
        _pack_rgb_bytes([image], second)
        self.assertTrue(torch.equal(first, retained))
        self.assertEqual(second.tolist(), [255] * 18)

    def test_empty_pack_preserves_destination(self):
        destination = torch.full((3,), 19, dtype=torch.uint8)
        _pack_rgb_bytes([], destination)
        self.assertEqual(destination.tolist(), [19] * 3)


class BatchPipelineTest(unittest.TestCase):
    def test_ready_queue_is_drained_even_with_expired_wait_window(self):
        scheduler = MMScheduler.__new__(MMScheduler)
        scheduler._stopped = threading.Event()
        scheduler._collect_wakeup = threading.Event()
        scheduler._collector_idle = threading.Event()
        scheduler._waiting = queue.Queue()
        scheduler._pending = None
        scheduler._max_batch_size = 3
        scheduler._max_batch_images = 3
        scheduler._batch_wait_ms = 0
        scheduler._work_budget = None
        scheduler._batch_stats = Counter()
        scheduler._batch_stats_lock = threading.Lock()
        scheduler._report_queue_depth = lambda: None
        for _ in range(4):
            scheduler._waiting.put(
                SimpleNamespace(
                    n_images=1,
                    work_estimate=None,
                    request=SimpleNamespace(cancelled=threading.Event()),
                )
            )
        demanded = threading.Event()
        demanded.set()
        self.assertEqual(len(scheduler._collect_batch(demanded)), 3)
        self.assertEqual(scheduler._waiting.qsize(), 1)

    def test_cancelled_partial_lookahead_is_not_forwarded(self):
        model = Backend(block=True)
        scheduler = MMScheduler(
            model,
            max_batch_size=126,
            max_batch_images=126,
            batch_wait_ms=1,
            prefetch_batches=1,
            prefetch_fill_batch=True,
            device="cpu",
        )
        self.addCleanup(scheduler.close)
        self.addCleanup(model.release.set)
        with ThreadPoolExecutor(2) as callers:
            first = callers.submit(scheduler.submit_and_wait, [Item(0)])
            self.assertTrue(model.entered.wait(5))
            second = callers.submit(
                scheduler.submit_and_wait, [Item(1, timeout_ms=200)]
            )
            try:
                with self.assertRaises(MMSchedulerTimeoutError):
                    second.result(5)
            finally:
                model.release.set()
            first.result(5)
        self.assertTrue(scheduler.close())
        self.assertEqual(model.forwarded, [[100]])

    def test_fill_lookahead_collects_late_arrivals_up_to_unchanged_cap(self):
        model = Backend(block=True)
        scheduler = MMScheduler(
            model,
            max_batch_size=2,
            max_batch_images=2,
            batch_wait_ms=1,
            prefetch_batches=1,
            prefetch_fill_batch=True,
            device="cpu",
        )
        self.addCleanup(scheduler.close)
        self.addCleanup(model.release.set)
        items = [Item(i) for i in range(4)]
        with ThreadPoolExecutor(4) as callers:
            first = callers.submit(scheduler.submit_and_wait, [items[0]])
            self.assertTrue(model.entered.wait(5))
            second = callers.submit(scheduler.submit_and_wait, [items[1]])
            try:
                # Old policy seals this singleton at 1 ms, even though the
                # consumer is blocked. New policy leaves its capacity open.
                self.assertFalse(model.prepared_events[1].wait(0.1))
                third = callers.submit(scheduler.submit_and_wait, [items[2]])
                self.assertTrue(model.prepared_events[1].wait(5))
                self.assertEqual(model.prepared[1], [101, 102])
                fourth = callers.submit(scheduler.submit_and_wait, [items[3]])
                self.assertFalse(model.prepared_events[2].wait(0.1))
            finally:
                model.release.set()
            for future in (first, second, third, fourth):
                future.result(5)
        self.assertTrue(scheduler.close())
        self.assertEqual(model.forwarded, [[100], [101, 102], [103]])
        self.assertEqual(
            [item.embedding_result for item in items], [100, 101, 102, 103]
        )
        self.assertEqual(model.closed, [0, 1, 2])

    def test_consumer_demand_flushes_partial_without_waiting_for_full_batch(self):
        model = Backend(block=True)
        scheduler = MMScheduler(
            model,
            max_batch_size=126,
            max_batch_images=126,
            batch_wait_ms=1,
            prefetch_batches=1,
            prefetch_fill_batch=True,
            device="cpu",
        )
        self.addCleanup(scheduler.close)
        self.addCleanup(model.release.set)
        with ThreadPoolExecutor(2) as callers:
            first = callers.submit(scheduler.submit_and_wait, [Item(0)])
            self.assertTrue(model.entered.wait(5))
            second = callers.submit(scheduler.submit_and_wait, [Item(1)])
            self.assertFalse(model.prepared_events[1].wait(0.1))
            model.release.set()
            first.result(5)
            second.result(5)
        self.assertEqual(model.forwarded, [[100], [101]])

    def test_fill_mode_rejects_missing_prefetch(self):
        with self.assertRaisesRegex(ValueError, "requires prefetch_batches"):
            MMScheduler(Backend(), prefetch_fill_batch=True)

    def test_close_wakes_partial_collector_and_does_not_forward_it(self):
        model = Backend(block=True)
        scheduler = MMScheduler(
            model,
            max_batch_size=126,
            max_batch_images=126,
            batch_wait_ms=1,
            prefetch_batches=1,
            prefetch_fill_batch=True,
            device="cpu",
        )
        self.addCleanup(scheduler.close)
        self.addCleanup(model.release.set)
        with ThreadPoolExecutor(2) as callers:
            first = callers.submit(scheduler.submit_and_wait, [Item(0)])
            self.assertTrue(model.entered.wait(5))
            second = callers.submit(scheduler.submit_and_wait, [Item(1)])
            try:
                self.assertFalse(model.prepared_events[1].wait(0.1))
                self.assertFalse(scheduler.close(timeout=0.01))
            finally:
                model.release.set()
            first.result(5)
            with self.assertRaises(MMSchedulerExecutionError):
                second.result(5)
        self.assertTrue(scheduler.close())
        self.assertEqual(model.forwarded, [[100]])

    def scheduler(self, model, enabled=1):
        scheduler = MMScheduler(
            model,
            max_batch_size=1,
            batch_wait_ms=0,
            prefetch_batches=enabled,
            device="cpu",
        )
        self.addCleanup(scheduler.close)
        self.addCleanup(model.release.set)
        return scheduler

    def test_one_lookahead_overlaps_forward_and_preserves_inputs(self):
        model = Backend(block=True)
        scheduler = self.scheduler(model)
        items = [Item(i) for i in range(3)]
        with ThreadPoolExecutor(3) as callers:
            first = callers.submit(scheduler.submit_and_wait, [items[0]])
            self.assertTrue(model.entered.wait(5))
            second = callers.submit(scheduler.submit_and_wait, [items[1]])
            self.assertTrue(model.prepared_events[1].wait(5))
            third = callers.submit(scheduler.submit_and_wait, [items[2]])
            try:
                self.assertFalse(
                    model.prepared_events[2].wait(0.1), "more than one lookahead"
                )
            finally:
                model.release.set()
            for future in (first, second, third):
                future.result(5)
        self.assertTrue(scheduler.close())
        self.assertEqual([item.embedding_result for item in items], [100, 101, 102])
        self.assertEqual([item.preprocess_result for item in items], [0, 1, 2])
        self.assertTrue(all(item.completed for item in items))
        self.assertEqual(model.closed, [0, 1, 2])
        self.assertEqual(model.waited, [0, 1, 2])
        self.assertTrue(
            all(name.startswith("mm-upload") for name in model.stage_threads)
        )
        self.assertEqual(set(model.forward_threads), {"mm-scheduler"})

    def test_cancelled_uploaded_batch_is_released_without_forward(self):
        model = Backend(block=True)
        scheduler = self.scheduler(model)
        with ThreadPoolExecutor(2) as callers:
            first = callers.submit(scheduler.submit_and_wait, [Item(0)])
            self.assertTrue(model.entered.wait(5))
            second = callers.submit(
                scheduler.submit_and_wait, [Item(1, timeout_ms=200)]
            )
            try:
                self.assertTrue(model.prepared_events[1].wait(5))
                with self.assertRaises(MMSchedulerTimeoutError):
                    second.result(5)
            finally:
                model.release.set()
            first.result(5)
        self.assertTrue(scheduler.close())
        self.assertEqual(model.forwarded, [[100]])
        self.assertCountEqual(model.closed, [0, 1])

    def test_close_drains_uploaded_batch_after_inflight_forward(self):
        model = Backend(block=True)
        scheduler = self.scheduler(model)
        with ThreadPoolExecutor(2) as callers:
            first = callers.submit(scheduler.submit_and_wait, [Item(0)])
            self.assertTrue(model.entered.wait(5))
            second = callers.submit(scheduler.submit_and_wait, [Item(1)])
            try:
                self.assertTrue(model.prepared_events[1].wait(5))
                self.assertFalse(scheduler.close(timeout=0.01))
            finally:
                model.release.set()
            first.result(5)
            with self.assertRaises(MMSchedulerExecutionError):
                second.result(5)
        self.assertTrue(scheduler.close())
        self.assertEqual(model.forwarded, [[100]])
        self.assertCountEqual(model.closed, [0, 1])

    def test_stage_failure_propagates_without_retry_and_next_request_works(self):
        model = Backend(fail_prepare=True)
        scheduler = self.scheduler(model)
        with self.assertRaisesRegex(MMSchedulerExecutionError, "upload failure"):
            scheduler.submit_and_wait([Item(0)])
        item = Item(1)
        scheduler.submit_and_wait([item])
        self.assertEqual(item.embedding_result, 101)
        self.assertEqual(model.forwarded, [[101]])

    def test_poll_timeout_racing_with_success_does_not_stop_scheduler(self):
        real_result = Future.result

        def result(future, timeout=None):
            if (
                threading.current_thread().name == "mm-scheduler"
                and timeout is not None
            ):
                # Force the old result(timeout) race: the producer succeeds just
                # before the consumer handles its wait timeout. The fixed path
                # waits for completion separately and reads result() untimed.
                real_result(future, timeout=5)
                raise FutureTimeoutError()
            return real_result(future, timeout)

        with patch.object(Future, "result", result):
            model = Backend()
            scheduler = self.scheduler(model)
            for value in (0, 1):
                item = Item(value)
                scheduler.submit_and_wait([item])
                self.assertEqual(item.embedding_result, value + 100)
            self.assertFalse(scheduler._stopped.is_set())
            self.assertTrue(scheduler._executor.is_alive())
            self.assertTrue(scheduler.close())

    def test_collection_failure_fails_dequeued_requests_and_continues(self):
        model = Backend()
        scheduler = MMScheduler(
            model, max_batch_size=2, batch_wait_ms=100, prefetch_batches=1, device="cpu"
        )
        self.addCleanup(scheduler.close)
        first_dequeued = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)
        real_report = scheduler._report_queue_depth
        dequeued = 0

        def report():
            nonlocal dequeued
            if threading.current_thread().name.startswith("mm-upload"):
                dequeued += 1
                if dequeued == 1:
                    first_dequeued.set()
                    if not release.wait(5):
                        raise RuntimeError("test release missing")
                elif dequeued == 2:
                    raise FutureTimeoutError("collection failure")
            real_report()

        with patch.object(scheduler, "_report_queue_depth", report):
            first = scheduler.submit([Item(0)])
            self.assertTrue(first_dequeued.wait(5))
            second = scheduler.submit([Item(1)])
            release.set()
            for request in (first, second):
                with self.assertRaisesRegex(
                    MMSchedulerExecutionError, "collection failure"
                ) as caught:
                    scheduler.wait(request)
                self.assertEqual(
                    caught.exception.source_type, FutureTimeoutError.__name__
                )
            item = Item(2)
            scheduler.submit_and_wait([item])
            self.assertEqual(item.embedding_result, 102)
        self.assertEqual(model.forwarded, [[102]])
        self.assertTrue(scheduler._executor.is_alive())

    def test_consumer_failure_only_fails_current_batch(self):
        for error in (
            ValueError("consumer failure"),
            FutureTimeoutError("task timeout"),
        ):
            with self.subTest(error=type(error).__name__):
                model = Backend(block=True)
                scheduler = self.scheduler(model)
                first = scheduler.submit([Item(0)])
                self.assertTrue(model.entered.wait(5))
                failed = scheduler.submit([Item(1)])
                self.assertTrue(model.prepared_events[1].wait(5))
                execute = scheduler._execute_batch

                def fail_once(batch, prepared):
                    scheduler._execute_batch = execute
                    raise error

                # Fail the next consumer batch outside the forward's own guard.
                scheduler._execute_batch = fail_once
                model.release.set()
                scheduler.wait(first)
                with self.assertRaisesRegex(MMSchedulerExecutionError, str(error)):
                    scheduler.wait(failed)
                last = Item(2)
                scheduler.submit_and_wait([last])
                self.assertEqual(last.embedding_result, 102)
                self.assertTrue(scheduler.close())
                self.assertEqual(model.closed, [0, 1, 2])

    def test_prepare_timeout_is_request_failure_not_poll_timeout(self):
        model = Backend()
        scheduler = self.scheduler(model)
        with patch.object(
            model,
            "prepare_embedding_batch",
            side_effect=FutureTimeoutError("upload timeout"),
        ):
            with self.assertRaisesRegex(MMSchedulerExecutionError, "upload timeout"):
                scheduler.submit_and_wait([Item(0)])
        item = Item(1)
        scheduler.submit_and_wait([item])
        self.assertEqual(item.embedding_result, 101)
        self.assertEqual(model.forwarded, [[101]])

    def test_forward_failure_releases_upload_and_next_request_works(self):
        model = Backend(fail_forward=True)
        scheduler = self.scheduler(model)
        with self.assertRaisesRegex(MMSchedulerExecutionError, "forward failure"):
            scheduler.submit_and_wait([Item(0)])
        scheduler.submit_and_wait([Item(1)])
        self.assertTrue(scheduler.close())
        self.assertEqual(model.closed, [0, 1])

    def test_explicit_disable_does_not_call_prepare(self):
        model = Backend()
        scheduler = self.scheduler(model, enabled=0)
        item = Item(7)
        scheduler.submit_and_wait([item])
        self.assertEqual(item.embedding_result, 7)
        self.assertEqual(model.prepared, [])

    def test_cancelled_subset_keeps_surviving_item_mapping(self):
        items = [Item(3), Item(9)]
        batch = PrefetchedMMBatch([SimpleNamespace(work_items=items)])
        batch.prepare(Backend())
        self.assertEqual(batch.inputs_for([items[1]]), [109])
        batch.close()
        batch.close()
        self.assertEqual(batch.data_by_item, {})

    def test_config_default_on_and_validation(self):
        config = VitConfig()
        self.assertEqual(config.embedding_scheduler_args()["prefetch_batches"], 1)
        config.gpu_prefetch_batches = 0
        self.assertNotIn("prefetch_batches", config.embedding_scheduler_args())
        config.gpu_prefetch_batches = 2
        with self.assertRaises(ValueError):
            config.embedding_scheduler_args()

    def test_default_config_reaches_input_staging(self):
        config = VitConfig()
        model = Backend()
        scheduler = MMScheduler(
            model, device="cpu", **config.embedding_scheduler_args()
        )
        self.addCleanup(scheduler.close)
        item = Item(7)
        scheduler.submit_and_wait([item])
        self.assertTrue(scheduler.close())
        self.assertEqual(model.prepared, [[107]])
        self.assertEqual(item.embedding_result, 107)
        self.assertEqual(model.closed, [0])

    def test_graph_effective_cap_and_tails(self):
        graph = PackedVisionGraph(128, 1061888)
        self.assertEqual(graph.effective_batch_images((112, 75)), 126)
        self.assertEqual(graph.effective_batch_images((1, 1)), 128)
        self.assertEqual(graph.effective_batch_images((2000, 2000)), 0)
        self.assertEqual(PackedVisionGraph(128).effective_batch_images((112, 75)), 128)
        with self.assertRaises(ValueError):
            PackedVisionGraph(128, 0)

    def test_upload_budget_falls_back_before_cuda_allocation(self):
        import torch

        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.gpu_input_transfer import (
            RGBBatchUploader,
        )

        uploader = RGBBatchUploader("cuda:0", max_bytes=5)
        self.assertIsNone(
            uploader.upload([[torch.zeros(2, 2, 3, dtype=torch.uint8)]], [True])
        )
        self.assertEqual(uploader.stats()["budget_fallbacks"], 1)
        self.assertEqual(uploader.stats()["bytes"], 0)
        self.assertIsNone(uploader._stream)


if __name__ == "__main__":
    unittest.main()

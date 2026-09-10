"""Source ownership across the fastsafetensors producer/consumer streams."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.model_loader.per_expert_parallel_loader import PerExpertParallelLoader


def file_buffer(tensor):
    factory = SimpleNamespace(
        tensors=(
            {}
            if tensor is None
            else {"weight": SimpleNamespace(get_raw=lambda: tensor)}
        )
    )
    return SimpleNamespace(
        _get_rank_lidx=lambda key: (1, 0), rank_loaders={1: [factory]}
    )


class SourceStreamLifetimeTest(unittest.TestCase):
    def test_records_source_on_current_device_stream(self):
        source = mock.Mock(is_cuda=True, device="cuda:1")
        fb = file_buffer(source)
        stream = object()
        with mock.patch.object(
            torch.cuda, "current_stream", return_value=stream
        ) as current:
            PerExpertParallelLoader._record_source_stream(fb, "weight")
        current.assert_called_once_with("cuda:1")
        source.record_stream.assert_called_once_with(stream)

    def test_cpu_and_remote_rank_sources_do_not_touch_cuda(self):
        with mock.patch.object(torch.cuda, "current_stream") as current:
            PerExpertParallelLoader._record_source_stream(
                file_buffer(torch.ones(2)), "weight"
            )
            PerExpertParallelLoader._record_source_stream(file_buffer(None), "weight")
        current.assert_not_called()

    def test_records_before_broadcast_can_free_source(self):
        for stacked in (False, True):
            with self.subTest(stacked=stacked):
                events = []
                source = mock.Mock(is_cuda=True, device="cuda:0")
                source.record_stream.side_effect = lambda stream: events.append(
                    "record"
                )
                fb = file_buffer(source)

                def get_tensor(key):
                    events.append("clone_and_free")
                    fb.rank_loaders[1][0].tensors.clear()
                    return torch.ones(2)

                fb.get_tensor = get_tensor
                fb.close = mock.Mock()
                batch = SimpleNamespace(
                    fb=fb,
                    keys=["weight"],
                    batch_id=0,
                    add_filenames_time=0,
                    copy_files_time=0,
                )
                loader = PerExpertParallelLoader.__new__(PerExpertParallelLoader)
                loader.batch_queue = mock.Mock()
                loader.batch_queue.get.return_value = batch
                loader.queue_size = 0
                loader.consumer_processed = mock.Mock()
                loader._log_message = mock.Mock()
                loader._log_error = mock.Mock()
                loader.stacked_key_config = (
                    {"weight": "expert.{expert_id}"} if stacked else {}
                )
                loader._broadcast_per_expert = lambda batch, key: iter(
                    [(key, get_tensor(key))]
                )
                with mock.patch.object(torch.cuda, "current_stream"):
                    self.assertEqual(len(list(loader._consume_single_batch())), 1)
                self.assertEqual(events, ["record", "clone_and_free"])
                fb.close.assert_called_once_with()

    def test_supported_fastsafetensors_versions(self):
        with mock.patch(
            "rtp_llm.model_loader.per_expert_parallel_loader.ParallelLoader.__init__",
            return_value=None,
        ):
            for version in ("0.1.19", "0.1.19+ali", "0.1.20+ali"):
                with self.subTest(version=version), mock.patch(
                    "rtp_llm.model_loader.per_expert_parallel_loader.fastsafetensors.__version__",
                    version,
                ):
                    self.assertEqual(PerExpertParallelLoader({}).stacked_key_config, {})
            with mock.patch(
                "rtp_llm.model_loader.per_expert_parallel_loader.fastsafetensors.__version__",
                "0.2.0",
            ):
                with self.assertRaisesRegex(RuntimeError, "Internal API"):
                    PerExpertParallelLoader({})

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA allocator")
    def test_producer_allocation_is_not_reused_during_consumer_clone(self):
        producer = torch.cuda.default_stream()
        consumer = torch.cuda.Stream()
        with torch.cuda.stream(producer):
            source = torch.full(
                (16 * 1024 * 1024,), 37, dtype=torch.uint8, device="cuda"
            )
        fb = file_buffer(source)
        with torch.cuda.stream(consumer):
            consumer.wait_stream(producer)
            # Force the producer to release/reallocate while its consumer read
            # is queued. No global synchronize is part of the lifetime guard.
            torch.cuda._sleep(50_000_000)
            PerExpertParallelLoader._record_source_stream(fb, "weight")
            copied = source.clone()
        del source
        fb.rank_loaders[1][0].tensors.clear()
        with torch.cuda.stream(producer):
            replacement = torch.full_like(copied, 71)
        consumer.synchronize()
        self.assertTrue(torch.all(copied == 37).item())
        producer.synchronize()
        self.assertTrue(torch.all(replacement == 71).item())


if __name__ == "__main__":
    unittest.main()

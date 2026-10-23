"""CUDA lifetime regressions for background prefill metadata preparation."""

import importlib.util
import pathlib
import types
import unittest
from unittest.mock import patch

import torch

SOURCE = pathlib.Path(__file__).resolve().parents[1] / "fp8/_head_prebuild.py"
spec = importlib.util.spec_from_file_location("head_prebuild_stream_test", SOURCE)
head = importlib.util.module_from_spec(spec)
spec.loader.exec_module(head)


class AsyncHeadStreamTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "CUDA hardware is required")
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.builder = torch.cuda.Stream(device=self.device)
        self.consumer = torch.cuda.Stream(device=self.device)
        self.addCleanup(torch.cuda.synchronize)

    def test_consumed_storage_survives_next_builder_allocation(self):
        with torch.cuda.stream(self.builder):
            value = torch.full((1 << 18,), 7.0, device=self.device)
            pointer = value.data_ptr()
            bundle = head.PrebuiltHead(
                (1,), types.SimpleNamespace(positions=value), {4: (value,)}, {}, {}, 0
            )
            bundle.event = torch.cuda.Event()
            bundle.event.record(self.builder)
        with torch.cuda.stream(self.consumer):
            head._fence_bundle(bundle, self.device)
            torch.cuda._sleep(100_000_000)
            observed = value.clone()
        del value, bundle
        with torch.cuda.stream(self.builder):
            replacement = torch.empty((1 << 18,), device=self.device)
            replacement.fill_(-3.0)
        torch.cuda.synchronize()
        self.assertNotEqual(replacement.data_ptr(), pointer)
        self.assertTrue(torch.equal(observed, torch.full_like(observed, 7.0)))

    def test_builder_waits_for_input_producer(self):
        value = torch.zeros(1024, device=self.device)
        torch.cuda.synchronize()
        with torch.cuda.stream(self.consumer):
            torch.cuda._sleep(50_000_000)
            value.fill_(11.0)
            ready = torch.cuda.Event()
            ready.record(self.consumer)

        def build(_kwargs, _inference_mode):
            return head.PrebuiltHead((1,), None, {4: value.clone()}, {}, {}, 0)

        with patch.object(
            head, "_builder_stream", return_value=self.builder
        ), patch.object(head, "_build_under_inference_mode", side_effect=build):
            bundle = head._build_with_stream_fence(
                {"device": self.device}, ready_event=ready
            )
        bundle.event.synchronize()
        self.assertTrue(
            torch.equal(bundle.meta_by_ratio[4], torch.full_like(value, 11.0))
        )

    def test_draft_fences_but_preserves_target_prediction(self):
        target, draft = object(), object()
        value = torch.zeros(1024, device=self.device)
        torch.cuda.synchronize()
        with torch.cuda.stream(self.builder):
            torch.cuda._sleep(50_000_000)
            value.fill_(13.0)
            bundle = head.PrebuiltHead((1,), None, {4: value}, {}, {}, 0, owner=target)
            bundle.event = torch.cuda.Event()
            bundle.event.record(self.builder)
        worker = head.AsyncHeadBuilder()
        worker._result = ((1,), bundle)
        with torch.cuda.stream(self.consumer), patch.object(
            torch.cuda.Event, "synchronize", side_effect=AssertionError("host wait")
        ):
            self.assertIsNone(worker.consume((1,), owner=draft))
            draft_observed = value.clone()
        self.assertIs(worker.consume((1,), owner=target), bundle)
        with torch.cuda.stream(self.consumer):
            head._fence_bundle(bundle, self.device)
            observed = value.clone()
        self.consumer.synchronize()
        self.assertTrue(
            torch.equal(draft_observed, torch.full_like(draft_observed, 13.0))
        )
        self.assertTrue(torch.equal(observed, torch.full_like(observed, 13.0)))
        self.assertIsNone(worker.consume((1,), owner=target))

    def test_failed_build_drains_submitted_device_work(self):
        value = torch.zeros(1024, device=self.device)
        torch.cuda.synchronize()
        for error_type in (RuntimeError, SystemExit):
            with self.subTest(error_type=error_type):

                def build(_kwargs, _inference_mode):
                    torch.cuda._sleep(50_000_000)
                    value.fill_(11.0)
                    raise error_type("failed prediction")

                with patch.object(
                    head, "_builder_stream", return_value=self.builder
                ), patch.object(head, "_build_under_inference_mode", side_effect=build):
                    with self.assertRaises(error_type):
                        head._build_with_stream_fence({"device": self.device})
                self.assertTrue(self.builder.query())
                self.assertTrue(torch.equal(value, torch.full_like(value, 11.0)))

    def test_builder_input_storage_survives_closure_release(self):
        value = torch.full((1 << 18,), 7.0, device=self.device)
        pointer = value.data_ptr()
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))
        tables = {"swa": value}

        def build(kwargs, _inference_mode):
            torch.cuda._sleep(100_000_000)
            observed = kwargs["block_tables_by_type"]["swa"].clone()
            return head.PrebuiltHead((1,), None, {4: observed}, {}, {}, 0)

        with patch.object(
            head, "_builder_stream", return_value=self.builder
        ), patch.object(head, "_build_under_inference_mode", new=build):
            bundle = head._build_with_stream_fence(
                {"device": self.device, "block_tables_by_type": tables},
                ready_event=ready,
            )
        del tables, value
        replacement = torch.empty((1 << 18,), device=self.device)
        replacement.fill_(-3.0)
        torch.cuda.synchronize()
        observed = bundle.meta_by_ratio[4]
        self.assertNotEqual(replacement.data_ptr(), pointer)
        self.assertTrue(torch.equal(observed, torch.full_like(observed, 7.0)))


if __name__ == "__main__":
    unittest.main()

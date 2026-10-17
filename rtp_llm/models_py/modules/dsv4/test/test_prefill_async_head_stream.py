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


if __name__ == "__main__":
    unittest.main()

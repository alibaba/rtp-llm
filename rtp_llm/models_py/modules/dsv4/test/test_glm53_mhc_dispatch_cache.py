"""Measure code-only TileLang factory caching without graph workspaces."""

import importlib
import json
import os
import statistics
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import test_glm53_mhc_flat as fixtures
import torch

pre = importlib.import_module(
    "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.ops.pre_big_fuse"
)
post = importlib.import_module(
    "rtp_llm.models_py.3rdparty.tile_kernels.mhc.post_kernel"
)


class TestMhcDispatchCache(unittest.TestCase):
    def setUp(self):
        self.helper = fixtures.TestMhcFlat()
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        self.variants = ("original", "product")

    @contextmanager
    def variant(self, name):
        with patch.object(pre, "_COMPILED_DISPATCH", name == "product"), patch.object(
            post, "_COMPILED_DISPATCH", name == "product"
        ):
            yield

    def test_training_layout_and_async_regressions(self):
        with self.variant("product"):
            self.helper.test_batched_training_and_flat_rejection()
            self.helper.test_exact_layouts_and_alias()
            self.helper.test_devices_stream_and_capture()

    @torch.inference_mode()
    def test_bytes_ownership_devices_streams_capture(self):
        for device in (0, 1):
            with torch.cuda.device(device):
                layer = self.helper.layer(4096)
                for tokens in (1, 13, 81, 419, 513, 5103, 9889):
                    x = torch.randn(
                        tokens, 4, 4096, device="cuda", dtype=torch.bfloat16
                    )
                    with self.variant("original"):
                        expected = layer.pre(x)
                        expected_post = layer.post(
                            expected[0], x.clone(), *expected[1:]
                        )
                    for name in self.variants:
                        with self.variant(name):
                            actual = layer.pre(x)
                            self.helper.assert_bits(expected, actual)
                            saved = tuple(v.clone() for v in actual)
                            scratch = x.clone()
                            result = layer.post(actual[0], scratch, *actual[1:])
                            self.assertEqual(result.data_ptr(), scratch.data_ptr())
                            self.helper.assert_bits((expected_post,), (result,))
                            layer.pre(x)
                            self.helper.assert_bits(saved, actual)
                    if tokens == 81:
                        for name in self.variants:
                            stream = torch.cuda.Stream()
                            stream.wait_stream(torch.cuda.current_stream())
                            with self.variant(name), torch.cuda.stream(stream):
                                torch.cuda._sleep(100_000_000)
                                pending = torch.cuda.Event()
                                pending.record()
                                actual = layer.pre(x)
                                self.assertFalse(pending.query())
                            stream.synchronize()
                            self.helper.assert_bits(expected, actual)
                            graph = torch.cuda.CUDAGraph()
                            with self.variant(name), torch.cuda.graph(
                                graph, stream=stream
                            ):
                                captured = layer.pre(x)
                            graph.replay()
                            self.helper.assert_bits(expected, captured)

    @torch.inference_mode()
    def test_complete_wrapper_cpu_and_event(self):
        layer = self.helper.layer(4096)
        rows = []
        for tokens in (1, 81, 419, 5103, 9889):
            source = torch.randn(tokens, 4, 4096, device="cuda", dtype=torch.bfloat16)
            scratch = source.clone()

            def call():
                values = layer.pre(scratch)
                return layer.post(values[0], scratch, *values[1:])

            for name in self.variants:
                with self.variant(name):
                    for _ in range(30):
                        scratch.copy_(source)
                        call()
            samples = {name: [] for name in self.variants}
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            for repeat in range(60):
                names = list(self.variants)
                if repeat % 2:
                    names.reverse()
                for name in names:
                    with self.variant(name):
                        scratch.copy_(source)
                        torch.cuda.synchronize()
                        start.record()
                        begin = time.thread_time_ns()
                        call()
                        cpu = (time.thread_time_ns() - begin) / 1000
                        end.record()
                        end.synchronize()
                    samples[name].append(
                        dict(cpu_us=cpu, event_us=start.elapsed_time(end) * 1000)
                    )
            medians = {
                name: {
                    key: statistics.median(s[key] for s in values) for key in values[0]
                }
                for name, values in samples.items()
            }
            rows.append(dict(tokens=tokens, samples=samples, medians=medians))
            print("MHC_DISPATCH_CACHE", tokens, medians, flush=True)
        Path(
            os.environ["TEST_UNDECLARED_OUTPUTS_DIR"], "mhc_dispatch_cache.json"
        ).write_text(
            json.dumps(
                dict(
                    rows=rows,
                    note="Unchanged full public pre+post; freshoutputs and adapter validation;30warm/60alternating cleanthreadCPU/CUDAevents;nocopygraph/privateexecutable",
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    unittest.main()

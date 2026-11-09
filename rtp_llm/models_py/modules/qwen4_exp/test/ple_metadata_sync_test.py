"""PLE per-step metadata agreement and host uploads must not sync per item.

Serving timelines (warm7, 49-layer steps) attribute the ``qwen4.ple.apply``
window's device readbacks precisely: 6 ``torch.equal(cuda, cuda)`` calls from
the state-vs-context metadata agreement check (one D2H scalar readback each,
13-44µs, draining the pipeline) plus 2 ``host_input_mirror`` readbacks. The
same window also contains pageable synchronous H2D uploads of host-built index
lists (the terminal-page row list and the prefix mask).

This test locks the fixes:

* the fused metadata-agreement check answers the same True/False as the
  per-name comparisons (equal values with distinct storages, a differing
  value, mixed host/device pairs, missing attributes) using at most ONE
  device readback instead of six;
* the pinned int64 upload helper moves host values with a pinned staging
  buffer and a non-blocking copy (no pageable HtoD) and round-trips values
  exactly, including the empty list.
"""

import json
import os
import tempfile
import unittest

import torch


class _Inputs:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _dtoh_count(trace_events):
    return sum(
        1
        for e in trace_events
        if e.get("cat") == "gpu_memcpy" and "DtoH" in (e.get("name") or "")
    )


def _pageable_htod_count(trace_events):
    return sum(
        1
        for e in trace_events
        if e.get("cat") == "gpu_memcpy" and "HtoD (Pageable" in (e.get("name") or "")
    )


class PleMetadataSyncTest(unittest.TestCase):
    NAMES = (
        "input_lengths",
        "prefix_lengths",
        "sequence_lengths",
        "cu_seqlens_device",
        "cu_kv_seqlens_device",
        "combo_position_ids",
    )

    def _pair(self, seed, differing=False):
        torch.manual_seed(seed)
        values = {
            "input_lengths": torch.randint(1, 500, (8,), device="cuda", dtype=torch.int32),
            "prefix_lengths": torch.randint(0, 500, (8,), device="cuda", dtype=torch.int32),
            "sequence_lengths": torch.randint(1, 900, (8,), device="cuda", dtype=torch.int32),
            "cu_seqlens_device": torch.arange(9, device="cuda", dtype=torch.int32),
            "cu_kv_seqlens_device": torch.arange(9, device="cuda", dtype=torch.int32),
            "combo_position_ids": torch.arange(8, device="cuda", dtype=torch.int64),
        }
        lhs = _Inputs(**values)
        rhs = _Inputs(**{k: v.clone() for k, v in values.items()})
        if differing:
            rhs.input_lengths[0] = rhs.input_lengths[0] + 1
        return lhs, rhs

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fused_agreement_matches_per_name_result(self):
        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_execution import (
            PLEExecution,
        )

        lhs, rhs = self._pair(3)
        self.assertTrue(PLEExecution._metadata_agrees(lhs, rhs, self.NAMES))
        lhs, rhs = self._pair(5, differing=True)
        self.assertFalse(PLEExecution._metadata_agrees(lhs, rhs, self.NAMES))
        # Missing on both sides stays equal; host tensors compare for free.
        lhs, rhs = _Inputs(), _Inputs()
        self.assertTrue(PLEExecution._metadata_agrees(lhs, rhs, self.NAMES))
        lhs, rhs = _Inputs(a=torch.arange(4, device="cuda")), _Inputs(
            a=torch.arange(4, device="cuda").cpu()
        )
        self.assertFalse(PLEExecution._metadata_agrees(lhs, rhs, ("a",)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fused_agreement_uses_one_device_readback(self):
        from torch.profiler import ProfilerActivity, profile

        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_execution import (
            PLEExecution,
        )

        lhs, rhs = self._pair(7)
        self.assertTrue(PLEExecution._metadata_agrees(lhs, rhs, self.NAMES))
        lhs, rhs = self._pair(9)  # fresh tensors: no cached fast paths
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            self.assertTrue(PLEExecution._metadata_agrees(lhs, rhs, self.NAMES))
            torch.cuda.synchronize()
        path = os.path.join(tempfile.mkdtemp(prefix="ple_meta_"), "t.json")
        prof.export_chrome_trace(path)
        with open(path) as handle:
            evs = json.load(handle)["traceEvents"]
        self.assertLessEqual(
            _dtoh_count(evs),
            1,
            "the fused metadata agreement must use at most one device readback, "
            f"saw {_dtoh_count(evs)}",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_pinned_int64_upload_has_no_pageable_htod(self):
        from torch.profiler import ProfilerActivity, profile

        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_execution import (
            PLEExecution,
        )

        values = [3, 1, 4, 1, 5, 9, 2, 6]
        got = PLEExecution._upload_int64(values, torch.device("cuda"))
        self.assertEqual(got.tolist(), values)
        self.assertEqual(got.dtype, torch.long)
        self.assertTrue(got.is_cuda)
        empty = PLEExecution._upload_int64([], torch.device("cuda"))
        self.assertEqual(empty.numel(), 0)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            PLEExecution._upload_int64(values, torch.device("cuda"))
            torch.cuda.synchronize()
        path = os.path.join(tempfile.mkdtemp(prefix="ple_up_"), "t.json")
        prof.export_chrome_trace(path)
        with open(path) as handle:
            evs = json.load(handle)["traceEvents"]
        self.assertEqual(
            _pageable_htod_count(evs),
            0,
            "host-built int64 uploads must stage through pinned memory, saw "
            f"{_pageable_htod_count(evs)} pageable HtoD",
        )


if __name__ == "__main__":
    unittest.main()

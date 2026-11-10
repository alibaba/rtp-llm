import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.cp_kv_prefetch import (
    CrossLayerKvPrefetch,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.cp_profile import (
    PROFILE_STAGES,
    profile_stage,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.layered_prefill_cache import (
    ContiguousPrefillCache,
)


class CpProfileStagesTest(unittest.TestCase):
    def test_timing_targets_future_layer_and_disabled_creates_no_events(self):
        pipeline = object.__new__(CrossLayerKvPrefetch)
        pipeline.capture_timing = False
        pipeline.timing_rows = {}
        stream = object()
        event = Mock()
        with patch.object(torch.cuda, "Event", return_value=event) as make_event:
            pipeline.record_timing(5, "history_reorder_begin", stream)
            make_event.assert_not_called()
            pipeline.capture_timing = True
            pipeline.record_timing(5, "history_reorder_begin", stream)
            make_event.assert_called_once_with(enable_timing=True)
            event.record.assert_called_once_with(stream)
        self.assertEqual(pipeline.timing_rows, {5: {"history_reorder_begin": event}})

    def test_pending_cuda_timing_is_not_read_until_done(self):
        root = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        cache = object.__new__(ContiguousPrefillCache)
        start = Mock()
        start.elapsed_time.return_value = 1.25
        end = Mock()
        done = Mock()
        done.query.return_value = False
        cache.pending_timings = [
            (
                done,
                [
                    {
                        "layer": 5,
                        "weight": 1,
                        "history_reorder_begin": start,
                        "history_reorder_end": end,
                    }
                ],
                str(root),
            )
        ]
        with patch.dict(os.environ, {"WORLD_RANK": "timing_test"}):
            cache.flush_timings()
            start.elapsed_time.assert_not_called()
            self.assertEqual(len(cache.pending_timings), 1)
            done.query.return_value = True
            cache.flush_timings()
        row = json.loads(
            (root / "prefill_copy_timing_ranktiming_test.jsonl").read_text()
        )["layers"][0]
        self.assertEqual(row["layer"], 5)
        self.assertEqual(row["history_reorder_span_ms"], 1.25)
        self.assertIsNone(row["cp_ready_wait_ms"])
        self.assertIsNone(row["new_reorder_span_ms"])
        self.assertEqual(cache.pending_timings, [])

    def test_disabled_returns_original_callable_and_freezes_startup_flag(self):
        def identity(value: int) -> int:
            return value

        opposite = "0" if PROFILE_STAGES else "1"
        with patch.dict(os.environ, {"RTP_LLM_CP_PROFILE_STAGES": opposite}):
            wrapped = profile_stage("cp.test")(identity)
        if PROFILE_STAGES:
            self.assertIsNot(wrapped, identity)
        else:
            self.assertIs(wrapped, identity)
            with patch(
                "torch.profiler.record_function",
                side_effect=AssertionError("disabled profiling constructed a scope"),
            ):
                self.assertEqual(wrapped(13), 13)

    def test_real_wait_methods_keep_dependency_calls_and_emit_exact_phase_counts(self):
        events = []
        ready = object()
        compute = SimpleNamespace(wait_event=lambda event: events.append(event))
        pipeline = object.__new__(CrossLayerKvPrefetch)
        pipeline.latest_ready = ready
        pipeline.compute = compute
        staging = object.__new__(ContiguousPrefillCache)
        staging.current_layer = 7
        staging.ready = [ready]
        staging._timing_event = lambda name: events.append(name)

        def layer_identity(receiver, layer: int) -> int:
            return layer

        layer_call = profile_stage("cp.test.layer", layer_arg=1)(layer_identity)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profiler:
            for _ in range(2):
                pipeline.wait_before_moe()
                staging._wait_ready(0, compute)
            self.assertEqual(layer_call(pipeline, 5), 5)
            self.assertEqual(layer_call(pipeline, layer=6), 6)
        self.assertEqual(
            events,
            [ready, "kv_wait_begin", ready, "kv_wait_end"] * 2,
        )
        expected = {
            "cp.collective.wait_before_moe": 2,
            "cp.staging.wait(layer=7)": 2,
            "cp.test.layer(layer=5)": 1,
            "cp.test.layer(layer=6)": 1,
        }
        names = [event.name for event in profiler.events()]
        counts = {name: names.count(name) for name in expected}
        self.assertEqual(
            counts, expected if PROFILE_STAGES else dict.fromkeys(expected, 0)
        )
        root = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        profiler.export_chrome_trace(str(root / "cp_profile_stages.json"))
        (root / "cp_profile_stages_counts.json").write_text(
            json.dumps(
                {"enabled": PROFILE_STAGES, "counts": counts, "wait_event_calls": 4},
                indent=2,
            )
        )


if __name__ == "__main__":
    unittest.main()

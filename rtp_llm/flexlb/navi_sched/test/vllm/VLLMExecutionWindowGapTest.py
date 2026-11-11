import unittest

from navi_sched.adapters.vllm.VLLMExecutionSample import (
    PrefillExecutionSample,
    PrefillStepRequest,
)
from navi_sched.adapters.vllm.VLLMExecutionWindow import PrefillExecutionWindowAssembler


class VLLMExecutionWindowGapTest(unittest.TestCase):
    def sample(self, identity, start):
        return PrefillExecutionSample(
            identity,
            (PrefillStepRequest(identity, 0, 8192, 0, identity),),
            start,
            start + 4000000,
            8192,
        )

    def complete(self, starts, order=None):
        assembler = PrefillExecutionWindowAssembler((len(starts),))
        for identity in range(1, len(starts) + 1):
            assembler.register(identity)
        windows = []
        for identity in order or range(1, len(starts) + 1):
            windows.extend(
                assembler.complete(
                    identity, self.sample(identity, starts[identity - 1])
                )
            )
        return assembler, windows

    def test_empty_control_steps_do_not_discard_prefill_samples(self):
        from types import SimpleNamespace

        from navi_sched.adapters.vllm.VLLMEngineObserver import VLLMEngineObserver

        observer = VLLMEngineObserver.__new__(VLLMEngineObserver)
        observer._native_ingress = True
        observer._v6d_scheduler = None
        observer._observed_execution_window_training = True
        assembler = PrefillExecutionWindowAssembler((4,))
        observer._prefill_execution_windows = assembler
        assembler.register(1)
        observer._prepare_execution_step(
            SimpleNamespace(num_scheduled_tokens={}),
            lambda identity: self.fail("empty control step must not read requests"),
            SimpleNamespace(started_ns=2, prefill_tokens=0, batch_progress=[]),
        )
        assembler.register(3)
        windows = list(assembler.complete(2, None))
        windows.extend(assembler.complete(1, self.sample(1, 1000000)))
        windows.extend(assembler.complete(3, self.sample(3, 3000000)))
        self.assertEqual([1, 3], [s.step_id for w in windows for s in w.steps])
        self.assertEqual(0, assembler.rejected_windows)

    def test_idle_gap_preserves_both_execution_spans(self):
        assembler, windows = self.complete([1000000, 100000000])
        self.assertEqual([1, 1], [w.execution_step_count for w in windows])
        self.assertEqual([4.0, 4.0], [w.elapsed_ms for w in windows])
        self.assertEqual(16384, sum(w.global_padded_tokens for w in windows))
        self.assertEqual(0, assembler.rejected_windows)

    def test_small_gaps_retain_every_hot_cache_step(self):
        assembler, windows = self.complete([1000000, 5000001, 9000002, 13000003])
        self.assertEqual([1, 2, 3, 4], [s.step_id for w in windows for s in w.steps])
        self.assertEqual(16.0, sum(w.elapsed_ms for w in windows))
        self.assertEqual(0, assembler.rejected_windows)
        self.assertEqual((), assembler.complete(4, self.sample(4, 13000003)))

    def test_out_of_order_completion_preserves_overlapping_groups(self):
        assembler, windows = self.complete(
            [1000000, 2000000, 10000000, 11000000], [4, 2, 3, 1]
        )
        self.assertEqual(
            [[1, 2], [3, 4]], [[s.step_id for s in w.steps] for w in windows]
        )
        self.assertEqual([5.0, 5.0], [w.elapsed_ms for w in windows])
        self.assertEqual([16384, 16384], [w.global_padded_tokens for w in windows])
        self.assertEqual(0, assembler.rejected_windows)

    def test_continuous_steps_keep_one_window(self):
        assembler, windows = self.complete([1000000, 2000000, 3000000, 4000000])
        self.assertEqual(1, len(windows))
        self.assertEqual(7.0, windows[0].elapsed_ms)
        self.assertEqual(4, windows[0].execution_step_count)
        self.assertEqual(0, assembler.rejected_windows)

    def test_reordered_start_times_cannot_emit_partial_windows(self):
        assembler, windows = self.complete([1000000, 100000000, 2000000])
        self.assertEqual([], windows)
        self.assertEqual(1, assembler.rejected_windows)


if __name__ == "__main__":
    unittest.main()

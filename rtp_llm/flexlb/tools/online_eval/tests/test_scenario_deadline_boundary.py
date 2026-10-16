"""Deterministic clock crossings previously passed negative values to sleep."""

import unittest

from scenario.runtime import Deadline, StageTimeout


class DeadlineBoundaryTest(unittest.TestCase):
    def test_sleep_crossing_uses_one_clock_sample_for_each_delay(self):
        values = iter([0, 0.09, 0.11, 0.12, 0.13, 0.14])
        delays = []
        Deadline(1, clock=lambda: next(values), sleeper=delays.append).sleep(0.1)
        self.assertEqual(len(delays), 1)
        self.assertGreater(delays[0], 0)
        self.assertAlmostEqual(delays[0], 0.01)

    def test_remaining_cannot_cross_deadline_between_check_and_return(self):
        values = iter([0.99, 1.01])
        deadline = Deadline(1, clock=lambda: next(values))
        self.assertAlmostEqual(deadline.remaining(), 0.01)
        with self.assertRaises(StageTimeout):
            deadline.remaining()

    def test_sleep_expiry_remains_timeout_without_negative_sleep(self):
        values = iter([0, 0.99, 1.01])
        delays = []
        with self.assertRaises(StageTimeout):
            Deadline(1, clock=lambda: next(values), sleeper=delays.append).sleep(2)
        self.assertEqual(len(delays), 1)
        self.assertGreater(delays[0], 0)
        self.assertAlmostEqual(delays[0], 0.01)


if __name__ == "__main__":
    unittest.main()


def test_policy_hook_cannot_consume_budget_and_still_start_operation(tmp_path):
    from types import SimpleNamespace
    from scenario import compile_scenarios
    from scenario.plan import CompiledInstance
    from scenario.context import RuntimeContext
    from scenario.stage_execution import StageExecutor
    from test_scenario_runtime import source, Clock, Backend
    clock = Clock()
    backend = Backend(clock)
    doc = source()
    compiled = compile_scenarios([(str(tmp_path / "case.yaml"), doc)])[0]
    ctx = RuntimeContext(compiled, backend, tmp_path, clock, clock.sleep, False)
    ctx.plan = CompiledInstance.read(compiled)
    policy = SimpleNamespace(attach=lambda ctx: None,
        before_stage=lambda ctx, spec: clock.sleep(100000),
        after_stage=lambda ctx, spec, row: None)
    summary = StageExecutor().run(ctx, {}, clock(), clock() + 1000, None, policy)
    assert summary.status == "TIMEOUT"
    assert not backend.calls

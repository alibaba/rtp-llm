"""Smoke coverage for the trace-memory pybind surface and the MoE skew defaults.

Both live here because both need the real C++ extension loaded, which the pure-math
test_warmup_skew.py deliberately avoids (it fakes the binding instead).

Irreversible-state constraint: the trace phase is one process-global state machine
(Pending -> Active -> Finished, see TraceMemoryState in ExecOps.h). This binding
exposes no way back, so finish_trace_memory() burns the phase for the whole process
and this file deliberately contains ONE test that walks the machine in order:

  1. read the phase and the boolean view while the phase is still whatever the
     process started in (Pending in a bare test process, since no engine ran),
  2. drive the single transition this binding exposes (-> Finished),
  3. assert Finished is observed and that repeating the call is a no-op.

"Finished is final" here means final for this warmup lifecycle and for everything
this binding can do. It is not final for the process: a second model build calls
setTraceMemory(true) on the C++ side, which re-activates the phase on purpose.

Do not add a second test that also calls finish_trace_memory(): unittest gives no
ordering guarantee across methods, and whichever ran second would see a phase the
first one already consumed. Anything needing a fresh phase belongs in a separate
process (see test_warmup_skew.py, which fakes the binding instead).

That constraint is about the trace phase only. MoeSkewDefaultTest below touches no
process-global state, so it is a legitimate second test class in this file -- do not
delete it while enforcing the paragraph above.
"""

import unittest

# TraceMemoryPhase values from rtp_llm/models_py/bindings/core/ExecOps.h. A
# static_assert next to that enum pins them at 0/1/2 precisely because this file
# and warmup_diagnostics.py compare the binding's return value against these
# literals: a C++ renumbering fails the build instead of silently diverging.
_PENDING, _ACTIVE, _FINISHED = 0, 1, 2


class WarmupBindingTest(unittest.TestCase):
    def test_trace_memory_bindings_walk_the_state_machine_in_order(self):
        from rtp_llm.ops.compute_ops import (
            finish_trace_memory,
            get_trace_memory_state,
            is_trace_memory,
        )

        # Step 1: observe. No engine ran in this process, so nothing should have
        # activated the phase; assert the pre-transition invariants rather than a
        # specific value beyond that.
        self.assertTrue(callable(is_trace_memory))
        self.assertTrue(callable(get_trace_memory_state))
        initial_state = get_trace_memory_state()
        self.assertIn(initial_state, (_PENDING, _ACTIVE, _FINISHED))
        self.assertIsInstance(is_trace_memory(), bool)
        # is_trace_memory() is exactly "phase == Active", the gate the MoE modules use.
        self.assertEqual(is_trace_memory(), initial_state == _ACTIVE)

        # Step 2: the one transition this binding exposes. Irreversible.
        finish_trace_memory()

        # Step 3: Finished is final as far as this binding goes, and idempotent.
        self.assertEqual(get_trace_memory_state(), _FINISHED)
        self.assertFalse(is_trace_memory())
        finish_trace_memory()
        self.assertEqual(get_trace_memory_state(), _FINISHED)


class MoeSkewDefaultTest(unittest.TestCase):
    """Pin the diagnostics singleton's initial skew default to MoeConfig.

    MoeConfig (rtp_llm/cpp/config/ConfigModules.h) owns the default and reaches
    the singleton twice: at import time (warmup_diagnostics reads
    MoeConfig().moe_skew_mult for the module-level singleton's initial value)
    and per model build through ModelFactory -> reload_runtime_diagnostics().
    This test pins the import-time wiring; a change that stops __init__ from
    reading the binding would otherwise only surface as a stale pre-reload
    value in tests that never call reload.
    """

    def test_diagnostics_singleton_starts_from_the_same_defaults(self):
        from rtp_llm.models_py.modules.factory.fused_moe.defs.warmup_diagnostics import (
            diagnostics,
        )
        from rtp_llm.ops import MoeConfig

        moe_config = MoeConfig()
        self.assertEqual(diagnostics.skew_mult, moe_config.moe_skew_mult)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

import rtp_llm.utils.backend_registry as backend_registry

from rtp_llm.utils.backend_registry import (
    register_backend_hook,
    reset_backend_registrations,
    run_backend_registrations,
)


class BackendRegistryTest(unittest.TestCase):
    def setUp(self):
        reset_backend_registrations()
        self._entrypoint_patcher = patch(
            "rtp_llm.utils.import_util.import_optional_internal_source_entrypoint",
            return_value=False,
        )
        self._entrypoint_patcher.start()

    def tearDown(self):
        self._entrypoint_patcher.stop()
        reset_backend_registrations()

    def test_hook_runs_with_context_when_slot_drained(self):
        seen = []
        register_backend_hook("linear", lambda factory: seen.append(factory))

        self.assertEqual(seen, [], "hook must not run before the slot is drained")
        run_backend_registrations("linear", factory="LinearFactory")
        self.assertEqual(seen, ["LinearFactory"])

    def test_hooks_run_in_registration_order(self):
        seen = []
        register_backend_hook("moe", lambda: seen.append("first"))
        register_backend_hook("moe", lambda: seen.append("second"))

        run_backend_registrations("moe")
        self.assertEqual(seen, ["first", "second"])

    def test_draining_twice_does_not_rerun_hooks(self):
        calls = []
        register_backend_hook("linear", lambda: calls.append(1))

        run_backend_registrations("linear")
        run_backend_registrations("linear")
        self.assertEqual(calls, [1])

    def test_repeatable_slot_replays_frozen_hooks_for_each_owner(self):
        seen = []
        register_backend_hook(
            "moe_strategy_choices", lambda parser: seen.append(parser)
        )

        run_backend_registrations(
            "moe_strategy_choices", repeatable=True, parser="first"
        )
        run_backend_registrations(
            "moe_strategy_choices", repeatable=True, parser="second"
        )

        self.assertEqual(seen, ["first", "second"])

    def test_repeatable_slot_rejects_late_hooks(self):
        run_backend_registrations("parser", repeatable=True, parser="first")

        with self.assertRaises(RuntimeError):
            register_backend_hook("parser", lambda parser: None)

    def test_slot_lifecycle_cannot_change_after_start(self):
        run_backend_registrations("parser", repeatable=True, parser="first")

        with self.assertRaises(RuntimeError):
            run_backend_registrations("parser", parser="second")

    @patch("rtp_llm.utils.import_util.import_optional_internal_source_entrypoint")
    def test_entrypoint_loads_before_slot_is_consumed(self, load_entrypoint):
        seen = []

        def load(relative_module):
            self.assertEqual(relative_module, "models_py")
            register_backend_hook("linear", lambda factory: seen.append(factory))
            return True

        load_entrypoint.side_effect = load
        run_backend_registrations("linear", factory="LinearFactory")

        self.assertEqual(seen, ["LinearFactory"])

    def test_other_slots_are_untouched(self):
        calls = []
        register_backend_hook("attention", lambda: calls.append(1))

        run_backend_registrations("linear")
        self.assertEqual(calls, [])

    def test_registering_after_drain_raises(self):
        run_backend_registrations("linear")

        # A hook recorded after its slot was drained would never run, so
        # surface it instead of silently dropping the backend.
        with self.assertRaises(RuntimeError):
            register_backend_hook("linear", lambda: None)

    def test_hook_exception_propagates(self):
        def broken():
            raise ValueError("backend is broken")

        register_backend_hook("linear", broken)

        # Swallowing this would leave the factory silently selecting a
        # different implementation, i.e. wrong numerics instead of a crash.
        with self.assertRaisesRegex(ValueError, "backend is broken"):
            run_backend_registrations("linear")

    def test_failed_hook_does_not_poison_slot(self):
        attempts = []

        def flaky_hook():
            attempts.append(len(attempts))
            if len(attempts) == 1:
                raise ValueError("transient backend failure")

        register_backend_hook("linear", flaky_hook)
        with self.assertRaisesRegex(ValueError, "transient backend failure"):
            run_backend_registrations("linear")
        run_backend_registrations("linear")
        self.assertEqual(attempts, [0, 1])

    def test_hooks_run_without_holding_registry_lock(self):
        lock_owned = []

        def check_lock_state():
            lock_owned.append(backend_registry._lock._is_owned())

        register_backend_hook("linear", check_lock_state)
        run_backend_registrations("linear")
        self.assertEqual(lock_owned, [False])

    def test_draining_slot_without_hooks_is_noop(self):
        run_backend_registrations("nobody_registered_here")


if __name__ == "__main__":
    unittest.main()

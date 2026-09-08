import unittest

from rtp_llm.utils.scr_template_lifecycle import (
    CallbackHook,
    TemplateLifecycle,
)


class TemplateLifecycleTest(unittest.TestCase):
    def test_hooks_run_in_order(self):
        events = []
        lifecycle = TemplateLifecycle()
        lifecycle.register(
            "one",
            CallbackHook(
                prepare=lambda generation: events.append(("prepare", generation)),
                fixup=lambda generation: events.append(("fixup", generation)),
                release=lambda generation: events.append(("release", generation)),
            ),
        )
        lifecycle.prepare_for_template("g1", "restore")
        lifecycle.restore_fixup("g1")
        lifecycle.release_template("g1")
        self.assertEqual(
            events,
            [("prepare", "g1"), ("fixup", "g1"), ("release", "g1")],
        )
        self.assertFalse(lifecycle.active)

    def test_failure_aborts_completed_hooks(self):
        events = []
        lifecycle = TemplateLifecycle()
        lifecycle.register(
            "one",
            CallbackHook(
                prepare=lambda _generation: events.append("prepare"),
                abort=lambda _generation: events.append("abort"),
            ),
        )
        lifecycle.register(
            "two",
            CallbackHook(prepare=lambda _generation: (_ for _ in ()).throw(ValueError("boom"))),
        )
        with self.assertRaises(ValueError):
            lifecycle.prepare_for_template("g1", "checkpoint")
        self.assertEqual(events, ["prepare", "abort"])
        self.assertFalse(lifecycle.active)

    def test_duplicate_prepare_is_idempotent(self):
        lifecycle = TemplateLifecycle()
        calls = []
        lifecycle.register("one", CallbackHook(prepare=lambda _generation: calls.append(1)))
        lifecycle.prepare_for_template("g1", "restore")
        lifecycle.prepare_for_template("g1", "restore")
        self.assertEqual(calls, [1])


if __name__ == "__main__":
    unittest.main()

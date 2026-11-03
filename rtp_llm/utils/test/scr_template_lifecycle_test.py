import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from rtp_llm.utils.scr_restore_context import RestoreContext
from rtp_llm.utils.scr_template_lifecycle import CallbackHook, TemplateLifecycle


class TemplateLifecycleTest(unittest.TestCase):
    def test_hooks_run_in_order(self):
        events = []
        lifecycle = TemplateLifecycle()
        lifecycle.register(
            "one",
            CallbackHook(
                prepare=lambda generation: events.append(("prepare", generation)),
                fixup=lambda context: events.append(("fixup", context.generation)),
                release=lambda generation: events.append(("release", generation)),
            ),
        )
        lifecycle.prepare_for_template("g1", "restore")
        lifecycle.restore_fixup(RestoreContext("g1", "192.0.2.50"))
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
            CallbackHook(
                prepare=lambda _generation: (_ for _ in ()).throw(ValueError("boom"))
            ),
        )
        with self.assertRaises(ValueError):
            lifecycle.prepare_for_template("g1", "checkpoint")
        self.assertEqual(events, ["prepare", "abort"])
        self.assertFalse(lifecycle.active)

    def test_duplicate_prepare_is_idempotent(self):
        lifecycle = TemplateLifecycle()
        calls = []
        lifecycle.register(
            "one", CallbackHook(prepare=lambda _generation: calls.append(1))
        )
        lifecycle.prepare_for_template("g1", "restore")
        lifecycle.prepare_for_template("g1", "restore")
        self.assertEqual(calls, [1])

    def test_fixup_only_components_share_inputs_before_release(self):
        lifecycle = TemplateLifecycle()
        events = []
        context = RestoreContext("seed", "192.0.2.20")
        for name in ("server-config", "backend-endpoints", "request-identity"):
            participant = SimpleNamespace(
                restore_fixup=lambda ctx, name=name: events.append((name, ctx))
            )
            lifecycle.register_fixup(name, participant)
            lifecycle.register_fixup(name, participant)
        lifecycle.register(
            "service", CallbackHook(release=lambda g: events.append(("ready", g)))
        )
        lifecycle.prepare_for_template("seed", "restore")
        lifecycle.restore_fixup(context)
        lifecycle.release_template("seed")
        self.assertEqual(
            events,
            [
                (name, context)
                for name in ("server-config", "backend-endpoints", "request-identity")
            ]
            + [("ready", "seed")],
        )
        for _, received in events[:-1]:
            self.assertIs(received, context)

    def test_failed_fixup_blocks_release_and_allows_reverse_abort(self):
        lifecycle = TemplateLifecycle()
        events = []
        release = Mock()
        lifecycle.register("one", CallbackHook(abort=lambda g: events.append("one")))
        lifecycle.register(
            "two",
            CallbackHook(
                fixup=Mock(side_effect=RuntimeError("endpoint fixup failed")),
                release=release,
                abort=lambda g: events.append("two"),
            ),
        )
        lifecycle.prepare_for_template("seed", "restore")
        with self.assertRaisesRegex(RuntimeError, "endpoint fixup failed"):
            lifecycle.restore_fixup(RestoreContext("seed", "192.0.2.20"))
        with self.assertRaisesRegex(RuntimeError, "before successful fixup"):
            lifecycle.release_template("seed")
        release.assert_not_called()
        lifecycle.abort_template("seed")
        self.assertEqual(events, ["two", "one"])
        self.assertFalse(lifecycle.active)

    def test_release_failure_preserves_abort_cleanup(self):
        lifecycle = TemplateLifecycle()
        abort = Mock()
        lifecycle.register(
            "service",
            CallbackHook(
                release=Mock(side_effect=RuntimeError("start failed")), abort=abort
            ),
        )
        lifecycle.prepare_for_template("seed", "restore")
        lifecycle.restore_fixup(RestoreContext("seed", "192.0.2.20"))
        with self.assertRaisesRegex(RuntimeError, "start failed"):
            lifecycle.release_template("seed")
        lifecycle.abort_template("seed")
        abort.assert_called_once_with("seed")
        self.assertFalse(lifecycle.active)

    def test_participant_set_cannot_change_during_restore(self):
        lifecycle = TemplateLifecycle()
        lifecycle.register("one", CallbackHook())
        lifecycle.prepare_for_template("seed", "restore")
        with self.assertRaisesRegex(RuntimeError, "cannot register"):
            lifecycle.register_fixup("late", SimpleNamespace(restore_fixup=Mock()))
        with self.assertRaisesRegex(RuntimeError, "cannot unregister"):
            lifecycle.unregister("one")
        with self.assertRaisesRegex(RuntimeError, "before successful fixup"):
            lifecycle.release_template("seed")


if __name__ == "__main__":
    unittest.main()

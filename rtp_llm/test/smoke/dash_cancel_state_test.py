import unittest

from dash_cancel_state import classify_dash_cancel


class DashCancelStateTest(unittest.TestCase):
    def test_cancel_wins_only_when_cancel_call_exercised(self):
        state = classify_dash_cancel(
            cancel_exercised=True,
            terminal_code="CANCELLED",
            cancelled_code="CANCELLED",
            ok_code="OK",
        )
        self.assertTrue(state.cancel_requested)
        self.assertTrue(state.cancel_exercised)
        self.assertTrue(state.cancelled)
        self.assertFalse(state.completed_before_cancel)

    def test_peer_cancel_is_not_reported_as_comparer_cancel(self):
        state = classify_dash_cancel(
            cancel_exercised=False,
            terminal_code="CANCELLED",
            cancelled_code="CANCELLED",
            ok_code="OK",
        )
        self.assertTrue(state.cancel_requested)
        self.assertFalse(state.cancel_exercised)
        self.assertFalse(state.cancelled)

    def test_normal_completion_before_cancel_is_explicit(self):
        state = classify_dash_cancel(
            cancel_exercised=False,
            terminal_code="OK",
            cancelled_code="CANCELLED",
            ok_code="OK",
        )
        self.assertTrue(state.completed_before_cancel)
        self.assertFalse(state.cancelled)

    def test_unexpected_terminal_code_is_rejected(self):
        with self.assertRaises(ValueError):
            classify_dash_cancel(
                cancel_exercised=True,
                terminal_code="INTERNAL",
                cancelled_code="CANCELLED",
                ok_code="OK",
            )


if __name__ == "__main__":
    unittest.main()

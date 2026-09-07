"""Shared fire/drain preserves dispatch and terminal ownership boundaries."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flexlb_ft.support.requests import drain_fired, fire_request


class RequestLifecycleTest(unittest.TestCase):
    def setup_ops(self, batch=True):
        response = SimpleNamespace(code=200, success=True, enqueued_by_master=batch)
        stream = Mock()
        stream.wait_end.return_value = True
        stream.snap = SimpleNamespace(completed=True, error=None)
        ops = Mock()
        ops.schedule.return_value = response
        ops.role_addr.return_value = "p:1"
        ops.addr_to_name.return_value = {"p:1": "prefill-0"}
        ops.start_stream.return_value = stream
        return ops, response, stream

    def test_batch_ack_does_not_consume_fetch_until_drain(self):
        ops, response, stream = self.setup_ops()
        fired, handles = [], {}
        self.assertEqual(fire_request(ops, 42, fired, handles), ("prefill-0", None))
        ops.start_stream.assert_not_called()
        self.assertEqual(fired, [(42, response)])
        self.assertEqual(
            drain_fired(ops, fired, handles), [(42, "prefill-0", True, None)]
        )
        ops.start_stream.assert_called_once_with(response, 42)
        ops.cancel.assert_not_called()

    def test_nonbatch_starts_once_with_original_input_and_reuses_stream(self):
        ops, response, stream = self.setup_ops(batch=False)
        fired, handles = [], {}
        fire_request(ops, 7, fired, handles, input_len=512, block_keys=[11])
        ops.build_generate_input.assert_called_once_with(
            7, input_len=512, block_keys=[11]
        )
        ops.start_stream.assert_called_once_with(
            response, 7, input_pb=ops.build_generate_input.return_value
        )
        self.assertIs(handles[7], stream)
        drain_fired(ops, fired, handles, wait_s=12)
        self.assertEqual(ops.start_stream.call_count, 1)
        stream.wait_end.assert_called_once_with(12)

    def test_reject_does_not_create_stream_or_tracked_request(self):
        ops, response, _ = self.setup_ops()
        response.success = False
        response.error_message = "queue full"
        fired, handles = [], {}
        self.assertEqual(
            fire_request(ops, 8, fired, handles), (None, "schedule failed: queue full")
        )
        self.assertEqual(fired, [])
        ops.start_stream.assert_not_called()

    def test_open_failure_keeps_scheduled_request_available_to_cleanup(self):
        ops, response, _ = self.setup_ops(batch=False)
        ops.start_stream.side_effect = RuntimeError("transport")
        fired, handles = [], {}
        name, error = fire_request(ops, 9, fired, handles)
        self.assertEqual(name, "prefill-0")
        self.assertIn("direct stream failed", error)
        self.assertEqual(fired, [(9, response)])
        result = drain_fired(ops, fired, handles)
        self.assertFalse(result[0][2])
        self.assertIn("transport", result[0][3])
        ops.cancel.assert_called_once_with(9, response)

    def test_in_band_error_is_failure_even_when_stream_ended(self):
        ops, response, stream = self.setup_ops()
        stream.snap.error = "8211 lack_mem"
        result = drain_fired(ops, [(10, response)], {})
        self.assertEqual(result, [(10, "prefill-0", False, "8211 lack_mem")])
        ops.cancel.assert_called_once_with(10, response)

    def test_timeout_cancels_and_cleanup_failure_does_not_hide_next_request(self):
        ops, response, stream = self.setup_ops()
        stream.wait_end.return_value = False
        ops.cancel.side_effect = RuntimeError("cancel unavailable")
        result = drain_fired(ops, [(10, response), (11, response)], {}, wait_s=3)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(not row[2] for row in result))
        self.assertEqual(ops.cancel.call_count, 2)
        self.assertEqual(stream.wait_end.call_args_list[0].args, (3,))


if __name__ == "__main__":
    unittest.main()

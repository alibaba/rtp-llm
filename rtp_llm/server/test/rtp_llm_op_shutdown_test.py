import unittest
from unittest.mock import Mock

from rtp_llm import ops


class RtpLLMOpShutdownTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ops.ensure_engine_ops_loaded()

    def test_cpp_stop_runs_shutdown_callback_once(self):
        op = ops.RtpLLMOp()
        callback = Mock()
        op.stop(callback)
        op.stop(callback)
        callback.assert_called_once()
        self.assertGreater(callback.call_args.args[0], 0)

    def test_cpp_stop_propagates_shutdown_callback_failure(self):
        op = ops.RtpLLMOp()
        callback = Mock(side_effect=RuntimeError("shutdown callback failed"))
        with self.assertRaisesRegex(RuntimeError, "shutdown callback failed"):
            op.stop(callback)
        callback.assert_called_once()
        self.assertGreater(callback.call_args.args[0], 0)
        op.stop()


if __name__ == "__main__":
    unittest.main()

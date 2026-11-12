import unittest
from contextlib import ExitStack
from unittest.mock import Mock, mock_open, patch

from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class MagaServerManagerStartTest(unittest.TestCase):
    def setUp(self):
        mocks = ExitStack()
        self.addCleanup(mocks.close)
        mocks.enter_context(
            patch.dict(
                "os.environ",
                {
                    "HOME": "/mock/home",
                    "TEST_UNDECLARED_OUTPUTS_DIR": "/mock/output",
                    "CUDA_VISIBLE_DEVICES": "0",
                },
                clear=True,
            )
        )
        mocks.enter_context(patch("rtp_llm.test.utils.maga_server_manager.os.makedirs"))
        self.log_open = mocks.enter_context(patch("builtins.open", mock_open()))
        mocks.enter_context(patch("resource.setrlimit"))
        self.process = Mock(pid=4242)
        self.popen = mocks.enter_context(
            patch(
                "rtp_llm.test.utils.maga_server_manager.subprocess.Popen",
                return_value=self.process,
            )
        )
        self.health_check = mocks.enter_context(
            patch("rtp_llm.utils.util.wait_sever_done", return_value=True)
        )
        self.manager = MagaServerManager(
            port="18088", device_ids=[2, 3], health_check_path="/ready"
        )

    def tearDown(self):
        with self.manager._state_lock:
            self.manager._server_process = None
        self.manager.stop_server()

    def test_start_without_readiness_does_not_probe(self):
        self.assertTrue(self.manager.start_server(wait_ready=False))

        self.popen.assert_called_once()
        self.health_check.assert_not_called()
        self.process.poll.assert_not_called()
        self.assertEqual(self.manager.server_pid, 4242)
        self.log_open.assert_called_once_with("/mock/output/main_logs/process.log", "w")
        self.assertEqual(
            self.popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2,3"
        )
        self.assertIs(self.popen.call_args.kwargs["stdout"], self.log_open.return_value)
        self.assertIs(self.popen.call_args.kwargs["stderr"], self.log_open.return_value)

    def test_start_defaults_to_readiness(self):
        self.assertTrue(self.manager.start_server(timeout=17))

        self.health_check.assert_called_once_with(self.process, 18088, 17, "/ready")
        self.assertEqual(
            self.popen.call_args.args[0],
            ["/opt/conda310/bin/python", "-m", "rtp_llm.start_server"],
        )

    def test_start_forwards_custom_module(self):
        self.manager._smoke_args_str = '--checkpoint "/mock/model path"'
        self.assertTrue(
            self.manager.start_server(wait_ready=False, server_module="custom.server")
        )

        self.assertEqual(
            self.popen.call_args.args[0],
            [
                "/opt/conda310/bin/python",
                "-m",
                "custom.server",
                "--checkpoint",
                "/mock/model path",
            ],
        )
        self.health_check.assert_not_called()

    def test_stop_before_launch_still_returns_false(self):
        self.assertTrue(self.manager.stop_server())
        parent = Mock()
        parent.children.return_value = []
        with (
            patch(
                "rtp_llm.test.utils.maga_server_manager.psutil.Process",
                return_value=parent,
            ) as get_parent,
            patch(
                "rtp_llm.test.utils.maga_server_manager.psutil.wait_procs",
                return_value=([], []),
            ),
        ):
            self.assertFalse(self.manager.start_server(wait_ready=False))

        self.popen.assert_called_once()
        self.health_check.assert_not_called()
        get_parent.assert_called_once_with(4242)
        parent.terminate.assert_called_once_with()
        self.assertIsNone(self.manager.server_pid)
        self.log_open.return_value.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

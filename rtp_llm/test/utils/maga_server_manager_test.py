import os
import unittest
from unittest.mock import Mock, patch

from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class MagaServerManagerTest(unittest.TestCase):
    @patch.object(MagaServerManager, "wait_sever_done", return_value=True)
    @patch("rtp_llm.test.utils.maga_server_manager.subprocess.Popen")
    def test_server_env_uses_absolute_overrides(self, popen, wait_done):
        process = Mock(pid=1234)
        popen.return_value = process
        manager = MagaServerManager(
            env_args={
                "TEST_UNDECLARED_OUTPUTS_DIR": "relative-output",
                "MAGA_SERVER_WORK_DIR": "relative-work",
            },
            port="12345",
        )
        # Guarantee cleanup even if an assertion below fails: otherwise __del__ ->
        # stop_server() runs with the Mock's pid=1234 and psutil.Process(1234)
        # could terminate a real process on the host.
        self.addCleanup(setattr, manager, "_server_process", None)

        self.assertTrue(manager.start_server(model_path="/model", log_to_file=False))

        popen_kwargs = popen.call_args.kwargs
        self.assertEqual(
            popen_kwargs["env"]["TEST_UNDECLARED_OUTPUTS_DIR"],
            os.path.abspath("relative-output"),
        )
        self.assertEqual(
            popen_kwargs["env"]["MAGA_SERVER_WORK_DIR"],
            os.path.abspath("relative-work"),
        )
        self.assertEqual(popen_kwargs["cwd"], os.path.abspath("relative-work"))
        wait_done.assert_called_once()


if __name__ == "__main__":
    unittest.main()

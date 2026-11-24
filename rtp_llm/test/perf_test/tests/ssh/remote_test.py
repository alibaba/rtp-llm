"""Performance framework behavior and failure regressions."""

import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

from rtp_llm.test.perf_test.ssh.remote import RemoteHost


class RemoteTest(unittest.TestCase):

    def test_ssh_uses_config_and_quotes_remote_arguments_as_data(self):
        host = RemoteHost("jump-alias", python_bin=sys.executable)
        argv = host.command(["printf", "%s", "path with space;$(echo unsafe)'quote"])
        self.assertNotIn("-F", argv)
        remote = argv[-1]
        result = subprocess.run(
            ["bash", "-c", remote], capture_output=True, text=True, check=True
        )
        self.assertEqual(result.stdout, "path with space;$(echo unsafe)'quote")
        self.assertIn("jump-alias", argv)

    def test_option_like_host_is_rejected(self):
        with self.assertRaises(ValueError):
            RemoteHost("-oProxyCommand=anything")

    def test_connection_overrides_are_explicit(self):
        host = RemoteHost("alias", user="tester", port=2222)
        self.assertIn("2222", host.command(["true"]))
        self.assertIn("tester@alias", host.command(["true"]))

    def test_cancel_pending_aborts_and_joins_active_transport_before_reuse(self):
        host = RemoteHost("alias")
        ready = threading.Event()
        errors = []
        original = subprocess.Popen

        def spawn(*args, **kwargs):
            process = original(*args, **kwargs)
            ready.set()
            return process

        def run():
            try:
                host._run([sys.executable, "-c", "import signal;signal.pause()"])
            except RuntimeError as error:
                errors.append(str(error))

        thread = threading.Thread(target=run)
        with patch(
            "rtp_llm.test.perf_test.ssh.remote.subprocess.Popen", side_effect=spawn
        ):
            thread.start()
            self.addCleanup(host.cancel_pending)
            self.assertTrue(ready.wait(5))
            host.cancel_pending()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertTrue(errors)
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            host._run([sys.executable, "-c", "pass"])
        host.reset_cancelled()
        self.assertEqual(host._run([sys.executable, "-c", "print(42)"]).strip(), "42")


if __name__ == "__main__":
    unittest.main()

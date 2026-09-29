"""CPU regressions for raced child exit and process-drain ordering."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

# Bypass application registration; neither config constants nor port allocation
# participates in stop_server. Import and execute the actual manager module.
ROOT = Path(__file__).resolve().parents[3]
for suffix in ("", ".config", ".test", ".test.utils"):
    name = "rtp_llm" + suffix
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
    sys.modules.setdefault(name, package)
config = types.ModuleType("rtp_llm.config.py_config_modules")
config.MIN_WORKER_INFO_PORT_NUM = 10000
sys.modules.setdefault(config.__name__, config)
ports = types.ModuleType("rtp_llm.test.utils.port_util")
ports.PortManager = mock.Mock(
    side_effect=AssertionError("test must not allocate ports")
)
sys.modules.setdefault(ports.__name__, ports)
SPEC = importlib.util.spec_from_file_location(
    "manager_under_test", Path(__file__).with_name("maga_server_manager.py")
)
manager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manager)


class ServerStopTest(unittest.TestCase):
    def make_owner(self):
        owner = manager.MagaServerManager(port="12345")
        owner._server_process = types.SimpleNamespace(pid=100)
        return owner

    def test_disappeared_child_does_not_abandon_live_sibling(self):
        owner = self.make_owner()
        gone, live, parent = mock.Mock(pid=101), mock.Mock(pid=102), mock.Mock(pid=100)
        gone.terminate.side_effect = manager.psutil.NoSuchProcess(101)
        parent.children.return_value = [gone, live]
        with mock.patch.object(
            manager.psutil, "Process", return_value=parent
        ), mock.patch.object(
            manager.psutil, "wait_procs", side_effect=[([gone, live], []), ([], [])]
        ):
            self.assertTrue(owner.stop_server())
        live.terminate.assert_called_once()
        parent.terminate.assert_called_once()
        parent.wait.assert_called_once_with(timeout=10)
        self.assertIsNone(owner._server_process)

    def test_killed_children_are_waited_before_parent_exit(self):
        owner = self.make_owner()
        child, parent = mock.Mock(pid=101), mock.Mock(pid=100)
        parent.children.return_value = [child]
        events = []
        child.kill.side_effect = lambda: events.append("kill")
        parent.terminate.side_effect = lambda: events.append("parent_term")

        def wait_procs(processes, timeout):
            events.append("wait")
            return ([], [child]) if len(events) == 1 else ([child], [])

        with mock.patch.object(
            manager.psutil, "Process", return_value=parent
        ), mock.patch.object(manager.psutil, "wait_procs", side_effect=wait_procs):
            self.assertTrue(owner.stop_server())
        self.assertEqual(events, ["wait", "kill", "wait", "parent_term"])

    def test_child_disappearing_before_kill_is_benign(self):
        owner = self.make_owner()
        child, parent = mock.Mock(pid=101), mock.Mock(pid=100)
        parent.children.return_value = [child]
        child.kill.side_effect = manager.psutil.NoSuchProcess(101)
        with mock.patch.object(
            manager.psutil, "Process", return_value=parent
        ), mock.patch.object(
            manager.psutil, "wait_procs", side_effect=[([], [child]), ([child], [])]
        ):
            self.assertTrue(owner.stop_server())
        parent.terminate.assert_called_once()

    def test_surviving_child_is_reported_as_failure(self):
        owner = self.make_owner()
        child, parent = mock.Mock(pid=101), mock.Mock(pid=100)
        parent.children.return_value = [child]
        with mock.patch.object(
            manager.psutil, "Process", return_value=parent
        ), mock.patch.object(manager.psutil, "wait_procs", return_value=([], [child])):
            self.assertFalse(owner.stop_server())
        parent.terminate.assert_called_once()

    def test_permission_error_is_not_reported_as_success(self):
        owner = self.make_owner()
        with mock.patch.object(
            manager.psutil, "Process", side_effect=manager.psutil.AccessDenied(100)
        ):
            self.assertFalse(owner.stop_server())

    def test_parent_already_gone_is_idempotent(self):
        owner = self.make_owner()
        with mock.patch.object(
            manager.psutil, "Process", side_effect=manager.psutil.NoSuchProcess(100)
        ):
            self.assertTrue(owner.stop_server())
        self.assertTrue(owner.stop_server())


if __name__ == "__main__":
    unittest.main()

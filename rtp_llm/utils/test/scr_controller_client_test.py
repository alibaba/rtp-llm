"""Unit tests for the out-of-process sCR controller client."""

from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from rtp_llm.utils.scr_controller_client import (
    ControllerPhase,
    ScrControllerBusy,
    ScrControllerClient,
    ScrControllerCommandError,
    ScrControllerCoordinator,
    ScrControllerNotReady,
)


def _completed(stdout: str = "", returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


class ScrControllerClientTest(unittest.TestCase):
    def test_check_builds_argv_and_strictly_parses_json(self) -> None:
        client = ScrControllerClient("/opt/controller", "/run/scr/socket", sid="17")
        with mock.patch(
            "subprocess.run",
            return_value=_completed(
                "Release Version: 1.6.0\n{\"errno\":0,\"checkpoint_ready\":true}\n"
            ),
        ) as run:
            result = client.check(timeout=12)

        run.assert_called_once_with(
            ["/opt/controller", "--uds", "/run/scr/socket", "--sid", "17", "check"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=12.0,
        )
        self.assertEqual(result.errno, 0)
        self.assertTrue(result.checkpoint_ready)
        self.assertEqual(result.argv[-1], "check")
        self.assertEqual(client.history, [result])

    def test_check_rejects_not_ready_and_nonzero_errno(self) -> None:
        client = ScrControllerClient()
        for payload in (
            '{"errno":0,"checkpoint_ready":false}',
            '{"errno":9,"checkpoint_ready":true}',
            '{"checkpoint_ready":true}',
        ):
            with self.subTest(payload=payload), mock.patch(
                "subprocess.run", return_value=_completed(payload)
            ):
                with self.assertRaises(ScrControllerNotReady):
                    client.check()

    def test_invalid_json_and_nonzero_rc_are_errors_with_result(self) -> None:
        client = ScrControllerClient()
        with mock.patch("subprocess.run", return_value=_completed("not-json")):
            with self.assertRaises(ScrControllerCommandError) as raised:
                client.check()
        self.assertIsNotNone(raised.exception.result)
        self.assertEqual(raised.exception.result.returncode, 0)

        with mock.patch(
            "subprocess.run",
            return_value=_completed('{"errno":0,"checkpoint_ready":true}', 3, "failed"),
        ):
            with self.assertRaises(ScrControllerCommandError) as raised:
                client.check()
        self.assertEqual(raised.exception.result.returncode, 3)

    def test_timeout_is_recorded_without_running_real_controller(self) -> None:
        client = ScrControllerClient(default_timeout=4)
        timeout = subprocess.TimeoutExpired(["scr_controller"], 4, output=b"partial")
        with mock.patch("subprocess.run", side_effect=timeout) as run:
            with self.assertRaises(ScrControllerCommandError):
                client.health()
        run.assert_called_once()
        self.assertEqual(client.history[-1].returncode, -1)
        self.assertIn("health", client.history[-1].argv)

    def test_command_arguments_for_dump_restore_and_wait(self) -> None:
        client = ScrControllerClient("ctl", "/uds")
        with mock.patch(
            "subprocess.run",
            side_effect=[
                _completed(),
                _completed(),
                _completed(),
                _completed(),
            ],
        ) as run:
            client.dump(
                "/template",
                bypass_cr_path="/cr",
                block_timeout_ms=1200,
                bypass_direct_io=True,
                cache_fs_speedup=True,
            )
            client.wait_cr_done(timeout_seconds=33)
            client.prepare_restore("/cr", cache_fs_speedup=True, timeout_seconds=8)
            client.restore("/template", bypass_cr_path="/cr", cache_fs_speedup=True)

        calls = [call.args[0] for call in run.call_args_list]
        self.assertEqual(
            calls[0],
            [
                "ctl",
                "--uds",
                "/uds",
                "dump",
                "--path",
                "/template",
                "--bypass-cr-path",
                "/cr",
                "--block-timeout-ms",
                "1200",
                "--bypass-direct-io",
                "--cache-fs-speedup",
            ],
        )
        self.assertEqual(calls[1][-3:], ["wait-cr-done", "--timeout", "33"])
        self.assertEqual(
            calls[2][-5:],
            ["--bypass-cr-path", "/cr", "--cache-fs-speedup", "--timeout", "8"],
        )
        self.assertEqual(
            calls[3][-5:],
            ["--path", "/template", "--bypass-cr-path", "/cr", "--cache-fs-speedup"],
        )
        self.assertEqual(run.call_args_list[1].kwargs["timeout"], 63.0)
        self.assertEqual(run.call_args_list[2].kwargs["timeout"], 38.0)

    def test_steady_state_requires_checkpoint_ready(self) -> None:
        client = ScrControllerClient()
        with mock.patch("subprocess.run", return_value=_completed('{"checkpoint_ready":true}')):
            self.assertTrue(client.check_steady_state().checkpoint_ready)
        with mock.patch("subprocess.run", return_value=_completed('{"checkpoint_ready":false}')):
            with self.assertRaises(ScrControllerNotReady):
                client.check_steady_state()
        with mock.patch(
            "subprocess.run", return_value=_completed('{"errno":3,"checkpoint_ready":true}')
        ):
            with self.assertRaises(ScrControllerNotReady):
                client.check_steady_state()


class ScrControllerCoordinatorTest(unittest.TestCase):
    def setUp(self) -> None:
        # Avoid a leaked lease from a failed test.
        ScrControllerCoordinator._leases.clear()

    def tearDown(self) -> None:
        ScrControllerCoordinator._leases.clear()

    def test_generation_is_single_owner_and_commands_are_idempotent(self) -> None:
        client = mock.Mock(spec=ScrControllerClient)
        client.controller = "ctl"
        client.uds = "/uds"
        client.check.return_value = _completed('{"errno":0,"checkpoint_ready":true}')
        client.block.return_value = _completed()
        client.dump.return_value = _completed()
        client.wait_cr_done.return_value = _completed()
        first = ScrControllerCoordinator(client, "generation-1", coordinator_id="one")
        with self.assertRaises(ScrControllerBusy):
            ScrControllerCoordinator(client, "generation-1", coordinator_id="two")
        first.check_ready()
        first.check_ready()
        first.block()
        first.dump("/template")
        first.dump("/template")
        first.wait_cr_done()
        self.assertEqual(client.check.call_count, 1)
        self.assertEqual(client.dump.call_count, 1)
        self.assertEqual(first.phase, ControllerPhase.WAIT_CR_DONE)
        first.close()

    def test_abort_fallback_is_explicit_and_idempotent(self) -> None:
        client = mock.Mock(spec=ScrControllerClient)
        client.controller = "ctl"
        client.uds = "/uds"
        client.fallback.return_value = _completed()
        coordinator = ScrControllerCoordinator(client, "generation-2")
        result = coordinator.abort("dump failed")
        self.assertEqual(coordinator.phase, ControllerPhase.FALLBACK)
        self.assertEqual(result, coordinator.abort("same failure"))
        client.fallback.assert_called_once_with(timeout=None)
        self.assertEqual(coordinator.failure_reason, "dump failed")
        coordinator.close()

    def test_state_machine_does_not_invoke_commands_implicitly(self) -> None:
        client = mock.Mock(spec=ScrControllerClient)
        client.controller = "ctl"
        client.uds = "/uds"
        coordinator = ScrControllerCoordinator(client, "generation-3")
        self.assertEqual(coordinator.phase, ControllerPhase.INIT)
        client.check.assert_not_called()
        client.dump.assert_not_called()
        coordinator.close()


if __name__ == "__main__":
    unittest.main()

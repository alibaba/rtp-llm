from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from rtp_llm.utils.kvcm_subscriber_launcher import (
    _log_optional_failure,
    _stop_subscriber_process,
    _supervise_kvcm_subscriber,
    build_kvcm_subscriber_command,
    is_kvcm_subscriber_required,
    start_kvcm_subscriber,
)


class _FakeStopEvent:
    def __init__(self, wait_results: list[bool]) -> None:
        self._is_set = False
        self._wait_results = list(wait_results)

    def is_set(self) -> bool:
        return self._is_set

    def set(self) -> None:
        self._is_set = True

    def wait(self, _timeout: float) -> bool:
        if self._is_set:
            return True
        result = self._wait_results.pop(0)
        if result:
            self._is_set = True
        return result


class _ServerConfig:
    def __init__(self, start_port: int = 8088, ip: str = "10.0.0.8") -> None:
        self.start_port = start_port
        self.rank_id = 0
        self.worker_info_port_num = 8
        self.ip = ip

    @property
    def server_port(self) -> int:
        return self.start_port + self.rank_id * self.worker_info_port_num

    @property
    def rpc_server_port(self) -> int:
        return self.server_port + 1


def _configs(
    *,
    start_port: int = 8088,
    world_rank: int = 0,
    dp_size: int = 1,
    ip: str = "10.0.0.8",
):
    return SimpleNamespace(
        server_config=_ServerConfig(start_port, ip),
        parallelism_config=SimpleNamespace(
            world_rank=world_rank,
            dp_size=dp_size,
            tp_size=2,
            pp_size=1,
        ),
        kv_cache_config=SimpleNamespace(seq_size_per_block=64),
        model_args=SimpleNamespace(
            ckpt_path="/models/Qwen2-0.5B",
            model_type="qwen_2",
            act_type="BF16",
        ),
    )


class KvcmSubscriberLauncherTest(unittest.TestCase):
    def test_required_mode_parses_common_boolean_values(self) -> None:
        for value in ("1", "true", "TRUE", "yes", "on"):
            with self.subTest(value=value), patch.dict(
                os.environ,
                {"KVCM_SUBSCRIBER_REQUIRED": value},
                clear=True,
            ):
                self.assertTrue(is_kvcm_subscriber_required())

        for value in ("", "0", "false", "FALSE", "no", "off"):
            with self.subTest(value=value), patch.dict(
                os.environ,
                {"KVCM_SUBSCRIBER_REQUIRED": value},
                clear=True,
            ):
                self.assertFalse(is_kvcm_subscriber_required())

    def test_invalid_required_value_defaults_to_optional(self) -> None:
        with patch.dict(
            os.environ,
            {"KVCM_SUBSCRIBER_REQUIRED": "invalid"},
            clear=True,
        ), patch("logging.warning") as warning:
            self.assertFalse(is_kvcm_subscriber_required())
            warning.assert_called_once()

    def test_absent_config_disables_subscriber(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(build_kvcm_subscriber_command(_configs()))

    def test_resolves_rank_zero_grpc_and_http_ports_from_start_port(self) -> None:
        with tempfile.NamedTemporaryFile() as config:
            env = {
                "KVCM_SUBSCRIBER_CONFIG": config.name,
                "KVCM_SUBSCRIBER_COMMAND": "python -m subscriber",
            }
            with patch.dict(os.environ, env, clear=True):
                command = build_kvcm_subscriber_command(
                    _configs(start_port=18088)
                )

        self.assertEqual(
            command,
            (
                "python",
                "-m",
                "subscriber",
                "--config",
                config.name,
                "--engine-type",
                "rtp_llm",
                "--rtp-endpoints",
                "127.0.0.1:18089",
                "--host-ip-port",
                "10.0.0.8:18088",
                "--block-size",
                "64",
                "--model-name",
                "Qwen2-0.5B",
                "--model-dtype",
                "bfloat16",
                "--tensor-parallel-size",
                "2",
                "--data-parallel-size",
                "1",
                "--pipeline-parallel-size",
                "1",
            ),
        )

    def test_explicit_dp_endpoints_override_auto_discovery(self) -> None:
        with tempfile.NamedTemporaryFile() as config:
            env = {
                "KVCM_SUBSCRIBER_CONFIG": config.name,
                "RTP_LLM_CACHE_SUBSCRIBER_ENDPOINTS": (
                    "10.0.0.1:8089,10.0.0.2:8089"
                ),
                "KVCM_HOST_IP_PORT": "service.example:80",
            }
            with patch.dict(os.environ, env, clear=True):
                command = build_kvcm_subscriber_command(
                    _configs(dp_size=2)
                )

        self.assertIsNotNone(command)
        assert command is not None
        self.assertEqual(
            command[command.index("--rtp-endpoints") + 1],
            "10.0.0.1:8089,10.0.0.2:8089",
        )
        self.assertEqual(
            command[command.index("--host-ip-port") + 1],
            "service.example:80",
        )

    def test_multi_dp_requires_explicit_endpoints(self) -> None:
        with tempfile.NamedTemporaryFile() as config:
            with patch.dict(
                os.environ,
                {"KVCM_SUBSCRIBER_CONFIG": config.name},
                clear=True,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "RTP_LLM_CACHE_SUBSCRIBER_ENDPOINTS is required",
                ):
                    build_kvcm_subscriber_command(_configs(dp_size=2))

    def test_multi_dp_requires_exactly_one_unique_endpoint_per_dp_rank(self) -> None:
        with tempfile.NamedTemporaryFile() as config:
            env = {
                "KVCM_SUBSCRIBER_CONFIG": config.name,
                "RTP_LLM_CACHE_SUBSCRIBER_ENDPOINTS": "rank-0:8089",
            }
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(ValueError, "expected 2, got 1"):
                    build_kvcm_subscriber_command(_configs(dp_size=2))

            env["RTP_LLM_CACHE_SUBSCRIBER_ENDPOINTS"] = (
                "rank-0:8089,rank-0:8089"
            )
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(ValueError, "duplicate endpoints"):
                    build_kvcm_subscriber_command(_configs(dp_size=2))

    def test_only_configured_world_rank_launches(self) -> None:
        with tempfile.NamedTemporaryFile() as config:
            with patch.dict(
                os.environ,
                {"KVCM_SUBSCRIBER_CONFIG": config.name},
                clear=True,
            ):
                self.assertIsNone(
                    build_kvcm_subscriber_command(_configs(world_rank=4))
                )

    def test_non_owner_rank_does_not_require_local_config_file(self) -> None:
        with patch.dict(
            os.environ,
            {"KVCM_SUBSCRIBER_CONFIG": "/not-mounted/on/non-owner.yaml"},
            clear=True,
        ):
            self.assertIsNone(
                build_kvcm_subscriber_command(_configs(world_rank=4))
            )

    def test_missing_config_file_fails_fast(self) -> None:
        with patch.dict(
            os.environ,
            {"KVCM_SUBSCRIBER_CONFIG": "/missing/subscriber.yaml"},
            clear=True,
        ):
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                build_kvcm_subscriber_command(_configs())

    def test_optional_start_failure_does_not_escape(self) -> None:
        with patch.dict(
            os.environ,
            {"KVCM_SUBSCRIBER_CONFIG": "/missing/subscriber.yaml"},
            clear=True,
        ), patch("logging.exception") as exception_log:
            self.assertIsNone(start_kvcm_subscriber(_configs()))
            exception_log.assert_called_once()

    def test_required_start_failure_escapes(self) -> None:
        with patch.dict(
            os.environ,
            {"KVCM_SUBSCRIBER_CONFIG": "/missing/subscriber.yaml"},
            clear=True,
        ):
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                start_kvcm_subscriber(_configs(), required=True)

    def test_optional_process_start_failure_does_not_escape(self) -> None:
        with tempfile.NamedTemporaryFile() as config:
            with patch.dict(
                os.environ,
                {"KVCM_SUBSCRIBER_CONFIG": config.name},
                clear=True,
            ), patch(
                "rtp_llm.utils.kvcm_subscriber_launcher.multiprocessing.Process"
            ) as process_cls, patch("logging.exception") as exception_log:
                process_cls.return_value.start.side_effect = OSError("spawn failed")

                self.assertIsNone(start_kvcm_subscriber(_configs()))
                exception_log.assert_called_once()

    def test_required_process_start_failure_escapes(self) -> None:
        with tempfile.NamedTemporaryFile() as config:
            with patch.dict(
                os.environ,
                {"KVCM_SUBSCRIBER_CONFIG": config.name},
                clear=True,
            ), patch(
                "rtp_llm.utils.kvcm_subscriber_launcher.multiprocessing.Process"
            ) as process_cls:
                process_cls.return_value.start.side_effect = OSError("spawn failed")

                with self.assertRaisesRegex(OSError, "spawn failed"):
                    start_kvcm_subscriber(_configs(), required=True)

    def test_start_registers_supervisor_with_expected_command(self) -> None:
        with tempfile.NamedTemporaryFile() as config:
            with patch.dict(
                os.environ,
                {"KVCM_SUBSCRIBER_CONFIG": config.name},
                clear=True,
            ), patch(
                "rtp_llm.utils.kvcm_subscriber_launcher.multiprocessing.Process"
            ) as process_cls:
                process = MagicMock()
                process_cls.return_value = process

                result = start_kvcm_subscriber(_configs())

        self.assertIs(result, process)
        process.start.assert_called_once_with()
        kwargs = process_cls.call_args.kwargs
        self.assertEqual(kwargs["name"], "kvcm_subscriber_supervisor")
        self.assertTrue(kwargs["daemon"])
        self.assertIs(kwargs["target"], _supervise_kvcm_subscriber)
        self.assertFalse(kwargs["args"][1])
        self.assertEqual(
            kwargs["args"][0][kwargs["args"][0].index("--rtp-endpoints") + 1],
            "127.0.0.1:8089",
        )

    def test_optional_supervisor_restarts_exited_subscriber(self) -> None:
        first = MagicMock(pid=101)
        first.poll.return_value = 7
        second = MagicMock(pid=102)
        second.poll.return_value = 8
        stop_event = _FakeStopEvent([False, False, False, True])

        with patch("signal.signal"), patch(
            "rtp_llm.utils.kvcm_subscriber_launcher.subprocess.Popen",
            side_effect=[first, second],
        ) as popen, patch("logging.warning") as warning:
            _supervise_kvcm_subscriber(
                ("subscriber", "--config", "/tmp/config.yaml"),
                False,
                0.01,
                stop_event=stop_event,
            )

        self.assertEqual(popen.call_count, 2)
        warning.assert_called_once()
        self.assertIn("exited with code 7", str(warning.call_args))

    def test_optional_supervisor_retries_exec_failure(self) -> None:
        stop_event = _FakeStopEvent([True])

        with patch("signal.signal"), patch(
            "rtp_llm.utils.kvcm_subscriber_launcher.subprocess.Popen",
            side_effect=FileNotFoundError("subscriber missing"),
        ), patch("logging.warning") as warning:
            _supervise_kvcm_subscriber(
                ("missing-subscriber",),
                False,
                0.01,
                stop_event=stop_event,
            )

        warning.assert_called_once()
        self.assertIn("FileNotFoundError", str(warning.call_args))
        self.assertIsInstance(warning.call_args.kwargs["exc_info"], tuple)

    def test_optional_restart_warning_is_throttled(self) -> None:
        with patch("logging.warning") as warning:
            for failure_count in range(1, 13):
                _log_optional_failure(failure_count, "exited", 5.0)

        self.assertEqual(warning.call_count, 2)

    def test_required_supervisor_propagates_subscriber_exit(self) -> None:
        child = MagicMock(pid=103)
        child.poll.return_value = 9
        stop_event = _FakeStopEvent([False])

        with patch("signal.signal"), patch(
            "rtp_llm.utils.kvcm_subscriber_launcher.subprocess.Popen",
            return_value=child,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "required KVCM Subscriber exited with code 9",
            ):
                _supervise_kvcm_subscriber(
                    ("subscriber",),
                    True,
                    0.01,
                    stop_event=stop_event,
                )

    def test_supervisor_forwards_shutdown_to_subscriber_group(self) -> None:
        child = MagicMock(pid=104)
        child.poll.return_value = None
        stop_event = _FakeStopEvent([True])

        with patch("signal.signal"), patch(
            "rtp_llm.utils.kvcm_subscriber_launcher.subprocess.Popen",
            return_value=child,
        ), patch("os.killpg") as killpg:
            _supervise_kvcm_subscriber(
                ("subscriber",),
                False,
                0.01,
                stop_event=stop_event,
            )

        killpg.assert_called_once_with(104, signal.SIGTERM)
        child.wait.assert_called_once()

    def test_shutdown_force_kills_subscriber_after_grace_timeout(self) -> None:
        child = MagicMock(pid=105)
        child.poll.return_value = None
        child.wait.side_effect = [
            subprocess.TimeoutExpired("subscriber", 10),
            None,
        ]

        with patch("os.killpg") as killpg:
            _stop_subscriber_process(child)

        self.assertEqual(
            killpg.call_args_list,
            [
                call(105, signal.SIGTERM),
                call(105, signal.SIGKILL),
            ],
        )


if __name__ == "__main__":
    unittest.main()

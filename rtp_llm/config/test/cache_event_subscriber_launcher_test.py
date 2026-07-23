import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock

_MODULE_PATH = Path(__file__).parents[2] / "cache_event_subscriber_launcher.py"
_SPEC = importlib.util.spec_from_file_location(
    "cache_event_subscriber_launcher", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
launch_cache_event_subscriber = _MODULE.launch_cache_event_subscriber
subscriber_command = _MODULE.subscriber_command
subscriber_owner_rank = _MODULE.subscriber_owner_rank
subscriber_required = _MODULE.subscriber_required


class CacheEventSubscriberLauncherTest(unittest.TestCase):
    def test_subscriber_command_is_disabled_by_default(self) -> None:
        self.assertEqual((), subscriber_command({}))

    def test_subscriber_command_parses_argv_without_shell(self) -> None:
        env = {
            "KVCM_CACHE_EVENT_SUBSCRIBER_COMMAND": (
                "python -m kv_cache_manager.cache_event_subscriber "
                "--instance-id 'model with spaces'"
            )
        }
        self.assertEqual(
            (
                "python",
                "-m",
                "kv_cache_manager.cache_event_subscriber",
                "--instance-id",
                "model with spaces",
            ),
            subscriber_command(env),
        )

    def test_launch_registers_process_compatible_child(self) -> None:
        process = Mock(pid=123)
        process_factory = Mock(return_value=process)
        result = launch_cache_event_subscriber(
            environ={"KVCM_CACHE_EVENT_SUBSCRIBER_COMMAND": "python -m subscriber"},
            process_factory=process_factory,
        )
        self.assertIs(process, result)
        process.start.assert_called_once_with()
        kwargs = process_factory.call_args.kwargs
        self.assertEqual("cache_event_subscriber", kwargs["name"])
        self.assertEqual(("python", "-m", "subscriber"), kwargs["args"][0])
        self.assertIs(kwargs["target"], _MODULE._supervise_subscriber)

    def test_required_subscriber_is_fail_closed(self) -> None:
        process = Mock(pid=123)
        process_factory = Mock(return_value=process)
        launch_cache_event_subscriber(
            environ={
                "KVCM_CACHE_EVENT_SUBSCRIBER_COMMAND": (
                    "subscriber --endpoint {rtp_endpoint} --rank {world_rank}"
                ),
                "KVCM_CACHE_EVENT_SUBSCRIBER_REQUIRED": "true",
                "KVCM_CACHE_EVENT_SUBSCRIBER_OWNER_RANK": "3",
            },
            world_rank=3,
            default_endpoint="127.0.0.1:19001",
            process_factory=process_factory,
        )
        kwargs = process_factory.call_args.kwargs
        self.assertIs(kwargs["target"], _MODULE._exec_subscriber)
        self.assertEqual(
            ("subscriber", "--endpoint", "127.0.0.1:19001", "--rank", "3"),
            kwargs["args"][0],
        )
        self.assertEqual(
            "127.0.0.1:19001", kwargs["args"][1]["RTP_CACHE_EVENT_ENDPOINT"]
        )

    def test_non_owner_rank_does_not_launch(self) -> None:
        process_factory = Mock()
        result = launch_cache_event_subscriber(
            environ={
                "KVCM_CACHE_EVENT_SUBSCRIBER_COMMAND": "subscriber",
                "KVCM_CACHE_EVENT_SUBSCRIBER_OWNER_RANK": "2",
            },
            world_rank=1,
            process_factory=process_factory,
        )
        self.assertIsNone(result)
        process_factory.assert_not_called()

    def test_launcher_config_validation(self) -> None:
        self.assertEqual(
            2, subscriber_owner_rank({"KVCM_CACHE_EVENT_SUBSCRIBER_OWNER_RANK": "2"})
        )
        self.assertTrue(
            subscriber_required({"KVCM_CACHE_EVENT_SUBSCRIBER_REQUIRED": "yes"})
        )
        with self.assertRaises(ValueError):
            subscriber_required({"KVCM_CACHE_EVENT_SUBSCRIBER_REQUIRED": "sometimes"})

    def test_invalid_quoted_command_fails_fast(self) -> None:
        with self.assertRaises(ValueError):
            subscriber_command({"KVCM_CACHE_EVENT_SUBSCRIBER_COMMAND": "python '"})


if __name__ == "__main__":
    unittest.main()

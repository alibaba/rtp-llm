from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rtp_llm.utils.kvcm_subscriber_launcher import (
    build_kvcm_subscriber_command,
    start_kvcm_subscriber,
)


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

    def test_start_registers_exec_child_with_expected_command(self) -> None:
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
        self.assertEqual(kwargs["name"], "kvcm_subscriber")
        self.assertEqual(
            kwargs["args"][0][kwargs["args"][0].index("--rtp-endpoints") + 1],
            "127.0.0.1:8089",
        )


if __name__ == "__main__":
    unittest.main()

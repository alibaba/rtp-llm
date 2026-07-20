from __future__ import annotations

import json
import sys
from concurrent.futures import TimeoutError as FutureTimeoutError
from types import ModuleType
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

from rtp_llm.dash_sc.server import (
    DashScGrpcServer,
    dash_sc_grpc_server_channel_options,
)


class _FakeDashScGrpcConfig:
    def __init__(self) -> None:
        self._server_config: dict[str, int] = {}

    def from_json(self, json_str: str) -> None:
        self._server_config = json.loads(json_str)["server_config"]

    def get_server_config(self) -> dict[str, int]:
        return self._server_config


class DashScGrpcServerChannelOptionsTest(TestCase):
    def test_default_config_applies_receive_message_limit(self) -> None:
        fake_ops = ModuleType("rtp_llm.ops")
        fake_ops.DashScGrpcConfig = _FakeDashScGrpcConfig

        with patch.dict(sys.modules, {"rtp_llm.ops": fake_ops}):
            options = dict(dash_sc_grpc_server_channel_options(None))

        self.assertEqual(
            options["grpc.max_receive_message_length"],
            1024 * 1024 * 1024,
        )

    def test_start_timeout_cancels_submitted_coroutine(self) -> None:
        future = MagicMock()
        future.result.side_effect = FutureTimeoutError()

        def submit(coro, _loop):
            coro.close()
            return future

        server = DashScGrpcServer(dash_sc_grpc_config=MagicMock())
        with patch(
            "rtp_llm.dash_sc.server.asyncio.run_coroutine_threadsafe",
            side_effect=submit,
        ):
            with self.assertRaises(FutureTimeoutError):
                server.start_on_loop(
                    MagicMock(),
                    port=12345,
                    servicer=MagicMock(),
                    startup_timeout_s=0.01,
                )

        future.cancel.assert_called_once_with()


if __name__ == "__main__":
    main()

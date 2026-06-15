"""Unit tests for ``rtp_llm.dash_sc.app`` helpers (echo_prefix startup derivation)."""

from __future__ import annotations

import asyncio
import os
from unittest import TestCase, main
from unittest.mock import patch

from rtp_llm.dash_sc import app as bg_app
from rtp_llm.dash_sc.app import (
    _create_proxy_servicer_on_loop,
    _derive_echo_prefix_ids,
    _is_proxy_mode_enabled,
    _pre_stop_drain_seconds,
)


class _EnvCfg:
    def __init__(self, think_mode: int = 1, think_start_tag: str = "<think>\n"):
        self.think_mode = think_mode
        self.think_start_tag = think_start_tag


class _FakeTokenizer:
    def __init__(self, *, ids=None, id_map=None, raise_exc: bool = False):
        self._ids = ids or []
        self._id_map = id_map or {}
        self._raise = raise_exc
        self.encode_calls: list[tuple[str, bool]] = []

    def encode(self, text, add_special_tokens=True):
        if self._raise:
            raise RuntimeError("tokenizer.encode failed")
        self.encode_calls.append((text, add_special_tokens))
        if text in self._id_map:
            return list(self._id_map[text])
        return list(self._ids)


class _BaseTok:
    def __init__(self, tok):
        self.tokenizer = tok


class DeriveEchoPrefixIdsTest(TestCase):
    def test_encodes_think_start_tag(self) -> None:
        tok = _FakeTokenizer(ids=[154841])
        ids = _derive_echo_prefix_ids(_EnvCfg(), _BaseTok(tok))
        self.assertEqual(ids, [154841])
        # Must encode without special tokens so only the tag bytes become ids.
        self.assertEqual(tok.encode_calls, [("<think>\n", False)])

    def test_disabled_when_think_mode_off(self) -> None:
        tok = _FakeTokenizer(ids=[154841])
        ids = _derive_echo_prefix_ids(_EnvCfg(think_mode=0), _BaseTok(tok))
        self.assertEqual(ids, [])
        self.assertEqual(tok.encode_calls, [])

    def test_disabled_when_tag_empty(self) -> None:
        tok = _FakeTokenizer(ids=[154841])
        ids = _derive_echo_prefix_ids(_EnvCfg(think_start_tag=""), _BaseTok(tok))
        self.assertEqual(ids, [])
        self.assertEqual(tok.encode_calls, [])

    def test_fail_open_on_tokenizer_error(self) -> None:
        ids = _derive_echo_prefix_ids(
            _EnvCfg(), _BaseTok(_FakeTokenizer(raise_exc=True))
        )
        self.assertEqual(ids, [])


class CreateProxyServicerOnLoopTest(TestCase):
    def test_constructs_inside_running_loop(self) -> None:
        created_loops = []
        sentinel = object()

        def fake_servicer():
            created_loops.append(asyncio.get_running_loop())
            return sentinel

        async def run():
            with patch.object(bg_app, "DashScProxyServicer", side_effect=fake_servicer):
                loop = asyncio.get_running_loop()
                servicer = await _create_proxy_servicer_on_loop()
            return loop, servicer

        loop, servicer = asyncio.run(run())
        self.assertIs(servicer, sentinel)
        self.assertEqual(created_loops, [loop])


class ProxyModeEnvTest(TestCase):
    def test_service_route_alone_does_not_enable_proxy_mode(self) -> None:
        with patch.dict(
            os.environ,
            {"SERVICE_ROUTE": ('{"type": "ip_port_list", "address": "127.0.0.1:1"}')},
            clear=True,
        ):
            self.assertFalse(_is_proxy_mode_enabled())

    def test_proxy_mode_enabled_only_by_one(self) -> None:
        with patch.dict(os.environ, {"DASH_SC_GRPC_PROXY_MODE": "1"}, clear=True):
            self.assertTrue(_is_proxy_mode_enabled())
        with patch.dict(os.environ, {"DASH_SC_GRPC_PROXY_MODE": "true"}, clear=True):
            self.assertFalse(_is_proxy_mode_enabled())

    def test_legacy_forward_addr_still_enables_proxy_mode(self) -> None:
        with patch.dict(
            os.environ,
            {"DASH_SC_GRPC_FORWARD_ADDR": "127.0.0.1:1"},
            clear=True,
        ):
            self.assertTrue(_is_proxy_mode_enabled())


class PreStopDrainSecondsTest(TestCase):
    def test_default_pre_stop_drain(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_pre_stop_drain_seconds(), 120.0)

    def test_env_pre_stop_drain(self) -> None:
        with patch.dict(
            os.environ, {"DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS": "2.5"}, clear=True
        ):
            self.assertEqual(_pre_stop_drain_seconds(), 2.5)

    def test_bad_pre_stop_drain_uses_default(self) -> None:
        with patch.dict(
            os.environ, {"DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS": "bad"}, clear=True
        ):
            self.assertEqual(_pre_stop_drain_seconds(), 120.0)

    def test_effective_pre_stop_drain_clamps_to_shutdown_timeout(self) -> None:
        app = bg_app.DashScApp.__new__(bg_app.DashScApp)

        class _ServerConfig:
            shutdown_timeout = 10

        app.server_config = _ServerConfig()
        with patch.dict(
            os.environ, {"DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS": "30"}, clear=True
        ):
            self.assertEqual(app._effective_pre_stop_drain_seconds(), 10.0)


class CloseServicerOnLoopTest(TestCase):
    def test_closes_servicer_on_enqueue_loop(self) -> None:
        app = bg_app.DashScApp.__new__(bg_app.DashScApp)
        loop = app._start_enqueue_loop()
        closed_loops = []

        class _Closable:
            async def close(self):
                closed_loops.append(asyncio.get_running_loop())

        try:
            app._close_servicer_on_loop(_Closable())
        finally:
            app._stop_enqueue_loop()

        self.assertEqual(closed_loops, [loop])


if __name__ == "__main__":
    main()

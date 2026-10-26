"""Stateless Mock control operations; the execution facade owns its endpoint."""

from typing import Optional

from runtime.network import http_get_json, http_post_json


class MockEngineControl:
    mock_http_port: int

    def snapshot(self) -> dict:
        data = http_get_json(f"http://127.0.0.1:{self.mock_http_port}/snapshot")
        if data is None:
            raise RuntimeError(f"snapshot failed on mock http port {self.mock_http_port}")
        return data

    def _post(self, action, body, **options):
        return http_post_json(f"http://127.0.0.1:{self.mock_http_port}/{action}", body, **options)

    def _engine_command(self, action, engine_name, fields=None):
        return self._post(action, {"engine": engine_name, **(fields or {})})

    def _require_engine_command(self, action, engine_name, **fields):
        status, body = self._engine_command(action, engine_name, fields)
        if status != 200:
            raise RuntimeError(f"{action}({engine_name}) failed: {status} {body}")
        return body or {}

    def inject(self, engine_name: str, config: dict) -> dict:
        return self._require_engine_command("inject", engine_name, mock={"fault": config})

    def clear_inject(self, engine_name: str) -> dict:
        return self._require_engine_command("clear_inject", engine_name)

    def set_perf(self, engine_name: str, **kwargs) -> bool:
        return self._engine_command("set_perf", engine_name, {"mock": kwargs})[0] == 200

    def set_kv_pressure(self, engine_name: str, active_kv_tokens: int) -> bool:
        return self._engine_command("set_kv_pressure", engine_name, {"mock": {"fault": {"active_kv_tokens": active_kv_tokens}}})[0] == 200

    def set_queue_depth(self, engine_name: str, queue_depth: int) -> bool:
        """Set the real enqueue-rejection limit, not a diagnostic display value."""
        return self._engine_command("set_queue_depth", engine_name, {"mock": {"fault": {"queue_depth_limit": queue_depth}}})[0] == 200

    def stop_engine(self, engine_name: str) -> dict:
        return self._require_engine_command("stop_engine", engine_name)

    def start_engine(self, engine_name: str) -> dict:
        return self._require_engine_command("start_engine", engine_name)

    def add_engine(self, role: str, port: Optional[int] = None) -> tuple[int, Optional[dict]]:
        """POST /add_engine {"role": ..., "port": optional} — dynamic scale-out.

        The Java mock's field name is ``port`` (gRPC port; auto-allocated as
        current max + 1 when omitted).  Returns (status, body) WITHOUT raising:
        200 → body carries ``engine`` (name) + ``port`` (gRPC) + ``http_port``;
        409 port-in-use / 400 bad role / 501 (cluster started without
        --mock.discovery_file) are surfaced to the caller (chaos cases exercise
        concurrent add/remove and treat those as expected outcomes).
        """
        body: dict = {"role": role}
        if port is not None:
            body["port"] = port
        return self._post("add_engine", body)

    def remove_engine(
        self, engine_name: Optional[str] = None, port: Optional[int] = None, mode: str = "graceful", drain_timeout_ms: Optional[int] = None
    ) -> tuple[int, Optional[dict]]:
        """POST /remove_engine {"engine": name} or {"port": grpcPort}.

        Default mode is the mock's GRACEFUL scale-in (strip the discovery
        entry first so the master stops routing, then wait bounded for all
        in-flight work to finish, then tear down) — the production rolling
        scale-in order (user ruling 2026-09: a planned scale-in under load
        must not lose or fail any request).  ``mode="abrupt"`` keeps the
        legacy immediate teardown (in-flight streams cut) for chaos-style
        fault cases.

        The graceful call BLOCKS until the drain settles (mock drain cap
        60s by default), so the HTTP timeout sits well above the bound;
        the response carries ``drained`` / ``drain_ms`` alongside the
        ``running_at_removal`` / ``waiting_at_removal`` counters.  Returns
        (status, body) without raising — 404 (unknown engine) is an
        expected outcome under concurrent add/remove racing.
        """
        body: dict = {}
        if engine_name:
            body["engine"] = engine_name
        if port is not None:
            body["port"] = port
        if not body:
            raise ValueError("remove_engine needs engine_name or port")
        body["mode"] = mode
        if drain_timeout_ms is not None:
            body["drain_timeout_ms"] = drain_timeout_ms
        # Graceful cap is 60s + teardown margin on the Java side; keep the
        # client out of the way of a legitimately slow drain.
        timeout = 5.0 if mode == "abrupt" else 95.0
        return self._post("remove_engine", body, timeout=timeout)

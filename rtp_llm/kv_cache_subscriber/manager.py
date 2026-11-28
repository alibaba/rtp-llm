from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import aiohttp

logger = logging.getLogger(__name__)


class KvcmResponseError(RuntimeError):
    def __init__(self, endpoint: str, code: str, message: str) -> None:
        super().__init__(f"KVCM {endpoint} failed: {code} {message}")
        self.code = code


@dataclass(frozen=True)
class _Endpoint:
    host: str
    port: int

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class KvcmHttpManagerClient:
    """Small async KVCM HTTP client with static/Spectrum and leader discovery."""

    def __init__(self, configured_url: str, timeout_s: float) -> None:
        self._configured_url = configured_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._session: aiohttp.ClientSession | None = None
        self._base_url = ""
        self._discovery_lock = asyncio.Lock()

    async def start(self) -> None:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        await self._resolve_service_endpoint()
        await self._discover_leader(required=False)

    async def _resolve_service_endpoint(self) -> None:
        if self._configured_url.startswith(("http://", "https://")):
            self._base_url = self._configured_url
            return
        if self._configured_url.startswith("static://"):
            endpoints = self._parse_static(self._configured_url.removeprefix("static://"))
            self._base_url = random.choice(endpoints).base_url
            return
        if self._configured_url.startswith("spectrum://"):
            self._base_url = (await self._resolve_spectrum()).base_url
            return
        raise ValueError(f"unsupported KVCM URL: {self._configured_url!r}")

    @staticmethod
    def _parse_static(value: str) -> list[_Endpoint]:
        endpoints: list[_Endpoint] = []
        for item in value.split(","):
            host, separator, port_text = item.strip().rpartition(":")
            if not separator or not host or not port_text.isdigit():
                raise ValueError(f"invalid static KVCM endpoint: {item!r}")
            port = int(port_text)
            if not 0 < port < 65536:
                raise ValueError(f"invalid static KVCM port: {port}")
            endpoints.append(_Endpoint(host, port))
        if not endpoints:
            raise ValueError("static KVCM URL contains no endpoints")
        return endpoints

    async def _resolve_spectrum(self) -> _Endpoint:
        if self._session is None:
            raise RuntimeError("KVCM manager client has not been started")
        parsed = urlsplit(self._configured_url)
        virtual_service_id = parsed.hostname or ""
        if not virtual_service_id:
            raise ValueError("Spectrum KVCM URL must contain a virtual service id")
        port_override = parsed.port
        url = (
            "http://127.0.0.1:8880/api/v1/discovery/virtual-services/"
            f"{virtual_service_id}/instances"
        )
        async with self._session.get(url) as response:
            response.raise_for_status()
            payload = await response.json()

        endpoints: list[_Endpoint] = []
        for item in payload.get("instances", []):
            if not isinstance(item, dict):
                continue
            host = item.get("ip")
            port = port_override if port_override is not None else item.get("port")
            if not isinstance(host, str) or not host:
                continue
            try:
                parsed_port = int(port)
            except (TypeError, ValueError):
                continue
            if 0 < parsed_port < 65536:
                endpoints.append(_Endpoint(host, parsed_port))
        if not endpoints:
            raise RuntimeError(
                f"Spectrum returned no KVCM endpoint for {virtual_service_id}"
            )
        return random.choice(endpoints)

    async def _discover_leader(self, *, required: bool) -> bool:
        if self._session is None:
            raise RuntimeError("KVCM manager client has not been started")
        async with self._discovery_lock:
            try:
                async with self._session.post(
                    self._base_url + "/api/getClusterInfo",
                    json={
                        "trace_id": f"rtp_subscriber_leader_{time.monotonic_ns()}",
                        "instance_id": "",
                    },
                ) as response:
                    response.raise_for_status()
                    payload = await response.json()
                code = payload.get("header", {}).get("status", {}).get("code")
                leader = payload.get("leader_endpoint", {})
                if code != "OK" or not leader.get("host") or not leader.get(
                    "meta_http_port"
                ):
                    if required:
                        raise RuntimeError("KVCM leader discovery returned no leader")
                    return False
                self._base_url = (
                    f"http://{leader['host']}:{int(leader['meta_http_port'])}"
                )
                return True
            except Exception:
                if required:
                    raise
                logger.warning("initial KVCM leader discovery failed", exc_info=True)
                return False

    @staticmethod
    def _check_payload(endpoint: str, payload: dict[str, Any]) -> None:
        status = payload.get("header", {}).get("status", {})
        code = str(status.get("code", ""))
        if code != "OK":
            raise KvcmResponseError(endpoint, code, str(status.get("message", "")))
        item_results = payload.get("item_results", [])
        failed = [result for result in item_results if result not in ("OK", 1)]
        if failed:
            raise KvcmResponseError(endpoint, "PARTIAL_FAILURE", repr(failed))

    async def _post(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("KVCM manager client has not been started")
        for attempt in range(2):
            try:
                async with self._session.post(
                    self._base_url + endpoint, json=data
                ) as response:
                    response.raise_for_status()
                    payload: dict[str, Any] = await response.json()
                try:
                    self._check_payload(endpoint, payload)
                except KvcmResponseError as exc:
                    if exc.code == "SERVER_NOT_LEADER" and attempt == 0:
                        await self._discover_leader(required=True)
                        continue
                    raise
                return payload
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
                if attempt == 0 and not self._configured_url.startswith(
                    ("http://", "https://")
                ):
                    await self._resolve_service_endpoint()
                    continue
                raise
        raise RuntimeError(f"KVCM request retry loop exhausted for {endpoint}")

    async def register_instance(self, data: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/api/registerInstance", data)

    async def report_event(self, data: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/api/reportEvent", data)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

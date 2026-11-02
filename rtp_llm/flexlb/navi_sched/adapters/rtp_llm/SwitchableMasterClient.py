import asyncio
import json
import os
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from navi_sched.route_control import RouteAddressSubscriber

from .NaviSchedAdapterConfig import NaviSchedAdapterConfig
from .NaviSchedClient import NaviSchedClient


def _probe_navi_sched_master(master_address: str, timeout_s: float) -> bool:
    if not master_address:
        return False
    request = Request(
        f"http://{master_address}/route-mode",
        data=b'{"update":false}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with build_opener(ProxyHandler({})).open(
            request, timeout=timeout_s
        ) as response:
            payload = json.loads(response.read().decode())
    except HTTPError:
        return False
    return (
        isinstance(payload, dict)
        and payload.get("accepted") is True
        and isinstance(payload.get("generation"), int)
    )


def _install_navi_sched_master_health_fallback(host_service) -> None:
    master_service = getattr(host_service, "master_service", None)
    original_probe = getattr(master_service, "_probe_single_host", None)
    if original_probe is None or getattr(
        master_service, "_navi_sched_health_fallback_installed", False
    ):
        return

    def probe(host):
        result = original_probe(host)
        if result is not None:
            return result
        master_address = f"{host.ip}:{host.port}"
        if not _probe_navi_sched_master(master_address, 0.2):
            return None
        return {
            "real_master_host": master_address,
            "queue_length": 0,
        }

    master_service._probe_single_host = probe
    master_service._navi_sched_health_fallback_installed = True


class AutoDetectMasterClient:
    """Temporary adapter that detects the Master implementation via /route-mode."""

    uses_prompt_token_ids = False

    def __init__(
        self,
        flexlb_client,
        host_service,
        config=None,
        navi_sched_client=None,
        probe=None,
        probe_interval_s=1.0,
        probe_timeout_s=0.2,
    ):
        self.config = config or NaviSchedAdapterConfig.from_env()
        self.config.validate_client()
        self._host_service = host_service
        self._flexlb_client = flexlb_client
        self._navi_sched_client = navi_sched_client or NaviSchedClient(
            self.config,
            host_service=host_service,
            enable_agent_forward=True,
        )
        self._probe = probe or _probe_navi_sched_master
        self._probe_interval_s = max(0.05, float(probe_interval_s))
        self._probe_timeout_s = max(0.01, float(probe_timeout_s))
        self._probe_lock = asyncio.Lock()
        self._last_probe_time = 0.0
        self._last_master_address = ""
        self._using_navi_sched = False

    def using_navi_sched(self) -> bool:
        return self._using_navi_sched

    def is_available(self) -> bool:
        return bool(self._host_service.get_master_addr())

    async def prepare_input(self, _input=None) -> None:
        await self._refresh_mode()

    def _selected_client(self):
        return (
            self._navi_sched_client
            if self._using_navi_sched
            else self._flexlb_client
        )

    def _master_probe_address(self) -> str:
        master_address = self._host_service.get_master_addr() or ""
        if master_address:
            return master_address
        master_vip = getattr(self._host_service, "master_vip", None)
        if master_vip is None:
            return ""
        try:
            hosts = master_vip.get_hosts()
        except Exception:
            return ""
        for host in hosts:
            ip = str(getattr(host, "ip", "")).strip()
            port = int(getattr(host, "port", 0) or 0)
            if ip and port > 0:
                return f"{ip}:{port}"
        return ""

    async def _refresh_mode(self, force=False) -> bool:
        now = time.monotonic()
        master_address = self._master_probe_address()
        if (
            not force
            and master_address == self._last_master_address
            and now - self._last_probe_time < self._probe_interval_s
        ):
            return False
        async with self._probe_lock:
            now = time.monotonic()
            master_address = self._master_probe_address()
            if (
                not force
                and master_address == self._last_master_address
                and now - self._last_probe_time < self._probe_interval_s
            ):
                return False
            detected = self._using_navi_sched
            if master_address:
                try:
                    detected = await asyncio.to_thread(
                        self._probe, master_address, self._probe_timeout_s
                    )
                except OSError:
                    pass
                except (ValueError, KeyError, json.JSONDecodeError):
                    detected = False
            else:
                detected = False
            changed = detected != self._using_navi_sched
            self._using_navi_sched = detected
            self._last_master_address = master_address
            self._last_probe_time = now
            return changed

    def get_latest_queue_length(self) -> int:
        return self._selected_client().get_latest_queue_length()

    async def get_backend_role_addrs(self, **kwargs):
        await self._refresh_mode()
        selected_client = self._selected_client()
        result = await selected_client.get_backend_role_addrs(**kwargs)
        if result.is_ok:
            return result
        changed = await self._refresh_mode(force=True)
        if not changed:
            return result
        return await self._selected_client().get_backend_role_addrs(**kwargs)

    async def close(self) -> None:
        await asyncio.gather(
            self._navi_sched_client.close(),
            self._flexlb_client.close(),
        )


class _FlexLBHostServiceProxy:
    def __init__(self, host_service, route_subscriber):
        self._host_service = host_service
        self._route_subscriber = route_subscriber

    def get_master_addr(self):
        return self._route_subscriber.master_address or None

    def get_slave_addr(self):
        return None

    def get_queue_length(self):
        return 0

    def __getattr__(self, name):
        return getattr(self._host_service, name)


class SwitchableMasterClient:
    uses_prompt_token_ids = False

    def __init__(
        self,
        flexlb_client,
        host_service,
        config=None,
        navi_sched_client=None,
        route_subscriber=None,
    ):
        self.config = config or NaviSchedAdapterConfig.from_env()
        self.config.validate_switching()
        self._route_subscriber = route_subscriber or RouteAddressSubscriber(
            self.config.master_address
        )
        self._flexlb_client = flexlb_client
        self._flexlb_client.host_service = _FlexLBHostServiceProxy(
            host_service, self._route_subscriber
        )
        self._navi_sched_client = navi_sched_client or NaviSchedClient(
            self.config,
            host_service=host_service,
            enable_agent_forward=True,
        )

    @property
    def flexlb_master_address(self) -> str:
        return self._route_subscriber.master_address

    def using_navi_sched(self) -> bool:
        return not self.flexlb_master_address

    def is_available(self) -> bool:
        if self.using_navi_sched():
            return self._navi_sched_client.is_available()
        return True

    def _selected_client(self):
        if self.using_navi_sched():
            return self._navi_sched_client
        return self._flexlb_client

    def get_latest_queue_length(self) -> int:
        return self._selected_client().get_latest_queue_length()

    async def prepare_input(self, input) -> None:
        prepare_input = getattr(self._selected_client(), "prepare_input", None)
        if prepare_input is not None:
            await prepare_input(input)

    async def get_backend_role_addrs(self, **kwargs):
        return await self._selected_client().get_backend_role_addrs(**kwargs)

    async def close(self) -> None:
        self._route_subscriber.stop()
        await asyncio.gather(
            self._navi_sched_client.close(),
            self._flexlb_client.close(),
        )


def create_switchable_master_client(flexlb_client, host_service):
    config = NaviSchedAdapterConfig.from_env()
    if not config.enabled:
        return flexlb_client
    return SwitchableMasterClient(flexlb_client, host_service, config)


def create_auto_detect_master_client(flexlb_client, host_service):
    role_type = os.environ.get("ROLE_TYPE", "").strip().upper()
    if role_type and role_type != "FRONTEND":
        return flexlb_client
    config = NaviSchedAdapterConfig.from_env()
    try:
        config.validate_client()
    except ValueError:
        return flexlb_client
    _install_navi_sched_master_health_fallback(host_service)
    return AutoDetectMasterClient(flexlb_client, host_service, config)

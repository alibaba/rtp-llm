import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import requests
from pydantic import BaseModel, Field

from rtp_llm.config.generate_config import RoleAddr, RoleType
from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import GaugeMetrics
from rtp_llm.vipserver import get_host_list_by_domain, get_host_list_by_domain_now
from rtp_llm.vipserver.host import Host

route_logger = logging.getLogger("route_logger")


class HostHealthStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HostHeartbeatInfo:
    host: Host
    health_status: HostHealthStatus = HostHealthStatus.UNKNOWN
    is_master: bool = False
    last_heartbeat_time: Optional[float] = None
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    queue_length: int = 0

    def mark_success(self, real_master_host: Optional[str] = None, queue_length: int = 0):
        self.health_status = HostHealthStatus.HEALTHY
        self.last_heartbeat_time = time.time()
        self.consecutive_failures = 0
        self.last_error = None
        self.queue_length = queue_length
        if real_master_host:
            self.is_master = (f"{self.host.ip}:{self.host.port}" == real_master_host)

    def mark_failure(self, error: str):
        self.last_heartbeat_time = time.time()
        self.consecutive_failures += 1
        self.last_error = error
        if self.consecutive_failures >= 2:
            self.health_status = HostHealthStatus.UNHEALTHY

    def host_key(self) -> str:
        return f"{self.host.ip}:{self.host.port}"


class VipServerWrapper:
    def __init__(self, domain: str, use_local: bool = False):
        self.cnt = 0
        self.use_local = use_local
        self.domain = domain
        if not self.domain:
            return
        if self.use_local:
            addr_list = domain.split(",")
            logging.info(f"Using local addresses: {addr_list}")
            hosts: List[Host] = []
            for addr in addr_list:
                ip, port = addr.split(":")
                hosts.append(Host(ip, port))
            self.hosts = hosts
        else:
            self.hosts = get_host_list_by_domain(self.domain)

    def get_host(self, refresh: bool = False) -> Optional[Host]:
        if not self.domain:
            return None
        get_hosts = get_host_list_by_domain_now if refresh else get_host_list_by_domain
        hosts = self.hosts if self.use_local else get_hosts(self.domain)
        if not hosts:
            return None
        cur_idx = self.cnt % len(hosts)
        self.cnt += 1
        return hosts[cur_idx]

    def get_hosts(self, refresh: bool = False) -> List[Host]:
        if not self.domain:
            return []
        get_hosts = get_host_list_by_domain_now if refresh else get_host_list_by_domain
        hosts = self.hosts if self.use_local else get_hosts(self.domain)
        return hosts if hosts else []


class EndPoint(BaseModel):
    type: str
    address: str
    protocol: str
    path: str


class GroupEndPoint(BaseModel):
    group: str
    prefill_endpoint: Optional[EndPoint] = None
    decode_endpoint: Optional[EndPoint] = None
    vit_endpoint: Optional[EndPoint] = None
    pd_fusion_endpoint: Optional[EndPoint] = None


class ServiceRoute(BaseModel):
    service_id: str
    role_endpoints: List[GroupEndPoint] = Field(default_factory=list)
    master_endpoint: Optional[EndPoint] = None
    use_local: bool = False


@dataclass
class HostServiceArgs:
    master_domain: str = ""
    pdfusion_domain: str = ""
    prefill_domain: str = ""
    decode_domain: str = ""
    vit_domain: str = ""
    use_local: bool = False

    @classmethod
    def create_from_env(cls):
        def _get_domain(endpoint: Optional[EndPoint]) -> str:
            if not endpoint:
                return ""
            return endpoint.address

        master_domain = ""
        pdfusion_domain = ""
        prefill_domain = ""
        decode_domain = ""
        vit_domain = ""
        use_local = False

        service_route_config_str = os.environ.get("MODEL_SERVICE_CONFIG", "")
        if service_route_config_str:
            server_route_config = ServiceRoute.model_validate(
                json.loads(service_route_config_str)
            )
            use_local = server_route_config.use_local
            master_domain = _get_domain(server_route_config.master_endpoint)

            if server_route_config.role_endpoints:
                group_endpoints = server_route_config.role_endpoints[0]
                pdfusion_domain = _get_domain(group_endpoints.pd_fusion_endpoint)
                prefill_domain = _get_domain(group_endpoints.prefill_endpoint)
                decode_domain = _get_domain(group_endpoints.decode_endpoint)
                vit_domain = _get_domain(group_endpoints.vit_endpoint)

        logging.info(
            f"""Model Service Configuration:
    Master Domain: {master_domain}
    PDFusion Domain: {pdfusion_domain}
    Prefill Domain: {prefill_domain}
    Decode Domain: {decode_domain}
    VIT Domain: {vit_domain}
    Use Local: {use_local}"""
        )

        return HostServiceArgs(
            master_domain=master_domain,
            pdfusion_domain=pdfusion_domain,
            prefill_domain=prefill_domain,
            decode_domain=decode_domain,
            vit_domain=vit_domain,
            use_local=use_local,
        )


class MasterService:
    def __init__(self, master_vip: VipServerWrapper):
        self.master_vip = master_vip
        self.cached_master_addr: Optional[str] = None
        self.cached_slave_addr: Optional[str] = None
        self.latest_queue_length: int = 0
        self.host_health_map: Dict[str, HostHeartbeatInfo] = {}
        self.health_map_lock = threading.Lock()
        self.unhealthy_node_timeout = 30.0
        self.backend_refresh_thread = threading.Thread(
            target=self.refresh_master_addr,
            name="rtp_llm_master_addr_refresh",
            daemon=True,
        )
        self.backend_refresh_thread.start()

    def refresh_master_addr(self):
        if not self.master_vip.domain:
            return

        while True:
            try:
                self._check_all_hosts_heartbeat()
                self._update_master_from_heartbeat_results()
                self._cleanup_unhealthy_nodes()
            except Exception as e:
                route_logger.error(f"Error in master refresh cycle: {e}")
            time.sleep(1)

    def _check_all_hosts_heartbeat(self):
        discovery_hosts = self.master_vip.get_hosts()
        if not discovery_hosts:
            route_logger.warning("No hosts available from VIP server")

        healthy_hosts = [
            info.host for info in self.host_health_map.values()
            if info.health_status == HostHealthStatus.HEALTHY
        ]

        seen_hosts = set()
        merged_hosts = []
        for host in discovery_hosts + healthy_hosts:
            host_key = f"{host.ip}:{host.port}"
            if host_key not in seen_hosts:
                seen_hosts.add(host_key)
                merged_hosts.append(host)

        if not merged_hosts:
            route_logger.warning("No hosts to probe")
            return

        for host in merged_hosts:
            try:
                result = self._probe_single_host(host)
                self._update_host_health(host, result)
            except Exception as e:
                route_logger.error(f"Error probing host {host.ip}:{host.port}: {e}")

    def _probe_single_host(self, host: Host) -> Optional[Dict]:
        master_addr = f"{host.ip}:{host.port}"
        headers = {"Content-Type": "application/json"}
        request_url = f"http://{master_addr}/rtp_llm/master/info"

        try:
            response = requests.post(request_url, headers=headers, json={}, timeout=0.5)
            if response.status_code == 200:
                result = response.json()
                return {
                    "real_master_host": result.get("real_master_host"),
                    "queue_length": result.get("queue_length", 0)
                }
        except Exception as e:
            route_logger.debug(f"Heartbeat failed for {master_addr}: {e}")
            return None

    def _update_host_health(self, host: Host, result: Optional[Dict]):
        host_addr = f"{host.ip}:{host.port}"

        with self.health_map_lock:
            if host_addr not in self.host_health_map:
                self.host_health_map[host_addr] = HostHeartbeatInfo(host=host)

            heartbeat_info = self.host_health_map[host_addr]

            if result is not None:
                heartbeat_info.mark_success(
                    real_master_host=result.get("real_master_host"),
                    queue_length=result.get("queue_length", 0)
                )
                route_logger.debug(
                    f"Host {host_addr} healthy, is_master={heartbeat_info.is_master}, "
                    f"queue_length={heartbeat_info.queue_length}"
                )
            else:
                heartbeat_info.mark_failure("Probe failed")
                route_logger.warning(
                    f"Host {host_addr} unhealthy, consecutive_failures={heartbeat_info.consecutive_failures}"
                )

    def _update_master_from_heartbeat_results(self):
        with self.health_map_lock:
            healthy_hosts = [
                info for info in self.host_health_map.values()
                if info.health_status == HostHealthStatus.HEALTHY
            ]

        if not healthy_hosts:
            route_logger.error("No healthy hosts available")
            self.cached_master_addr = None
            return

        # Priority 1: Use server-marked masters (ZK elected)
        master_candidates = [info for info in healthy_hosts if info.is_master]

        if master_candidates:
            selected_master = master_candidates[0]
            master_addr = f"{selected_master.host.ip}:{selected_master.host.port}"

            self.cached_master_addr = master_addr
            self.latest_queue_length = selected_master.queue_length
            self.update_slave_addr()

            kmonitor.report(GaugeMetrics.MASTER_HOST_METRIC, 1, tags={"master_host": master_addr})
            kmonitor.report(GaugeMetrics.MASTER_QUEUE_LENGTH_METRIC, self.latest_queue_length)
            return

        # Priority 2: Client-side failover - select from healthy hosts when no ZK master exists
        # This is temporary - ZK will eventually elect the real master
        if self.cached_master_addr:
            current_master_info = self.host_health_map.get(self.cached_master_addr)
            if current_master_info:
                if current_master_info.health_status == HostHealthStatus.UNHEALTHY:
                    backup_candidates = sorted([info for info in healthy_hosts if info.host_key() != self.cached_master_addr], key=lambda x: x.host.ip)
                    if backup_candidates:
                        self._select_backup_master(backup_candidates[0], "Current master unhealthy, client-side failover to")
                else:
                    self.latest_queue_length = current_master_info.queue_length
                    kmonitor.report(GaugeMetrics.MASTER_HOST_METRIC, 1, tags={"master_host": self.cached_master_addr})
                    kmonitor.report(GaugeMetrics.MASTER_QUEUE_LENGTH_METRIC, self.latest_queue_length)
        else:
            backup_candidates = sorted(healthy_hosts, key=lambda x: x.host.ip)
            if backup_candidates:
                self._select_backup_master(backup_candidates[0], "No cached master, client-side selection to")

    def _cleanup_unhealthy_nodes(self):
        current_time = time.time()
        nodes_to_remove = []

        with self.health_map_lock:
            for host_addr, info in self.host_health_map.items():
                if (info.health_status == HostHealthStatus.UNHEALTHY and
                    info.last_heartbeat_time and
                    (current_time - info.last_heartbeat_time) > self.unhealthy_node_timeout):
                    nodes_to_remove.append(host_addr)

            for host_addr in nodes_to_remove:
                del self.host_health_map[host_addr]
                route_logger.info(f"Removed unhealthy node: {host_addr}")

        return len(nodes_to_remove)

    def _select_backup_master(self, backup_master: HostHeartbeatInfo, reason: str):
        backup_addr = f"{backup_master.host.ip}:{backup_master.host.port}"
        self.cached_master_addr = backup_addr
        self.latest_queue_length = backup_master.queue_length
        kmonitor.report(GaugeMetrics.MASTER_HOST_METRIC, 1, tags={"master_host": backup_addr})
        kmonitor.report(GaugeMetrics.MASTER_QUEUE_LENGTH_METRIC, self.latest_queue_length)
        route_logger.warning(
            f"{reason}: {backup_addr}, queue_length={self.latest_queue_length}"
        )

    def get_master_addr(self) -> Optional[str]:
        return self.cached_master_addr

    def get_slave_addr(self) -> Optional[str]:
        return self.cached_slave_addr

    def update_slave_addr(self):
        healthy_hosts = [
            info for info in self.host_health_map.values()
            if info.health_status == HostHealthStatus.HEALTHY
        ]

        if not healthy_hosts:
            return

        slave_candidates = sorted(
            [info for info in healthy_hosts if not info.is_master],
            key=lambda x: x.host.ip
        )

        if slave_candidates and self.cached_master_addr:
            slave_candidates = [
                info for info in slave_candidates
                if info.host_key() != self.cached_master_addr
            ]

        if slave_candidates:
            selected_slave = slave_candidates[0]
            self.cached_slave_addr = selected_slave.host_key()

    def get_queue_length(self) -> int:
        return self.latest_queue_length

    def get_host_health_status(self) -> Dict[str, Dict]:
        return {
            addr: {
                "health": info.health_status.value,
                "is_master": info.is_master,
                "consecutive_failures": info.consecutive_failures,
                "last_heartbeat_time": info.last_heartbeat_time,
                "queue_length": info.queue_length
            }
            for addr, info in self.host_health_map.items()
        }


class HostService:
    def __init__(self, args: Optional[HostServiceArgs] = None):
        if args is None:
            args = HostServiceArgs.create_from_env()
        use_local = args.use_local
        self.master_vip = VipServerWrapper(args.master_domain, use_local)
        self.master_service = MasterService(self.master_vip)
        self.role_vip_map: Dict[RoleType, str] = {
            RoleType.PDFUSION: (
                VipServerWrapper(args.pdfusion_domain, use_local)
                if args.pdfusion_domain
                else None
            ),
            RoleType.PREFILL: (
                VipServerWrapper(args.prefill_domain, use_local)
                if args.prefill_domain
                else None
            ),
            RoleType.DECODE: (
                VipServerWrapper(args.decode_domain, use_local)
                if args.decode_domain
                else None
            ),
            RoleType.VIT: (
                VipServerWrapper(args.vit_domain, use_local)
                if args.vit_domain
                else None
            ),
        }
        self.master_service = MasterService(self.master_vip)

        self.service_available = bool(self.master_vip.domain) or any(
            self.role_vip_map.values()
        )

    def get_master_addr(self) -> Optional[str]:
        return self.master_service.get_master_addr()

    def get_queue_length(self) -> int:
        return self.master_service.get_queue_length()

    def get_slave_addr(self) -> Optional[str]:
        return self.master_service.get_slave_addr()

    def get_backend_role_addrs(
        self, role_list: List[RoleType], refresh: bool = False
    ) -> List[RoleAddr]:
        def _create_role_addr(
            role: RoleType, vip: VipServerWrapper
        ) -> Optional[RoleAddr]:
            host = vip.get_host(refresh)
            if host:
                return RoleAddr(role=role, ip=host.ip, grpc_port=int(host.port) + 1, http_port=int(host.port))  # type: ignore
            return None

        role_addrs: List[RoleAddr] = []
        for role in role_list:
            role_addr = _create_role_addr(role, self.role_vip_map.get(role))
            if role_addr:
                role_addrs.append(role_addr)
        return role_addrs

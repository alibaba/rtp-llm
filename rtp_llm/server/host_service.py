import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests
from pydantic import BaseModel, Field

from rtp_llm.config.generate_config import RoleAddr, RoleType
from rtp_llm.vipserver import get_host_list_by_domain, get_host_list_by_domain_now
from rtp_llm.vipserver.host import Host

route_logger = logging.getLogger("route_logger")


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
            host = self.master_vip.get_host(refresh=True)
            master_addr = f"{host.ip}:{host.port}" if host else None
            route_logger.debug(f"master address from vipserver: {master_addr}")
            if master_addr:
                try:
                    # request master_addr to get real master address
                    headers = {"Content-Type": "application/json"}
                    request_url = "http://" + master_addr + "/rtp_llm/master"
                    response = requests.post(
                        request_url, headers=headers, json={}, timeout=0.5
                    )
                    if response.status_code == 200:
                        master_addr = response.json()["real_master_host"]
                except Exception as e:
                    route_logger.error(
                        f"Failed to get master address from {master_addr}, error: {e}"
                    )
            route_logger.debug(f"master address refreshed: {master_addr}")
            self.cached_master_addr = master_addr
            time.sleep(1)

    def get_master_addr(self) -> Optional[str]:
        return self.cached_master_addr


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

    def get_backend_role_addrs(
        self, role_list: List[RoleType], refresh: bool = False
    ) -> List[RoleAddr]:
        def _create_role_addr(
            role: RoleType, vip: VipServerWrapper
        ) -> Optional[RoleAddr]:
            host = vip.get_host(refresh)
            if host:
                # FIXME(zhangjiangning.zjn): should use the logic from WorkerInfo.grpc_server_port to determine the grpc_port
                return RoleAddr(role=role, ip=host.ip, grpc_port=int(host.port) + 7, http_port=int(host.port))  # type: ignore
            return None

        role_addrs: List[RoleAddr] = []
        for role in role_list:
            role_addr = _create_role_addr(role, self.role_vip_map.get(role))
            if role_addr:
                role_addrs.append(role_addr)
        return role_addrs

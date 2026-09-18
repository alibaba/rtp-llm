"""Bootstrap CSR before Carbon makes a Worker visible to normal discovery."""

import json
import logging
import os
import random
import threading
from typing import Callable, Optional

import requests


class ConstraintTreeBootstrap:
    PATH = "/rtp_llm/constraint_tree/register"

    def __init__(
        self,
        master_address: Callable[[], Optional[str]],
        service_id: str,
        http_port: int,
        role: str,
        interval: float = 10,
    ):
        self.master_address = master_address
        self.body = {
            "service_id": service_id,
            "http_port": http_port,
            "role": role,
        }
        self.health_url = f"http://127.0.0.1:{http_port}/health"
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="constraint-tree-bootstrap", daemon=True
        )

    @classmethod
    def from_env(cls, host_service, http_port: int, role):
        if os.environ.get("CONSTRAINT_TREE_REQUIRED", "").lower() not in (
            "1",
            "true",
            "on",
        ):
            return None
        role_name = getattr(role, "name", str(role)).removeprefix("RoleType.")
        if role_name not in ("DECODE", "PDFUSION"):
            return None
        config = json.loads(os.environ.get("MODEL_SERVICE_CONFIG", "{}"))
        if not config.get("service_id"):
            raise ValueError(
                "CONSTRAINT_TREE_REQUIRED bootstrap needs MODEL_SERVICE_CONFIG.service_id"
            )
        domain = os.environ.get("CONSTRAINT_TREE_MASTER_ENDPOINT", "").strip()
        if domain:
            # Tree delivery must not enable Master scheduling for inference.
            # Resolve only on the bootstrap thread; registration already follows
            # the Master's leader redirect, so no routing poller is needed here.
            master_address = lambda: cls._master_from_vip(domain)
        elif (config.get("master_endpoint") or {}).get("address"):
            # Existing FlexLB deployments may share their inference Master.
            master_address = host_service.get_master_addr
        else:
            raise ValueError(
                "CONSTRAINT_TREE_REQUIRED bootstrap needs CONSTRAINT_TREE_MASTER_ENDPOINT "
                "or MODEL_SERVICE_CONFIG.master_endpoint.address"
            )
        return cls(master_address, config["service_id"], http_port, role_name)

    @staticmethod
    def _master_from_vip(domain: str) -> Optional[str]:
        from rtp_llm.vipserver import get_host_list_by_domain_now

        hosts = get_host_list_by_domain_now(domain)
        if not hosts:
            return None
        host = random.choice(hosts)
        return f"{host.ip}:{host.port}"

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        # HTTP calls are bounded; do not race Session.close() with an in-flight call.
        if self._thread.is_alive():
            self._thread.join(timeout=12)

    def _register_once(self, session) -> bool:
        master = self.master_address()
        if not master:
            raise RuntimeError("constraint-tree Master is not yet discoverable")
        response = session.post(
            f"http://{master}{self.PATH}", json=self.body, timeout=(2, 5)
        )
        response.raise_for_status()
        # Do not stop renewing on delivery/activation alone: Carbon may not have
        # registered us yet. Conversely discovery alone does not imply readiness.
        if response.json().get("discovered") is not True:
            return False
        health = session.get(self.health_url, timeout=(1, 2))
        return health.status_code == 200 and health.json() == "ok"

    def _run(self):
        failures = 0
        with requests.Session() as session:
            # VIP returns a direct internal address. A proxy/NAT would change the
            # peer IP which Master uses as the native callback address.
            session.trust_env = False
            while not self._stop.is_set():
                try:
                    if self._register_once(session):
                        logging.info(
                            "constraint-tree bootstrap complete; discovery took over"
                        )
                        return
                    failures = 0
                except Exception as error:
                    failures += 1
                    if failures == 1 or failures % 6 == 0:
                        logging.warning(
                            "constraint-tree bootstrap will retry: %s", error
                        )
                self._stop.wait(self.interval * random.uniform(0.8, 1.2))

"""Run a Prefill Master against the live nodes of one Spectrum deployment."""

import ipaddress
import json
import logging
import os
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

LOG = logging.getLogger(__name__)
DISCOVERY_URL = (
    "http://127.0.0.1:8880/api/v1/discovery/virtual-services/" "{service_id}/instances"
)


def prefill_ips(
    payload, *, service_id, workspace, application, deployment, engine_port, own_ip
):
    """Accept only healthy instances tagged as this exact Prefill deployment."""
    if not isinstance(payload, dict):
        raise ValueError("invalid Spectrum discovery response")
    if payload.get("virtual_service_id") != service_id:
        raise ValueError("Spectrum virtual service ID changed")
    tags = payload.get("virtual_service_tags") or {}
    physical_tags = payload.get("physical_service_tags") or {}
    if not isinstance(tags, dict) or not isinstance(physical_tags, dict):
        raise ValueError("invalid Spectrum discovery tags")
    if (
        tags.get("dashscope.ai/workspace-id") != workspace
        or tags.get("dashscope.ai/application-name") != application
    ):
        raise ValueError("Spectrum virtual service scope changed")
    instances = payload.get("instances")
    if not isinstance(instances, list) or len(instances) > 256:
        raise ValueError("invalid Spectrum instance list")
    selected = set()
    for item in instances:
        if not isinstance(item, dict):
            continue
        service_tags = physical_tags.get(item.get("physical_service_id"), {})
        if not isinstance(service_tags, dict):
            continue
        if service_tags.get("dashscope.ai/deployment-name") != deployment:
            continue
        if item.get("port") != engine_port:
            continue
        try:
            address = ipaddress.IPv4Address(item.get("ip"))
        except (ipaddress.AddressValueError, TypeError):
            continue
        selected.add(str(address))
    if own_ip not in selected:
        raise ValueError("own Prefill Pod missing from Spectrum discovery")
    return tuple(sorted(selected, key=lambda ip: int(ipaddress.IPv4Address(ip))))


def discover_prefill_ips(
    *, service_id, workspace, application, deployment, engine_port, own_ip
):
    if not service_id.startswith("v-"):
        raise ValueError("invalid Spectrum virtual service ID")
    url = DISCOVERY_URL.format(service_id=service_id)
    with urlopen(url, timeout=2) as response:
        payload = json.load(response)
    return prefill_ips(
        payload,
        service_id=service_id,
        workspace=workspace,
        application=application,
        deployment=deployment,
        engine_port=engine_port,
        own_ip=own_ip,
    )


def _stop_child(child):
    if child is None or child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=5)


def _start_child(config, addresses, agent_port, agent_biz):
    role = dict(config)
    runtime = dict(role.get("runtime_config") or {})
    runtime["local_nodes"] = [
        {
            "biz_name": agent_biz,
            "ip": ip,
            "grpc_port": agent_port,
            "support_heartbeat": True,
        }
        for ip in addresses
    ]
    role["runtime_config"] = runtime
    child_env = os.environ.copy()
    child_env["NAVI_SCHED_MASTER_CONFIG_JSON"] = json.dumps(role)
    child_env.pop("LD_PRELOAD", None)
    LOG.info(
        "starting Master with %d Prefill Agents: %s",
        len(addresses),
        ",".join(addresses),
    )
    return subprocess.Popen(
        [sys.executable, "-m", "navi_sched.master_process"],
        env=child_env,
        start_new_session=True,
    )


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    env = os.environ
    service_id = env["SPECTRUM_APPLICATION_SERVICE_ID"]
    workspace = env["SPECTRUM_WORKSPACE_ID"]
    application = env["SPECTRUM_APPLICATION_NAME"]
    deployment = env["SPECTRUM_DEPLOYMENT_NAME"]
    expected = env["NAVI_SCHED_PREFILL_DEPLOYMENT"]
    if deployment != expected or not deployment.startswith(application + "-"):
        raise ValueError("not the configured Prefill deployment")
    own_ip = str(ipaddress.IPv4Address(env["NAVI_SCHED_POD_IP"]))
    engine_port = int(env["NAVI_SCHED_ENGINE_PORT"])
    agent_port = int(env["NAVI_SCHED_AGENT_GRPC_PORT"])
    agent_biz = env["NAVI_SCHED_AGENT_BIZ_NAME"]
    with open(env["NAVI_SCHED_MASTER_CONFIG_FILE"], encoding="utf-8") as stream:
        config = json.load(stream)
    options = dict(
        service_id=service_id,
        workspace=workspace,
        application=application,
        deployment=deployment,
        engine_port=engine_port,
        own_ip=own_ip,
    )
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    child = None
    current = None
    candidate = None
    candidate_count = 0
    try:
        while not stopping:
            try:
                discovered = discover_prefill_ips(**options)
            except (OSError, ValueError, KeyError, TypeError) as error:
                LOG.warning("Spectrum Prefill discovery unavailable: %s", error)
                discovered = None
            if discovered is not None:
                if discovered == candidate:
                    candidate_count += 1
                else:
                    candidate, candidate_count = discovered, 1
            if child is not None and child.poll() is not None:
                LOG.error("Master exited with status %s", child.returncode)
                child = None
            if candidate_count >= 2 and candidate != current:
                _stop_child(child)
                child = None
                current = candidate
            if child is None and current is not None:
                child = _start_child(config, current, agent_port, agent_biz)
            for _ in range(50):
                if stopping:
                    break
                time.sleep(0.1)
    finally:
        _stop_child(child)


if __name__ == "__main__":
    main()

"""Translate FlexLB KVCM topology into the Master query Resource config."""

import json
import os
import re


def _object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate MODEL_SERVICE_CONFIG field: {name}")
        result[name] = value
    return result


def _positive_integer(value, name, maximum=None):
    if type(value) is not int or value <= 0 or (maximum and value > maximum):
        raise ValueError(f"invalid KVCM {name}")
    return value


def make_tair_kvcm_resource_config(explicit=None, env=None):
    """An absent topology is dormant; Worker capability still gates queries.

    Explicit Resource settings take precedence. The environment uses FlexLB's
    current MODEL_SERVICE_CONFIG object, whose top-level kvcm is the topology.
    Only fields consumed by the Resource are copied, never the full document.
    """
    if explicit is None:
        raw = (os.environ if env is None else env).get("MODEL_SERVICE_CONFIG", "")
        if not raw.strip():
            return {}
        document = json.loads(raw, object_pairs_hook=_object)
        if not isinstance(document, dict):
            raise ValueError("MODEL_SERVICE_CONFIG must be an object")
        explicit = document.get("kvcm")
        if explicit is None:
            return {}
    if not isinstance(explicit, dict):
        raise ValueError("KVCM Resource config must be an object")
    if not explicit:
        return {}
    address = explicit.get("address")
    if not isinstance(address, str) or not address.strip():
        raise ValueError("KVCM address must not be blank")
    discovery = explicit.get("discovery")
    if not isinstance(discovery, dict):
        raise ValueError("KVCM discovery must be configured")
    discovery_type = discovery.get("type")
    if not isinstance(discovery_type, str):
        raise ValueError("KVCM discovery.type must be configured")
    # FlexLB calls the Spectrum sidecar provider 'dashscope'.
    discovery_type = discovery_type.lower()
    if discovery_type == "dashscope":
        discovery_type = "spectrum"
    if discovery_type not in ("spectrum", "direct", "vipserver", "static-env"):
        raise ValueError(f"unsupported KVCM discovery.type: {discovery_type}")
    if discovery_type == "spectrum":
        if not re.fullmatch(r"[A-Za-z0-9_-]+", address):
            raise ValueError("invalid KVCM Spectrum service address")
        if discovery.get("base_url", "http://127.0.0.1:8880") != "http://127.0.0.1:8880":
            raise ValueError("KVCM Spectrum discovery requires the local sidecar")
    elif discovery_type == "direct" and not re.fullmatch(r"[A-Za-z0-9_.-]+", address):
        raise ValueError("invalid KVCM direct host")
    result = {
        "address": address,
        "port": _positive_integer(explicit.get("port", 6381), "port", 65535),
        "discovery": {"type": discovery_type},
        "thread_num": _positive_integer(explicit.get("thread_num", 4), "thread_num"),
    }
    if discovery_type in ("vipserver", "static-env") and any(c.isspace() for c in address):
        raise ValueError("invalid KVCM discovery address")
    if discovery_type == "vipserver" and "jmenv_dom" in discovery:
        domain = discovery["jmenv_dom"]
        if not isinstance(domain, str):
            raise ValueError("KVCM discovery.jmenv_dom must be a string")
        if domain.strip():
            result["discovery"]["jmenv_dom"] = domain.strip()
    if discovery_type == "static-env":
        hosts = discovery.get("hosts")
        if not isinstance(hosts, list) or not hosts:
            raise ValueError("KVCM static-env discovery.hosts must be nonempty")
        for host in hosts:
            if not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+:[0-9]+", host):
                raise ValueError("invalid KVCM static-env host")
            _positive_integer(int(host.rsplit(":", 1)[1]), "static-env host port", 65535)
        result["discovery"]["hosts"] = list(dict.fromkeys(hosts))
    namespace = explicit.get("namespace")
    if namespace is not None:
        if not isinstance(namespace, str):
            raise ValueError("KVCM namespace must be a string")
        if namespace.strip():
            result["namespace"] = namespace.strip()
    return result

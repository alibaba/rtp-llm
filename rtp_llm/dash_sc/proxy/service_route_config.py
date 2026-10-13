"""Configuration parsing for dash_sc proxy service routes."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Mapping, Optional

SERVICE_ROUTE_ENV_KEY = "SERVICE_ROUTE"
LEGACY_FORWARD_ENV_KEY = "DASH_SC_GRPC_FORWARD_ADDR"

SERVICE_ROUTE_TYPE_IP_PORT_LIST = "ip_port_list"
SERVICE_ROUTE_TYPE_VIPSERVER = "vipserver"


@dataclass(frozen=True)
class ServiceRouteConfig:
    type: str
    address: str


def load_service_route_config_from_env(
    env: Optional[Mapping[str, str]] = None,
) -> ServiceRouteConfig:
    if env is None:
        env = os.environ
    raw = env.get(SERVICE_ROUTE_ENV_KEY, "").strip()
    if raw:
        return parse_service_route_config(raw)

    legacy = env.get(LEGACY_FORWARD_ENV_KEY, "").strip()
    if legacy:
        addrs = parse_legacy_forward_addrs(legacy)
        if addrs:
            return ServiceRouteConfig(
                type=SERVICE_ROUTE_TYPE_IP_PORT_LIST,
                address=";".join(addrs),
            )
        raise RuntimeError(
            f"{LEGACY_FORWARD_ENV_KEY} did not contain any forward addresses"
        )

    raise RuntimeError(
        f"No service route provided. Set {SERVICE_ROUTE_ENV_KEY} env or "
        f"legacy {LEGACY_FORWARD_ENV_KEY} env."
    )


def parse_service_route_config(env_value: str) -> ServiceRouteConfig:
    try:
        data = json.loads(env_value.strip())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"{SERVICE_ROUTE_ENV_KEY} is not valid JSON: {e}") from e

    if not isinstance(data, dict):
        raise RuntimeError(
            f"{SERVICE_ROUTE_ENV_KEY} must be a JSON object with 'type' and 'address'"
        )

    try:
        cfg = ServiceRouteConfig(**data)
    except TypeError as e:
        raise RuntimeError(
            f"{SERVICE_ROUTE_ENV_KEY} must contain only 'type' and 'address'"
        ) from e

    if not isinstance(cfg.type, str) or not isinstance(cfg.address, str):
        raise RuntimeError(
            f"{SERVICE_ROUTE_ENV_KEY} 'type' and 'address' must both be strings"
        )

    cfg = ServiceRouteConfig(type=cfg.type.strip(), address=cfg.address.strip())
    if cfg.type == SERVICE_ROUTE_TYPE_IP_PORT_LIST:
        if not parse_ip_port_list(cfg.address):
            raise RuntimeError(f"{SERVICE_ROUTE_TYPE_IP_PORT_LIST} address is empty")
    elif cfg.type == SERVICE_ROUTE_TYPE_VIPSERVER:
        if not cfg.address:
            raise RuntimeError(f"{SERVICE_ROUTE_TYPE_VIPSERVER} address is empty")
    else:
        raise RuntimeError(
            f"{SERVICE_ROUTE_ENV_KEY} unsupported 'type': {cfg.type!r} "
            f"(expected one of: {SERVICE_ROUTE_TYPE_IP_PORT_LIST}, "
            f"{SERVICE_ROUTE_TYPE_VIPSERVER})"
        )
    return cfg


def parse_legacy_forward_addrs(env_value: str) -> list[str]:
    env_value = env_value.strip()
    if not env_value:
        return []

    if env_value.startswith("["):
        try:
            data = json.loads(env_value)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            return _normalise_addrs(str(addr) for addr in data)

    return _normalise_addrs(env_value.split(","))


def parse_ip_port_list(address: str) -> list[str]:
    return _normalise_addrs(address.split(";"))


def _normalise_addrs(addrs) -> list[str]:
    out = []
    seen = set()
    for addr in addrs:
        addr = str(addr).strip()
        if not addr or addr in seen:
            continue
        out.append(addr)
        seen.add(addr)
    return out

import json
import logging
import os
import re
from typing import Any, Set

from rtp_llm.frontend.sleep_validation import dedupe_addresses

SLEEP_CONTROL_ADDRESSES_ENV = "RTP_LLM_SLEEP_CONTROL_ADDRESSES"
SLEEP_INFER_CONTROL_ADDRESSES_ENV = "RTP_LLM_SLEEP_INFER_CONTROL_ADDRESSES"


def _parse_address_list(raw_value: str) -> list[str]:
    value = raw_value.strip()
    if not value:
        return []
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("JSON value must be a list")
        addresses = [str(item).strip() for item in parsed]
    else:
        addresses = [item.strip() for item in value.replace(";", ",").split(",")]

    addresses = [address for address in addresses if address]
    for address in addresses:
        if ":" not in address:
            raise ValueError(f"invalid address [{address}], expected host:port")
    return dedupe_addresses(addresses)


def get_control_addrs_from_env(
    env_name: str = SLEEP_CONTROL_ADDRESSES_ENV,
) -> list[str]:
    """Get lifecycle control addresses from an explicit env override.

    This is for separated frontend or multi-part deployments where local
    world_info may only contain the current node. The value accepts either a
    comma/semicolon-separated list or a JSON string list.
    """
    raw_value = os.environ.get(env_name, "")
    if not raw_value.strip():
        return []
    addresses = _parse_address_list(raw_value)
    logging.info("using control-plane addresses from %s: %s", env_name, addresses)
    return addresses


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _parse_gang_member_items(raw_value: str) -> list[dict[str, str]]:
    members: list[dict[str, str]] = []
    for member_str in raw_value.split(";"):
        member_str = member_str.strip()
        if not member_str:
            continue
        member: dict[str, str] = {}
        for item in member_str.split(","):
            if ":" not in item:
                continue
            key, value = item.split(":", 1)
            member[key.strip()] = value.strip()
        if member.get("ip"):
            members.append(member)
    return members


def _parse_gang_json_items(raw_value: str) -> list[dict[str, str]]:
    try:
        data = json.loads(raw_value)
    except Exception as e:
        logging.warning("failed to parse gang json for sleep control addresses: %s", e)
        return []
    if not isinstance(data, dict):
        return []
    members: list[dict[str, str]] = []
    for name, info in data.items():
        if not isinstance(info, dict) or not info.get("ip"):
            continue
        members.append(
            {
                "name": str(name),
                "ip": str(info.get("ip", "")),
                "port": str(info.get("port", "")),
            }
        )
    return members


def _part_sort_key(member: dict[str, str]) -> tuple[int, str]:
    name = member.get("name", "")
    match = re.search(r"part(\d+)$", name)
    if match:
        return (int(match.group(1)), name)
    return (10**9, name)


def _control_addrs_from_node_members(
    members: list[dict[str, str]],
    server_config: Any,
    parallelism_config: Any,
) -> list[str]:
    if not members:
        return []
    world_size = int(parallelism_config.world_size)
    local_world_size = int(parallelism_config.local_world_size)
    worker_info_port_num = int(server_config.worker_info_port_num)
    default_start_port = int(server_config.start_port)
    addresses: list[str] = []
    for node_index, member in enumerate(sorted(members, key=_part_sort_key)):
        ip = member.get("ip", "")
        if not ip:
            continue
        try:
            base_port = int(member.get("port") or default_start_port)
        except ValueError:
            logging.warning(
                "invalid gang member port for sleep control address: %s", member
            )
            continue
        for local_rank in range(local_world_size):
            world_rank = node_index * local_world_size + local_rank
            if world_rank >= world_size:
                break
            rpc_port = base_port + local_rank * worker_info_port_num + 1
            addresses.append(f"{ip}:{rpc_port}")
    return dedupe_addresses(addresses)


def infer_control_addrs_from_gang_metadata(
    server_config: Any,
    distribute_config: Any,
    parallelism_config: Any,
    env_name: str = SLEEP_INFER_CONTROL_ADDRESSES_ENV,
) -> list[str]:
    """Test-only helper to infer global rank RPC addresses from gang metadata.

    This is intentionally behind RTP_LLM_SLEEP_INFER_CONTROL_ADDRESSES because it
    assumes rank blocks are ordered by partN and ports follow RTP-LLM's local
    rank layout. Production should prefer control-plane membership or explicit
    RTP_LLM_SLEEP_CONTROL_ADDRESSES.
    """
    if not _truthy_env(env_name):
        return []

    members: list[dict[str, str]] = []
    if getattr(distribute_config, "gang_config_string", ""):
        members = _parse_gang_member_items(distribute_config.gang_config_string)
    elif getattr(distribute_config, "distribute_config_file", ""):
        try:
            with open(distribute_config.distribute_config_file, "r") as reader:
                members = _parse_gang_json_items(reader.read())
        except Exception as e:
            logging.warning(
                "failed to read distribute_config_file for sleep control addresses: %s",
                e,
            )

    addresses = _control_addrs_from_node_members(
        members, server_config, parallelism_config
    )
    if addresses:
        logging.warning(
            "using test-only inferred sleep control addresses from gang metadata: %s",
            addresses,
        )
    return addresses


def get_dp_addrs_from_world_info(world_info: Any, parallelism_config: Any) -> list[str]:
    """Get data parallel addresses from world_info."""
    ffn_disaggregate_config = parallelism_config.ffn_disaggregate_config
    logging.info(
        f"frontend worker ffn_disaggregate_config: {ffn_disaggregate_config.to_string()}"
    )
    # If FFN disaggregate is enabled, use only serving ranks; additional ranks
    # are internal to that node.
    if ffn_disaggregate_config.enable_ffn_disaggregate:
        serving_ranks = (
            ffn_disaggregate_config.attention_tp_size
            * ffn_disaggregate_config.attention_dp_size
        )
        members = world_info.members[:serving_ranks]
        logging.info(
            f"FFN disaggregate enabled, limiting addresses to {serving_ranks} serving ranks: {members}"
        )
    else:
        members = [
            member
            for member in world_info.members
            if (member.world_rank % parallelism_config.tp_size) == 0
        ]

    addresses = [f"{member.ip}:{member.rpc_server_port}" for member in members]
    logging.info(
        f"[world_rank: {parallelism_config.world_rank}] "
        f"using addresses from world_info: {addresses}"
    )
    return addresses


def get_control_addrs_from_world_info(world_info: Any) -> list[str]:
    """Get all backend RPC addresses that must receive lifecycle control."""
    addresses: list[str] = []
    for member in sorted(world_info.members, key=lambda x: x.world_rank):
        addresses.append(f"{member.ip}:{member.rpc_server_port}")
    addresses = dedupe_addresses(addresses)
    logging.info("using control-plane addresses from world_info: %s", addresses)
    return addresses

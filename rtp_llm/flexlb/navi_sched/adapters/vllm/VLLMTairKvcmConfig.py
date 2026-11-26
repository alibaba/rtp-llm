"""Capture Native V6D's registered remote layout once for the Agent handshake."""

import logging

logger = logging.getLogger("vllm.navi_sched.agent")


def load_tair_kvcm_config(source, cache_policy, navi_module):
    try:
        return _capture_tair_kvcm_config(source, cache_policy, navi_module)
    except Exception:
        logger.exception("Could not capture Native V6D remote query configuration")
        return None


def _capture_tair_kvcm_config(source, cache_policy, navi_module):
    context = source._client_context
    if context is None:
        logger.info("NaviSched Tair query capability: enabled=False reason=v6d_client_uninitialized")
        return None
    client = context.client
    clients = getattr(client, "_clients", (client,))
    owner = clients[0]
    # The connection predates spec registration and may cache an older mode.
    # Read capability and layout from the same refreshed Native metadata response.
    info = owner.rpc.refresh_peer_info()
    status = info.get("kvcm_status", {})
    if status.get("mode") != "report-event-and-cache-transfer":
        logger.info("NaviSched Tair query capability: enabled=False native_mode=%s",
                    status.get("mode"))
        return None
    remote = info.get("tair_kvcm_query_config")
    if remote is None:
        logger.warning("Native V6D has no registered Tair KVCM query layout")
        return None
    if (remote["hash_block_size"] != source.hash_block_size
            or remote["hash_block_size"] != cache_policy.block_size):
        raise ValueError("V6D remote query hash block size differs from Native cache policy")
    groups = remote["groups"]
    if sorted(group["group_id"] for group in groups) != list(range(len(cache_policy.group_roles))):
        raise ValueError("V6D remote query group IDs differ from Native cache policy")
    config = navi_module.NaviSchedTairKvcmConfig()
    config.manager_endpoint = remote["manager_endpoint"]
    config.instance_id = remote["instance_id"]
    config.hash_block_size = remote["hash_block_size"]
    config.node_count = remote["node_count"]
    config.discovery_refresh_interval_us = remote["discovery_refresh_interval_ms"] * 1000
    typed_groups = []
    for group in sorted(groups, key=lambda entry: entry["group_id"]):
        group_id = group["group_id"]
        spec = source.kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
        if (group["spec_type"] != type(spec).__name__
                or group["object_size_bytes"] != source._group_block_bytes[group_id][0]):
            raise ValueError("V6D remote query object layout differs from Native registration")
        role = cache_policy.group_roles[group_id]
        if role not in ("full", "tail"):
            raise ValueError("Unsupported Native cache group role for remote query")
        typed = navi_module.NaviSchedTairKvcmGroupConfig()
        typed.group_id = group_id
        typed.hash_block_scale = cache_policy.group_block_scales[group_id]
        typed.tail = role == "tail"
        typed.object_size_bytes = group["object_size_bytes"]
        typed.location_spec_names = group["location_spec_names"]
        typed_groups.append(typed)
    config.groups = typed_groups
    if not config.valid():
        raise ValueError("Native V6D remote query layout is incomplete")
    logger.info("NaviSched Tair query capability: enabled=True instance_id=%s groups=%d nodes=%d",
                config.instance_id, len(config.groups), config.node_count)
    return config

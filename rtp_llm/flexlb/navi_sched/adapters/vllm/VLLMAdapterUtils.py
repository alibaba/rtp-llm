import os

NAVI_SCHED_MEDIA_ERROR_PREFIX = "navi_sched_media_error:"


def split_address(address: str):
    value = address.strip()
    host, separator, port = value.rpartition(":")
    if not separator or not host or ":" in host:
        raise ValueError(f"invalid server address: {address}")
    try:
        port_number = int(port)
    except ValueError as exc:
        raise ValueError(f"invalid server address: {address}") from exc
    if not 0 < port_number <= 65535:
        raise ValueError(f"invalid server address: {address}")
    return host, port_number


def connect_event_endpoint(endpoint: str) -> str:
    replacements = {
        "tcp://*:": "tcp://127.0.0.1:",
        "tcp://0.0.0.0:": "tcp://127.0.0.1:",
    }
    for bind_prefix, connect_prefix in replacements.items():
        if endpoint.startswith(bind_prefix):
            return connect_prefix + endpoint[len(bind_prefix) :]
    return endpoint


def make_kmon_tags(role: str, environ=None) -> dict[str, str]:
    values = os.environ if environ is None else environ
    tags = {"role": role}
    deployment_name = (
        values.get("DEPLOYMENT_NAME", "").strip()
        or values.get("HIPPO_ROLE", "").strip()
    )
    if deployment_name:
        tags["deployment_name"] = deployment_name
    for env_name, tag_name in (
        ("HIPPO_APP", "hippo_app"),
        ("HIPPO_ROLE", "hippo_role"),
        ("C2_GROUP", "hippo_group"),
        ("HIPPO_GROUP", "hippo_group"),
        ("HIPPO_SLOT_ID", "slot_id"),
    ):
        value = values.get(env_name, "").strip()
        if value:
            tags[tag_name] = value
    physical_host_ip = values.get("HIPPO_SLAVE_IP", "").strip()
    instance_ip = (
        values.get("VLLM_NAVI_SCHED_SERVER_IP", "").strip()
        or values.get("POD_IP", "").strip()
        or physical_host_ip
    )
    if instance_ip:
        tags.update(
            {
                "host": instance_ip,
                "host_ip": instance_ip,
                "container_ip": instance_ip,
            }
        )
    if physical_host_ip:
        tags["physical_host_ip"] = physical_host_ip
    return tags

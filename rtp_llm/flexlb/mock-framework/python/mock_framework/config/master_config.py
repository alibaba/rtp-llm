"""Validate Master settings supplied by a Case or a local launch file."""

import json

from .validation import ConfigError, text


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"Duplicate native JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def validate_master_config(config):
    if "FLEXLB_CONFIG" not in config:
        return
    encoded = text(config["FLEXLB_CONFIG"], "FLEXLB_CONFIG")
    try:
        value = json.loads(
            encoded,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (ValueError, TypeError) as error:
        raise ConfigError(f"Invalid FLEXLB_CONFIG JSON: {error}") from error
    if not isinstance(value, dict):
        raise ConfigError("FLEXLB_CONFIG must contain a JSON object")


def validate_master_launch(config, launch_env, *, local_election):
    declared = {key for key in config if key.isupper()}
    duplicate = declared & launch_env.keys()
    if duplicate:
        key = min(duplicate)
        raise ConfigError(
            f"{key} is declared in both environment.master.config and launch.components.master.env"
        )
    assigned = {
        "MOCK_TOPOLOGY_FILE",
        "MOCK_NODE_ID",
        "MOCK_CONFIG_FILE",
        "MOCK_READY_FILE",
        "MOCK_OUTPUT_DIR",
    }
    if local_election:
        assigned.update(
            {
                "HIPPO_ROLE",
                "FLEXLB_SYNC_CONSISTENCY_CONFIG",
                "FLEXLB_ADVERTISED_IP",
                "MOCK_MASTER_PORT",
            }
        )
    for source, keys in [
        ("environment.master.config", declared),
        ("launch.components.master.env", launch_env.keys()),
    ]:
        conflict = assigned & keys
        if conflict:
            raise ConfigError(
                f"{source}.{min(conflict)} is assigned by the local runner"
            )
